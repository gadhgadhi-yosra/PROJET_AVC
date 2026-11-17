"""2.5D U-Net (ResNet-34 encoder, ImageNet-pretrained) fine-tuned for stroke lesions - fast baseline.

Input = 3 adjacent axial slices as the 3 RGB channels (fits the ImageNet encoder, gives some 3D context).
Each run gets the same wall-clock training budget, so the freezing variants are compared at equal cost:
  --freeze none     fine-tune everything
  --freeze partial  freeze stem + layer1 + layer2 (low-level filters), train layer3-4 + decoder
  --freeze encoder  freeze the whole encoder, train the decoder only
  --random          no pretraining (random init), everything trained - control for the value of ImageNet
Evaluation on ID val (other patients, training centres) and OOD (held-out centres), whole 3D volumes,
threshold 0.5 + left-right flip test-time augmentation. Writes outputs/fast_results/<name>_cases.csv.
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RES, case_metrics, load_case, load_index, save_pred

H, W = 224, 192  # axial slice (y, x) after 1 mm crop: median ~ 175 x 140


def stack3(img, zs):
    nz = img.shape[0]
    return np.stack([img[np.clip([z - 1, z, z + 1], 0, nz - 1)] for z in zs]).astype(np.float32)


class SliceDS(torch.utils.data.Dataset):
    def __init__(self, cases, les_slices, n):
        self.cases, self.les, self.n = cases, les_slices, n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng()
        c = self.cases[rng.integers(len(self.cases))]
        img, seg = load_case(c, mmap=True)
        les = self.les[c]
        z = int(rng.choice(les)) if len(les) and rng.random() < 0.6 else int(rng.integers(img.shape[0]))
        x, off = fit(stack3(img, [z])[0], rng)
        y, _ = fit(np.asarray(seg[z], np.float32)[None], rng, off)
        if rng.random() < 0.5:
            x, y = x[..., ::-1], y[..., ::-1]
        x = x * rng.uniform(0.85, 1.15) + rng.uniform(-0.15, 0.15)
        x = x + rng.normal(0, rng.uniform(0, 0.1), x.shape).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))


def fit(a, rng, off=None):
    """random crop / zero-pad (c, h, w) to (c, H, W); pass the returned offsets again for the mask"""
    c, h, w = a.shape
    if off is None:
        off = (rng.integers(0, max(h - H, 0) + 1), rng.integers(0, max(w - W, 0) + 1),
               rng.integers(0, max(H - h, 0) + 1), rng.integers(0, max(W - w, 0) + 1))
    oy, ox, py, px = off
    a = a[:, oy:oy + H, ox:ox + W]
    out = np.zeros((c, H, W), np.float32)
    out[:, py:py + a.shape[1], px:px + a.shape[2]] = a
    return out, off


def build(freeze, pretrained):
    m = smp.Unet("resnet34", encoder_weights="imagenet" if pretrained else None, in_channels=3, classes=1)
    enc = m.encoder
    frozen = {"none": [], "partial": [enc.conv1, enc.bn1, enc.layer1, enc.layer2], "encoder": [enc]}[freeze]
    for mod in frozen:
        for p in mod.parameters():
            p.requires_grad = False
    return m


def loss_fn(logits, y):
    p = torch.sigmoid(logits)
    inter = (p * y).sum()
    dice = 1 - (2 * inter + 1) / (p.sum() + y.sum() + 1)
    return F.binary_cross_entropy_with_logits(logits, y) + dice


@torch.no_grad()
def predict_volume(model, img):
    nz, h, w = img.shape
    ph, pw = (-h) % 32, (-w) % 32
    prob = np.zeros((nz, h, w), np.float32)
    for s in range(0, nz, 48):
        zs = list(range(s, min(s + 48, nz)))
        x = torch.from_numpy(stack3(img, zs)).cuda()
        x = F.pad(x, (0, pw, 0, ph))
        with torch.autocast("cuda", dtype=torch.float16):
            p = torch.sigmoid(model(x)) + torch.sigmoid(model(x.flip(-1))).flip(-1)
        prob[s:s + len(zs)] = (p[:, 0, :h, :w] / 2).float().cpu().numpy()
    return prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", default="none", choices=["none", "partial", "encoder"])
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--minutes", type=float, default=9)
    ap.add_argument("--lr", type=float, default=5e-4)
    a = ap.parse_args()
    name = "cnn_random_init" if a.random else f"cnn_pretrained_freeze-{a.freeze}"
    torch.manual_seed(0)

    idx = load_index()
    train = idx.loc[idx["set"] == "train", "case"].tolist()
    les = {c: np.flatnonzero(load_case(c, mmap=True)[1].any(axis=(1, 2))) for c in train}
    dl = torch.utils.data.DataLoader(SliceDS(train, les, 10 ** 7), batch_size=16, num_workers=4,
                                     pin_memory=True, persistent_workers=True)
    model = build(a.freeze, not a.random).cuda()
    params = [p for p in model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in params)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"{name}: trainable {n_tr / 1e6:.1f}M / {n_all / 1e6:.1f}M params, budget {a.minutes} min", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler()
    model.train()
    T, t0, it, run = a.minutes * 60, time.time(), 0, []
    for x, y in dl:
        el = time.time() - t0
        if el > T:
            break
        for g in opt.param_groups:
            g["lr"] = a.lr * (1 - el / T) ** 0.9
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = loss_fn(model(x).float(), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        run.append(loss.item())
        it += 1
        if it % 200 == 0:
            print(f"  it {it}  {el / 60:.1f} min  loss {np.mean(run[-200:]):.3f}  {it / el:.1f} it/s", flush=True)
    print(f"trained {it} iterations ({it * 16} slices) in {(time.time() - t0) / 60:.1f} min", flush=True)
    del dl  # stop the loader workers before the metrics pool starts
    RES.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), RES / f"{name}.pt")

    model.eval()
    ev = idx[idx["set"].isin(["id_val", "ood_test"])]
    t1, futs = time.time(), []
    with ProcessPoolExecutor(3) as ex:
        for c, s in zip(ev["case"], ev["set"]):
            img, _ = load_case(c)
            pred = predict_volume(model, img.astype(np.float32)) > 0.5
            save_pred(name, c, pred)
            futs.append(ex.submit(case_metrics, c, "ID" if s == "id_val" else "OOD", np.packbits(pred), pred.shape))
        rows = [f.result() for f in futs]
    df = pd.DataFrame(rows).assign(model=name, train_iters=it)
    df.to_csv(RES / f"{name}_cases.csv", index=False)
    d = df[df["gt_ml"] > 0].groupby("split")[["dice", "lesion_f1", "abs_vol_diff_ml"]].mean().round(3)
    print(f"eval {len(df)} cases in {(time.time() - t1) / 60:.1f} min\n{d}", flush=True)


if __name__ == "__main__":
    main()
