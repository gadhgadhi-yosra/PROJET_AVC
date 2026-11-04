# Stroke lesion segmentation on T1 MRI – baselines (Projet 6)

## What we did

We used **1,416 T1 brain MRIs from 54 hospitals** (ATLAS v3 / ISLES'26; the 37 scans from centre R032 were excluded because of broken headers). Each scan has a manual lesion mask. We split the data **by hospital**:
- **OOD test:** 287 patients from 14 hospitals removed completely from training.
- **ID validation:** 226 new patients from the 40 training hospitals.
- **Training:** 903 patients.

All scans were reoriented to the same orientation (RAS), resampled to 1 mm voxels, cropped to the brain and intensity-normalised (z-score). We trained 4 variants of a pretrained CNN and 3 classical machine-learning models, then measured how well each one segments lesions on known vs. unseen hospitals. Everything ran on a laptop: RTX 2050 GPU (4 GB) and 16 GB of RAM.

## Results (Dice, same 300 test patients for every model)

| Model | ID (known hospitals) | OOD (unseen hospitals) |
|---|---|---|
| CNN pretrained, full fine-tune | **0.50** | **0.52** |
| CNN pretrained, early encoder frozen | 0.49 | 0.52 |
| CNN pretrained, whole encoder frozen | 0.47 | 0.51 |
| CNN random init (no pretraining) | 0.40 | 0.44 |
| ML gradient boosting | 0.33 | 0.38 |
| ML random forest | 0.28 | 0.33 |
| ML logistic regression | 0.16 | 0.21 |

Best CNN on all 512 test patients: **Dice 0.47 ID / 0.53 OOD**. By lesion size (OOD): 0.20 for lesions under 1 mL, 0.44 for 1–10 mL, 0.70 for 10–50 mL, 0.84 over 50 mL. The OOD hospitals score slightly higher only because their lesions are larger (median 5.6 mL vs 3.7 mL), so there is no measurable drop on unseen hospitals in this split. ImageNet pretraining adds about **+0.10 Dice**. The best model was chosen on ID validation only, never on the OOD test set.

## The CNN

| | |
|---|---|
| Architecture | **U-Net** with a **ResNet-34 encoder** pretrained on ImageNet (`segmentation_models_pytorch`), 24.4 M parameters |
| Input | **2.5D**: 3 adjacent axial slices (z−1, z, z+1) stacked as the 3 channels, 224 × 192 pixels (see below) |
| Output | 1 channel per pixel → **sigmoid** → lesion probability; threshold 0.5 |
| Activations | **ReLU** everywhere (encoder and decoder), **Batch Normalisation** after each convolution, sigmoid at the output |
| Loss | **Binary cross-entropy + soft Dice loss** (sum of the two) |
| Optimiser | **AdamW**, learning rate **5 × 10⁻⁴**, weight decay 10⁻⁴ |
| LR schedule | polynomial decay: lr × (1 − t/T)^0.9, where T is the time budget |
| Batch size | 16 slices |
| Training budget | **8 minutes per variant** (2,600–4,500 iterations ≈ 41,000–72,000 slices), same for every variant so the comparison is fair |
| Precision | mixed precision (float16 + gradient scaling) to fit in 4 GB and run faster |
| Slice sampling | 60 % of slices contain a lesion, 40 % random brain slices (lesions are only ~1 % of the brain) |
| Augmentation | random left–right flip, intensity scale ×0.85–1.15, shift ±0.15, Gaussian noise |
| Inference | every slice of the 3D volume is predicted, plus its left–right mirror (averaged = test-time augmentation), then the slices are stacked back into a 3D mask |

**The 4 variants (transfer learning / freezing):**
- **Full fine-tune:** all 24.4 M weights are updated.
- **Early encoder frozen:** the first ResNet blocks (conv1 + layer1 + layer2, generic edges/textures) are fixed; 23.1 M are trained.
- **Whole encoder frozen:** only the decoder learns (3.2 M weights, 13 %).
- **Random init:** same network, no ImageNet weights; the control that measures what pretraining brings.

**Backpropagation:** for each batch, the loss is computed and its gradient is propagated backwards from the output layer through the decoder (and the encoder layers that are not frozen). AdamW then updates those weights. Frozen layers have `requires_grad = False`: no gradient is computed for them and their weights never change. When the whole encoder is frozen, the backward pass stops at the decoder, which is why that variant ran fastest (9 iterations/s vs 5.4).

**How a 3D scan becomes a "3-channel image" (2.5D):** an ImageNet network expects 2D images with 3 colour channels (R, G, B). A 3D MRI is a stack of about 150 axial slices. For each slice z we put slice z−1 in the "red" channel, slice z in "green" and slice z+1 in "blue". The network therefore sees the slice plus its neighbours above and below (a little 3D context) and can reuse the ImageNet weights. It predicts the mask of the middle slice. At test time we do this for every slice and stack the 2D predictions back into a 3D mask. A full 3D network would need more GPU memory than the 4 GB available.

## The machine-learning models

Each brain voxel is classified as lesion / not lesion from **11 hand-made features**:
- **Intensity:** the voxel's intensity and blurred versions (Gaussian σ = 1, 2, 4 mm).
- **Texture:** local standard deviation and edge strength (gradient).
- **Left–right asymmetry:** intensity minus the mirror voxel in the other hemisphere. This is the most important feature (28 % of the random-forest importance), because a stroke damages one side only.
- **Position:** distance to the edge of the brain, and position in the brain (z, y, distance from the midline).

| | |
|---|---|
| Training data | 1.39 M voxels sampled from 250 training patients (up to 2,000 lesion + 4,000 healthy voxels each, 28 % lesion) |
| Logistic regression | standardised features, L2 regularisation (C = 1), up to 500 iterations; threshold 0.7 |
| Random forest | 30 trees, max depth 14, min 20 voxels per leaf, each tree on 50 % of the data; threshold 0.85 |
| Gradient boosting | histogram gradient boosting, 200 trees, max 63 leaves, learning rate 0.1; threshold 0.9 |
| Threshold | chosen on 15 other training patients (best mean Dice), because training is balanced but real brains are ~99 % healthy |
| Evaluation | random 150 ID + 150 OOD patients (the full set would take > 1 h on the laptop CPU) |

## Terms

- **Lesion:** the damaged brain area after the stroke; in the data it is a binary mask (1 = lesion voxel, 0 = healthy).
- **Voxel:** a 3D pixel; here 1 × 1 × 1 mm, so 1,000 voxels = 1 mL.
- **Segmentation:** labelling every voxel as lesion or not (drawing the lesion), as opposed to classification (one label per scan).
- **Dice:** overlap between predicted and true lesion = 2 × |overlap| / (|predicted| + |true|). 0 = no overlap, 1 = perfect. The standard metric; accuracy is useless here because over 99 % of voxels are healthy.
- **HD95 (Hausdorff distance, 95th percentile):** how far the predicted border is from the true border, in mm, ignoring the worst 5 % of points. Lower is better.
- **Absolute volume difference:** |predicted volume − true volume| in mL; important clinically (lesion size).
- **Lesion-wise F1:** counts lesions instead of voxels. Each connected blob is one lesion; a true lesion is "found" if the prediction touches it, and a predicted blob touching no true lesion is a false alarm. F1 combines found lesions and false alarms (1 = perfect).
- **Lesion count difference:** |number of predicted blobs − number of true lesions|.
- **ID (in-distribution):** test patients from hospitals that were in training. **OOD (out-of-distribution):** patients from hospitals never seen.
- **Domain shift:** images differ between hospitals (scanner, protocol, resolution), so a model can drop on a new hospital.
- **Generalisation gap:** Dice(ID) − Dice(OOD). Positive means a drop on new hospitals. **Size-adjusted gap:** the same, after giving both sets the same mix of lesion sizes.
- **Bootstrap 95 % CI:** uncertainty of a result, obtained by re-sampling the patients 3,000 times.
- **RAS:** standard orientation (x → Right, y → Anterior, z → Superior), so every scan is stored the same way round.
- **Resampling:** changing voxel size (to 1 mm) so all scans have the same resolution.
- **z-score normalisation:** (intensity − mean) / standard deviation inside the brain, so all scanners share the same intensity scale.
- **CNN:** convolutional neural network. **U-Net:** a CNN with an **encoder** (shrinks the image, extracts features) and a **decoder** (expands back to full size); **skip connections** copy encoder details to the decoder for precise borders.
- **ResNet-34:** a 34-layer encoder with residual (shortcut) connections. **ImageNet:** 1.2 M natural photos used to pretrain it.
- **Transfer learning / fine-tuning:** start from weights learned on another task (ImageNet), then continue training on our data. **Freezing:** keeping some layers fixed during fine-tuning.
- **Epoch / iteration:** an iteration = one batch (16 slices) → one weight update. We trained on a time budget instead of epochs.
- **Loss:** the error the network minimises. **BCE** penalises each pixel; **soft Dice loss** directly optimises the overlap and handles the tiny lesion fraction.
- **Learning rate:** step size of each weight update. **AdamW:** Adam optimiser (adaptive step per weight) with decoupled weight decay (a penalty that keeps weights small).
- **ReLU:** activation max(0, x). **Sigmoid:** squashes the output into a 0–1 probability. **Batch Normalisation:** re-scales the activations of each layer to stabilise training.
- **Test-time augmentation (TTA):** predicting the image and its mirror and averaging the two.
- **nnU-Net:** a self-configuring 3D U-Net, the reference baseline in medical segmentation; prepared in `nnunet_full/` but too heavy for this laptop (needs a bigger GPU, e.g. Kaggle).

## Limits

- Short training (8 minutes per CNN); a fully trained 3D model would score higher. Published 3D models reach roughly 0.6–0.65 Dice on this kind of data.
- One hospital split only; leave-centres-out cross-validation (several held-out groups) is needed to confirm "no drop on new hospitals".
- Metrics are computed in the common 1 mm space, not on the original scan grid.
- ML models were evaluated on a 300-patient subset.

## Files

`preprocessing/` (split + data fixes, arrays), `training/` (`train_cnn.py`, `train_ml.py`), `results/make_results.py` → `outputs/results.txt` and `outputs/fig1–3`, `nnunet_full/` (full nnU-Net, later), `common.py` (shared metrics), `run_all.sh` (everything in order).
