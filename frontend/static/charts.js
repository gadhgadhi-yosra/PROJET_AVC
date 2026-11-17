/* Graphiques SVG colorés sans dépendance (fonctionnent hors ligne). */
(function () {
  const NS = 'http://www.w3.org/2000/svg';
  let uid = 0;
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const fmt = (v, d = 2) => (v == null || v === '' || Number.isNaN(Number(v)) ? '—' : Number(v).toLocaleString('fr-FR', { maximumFractionDigits: d }));

  function niceMax(v) {
    if (!(v > 0)) return 1;
    const p = Math.pow(10, Math.floor(Math.log10(v)));
    const n = v / p;
    return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 2.5 ? 2.5 : n <= 5 ? 5 : 10) * p;
  }
  const tickN = (max) => { const n = max / Math.pow(10, Math.floor(Math.log10(max))); return n === 2.5 ? 5 : 4; };
  const ticks = (min, max, count = tickN(max - min || 1)) => Array.from({ length: count + 1 }, (_, i) => min + ((max - min) * i) / count);
  const width = (el, min = 340) => Math.max(min, Math.round(el.clientWidth || min));

  // Dégradé linéaire : renvoie [defs, url]
  function gradient(c1, c2, vertical = true) {
    const id = `cg${++uid}`;
    const dir = vertical ? 'x1="0" y1="0" x2="0" y2="1"' : 'x1="0" y1="0" x2="1" y2="0"';
    return [`<linearGradient id="${id}" ${dir}><stop offset="0" stop-color="${c1}"/><stop offset="1" stop-color="${c2 || c1}" stop-opacity="${c2 ? 1 : 0.7}"/></linearGradient>`, `url(#${id})`];
  }
  const svg = (w, h, defs, body) => `<svg class="chart" viewBox="0 0 ${w} ${h}" role="img" xmlns="${NS}"><defs>${defs}</defs>${body}</svg>`;
  const legend = (series) => (!series || series.length < 2 ? '' :
    `<div class="legend">${series.map((s) => `<span><i style="background:linear-gradient(135deg,${s.color},${s.color2 || s.color})"></i>${esc(s.name)}</span>`).join('')}</div>`);

  function splitLabel(text, max) {
    const words = text.split(/\s+/), out = [];
    let cur = '';
    words.forEach((wd) => {
      if ((cur + ' ' + wd).trim().length > max && cur) { out.push(cur); cur = wd; } else cur = (cur + ' ' + wd).trim();
    });
    if (cur) out.push(cur);
    return out.slice(0, 3);
  }

  // Barres verticales groupées
  function bars(el, { labels, series, height = 260, yMax, format = (v) => fmt(v), valueLabels = true, wrap = 14 }) {
    const w = width(el), h = height;
    const lines = labels.map((l) => splitLabel(String(l), wrap));
    const nLines = Math.max(...lines.map((l) => l.length));
    const m = { t: 22, r: 8, b: 16 + nLines * 13, l: 44 };
    const iw = w - m.l - m.r, ih = h - m.t - m.b;
    const max = yMax ?? niceMax(Math.max(...series.flatMap((s) => s.values.filter((v) => v != null))) * 1.08);
    const band = iw / labels.length, inner = band * 0.74, bw = inner / series.length;
    let defs = '', body = '';
    const fills = series.map((s) => { const [d, u] = gradient(s.color, null); defs += d; return u; });
    ticks(0, max).forEach((t) => {
      const y = m.t + ih - (t / max) * ih;
      body += `<line class="grid-line" x1="${m.l}" x2="${w - m.r}" y1="${y}" y2="${y}"/><text x="${m.l - 8}" y="${y + 4}" text-anchor="end">${format(t)}</text>`;
    });
    labels.forEach((lab, i) => {
      const x0 = m.l + i * band + (band - inner) / 2;
      series.forEach((s, j) => {
        const v = s.values[i];
        if (v == null) return;
        const bh = (Math.max(0, v) / max) * ih, x = x0 + j * bw, y = m.t + ih - bh;
        const bwid = Math.max(2, Math.min(bw - 6, 46)), bx = x + (bw - bwid) / 2, r = Math.min(7, bwid / 2, bh);
        body += `<path class="grow-y" style="animation-delay:${(i * series.length + j) * 40}ms" d="M${bx},${y + bh} V${y + r} Q${bx},${y} ${bx + r},${y} H${bx + bwid - r} Q${bx + bwid},${y} ${bx + bwid},${y + r} V${y + bh} Z" fill="${fills[j]}" data-tip="${esc(`${lab} · ${s.name} : ${format(v)}`)}"/>`;
        if (valueLabels && bw > 26) body += `<text class="value-label pop" x="${x + bw / 2 - 1}" y="${y - 6}" text-anchor="middle">${format(v)}</text>`;
      });
      const cx = m.l + i * band + band / 2;
      lines[i].forEach((ln, k) => { body += `<text x="${cx}" y="${m.t + ih + 16 + k * 13}" text-anchor="middle">${esc(ln)}</text>`; });
    });
    body += `<line class="axis" x1="${m.l}" x2="${w - m.r}" y1="${m.t + ih}" y2="${m.t + ih}"/>`;
    el.innerHTML = svg(w, h, defs, body) + legend(series);
  }

  // Barres horizontales empilées
  function stacked(el, { rows, series, format = (v) => fmt(v, 0), rowHeight = 22 }) {
    const w = width(el), m = { t: 6, r: 44, b: 24, l: 58 };
    const h = m.t + m.b + rows.length * rowHeight, iw = w - m.l - m.r;
    const max = niceMax(Math.max(...rows.map((r) => r.values.reduce((a, b) => a + (b || 0), 0))));
    let defs = '', body = '';
    const fills = series.map((s) => { const [d, u] = gradient(s.color, s.color2, false); defs += d; return u; });
    ticks(0, max).forEach((t) => {
      const x = m.l + (t / max) * iw;
      body += `<line class="grid-line" x1="${x}" x2="${x}" y1="${m.t}" y2="${h - m.b}"/><text x="${x}" y="${h - 6}" text-anchor="middle">${format(t)}</text>`;
    });
    rows.forEach((r, i) => {
      const y = m.t + i * rowHeight;
      let x = m.l;
      body += `<text x="${m.l - 10}" y="${y + rowHeight * 0.66}" text-anchor="end">${esc(r.label)}</text>`;
      r.values.forEach((v, j) => {
        if (!v) return;
        const bw = (v / max) * iw;
        body += `<rect class="grow-x" style="animation-delay:${i * 25}ms" x="${x}" y="${y + 4}" width="${bw}" height="${rowHeight - 8}" rx="${(rowHeight - 8) / 2}" fill="${fills[j]}" ${r.site ? `data-site="${esc(r.site)}"` : ''} data-tip="${esc(`${r.label} · ${series[j].name} : ${format(v)}`)}"/>`;
        x += bw;
      });
      body += `<text class="value-label pop" x="${x + 8}" y="${y + rowHeight * 0.66}">${format(r.values.reduce((a, b) => a + (b || 0), 0))}</text>`;
    });
    el.innerHTML = svg(w, h, defs, body) + legend(series);
  }

  // Points + intervalles de confiance (horizontal)
  function intervals(el, { items, min = 0, max = 1, ref, refLabel = '', format = (v) => fmt(v, 3), tickCount = 4, labelWidth = 200 }) {
    const w = width(el), rh = 34, m = { t: 16, r: 24, b: 28, l: Math.min(labelWidth, w * 0.42) };
    const h = m.t + m.b + items.length * rh, iw = w - m.l - m.r;
    const X = (v) => m.l + ((Math.min(max, Math.max(min, v)) - min) / (max - min)) * iw;
    let defs = '', body = '';
    ticks(min, max, tickCount).forEach((t) => {
      body += `<line class="grid-line" x1="${X(t)}" x2="${X(t)}" y1="${m.t}" y2="${h - m.b}"/><text x="${X(t)}" y="${h - 8}" text-anchor="middle">${format(t)}</text>`;
    });
    if (ref != null) {
      body += `<line x1="${X(ref)}" x2="${X(ref)}" y1="${m.t - 6}" y2="${h - m.b}" style="stroke:#e11d48" stroke-width="1.5" stroke-dasharray="5 4"/>`;
      if (refLabel) body += `<text x="${X(ref) + 5}" y="${m.t + 2}" style="fill:#e11d48;font-weight:700">${esc(refLabel)}</text>`;
    }
    items.forEach((it, i) => {
      const y = m.t + i * rh + rh / 2;
      body += `<text x="${m.l - 12}" y="${y + 4}" text-anchor="end">${esc(it.label)}</text>`;
      if (it.low != null) {
        const [d, u] = gradient(it.color, it.color2 || it.color, false); defs += d;
        body += `<rect class="grow-x" style="animation-delay:${i * 60}ms" x="${X(it.low)}" y="${y - 4}" width="${Math.max(2, X(it.high) - X(it.low))}" height="8" rx="4" fill="${u}" opacity=".45"/>`;
      }
      body += `<circle class="pop" style="animation-delay:${i * 60 + 200}ms;fill:${it.color};stroke:var(--panel)" stroke-width="2.5" cx="${X(it.value)}" cy="${y}" r="7.5" data-tip="${esc(`${it.label} : ${format(it.value)}${it.low != null ? ` [${format(it.low)} ; ${format(it.high)}]` : ''}`)}"/>`;
    });
    el.innerHTML = svg(w, h, defs, body);
  }

  // Nuage de points (axe x log optionnel)
  function scatter(el, { points, xLabel, yLabel, height = 320, series = [], format = (v) => fmt(v, 2), logX = false }) {
    const w = width(el), h = height, m = { t: 26, r: 20, b: 44, l: 50 };
    const iw = w - m.l - m.r, ih = h - m.t - m.b;
    const tx = (v) => (logX ? Math.log10(Math.max(v, 1)) : v);
    const x1 = logX ? Math.max(1, Math.ceil(Math.max(...points.map((p) => tx(p.x))))) : niceMax(Math.max(...points.map((p) => p.x)));
    const y1 = niceMax(Math.max(...points.map((p) => p.y)) * 1.05);
    const X = (v) => m.l + (tx(v) / x1) * iw, Y = (v) => m.t + ih - (v / y1) * ih;
    let body = '';
    ticks(0, y1).forEach((t) => {
      body += `<line class="grid-line" x1="${m.l}" x2="${w - m.r}" y1="${Y(t)}" y2="${Y(t)}"/><text x="${m.l - 8}" y="${Y(t) + 4}" text-anchor="end">${format(t)}</text>`;
    });
    (logX ? Array.from({ length: x1 + 1 }, (_, i) => 10 ** i) : ticks(0, x1)).forEach((t) => {
      body += `<text x="${X(t)}" y="${m.t + ih + 18}" text-anchor="middle">${fmt(t, 0)}</text>`;
    });
    body += `<line class="axis" x1="${m.l}" x2="${w - m.r}" y1="${m.t + ih}" y2="${m.t + ih}"/>`;
    points.forEach((p, i) => {
      body += `<circle class="pop" style="animation-delay:${i * 12}ms;fill:${p.color};fill-opacity:.78;stroke:var(--panel)" stroke-width="1.5" cx="${X(p.x)}" cy="${Y(p.y)}" r="${p.r || 6}" ${p.site ? `data-site="${esc(p.site)}"` : ''} data-tip="${esc(p.label)}"/>`;
    });
    body += `<text x="${m.l + iw / 2}" y="${h - 4}" text-anchor="middle" class="label-strong">${esc(xLabel)}</text>`;
    body += `<text x="${m.l}" y="${m.t - 10}" class="label-strong">${esc(yLabel)}</text>`;
    el.innerHTML = svg(w, h, '', body) + legend(series);
  }

  // Courbe avec aire dégradée
  function line(el, { points, color, color2, xLabel, yLabel, height = 260, ideal, format = (v) => fmt(v, 2) }) {
    const w = width(el), h = height, m = { t: 26, r: 20, b: 44, l: 44 };
    const iw = w - m.l - m.r, ih = h - m.t - m.b;
    const xMax = Math.max(...points.map((p) => p.x));
    const yMax = niceMax(Math.max(...points.map((p) => p.y), ...(ideal || []).map((p) => p.y)));
    const X = (v) => m.l + (v / xMax) * iw, Y = (v) => m.t + ih - (v / yMax) * ih;
    const [dArea, uArea] = gradient(color, null, true);
    const [dLine, uLine] = gradient(color, color2 || color, false);
    let body = '';
    ticks(0, yMax).forEach((t) => {
      body += `<line class="grid-line" x1="${m.l}" x2="${w - m.r}" y1="${Y(t)}" y2="${Y(t)}"/><text x="${m.l - 8}" y="${Y(t) + 4}" text-anchor="end">${format(t)}</text>`;
    });
    points.forEach((p) => { body += `<text x="${X(p.x)}" y="${m.t + ih + 18}" text-anchor="middle">${p.x}</text>`; });
    if (ideal) body += `<polyline style="fill:none;stroke:#a6a2cf" stroke-width="1.5" stroke-dasharray="5 5" points="${ideal.map((p) => `${X(p.x)},${Y(p.y)}`).join(' ')}"/>`;
    const pts = points.map((p) => `${X(p.x)},${Y(p.y)}`).join(' ');
    body += `<polygon class="pop" fill="${uArea}" opacity=".35" points="${X(points[0].x)},${Y(0)} ${pts} ${X(points[points.length - 1].x)},${Y(0)}"/>`;
    body += `<polyline class="pulse-line" style="fill:none" stroke="${uLine}" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round" points="${pts}"/>`;
    points.forEach((p, i) => { body += `<circle class="pop" style="animation-delay:${400 + i * 120}ms;fill:${color2 || color};stroke:var(--panel)" stroke-width="2.5" cx="${X(p.x)}" cy="${Y(p.y)}" r="6.5" data-tip="${esc(p.label)}"/>`; });
    body += `<text x="${m.l + iw / 2}" y="${h - 4}" text-anchor="middle" class="label-strong">${esc(xLabel)}</text>`;
    body += `<text x="${m.l}" y="${m.t - 10}" class="label-strong">${esc(yLabel)}</text>`;
    el.innerHTML = svg(w, h, dArea + dLine, body);
  }

  // Anneau (donut)
  function donut(el, { parts, center, sub, size = 220 }) {
    const total = parts.reduce((a, p) => a + p.value, 0), r = 80, c = 2 * Math.PI * r;
    let offset = 0, defs = '', body = '';
    parts.forEach((p, i) => {
      const [d, u] = gradient(p.color, p.color2 || p.color, false); defs += d;
      const len = (p.value / total) * c;
      body += `<circle class="pop" style="animation-delay:${i * 150}ms" cx="110" cy="110" r="${r}" fill="none" stroke="${u}" stroke-width="26" stroke-dasharray="${Math.max(0, len - 3)} ${c}" stroke-dashoffset="${-offset}" transform="rotate(-90 110 110)" data-tip="${esc(`${p.name} : ${fmt(p.value, 0)} (${fmt((p.value / total) * 100, 1)} %)`)}"/>`;
      offset += len;
    });
    body += `<text x="110" y="108" text-anchor="middle" class="donut-center">${esc(center)}</text><text x="110" y="132" text-anchor="middle">${esc(sub)}</text>`;
    el.innerHTML = `<div style="max-width:${size}px;margin:0 auto">${svg(220, 220, defs, body)}</div>` + legend(parts);
  }

  // Infobulle globale
  const tip = document.createElement('div');
  tip.className = 'tooltip'; tip.hidden = true;
  document.addEventListener('DOMContentLoaded', () => document.body.appendChild(tip));
  document.addEventListener('mousemove', (e) => {
    const t = e.target.closest && e.target.closest('[data-tip]');
    if (!t) { tip.hidden = true; return; }
    tip.textContent = t.getAttribute('data-tip');
    tip.hidden = false;
    const x = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
    tip.style.left = `${x}px`; tip.style.top = `${e.clientY + 16}px`;
  });

  const hideTip = () => { tip.hidden = true; };
  window.addEventListener('hashchange', hideTip);
  document.addEventListener('click', hideTip);
  document.addEventListener('scroll', hideTip, true);

  window.Charts = { bars, stacked, intervals, scatter, line, donut, fmt, esc };
})();
