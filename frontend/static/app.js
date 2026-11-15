/* Application monopage : une fonction de rendu par section du projet. */
(function () {
  const { bars, stacked, intervals, scatter, line, donut, fmt, esc } = window.Charts;
  const $ = (s, root = document) => root.querySelector(s);
  const content = $('#content');
  const cache = {};
  const days = (v) => `${fmt(v, 0)} jour${Math.abs(v) >= 2 ? 's' : ''}`;
  const pct = (v, d = 1) => (v == null ? '—' : `${fmt(v * 100, d)} %`);

  // Palette vive (couples de dégradés)
  const P = {
    violet: ['#8b5cf6', '#6d28d9'], pink: ['#f472b6', '#db2777'], orange: ['#fb923c', '#ea580c'],
    amber: ['#fbbf24', '#f59e0b'], cyan: ['#22d3ee', '#0891b2'], teal: ['#2dd4bf', '#0d9488'],
    blue: ['#60a5fa', '#2563eb'], green: ['#4ade80', '#16a34a'], red: ['#fb7185', '#e11d48'],
  };
  const ID = P.blue, OOD = P.orange;

  const ICONS = {
    brain: '<path d="M12 4c-3 0-5 2-5 4.5-2 .5-3 2-3 4s1.5 3.5 3 3.8C7.5 19 9.5 20 12 20s4.5-1 5-3.7c1.5-.3 3-1.8 3-3.8s-1-3.5-3-4C17 6 15 4 12 4zM12 4v16"/>',
    hospital: '<path d="M4 21V5a1 1 0 0 1 1-1h14a1 1 0 0 1 1 1v16M9 21v-4h6v4M12 7v6M9 10h6"/>',
    filter: '<path d="M3 5h18l-7 8v6l-4 2v-8z"/>',
    check: '<path d="M20 6L9 17l-5-5"/>',
    clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    db: '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/>',
    zap: '<path d="M13 2L4 14h7l-1 8 9-12h-7z"/>',
    shift: '<path d="M4 7h13l-3-3M20 17H7l3 3"/>',
    target: '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
    chart: '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
    file: '<path d="M14 3H6a1 1 0 0 0-1 1v16a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V8z"/><path d="M14 3v5h5"/>',
    layers: '<path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/>',
    alert: '<path d="M12 3l10 18H2z"/><path d="M12 10v4M12 17v.5"/>',
    upload: '<path d="M12 16V4M6 10l6-6 6 6M4 20h16"/>',
  };
  const icon = (name) => `<svg viewBox="0 0 24 24">${ICONS[name] || ''}</svg>`;
  const g = (c) => `--k1:${c[0]};--k2:${c[1]}`;

  async function api(path) {
    if (!cache[path]) {
      cache[path] = fetch(path).then((r) => {
        if (!r.ok) throw new Error(`${path} : HTTP ${r.status}`);
        return r.json();
      });
    }
    return cache[path];
  }

  // value numérique => compteur animé ; texte sinon
  const kpi = (label, value, hint, ic, color, digits = 0) => {
    const num = typeof value === 'number';
    return `<div class="card kpi" style="${g(color)}">
      <div class="kpi-top"><span class="label">${label}</span><span class="ico tint">${icon(ic)}</span></div>
      <span class="value" ${num ? `data-count="${value}" data-digits="${digits}"` : String(value).length > 9 ? 'style="font-size:1.4rem"' : ''}>${num ? '0' : value}</span>
      <span class="hint">${hint || ''}</span></div>`;
  };
  const OWNERS = {
    yosra: ['Ma partie · Yosra', 'violet'], anas: ['Équipe · Anas', 'id'], fayfa: ['Équipe · Fayfa', 'ood'],
    equipe: ['Équipe', ''], mixte: ['Équipe + ma partie', 'warn'], projet: ['Projet complet', ''],
  };
  const ownerChip = (who) => { const [label, kind] = OWNERS[who] || [who, '']; return `<span class="badge owner-chip ${kind}">${label}</span>`; };
  const own = (who, title) => `${title}${ownerChip(who)}`;
  const card = (title, body, sub = '', color) =>
    `<section class="card" ${color ? `style="--dot:linear-gradient(135deg,${color[0]},${color[1]})"` : ''}><header><h2>${title}</h2>${sub ? `<p>${sub}</p>` : ''}</header>${body}</section>`;
  const notes = (sections, sub = '') => `<details class="card notes"><summary><div>Notes méthodologiques et limites<span>${sub}</span></div></summary>
    <div class="notes-body">${sections.map(([title, body]) => `<div><h3>${title}</h3>${body}</div>`).join('')}</div></details>`;
  const list = (items) => `<ul class="clean">${items.map((l) => `<li>${esc(l)}</li>`).join('')}</ul>`;
  const badge = (text, kind = '') => `<span class="badge ${kind}">${esc(text)}</span>`;

  function countUp() {
    document.querySelectorAll('[data-count]').forEach((el) => {
      const target = Number(el.dataset.count), digits = Number(el.dataset.digits || 0), t0 = performance.now();
      const step = (t) => {
        const k = Math.min(1, (t - t0) / 1100), e = 1 - Math.pow(1 - k, 3);
        el.textContent = fmt(target * e, digits);
        if (k < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    });
  }

  function table(el, rows, columns, { search = true, rowSite } = {}) {
    let sortKey = null, dir = 1, query = '';
    const draw = () => {
      let data = rows.filter((r) => !query || columns.some((c) => String(r[c.key] ?? '').toLowerCase().includes(query)));
      if (sortKey) data = [...data].sort((a, b) => {
        const x = a[sortKey], y = b[sortKey];
        if (x == null) return 1; if (y == null) return -1;
        return (typeof x === 'number' && typeof y === 'number' ? x - y : String(x).localeCompare(String(y))) * dir;
      });
      el.querySelector('tbody').innerHTML = data.map((r) =>
        `<tr${rowSite ? ` class="clickable" data-site="${esc(r[rowSite])}" title="Ouvrir la fiche du centre"` : ''}>${columns.map((c) => `<td>${c.render ? c.render(r[c.key], r) : esc(r[c.key] ?? '—')}</td>`).join('')}</tr>`).join('');
      el.querySelector('.count').textContent = `${data.length} ligne${data.length > 1 ? 's' : ''}`;
    };
    el.innerHTML = `<div class="toolbar">${search ? '<input type="search" placeholder="Rechercher…">' : ''}<span class="count"></span></div>
      <div class="table-wrap"><table><thead><tr>${columns.map((c) => `<th data-key="${c.key}">${c.label}</th>`).join('')}</tr></thead><tbody></tbody></table></div>`;
    el.querySelectorAll('th').forEach((th) => th.addEventListener('click', () => {
      dir = sortKey === th.dataset.key ? -dir : 1; sortKey = th.dataset.key;
      el.querySelectorAll('th').forEach((t) => t.removeAttribute('data-dir'));
      th.dataset.dir = dir > 0 ? 'asc' : 'desc'; draw();
    }));
    const input = el.querySelector('input');
    if (input) input.addEventListener('input', () => { query = input.value.toLowerCase(); draw(); });
    draw();
  }

  const heroArt = `<svg class="hero-art" viewBox="0 0 300 220" aria-hidden="true">
    <defs><radialGradient id="hb" cx=".5" cy=".45" r=".6"><stop offset="0" stop-color="#fff" stop-opacity=".45"/><stop offset="1" stop-color="#fff" stop-opacity=".05"/></radialGradient></defs>
    <path d="M150 28c-34 0-56 22-58 46-24 4-38 22-38 44 0 24 18 42 40 44 6 24 28 38 56 38s50-14 56-38c22-2 40-20 40-44 0-22-14-40-38-44-2-24-24-46-58-46z" fill="url(#hb)" stroke="#fff" stroke-opacity=".7" stroke-width="2.5"/>
    <path d="M150 30v170M118 70c10 8 10 22 0 30M182 70c-10 8-10 22 0 30M104 130c14-4 26 2 30 14M196 130c-14-4-26 2-30 14" stroke="#fff" stroke-opacity=".55" stroke-width="2" fill="none" stroke-linecap="round"/>
    <circle cx="178" cy="112" r="13" fill="#fde047" opacity=".9"/><circle cx="178" cy="112" r="22" fill="none" stroke="#fde047" stroke-opacity=".5" stroke-width="2"/>
    <path class="pulse-line" d="M10 196h70l14-26 18 44 16-30 10 12h152" stroke="#fff" stroke-width="3" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
  </svg>`;

  /* ---------- Vue d’ensemble ---------- */
  async function accueil() {
    const d = await api('/api/overview');
    const c = d.counts, s = d.split, b = d.best_model;
    content.innerHTML = `
      <section class="card hero"><div class="hero-grid">
        <div>
          <h2>Segmentation des lésions d’AVC sur IRM T1 et domain shift inter-centres</h2>
          <p>Un modèle entraîné dans certains hôpitaux reste-t-il fiable dans des centres qu’il n’a jamais vus ?
          Ce tableau de bord suit le projet de bout en bout : données, prétraitement distribué, mesure du décalage,
          performances des modèles et adaptation de domaine.</p>
          <div class="chips"><span>IRM T1</span><span>Spark</span><span>Wasserstein · MMD</span><span>CNN · nnU-Net</span><span>DANN · TENT</span></div>
        </div>${heroArt}</div>
      </section>
      <div class="grid g4">
        ${kpi('Cas validés et exportés', c.retained, `sur ${fmt(c.inventoried, 0)} inventoriés`, 'brain', P.violet)}
        ${kpi('Centres retenus', d.centers.retained, `sur ${fmt(d.centers.inventoried, 0)} centres`, 'hospital', P.cyan)}
        ${kpi('Exclusions documentées', c.excluded, `${fmt(c.errors, 0)} erreur de lecture`, 'filter', P.orange)}
        ${kpi('Tests automatisés', d.tests.total, d.tests.status === 'passed' ? 'tous réussis ✓' : d.tests.status, 'check', P.green)}
      </div>
      ${card('Parcours du projet', `<div class="timeline">
          <a class="step" href="#qc" style="${g(P.cyan)}"><span class="n tint">S1</span><span>Exploration</span><small>Empreinte des centres, QC initial</small></a>
          <a class="step" href="#pretraitement" style="${g(P.teal)}"><span class="n tint">S3</span><span>Prétraitement</span><small>Pipeline Spark, split v2</small></a>
          <a class="step" href="#shift" style="${g(P.orange)}"><span class="n tint">S5</span><span>Domain shift</span><small>Distances, reconnaissance du centre</small></a>
          <a class="step" href="#modeles" style="${g(P.pink)}"><span class="n tint">S7</span><span>Modèles</span><small>Dice ID / OOD, délai</small></a>
          <a class="step" href="#adaptation" style="${g(P.amber)}"><span class="n tint">S11</span><span>Adaptation</span><small>DANN, TENT, intégration</small></a>
          <a class="step" href="#demo" style="${g(P.violet)}"><span class="n tint">S12</span><span>Démo</span><small>IRM, masque, vue 3D</small></a>
        </div>`, 'Cliquez une étape pour l’ouvrir')}
      <div class="grid g3">
        ${card('Répartition du split v2', '<div id="split-chart"></div>', `${fmt(s.cases, 0)} cas`, P.violet)}
        ${card('Le prétraitement réduit le décalage', `
          <small class="muted">Distance relative médiane entre centres</small>
          <div class="big-compare"><span class="from">${fmt(d.shift.before, 3)}</span><span class="arrow">→</span><span class="to">${fmt(d.shift.after, 3)}</span></div>
          <small class="muted">Reconnaissance du centre (random forest)</small>
          <div class="big-compare"><span class="from">${pct(d.shift.fingerprint_before, 0)}</span><span class="arrow">→</span><span class="to">${pct(d.shift.fingerprint_after, 0)}</span></div>
          <p class="callout">Hasard : ${pct(d.shift.chance, 0)}. Le décalage diminue mais le centre reste reconnaissable.</p>`, '', P.orange)}
        ${card('Meilleure baseline (OOD)', b ? `
          <div class="metric-row">
            <div class="metric tint" style="${g(ID)}"><b>${fmt(b.dice_ID, 3)}</b><span>Dice ID</span></div>
            <div class="metric tint" style="${g(OOD)}"><b>${fmt(b.dice_OOD, 3)}</b><span>Dice OOD</span></div>
          </div>
          <dl class="kv"><dt>Modèle</dt><dd>${esc(b.description)}</dd>
            <dt>Écart (IC 95 %)</dt><dd>${fmt(b.gap, 3)} [${fmt(b.gap_ci_low, 3)} ; ${fmt(b.gap_ci_high, 3)}]</dd></dl>
          <p class="callout warn">Protocole historique (903 / 226 / 287), distinct du split v2.</p>` : '<p class="muted">Résultats absents.</p>', '', P.pink)}
      </div>`;
    donut($('#split-chart'), {
      parts: [
        { name: 'Train / validation', value: s.train_validation, color: P.violet[0], color2: P.violet[1] },
        { name: 'Test ID', value: s.id_test, color: ID[0], color2: ID[1] },
        { name: 'Test OOD', value: s.ood_test, color: OOD[0], color2: OOD[1] },
      ], center: fmt(s.cases, 0), sub: 'cas',
    });
  }

  /* ---------- Ma contribution ---------- */
  const md = (t) => esc(t).replace(/\*\*(.+?)\*\*/g, '<b>$1</b>').replace(/`(.+?)`/g, '<code>$1</code>');
  async function contribution() {
    const d = await api('/api/contribution');
    const links = { 'S1–S2': 'qc', S3: 'pretraitement', S4: 'pretraitement', S5: 'shift', S6: 'shift', S7: 'modeles', S8: 'modeles', 'S9–S10': 'demo', S11: 'demo', S12: 'conclusions', 'S13–S14': 'conclusions' };
    content.innerHTML = `
      <section class="card hero"><div class="hero-grid"><div>
        <h2>Mon rôle : des IRM brutes à l’analyse du domain shift</h2>
        <p>J’ai pris en charge l’exploration des données, le pipeline de prétraitement distribué, le contrôle qualité et le split,
        l’analyse du décalage entre centres, l’analyse délai / Dice, la démo IRM et l’intégration du travail de l’équipe.</p>
        <div class="chips"><span>Exploration & QC</span><span>Pipeline Spark</span><span>Split v2</span><span>Domain shift</span><span>Démo & Docker</span><span>Intégration</span></div>
      </div>${heroArt}</div></section>
      <div class="grid g4">
        ${kpi('Volumes validés et exportés', d.counts.retained, `sur ${fmt(d.counts.inventoried, 0)} inventoriés`, 'brain', P.violet)}
        ${kpi('Centres analysés', d.centers, 'statistiques et distances', 'hospital', P.cyan)}
        ${kpi('Sujets pour l’empreinte S6', d.fingerprint_scans, 'reconnaissance du centre', 'target', P.orange)}
        ${kpi('Mes tests automatisés', d.tests, 'pipeline et analyses ✓', 'check', P.green)}
      </div>
      ${card('Mon parcours semaine par semaine', `<ol class="weeks">${d.weeks.map((w) => `
        <li><a class="wk" href="#${links[w.weeks] || 'accueil'}">${esc(w.weeks)}</a><p>${md(w.text)}</p></li>`).join('')}</ol>`,
        'repris de docs/yosra/SUIVI_SEMAINES.md', P.violet)}
      ${card('Ce qui vient de l’équipe', `<ul class="todo">${d.team.map((t) =>
        `<li><span>${ownerChip(t.who === 'Anas' ? 'anas' : t.who === 'Fayfa' ? 'fayfa' : 'equipe')}</span><span>${esc(t.what)}</span><a class="more" href="#${t.page}">voir →</a></li>`).join('')}</ul>`,
        'intégré et présenté, mais pas réalisé par moi', P.blue)}`;
  }

  /* ---------- Données & QC ---------- */
  async function qc() {
    const d = await api('/api/qc');
    const c = d.summary.counts;
    const centers = [...d.centers].sort((a, b) => b.inventoried - a.inventoried);
    content.innerHTML = `
      <div class="grid g4">
        ${kpi('Masques vides', c.empty_masks, 'conservés comme cas négatifs', 'layers', P.violet)}
        ${kpi('Délai disponible', c.delay_available, `${fmt(c.delay_missing_or_invalid, 0)} absents ou invalides`, 'clock', P.cyan)}
        ${kpi('Chronicité non renseignée', c.chronicity_unspecified, 'ni aigu ni chronique par défaut', 'file', P.amber)}
        ${kpi('Alertes métadonnées', c.metadata_warning_cases, 'cas signalés', 'alert', P.red)}
      </div>
      <div class="grid g2">
        ${card('Sujets par centre', '<div class="toolbar"><button class="btn soft" id="toggle-centers" type="button">Afficher les 54 centres</button></div><div id="centers-chart"></div>', 'Retenus et exclus', P.cyan)}
        <div class="grid">
          ${card('Statut des sujets', '<div id="status-donut"></div>', `${fmt(c.inventoried, 0)} inventoriés`, P.teal)}
          ${card('Exclusions', `<ul class="clean">${d.exclusions.map((e) => `<li><b>${esc(e.site)}</b> — ${esc(e.error.replace(/_/g, ' '))} ${badge(`${e.n} cas`, 'bad')}</li>`).join('')}</ul>`, '', P.red)}
        </div>
      </div>
      ${card('Tableau QC par centre', '<div id="qc-table"></div>', 'Cliquez une ligne pour ouvrir la fiche du centre', P.blue)}
      ${notes([['Limites de lecture', list(d.summary.limits)], ['Dénominateurs', list(Object.values(d.summary.denominators))]], 'cliquer pour afficher')}`;
    let all = false;
    const draw = () => stacked($('#centers-chart'), {
      rows: (all ? centers : centers.slice(0, 15)).map((r) => ({ label: r.site, site: r.site, values: [r.retained, r.excluded] })),
      series: [{ name: 'Retenus', color: P.cyan[0], color2: P.violet[0] }, { name: 'Exclus', color: P.red[0], color2: P.red[1] }],
    });
    draw();
    $('#toggle-centers').addEventListener('click', (e) => {
      all = !all; e.target.textContent = all ? 'Afficher les 15 plus grands' : 'Afficher les 54 centres'; draw();
    });
    donut($('#status-donut'), {
      parts: [
        { name: 'Masque non vide', value: c.nonempty_masks, color: P.teal[0], color2: P.teal[1] },
        { name: 'Masque vide', value: c.empty_masks, color: P.violet[0], color2: P.violet[1] },
        { name: 'Exclus', value: c.excluded, color: P.red[0], color2: P.red[1] },
      ], center: fmt(c.retained, 0), sub: 'retenus', size: 200,
    });
    const delays = Object.fromEntries(d.delays.map((r) => [r.site, r.median_days]));
    table($('#qc-table'), d.centers.map((r) => ({ ...r, median_days: delays[r.site] ?? null })), [
      { key: 'site', label: 'Centre', render: (v) => `<b>${esc(v)}</b>` },
      { key: 'inventoried', label: 'Inventoriés' },
      { key: 'retained', label: 'Retenus' },
      { key: 'excluded', label: 'Exclus', render: (v) => (v ? badge(v, 'bad') : '0') },
      { key: 'empty_masks', label: 'Masques vides', render: (v) => (v ? badge(v, 'violet') : '0') },
      { key: 'delay_missing_or_invalid_percent', label: 'Délai manquant', render: (v) => (v == null ? '—' : badge(`${fmt(v, 1)} %`, v >= 50 ? 'warn' : v > 0 ? '' : 'ok')) },
      { key: 'median_days', label: 'Délai médian (j)', render: (v) => fmt(v, 0) },
    ], { rowSite: 'site' });
  }

  /* ---------- Prétraitement ---------- */
  async function pretraitement() {
    const d = await api('/api/preprocessing');
    const r = d.release, bench = d.benchmark;
    const steps = [
      ['Inventaire', `${fmt(d.qc.subjects, 0)} IRM T1 + masques`, P.cyan], ['QC', 'Espacement, masques, métadonnées', P.blue],
      ['Réorientation', 'RAS, grille 1 mm', P.violet], ['Normalisation', 'z-score dans le cerveau', P.pink],
      ['Split v2', 'Centres OOD tenus à l’écart', P.orange], ['Export', 'Format nnU-Net / C1', P.teal],
    ];
    content.innerHTML = `
      <div class="grid g4">
        ${kpi('Sujets traités', d.qc.subjects, `${fmt(d.qc.sites, 0)} centres`, 'brain', P.violet)}
        ${kpi('Durée du QC (min)', d.qc.seconds / 60, `${d.qc.workers} workers Spark`, 'clock', P.cyan, 1)}
        ${kpi('Export complet', d.audit.ready_for_full_export ? 'Prêt ✓' : 'Incomplet', `${fmt(d.audit.available, 0)} / ${fmt(d.audit.expected_in_splits, 0)} cas`, 'db', P.teal)}
        ${kpi('Tests unitaires', d.tests.tests, d.tests.status === 'passed' ? 'réussis ✓' : d.tests.status, 'check', P.green)}
      </div>
      ${card('Chaîne de prétraitement', `<div class="timeline">${steps.map(([t, s, c], i) =>
        `<div class="step" style="${g(c)}"><span class="n tint">${i + 1}</span><span>${t}</span><small>${s}</small></div>`).join('')}</div>`, 'Lecture → contrôle → normalisation → export', P.teal)}
      <div class="grid g2">
        ${card('Validation de l’export', `
          <div class="checks">${d.validation.checks.map((c) => badge(`✓ ${c.replace(/_/g, ' ')}`, 'ok')).join('')}</div>
          <p class="muted">Contrôlé sur ${d.validation.subjects} sujets · écart max de la moyenne cérébrale ${d.validation.max_abs_brain_mean.toExponential(1)}</p>
          <div class="metric-row">
            <div class="metric tint" style="${g(P.violet)}"><b>${fmt(r.train_validation, 0)}</b><span>Train / validation</span></div>
            <div class="metric tint" style="${g(ID)}"><b>${fmt(r.id_test, 0)}</b><span>Test ID</span></div>
            <div class="metric tint" style="${g(OOD)}"><b>${fmt(r.ood_test, 0)}</b><span>Test OOD</span></div>
          </div>
          <dl class="kv"><dt>Quarantaine</dt><dd>${r.quarantined.map(esc).join(', ')}</dd>
            <dt>Données brutes modifiées</dt><dd>${r.raw_data_modified ? 'oui' : 'non'}</dd></dl>`, '', P.green)}
        ${card('Accélération Spark', '<div id="bench"></div>', esc(d.benchmark_scope?.scope || ''), P.violet)}
      </div>
      ${card('Mesures du benchmark', '<div id="bench-table"></div>', '', P.blue)}`;
    line($('#bench'), {
      points: bench.map((b) => ({ x: b.workers, y: b.speedup, label: `${b.workers} workers : ×${fmt(b.speedup, 2)} (${fmt(b.compute_seconds, 1)} s)` })),
      ideal: bench.map((b) => ({ x: b.workers, y: b.workers })),
      color: P.violet[0], color2: P.pink[1], xLabel: 'Workers', yLabel: 'Accélération (pointillé : idéale)',
    });
    table($('#bench-table'), bench, [
      { key: 'workers', label: 'Workers', render: (v) => `<b>${v}</b>` },
      { key: 'compute_seconds', label: 'Calcul (s)', render: (v) => fmt(v, 1) },
      { key: 'speedup', label: 'Accélération', render: (v) => `×${fmt(v, 2)}` },
      { key: 'efficiency', label: 'Efficacité', render: (v) => badge(pct(v, 0), v >= 0.5 ? 'ok' : 'warn') },
    ], { search: false });
  }

  /* ---------- Domain shift ---------- */
  async function shift() {
    const d = await api('/api/shift');
    const s = d.stats, f = d.fingerprint;
    const etaLabels = { raw_mean: 'Moyenne brute', log_raw_mean: 'log moyenne', log_raw_std: 'log écart-type', z_p1: 'z p1', z_median: 'z médiane', z_p99: 'z p99', z_skew: 'z asymétrie' };
    content.innerHTML = `
      <div class="grid g4">
        ${kpi('Distance relative médiane', `${fmt(s.relative_distance_median_before, 3)} → ${fmt(s.relative_distance_median_after, 3)}`, 'avant → après prétraitement', 'shift', P.orange)}
        ${kpi('Wasserstein médian ID', s.wasserstein_median_by_group.ID, 'centres vus à l’entraînement', 'target', P.blue, 3)}
        ${kpi('Wasserstein médian OOD', s.wasserstein_median_by_group.OOD, 'centres tenus à l’écart', 'target', P.pink, 3)}
        ${kpi('Accord Wasserstein / MMD', `ρ = ${fmt(s.spearman_wasserstein_mmd, 2)}`, 'corrélation de Spearman', 'chart', P.violet)}
      </div>
      <div class="grid g2">
        ${card('Peut-on reconnaître le centre ?', '<div id="fp"></div>', `${f.scans} scans · ${f.centers} centres · bootstrap ${fmt(f.bootstrap_repeats, 0)}×`, P.orange)}
        ${card('Part de variance due au centre (η²)', '<div id="eta"></div>', 'Par statistique d’intensité', P.violet)}
      </div>
      ${card('Distance de chaque centre à la référence d’entraînement', '<div id="dist"></div>', 'Cliquez un point pour ouvrir la fiche du centre', P.cyan)}
      ${notes([['Méthode', `<p class="muted" style="font-size:.9rem">${esc(f.method)}</p><p class="callout">${esc(f.scope)}</p>`], ['Limites', list(f.limitations)]], 'cliquer pour afficher')}`;
    const names = { avant_32: 'avant', apres_30: 'après', apres_150: 'après (150 bins)', controle: 'contrôle mélangé' };
    intervals($('#fp'), {
      items: f.scores.map((x) => {
        const col = x.representation === 'controle' ? ['#94a3b8', '#64748b'] : x.representation.startsWith('avant') ? P.orange : P.teal;
        return { label: `${x.model === 'logreg' ? 'Rég. logistique' : 'Random forest'} · ${names[x.representation] || x.representation}`,
          value: x.balanced_accuracy, low: x.ci_low, high: x.ci_high, color: col[1], color2: col[0] };
      }),
      ref: f.chance, refLabel: 'hasard', format: (v) => pct(v, 0), labelWidth: 210,
    });
    const keys = Object.keys(s.eta_squared);
    bars($('#eta'), {
      labels: keys.map((k) => etaLabels[k] || k),
      series: [{ name: 'η²', color: P.pink[0], color2: P.violet[1], values: keys.map((k) => s.eta_squared[k]) }],
      yMax: 1, height: 300, wrap: 9,
    });
    scatter($('#dist'), {
      points: d.distances.map((r) => ({ site: r.site, x: r.n, y: r.wasserstein, color: r.group === 'OOD' ? OOD[1] : ID[1], r: 5 + Math.sqrt(r.n) / 1.8,
        label: `${r.site} (${r.group}) · n = ${r.n} · W = ${fmt(r.wasserstein, 3)} · MMD = ${fmt(r.mmd, 3)}` })),
      series: [{ name: 'Centre ID', color: ID[0], color2: ID[1] }, { name: 'Centre OOD', color: OOD[0], color2: OOD[1] }],
      xLabel: 'Nombre de scans du centre', yLabel: 'Distance de Wasserstein', logX: true, height: 260, format: (v) => fmt(v, 2),
    });
  }

  /* ---------- Modèles ---------- */
  async function modeles() {
    const d = await api('/api/models');
    const m = d.baselines, t = d.temporal.temporal_by_role;
    const short = (r) => r.model.replace('cnn_pretrained_freeze-', 'CNN pré-entraîné ').replace('cnn_random_init', 'CNN aléatoire')
      .replace('ml_', 'ML ').replace(/_/g, ' ').replace('logistic regression', 'régression log.').replace('gradient boosting', 'gradient boost.');
    content.innerHTML = `
      <p class="callout warn" style="margin:0">Résultats réels des baselines d’Anas, sur le <b>protocole historique</b> (903 train / 226 ID / 287 OOD).
        Ils ne remplacent pas les cinq plis du split v2, en attente des checkpoints.</p>
      ${card(own('anas', 'Dice par modèle : centres connus (ID) vs inconnus (OOD)'), '<div id="dice"></div>', `${m[0]?.n_ID ?? '—'} cas ID · ${m[0]?.n_OOD ?? '—'} cas OOD`, P.pink)}
      <div class="grid g2">
        ${card(own('anas', 'Écart de Dice ID − OOD (IC 95 %)'), '<div id="gap"></div>', 'Négatif = meilleur sur OOD', P.violet)}
        ${card(own('yosra', 'Le délai depuis l’AVC explique-t-il le Dice ?'), `<div id="r2"></div>
          <div class="metric-row" style="margin-top:18px">
            <div class="metric tint" style="${g(ID)}"><b>${fmt(t.ID.incremental_r2_time, 4)}</b><span>ID · R² ajouté par le temps (p = ${fmt(t.ID.permutation_p_time_given_center, 3)})</span></div>
            <div class="metric tint" style="${g(OOD)}"><b>${fmt(t.OOD.incremental_r2_time, 3)}</b><span>OOD · R² ajouté par le temps (p = ${fmt(t.OOD.permutation_p_time_given_center, 3)})</span></div>
          </div>
          <p class="callout">En ID, c’est le centre qui explique la variation du Dice (R² ajouté ${fmt(t.ID.incremental_r2_center, 3)}) ; le délai n’ajoute rien.
          En OOD, un faible effet du délai apparaît, mais il disparaît une fois le volume lésionnel contrôlé (${fmt(t.OOD.volume_sensitivity.incremental_r2_time_given_center_and_volume, 4)}).</p>`, 'R² par groupe', P.amber)}
      </div>
      ${card(own('anas', 'Toutes les métriques'), '<div id="metrics"></div>', '', P.blue)}
      ${notes([['Limites de l’analyse temporelle', list(d.temporal.limitations || [])]], 'cliquer pour afficher')}`;
    bars($('#dice'), {
      labels: m.map(short),
      series: [{ name: 'Dice ID', color: ID[0], color2: ID[1], values: m.map((r) => r.dice_ID) },
               { name: 'Dice OOD', color: OOD[0], color2: OOD[1], values: m.map((r) => r.dice_OOD) }],
      yMax: 0.6, height: 320, format: (v) => fmt(v, 2), wrap: 13,
    });
    intervals($('#gap'), {
      items: m.map((r) => ({ label: short(r), value: r.gap, low: r.gap_ci_low, high: r.gap_ci_high,
        color: (r.gap_ci_high < 0 ? P.pink : P.violet)[1], color2: (r.gap_ci_high < 0 ? P.pink : P.violet)[0] })),
      min: -0.15, max: 0.1, ref: 0, refLabel: '0', tickCount: 5, labelWidth: 190,
    });
    bars($('#r2'), {
      labels: ['Centre seul', 'Temps seul', 'Centre + temps'],
      series: ['ID', 'OOD'].map((r) => ({ name: r, color: (r === 'ID' ? ID : OOD)[0], color2: (r === 'ID' ? ID : OOD)[1], values: [t[r].r2.center, t[r].r2.time, t[r].r2.both] })),
      yMax: 0.4, height: 280, format: (v) => fmt(v, 2),
    });
    table($('#metrics'), m.map((r) => ({ ...r, name: short(r) })), [
      { key: 'name', label: 'Modèle', render: (v) => `<b>${esc(v)}</b>` },
      { key: 'dice_ID', label: 'Dice ID', render: (v) => fmt(v, 3) },
      { key: 'dice_OOD', label: 'Dice OOD', render: (v) => fmt(v, 3) },
      { key: 'gap', label: 'Écart', render: (v) => fmt(v, 3) },
      { key: 'hd95_ID', label: 'HD95 ID (mm)', render: (v) => fmt(v, 1) },
      { key: 'hd95_OOD', label: 'HD95 OOD (mm)', render: (v) => fmt(v, 1) },
      { key: 'lesion_f1_ID', label: 'F1 lésion ID', render: (v) => fmt(v, 3) },
      { key: 'lesion_f1_OOD', label: 'F1 lésion OOD', render: (v) => fmt(v, 3) },
    ], { search: false });
  }

  /* ---------- Adaptation ---------- */
  async function adaptation() {
    const d = await api('/api/adaptation');
    const v = d.verification, exp = Object.entries(d.readiness?.experiments || {});
    content.innerHTML = `
      <div class="grid g4">
        ${kpi('Tests Yosra', v.yosra_tests.passed, 'pipeline et analyses', 'check', P.violet)}
        ${kpi('Généralisation + adaptation', v.generalization_adaptation_without_calibration.passed, 'tests, dont vrais patches C1', 'layers', P.cyan)}
        ${kpi('Tests calibration', v.calibration_tests.passed, 'recalculée sur 916 sujets C1', 'target', P.amber)}
        ${kpi('Total', v.total_passed, v.status === 'passed' ? 'tous réussis ✓' : v.status, 'zap', P.green)}
      </div>
      <div class="grid g2">
        ${card(own('fayfa', 'DANN — adaptation adversariale'), `
          <p>Un classifieur de centre est branché sur l’encodeur via une <b>couche d’inversion de gradient</b> :
          le réseau apprend des représentations utiles à la segmentation mais peu informatives sur le centre.</p>
          <p class="callout">Module <code>adaptation.dann.DANNWrapper</code> — utilise les centres du train de chaque pli.</p>`, '', P.pink)}
        ${card(own('fayfa', 'TENT — adaptation au moment du test'), `
          <p>Sur un nouveau centre, seuls les paramètres de normalisation sont ajustés en minimisant l’entropie des prédictions,
          <b>sans étiquette</b>. Le modèle est réinitialisé entre centres.</p>
          <p class="callout">Module <code>adaptation.tent.Tent</code> — restreint au masque cérébral.</p>`, '', P.amber)}
      </div>
      <div class="grid g2">
        ${card(own('fayfa', 'Centres tenus à l’écart (OOD)'), `<div class="checks">${d.held_out_sites.map((s) => badge(s, 'ood')).join('')}</div>
          <p class="muted">${d.training_sites} centres d’entraînement · split v${d.splits?.version} · empreinte ${esc(d.splits?.sha256_16 || '')}</p>`, '', P.orange)}
        ${card(own('yosra', 'État de l’intégration'), `
          <dl class="kv">
            <dt>Branches fusionnées</dt><dd>${Object.keys(v.branches).map((b) => badge(b, 'violet')).join(' ')}</dd>
            <dt>Calibration</dt><dd>recalculée sur le split v${v.calibration.split_version}</dd>
            <dt>pip check</dt><dd>${badge(v.pip_check === 'passed' ? 'réussi' : v.pip_check, 'ok')}</dd>
            <dt>Checkpoints entraînés</dt><dd>${v.trained_checkpoints_present ? badge('présents', 'ok') : badge('absents', 'warn')}</dd>
          </dl>`, '', P.violet)}
      </div>
      ${resultsCard(d)}
      ${card(own('yosra', 'Expériences restantes (C1 v2)'), `
        <div class="checks">${exp.map(([k, e]) => badge(`${k.replace('baseline_fold_', 'Pli ')} : ${e.status === 'missing' ? 'en attente' : e.status}`, e.status === 'missing' ? 'warn' : 'ok')).join('')}</div>
        <p class="callout warn">${esc(d.readiness?.scope || '')}. Aucun Dice DANN/TENT n’est affiché tant que les poids entraînés ne sont pas fournis.</p>`, '', P.amber)}`;
  }

  function resultsCard(d) {
    const r = d.results;
    if (!r) {
      return card(own('equipe', 'Comparaison baseline · DANN · TENT'), `
        <div class="empty">${icon('chart')}<span><b>En attente des résultats entraînés.</b><br>
        Dès que le fichier <code>${esc(d.expected.path)}</code> est livré, cette section affiche automatiquement
        le Dice ID / OOD de chaque technique.</span></div>
        <p class="callout">Colonnes attendues : <code>${d.expected.columns.join(', ')}</code> — <code>technique</code> = baseline, dann ou tent.
        Le rôle ID / OOD est pris dans le split v2 actif ; les cas de validation sont ignorés.</p>`, 'aucun chiffre simulé', P.violet);
    }
    if (r.status !== 'ready') {
      return card(own('equipe', 'Comparaison baseline · DANN · TENT'), `<p class="status err">${esc(r.reason || r.status)}</p>`, esc(r.path), P.red);
    }
    setTimeout(() => {
      const val = (t, role) => (r.summary.find((x) => x.technique === t && x.role === role) || {}).mean_dice ?? null;
      bars($('#tech-chart'), {
        labels: r.techniques.map((t) => t.toUpperCase()),
        series: [{ name: 'Dice ID', color: ID[0], color2: ID[1], values: r.techniques.map((t) => val(t, 'ID')) },
                 { name: 'Dice OOD', color: OOD[0], color2: OOD[1], values: r.techniques.map((t) => val(t, 'OOD')) }],
        yMax: 1, height: 280, format: (v) => fmt(v, 3),
      });
      table($('#tech-table'), r.by_fold, [
        { key: 'technique', label: 'Technique', render: (v) => `<b>${esc(String(v).toUpperCase())}</b>` },
        { key: 'fold', label: 'Pli' },
        { key: 'role', label: 'Rôle', render: (v) => badge(v, v === 'OOD' ? 'ood' : 'id') },
        { key: 'n', label: 'Cas' },
        { key: 'mean_dice', label: 'Dice moyen', render: (v) => fmt(v, 3) },
      ], { search: false });
    });
    return card(own('equipe', 'Comparaison baseline · DANN · TENT'), `<div id="tech-chart"></div><div id="tech-table" style="margin-top:18px"></div>
      <p class="callout">Source : <code>${esc(r.path)}</code> · empreinte ${esc(r.sha256_16)} · ${fmt(r.ignored, 0)} lignes de validation ou invalides ignorées.</p>`,
      `${r.techniques.length} technique(s)`, P.violet);
  }

  /* ---------- Fiche centre (tiroir) ---------- */
  const drawer = $('#drawer'), drawerScrim = $('#drawer-scrim');
  let centerList = [];
  function closeDrawer() { drawer.classList.remove('open'); drawer.setAttribute('aria-hidden', 'true'); drawerScrim.hidden = true; }

  async function openCenter(site) {
    const d = await api('/api/centers');
    centerList = d.centers;
    const i = centerList.findIndex((c) => c.site === site);
    if (i < 0) return;
    const c = centerList[i], q = c.qc, dist = c.distance, st = c.stats, dl = c.delay;
    const role = c.role === 'OOD' ? badge('Centre OOD · tenu à l’écart', 'ood') : c.role === 'Exclu' ? badge('Centre exclu', 'bad') : badge('Centre ID · entraînement', 'id');
    const maxW = Math.max(...centerList.map((x) => x.distance?.wasserstein || 0));
    const pos = (v) => `${Math.max(0, Math.min(100, (v / maxW) * 100))}%`;
    const prev = centerList[(i - 1 + centerList.length) % centerList.length].site, next = centerList[(i + 1) % centerList.length].site;
    drawer.innerHTML = `
      <header class="drawer-head">
        <div><span class="eyebrow">Fiche centre</span><h2>${esc(c.site)}</h2>${role}</div>
        <button class="icon-btn" id="drawer-close" type="button" aria-label="Fermer">✕</button>
      </header>
      <div class="drawer-body">
        <div class="metric-row">
          <div class="metric tint" style="${g(P.violet)}"><b>${fmt(q.inventoried, 0)}</b><span>Inventoriés</span></div>
          <div class="metric tint" style="${g(P.teal)}"><b>${fmt(q.retained, 0)}</b><span>Retenus</span></div>
          <div class="metric tint" style="${g(P.red)}"><b>${fmt(q.excluded, 0)}</b><span>Exclus</span></div>
        </div>
        <section><h3>Distance à la référence d’entraînement</h3>
          ${dist ? `
            <div class="track"><span class="median" style="left:${pos(d.median_wasserstein)}" title="Médiane des centres"></span>
              <span class="dot" style="left:${pos(dist.wasserstein)}"></span></div>
            <div class="track-legend"><span>proche</span><span>médiane ${fmt(d.median_wasserstein, 3)}</span><span>éloigné</span></div>
            <dl class="kv">
              <dt>Wasserstein</dt><dd>${fmt(dist.wasserstein, 3)} · ${dist.rank}<sup>e</sup> sur ${dist.of}</dd>
              <dt>MMD</dt><dd>${fmt(dist.mmd, 3)}</dd>
              <dt>Distance relative</dt><dd>${fmt(dist.relative_before, 3)} → ${fmt(dist.relative_after, 3)} <span class="muted">(avant → après)</span></dd>
            </dl>` : '<p class="muted">Pas de distance calculée pour ce centre.</p>'}
        </section>
        <section><h3>Contrôle qualité</h3>
          <dl class="kv">
            <dt>Masques vides</dt><dd>${fmt(q.empty_masks, 0)}</dd>
            <dt>Délai manquant</dt><dd>${q.delay_missing_or_invalid_percent == null ? '—' : `${fmt(q.delay_missing_or_invalid_percent, 1)} %`}</dd>
            <dt>Chronicité non renseignée</dt><dd>${fmt(q.chronicity_unspecified, 0)}</dd>
            <dt>Alertes métadonnées</dt><dd>${fmt(q.metadata_warning_cases, 0)}</dd>
          </dl>
        </section>
        <section><h3>Délai depuis l’AVC</h3>
          ${dl.median != null ? `<dl class="kv">
            <dt>Médiane</dt><dd>${days(dl.median)}</dd>
            <dt>Étendue</dt><dd>${fmt(dl.min, 0)} – ${days(dl.max)}</dd>
            <dt>Disponible</dt><dd>${fmt(dl.available, 0)} sujets</dd></dl>` : '<p class="muted">Aucun délai valide pour ce centre.</p>'}
        </section>
        ${st ? `<section><h3>Volumes (médianes du centre)</h3>
          <dl class="kv"><dt>Cerveau</dt><dd>${fmt(st.brain_ml, 0)} mL</dd><dt>Lésion</dt><dd>${fmt(st.lesion_ml, 1)} mL</dd></dl></section>` : ''}
      </div>
      <footer class="drawer-foot">
        <button class="btn soft" type="button" data-site="${esc(prev)}">← ${esc(prev)}</button>
        <button class="btn soft" type="button" data-site="${esc(next)}">${esc(next)} →</button>
      </footer>`;
    drawer.classList.add('open'); drawer.setAttribute('aria-hidden', 'false'); drawerScrim.hidden = false;
    $('#drawer-close').addEventListener('click', closeDrawer);
  }

  /* ---------- Conclusions ---------- */
  async function conclusions() {
    const d = await api('/api/conclusions');
    const c = d.counts, s = d.shift, b = d.best, t = d.temporal;
    const excl = d.exclusions.map((e) => `${e.site} (${e.n})`).join(', ');
    const msg = (n, color, title, big, text, link) => `
      <a class="card message" href="#${link}" style="${g(color)}">
        <span class="num tint">${n}</span>
        <div><h3>${title}</h3><div class="big">${big}</div><p>${text}</p><span class="more">Voir le détail →</span></div>
      </a>`;
    const exp = Object.entries(d.experiments || {});
    const missingFolds = exp.filter(([, e]) => e.status !== 'ready' && e.status !== 'completed').length;
    const todo = [
      [`Baselines C1 v2 sur les ${exp.length || 5} plis`, missingFolds ? `${missingFolds} pli(s) en attente` : 'livrés', !missingFolds],
      ['Checkpoints entraînés fournis', d.checkpoints ? 'présents' : 'absents', !!d.checkpoints],
      ['Évaluation DANN et TENT', d.adaptation_results?.status === 'ready' ? `${d.adaptation_results.techniques.join(', ')}` : 'en attente des poids', d.adaptation_results?.status === 'ready'],
      ['Revue du split v2 par l’équipe', d.team_review_pending ? 'en attente' : 'faite', !d.team_review_pending],
    ];
    content.innerHTML = `
      <div class="grid g2 messages">
        ${msg(1, P.violet, 'Une cohorte large et contrôlée', `${fmt(c.retained, 0)} IRM · ${fmt(d.centers, 0)} centres`,
          `${fmt(c.excluded, 0)} exclusions documentées : ${esc(excl)}. Les ${fmt(c.empty_masks, 0)} masques vides sont conservés comme cas négatifs.`, 'qc')}
        ${msg(2, P.teal, 'Le prétraitement réduit le décalage', `${fmt(s.before, 3)} → ${fmt(s.after, 3)}`,
          `La distance relative médiane entre centres est divisée par ${fmt(s.before / s.after, 1)} après normalisation.`, 'shift')}
        ${msg(3, P.orange, '… sans l’effacer', `${pct(s.fp_before, 0)} → ${pct(s.fp_after, 0)}`,
          `Le centre reste reconnaissable à partir de l’intensité (hasard : ${pct(s.chance, 0)}) ; η² atteint encore ${fmt(s.eta_max_after, 2)} sur les statistiques normalisées.`, 'shift')}
        ${msg(4, P.pink, 'Pas de chute mesurée en OOD', `Dice ${fmt(b.dice_ID, 2)} (ID) · ${fmt(b.dice_OOD, 2)} (OOD)`,
          `Meilleure baseline (${esc(b.description || '')}) : écart ${fmt(b.gap, 3)}, IC 95 % [${fmt(b.gap_ci_low, 3)} ; ${fmt(b.gap_ci_high, 3)}] contenant 0. Protocole historique.`, 'modeles')}
        ${msg(5, P.amber, 'Le délai depuis l’AVC n’explique pas le Dice', `p = ${fmt(t.ID?.permutation_p_time_given_center, 3)} (ID)`,
          `En ID, le centre explique la variation ; en OOD, le faible effet du délai disparaît une fois le volume lésionnel contrôlé.`, 'modeles')}
        ${msg(6, P.cyan, 'L’adaptation de domaine est prête', `${fmt(d.tests, 0)} tests réussis`,
          `DANN et TENT sont intégrés et testés sur les vrais patches C1 ; il manque les poids entraînés pour mesurer leur effet.`, 'adaptation')}
      </div>
      ${card('Ce qui reste à faire', `<ul class="todo">${todo.map(([label, state, done]) =>
        `<li class="${done ? 'done' : ''}"><span class="check">${done ? '✓' : ''}</span><span>${label}</span>${badge(state, done ? 'ok' : 'warn')}</li>`).join('')}</ul>`,
        'mis à jour automatiquement depuis les rapports', P.amber)}`;
  }

  /* ---------- Démo IRM ---------- */
  let niivue = null;
  async function getNiivue() {
    if (niivue) return niivue;
    const { esm } = await import('/vendor/niivue.es.js');
    const { Niivue } = await import('data:text/javascript;charset=utf-8,' + esm);
    niivue = new Niivue({ backColor: [0.06, 0.04, 0.16, 1] });
    await niivue.attachTo('nv');
    return niivue;
  }

  const drop = (id, label, hint, ic, color, required) => `
    <div class="drop" style="${g(color)}"><span class="ico tint">${icon(ic)}</span><div class="txt">
      <label for="${id}">${label}${required ? ' *' : ''}</label><small id="${id}-name">${hint}</small></div>
      <input id="${id}" name="${id}" type="file" accept=".nii,.gz" ${required ? 'required' : ''}></div>`;

  async function demo() {
    niivue = null;
    content.innerHTML = `
      <div class="grid g2">
        ${card('Analyser une IRM', `
          <form class="form" id="form">
            ${drop('t1', 'IRM T1', 'Cliquez ou déposez un .nii / .nii.gz', 'brain', P.violet, true)}
            ${drop('mask', 'Masque lésionnel', 'Même grille que l’IRM · facultatif si un modèle est branché', 'target', P.pink, false)}
            ${drop('uncertainty', 'Carte d’incertitude', 'Facultative', 'layers', P.cyan, false)}
            <div class="range"><label for="pos"><b>Position des coupes</b></label><input id="pos" type="range" min="0" max="1" step="0.01" value="0.5"><span id="pos-v" class="badge violet">50 %</span></div>
            <div><button class="btn" id="go" type="submit">${icon('zap').replace('<svg', '<svg width="18" height="18" style="fill:none;stroke:#fff;stroke-width:2"')}Analyser</button></div>
            <p class="status" id="status">Les fichiers sont envoyés au serveur local uniquement, puis supprimés.</p>
          </form>`, '', P.violet)}
        ${card('Rapport', `<div id="report"><div class="empty">${icon('file')}<span>Aucune analyse pour l’instant.<br>Chargez une IRM et son masque puis cliquez « Analyser ».</span></div></div>`, '', P.pink)}
      </div>
      ${card('Coupes et lésion', `<div id="slices" class="slices"><div class="empty">${icon('brain')}<span>Les coupes sagittale, coronale et axiale apparaîtront ici.</span></div></div>`, '', P.orange)}
      ${card('Vue 3D interactive (NiiVue)', `
        <div class="toolbar"><button class="btn soft" id="mode" type="button">Basculer coupes / 3D</button><span id="nv-status" class="status">Chargée dans le navigateur dès que vous choisissez les fichiers.</span></div>
        <canvas id="nv" class="viewer"></canvas>`, 'WebGL2 requis', P.cyan)}`;

    const pos = $('#pos');
    pos.addEventListener('input', () => { $('#pos-v').textContent = `${Math.round(pos.value * 100)} %`; });
    let render = false;
    $('#mode').addEventListener('click', async () => {
      try { const nv = await getNiivue(); render = !render; nv.setSliceType(render ? nv.sliceTypeRender : nv.sliceTypeMultiplanar); } catch (e) { $('#nv-status').textContent = String(e); }
    });

    let urls = [];
    const show3d = async () => {
      const image = $('#t1').files[0], mask = $('#mask').files[0];
      if (!image) return;
      try {
        const nv = await getNiivue();
        urls.forEach(URL.revokeObjectURL); urls = [];
        const vol = (file, opts = {}) => { const url = URL.createObjectURL(file); urls.push(url); return { url, name: file.name, ...opts }; };
        const volumes = [vol(image)];
        if (mask) volumes.push(vol(mask, { colormap: 'red', opacity: 0.6 }));
        await nv.loadVolumes(volumes);
        $('#nv-status').textContent = 'Chargé. Vérifiez visuellement l’alignement du masque.';
      } catch (e) {
        $('#nv-status').textContent = `Vue 3D indisponible (lancer demo/setup_viewer.py) : ${e.message || e}`;
      }
    };
    ['t1', 'mask', 'uncertainty'].forEach((id) => $(`#${id}`).addEventListener('change', (e) => {
      const f = e.target.files[0];
      if (f) { $(`#${id}-name`).textContent = `✓ ${f.name}`; e.target.closest('.drop').classList.add('filled'); }
      if (id !== 'uncertainty') show3d();
    }));

    $('#form').addEventListener('submit', async (e) => {
      e.preventDefault();
      const status = $('#status'), go = $('#go');
      const body = new FormData();
      ['t1', 'mask', 'uncertainty'].forEach((k) => { const f = $(`#${k}`).files[0]; if (f) body.append(k, f); });
      go.disabled = true; status.className = 'status'; status.textContent = 'Analyse en cours…';
      try {
        const r = await fetch(`/api/analyze?position=${pos.value}`, { method: 'POST', body });
        const data = await r.json();
        if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
        $('#slices').innerHTML = `<img alt="Coupes sagittale, coronale et axiale avec la lésion en surimpression" src="data:image/png;base64,${data.slices_png}">`;
        const u = data.uncertainty;
        const { slices_png, ...json } = data;
        $('#report').innerHTML = `
          <div class="metric-row">
            <div class="metric tint" style="${g(P.pink)}"><b>${fmt(data.lesion_volume_ml, 2)} mL</b><span>Volume lésionnel</span></div>
            <div class="metric tint" style="${g(P.violet)}"><b>${fmt(data.lesion_voxels, 0)}</b><span>Voxels lésionnels</span></div>
          </div>
          <dl class="kv">
            <dt>Provenance</dt><dd>${esc(data.provenance)}</dd>
            <dt>Dimensions</dt><dd>${data.shape.join(' × ')}</dd>
            <dt>Taille voxel</dt><dd>${data.voxel_mm.join(' × ')} mm</dd>
            ${u ? `<dt>Incertitude moyenne</dt><dd>${fmt(u.mean_volume, 4)} (lésion : ${fmt(u.mean_lesion, 4)})</dd>` : ''}
          </dl>
          <p class="callout warn">${esc(data.usage)}</p>
          <a class="btn" style="margin-top:12px" download="rapport_avc.json" href="data:application/json;charset=utf-8,${encodeURIComponent(JSON.stringify(json, null, 2))}">Télécharger le rapport JSON</a>`;
        status.className = 'status okk'; status.textContent = 'Analyse terminée ✓';
      } catch (err) {
        status.className = 'status err'; status.textContent = err.message;
      } finally { go.disabled = false; }
    });
  }

  /* ---------- Galerie ---------- */
  async function galerie() {
    const figs = await api('/api/figures');
    const groups = ['Tout', ...new Set(figs.map((f) => f.group))];
    const kinds = { Exploration: 'id', QC: 'violet', 'Prétraitement': 'ok', 'Domain shift': 'ood', 'Modèles': 'bad', Temps: 'warn' };
    content.innerHTML = `<div class="filters" id="filters">${groups.map((x, i) => `<button type="button" class="${i ? '' : 'on'}" data-g="${esc(x)}">${esc(x)}</button>`).join('')}</div><div class="gallery" id="gallery"></div>`;
    const draw = (sel) => {
      $('#gallery').innerHTML = figs.filter((f) => sel === 'Tout' || f.group === sel).map((f) =>
        `<figure data-src="/files/${f.path}" data-cap="${esc(f.caption)}"><img loading="lazy" src="/files/${f.path}" alt="${esc(f.caption)}"><figcaption>${badge(f.group, kinds[f.group] || '')}${esc(f.caption)}</figcaption></figure>`).join('');
    };
    draw('Tout');
    $('#filters').addEventListener('click', (e) => {
      const b = e.target.closest('button'); if (!b) return;
      document.querySelectorAll('#filters button').forEach((x) => x.classList.toggle('on', x === b)); draw(b.dataset.g);
    });
    $('#gallery').addEventListener('click', (e) => {
      const f = e.target.closest('figure'); if (!f) return;
      const box = $('#lightbox'); box.querySelector('img').src = f.dataset.src; box.querySelector('p').textContent = f.dataset.cap; box.hidden = false;
    });
  }

  /* ---------- Routage ---------- */
  const pages = {
    accueil: ['Projet AVC', 'Vue d’ensemble', accueil, 'Le projet en un coup d’œil : données, résultats clés et parcours.', 'projet'],
    contribution: ['Yosra', 'Ma contribution', contribution, 'Ce que j’ai réalisé dans le projet, semaine par semaine.', 'yosra'],
    qc: ['S1–S2 · Exploration', 'Données & contrôle qualité', qc, 'Ce que contient la cohorte et ce qui a été écarté, centre par centre.', 'yosra'],
    pretraitement: ['S3–S4 · Pipeline', 'Prétraitement distribué', pretraitement, 'De l’IRM brute au volume normalisé, avec Spark.', 'yosra'],
    shift: ['S5–S6 · Analyse', 'Domain shift entre centres', shift, 'À quel point les centres diffèrent, avant et après normalisation.', 'yosra'],
    modeles: ['S7–S10 · Évaluation', 'Modèles & Dice', modeles, 'Performances des baselines sur les centres connus et inconnus.', 'mixte'],
    adaptation: ['S11–S12 · Généralisation', 'Adaptation de domaine', adaptation, 'DANN et TENT : prêts, testés, en attente des poids entraînés.', 'mixte'],
    demo: ['S12–S14 · Démo', 'Démo IRM', demo, 'Chargez une IRM et son masque pour obtenir coupes, volume et vue 3D.', 'yosra'],
    conclusions: ['Synthèse', 'Conclusions', conclusions, 'Les messages clés du projet et ce qui reste à faire.', 'projet'],
    galerie: ['Figures', 'Galerie des figures', galerie, 'Toutes les figures du projet, cliquez pour agrandir.', 'projet'],
  };

  const sidebar = $('#sidebar'), scrim = $('#scrim');
  const closeMenu = () => { sidebar.classList.remove('open'); scrim.classList.remove('show'); };

  async function route() {
    const key = pages[location.hash.slice(1)] ? location.hash.slice(1) : 'accueil';
    const [eyebrow, title, render, subtitle, owner] = pages[key];
    $('#page-owner').innerHTML = owner ? ownerChip(owner) : '';
    $('#page-eyebrow').textContent = eyebrow; $('#page-title').textContent = title; $('#page-sub').textContent = subtitle;
    document.title = `${title} · Projet AVC`;
    document.querySelectorAll('#nav a').forEach((a) => a.classList.toggle('active', a.dataset.page === key));
    closeMenu();
    content.innerHTML = '<div class="skeleton" style="height:180px"></div><div class="grid g4"><div class="skeleton"></div><div class="skeleton"></div><div class="skeleton"></div><div class="skeleton"></div></div>';
    try { await render(); countUp(); } catch (e) {
      content.innerHTML = card('Impossible de charger cette section', `<p class="status err">${esc(e.message)}</p>`, '', P.red);
    }
    window.scrollTo(0, 0);
  }

  // Redessine les graphiques à la bonne largeur après un redimensionnement
  let lastW = window.innerWidth, timer;
  window.addEventListener('resize', () => {
    clearTimeout(timer);
    timer = setTimeout(() => { if (Math.abs(window.innerWidth - lastW) > 80 && location.hash !== '#demo') { lastW = window.innerWidth; route(); } }, 300);
  });

  // Thème
  const root = document.documentElement;
  try { const t = localStorage.getItem('avc-theme'); if (t) root.dataset.theme = t; } catch (_) { /* stockage indisponible */ }
  $('#theme').addEventListener('click', () => {
    const dark = root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
    root.dataset.theme = dark ? 'light' : 'dark';
    try { localStorage.setItem('avc-theme', root.dataset.theme); } catch (_) { /* ignore */ }
  });
  $('#menu').addEventListener('click', () => { sidebar.classList.add('open'); scrim.classList.add('show'); });
  scrim.addEventListener('click', closeMenu);
  $('#lightbox').addEventListener('click', (e) => { e.currentTarget.hidden = true; });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') { $('#lightbox').hidden = true; closeMenu(); closeDrawer(); } });
  window.addEventListener('hashchange', () => { closeDrawer(); route(); });
  document.addEventListener('click', (e) => {
    const el = e.target.closest('[data-site]');
    if (el) openCenter(el.getAttribute('data-site'));
  });
  drawerScrim.addEventListener('click', closeDrawer);
  route();
})();
