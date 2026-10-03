"""A small, local-only browser UI for Neural's model advisor."""

from __future__ import annotations

import json
import ipaddress
import math
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlencode, urlsplit


_PAGE = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <title>Neural model advisor</title>
  <style>
    :root { color-scheme: light dark; --bg:#f4f6f8; --panel:#fff; --ink:#17212b; --muted:#596979; --line:#d9e0e6; --accent:#235f52; --accent-soft:#e4f2ed; --warn:#80530b; --warn-soft:#fff3d5; --bad:#922f39; --bad-soft:#fae8e9; }
    @media (prefers-color-scheme: dark) { :root { --bg:#10171d; --panel:#18232c; --ink:#e7edf1; --muted:#aab7c1; --line:#354550; --accent:#8ed4ba; --accent-soft:#213c36; --warn:#f2ca73; --warn-soft:#43371d; --bad:#f19ca2; --bad-soft:#43272b; } }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
    header { border-bottom:1px solid var(--line); background:var(--panel); }
    .wrap { width:min(1120px,calc(100% - 36px)); margin:0 auto; }
    .head { display:flex; align-items:center; justify-content:space-between; gap:20px; padding:24px 0; }
    .brand { display:flex; align-items:center; gap:13px; }
    .mark { width:38px; height:38px; display:grid; place-items:center; background:var(--accent); color:var(--panel); border-radius:12px; font-weight:800; font-size:18px; }
    h1 { margin:0; font-size:22px; line-height:1.15; letter-spacing:-.03em; }
    .sub { margin:5px 0 0; color:var(--muted); font-size:13px; }
    .local { color:var(--muted); font-size:12px; white-space:nowrap; }
    main { padding:28px 0 60px; }
    .controls,.panel,.hero,.notice { background:var(--panel); border:1px solid var(--line); border-radius:16px; box-shadow:none; }
    .controls { padding:20px; }
    .control-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; align-items:end; }
    .advanced { margin-top:14px; border-top:1px solid var(--line); padding-top:12px; }
    .advanced summary { cursor:pointer; color:var(--muted); font-size:13px; font-weight:700; }
    .advanced-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; margin-top:12px; }
    .control-actions { display:flex; justify-content:flex-end; margin-top:14px; }
    label { display:block; font-size:12px; color:var(--muted); font-weight:650; margin-bottom:6px; }
    select,input { width:100%; color:var(--ink); background:var(--bg); border:1px solid var(--line); border-radius:9px; padding:10px 11px; font:inherit; min-height:42px; }
    input::placeholder { color:var(--muted); opacity:.75; }
    button { border:0; border-radius:9px; min-height:42px; padding:10px 18px; font:inherit; font-weight:700; cursor:pointer; color:#fff; background:#235f52; }
    @media (prefers-color-scheme:dark) { button { color:#0e1915; background:#8ed4ba; } }
    button:disabled { opacity:.6; cursor:wait; }
    .help { margin:12px 0 0; color:var(--muted); font-size:12px; }
    #error { margin:14px 0 0; color:var(--bad); font-weight:600; }
    .result-top { display:flex; justify-content:space-between; align-items:end; gap:12px; margin:26px 2px 12px; }
    h2 { margin:0; font-size:18px; letter-spacing:-.02em; }
    .stamp { color:var(--muted); font-size:12px; }
    .notice { padding:12px 15px; margin:10px 0; color:var(--warn); background:var(--warn-soft); box-shadow:none; }
    .hero { border-color:color-mix(in srgb,var(--accent) 45%,var(--line)); padding:22px; margin:12px 0 18px; }
    .hero-top { display:flex; justify-content:space-between; gap:14px; align-items:flex-start; }
    .eyebrow { text-transform:uppercase; letter-spacing:.09em; font-size:10px; font-weight:800; color:var(--accent); }
    .hero h3 { margin:3px 0 5px; font-size:22px; letter-spacing:-.025em; }
    .reason { color:var(--muted); margin:0; }
    .badge { display:inline-flex; align-items:center; border:1px solid var(--line); border-radius:999px; padding:4px 9px; font-size:11px; color:var(--muted); white-space:nowrap; font-weight:700; }
    .badge.good { color:var(--accent); background:var(--accent-soft); border-color:transparent; }
    .badge.warn { color:var(--warn); background:var(--warn-soft); border-color:transparent; }
    .badge.bad { color:var(--bad); background:var(--bad-soft); border-color:transparent; }
    .hero-facts { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin-top:18px; }
    .fact { background:var(--bg); border-radius:11px; padding:12px; min-height:77px; }
    .fact-label { display:block; color:var(--muted); font-size:11px; margin-bottom:4px; }
    .fact-value { display:block; font-size:16px; font-weight:750; }
    .measurement { margin-top:14px; padding:12px 14px; border-left:3px solid var(--accent); background:var(--accent-soft); border-radius:8px; }
    .measurement strong { display:block; font-size:14px; }
    .measurement .small { margin-top:2px; }
    .panel { padding:18px; margin-top:16px; box-shadow:none; }
    .panel h2 { margin-bottom:12px; }
    .hardware { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
    .hardware-card { border:1px solid var(--line); background:var(--bg); border-radius:12px; padding:14px; }
    .hardware-card h3 { margin:0 0 8px; font-size:14px; }
    .kv { display:grid; grid-template-columns:1fr auto; gap:5px 14px; font-size:13px; }
    .kv span:nth-child(odd) { color:var(--muted); }
    .bandwidth { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
    .band-card { border:1px solid var(--line); border-radius:12px; padding:14px; }
    .band-card h3 { font-size:13px; margin:0 0 4px; }
    .band-number { font-size:20px; font-weight:750; }
    .notes-details { margin:12px 0 16px; border:1px solid var(--line); border-radius:12px; padding:12px 15px; background:var(--panel); }
    .notes-details summary,.more summary { cursor:pointer; font-weight:700; }
    .small { color:var(--muted); font-size:12px; }
    .rows { display:grid; gap:10px; }
    details.row { border:1px solid var(--line); border-radius:12px; padding:0 14px; background:var(--panel); }
    details.row summary { cursor:pointer; list-style:none; display:flex; gap:12px; justify-content:space-between; align-items:center; padding:14px 0; }
    details.row summary::-webkit-details-marker { display:none; }
    .row-name { font-weight:700; }
    .row-left { min-width:0; }
    .row-meta { display:flex; gap:6px; flex-wrap:wrap; margin-top:6px; }
    .row-body { border-top:1px solid var(--line); padding:14px 0; display:grid; gap:14px; }
    .row-body h4 { margin:0 0 5px; font-size:12px; }
    .row-body p { margin:0; }
    .chips { display:flex; gap:7px; flex-wrap:wrap; }
    .chip { background:var(--bg); border-radius:7px; padding:5px 8px; color:var(--muted); font-size:12px; }
    a { color:var(--accent); overflow-wrap:anywhere; }
    .empty { color:var(--muted); padding:12px 2px; }
    footer { text-align:center; color:var(--muted); padding-top:24px; font-size:12px; }
    @media(max-width:800px) { .control-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } .head { align-items:flex-start; } }
    @media(max-width:560px) { .wrap { width:min(100% - 24px,1120px); } .head { padding:17px 0; } .local { display:none; } .control-grid { grid-template-columns:1fr 1fr; gap:10px; } .hero-top { display:block; } .hero-top .badge { margin-top:10px; } .hero-facts,.hardware,.bandwidth { grid-template-columns:1fr; } .result-top { align-items:flex-start; flex-direction:column; } details.row summary { align-items:flex-start; } }
  </style>
</head>
<body>
  <header><div class="wrap head"><div class="brand"><div class="mark" aria-hidden="true">N</div><div><h1>Neural model advisor</h1><p class="sub">Compare local model options against your workload and machine.</p></div></div><div class="local">Local page · no model downloads or inference</div></div></header>
  <main class="wrap">
    <section class="panel" style="margin-top:0;margin-bottom:16px">
      <h2>Start with GPT-OSS 120B</h2>
      <p>No model choice is needed. Run <strong>install_neural.bat</strong> once, then <strong>start_neural.bat</strong> to open chat. Neural checks your PC before loading the model.</p>
      <p class="small"><a href="https://github.com/x3r081/neural#readme" target="_blank" rel="noopener noreferrer">Easy installation guide</a> · <a href="https://x3r081.github.io/neural-site/" target="_blank" rel="noopener noreferrer">How Neural works and benchmarks</a></p>
    </section>
    <section class="controls" aria-label="Recommendation settings">
      <div class="control-grid">
        <div><label for="context">Context length</label><select id="context"><option value="8192">8k tokens</option><option value="16384" selected>16k tokens</option><option value="32768">32k tokens</option><option value="65536">64k tokens</option><option value="131072">128k tokens</option></select></div>
        <div><label for="workload">Workload</label><select id="workload"><option value="coding" selected>Coding</option><option value="general">General</option></select></div>
        <div><label for="scenario">Machine scenario</label><select id="scenario"><option value="dedicated" selected>Dedicated machine</option><option value="current">Current machine</option></select></div>
      </div>
      <details class="advanced"><summary>Advanced bandwidth assumptions</summary>
        <div class="advanced-grid">
          <div><label for="cpu">CPU bandwidth · GB/s</label><input id="cpu" type="number" min="0.1" max="100000" step="0.1" placeholder="Auto from hardware"></div>
          <div><label for="gpu">GPU bandwidth · GB/s</label><input id="gpu" type="number" min="0.1" max="100000" step="0.1" placeholder="Auto from hardware"></div>
        </div>
        <p class="help">Optional assumptions, from 0.1 to 100,000 GB/s. Leave blank to use the advisor’s hardware estimate.</p>
      </details>
      <div class="control-actions"><button id="scan" type="button">Scan recommendations</button></div>
      <p id="error" role="alert" hidden></p>
    </section>
    <div class="result-top"><h2>Your default model</h2><div id="stamp" class="stamp"></div></div>
    <div id="recommended" aria-live="polite"><div class="empty">Loading recommendations…</div></div>
    <div id="notices"></div>
    <section class="panel"><h2>Machine details</h2><div id="hardware" class="hardware"></div></section>
    <section class="panel"><h2>Bandwidth assumptions</h2><div id="bandwidth" class="bandwidth"></div></section>
    <details class="panel more"><summary>Advanced: other model experiments</summary><p class="small">These are optional experiments. Neural does not switch to them automatically if GPT-OSS cannot run.</p><div id="others" class="rows"></div></details>
    <details class="panel more"><summary>Advanced: excluded models</summary><div id="excluded" class="rows"></div></details>
    <footer>Advice is based on the stated hardware and assumptions. Actual output may fall below theoretical values.</footer>
  </main>
<script>
  const INITIAL_SETTINGS = /*INITIAL_SETTINGS*/{};
  const el = (id) => document.getElementById(id);
  const make = (tag, cls, value) => { const n = document.createElement(tag); if (cls) n.className = cls; if (value !== undefined && value !== null) n.textContent = String(value); return n; };
  const shown = (value, suffix='') => value === null || value === undefined || value === '' ? 'Unavailable' : `${value}${suffix}`;
  const number1 = (value) => value === null || value === undefined || !Number.isFinite(Number(value)) ? 'Unavailable' : Number(value).toFixed(1);
  const tps = (value) => value === null || value === undefined ? 'Unavailable' : number1(value);
  const gib = (bytes) => bytes === null || bytes === undefined ? 'Unavailable' : `${(Number(bytes) / 1073741824).toFixed(1)} GiB`;
  const capacityGib = (bytes, reserve) => bytes === null || bytes === undefined ? 'Unavailable' : `${Math.max(0, Number(bytes) / 1073741824 - reserve).toFixed(1)} GiB`;
  const statusClass = (status) => status === 'recommended' ? 'good' : status === 'blocked' ? 'bad' : 'warn';
  function renderHardware(h) {
    const root = el('hardware'); root.replaceChildren();
    const cards = [
      ['Current machine · observed', [['GPU',h.gpu_name],['GPU memory total',gib(h.gpu_total_bytes)],['GPU memory free now',gib(h.gpu_free_bytes)],['System RAM total',gib(h.ram_total_bytes)],['System RAM available now',gib(h.ram_available_bytes)],['Physical cores',h.physical_cores],['Architecture',h.machine || (typeof h.cpu_signature === 'string' ? h.cpu_signature.split('|')[1] : null)]]],
      ['Dedicated machine · planning capacity', [['GPU capacity after 1 GiB reserve',capacityGib(h.gpu_total_bytes,1)],['RAM capacity after 6 GiB reserve',capacityGib(h.ram_total_bytes,6)],['Planning basis','Total capacity less fixed operating reserve']]]
    ];
    for (const [title, pairs] of cards) { const c=make('div','hardware-card'); c.append(make('h3','',title)); const grid=make('div','kv'); for (const [k,v] of pairs) { grid.append(make('span','',k),make('strong','',shown(v))); } c.append(grid); root.append(c); }
    const scenario = document.querySelector('#scenario option:checked').textContent;
    const info = make('p','small',`Selected scenario: ${scenario}. Current free memory is observed now; dedicated-machine capacity is estimated from total memory after fixed reserves.`); root.append(info);
  }
  function renderBandwidth(b) {
    const root=el('bandwidth'); root.replaceChildren();
    for (const [label,item] of [['CPU',b?.cpu],['GPU',b?.gpu]]) { const c=make('div','band-card'); c.append(make('h3','',`${label} bandwidth`)); const low=item?.gbps_low, high=item?.gbps_high; c.append(make('div','band-number',low == null || high == null ? 'Unavailable' : `${number1(low)}–${number1(high)} GB/s`)); c.append(make('div','small',item?.evidence || 'Evidence unavailable')); if(item?.note)c.append(make('p','small',item.note)); root.append(c); }
  }
  function renderSpeed(speed) {
    if (!speed || (speed.low_tps == null && speed.high_tps == null)) return 'No speed estimate available';
    return `${tps(speed.low_tps)}–${tps(speed.high_tps)} tok/s`;
  }
  function section(title, content, parent) { const box=make('div'); box.append(make('h4','',title)); box.append(content); parent.append(box); }
  function detailsFor(item) {
    const body=make('div','row-body');
    body.append(make('p','',item.reason || 'No explanation supplied.'));
    const memory=item.memory || {}; const mem=make('div','chips');
    for (const [k,v] of [['Model',memory.model_gib],['GPU core',memory.gpu_core_gib],['KV cache',memory.kv_gib],['GPU experts',memory.gpu_expert_gib],['RAM needed',memory.ram_needed_gib]]) if(v != null) mem.append(make('span','chip',`${k}: ${number1(v)} GiB`));
    for (const [k,v] of [['Fits dedicated',memory.fits_dedicated],['Fits this machine now',memory.fits_now]]) if(v != null) mem.append(make('span','chip',`${k}: ${v ? 'Yes' : 'No'}`));
    section('Memory estimate',mem,body);
    const speed=item.speed || {}; const speedInfo=make('div'); speedInfo.append(make('p','',renderSpeed(speed)));
    if(speed.label)speedInfo.append(make('p','small',speed.label));
    if(speed.evidence)speedInfo.append(make('p','small',`Evidence: ${speed.evidence}`));
    if(speed.confidence)speedInfo.append(make('p','small',`Confidence: ${speed.confidence}`));
    speedInfo.append(make('p','small','Actual speed can be below the shown range.'));
    if(item.support_tier === 'reference')speedInfo.append(make('p','small','CPU-expert bandwidth scenario; compute and transfer costs omitted. Staged PCIe streaming is not estimated.'));
    if(speed.measured){
      const measured=make('div','measurement'); measured.append(make('strong','',`Historical measurement: ${tps(speed.measured.low_tps)}–${tps(speed.measured.high_tps)} tok/s`));
      if(speed.measured.label)measured.append(make('div','small',speed.measured.label));
      if(speed.measured.note)measured.append(make('div','small',speed.measured.note));
      const url=typeof speed.measured.source==='string' && speed.measured.source.startsWith('https://') ? speed.measured.source : '';
      if(url){const a=make('a','small','View measurement source');a.href=url;a.target='_blank';a.rel='noopener noreferrer';measured.append(a);}
      speedInfo.append(measured);
    }
    if(Array.isArray(speed.assumptions) && speed.assumptions.length){const ul=make('ul');for(const a of speed.assumptions)ul.append(make('li','',a));speedInfo.append(ul);}
    section('Speed and evidence',speedInfo,body);
    for (const [title, values] of [['Notes',item.notes],['Sources',item.sources]]) {
      const box=make('div'); box.append(make('h4','',title));
      if (title==='Sources' && Array.isArray(values)) { const list=make('ul'); for(const source of values){const li=make('li');const url=typeof source?.url==='string' && source.url.startsWith('https://') ? source.url : ''; if(url){const a=make('a','',url);a.href=url;a.target='_blank';a.rel='noopener noreferrer';li.append(a);} if(source?.kind)li.append(document.createTextNode(` (${source.kind})`)); list.append(li);} box.append(list); }
      else if(Array.isArray(values)){const ul=make('ul');for(const note of values)ul.append(make('li','',note));box.append(ul);} else box.append(make('p','small','None supplied.'));
      body.append(box);
    }
    return body;
  }
  function row(item) {
    const d=make('details','row'); const s=make('summary'); const left=make('div','row-left'); const rank=item.rank == null ? '' : `${item.rank}. `; left.append(make('div','row-name',`${rank}${item.name || item.id || 'Unnamed option'}`));
    const meta=make('div','row-meta'); for(const val of [item.support_tier,item.validation])if(val)meta.append(make('span','badge',val)); left.append(meta);
    const speed=item.speed || {}; const speedSummary=speed.low_tps == null && speed.high_tps == null ? 'Speed unavailable' : `Theoretical · ${renderSpeed(speed)}`;
    s.append(left,make('span','badge',speedSummary),make('span',`badge ${statusClass(item.status)}`,item.status || 'unknown')); d.append(s,detailsFor(item)); return d;
  }
  function renderList(root, items, empty) { root.replaceChildren(); if(!Array.isArray(items)||!items.length){root.append(make('div','empty',empty));return;} for(const item of items)root.append(row(item)); }
  function renderRecommended(items, defaultModel) {
    const root=el('recommended');root.replaceChildren();
    if(defaultModel)items=[defaultModel];
    if(!Array.isArray(items)||!items.length){root.append(make('div','empty','No recommendation is available for these settings.'));return;}
    const item=items[0], hero=make('article','hero'), top=make('div','hero-top'), left=make('div');
    left.append(make('div','eyebrow',defaultModel ? 'Default model' : 'Recommendation'),make('h3','',item.name || item.id || 'Unnamed option'),make('p','reason',item.reason || 'No explanation supplied.'));
    if(defaultModel && item.status==='blocked')left.append(make('p','small','GPT-OSS stays the default. Resolve the setup or memory requirement above; Neural will not choose an experimental model for you.'));
    if(item.memory?.fits_now === false){
      const notes=Array.isArray(item.notes)?item.notes:[];
      const fitNote=notes.find(note=>/does not fit currently available memory|currently available memory/i.test(String(note))) || 'Does not fit in currently available memory.';
      left.append(make('div','notice',fitNote));
    }
    top.append(left,make('span',`badge ${statusClass(item.status)}`,item.status || 'unknown'));hero.append(top);
    const facts=make('div','hero-facts');
    for(const [label,value] of [['Theoretical output speed',renderSpeed(item.speed)],['Support',item.support_tier || 'Unknown'],['Validation',item.validation || 'Unknown']]){const f=make('div','fact');f.append(make('span','fact-label',label),make('span','fact-value',value));if(label==='Theoretical output speed'&&item.speed?.label)f.append(make('span','fact-label',item.speed.label));facts.append(f);} hero.append(facts);
    const speed=item.speed || {};
    if(speed.measured){const measured=make('div','measurement');measured.append(make('strong','',`Historical measurement: ${tps(speed.measured.low_tps)}–${tps(speed.measured.high_tps)} tok/s`));if(speed.measured.label)measured.append(make('div','small',speed.measured.label));if(speed.measured.note)measured.append(make('div','small',speed.measured.note));const url=typeof speed.measured.source==='string' && speed.measured.source.startsWith('https://') ? speed.measured.source : '';if(url){const a=make('a','small','View measurement source');a.href=url;a.target='_blank';a.rel='noopener noreferrer';measured.append(a);}hero.append(measured);}
    hero.append(make('p','small','Actual output speed may fall below both theoretical values.'));
    const more=make('details','more');more.append(make('summary','','Memory, assumptions, notes and sources'),detailsFor(item));hero.append(more);root.append(hero);
  }
  function render(report) {
    el('stamp').textContent=report.generated_at_utc ? `Generated ${report.generated_at_utc}` : '';
    const notices=el('notices');notices.replaceChildren();
    const notes=Array.isArray(report.notices) ? report.notices : [];
    const warnings=notes.filter(n=>/\b(warning|blocked|stop|error|insufficient)\b/i.test(String(n)));
    const explanations=notes.filter(n=>!warnings.includes(n));
    for(const n of warnings)notices.append(make('div','notice',n));
    if(explanations.length){const d=make('details','notes-details');d.append(make('summary','',`Read about assumptions and limits (${explanations.length})`));const ul=make('ul');for(const n of explanations)ul.append(make('li','',n));d.append(ul);notices.append(d);}
    renderRecommended(report.recommendations,report.default_model);renderHardware(report.hardware || {});renderBandwidth(report.bandwidth || {});
    const primary=report.default_model?.id || report.recommendations?.[0]?.id;
    renderList(el('others'),(report.recommendations || []).filter(item=>item.id!==primary),'No additional compatible experiments.');
    renderList(el('excluded'),(report.excluded || []).filter(item=>item.id!==primary),'No other options were excluded.');
  }
  async function scan(){const button=el('scan'),error=el('error');button.disabled=true;button.textContent='Scanning…';error.hidden=true;
    try{const q=new URLSearchParams({context:el('context').value,workload:el('workload').value,scenario:el('scenario').value});for(const [key,id] of [['cpu_bandwidth_gbps','cpu'],['gpu_bandwidth_gbps','gpu']]){if(el(id).value.trim())q.set(key,el(id).value.trim());}
      const response=await fetch(`/api/recommend?${q}`);const body=await response.json();if(!response.ok)throw new Error(body.error || 'Could not load recommendations.');render(body);
    }catch(e){error.textContent=e.message || 'Could not load recommendations.';error.hidden=false;}
    finally{button.disabled=false;button.textContent='Scan recommendations';}}
  if (INITIAL_SETTINGS.context !== undefined) {
    const context = String(INITIAL_SETTINGS.context);
    if (![...el('context').options].some(option => option.value === context)) {
      const option = document.createElement('option'); option.value = context;
      option.textContent = `${Number(context).toLocaleString()} tokens · custom`;
      el('context').append(option);
    }
    el('context').value = context;
  }
  if (INITIAL_SETTINGS.workload) el('workload').value = INITIAL_SETTINGS.workload;
  if (INITIAL_SETTINGS.scenario) el('scenario').value = INITIAL_SETTINGS.scenario;
  if (INITIAL_SETTINGS.cpu_bandwidth_gbps != null) el('cpu').value = INITIAL_SETTINGS.cpu_bandwidth_gbps;
  if (INITIAL_SETTINGS.gpu_bandwidth_gbps != null) el('gpu').value = INITIAL_SETTINGS.gpu_bandwidth_gbps;
  el('scan').addEventListener('click',scan);scan();
</script>
</body>
</html>'''


def _parse_request(query: str) -> dict[str, object]:
    params = parse_qs(query, keep_blank_values=True)

    def one(name: str, default: str) -> str:
        values = params.get(name)
        return values[-1].strip() if values else default

    try:
        context = int(one("context", "16384"))
    except ValueError as exc:
        raise ValueError("context must be a whole number of tokens") from exc
    if not 128 <= context <= 1_048_576:
        raise ValueError("context must be an integer from 128 to 1048576 tokens")
    workload = one("workload", "coding")
    if workload not in {"coding", "general"}:
        raise ValueError("workload must be coding or general")
    scenario = one("scenario", "dedicated")
    if scenario not in {"dedicated", "current"}:
        raise ValueError("scenario must be dedicated or current")

    def optional_bandwidth(name: str) -> float | None:
        raw = one(name, "")
        if not raw:
            return None
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be a positive number") from exc
        if not math.isfinite(value) or not 0.1 <= value <= 100_000:
            raise ValueError(f"{name} must be between 0.1 and 100000 GB/s")
        return value

    return {
        "context": context,
        "workload": workload,
        "scenario": scenario,
        "cpu_bandwidth_gbps": optional_bandwidth("cpu_bandwidth_gbps"),
        "gpu_bandwidth_gbps": optional_bandwidth("gpu_bandwidth_gbps"),
    }


def _normalized_settings(settings: dict[str, object] | None) -> dict[str, object]:
    if not settings:
        return {"context": 16384, "workload": "coding", "scenario": "dedicated", "cpu_bandwidth_gbps": None, "gpu_bandwidth_gbps": None}
    query = urlencode({key: value for key, value in settings.items() if value is not None})
    return _parse_request(query)


def _render_page(settings: dict[str, object] | None) -> bytes:
    normalized = _normalized_settings(settings)
    serialized = json.dumps(normalized, ensure_ascii=False, allow_nan=False)
    serialized = serialized.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return _PAGE.replace("/*INITIAL_SETTINGS*/{}", serialized).encode("utf-8")


def _is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    normalized = host.lower().rstrip(".")
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _make_server(report_factory, host: str, port: int, initial_settings: dict[str, object] | None = None) -> ThreadingHTTPServer:
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ValueError("The advisor server must bind to the IPv4 loopback address 127.0.0.1") from exc
    if not address.is_loopback or address.version != 4:
        raise ValueError("The advisor server must bind to the IPv4 loopback address 127.0.0.1")

    class AdvisorHandler(BaseHTTPRequestHandler):
        server_version = "NeuralAdvisor/1.0"

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            if not self._valid_local_headers():
                self._json(403, {"error": "Requests must come from this local advisor page."})
                return
            parsed = urlsplit(self.path)
            if parsed.path in {"/", "/page"}:
                payload = _render_page(initial_settings)
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'")
                self.end_headers()
                self.wfile.write(payload)
                return
            if parsed.path != "/api/recommend":
                self._json(404, {"error": "Not found"})
                return
            try:
                arguments = _parse_request(parsed.query)
                report = report_factory(**arguments)
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
                return
            except Exception:
                self._json(500, {"error": "The advisor could not build a recommendation."})
                return
            self._json(200, report)

        def _valid_local_headers(self) -> bool:
            try:
                host_header = self.headers.get("Host", "")
                host_parts = urlsplit(f"//{host_header}")
                if (
                    not _is_loopback_host(host_parts.hostname)
                    or host_parts.port != self.server.server_address[1]
                    or host_parts.username
                    or host_parts.password
                    or host_parts.path
                    or host_parts.query
                    or host_parts.fragment
                ):
                    return False
                origin = self.headers.get("Origin")
                if origin:
                    origin_parts = urlsplit(origin)
                    if origin_parts.scheme != "http" or not _is_loopback_host(origin_parts.hostname):
                        return False
                    if origin_parts.port != self.server.server_address[1] or origin_parts.path not in {"", "/"}:
                        return False
                    if origin_parts.username or origin_parts.password or origin_parts.query or origin_parts.fragment:
                        return False
                return True
            except ValueError:
                return False

        def _json(self, status: int, value: object) -> None:
            payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt: str, *args) -> None:
            # Keep the local server quiet by default; failures are returned to the page.
            return

    return ThreadingHTTPServer((host, port), AdvisorHandler)


def serve_advisor(
    report_factory,
    host: str = "127.0.0.1",
    port: int = 8002,
    open_browser: bool = False,
    initial_settings: dict[str, object] | None = None,
) -> None:
    """Serve the advisor locally until interrupted by the caller."""
    server = _make_server(report_factory, host, port, initial_settings)
    address, actual_port = server.server_address[:2]
    url = f"http://{address}:{actual_port}/"
    print(f"Neural model advisor is available at {url}", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
