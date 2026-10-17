/* Qwen3-TTS Studio — script.js (split from index.html) */
/* ═════════ script writer (free local LLM) ═════════ */
let SCRIPT_META=null, selScenario='story', selLength='standard', selWriter='darkidol', selProvider='local';
const TONES=['Gritty','Playful','Dark','Sultry','Whimsical','Deadpan'];
async function loadScriptMeta(){
  TONES.forEach(t=>{ const b=document.createElement('button'); b.className='text-[11px] px-2.5 py-1 rounded-full bg-white/5 border border-white/10 hover:border-violet-400/60'; b.textContent=t; b.title='Set tone to “'+t+'”'; b.onclick=()=>{ $('s_tone').value=t; }; $('toneChips').appendChild(b); });
  document.querySelectorAll('#lenSeg .lseg').forEach(b=>b.onclick=()=>{ selLength=b.dataset.len; document.querySelectorAll('#lenSeg .lseg').forEach(x=>x.classList.toggle('active',x===b)); });
  try{ const r=await fetch('/api/script/models'); SCRIPT_META=await r.json(); }
  catch(e){ $('scriptGrid').innerHTML='<div class="col-span-full text-[12px] text-red-300 text-center py-4">Writer backend unreachable.</div>'; return; }
  renderScriptGrid();
  if(!SCRIPT_META.ollama_up) $('scriptStatus').innerHTML='Ollama is down — local drafting unavailable, but <b>Groq</b>/<b>Gemini</b> above work without it (free key). Or start Ollama with <span class="font-mono">ollama serve</span>.';
  else $('scriptStatus').textContent='Writer ready ✓ free, local, unfiltered.';
  renderWriterSeg(); renderProviderSeg(); updateKeyBox();
}
function renderScriptGrid(){
  const box=$('scriptGrid'); if(!box||!SCRIPT_META||!SCRIPT_META.scenarios) return; box.innerHTML='';
  for(const sc of SCRIPT_META.scenarios){
    const b=document.createElement('button');
    b.className='card-hover text-left rounded-xl border px-3 py-2.5 '+(sc.id===selScenario?'border-violet-400 bg-violet-500/10':'border-white/10 bg-white/[.03]');
    b.innerHTML=`<div class="font-bold text-[12.5px]">${sc.label}</div><div class="text-[11px] text-slate-400">${sc.desc}</div>`;
    b.title=sc.label+': '+sc.desc;
    b.onclick=()=>{ selScenario=sc.id; renderScriptGrid(); };
    box.appendChild(b);
  }
}
function writerModels(){
  if(!SCRIPT_META) return [];
  if(selProvider==='local') return (SCRIPT_META.llm_models||[]).map(m=>({id:m.id,label:m.label,
    sub:`${m.size_gb} GB · ${m.note}`,
    warn:m.fits_ram===false?'⚠ tight on RAM — may be slow or stall':null, avail:!!m.available,
    why:m.available?((m.note||'')+' ('+m.tag+')'+(m.fits_ram===false?' WARNING: bigger than comfortably fits free RAM.':'')):'Not downloaded (`ollama pull '+m.tag+'`)'}));
  const pv=(SCRIPT_META.providers||[]).find(p=>p.id===selProvider);
  if(!pv) return [];
  return pv.models.map(m=>({id:m.id,label:m.label,sub:'free tier · '+pv.label+(pv.configured?'':' · needs key'),
    warn:null,avail:pv.configured,
    why:pv.configured?('Free '+pv.label+' quota · '+m.id):('Save your free '+pv.label+' key above first — '+pv.key_url)}));
}
function renderProviderSeg(){
  const box=$('providerSeg'); if(!box||!SCRIPT_META) return; box.innerHTML='';
  const opts=[{id:'local',label:'🏠 Local Ollama',sub:'free forever · private'}].concat((SCRIPT_META.providers||[]).map(p=>({id:p.id,label:p.label,sub:p.configured?'key saved ✓':'free key needed'})));
  for(const o of opts){
    const b=document.createElement('button');
    b.className='text-[12px] px-3 py-2 rounded-xl border '+(o.id===selProvider?'border-violet-400 bg-violet-500/15 font-bold':'border-white/10 bg-white/5');
    b.innerHTML=`${o.label}<br><span class="opacity-70 text-[11px] font-normal">${o.sub}</span>`;
    const pv=(SCRIPT_META.providers||[]).find(p=>p.id===o.id);
    b.title=o.id==='local'?'100% free local models on this machine. No key, no cloud, private.':('Free-tier cloud backend. '+(pv?pv.note:''));
    b.onclick=()=>{ selProvider=o.id;
      if(o.id==='local'){ if(!['best','balanced','darkidol','fast'].includes(selWriter)) selWriter='darkidol'; }
      else { const q=(SCRIPT_META.providers||[]).find(p=>p.id===o.id); selWriter=q?q.default:selWriter; }
      renderProviderSeg(); renderWriterSeg(); updateKeyBox();
    };
    box.appendChild(b);
  }
}
function renderWriterSeg(){
  const box=$('writerSeg'); if(!box) return; box.innerHTML='';
  const list=writerModels();
  for(const m of list){
    const b=document.createElement('button');
    b.disabled=!m.avail;
    b.className='wseg text-left text-[12px] px-3 py-2 rounded-xl border '+(m.id===selWriter?'border-violet-400 bg-violet-500/10':'border-white/10 bg-white/5')+(m.avail?'':' opacity-40 cursor-not-allowed');
    b.innerHTML=`<b>${m.label}</b><br><span class="opacity-70 text-[11px]">${m.sub}</span>`+(m.warn?`<br><span class="text-amber-300 text-[11px]">${m.warn}</span>`:'');
    b.title=m.why||m.label;
    b.onclick=()=>{ selWriter=m.id; renderWriterSeg(); };
    box.appendChild(b);
  }
  if(!list.some(m=>m.id===selWriter&&m.avail)){
    const av=list.find(m=>m.avail); if(av){ selWriter=av.id; renderWriterSeg(); }
  }
}
function updateKeyBox(){
  const box=$('keyBox'); if(!box) return;
  const pv=(SCRIPT_META&&SCRIPT_META.providers||[]).find(p=>p.id===selProvider);
  if(!pv){ box.classList.add('hidden'); return; }
  box.classList.remove('hidden');
  $('keyLink').href=pv.key_url;
  $('keyStatus').innerHTML=pv.configured?'<span class="text-emerald-300">✓ key saved — '+pv.label+' ready</span>':'<span class="text-amber-300">No key yet — grab a free one (60s), paste, save:</span>';
  $('keyForget').style.display=pv.configured?'':'none';
}
async function saveKey(){
  const k=$('keyInput').value.trim(); if(k.length<8) return toast('Paste the full key first',false);
  $('keyStatus').textContent='Verifying with provider…';
  try{
    const r=await fetch('/api/script/keys',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({provider:selProvider,key:k})});
    const j=await r.json(); if(!r.ok) throw new Error(j.detail||'rejected');
    $('keyInput').value=''; toast('Key verified & saved ✓'); await loadScriptMeta();
  }catch(e){ $('keyStatus').innerHTML='<span class="text-red-300">Rejected: '+friendlyErr(e)+'</span>'; }
}
async function forgetKey(){
  await fetch('/api/script/keys/'+selProvider,{method:'DELETE'}); toast('Key forgotten'); await loadScriptMeta();
}
async function generateScript(){
  const topic=$('s_topic').value.trim(); if(!topic) return toast('Give me a topic first ✍️',false);
  const btn=$('scriptBtn'); $('scriptStatus').textContent='';
  try{
    const j=await runJob('script','/api/jobs/script',{topic,scenario:selScenario,tone:$('s_tone').value,length:selLength,model:selWriter,provider:selProvider},btn);
    $('scriptOut').value=j.script||'';
    const n=j.words||(j.script||'').split(/\s+/).filter(Boolean).length;
    $('scriptWords').textContent=`· ${n} words in ${j.took_sec}s`;
    $('scriptStatus').textContent=`Done ✓ ${n} words · ${j.scenario}.`+(j.tts_parked?' '+j.tts_parked:'');
    toast(`Script ready — ${n} words ✍️`); checkHealth();
  }catch(e){ if(String(e.message)==='superseded'||String(e.message)==='__CANCELLED__') return; const m=friendlyErr(e); renderJBar('script',{phase:'error',detail:m,elapsed_sec:0}); $('scriptStatus').textContent='Error: '+m; toast('Error: '+m,false); checkHealth(); }
}
function fillPipeVoices(){
  const sel=$('pipeVoice'); if(!sel||!VOICES.length) return;
  sel.innerHTML=VOICES.map(v=>`<option value="${v.id}" ${v.id===selSpeaker?'selected':''}>${v.flag||''} ${v.label}</option>`).join('');
  sel.onchange=()=>{ if(sel.value) selectSpeaker(sel.value); };
}
function copyScript(){ const t=$('scriptOut').value; if(!t.trim()) return toast('Nothing to copy yet',false); navigator.clipboard.writeText(t).then(()=>toast('Copied 📋')).catch(()=>toast('Copy blocked by the browser',false)); }
function sendToStudio(){
  const t=$('scriptOut').value.trim(); if(!t) return toast('Write a script first ✍️',false);
  $('text').value=t; syncCounts();
  document.querySelector('.tab-btn[data-tab="studio"]').click();
  toast('Script loaded into Studio 🎙️ — hit Generate');
}
function sendToDuet(){
  const t=$('scriptOut').value.trim(); if(!t) return toast('Write a script first ✍️',false);
  $('duetText').value=t;
  document.querySelector('.tab-btn[data-tab="duet"]').click();
  toast('Script loaded into Duet 🎭 — A: / B: lines pick the voice');
}
function fillScriptResult(j){
  $('scriptOut').value=j.script||'';
  const n=j.words||(j.script||'').split(/\s+/).filter(Boolean).length;
  $('scriptWords').textContent=`· ${n} words in ${j.took_sec}s`;
  return n;
}
async function writeAndPerform(){
  const topic=$('s_topic').value.trim(); if(!topic) return toast('Give me a topic first ✍️',false);
  const btn=$('pipeBtn'); $('scriptStatus').textContent='';
  try{
    toast('Stage 1/2 — drafting the duet script ✍️');
    const s=await runJob('script','/api/jobs/script',{topic,scenario:selScenario,tone:$('s_tone').value,length:selLength,model:selWriter,provider:selProvider},btn);
    fillScriptResult(s);
    $('scriptStatus').textContent=`Script done ✓ ${(s.words||'')} words. Stage 2/2 — performing duet…`;
    document.querySelector('.tab-btn[data-tab="duet"]').click();
    await generateDuet();
  }catch(e){ if(String(e.message)==='superseded'||String(e.message)==='__CANCELLED__') return; const m=friendlyErr(e); renderJBar('script',{phase:'error',detail:m,elapsed_sec:0}); $('scriptStatus').textContent='Error: '+m; toast('Error: '+m,false); checkHealth(); }
}
async function writeAndSpeak(){
  const topic=$('s_topic').value.trim(); if(!topic) return toast('Give me a topic first ✍️',false);
  const btn=$('pipeBtn'); $('scriptStatus').textContent='';
  try{
    toast('Stage 1/2 — drafting script ✍️');
    const s=await runJob('script','/api/jobs/script',{topic,scenario:selScenario,tone:$('s_tone').value,length:selLength,model:selWriter,provider:selProvider},btn);
    $('scriptOut').value=s.script||'';
    const n=s.words||(s.script||'').split(/\s+/).filter(Boolean).length;
    $('scriptWords').textContent=`· ${n} words in ${s.took_sec}s`;
    $('scriptStatus').textContent=`Script done ✓ ${n} words. Stage 2/2 — speaking…`;
    const voice=($('pipeVoice')&&$('pipeVoice').value)||selSpeaker;
    toast(`Stage 2/2 — ${voice} is speaking 🔊`);
    const a=await runJob('script','/api/jobs/custom-voice',{text:s.script,language:$('language').value,speaker:voice,instruct:$('instruct').value,model:selModel},btn);
    playURL(a.file);
    $('outInfo').innerHTML=`<b class="text-slate-100">Pipeline 🎙️ ${voice}</b> · ${a.duration_sec}s · ${a.sample_rate/1000}kHz · ${a.language}${a.instruct_applied?' · <span class="text-violet-300">style applied</span>':''}`;
    $('scriptStatus').textContent=`Done ✓ script + ${a.duration_sec}s of audio with ${voice}.`;
    toast(`Pipeline complete — ${a.duration_sec}s of audio 🔊`); loadHistory(); checkHealth();
  }catch(e){ if(String(e.message)==='superseded'||String(e.message)==='__CANCELLED__') return; const m=friendlyErr(e); renderJBar('script',{phase:'error',detail:m,elapsed_sec:0}); $('scriptStatus').textContent='Error: '+m; toast('Error: '+m,false); checkHealth(); }
}

