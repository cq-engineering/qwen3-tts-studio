/* Qwen3-TTS Studio — core.js (split from index.html) */
/* ═════════ state ═════════ */
let VOICES=[], LANGS=[], MODELS={};
let selSpeaker='Aiden', selModel='custom-1.7B';
let autoIntroText='Hi, my name is Aiden.';
let lastURL=null, audioCtx=null, analyser=null, waveAnim=null;

const SAMPLES={
 ad:{t:"Tired of boring mornings? Meet Sunrise Brew — bold flavor, zero bitterness, ready in thirty seconds. Sunrise Brew. Wake up and win the day.",i:"Speak like an excited commercial narrator, punchy and confident with strong rhythmic drive."},
 story:{t:"The old lighthouse had been dark for forty years — until one stormy night, against all reason, its light flickered back on.",i:"Speak like a mysterious storyteller, slow and suspenseful with a low intimate tone."},
 podcast:{t:"Welcome back to Deep Dive, the show where we unpack the ideas shaping tomorrow. I'm your host, and today we're talking about the future of voice.",i:"Speak like a warm podcast host, friendly and conversational with natural pacing."},
 emotional:{t:"I waited all evening by the window, hoping you'd come. And when you finally did… I remembered why I never stopped believing.",i:"Speak with deep tenderness and quiet emotion, slow and heartfelt."}
};

const $=id=>document.getElementById(id);
/* Report browser-side crashes back to errors.log so UI bugs are diagnosable too. */
window.addEventListener('error',e=>{ try{fetch('/api/log/client',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({msg:String((e&&e.message)||(e&&e.error)||'error').slice(0,500),url:location.href.slice(0,200),line:String(e.lineno||'')})})}catch(_){} });
window.addEventListener('unhandledrejection',e=>{ try{fetch('/api/log/client',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({msg:('promise: '+String(e.reason)).slice(0,500),url:location.href.slice(0,200),line:''})})}catch(_){} });
/* Viewport-aware tooltips: CSS alone can't see screen edges, so one shared
   popup (appended to body = no clipping from scroll panels or glass filters)
   is measured and clamped inside the viewport on every show. */
const tipEl=document.createElement('div'); tipEl.id='tippy'; tipEl.setAttribute('role','tooltip'); document.body.appendChild(tipEl);
let tipFor=null;
function showTip(el){
  const txt=(el.dataset&&el.dataset.tip)||el.title||'';
  if(!txt) return;
  tipEl.textContent=txt; tipEl.classList.add('on');
  tipEl.style.visibility='hidden'; tipEl.style.left='0px'; tipEl.style.top='0px';
  const r=el.getBoundingClientRect(), gap=9, pad=10;
  const w=tipEl.offsetWidth, h=tipEl.offsetHeight;
  let left=Math.round(r.left+r.width/2-w/2);
  left=Math.max(pad,Math.min(left,window.innerWidth-w-pad));
  let top=Math.round(r.top-h-gap);
  if(top<pad) top=Math.round(r.bottom+gap);           // flip below when headroom is short
  top=Math.max(pad,Math.min(top,window.innerHeight-h-pad));
  tipEl.style.left=left+'px'; tipEl.style.top=top+'px';
  tipEl.style.visibility='visible'; tipFor=el;
}
function hideTip(){ tipEl.classList.remove('on'); tipFor=null; }
document.addEventListener('mouseover',e=>{ const h=e.target&&e.target.closest?e.target.closest('.hint'):null; if(h){ if(tipFor!==h) showTip(h); } else if(tipFor) hideTip(); });
document.addEventListener('mouseout',e=>{ if(tipFor&&!(e.relatedTarget&&e.relatedTarget.closest&&e.relatedTarget.closest('.hint'))) hideTip(); });
document.addEventListener('click',e=>{ const h=e.target&&e.target.closest?e.target.closest('.hint'):null; if(h){ if(tipFor===h) hideTip(); else showTip(h); } else hideTip(); });
document.addEventListener('focusin',e=>{ const h=e.target&&e.target.closest?e.target.closest('.hint'):null; if(h) showTip(h); });
document.addEventListener('focusout',()=>hideTip());
window.addEventListener('scroll',()=>hideTip(),true);
window.addEventListener('resize',()=>hideTip());
document.addEventListener('keydown',e=>{ if(e.key==='Escape') hideTip(); });
function toast(msg, ok=true){ const d=document.createElement('div'); d.className='toast glass rounded-xl px-4 py-2.5 text-[13px] max-w-sm '+(ok?'border-emerald-400/40':'border-red-400/50'); d.textContent=msg; $('toasts').appendChild(d); setTimeout(()=>{d.style.opacity='0';d.style.transition='opacity .4s';setTimeout(()=>d.remove(),400)},3600); }
/* Friendly error: "Failed to fetch" almost always means the backend is down. */
function friendlyErr(e){
  const m=String((e&&e.message)||e||'');
  if(/failed to fetch|networkerror|load failed|abort/i.test(m))
    return 'Cannot reach the server — it may have stopped. Start it with ./start.sh, refresh, and try again.';
  return m;
}

/* ═════════ init ═════════ */
async function init(){
  try{
    const r=await fetch('/api/voices'); const j=await r.json();
    VOICES=j.voices; LANGS=j.languages; MODELS=j.models;
  }catch(e){ VOICES=[]; }
  if(!VOICES.length){ // offline fallback mirror
    VOICES=[{id:'Aiden',label:'Aiden',gender:'Male',native_language:'English',region:'American',flag:'🇺🇸',description:'Sunny American male voice with a clear midrange.',tags:['american','male'],featured:true,best_for:'Flagship US voice'},{id:'Ryan',label:'Ryan',gender:'Male',native_language:'English',region:'American',flag:'🇺🇸',description:'Dynamic male voice with strong rhythmic drive.',tags:['american','male'],featured:true,best_for:'Promos'}];
    LANGS=['Auto','English','Chinese','Japanese','Korean','French','German','Spanish','Portuguese','Russian','Italian'];
  }
  for(const id of ['language','d_language','c_language']){ $(id).innerHTML=LANGS.map(l=>`<option ${l==='English'?'selected':''}>${l}</option>`).join(''); }
  renderVoices(); selectSpeaker('Aiden'); fillPipeVoices(); fillDuetVoices(); checkHealth(); loadHistory(); syncCounts(); drawIdleWave(); loadScriptMeta();
  document.addEventListener('keydown',e=>{ if((e.ctrlKey||e.metaKey)&&e.key==='Enter'){ e.preventDefault(); const tab=document.querySelector('.tab-btn.active').dataset.tab; if(tab==='studio')generate(); else if(tab==='design')generateDesign(); else if(tab==='script')generateScript(); else if(tab==='duet')generateDuet(); else generateClone(); }});
}
function syncCounts(){ $('charCount').textContent=$('text').value.length; }
document.addEventListener('input',e=>{ if(e.target.id==='text')syncCounts(); });

/* ═════════ voices sidebar ═════════ */
function renderVoices(){
  const sel=$('voiceSelect'); if(!sel||!VOICES.length) return;
  const us=VOICES.filter(v=>(v.tags||[]).includes('american'));
  const rest=VOICES.filter(v=>!(v.tags||[]).includes('american'));
  const opt=v=>`<option value="${v.id}"${v.id===selSpeaker?' selected':''}>${v.flag||''} ${v.label} — ${v.description||''}</option>`;
  sel.innerHTML=`<optgroup label="🇺🇸 American (best in English)">${us.map(opt).join('')}</optgroup><optgroup label="Other premade voices">${rest.map(opt).join('')}</optgroup>`;
  sel.title='Premade voice — grouped American-first. Every voice speaks all 10 languages.';
}
function selectSpeaker(id){
  selSpeaker=id; renderVoices();
  const pv=$('pipeVoice'); if(pv&&pv.options.length) pv.value=id;
  const vs=$('voiceSelect'); if(vs) vs.value=id;
  const v=VOICES.find(x=>x.id===id)||{label:id,description:'',flag:''};
  $('selDesc').textContent=(v.flag?v.flag+' ':'')+(v.description||'')+` · ${v.gender||'?'} · ${v.native_language||'multilingual'}`;
  const badge=$('usBadge'); if(badge) badge.style.display=(v.tags||[]).includes('american')?'':'none';
  const nm=v.label||id, t=$('text').value.trim();
  if(!t||t===autoIntroText){ autoIntroText='Hi, my name is '+nm+'.'; $('text').value=autoIntroText; syncCounts(); }
}
function clearText(){ $('text').value=''; syncCounts(); $('text').focus(); }
document.querySelectorAll('.sampleBtn').forEach(b=>b.onclick=()=>{ const s=SAMPLES[b.dataset.sample]; $('text').value=s.t; $('instruct').value=s.i; syncCounts(); toast('Sample loaded ✍️'); });
function applyStyle(s){ $('instruct').value=s||''; if(selModel==='custom-0.6B'&&s) toast('Style needs the 1.7B model — switched back to High Quality ⭐',false), setModel('custom-1.7B'); }
document.querySelectorAll('#styleChips button').forEach(b=>{ b.onclick=()=>applyStyle(b.dataset.style); if(!b.title) b.title='Fill the Style box with this direction'+(b.dataset.style?': “'+b.dataset.style+'”':' (clear)'); });
const EXTRA_STYLES=[
 ["😏 Sarcastic","Speak with dry sarcasm and an eye-rolling tone, dripping with irony.","Fills the Style box with a sarcastic direction. Best on witty or deadpan lines."],
 ["🏈 Sportscaster","Speak like an excited sports commentator in the final seconds, fast, loud and explosive.","High-energy play-by-play delivery for hype content."],
 ["🌙 Bedtime","Speak slowly and softly like a bedtime story, warm, drowsy, with a gentle rocking rhythm.","Slow, soft narration for sleep stories and calm content."],
 ["🦹 Villain","Speak like a sinister villain, low and menacing, savoring every single word.","Menacing character voice for stories and games."],
 ["🦸 Heroic","Speak like a heroic leader rallying the troops, bold, loud and inspiring.","Bold trailer-style heroic delivery."],
 ["🕵️ Noir","Speak like a weary noir detective, gravelly and slow, world-weary with dry wit.","Gravelly detective character for mysteries."],
 ["🏴‍☠️ Pirate","Speak like a boisterous pirate captain, hearty and rough, with rolling Rs.","Fun character voice for kids content and games."],
 ["📣 Coach","Speak like a motivational coach, high energy and punchy, with emphatic pauses.","Punchy motivational delivery for self-improvement content."],
 ["💘 Romantic","Speak tenderly and romantically, soft and breathy, slow and intimate.","Intimate delivery for love stories and dedications."],
 ["👻 Spooky","Speak in a creepy haunted whisper, slow and echoing, genuinely unsettling.","Horror narration and Halloween content."],
 ["🤖 Robot","Speak like a friendly retro robot, flat and mechanical with precise clipped pacing.","Character effect for sci-fi and comedy."],
 ["⏩ Auctioneer","Speak extremely fast like an auctioneer, breathless and rhythmic, barely pausing.","Maximum-pace novelty delivery."]
];
EXTRA_STYLES.forEach(s=>{ const b=document.createElement('button'); b.className='text-[11px] px-2.5 py-1 rounded-full bg-white/5 border border-white/10 hover:border-violet-400/60'; b.dataset.style=s[1]; b.textContent=s[0]; b.title=s[2]; b.onclick=()=>applyStyle(s[1]); $('styleChips').appendChild(b); });
document.querySelectorAll('.seg').forEach(b=>b.onclick=()=>setModel(b.dataset.model));
function setModel(m){ selModel=m; document.querySelectorAll('.seg').forEach(x=>x.classList.toggle('active',x.dataset.model===m)); $('modelBadge').textContent=m==='custom-1.7B'?'1.7B high-quality':'0.6B fast'; const has=m==='custom-1.7B'; $('instruct').disabled=!has; $('instruct').style.opacity=has?1:.4; $('instructNote').textContent=has?'● active on 1.7B':'○ needs 1.7B — ignored on 0.6B'; $('instructNote').className='ml-auto text-[11px] '+(has?'text-emerald-300':'text-amber-300'); }
function syncAdv(){ const t=$('temperature').value; $('tempVal').textContent=(t==='1'?'default':t); const p=$('top_p').value; $('toppVal').textContent=(p==='1'?'default':p); }
function resetAdv(id){ $(id).value=1; syncAdv(); }
function randomSeed(){ $('seed').value=Math.floor(Math.random()*999999); }

/* ═════════ tabs ═════════ */
document.querySelectorAll('.tab-btn').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('.tab-btn').forEach(x=>x.classList.remove('active')); b.classList.add('active');
  for(const t of ['studio','design','clone','script','duet']) $('panel-'+t).classList.toggle('hidden',b.dataset.tab!==t);
});
document.querySelectorAll('.dpreset').forEach(b=>{ b.onclick=()=>{ $('d_instruct').value=b.dataset.p; }; if(!b.title) b.title='Use this voice recipe: “'+b.dataset.p+'”'; });
const EXTRA_DESIGN=[
 ["🕵️ Detective","Middle-aged male detective, gravelly low voice, slow deliberate pace, world-weary and dry."],
 ["🧙 Witch","Ancient female voice, raspy and thin with a cackling undertone, slow with eerie pauses."],
 ["🧘 Therapist","Calm middle-aged female voice, soft and even, slow reassuring cadence, warm and safe."],
 ["👹 Overlord","Deep booming male voice, theatrical and menacing, dramatic pauses, powerful projection."],
 ["🤖 Robot","Genderless synthetic voice, flat and mechanical, precise clipped syllables, faint electronic edge."]
];
EXTRA_DESIGN.forEach(s=>{ const b=document.createElement('button'); b.className='dpreset text-[11px] px-2 py-1 rounded-full bg-white/5 border border-white/10'; b.dataset.p=s[1]; b.textContent=s[0]; b.title='Use this voice recipe: “'+s[1]+'”'; b.onclick=()=>{ $('d_instruct').value=s[1]; }; const box=$('designPresets'); if(box) box.appendChild(b); });

/* ═════════ health ═════════ */
async function checkHealth(){
  try{ const r=await fetch('/api/health'); const j=await r.json();
    const b=j.backend||{};
    $('backendBadge').textContent=`backend: ${b.torch&&String(b.torch).startsWith('missing')?'⏳ installing':'✓ torch '+String(b.torch||'').split('+')[0]} · ${b.cuda?'GPU':'CPU'} · qwen-tts ${String(b.qwen_tts||'').slice(0,18)}`+((j.boot&&j.boot.actual_sec!=null)?` · boot ${j.boot.actual_sec}s (avg ${j.boot.average_sec}s)`:'');
    $('backendBadge').title=(j.boot&&j.boot.actual_sec!=null)?`This session booted in ${j.boot.actual_sec}s · ${j.boot.average_sec}s average over ${j.boot.samples} recorded boots · ~${j.boot.estimated_sec}s estimated next boot. Click ↻ Status to refresh.`:'Backend status. Click ↻ Status to refresh.';
    $('footInfo').textContent=`device ${b.device} · loaded [${(j.loaded_models||[]).join(', ')||'none yet'}]`;
    $('offlineBanner').classList.add('hidden');
    return true;
  }catch(e){ $('backendBadge').textContent='backend: unreachable — run ./start.sh'; $('offlineBanner').classList.remove('hidden'); return false; }
}
setInterval(checkHealth, 30000);
async function restartServer(){
  if(!confirm('Restart the server?\n\nActive generations will be cancelled, loaded models unloaded, and a fresh server will boot on the same port (:8000). The page reconnects automatically.')) return;
  for(const k of Object.keys(activeJobs)) delete activeJobs[k];  // stop local poll loops silently
  toast('Restarting server… reconnecting automatically ⏻');
  try{ await fetch('/api/server/restart',{method:'POST'}); }catch(e){ /* response may die with the process — expected */ }
  for(let i=0;i<30;i++){
    await new Promise(res=>setTimeout(res,3000));
    if(await checkHealth()){ toast('Server back online ✓'); loadHistory(); return; }
  }
  toast('Server is taking unusually long — check server.log or run ./start.sh',false);
}
async function exitServer(){
  if(!confirm('Stop everything?\n\nActive generations are cancelled, TTS + writer models are unloaded (gigabytes of RAM freed), and the server process terminates.\n\nRelaunch anytime from the desktop icon.')) return;
  for(const k of Object.keys(activeJobs)) delete activeJobs[k];  // stop local poll loops silently
  let info='';
  try{
    const r=await fetch('/api/server/shutdown',{method:'POST'});
    const j=await r.json().catch(()=>({}));
    if(j.tts_unloaded&&j.tts_unloaded.length) info=`Freed ${(j.tts_freed_mb/1024).toFixed(1)} GB from ${j.tts_unloaded.join(', ')}.`;
  }catch(e){ /* process may die before responding — still show the overlay */ }
  const ov=$('stoppedOverlay');
  if(info) $('stoppedDetail').textContent=info;
  ov.classList.remove('hidden');
  checkHealth();
}

