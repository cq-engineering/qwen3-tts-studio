/* Qwen3-TTS Studio — jobs.js (split from index.html) */
/* ═════════ live job status bar (real backend phases) ═════════ */
const JSTEPS=[['queued','Queued'],['loading_model','Loading'],['synthesizing','Synthesizing'],['saving','Saving']];
const JWIDTH={queued:4,loading_model:14,synthesizing:45,saving:92,done:100,error:100};
const activeJobs={};
function fmtT(s){ s=Math.max(0,Math.floor(s||0)); return String(Math.floor(s/60)).padStart(2,'0')+':'+String(s%60).padStart(2,'0'); }
function renderJBar(panel, job){
  const el=$('jbar-'+panel); if(!el||!job) return;
  const err=job.phase==='error', done=job.phase==='done', cx=job.phase==='cancelled', active=!done&&!err&&job.phase!=='cancelled';
  const STEPS=(job.steps&&job.steps.length?job.steps:JSTEPS);
  const cur=STEPS.findIndex(s=>s[0]===job.phase);
  let steps=STEPS.map((s,i)=>{
    const st=done?'done':(err?'todo':(i<cur?'done':(i===cur?'cur':'todo')));
    const icon=st==='done'?'✓':(st==='cur'?'●':'○');
    return `<div class="jstep ${st}"><span>${icon}</span>${s[1]}</div>${i<3?'<div class="jline"></div>':''}`;
  }).join('');
  if(done) steps+='<div class="jline"></div><div class="jstep done"><span>✓</span>Done</div>';
  if(err) steps+='<div class="jline err"></div><div class="jstep err"><span>✕</span>Failed</div>';
  if(cx) steps+='<div class="jline cx"></div><div class="jstep cx"><span>■</span>Cancelled</div>';
  el.innerHTML=`<div class="jbar glass rounded-xl px-4 py-3 ${err?'is-err':''} ${done?'is-done':''} ${cx?'is-cancel':''}">
    <div class="flex items-center gap-2"><span class="jdot"></span>
      <b class="text-[12px]">${err?'Failed':(done?'Complete ✓':(cx?'Cancelled':(job.label||'Working…')))}</b>
      ${active?`<button class="jcancel" onclick="cancelJob('${panel}',this)" title="Stop this generation at the next safe point, discard partial work, and unload the model from memory.">⏹ Cancel</button>`:''}
      <span class="ml-auto font-mono text-[11px] text-slate-400">⏱ ${fmtT(job.elapsed_sec)}</span>
      <span class="jpct">${done?100:Math.round(job.percent??JWIDTH[job.phase]??4)}%</span></div>
    <div class="jsteps">${steps}</div>
    <div class="pbar"><div class="pfill" style="width:${done?100:(job.percent??JWIDTH[job.phase]??4)}%"></div></div>
    <div class="jdetail">${job.queue_position?`Queued #${job.queue_position+1} · `:''}${job.detail||''}${job.overtime?` <span class="text-amber-300">• past estimate — still working, normal on CPU.</span>`:''}</div>
    ${(err||cx)?'':(done
      ? `<div class="jcal">Finished in ${fmtT(job.elapsed_sec)} · predicted ~${job.predicted_sec??'?'}s · every run sharpens future estimates</div>`
      : `<div class="jcal">~${job.predicted_sec??'?'}s predicted · live estimate, calibrated from ${job.calib_runs??0} run${(job.calib_runs??0)===1?'':'s'} on this machine <span class="hint sm" tabindex="0" data-tip="True per-token progress isn't exposed by the Qwen3-TTS engine (one monolithic generation call, no streaming hooks), so the % is a self-calibrating live estimate: wall-seconds per input character, learned from real completed jobs on this machine and saved to calib.json. It updates every poll, converges with use, and always lands on exactly 100 at completion. The phases above (queued → loading → synthesizing → saving) are 100% real backend state.">?</span></div>`)}
  </div>`;
}
function clearJBar(panel){ const el=$('jbar-'+panel); if(el) el.innerHTML=''; }
/* Submit a job, poll live status, drive the bar. Resolves with result, throws on error. */
async function runJob(panel, url, body, btn, onPoll){
  const isFD=(typeof FormData!=='undefined')&&body instanceof FormData;
  let r;
  try{ r=await fetch(url,{method:'POST',headers:isFD?{}:{'Content-Type':'application/json'},body:isFD?body:JSON.stringify(body)}); }
  catch(e){ throw new Error(friendlyErr(e)); }
  const q=await r.json().catch(()=>({}));
  if(!r.ok) throw new Error(q.detail||('HTTP '+r.status));
  const id=q.job_id, myKey={}; activeJobs[panel]={key:myKey,id};
  clearJBar(panel); setBusy(btn,true,'Working');
  let fails=0;
  try{
    for(;;){
      await new Promise(res=>setTimeout(res,1500));
      if(activeJobs[panel]?.key!==myKey) throw new Error('superseded');
      let jr=null;
      try{ const pr=await fetch('/api/jobs/'+id); jr=await pr.json(); if(!pr.ok) throw new Error(jr.detail||'job lookup failed'); fails=0; }
      catch(e){ if(++fails>=4){ renderJBar(panel,{phase:'error',detail:friendlyErr(e),elapsed_sec:0}); throw new Error(friendlyErr(e)); } continue; }
      renderJBar(panel,jr);
      if(onPoll) try{onPoll(jr)}catch(_){}
      if(panel==='script'&&jr.partial&&document.activeElement!==$('scriptOut')){
        $('scriptOut').value=jr.partial;
        $('scriptWords').textContent='· '+jr.partial.split(/\s+/).filter(Boolean).length+' words (live…)';
      }
      if(jr.status==='done') return jr.result;
      if(jr.status==='cancelled') throw new Error('__CANCELLED__');
      if(jr.status==='error') throw new Error(jr.error||jr.detail||'job failed');
    }
  }finally{ if(activeJobs[panel]?.key===myKey) delete activeJobs[panel]; setBusy(btn,false); }
}
async function cancelJob(panel, btn){
  const a=activeJobs[panel]; if(!a) return;
  if(btn){ btn.disabled=true; btn.textContent='Cancelling…'; }
  try{ await fetch('/api/jobs/'+a.id+'/cancel?unload=true',{method:'POST'}); }
  catch(e){ toast('Cancel signal failed: '+friendlyErr(e),false); if(btn){ btn.disabled=false; btn.textContent='⏹ Cancel'; } }
  /* the poll loop picks up the cancelled state (with freed-memory note) within ~2s */
}

/* ═════════ generation ═════════ */
function advPayload(){
  const p={ max_new_tokens:parseInt($('max_tokens').value)||2048 };
  if($('temperature').value!=='1') p.temperature=parseFloat($('temperature').value);
  if($('top_p').value!=='1') p.top_p=parseFloat($('top_p').value);
  if($('top_k').value) p.top_k=parseInt($('top_k').value);
  if($('seed').value!=='') p.seed=parseInt($('seed').value);
  return p;
}
function setBusy(btn,on,label){ if(on){btn.dataset.orig=btn.innerHTML;btn.disabled=true;btn.classList.add('generating');btn.innerHTML='<span class="spinner"></span> '+label+'… (first run downloads model)';} else {btn.disabled=false;btn.classList.remove('generating');btn.innerHTML=btn.dataset.orig;} }

let streamSession=0;
async function generate(){
  const text=$('text').value.trim(); if(!text) return toast('Type something first ✍️',false);
  streamSession++; $('player').onended=null;
  const btn=$('genBtn'); $('genStatus').textContent='';
  const payload={text,language:$('language').value,speaker:selSpeaker,instruct:$('instruct').value,model:selModel,...advPayload()};
  try{
    if($('streamMode')&&$('streamMode').checked){ await generateStreamed(payload,btn); return; }
    const j=await runJob('studio','/api/jobs/custom-voice',payload,btn);
    playURL(j.file); $('outInfo').innerHTML=`<b class="text-slate-100">${selSpeaker} 🇺🇸</b> · ${j.duration_sec}s · ${j.sample_rate/1000}kHz · ${j.language} · ${j.model}${j.instruct_applied?' · <span class="text-violet-300">style applied</span>':''}`;
    $('genStatus').textContent=`Done in studio ✓ ${j.duration_sec}s of audio.`; toast(`Done — ${j.duration_sec}s from ${selSpeaker} 🔊`); loadHistory(); checkHealth();
  }catch(e){ if(String(e.message)==='superseded'||String(e.message)==='__CANCELLED__') return; const m=friendlyErr(e); renderJBar('studio',{phase:'error',detail:m,elapsed_sec:0}); $('genStatus').textContent='Error: '+m; toast('Error: '+m,false); checkHealth(); }
}
async function generateStreamed(payload,btn){
  const mySession=streamSession, p=$('player');
  const st={urls:[],i:0,waiting:false,active:true};
  p.onended=()=>{
    if(!st.active||mySession!==streamSession) return;
    if(st.i+1<st.urls.length){ st.i++; playURL(st.urls[st.i]); }
    else st.waiting=true;
  };
  const onPoll=(jr)=>{
    if(mySession!==streamSession) return;
    const names=jr.parts_done||[];
    if(names.length>st.urls.length){
      st.urls=names.map(f=>'/outputs/'+f);
      const n=jr.total_parts||st.urls.length;
      $('genStatus').textContent=`Part ${st.urls.length}/${n} ready — playing through…`;
      $('outInfo').innerHTML=`<b class="text-slate-100">${selSpeaker} 🇺🇸 stream</b> · part ${st.urls.length}/${n} playing, rest rendering…`;
      if(st.urls.length===1){ playURL(st.urls[0]); toast('🔊 Part 1 playing — rest still rendering'); }
      else if(st.waiting){ st.waiting=false; st.i++; playURL(st.urls[st.i]); }
    }
  };
  try{
    const j=await runJob('studio','/api/jobs/chunked',payload,btn,onPoll);
    if(mySession!==streamSession) return;
    st.active=false; p.onended=null;
    if(!j.parts||!j.parts.length){ // single-sentence fallback: normal result
      playURL(j.file); $('outInfo').innerHTML=`<b class="text-slate-100">${selSpeaker} 🇺🇸</b> · ${j.duration_sec}s · ${j.sample_rate/1000}kHz · ${j.language} · ${j.model}`;
      $('genStatus').textContent=`Done in studio ✓ ${j.duration_sec}s of audio.`; toast(`Done — ${j.duration_sec}s 🔊`);
    }else{
      playURL(j.file);
      $('outInfo').innerHTML=`<b class="text-slate-100">${selSpeaker} 🇺🇸 stream</b> · ${j.parts.length} parts · ${j.duration_sec}s total · stitched file ready`;
      $('genStatus').textContent=`Done ✓ ${j.parts.length} parts streamed, ${j.duration_sec}s total.`;
      toast(`Stream complete — ${j.parts.length} parts, ${j.duration_sec}s 🔊`);
    }
    loadHistory(); checkHealth();
  }catch(e){ if(String(e.message)==='superseded'||String(e.message)==='__CANCELLED__') return; const m=friendlyErr(e); renderJBar('studio',{phase:'error',detail:m,elapsed_sec:0}); $('genStatus').textContent='Error: '+m; toast('Error: '+m,false); checkHealth(); }
}
async function previewVoice(){ await generate(); }
async function generateDesign(){
  const btn=$('designBtn'); $('designStatus').textContent='';
  try{
    const j=await runJob('design','/api/jobs/voice-design',{text:$('d_text').value,language:$('d_language').value,instruct:$('d_instruct').value,...advPayload()},btn);
    playURL(j.file); $('outInfo').innerHTML=`<b class="text-slate-100">Designed voice ✨</b> · ${j.duration_sec}s · ${j.sample_rate/1000}kHz`;
    $('designStatus').textContent='Done ✓'; toast('Voice designed ✨'); loadHistory(); checkHealth();
  }catch(e){ if(String(e.message)==='superseded'||String(e.message)==='__CANCELLED__') return; const m=friendlyErr(e); renderJBar('design',{phase:'error',detail:m,elapsed_sec:0}); $('designStatus').textContent='Error: '+m; toast('Error: '+m,false); checkHealth(); }
}
let mediaRec=null, chunks=[];
async function toggleRec(){
  if(mediaRec&&mediaRec.state==='recording'){ mediaRec.stop(); $('recBtn').textContent='🎤 Record mic'; return; }
  try{ const s=await navigator.mediaDevices.getUserMedia({audio:true}); mediaRec=new MediaRecorder(s); chunks=[];
    mediaRec.ondataavailable=e=>chunks.push(e.data);
    mediaRec.onstop=()=>{ const blob=new Blob(chunks,{type:'audio/webm'}); const url=URL.createObjectURL(blob); const rp=$('refPreview'); rp.src=url; rp.classList.remove('hidden'); webmToWavFile(blob); };
    mediaRec.start(); $('recBtn').textContent='⏹ Stop recording';
  }catch(e){ toast('Mic blocked — allow microphone or upload a file',false); }
}
let clonedFile=null;
$('c_file').addEventListener('change',e=>{ clonedFile=e.target.files[0]||null; if(clonedFile){ const rp=$('refPreview'); rp.src=URL.createObjectURL(clonedFile); rp.classList.remove('hidden'); }});
async function webmToWavFile(blob){ try{ const ab=await blob.arrayBuffer(); const ac=new (window.AudioContext||window.webkitAudioContext)(); const buf=await ac.decodeAudioData(ab); const wav=audioBufferToWav(buf); clonedFile=new File([wav],'mic-record.wav',{type:'audio/wav'}); toast('Mic take captured 🎤'); }catch(e){} }
function audioBufferToWav(buf){ const n=buf.length, sr=buf.sampleRate, ch=buf.getChannelData(0); const b=new ArrayBuffer(44+n*2), v=new DataView(b); const ws=(o,s)=>{for(let i=0;i<s.length;i++)v.setUint8(o+i,s.charCodeAt(i))}; ws(0,'RIFF');v.setUint32(4,36+n*2,true);ws(8,'WAVE');ws(12,'fmt ');v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);v.setUint32(24,sr,true);v.setUint32(28,sr*2,true);v.setUint16(32,2,true);v.setUint16(34,16,true);ws(36,'data');v.setUint32(40,n*2,true); for(let i=0;i<n;i++){const s=Math.max(-1,Math.min(1,ch[i]));v.setInt16(44+i*2,s<0?s*0x8000:s*0x7FFF,true)} return new Blob([b],{type:'audio/wav'}); }
async function generateClone(){
  const f=clonedFile||$('c_file').files[0]; if(!f) return toast('Add reference audio first (upload or record) 🎤',false);
  if(!$('c_text').value.trim()) return toast('Type what the clone should say',false);
  const btn=$('cloneBtn'); $('cloneStatus').textContent='';
  try{
    const fd=new FormData(); fd.append('ref_audio',f); fd.append('target_text',$('c_text').value); fd.append('language',$('c_language').value); fd.append('ref_text',$('c_reftext').value); fd.append('model',$('c_model').value); fd.append('x_vector_only',$('c_xvec').checked); fd.append('max_new_tokens','2048');
    const j=await runJob('clone','/api/jobs/voice-clone',fd,btn);
    playURL(j.file); $('outInfo').innerHTML=`<b class="text-slate-100">Cloned voice 🧬</b> · ${j.duration_sec}s · ${j.sample_rate/1000}kHz`;
    $('cloneStatus').textContent='Done ✓'; toast('Clone complete 🧬'); loadHistory(); checkHealth();
  }catch(e){ if(String(e.message)==='superseded'||String(e.message)==='__CANCELLED__') return; const m=friendlyErr(e); renderJBar('clone',{phase:'error',detail:m,elapsed_sec:0}); $('cloneStatus').textContent='Error: '+m; toast('Error: '+m,false); checkHealth(); }
}

