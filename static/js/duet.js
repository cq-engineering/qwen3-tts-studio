/* Qwen3-TTS Studio — duet.js (split from index.html) */
/* ═════════ duet (two-speaker dialogue) ═════════ */
const DUET_SAMPLE=`A: Did you hear that? The lighthouse just turned itself on.\nB: Impossible. No keeper has climbed those stairs in forty years.\nA: Then explain the light moving across the water right now.\nB: ...Get in the boat. We are going out there.`;
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function fillDuetVoices(){
  for(const [id,def] of [['duetA','Aiden'],['duetB','Ryan']]){
    const sel=$(id); if(!sel||!VOICES.length) continue;
    sel.innerHTML=VOICES.map(v=>`<option value="${v.id}"${v.id===def?' selected':''}>${v.flag||''} ${v.label}</option>`).join('');
  }
}
async function generateDuet(){
  const text=$('duetText').value.trim(); if(!text) return toast('Write some dialogue first 🎭',false);
  const btn=$('duetBtn'); $('duetStatus').textContent=''; $('duetLines').innerHTML='';
  try{
    const j=await runJob('duet','/api/jobs/dialogue',{text,speaker_a:$('duetA').value,speaker_b:$('duetB').value,language:$('language').value,model:selModel,instruct:$('instruct').value},btn);
    playURL(j.file);
    const an=esc($('duetA').selectedOptions[0].text), bn=esc($('duetB').selectedOptions[0].text);
    $('outInfo').innerHTML=`<b class="text-slate-100">Duet 🎭 ${an} + ${bn}</b> · ${j.segments.length} lines · ${j.duration_sec}s`;
    $('duetStatus').textContent=`Done ✓ ${j.segments.length} lines, ${j.duration_sec}s.`;
    $('duetLines').innerHTML=j.segments.map(s=>`<button onclick="playURL('${s.url}')" title="Play this line" class="w-full text-left rounded-xl border border-white/10 bg-white/[.03] hover:bg-white/[.07] px-3 py-2 flex items-center gap-2 transition"><span class="text-[10px] font-bold px-1.5 py-0.5 rounded shrink-0 ${s.who==='A'?'bg-violet-500/25 text-violet-200':'bg-cyan-500/25 text-cyan-200'}">${s.who}</span><span class="min-w-0 flex-1 truncate text-[12px] text-slate-300">${esc(s.speaker)}: ${esc(s.text)}</span><span class="font-mono text-[10px] text-slate-500 shrink-0">${s.duration_sec}s</span><span class="text-[12px] shrink-0">▶</span></button>`).join('');
    toast(`Duet complete — ${j.segments.length} lines 🔊`); loadHistory(); checkHealth();
  }catch(e){ if(String(e.message)==='superseded'||String(e.message)==='__CANCELLED__') return; const m=friendlyErr(e); renderJBar('duet',{phase:'error',detail:m,elapsed_sec:0}); $('duetStatus').textContent='Error: '+m; toast('Error: '+m,false); checkHealth(); }
}

