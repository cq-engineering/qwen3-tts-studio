/* Qwen3-TTS Studio — media.js (split from index.html) */
/* ═════════ player + waveform ═════════ */
function playURL(url){
  lastURL=url; const p=$('player'); p.src=url; p.play().catch(()=>{});
  const a=$('dlBtn'); a.href=url; a.download=url.split('/').pop(); a.classList.remove('pointer-events-none','opacity-40');
  startWave(p);
}
function replay(){ if(lastURL){ $('player').currentTime=0; $('player').play().catch(()=>{});} }
function drawIdleWave(){
  const c=$('wave'), ctx=c.getContext('2d'); const dpr=window.devicePixelRatio||1;
  c.width=c.clientWidth*dpr; c.height=84*dpr; ctx.scale(dpr,dpr);
  ctx.clearRect(0,0,c.clientWidth,84); ctx.fillStyle='rgba(139,92,246,.55)';
  const mid=42; for(let x=0;x<c.clientWidth;x+=4){ const h=3+Math.abs(Math.sin(x*.08))*10+Math.random()*4; ctx.fillRect(x,mid-h/2,2,h); }
}
function startWave(media){
  try{
    audioCtx=audioCtx||new (window.AudioContext||window.webkitAudioContext)();
    if(!analyser){ const src=audioCtx.createMediaElementSource(media); analyser=audioCtx.createAnalyser(); analyser.fftSize=256; src.connect(analyser); analyser.connect(audioCtx.destination); }
    audioCtx.resume();
    const c=$('wave'), ctx=c.getContext('2d');
    cancelAnimationFrame(waveAnim);
    const draw=()=>{ waveAnim=requestAnimationFrame(draw);
      const dpr=window.devicePixelRatio||1; if(c.width!==c.clientWidth*dpr){c.width=c.clientWidth*dpr;c.height=84*dpr;}
      ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,c.clientWidth,84);
      const data=new Uint8Array(analyser.frequencyBinCount); analyser.getByteFrequencyData(data);
      const n=64, bw=c.clientWidth/n;
      for(let i=0;i<n;i++){ const v=data[Math.floor(i/data.length*n*1.4)]/255; const h=4+v*68;
        const g=ctx.createLinearGradient(0,42-h/2,0,42+h/2); g.addColorStop(0,'#8b5cf6'); g.addColorStop(1,'#06b6d4');
        ctx.fillStyle=g; const x=i*bw+2; ctx.beginPath(); ctx.roundRect(x,42-h/2,bw-4,h,4); ctx.fill(); }
    }; draw();
  }catch(e){ drawIdleWave(); }
}

/* ═════════ history ═════════ */
async function loadHistory(){
  try{ const r=await fetch('/api/history?limit=30'); const j=await r.json();
    $('histCount').textContent='· '+j.items.length;
    const box=$('history'); box.innerHTML=j.items.length?'':'<div class="text-[12px] text-slate-500 text-center py-4">No generations yet.<br>Your WAVs will live here.</div>';
    for(const it of j.items){
      const d=document.createElement('div'); d.className='rounded-xl border border-white/10 bg-white/[.03] px-3 py-2 flex items-center gap-2';
      d.innerHTML=`<input type="checkbox" class="w-4 h-4 shrink-0 accent-violet-500" title="Select for stitching"${stitchSel.includes(it.filename)?' checked':''}>
        <button class="w-8 h-8 rounded-full grid place-items-center bg-violet-600 shrink-0" title="Play">▶</button>
        <div class="min-w-0 flex-1"><div class="truncate font-mono text-[11px]">${it.filename}</div><div class="text-[10px] text-slate-500 font-mono">${it.created} · ${it.size_kb} KB</div></div>
        <a href="${it.url}" download class="text-[13px] opacity-70 hover:opacity-100" title="Download">⬇</a>
        <button class="text-[13px] opacity-50 hover:opacity-100 hover:text-red-300" title="Delete">✕</button>`;
      const cb=d.children[0], play=d.children[1], del=d.children[4];
      cb.onchange=()=>toggleStitch(it.filename,cb.checked);
      play.onclick=()=>playURL(it.url);
      del.onclick=async()=>{ stitchSel=stitchSel.filter(f=>f!==it.filename); renderStitchBar(); await fetch('/api/history/'+it.filename,{method:'DELETE'}); loadHistory(); };
      box.appendChild(d);
    }
    renderStitchBar();
  }catch(e){}
}
let stitchSel=[];
function toggleStitch(fn,on){ if(on){ if(!stitchSel.includes(fn)) stitchSel.push(fn); } else stitchSel=stitchSel.filter(f=>f!==fn); renderStitchBar(); }
function renderStitchBar(){
  const bar=$('stitchBar'); if(!bar) return;
  if(stitchSel.length<2){ bar.style.display='none'; return; }
  bar.style.display='flex';
  bar.innerHTML=`<span class="text-[12px] text-slate-300"><b>${stitchSel.length}</b> selected, in order</span><button onclick="stitchSelected()" title="Stitch the ticked files top-to-bottom into one WAV (same sample rate required)." class="text-[12px] font-bold px-3 py-1.5 rounded-full text-white" style="background:linear-gradient(135deg,#8b5cf6,#06b6d4)">⛓ Stitch into one file</button><button onclick="stitchSel=[];renderStitchBar();loadHistory()" class="text-[11px] text-slate-500 underline">clear</button>`;
}
async function stitchSelected(){
  if(stitchSel.length<2) return toast('Tick at least 2 files to stitch',false);
  const names=[...stitchSel];
  toast('Stitching '+names.length+' files… ⛓');
  try{
    const r=await fetch('/api/stitch',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({filenames:names})});
    const j=await r.json(); if(!r.ok) throw new Error(j.detail||'stitch failed');
    stitchSel=[]; renderStitchBar(); loadHistory(); playURL(j.file);
    $('outInfo').innerHTML=`<b class="text-slate-100">Stitched ⛓</b> · ${names.length} parts · ${j.duration_sec}s`;
    toast(`Stitched — ${j.duration_sec}s combined 🔊`);
  }catch(e){ toast('Stitch failed: '+friendlyErr(e),false); }
}
