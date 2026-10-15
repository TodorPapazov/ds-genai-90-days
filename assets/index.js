fetch('days/days.json').then(r=>r.json()).then(D=>{
 roadmap.innerHTML=D.weeks.map(w=>`<div class="wk"><b>Weeks ${w.weeks}</b><h3>${w.title}</h3><p>${w.desc}</p></div>`).join('');
 const by={};D.days.forEach(d=>by[d.day]=d);
 function render(){const q=search.value.toLowerCase(),p=getP();
  grid.innerHTML='';for(let i=1;i<=D.totalDays;i++){const d=by[i];
   const txt=d?(d.title+' '+d.tags.join(' ')+' day '+i).toLowerCase():('day '+i);
   if(q&&!txt.includes(q))continue;const wk=Math.ceil(i/7.5);
   const el=document.createElement(d?'a':'div');el.className='card'+(d?'':' future')+(p[i]?' done':'');
   if(d)el.href='day.html?d='+i;
   el.innerHTML=`<span class="dn">Day ${i}</span><span class="wkl">Wk ${Math.min(12,wk)}</span><p>${d?d.title:'Coming soon'}</p>`+
    (d?`<label onclick="event.stopPropagation()"><input type="checkbox" ${p[i]?'checked':''} data-d="${i}"> done</label>`:'');
   grid.appendChild(el);}
  grid.querySelectorAll('input').forEach(c=>c.onchange=e=>{e.preventDefault();setP(c.dataset.d,c.checked);render();});
  const n=Object.keys(p).length;bar.style.width=(n/D.totalDays*100)+'%';ptext.textContent=`${n} / ${D.totalDays} days complete · ${D.days.length} published`;}
 grid.addEventListener('click',e=>{if(e.target.tagName==='INPUT')e.stopPropagation();});
 search.oninput=render;render();});
