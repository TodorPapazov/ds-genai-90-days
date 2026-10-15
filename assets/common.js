const KEY='ds90-progress';
const getP=()=>JSON.parse(localStorage.getItem(KEY)||'{}');
const setP=(d,v)=>{const p=getP();v?p[d]=1:delete p[d];localStorage.setItem(KEY,JSON.stringify(p));};
const pad=n=>String(n).padStart(2,'0');
const esc=s=>s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function copyText(t,btn){navigator.clipboard.writeText(t).then(()=>{const o=btn.textContent;btn.textContent='Copied!';setTimeout(()=>btn.textContent=o,1500);});}
const KW={python:'import|from|as|def|return|for|in|if|else|elif|print|True|False|None|with|lambda|class',
sql:'SELECT|FROM|WHERE|GROUP|BY|ORDER|AS|AVG|MEDIAN|STDDEV|PERCENTILE_CONT|WITHIN|JOIN|ON|OVER|PARTITION|LIMIT|AND|OR',
bash:'conda|create|activate|git|add|commit|push|python|pip|cd|mkdir'};
function highlight(code,lang){
 const kw=new RegExp('^(?:'+(KW[lang]||'')+')$', lang==='sql'?'i':'');
 const re=/(#[^\n]*|--[^\n]*)|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|(\b\d+(?:\.\d+)?\b)|([A-Za-z_]\w*)/g;
 let out='',last=0,m;
 while((m=re.exec(code))){out+=esc(code.slice(last,m.index));last=re.lastIndex;
  if(m[1]&&(lang!=='sql'?m[1][0]==='#':m[1][0]==='-'))out+=`<span class="c">${esc(m[1])}</span>`;
  else if(m[1])out+=esc(m[1]);
  else if(m[2])out+=`<span class="s">${esc(m[2])}</span>`;
  else if(m[3])out+=`<span class="n">${m[3]}</span>`;
  else out+=kw.test(m[4])?`<span class="k">${m[4]}</span>`:m[4];}
 return out+esc(code.slice(last));}
