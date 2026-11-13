import {jobs} from '../api/entities.js';
import {workspace} from '../stores/workspace.js';
import {session} from '../stores/session.js';
import {heading} from '../components/common/shell.js';
import {chart} from '../components/common/chart.js';
import {table} from '../components/common/table.js';
import {status} from '../components/common/status.js';
import {esc,fmt,date,notify} from '../utils/dom.js';
function splitSheet(s){
 if(!s)return '';
 const sides=[['训练',s.train],['留出验证',s.validation]];
 return '<section class="panel split-sheet"><div class="section-title"><h2>读者隔离分组</h2><span>同一位读者的全部周窗口只进入一侧</span></div>'
  +'<div class="split-basis"><p class="eyebrow">SPLIT BASIS</p><p>'+esc(s.basis)+'</p></div>'
  +'<div class="metrics">'+sides.map(([name,g])=>'<div><span>'+esc(name)+' · 读者</span><strong>'+esc(fmt(g?.readers??0))+'<small>人</small></strong></div><div><span>'+esc(name)+' · 样本条</span><strong>'+esc(fmt(g?.rows??0))+'<small>条</small></strong></div>').join('')+'</div>'
  +'<p class="hint">输入 '+esc(fmt(s.input_rows||0))+' 条 · 去重后 '+esc(fmt(s.unique_rows||0))+' 条 · 忽略完全重复 '+esc(fmt(s.duplicate_rows_ignored||0))+' 条；训练未续读 '+esc(fmt(s.train?.not_renewed??0))+' 条 / 已续读 '+esc(fmt(s.train?.renewed??0))+' 条，验证未续读 '+esc(fmt(s.validation?.not_renewed??0))+' 条 / 已续读 '+esc(fmt(s.validation?.renewed??0))+' 条。</p>'
  +'<h3 class="detail-heading">留出读者（'+esc(fmt((s.holdout_readers||[]).length))+' 位）</h3>'
  +'<div class="holdout-list">'+(s.holdout_readers||[]).map(id=>'<span>'+esc(id)+'</span>').join('')+'</div>'
  +'<p class="hint">'+esc(s.method)+'</p></section>';
}
export async function results(id){
 workspace.jobs=await jobs.list();const j=id?await jobs.get(id):(workspace.jobs.find(j=>j.id===workspace.selectedJob)||workspace.jobs[0]);
 if(j&&!workspace.jobs.some(x=>x.id===j.id))workspace.jobs.push(j);
 if(!j)return heading('04 / REVIEW','结果复核','尚无分析结果。')+'<div class="empty">先到分析工作台，选择批次并提交一次计算。<a href="#/workbench">前往工作台 ↗</a></div>';
 workspace.selectedJob=j.id;const r=j.result||{};
 const detailRows=r.table?.rows||[],holdoutRows=detailRows.filter(x=>x.split==='留出验证'),reviewReaders=r.review?r.review.readers.length:holdoutRows.filter(x=>x.decision==='建议编辑复核').length;
 return heading('04 / REVIEW','结果复核与业务解释',j.name,'<button class="button" id="refresh-results">刷新结果 ↻</button>')+'<div class="result-selector"><label>分析记录<select id="select-job">'+workspace.jobs.map(x=>'<option value="'+x.id+'" '+(x.id===j.id?'selected':'')+'>'+esc(x.name)+' · '+esc(x.status)+'</option>').join('')+'</select></label>'+status(j.status)+'<span>'+date(j.created_at)+'</span><span class="approval">'+(j.approved?'已获业务批准':'待负责人复核')+'</span></div>'+(j.status==='completed'?'<section class="result-lead"><p class="eyebrow">COMPUTED FINDING</p><h2>'+esc(r.summary)+'</h2><div class="metrics">'+(r.metrics||[]).map(m=>'<div><span>'+esc(m.label)+'</span><strong>'+esc(fmt(m.value))+'<small>'+esc(m.unit||'')+'</small></strong></div>').join('')+'</div></section>'+splitSheet(r.split)+'<section class="panel">'+chart(r.chart)+'</section><div class="two-columns"><section class="panel"><div class="section-title"><h2>业务解释</h2><span>由计算结果生成</span></div><ol class="questions">'+(r.findings||[]).map(v=>'<li>'+esc(v)+'</li>').join('')+'</ol>'+(r.warnings||[]).map(v=>'<p class="warning">'+esc(v)+'</p>').join('')+'</section><section class="method-sheet compact"><p class="eyebrow">REPRODUCIBILITY</p><h3>计算口径与参数</h3><p>'+esc(r.method)+'</p><pre>'+esc(JSON.stringify(j.params,null,2))+'</pre>'+(session.user.role==='admin'&&!j.approved?'<button class="primary" id="approve-job">批准此业务结论</button>':'')+'<p class="hint">仅批准结果可进入定时分发档案。</p></section></div><section class="panel"><div class="section-title"><h2>结果明细</h2><span>留出验证 '+holdoutRows.length+' 条 · 复核名单 '+reviewReaders+' 位读者 · 全部 '+detailRows.length+' 条</span></div>'+table(r.table?.columns||[],detailRows)+'</section>':'<section class="panel empty">'+(j.status==='failed'?'计算失败：'+esc(j.error)+'。请返回工作台调整参数或数据。':'后台正在执行计算。点击刷新结果查看最新进展。')+'<a href="#/workbench">返回工作台 ↗</a></section>');
}
export function bindResults(refresh){document.querySelector('#refresh-results')?.addEventListener('click',refresh);document.querySelector('#select-job')?.addEventListener('change',e=>{workspace.selectedJob=Number(e.target.value);location.hash='#/results/'+e.target.value;});document.querySelector('#approve-job')?.addEventListener('click',async()=>{try{await jobs.approve(workspace.selectedJob);notify('已批准，可创建分发计划');await refresh();}catch(e){notify(e.message,true);}});}
