const assert = require('node:assert/strict');
const ui = require('../src/wmal/monitor/static/monitor.js');
assert.equal(ui.format(null), '未记录');
assert.equal(ui.format(NaN), '未记录');
assert.equal(ui.format(.25), '0.250');
assert.equal(ui.alignment({episode_id:'e1', step_id:4}, {episode_id:'e1', step_id:4}), 'same_step');
assert.equal(ui.alignment({episode_id:'e1', step_id:4,sim_time_s:1.2}, {episode_id:'e1', step_id:4,sim_time_s:1.2}), 'exact');
assert.equal(ui.alignment({episode_id:'e1', step_id:4,sim_time_s:1.4}, {episode_id:'e1', step_id:4,sim_time_s:1.2}), 'different_time');
assert.equal(ui.alignment({episode_id:'e2', step_id:4}, {episode_id:'e1', step_id:4}), 'different_episode');
assert.equal(ui.alignment({step_id:4}, {episode_id:'e1', step_id:4}), 'unknown');
const liveFrame={episode_id:'g1-episode-9',step_id:2,sim_time_s:.04};
const g1PlanEvent={episode_id:'g1-episode-9',observation_step:2,model_version:'motion-smoke'};
assert.equal(ui.alignment(liveFrame,g1PlanEvent),'same_step',
  'G1Agent plan evidence uses observation_step, while the live frame uses step_id');
assert.equal(ui.alignmentStatusMessage(liveFrame,g1PlanEvent),
  '回合 / 步号一致，但缺少仿真时间，无法精确对齐');
const armObservationEvent={episode_id:'arm-episode-4',step_id:2,sim_time_s:.04};
assert.equal(ui.alignmentStatusMessage({...liveFrame,episode_id:'arm-episode-4'},armObservationEvent),
  '画面与所选事件的回合 / 步号 / 仿真时间一致',
  'VisualWorkcellSession observations carry episode_id, step_id and sim_time_s');
assert.equal(ui.alignmentStatusMessage(liveFrame,{}),
  '所选事件缺少回合或步号，无法与实时帧精确对齐',
  'a frame with metadata but no selected event must identify the missing event side');
assert.deepEqual(ui.mergeEvents([{cursor:1,event:'plan'}], [{cursor:1,event:'plan'}, {cursor:2,event:'feedback'}]).map(r=>r.cursor), [1,2]);
assert.deepEqual(ui.mergeEvents([{cursor:1}], [{cursor:3}], true), [{cursor:3}]);
const element = {textContent: ''};
ui.setText(element, '<img src=x onerror=alert(1)>');
assert.equal(element.textContent, '<img src=x onerror=alert(1)>');
let storedText='unchanged',textAssignments=0;
const stableTextElement={get textContent(){return storedText;},set textContent(value){storedText=value;textAssignments++;}};
ui.setText(stableTextElement,'unchanged');
assert.equal(textAssignments,0,'unchanged values must not rewrite visible text during polling');
ui.setText(stableTextElement,'changed');
assert.equal(textAssignments,1,'changed values must still update the visible text');
assert.equal(typeof ui.RequestSequence, 'function');
const requests = new ui.RequestSequence();
const slow=requests.next(), fast=requests.next();
assert.equal(requests.isCurrent(slow), false);
assert.equal(requests.isCurrent(fast), true);
assert.equal(typeof ui.currentTaskState, 'function');
assert.deepEqual(ui.currentTaskState({task_result:{state:{status:'succeeded',task_goal:[1]}}},
                                    {task_id:'legacy-1',status:'incomplete'}), {});
assert.equal(typeof ui.candidateViews, 'function', 'Both visual and floating-G1 evidence must be projected');
assert.deepEqual(ui.candidateViews({planning_evidence:{selected_candidate:2,candidates:[
  {candidate_id:2,cost:.3,rejection:null},{candidate_id:3,cost:null,rejection:'predicted_constraint'}]}}), [
    {id:2,skill:null,score:.3,prefix:null,chosen:true,rejection:null},
    {id:3,skill:null,score:null,prefix:null,chosen:false,rejection:'predicted_constraint'}]);
assert.deepEqual(ui.candidateViews({evidence:{selected_candidate:0,candidates:[
  {candidate:0,skill:'arm_delta',score:.2,trusted_prefix:2}]}}), [
    {id:0,skill:'arm_delta',score:.2,prefix:2,chosen:true,rejection:null}]);
assert.equal(typeof ui.clearEvidenceViews, 'function', 'Run switches must clear previously displayed evidence');
const retained = new Map();
for (const id of ['eventDetail','eventDetailTitle','predictionAlignment','predictionStepLabel',
                 'artifactDetail','artifactTitle','frameAlignment','frameMeta','stateRows']) {
  retained.set(id, {textContent:'previous run evidence', hidden:false});
}
const getElement = id => retained.get(id);
ui.clearEvidenceViews(getElement);
for (const el of retained.values()) assert.notEqual(el.textContent, 'previous run evidence');
console.log('UI behavior: missing values, alignment, reconnect de-duplication, source reset and text-only rendering passed');
assert.equal(typeof ui.taskGraphRows, 'function');
assert.deepEqual(ui.taskGraphRows({nodes:{hold_A:{status:'running',actions:3,hold_count:2,recoveries:1}}}),
                [['hold_A','running',3,2,1]]);
assert.deepEqual(ui.taskGraphRows({}), []);
const taskEvidence=ui.taskEvidence([
  {event:'prediction',timestamp:1,payload:{predicted_state:[.4,.9]}},
  {event:'observation',timestamp:2,payload:{observed_state:[.42,.91]}},
  {event:'progress',timestamp:3,payload:{stage:'执行中'}}
]);
assert.deepEqual(taskEvidence.predictions,[{predicted_state:[.4,.9]}],
  'LiveTaskService event records use the event field for model prediction evidence');
assert.deepEqual(taskEvidence.observations,[{observed_state:[.42,.91]}],
  'LiveTaskService event records use the event field for actual simulator observations');
class FakeElement {
  constructor(tag){this.tagName=tag;this.children=[];this.parentNode=null;this.attributes={};this.textContent='';this.className='';this.hidden=false;}
  append(...nodes){for(const node of nodes)this.appendChild(node);}
  appendChild(node){if(node.parentNode)node.parentNode.removeChild(node);this.children.push(node);node.parentNode=this;return node;}
  insertBefore(node,reference){if(node.parentNode)node.parentNode.removeChild(node);const index=reference?this.children.indexOf(reference):-1;if(index<0)this.children.push(node);else this.children.splice(index,0,node);node.parentNode=this;return node;}
  removeChild(node){const index=this.children.indexOf(node);if(index>=0)this.children.splice(index,1);node.parentNode=null;return node;}
  replaceChildren(...nodes){for(const node of this.children)node.parentNode=null;this.children=[];this.append(...nodes);}
  setAttribute(name,value){this.attributes[name]=String(value);}
  getAttribute(name){return this.attributes[name]??null;}
}
const makeElement=(tag,text,cls)=>{const node=new FakeElement(tag);if(text!==undefined)node.textContent=String(text);if(cls)node.className=cls;return node;};
const taskRoot=new FakeElement('div'),taskCardCache=new Map(),cardOptions={pretty:value=>JSON.stringify(value),
  modeNames:{g1_navigation:'G1'},taskStateNames:{succeeded:'已完成'},onCancel(){}};
let readyTask={task_id:'task-1',state:'succeeded',spec:{mode:'g1_navigation'},
  progress:{stage:'encoding',fraction:1},
  detail:{state:'succeeded',progress:{stage:'encoding',fraction:1},video_status:'ready',
    video_url:'/api/live/run/tasks/task-1/video.mp4',events:[]}};
ui.reconcileTaskCards(taskRoot,[readyTask],taskCardCache,makeElement,cardOptions);
const statusOnlyTask={task_id:'task-1',state:'succeeded',spec:readyTask.spec,video_status:'ready',progress:{stage:'succeeded'}};
const retainedTask=ui.mergeClientTaskDetails([statusOnlyTask],[readyTask])[0];
assert.equal(retainedTask.detail,readyTask.detail,
  'a summary-only poll must retain the already loaded event and video evidence');
assert.equal(taskRoot.children[0]._taskView.progress.textContent,'已完成',
  'terminal task state must replace the last nonterminal stage such as encoding');
const player=taskRoot.children[0]._taskView.video;player.currentTime=5.25;
ui.reconcileTaskCards(taskRoot,[{...readyTask,detail:{...readyTask.detail,events:[
  {event:'prediction',payload:{state:[.4,.9]}},{event:'observation',payload:{state:[.41,.91]}}]}}],taskCardCache,makeElement,cardOptions);
assert.equal(taskRoot.children[0]._taskView.video,player,'repeated task refresh must retain the ready video DOM node');
assert.equal(taskRoot.children[0]._taskView.video.currentTime,5.25,'retained video node must preserve playback position');

// Execute the actual rendering function, not just projection helpers.
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(require.resolve('../src/wmal/monitor/static/monitor.js'),'utf8');
const renderBody = source.slice(source.indexOf('  function renderSummary(){'),source.indexOf('  function renderEvents(){'));
function renderedSummary(summary) {
  const nodes=new Map(), pairsRendered={};
  const $=id=>{
    if(!nodes.has(id))nodes.set(id,{id,textContent:'',replaceChildren(){},querySelectorAll(){return [];}});
    return nodes.get(id);
  };
  const context={...ui,summary,records:[],$,statusNames:{},
    badge:(id,text)=>ui.setText($(id),text),
    pairs:(node,rows)=>{pairsRendered[node.id]=Object.fromEntries(rows);},cells(){},plot(){},
    pretty:value=>JSON.stringify(value??'未记录',null,2)};
  vm.runInNewContext(renderBody+'\nrenderSummary();',context);
  return {nodes,pairs:pairsRendered};
}
const mixed=renderedSummary({tasks:[{task_id:'A',status:'succeeded'},{task_id:'B',status:'canceled',cycles:0}],
  latest:{task_started:{task_id:'B',goal:[.35,.87]},
    feedback:{task_id:'A',observed_state:[.43,.95,.43,.95]},
    observation:{task_id:'A',state:[9,9]},task_result:{task_id:'A',baseline:'A3',model_version:'old'},
    agent_state:{task_id:'B',state:{status:'canceled',executed_cycles:0}}}});
assert.equal(mixed.nodes.get('robotState').textContent,JSON.stringify('未记录'));
assert.equal(mixed.nodes.get('recentFeedback').textContent,JSON.stringify('未记录'));
assert.equal(mixed.pairs.modelState['模型版本'],undefined);
assert.equal(mixed.pairs.modelState['基线'],undefined);
const legacy=renderedSummary({tasks:[{task_id:'legacy',status:'failed'}],latest:{
  task_started:{task_id:'legacy'},mission_validated:{task_id:'legacy',goals:[{x:1,y:2}]},
  planning_event:{task_id:'legacy',reason:'obstacle_blocks_goal'}}});
assert.equal(JSON.stringify(legacy.pairs.agentState['任务目标']),JSON.stringify([{x:1,y:2}]));
assert.equal(legacy.pairs.agentState['最近失败原因'],'obstacle_blocks_goal');

const g1Plan={episode_id:'walk-1',observation_step:9,action:{vx:.29,vy:0,yaw_rate:-.06,duration_s:.5},
  predicted_state:{episode_id:'walk-1',step_id:10,sim_time_s:5.4,x:1.1,y:-.04,z:.78,yaw:-.15,
    vx:.29,vy:0,vz:0,yaw_rate:-.06,roll:0,pitch:0,pelvis_height:.78,joint_positions:{left_hip_pitch:.2}}};
const g1Final={phase:'final',episode_id:'walk-1',step_id:9,sim_time_s:5.26,x:1.15,y:-.07,z:.786,
  yaw:-.154,vx:.25,vy:-.04,vz:.01,yaw_rate:-.09,roll:.02,pitch:0,pelvis_height:.786,
  joint_positions:{left_hip_pitch:.21}};
ui.reconcileTaskCards(taskRoot,[{task_id:'walk-unaligned',state:'failed',spec:{mode:'g1_navigation'},
  detail:{state:'failed',events:[{event:'prediction',payload:g1Plan},{event:'observation',payload:g1Final}]}}],
  taskCardCache,makeElement,cardOptions);
const unalignedView=taskCardCache.get('walk-unaligned')._taskView;
assert.match(unalignedView.help.textContent,/执行动作前.*执行动作后/,
  'the card must explain that prediction is made before execution and observation after it');
assert.match(unalignedView.alignment.textContent,/暂不比较/,
  'a future prediction and an earlier final observation must not be presented as a matched error');
assert.match(unalignedView.alignment.textContent,/第 10 步.*第 9 步/,
  'the comparison status must name both time steps');
assert.equal(unalignedView.tableBody.children[0].children[1].textContent,'1.100 m',
  'G1 positions should be readable with units instead of hidden in JSON');
assert.equal(unalignedView.tableBody.children[0].children[3].textContent,'—',
  'misaligned evidence must not display a numeric prediction error');
assert.equal(unalignedView.predictionRaw.open,false,'raw prediction evidence starts collapsed');
assert.equal(unalignedView.observationRaw.open,false,'raw observation evidence starts collapsed');

ui.reconcileTaskCards(taskRoot,[{task_id:'walk-aligned',state:'succeeded',spec:{mode:'g1_navigation'},
  detail:{state:'succeeded',events:[{event:'prediction',payload:g1Plan},{event:'observation',payload:{...g1Final,
    step_id:10,sim_time_s:5.4,x:1.15}}]}}],taskCardCache,makeElement,cardOptions);
const alignedView=taskCardCache.get('walk-aligned')._taskView;
assert.match(alignedView.alignment.textContent,/可直接对照/,
  'matching episode, step and simulation time should enable a comparison');
assert.equal(alignedView.tableBody.children[0].children[3].textContent,'0.050 m',
  'aligned G1 evidence should show the measured absolute position difference');

const armPlan={episode_id:'arm-1',step_id:4,prefix_length:2,model_version:'arm-smoke',
  uncertainty_kind:'calibrated_error_bound',
  predicted_states:[[.1,.2,.3,.4],[.15,.25,.35,.45]]};
const armObservation={episode_id:'arm-1',step_id:6,sim_time_s:1.2,
  observed_state:[.16,.24,.36,.44],state_order:['left_shoulder_joint','left_elbow_joint',
    'left_shoulder_target','left_elbow_target']};
const armAlignedTask={task_id:'arm-aligned',state:'succeeded',spec:{mode:'arm_joint_targets'},
  detail:{state:'succeeded',events:[{event:'prediction',payload:armPlan},
    {event:'observation',payload:armObservation}]}};
ui.reconcileTaskCards(taskRoot,[armAlignedTask],taskCardCache,makeElement,cardOptions);
const armView=taskCardCache.get('arm-aligned')._taskView;
assert.match(armView.alignment.textContent,/可直接对照/,
  'the executed arm prefix should align its final predicted row with the real observation');
assert.equal(armView.tableBody.children[0].children[0].textContent,'左肩关节',
  'arm state labels should use the recorded order in readable Chinese');
assert.equal(armView.tableBody.children[0].children[3].textContent,'0.010 rad',
  'arm differences should use the recorded joint-state units');
const armEvidenceRow=armView.tableBody.children[0];
ui.reconcileTaskCards(taskRoot,[armAlignedTask],taskCardCache,makeElement,cardOptions);
assert.equal(armView.tableBody.children[0],armEvidenceRow,
  'unchanged task refreshes must retain evidence rows instead of flashing the whole table');

const armPlanWithoutPrefix={...armPlan,prefix_length:undefined};
ui.reconcileTaskCards(taskRoot,[{task_id:'arm-prefix-unrecorded',state:'succeeded',spec:{mode:'arm_joint_targets'},
  detail:{state:'succeeded',events:[{event:'prediction',payload:armPlanWithoutPrefix},
    {event:'observation',payload:armObservation}]}}],taskCardCache,makeElement,cardOptions);
const unverifiedArmView=taskCardCache.get('arm-prefix-unrecorded')._taskView;
assert.match(unverifiedArmView.alignment.textContent,/执行前缀.*未记录.*暂不比较/,
  'an arm prediction without an execution-prefix record must not be treated as aligned');
assert.equal(unverifiedArmView.tableBody.children[0].children[3].textContent,'—',
  'missing execution-prefix evidence must suppress numeric arm differences');
