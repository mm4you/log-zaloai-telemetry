"""Sanitised public telemetry only; never write credentials, notebook source or dataset rows."""
import base64,json,os,time,urllib.request,urllib.error
from pathlib import Path
REPO=os.environ.get('GITHUB_REPOSITORY','mm4you/log-zaloai-telemetry')
BRANCH='telemetry'

def request(path,body=None,method='GET'):
 token=os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')
 if not token:raise RuntimeError('Cloud GitHub token missing')
 headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','Content-Type':'application/json','X-GitHub-Api-Version':'2022-11-28'}
 req=urllib.request.Request('https://api.github.com/repos/'+REPO+path,data=json.dumps(body).encode() if body is not None else None,headers=headers,method=method)
 with urllib.request.urlopen(req,timeout=30) as r:return json.load(r) if r.status!=204 else None

def load():
 try:
  obj=request('/contents/latest.json?ref='+BRANCH)
  return json.loads(base64.b64decode(obj['content']))
 except urllib.error.HTTPError as error:
  if error.code!=404:raise
  return json.loads((Path(__file__).resolve().parents[1]/'public/snapshot.json').read_text(encoding='utf-8-sig'))

def publish(state):
 state['events']=state.get('events',[])[-300:]
 current=state['config']['version'];history=state.setdefault('history',[])
 entry=next((r for r in history if r['version']==current),None)
 if entry is None:
  entry={'version':current,'status':state['status'],'detail':''};history.insert(0,entry)
 entry.update(status=state['status'],detail='ASR' if 'whisper' in (state['config'].get('model') or '').lower() else 'Qwen')
 for row in history:
  if row['version']!=current and row.get('detail')=='Phiên đang xem':row['detail']='Phiên trước'
 # The producer only constructs numeric/config/event fields; raw logs never reach this function.
 allowed={'schema','config','asr','asrLogs','agentEvaluations','events','updatedAt','source','status','step','hours','baseline','phase','train','evaluations','references','history','benchmarks','peers','peerAsr'}
 payload={k:v for k,v in state.items() if k in allowed}
 raw=json.dumps(payload,ensure_ascii=False,allow_nan=False).encode()
 if len(raw)>200000:raise ValueError('Telemetry payload too large')
 try:
  old=request('/contents/latest.json?ref='+BRANCH)
 except urllib.error.HTTPError as error:
  if error.code!=404:raise
  try:request('/git/ref/heads/'+BRANCH)
  except urllib.error.HTTPError as missing:
   if missing.code!=404:raise
   # An orphan data branch cannot overwrite source history or trigger application builds.
   tree=request('/git/trees',{'tree':[{'path':'vercel.json','mode':'100644','type':'blob','content':'{"git":{"deploymentEnabled":false}}'},{'path':'latest.json','mode':'100644','type':'blob','content':raw.decode()}]},'POST')
   commit=request('/git/commits',{'message':'Initial sanitised telemetry [skip ci]','tree':tree['sha'],'parents':[]},'POST')
   request('/git/refs',{'ref':'refs/heads/'+BRANCH,'sha':commit['sha']},'POST');frame(raw);return
  old=None
 body={'branch':BRANCH,'message':'Update sanitised telemetry [skip ci]','content':base64.b64encode(raw).decode()}
 if old:body['sha']=old['sha']
 request('/contents/latest.json',body,'PUT')
 frame(raw)
 if state['status'] in ['complete','error']:
  body={'branch':BRANCH,'message':'Preserve final notebook status [skip ci]','content':base64.b64encode(raw).decode()}
  try:body['sha']=request('/contents/terminal.json?ref='+BRANCH)['sha']
  except urllib.error.HTTPError as error:
   if error.code!=404:raise
  request('/contents/terminal.json',body,'PUT')

def frame(raw):
 # Write the next minute before readers request it. Each path is immutable once created.
 for slot in [int(time.time()//60),int(time.time()//60)+1]:
  path=f'/contents/frames/{slot}.json'
  try:request(path+'?ref='+BRANCH);continue
  except urllib.error.HTTPError as error:
   if error.code!=404:raise
  request(path,{'branch':BRANCH,'message':'Publish immutable minute frame [skip ci]','content':base64.b64encode(raw).decode()},'PUT')

def dispatch_successor():
 request('/actions/workflows/kaggle-telemetry.yml/dispatches',{'ref':'main'},'POST')
 print('Next cloud worker dispatched',flush=True)
