"""One bounded cloud sync; no competition data or raw logs are published."""
import ast,hashlib,json,os,re,shutil,subprocess,tempfile
from pathlib import Path
from collect import utc,parse_line
from github_state import load as load_state,publish as publish_state

KAGGLE=shutil.which('kaggle') or str(Path.home()/'.local/bin/kaggle.exe')
ASR='khangung/lifemate-v15-1-asr'
TRAIN='khangung/lifemate-v15-2-train'
ASR_VERSION=4
SOURCE_CACHE=Path(tempfile.mkdtemp(prefix='lifemate-source-'))

def cmd(*args):
 p=subprocess.run([KAGGLE,*args],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90)
 if p.returncode:raise RuntimeError('Kaggle command failed; inspect credentials or kernel permissions')
 return p.stdout

def pull(handle,directory):
 cached=SOURCE_CACHE/handle.replace('/','_')
 if not (cached/'kernel-metadata.json').exists():cmd('kernels','pull',handle,'-p',str(cached),'-m')
 directory=cached
 meta=json.loads((directory/'kernel-metadata.json').read_text(encoding='utf-8-sig'))
 path=directory/Path(meta['code_file']).name
 return meta,path

def config(path,handle,asr):
 notebook=json.loads(path.read_text(encoding='utf-8-sig'))
 source='\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code' and not ''.join(c['source']).startswith('%%'))
 tree=ast.parse(source);values={}
 for node in tree.body:
  if not isinstance(node,ast.Assign):continue
  try:
   if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='loads':value=json.loads(ast.literal_eval(node.value.args[0]))
   else:value=ast.literal_eval(node.value)
  except (ValueError,TypeError,SyntaxError):continue
  for target in node.targets:
   if isinstance(target,ast.Name):values[target.id]=value
 cfg=values['CONFIG'];t=cfg['training']
 return dict(kernel=handle,version='V15',model=values['ASR']['id'] if asr else cfg['model']['id'],parentVersion=None,parentStep=0,targetStep=None if asr else t['max_additional_steps'],maxHours=None if asr else t['max_train_hours'],evalEvery=None if asr else t['eval_every'],evalOrigin=0,saveEvery=None if asr else t['save_every'],patience=None if asr else t['early_stopping_patience'],probeLimit=None if asr else cfg['evaluation']['probe_samples'],loraRank=None if asr else t['lora_rank'],gpu=None,source='version-code',sourceHash=hashlib.sha256(path.read_bytes()).hexdigest())

def stream(handle):
 p=subprocess.Popen([KAGGLE,'kernels','logs',handle,'--follow'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8'))
 try:out,_=p.communicate(timeout=25)
 except subprocess.TimeoutExpired:
  p.terminate();out,_=p.communicate(timeout=10)
 text=out.decode('utf-8',errors='replace')
 if 'Permission' in text or 'Cannot stream' in text:raise RuntimeError('Cannot access live log stream')
 return text

def parse(state,text,asr):
 current=0;total=None
 for line in text.splitlines():
  progress=re.search(r'(\d+)\s*/\s*(\d+)\s*\[',line)
  if progress:current,total=map(int,progress.groups())
  if asr and re.search(r"['\"]loss['\"]\s*:",line):
   state['step']=current
   if total:state['config']['targetStep']=total
   state['phase']='training'
  parse_line(state,line)
 if not asr:
  agent=[]
  for m in re.finditer(r'AGENT_EVAL: (\{[^\n]+\})',text):
   try:
    d=json.loads(m[1]);row={'step':int(d['step']),'agentScore':float(d['agent_score']),'stateMatch':float(d['state_match_rate']),'toolF1':float(d['tool_f1']),'improved':bool(d['improved'])}
    if d.get('diagnostic_loss') is not None:row['diagnosticLoss']=float(d['diagnostic_loss'])
    agent.append(row)
   except (ValueError,KeyError,TypeError):pass
  state['agentEvaluations']=list({r['step']:r for r in agent}.values())[-1000:]
 if asr:
  logs=[]
  seen=set()
  for line in text.splitlines():
   if not line.strip().startswith('{'):continue
   try:
    d=ast.literal_eval(line.strip())
    if 'loss' not in d:continue
    row={'loss':float(d['loss']),'epoch':float(d['epoch']) if 'epoch' in d else None}
    for a,b in [('learningRate','learning_rate'),('gradNorm','grad_norm')]:
     if b in d:row[a]=float(d[b])
    key=json.dumps(row,sort_keys=True)
    if key not in seen:logs.append(row);seen.add(key)
   except (ValueError,SyntaxError,TypeError):pass
  state['asrLogs']=logs[-2000:]
  info=state.setdefault('asr',{'stage':'preparing','baselineWer':None,'fineTunedWer':None,'cachedClips':None,'totalClips':None})
  for marker,key in [('ASR BASE','baselineWer'),('ASR FINETUNED','fineTunedWer')]:
   for match in re.finditer(marker+r' (\{[^\n]+\})',text):
    try:
     metric=ast.literal_eval(match[1]);info[key]=float(metric['wer'])
     side='Before' if key=='baselineWer' else 'After'
     for a,b in [('cer','cer'),('exact','exact_match'),('clips','clips')]:
      if b in metric:info[a+side]=float(metric[b])
    except (ValueError,KeyError,SyntaxError):pass
  if info['baselineWer'] is not None:info['stage']='training'
  if 'train_runtime' in text:info['stage']='evaluating'
  if 'ASR FINETUNED' in text:info['stage']='evaluating'
  if 'ASR SELECTION' in text:info['stage']='transcribing'
  shards={}
  for match in re.finditer(r'ASR shard (\d+)/(\d+): (\d+)/(\d+)',text):
   shards[int(match[1])]=(int(match[2]),int(match[3]),int(match[4]))
  if shards and len(shards)==next(iter(shards.values()))[0]:
   info['cachedClips']=sum(v[1] for v in shards.values());info['totalClips']=sum(v[2] for v in shards.values())
  if 'ASR DONE:' in text:info['stage']='finished'
  # Trainer progress remains useful even when its logger does not print loss dictionaries.
  training_text=text.split('trainable params:',1)[-1].split('ASR FINETUNED',1)[0]
  progress=list(re.finditer(r'(\d+)\s*/\s*(\d+)\s*\[([\d:]+)<',training_text))
  if progress:
   horizon=int(progress[0][2]);progress=[x for x in progress if int(x[2])==horizon]
   latest=progress[-1];state['step']=int(latest[1]);state['config']['targetStep']=int(latest[2]);state['phase']='evaluating' if info['stage']=='evaluating' else 'training'
   parts=[int(x) for x in latest[3].split(':')]
   state['hours']=sum(value*60**i for i,value in enumerate(reversed(parts)))/3600
 state['events']=events(text,'ASR' if asr else 'Qwen',state.get('events',[]))
 return state

def events(text,stage,previous):
 rows=[r for r in previous if r['stage']!=stage];seen=set()
 def add(message,level='info'):
  if message not in seen:rows.append({'stage':stage,'message':message[:240],'level':level});seen.add(message)
 for line in text.splitlines():
  line=line.strip()
  if line.startswith('EVAL_SAMPLE '):
   try:
    d=json.loads(line.split(' ',1)[1]);add(f"Đánh giá {d['n']}/{d['total']} · replay tạm {d['avg']:.2f}% · {d['sec']:.1f}s")
   except (ValueError,KeyError):pass
  if line.startswith('ATTENTION_PREFLIGHT_OK:'):add('Attention preflight đạt · forward khớp tham chiếu, backward hữu hạn')
  if line.startswith('ATTENTION_BACKEND:'):add('T4 · memory-efficient SDPA · math fallback đã tắt')
  if line.startswith('MEMORY_SMOKE_START:'):add('Bắt đầu kiểm tra VRAM ở giới hạn context')
  if line.startswith('MEMORY_SMOKE_OK:'):
   try:
    d=json.loads(line.split(':',1)[1]);add(f"VRAM smoke đạt · {d['context_tokens']} token · peak {d['peak_vram_gib']:.3f} GiB")
   except (ValueError,KeyError):pass
  if line.startswith('GPU_PREFLIGHT '):
   try:
    d=json.loads(line.split(' ',1)[1]);add(f"GPU preflight · CUDA {d['cuda_build']} · {d['count']} GPU · {'sẵn sàng' if d['available'] else 'không khả dụng'}")
   except (ValueError,KeyError):pass
  if line.startswith('GPU_AFTER_INSTALL') and 'True' in line:add('Sau cài thư viện · CUDA vẫn khả dụng')
  if line.startswith('+ pip install'):add('Cài thư viện')
  if 'Fetching ' in line and 'files' in line:add('Tải model từ Hugging Face')
  p=re.search(r'(\d+)\s*/\s*(\d+)\s*\[([\d:]+)<',line)
  if p:add(f'Tiến độ {p[1]}/{p[2]} · thời gian {p[3]}')
  for marker in ['ASR BASE','ASR FINETUNED']:
   if line.startswith(marker+' '):
    try:
     d=ast.literal_eval(line[len(marker)+1:]);add(f"{marker} · WER {100*d['wer']:.2f}% · CER {100*d['cer']:.2f}% · {d['clips']} audio")
    except (ValueError,KeyError,SyntaxError):pass
  if line.startswith('ASR shard '):
   m=re.search(r'ASR shard (\d+)/(\d+): (\d+)/(\d+)',line)
   if m:add(f'Transcript · GPU {m[1]}/{m[2]} · {m[3]}/{m[4]} audio')
  if line.startswith('ASR SELECTION'):add('Đã chọn ASR · bắt đầu tạo transcript')
  if line.startswith('ASR DONE:'):add('ASR hoàn tất')
  if line.startswith('LONG RUN PROGRESS:'):
   m=re.search(r'global=(\d+).*hours=([\d.]+)',line)
   if m:add(f"Train · bước {m[1]} · {m[2]} giờ")
  if line.startswith('AGENT_EVAL:'):
   try:
    d=json.loads(line.split(':',1)[1]);add(f"Đánh giá · bước {d['step']} · điểm agent {d['agent_score']:.2f} · state {d['state_match_rate']:.4f} · tool F1 {d['tool_f1']:.4f}")
   except (ValueError,KeyError,TypeError):pass
  if line.startswith('{') and ('loss' in line):
   try:
    d=ast.literal_eval(line)
    for k in ['loss','eval_loss']:
     if k in d:add(f'{k}: {float(d[k]):.6f}')
   except (ValueError,SyntaxError):pass
  if 'Saving model checkpoint' in line:
   m=re.search(r'checkpoint-(\d+)',line)
   if m:add(f'Lưu checkpoint · bước {m[1]}')
  if line.startswith('LONG RUN COMPLETE:'):add('Huấn luyện Qwen hoàn tất · chuyển sang đánh giá / tạo kết quả')
  if 'OutOfMemoryError:' in line:add('CUDA hết VRAM','error')
  elif line.startswith('Traceback (most recent call last):'):add('Notebook ghi nhận lỗi · mở Kaggle để xem traceback','error')
 return rows[-300:]

def main():
 previous=load_state()
 with tempfile.TemporaryDirectory() as temp:
  root=Path(temp);meta,train_path=pull(TRAIN,root/'train')
  # Version 1 is the review-only save. Any later launched version is observed, never launched again.
  from kaggle.api.kaggle_api_extended import KaggleApi
  from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
  api=KaggleApi();api.authenticate()
  with api.build_kaggle_client() as client:
   request=ApiGetKernelRequest();request.user_name=TRAIN.split('/')[0];request.kernel_slug=TRAIN.split('/')[1]
   train_version=client.kernels.kernels_api_client.get_kernel(request).metadata.current_version_number
  if not isinstance(train_version,int):raise RuntimeError('Missing train version provenance')
  # Keep following the last actual run; a review-only save must not look like completed training.
  previous_handle=previous.get('config',{}).get('kernel','')
  if previous_handle.startswith(TRAIN+'/'):
   train_version=int(previous_handle.rsplit('/',1)[1])
  asr=train_version<=1
  handle=f'{ASR}/{ASR_VERSION}' if asr else f'{TRAIN}/{train_version}'
  _,path=pull(handle,root/'active')
  c=config(path,handle,asr)
  status=cmd('kernels','status',handle)
  text=stream(handle)
  same=previous['config']['kernel']==handle
  state=previous if same else dict(schema=2,config=c,updatedAt=utc(),source='kaggle',status='unknown',phase='preparing',step=0,hours=0,baseline=None,train=[],evaluations=[],references=[],history=previous.get('history',[]),events=previous.get('events',[]),asrLogs=previous.get('asrLogs',[]),agentEvaluations=previous.get('agentEvaluations',[]),**({'asr':previous['asr']} if 'asr' in previous else {}))
  state['config']=c
  state=parse(state,text,asr)
  state.update(updatedAt=utc(),source='kaggle')
  if 'COMPLETE' in status:state.update(status='complete',phase='finished')
  elif 'ERROR' in status or 'CANCEL' in status:state.update(status='error')
  elif any(s in status for s in ['RUNNING','QUEUED']):
   state['status']='running'
   if 'QUEUED' in status:state['events'].append({'stage':'ASR' if asr else 'Qwen','level':'info','message':'Kaggle đang xếp hàng cấp GPU; chưa thực thi model'})
  else:state['status']='unknown'
  publish_state(state)
  print(f'Synced {handle}: {state["status"]}; step {state["step"]}; {len(state["train"])} loss rows')
  if asr and state['status']=='complete':
   dest=root/'result'
   cmd('kernels','output',handle,'-p',str(dest),'--file-pattern','ASR_SUMMARY.json$','--quiet')
   found=list(dest.rglob('ASR_SUMMARY.json'))
   if len(found)!=1 or json.loads(found[0].read_text()).get('smoke') is not False:raise RuntimeError('Full ASR summary missing; Qwen not launched')
   # Attach output only once it exists. Earlier quick-save cannot attach a running kernel.
   meta['kernel_sources']=[ASR];meta['enable_gpu']=True;meta['enable_internet']=True;meta['machine_shape']='NvidiaTeslaT4';meta.pop('docker_image',None)
   (train_path.parent/'kernel-metadata.json').write_text(json.dumps(meta),encoding='utf-8')
   result=cmd('kernels','push','-p',str(train_path.parent),'--accelerator','NvidiaTeslaT4')
   match=re.search(r'Kernel version (\d+)',result)
   if not match:raise RuntimeError('Train launch response uncertain; next sync checks existing version before any retry')
   version=int(match[1]);newconfig=config(train_path,f'{TRAIN}/{version}',False)
   state.update(config=newconfig,status='running',phase='preparing',step=0,hours=0,baseline=None,train=[],evaluations=[],updatedAt=utc())
   publish_state(state)
   print('ASR completed; Qwen training launched. No competition submission was sent.')
  return state

if __name__=='__main__':main()



