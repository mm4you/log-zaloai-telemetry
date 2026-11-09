"""One bounded cloud sync; no competition data or raw logs are published."""
import ast,hashlib,json,os,re,shutil,subprocess,tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collect import utc,parse_line
from github_state import load as load_state,publish as publish_state

KAGGLE=shutil.which('kaggle') or str(Path.home()/'.local/bin/kaggle.exe')
ASR='khangung/lifemate-v15-1-asr'
TRAIN='khangung/lifemate-v15-2-train'
ASR_VERSION=4
SIDE='qminh7706/lifemate-v17-1-asr'  # log-only: its lines fill the ASR rows of the notebook log; the pinned run is untouched
SIDE_TRAIN='qminh7706/lifemate-v16-2-train'  # log-only too: shown as "Minh · ..." rows after the pinned run's own rows
DUY_EVAL='castoraletheia/lifemate-v17-eval-307-public-634'  # log-only: Duy's 307 + public eval, shown inside Duy's panel as eval progress
DUY_TRAIN='castoraletheia/lifemate-v17-train'  # log-only: Duy's training run, shown as a "Train · Duy" peer panel
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
 declared=re.match(r'^(V\d+)\b',cfg.get('note',''))
 version=cfg.get('version') or (declared[1] if declared else 'V15')
 mode=values.get('RUN_MODE','train');benchmark=mode in ('benchmark','validate')
 parent=int(cfg['model'].get('parent_step') or 0)
 return dict(kernel=handle,version=version,runMode=mode,model=values['ASR']['id'] if asr else cfg['model']['id'],parentVersion=cfg['model'].get('parent_version'),parentStep=parent,targetStep=None if asr or benchmark else parent+t['max_additional_steps'],maxHours=None if asr or benchmark else t['max_train_hours'],evalEvery=None if asr or benchmark else t['eval_every'],evalOrigin=parent,saveEvery=None if asr or benchmark else t['save_every'],patience=None if asr or benchmark else t['early_stopping_patience'],probeLimit=None if asr else cfg['evaluation']['probe_samples'],loraRank=None if asr else t['lora_rank'],gpu=None,source='version-code',sourceHash=hashlib.sha256(path.read_bytes()).hexdigest())

def stream(handle):
 p=subprocess.Popen([KAGGLE,'kernels','logs',handle,'--follow'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8'))
 try:out,_=p.communicate(timeout=25)
 except subprocess.TimeoutExpired:
  p.terminate();out,_=p.communicate(timeout=10)
 text=out.decode('utf-8',errors='replace')
 if 'Permission' in text or 'Cannot stream' in text:raise RuntimeError('Cannot access live log stream')
 return text

def parse(state,text,asr):
 # Kaggle replays the full log on every poll. Rebuild, never carry its last step into old records.
 if not asr:
  state.update(step=state['config']['parentStep'],hours=0,baseline=None,train=[],evaluations=[],agentEvaluations=[],phase='preparing',benchmarks=[])
 current=0;total=None
 for line in text.splitlines():
  if not asr and line.strip().startswith('INFERENCE_BENCHMARK:'):
   try:
    d=ast.literal_eval(line.split(':',1)[1].strip())
    state['benchmarks'].append(dict(profile=str(d['inference_profile'])[:60],samples=int(d['sample_count']),score30=float(d['agent_score_30s']),p50=d.get('latency_p50_seconds'),p95=d.get('latency_p95_seconds'),max=d.get('latency_max_seconds'),within20=d.get('within_target_20s'),over30=d.get('over_budget_30s'),asr=None,deadline=None))
   except (ValueError,SyntaxError,KeyError,TypeError):pass
  progress=re.search(r'(\d+)\s*/\s*(\d+)\s*\[',line)
  if progress:current,total=map(int,progress.groups())
  if asr and re.search(r"['\"]loss['\"]\s*:",line):
   state['step']=current
   if total:state['config']['targetStep']=total
   state['phase']='training'
  if asr or not re.search(r"['\"]eval_loss['\"]\s*:",line):parse_line(state,line)
  if not asr and line.strip().startswith('AGENT_EVAL_START:'):state['phase']='evaluating'
  if not asr and line.strip().startswith('AGENT_EVAL:'):state['phase']='training'
 if not asr:
  agent=[]
  for m in re.finditer(r'AGENT_EVAL: (\{[^\n]+\})',text):
   try:
    d=json.loads(m[1]);row={'step':int(d['step']),'agentScore':float(d['agent_score']),'stateMatch':float(d['state_match_rate']),'toolF1':float(d['tool_f1']),'improved':bool(d['improved'])}
    if d.get('diagnostic_loss') is not None:row['diagnosticLoss']=float(d['diagnostic_loss'])
    agent.append(row)
   except (ValueError,KeyError,TypeError):pass
  state['agentEvaluations']=sorted({r['step']:r for r in agent}.values(),key=lambda r:r['step'])[-1000:]
  for row in state['agentEvaluations']:
   if 'diagnosticLoss' not in row:continue
   if row['step']==state['config']['parentStep']:state['baseline']=row['diagnosticLoss']
   else:state['evaluations'].append({'step':row['step'],'loss':row['diagnosticLoss']})
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
  tagged=re.match(r'^\[([AB])\]\s*(.*)$',line)
  tag=f'Run {tagged[1]} · ' if tagged else ''
  if tagged:line=tagged[2]
  for marker in ('PHASE ','ANALYZE shard','PREPARE','BATCH PROBE:','RUN DONE','TRANSCRIBE ','WINNER (dev','DECODE CHOICE','TEST vs','FINAL ASR:','ASR V16 DONE:'):
   if line.startswith(marker):add(tag+line[:200])
  if line.startswith('INFERENCE_BENCHMARK:'):
   try:
    d=ast.literal_eval(line.split(':',1)[1].strip());add(f"Benchmark {d['inference_profile']} · {d['sample_count']} mẫu · điểm 30s {d['agent_score_30s']:.2f} · p50 {d['latency_p50_seconds']:.1f}s · p95 {d['latency_p95_seconds']:.1f}s")
   except (ValueError,SyntaxError,KeyError):pass
  if line.startswith('V16_PREFLIGHT:'):add('V16 · parent125 và split validation đã xác minh')
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
     if k in d:add(f"{tag}{k}: {float(d[k]):.6f}" + (f" · epoch {d['epoch']}" if tag and 'epoch' in d else ''))
   except (ValueError,SyntaxError):pass
  if 'Saving model checkpoint' in line:
   m=re.search(r'checkpoint-(\d+)',line)
   if m:add(f'Lưu checkpoint · bước {m[1]}')
  if line.startswith('LONG RUN COMPLETE:'):add('Huấn luyện Qwen hoàn tất · chuyển sang đánh giá / tạo kết quả')
  if 'OutOfMemoryError:' in line:add('CUDA hết VRAM','error')
  elif line.startswith('Traceback (most recent call last):'):add('Notebook ghi nhận lỗi · mở Kaggle để xem traceback','error')
 return rows[-300:]

def peer_asr(text,handle,status):
 """Headline numbers of another person's ASR notebook, read from its own log. Nothing here changes the pinned run."""
 def num(v):
  try:return float(v)
  except (TypeError,ValueError):return None
 out=dict(label='ASR · Minh',kernel=handle,status=status,stage=None,baseDevWer=None,baseTestWer=None,devWer=None,winner=None,testV16=None,testV15=None,final=None,loss={'A':None,'B':None},updatedAt=utc())
 for raw in text.splitlines():
  line=raw.strip();tag=re.match(r'^\[([AB])\]\s*(.*)$',line)
  if tag:line=tag[2]
  if line.startswith('PHASE '):out['stage']=line.split(':',1)[0][6:][:60]
  elif line.startswith('PREPARE {'):
   try:
    d=json.loads(line[8:]);out['baseDevWer']=num(d.get('base_dev_wer'));out['baseTestWer']=num(d.get('base_test_wer'))
   except ValueError:pass
  elif line.startswith('WINNER (dev, strict WER):'):
   m=re.match(r'WINNER \(dev, strict WER\): (\w+) (\{.*\})',line)
   if m:
    try:
     dev=ast.literal_eval(m[2]);out['winner']=m[1];out['devWer']={k:num(v) for k,v in dev.items() if k in ('A','B')}
    except (ValueError,SyntaxError):pass
  elif line.startswith('TEST vs '):
   try:
    d=json.loads(line.split(' ',3)[3]);pick=lambda r:{k:num(r.get(k)) for k in ('wer','cer','exact_match')} if isinstance(r,dict) else None
    out['testV16']=pick(d.get('v16'));out['testV15']=pick(next((v for k,v in d.items() if k!='v16'),None))
   except (ValueError,IndexError):pass
  elif line.startswith('FINAL ASR:'):out['final']=line.split(':',1)[1].strip()[:40]
  elif tag and re.search(r"['\"]loss['\"]\s*:",line):
   try:
    d=ast.literal_eval(line);out['loss'][tag[1]]=num(d.get('loss'))
   except (ValueError,SyntaxError):pass
 if 'ASR V16 DONE' in text or 'ASR V17 DONE' in text:out['stage']='done'
 return out

def peer_train(text,handle,status,label='Train · Minh'):
 """Loss series of another person's training notebook, read only from its own log; the pinned run is never touched."""
 ver=re.search(r'-v(\d+)-',handle)
 out=dict(label=label,kernel=handle,version=f'V{ver[1]}' if ver else 'V16',status=status,phase='preparing',step=0,targetStep=None,hours=0.0,evalEvery=None,train=[],evaluations=[],baseline=None,evalProgress=None,stopped=None,updatedAt=utc())
 points={};evals={};agents={};cur=0
 for raw in text.splitlines():
  line=raw.strip()
  m=re.match(r'^(?:LONG RUN|TRAIN) PROGRESS: global=(\d+),? new=(\d+)/(\d+),? hours=([\d.]+)',line)
  if m:
   cur=int(m[1]);out.update(step=cur,targetStep=cur-int(m[2])+int(m[3]),hours=float(m[4]),phase='training')
  elif label.endswith('Duy') and line.startswith('AGENT_EVAL:'):
   # Duy's notebook logs the held-out probe as AGENT_EVAL {step, diagnostic_loss, ...}; the baseline probe (first line) is the starting loss.
   try:
    d=json.loads(line.split(':',1)[1]);evals[int(d['step'])]=float(d['diagnostic_loss'])
    agents[int(d['step'])]=dict(step=int(d['step']),agentScore=float(d['agent_score_30s']),stateMatch=float(d['state_match_rate']),toolF1=float(d['tool_f1']),diagnosticLoss=float(d['diagnostic_loss']))
   except (ValueError,KeyError,TypeError):pass
  elif line.startswith('{') and re.search(r"['\"]loss['\"]\s*:",line) and cur:
   try:
    d=ast.literal_eval(line);points[cur]=float(d['loss'])
   except (ValueError,SyntaxError,KeyError,TypeError):pass
  elif line.startswith('VAL_LOSS:'):
   try:
    d=json.loads(line.split(':',1)[1]);evals[int(d['step'])]=float(d['val_loss'])
   except (ValueError,KeyError,TypeError):pass
  elif line.startswith('EARLY STOP') or line.startswith('LONG RUN COMPLETE') or line.startswith('TRAIN DONE'):
   out.update(phase='finished',stopped=line[:100])
  else:
   t=re.match(r'^\[([\w.-]{1,40})\]\s*EVAL_SAMPLE (\{.*\})',line)
   if t:
    try:
     d=json.loads(t[2]);out['evalProgress']=dict(name=t[1],n=int(d['n']),total=int(d['total']),avg=float(d['avg']))
    except (ValueError,KeyError,TypeError):pass
 if 'PHASE train' in text and out['phase']=='training':out['phase']='finished'
 out['train']=[dict(step=s,loss=l) for s,l in sorted(points.items())][-600:]
 ev=sorted(evals.items())
 out['evaluations']=[dict(step=s,loss=l) for s,l in ev][-100:]
 if len(ev)>1:out['evalEvery']=max(1,min(b[0]-a[0] for a,b in zip(ev,ev[1:])))
 if agents:
  rows=[];best=None
  for s,a in sorted(agents.items()):
   rows.append(dict(a,improved=best is not None and a['agentScore']>best))
   best=a['agentScore'] if best is None else max(best,a['agentScore'])
  out['agent']=rows[-100:]
 return out

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
  side={}
  with ThreadPoolExecutor(max_workers=2) as pool:
   jobs={} if asr else {h:pool.submit(stream,h) for h in (SIDE,SIDE_TRAIN,DUY_TRAIN,DUY_EVAL)}
   text=stream(handle)
   for h,job in jobs.items():
    try:side[h]=job.result(timeout=60)
    except Exception as error:print(f'Side log skipped: {type(error).__name__}',flush=True)
  same=previous['config']['kernel']==handle
  state=previous if same else dict(schema=2,config=c,updatedAt=utc(),source='kaggle',status='unknown',phase='preparing',step=0,hours=0,baseline=None,train=[],evaluations=[],references=[],history=previous.get('history',[]),events=previous.get('events',[]),asrLogs=previous.get('asrLogs',[]),agentEvaluations=previous.get('agentEvaluations',[]),**({'asr':previous['asr']} if 'asr' in previous else {}))
  state['config']=c
  state=parse(state,text,asr)
  if side.get(SIDE):
   state['events']=events(side[SIDE],'ASR',state['events'])
   try:
    st=cmd('kernels','status',SIDE)
    state['peerAsr']=peer_asr(side[SIDE],SIDE,'complete' if 'COMPLETE' in st else 'error' if ('ERROR' in st or 'CANCEL' in st) else 'running' if ('RUNNING' in st or 'QUEUED' in st) else 'unknown')
   except Exception as error:print(f'Peer ASR skipped: {type(error).__name__}',flush=True)
  if side.get(SIDE_TRAIN):
   # Rebuilt from the full log on every sync (the main events() call above dropped the previous Minh rows with the Qwen rows).
   state['events']+=[dict(r,stage='Qwen',message=('Minh · '+r['message'])[:240]) for r in events(side[SIDE_TRAIN],'Minh',[])[-40:]]
   state['events']=state['events'][-300:]
   try:
    from kaggle.api.kaggle_api_extended import KaggleApi as _Api
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest as _Req
    _api=_Api();_api.authenticate()
    with _api.build_kaggle_client() as _client:
     _r=_Req();_r.user_name,_r.kernel_slug=SIDE_TRAIN.split('/')
     number=_client.kernels.kernels_api_client.get_kernel(_r).metadata.current_version_number
    _st=cmd('kernels','status',SIDE_TRAIN)
    state['peers']=[peer_train(side[SIDE_TRAIN],f'{SIDE_TRAIN}/{number}','complete' if 'COMPLETE' in _st else 'error' if ('ERROR' in _st or 'CANCEL' in _st) else 'running' if ('RUNNING' in _st or 'QUEUED' in _st) else 'unknown')]
   except Exception as error:print(f'Peer train skipped: {type(error).__name__}',flush=True)
  if side.get(DUY_TRAIN):
   # Duy's run: its own panel only (no Minh-style event rows). Kept after Minh's panel; a failure here never touches the pinned run.
   try:
    from kaggle.api.kaggle_api_extended import KaggleApi as _ApiD
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest as _ReqD
    _apiD=_ApiD();_apiD.authenticate()
    with _apiD.build_kaggle_client() as _clientD:
     _rd=_ReqD();_rd.user_name,_rd.kernel_slug=DUY_TRAIN.split('/')
     numberD=_clientD.kernels.kernels_api_client.get_kernel(_rd).metadata.current_version_number
    _sd=cmd('kernels','status',DUY_TRAIN)
    duy=peer_train(side[DUY_TRAIN],f'{DUY_TRAIN}/{numberD}','complete' if 'COMPLETE' in _sd else 'error' if ('ERROR' in _sd or 'CANCEL' in _sd) else 'running' if ('RUNNING' in _sd or 'QUEUED' in _sd) else 'unknown','Train · Duy')
    try:
     # Duy's eval notebook: last EVAL_SAMPLE line = live 307 progress (score avg so far); only shown while that run is live or just finished.
     _rows={}
     for _l in (side.get(DUY_EVAL) or '').splitlines():
      _m=re.match(r'^(?:\[[\w.-]{1,40}\]\s*)?EVAL_SAMPLE (\{.*\})',_l.strip())
      if _m:
       _d=json.loads(_m[1]);_rows[int(_d['n'])]=_d
     if _rows:
      _se=cmd('kernels','status',DUY_EVAL)
      _all=[_rows[k] for k in sorted(_rows)];_last=_all[-1];_tot=int(_last['total']);_secs=[float(r['sec']) for r in _all];_mean=sum(_secs)/len(_secs)
      _name='307 · Duy' if _tot<=400 else 'public · Duy'
      duy['evalProgress']=dict(name=_name,n=int(_last['n']),total=_tot,avg=float(_last['avg']))
      duy['evalLive']=dict(name=_name,status='complete' if 'COMPLETE' in _se else 'error' if ('ERROR' in _se or 'CANCEL' in _se) else 'running',n=len(_all),total=_tot,avg=float(_last['avg']),stateOk=sum(1 for r in _all if r.get('state') is True),within20=sum(1 for s in _secs if s<=20),over30=sum(1 for s in _secs if s>=30),meanSec=_mean,etaSeconds=max(0,(_tot-len(_all))*_mean),recent=[dict(n=int(r['n']),id=str(r['id'])[:40],sec=float(r['sec']),score=float(r['score']),calls=len(r.get('pred') or [])) for r in _all[-700:]])
     # Public 634 phase of the same notebook (INFER_SAMPLE lines): no gold, so only latency/errors; the finished 307 result stays attached as final307.
     _pub={}
     for _l in (side.get(DUY_EVAL) or '').splitlines():
      _m=re.match(r'^INFER_SAMPLE (\{.*\})',_l.strip())
      if _m:
       _d=json.loads(_m[1]);_pub[int(_d['n'])]=_d
     if _pub:
      _se=cmd('kernels','status',DUY_EVAL)
      _pa=[_pub[k] for k in sorted(_pub)];_pt=int(_pa[-1]['total']);_ps=[float(r['sec']) for r in _pa];_pm=sum(_ps)/len(_ps)
      _prev=duy.get('evalLive')
      _f307=None
      if _prev and _prev.get('total',0)<=400:_f307={k:v for k,v in _prev.items() if k!='recent'}
      duy['evalLive']=dict(name='public · Duy',kind='public',status='complete' if 'COMPLETE' in _se else 'error' if ('ERROR' in _se or 'CANCEL' in _se) else 'running',n=len(_pa),total=_pt,avg=(_f307['avg'] if _f307 else 0.0),stateOk=sum(1 for r in _pa if r.get('stopped') is True),within20=sum(1 for s in _ps if s<=20),over30=sum(1 for r in _pa if r.get('over30') or float(r['sec'])>=30),meanSec=_pm,lastN=int(_pa[-1]['n']),etaSeconds=max(0,(_pt-int(_pa[-1]['n']))*_pm),recent=[dict(n=int(r['n']),id=str(r['id'])[:40],sec=float(r['sec']),score=-1.0 if not r.get('err') else -2.0,calls=len(r.get('pred') or [])) for r in _pa[-700:]],**({'final307':_f307} if _f307 else {}))
      duy['evalProgress']=dict(name='public · Duy',n=len(_pa),total=_pt,avg=_pm)
    except Exception as error:print(f'Duy eval progress skipped: {type(error).__name__}',flush=True)
    state['peers']=[p for p in state.get('peers',[]) if not p['label'].endswith('Duy')]+[duy]
    state['events']+=[dict(r,stage='Qwen',message=('Duy · '+r['message'])[:240]) for r in events(side[DUY_TRAIN],'Duy',[])[-40:]]
    try:
     # Duy's eval notebook in the Log notebook feed: milestones only (every 25th sample) plus every sample that hit the 30 s deadline.
     _evx=[];_seenx=set()
     for _l in (side.get(DUY_EVAL) or '').splitlines():
      _s=_l.strip()
      _m=re.match(r'^(?:\[[\w.-]{1,40}\]\s*)?EVAL_SAMPLE (\{.*\})',_s)
      if _m:
       _d=json.loads(_m[1]);_n=int(_d['n'])
       if ('c',_n) in _seenx:continue
       _seenx.add(('c',_n))
       if _n%25==0 or _n==int(_d['total']):_evx.append({'stage':'Qwen','level':'info','message':f"Duy · Chấm 307 · mẫu {_n}/{_d['total']} · điểm tạm {float(_d['avg']):.2f}"})
       continue
      _m=re.match(r'^INFER_SAMPLE (\{.*\})',_s)
      if _m:
       _d=json.loads(_m[1]);_n=int(_d['n'])
       if ('p',_n) in _seenx:continue
       _seenx.add(('p',_n))
       if _d.get('err') or _d.get('over30'):_evx.append({'stage':'Qwen','level':'error','message':f"Duy · Public · #{_n} {_d['id']} quá 30 giây ({float(_d['sec']):.1f}s)"})
       elif _n%25==0:_evx.append({'stage':'Qwen','level':'info','message':f"Duy · Public · mẫu {_n}/{_d['total']} · {float(_d['sec']):.1f} giây"})
     state['events']+=_evx[-40:]
    except Exception as error:print(f'Duy eval events skipped: {type(error).__name__}',flush=True)
    state['events']=state['events'][-300:]
   except Exception as error:print(f'Peer train (Duy) skipped: {type(error).__name__}',flush=True)
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



