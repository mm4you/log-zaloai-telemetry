import re
from datetime import datetime,timezone

def utc():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

def parse_line(state, line):
    changed = False
    gpu = re.search(r'GPU count: (\d+).*device: (.+)', line)
    if gpu:
        state['config']['gpu'] = {'count': int(gpu[1]), 'name': gpu[2].strip()}
    m = re.search(r'LONG RUN PROGRESS: global=(\d+).*hours=([\d.]+)', line)
    if m:
        state.update(step=int(m[1]), hours=float(m[2]), status='running', phase='training')
        every = state['config']['evalEvery']
        if every and (state['step'] - state['config']['evalOrigin']) % every == 0 and state['step'] > state['config']['parentStep']:
            state['phase'] = 'evaluating'
        changed = True
    m = re.search(r"'baseline_loss':\s*'?([\d.eE+-]+)", line)
    if m:
        state['baseline'] = float(m[1])
    for key, pattern in [('train', r"'loss':\s*'?([\d.eE+-]+)"), ('evaluations', r"'eval_loss':\s*'?([\d.eE+-]+)")]:
        m = re.search(pattern, line)
        if m:
            row = {'step': state['step'], 'loss': float(m[1])}
            if key == 'train':
                for field, pattern in [('learningRate', r"'learning_rate':\s*'?([\d.eE+-]+)"), ('gradNorm', r"'grad_norm':\s*'?([\d.eE+-]+)")]:
                    extra = re.search(pattern, line)
                    if extra: row[field] = float(extra[1])
            if row['step'] > state['config']['parentStep']:
                rows = {r['step']: r for r in state[key]}
                rows[row['step']] = row
                state[key] = sorted(rows.values(), key=lambda r: r['step'])[-(1000 if key == 'evaluations' else 2000):]
                if key == 'evaluations': state['phase'] = 'training'
                changed = True
    if 'LONG RUN COMPLETE:' in line:
        state.update(status='complete', phase='finished')
        changed = True
    if 'Traceback' in line or 'OutOfMemoryError' in line:
        state['status'] = 'error'
    return changed
