"""Bounded cloud worker with an explicit successor; cron is only a backup."""
import os,time
from pathlib import Path
from cloud_sync import main
if __name__=='__main__':
 seconds=max(60,min(3300,int(os.environ.get('WATCH_SECONDS','3300'))))
 end=time.monotonic()+seconds
 while time.monotonic()<end:
  started=time.monotonic()
  try:
   state=main()
   if state and (state['status']=='error' or (state['status']=='complete' and '/lifemate-v15-2-train/' in state['config']['kernel'])):
    marker=Path('.local-state/monitor-terminal');marker.parent.mkdir(exist_ok=True);marker.write_text(state['status']);break
  except Exception as error:
   print(f'Cloud sync retry pending: {type(error).__name__}',flush=True)
  time.sleep(max(1,min(end-time.monotonic(),60-(time.monotonic()-started))))
