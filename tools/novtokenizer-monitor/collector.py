#!/usr/bin/env python3
"""One read-only SSH snapshot. No AI APIs, no background daemon."""
import argparse
import fcntl
import json
import shlex
import subprocess
import time
from pathlib import Path

REMOTE = r'''
import json,time,subprocess,os
from pathlib import Path
cfg=json.loads(CONFIG)
root=Path(cfg['root']); ev=Path(cfg['evaluation_root'])/'results'
def read(p):
 try: return json.loads(p.read_text())
 except (OSError,ValueError): return {}
a=root/'results/cuda_same_source/runs/seed1/j-reversible-sp-unigram-64k/artifacts'
p=read(a/'progress.json');m=read(a/'metrics.json');launch=read(root/'results/phase551_controller/launch.json')
train=m.get('training',p);chars=train.get('source_chars',0);wall=train.get('train_wall_seconds',0)
now=time.time();start=launch.get('started_unix');done=read(a.parent/'done.json');end=done.get('finished_unix',now);elapsed=end-start if start else None
try:
 out=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu','--format=csv,noheader,nounits'],text=True,timeout=5)
 gpus=[]
 for line in out.strip().splitlines():
  fields=[s.strip() for s in line.split(',')]
  def number(s):
   try:return float(s)
   except ValueError:return None
  gpus.append(dict(zip(['index','name','utilization','memory_used','memory_total','power','temperature'],[fields[0],fields[1]]+[number(x) for x in fields[2:]])))
except (OSError,subprocess.SubprocessError):gpus=[]
complete=m.get('complete',False);budget=launch.get('source_chars',30000000)
alive=None
if launch.get('pid'):
 try:os.kill(launch['pid'],0);alive=True
 except ProcessLookupError:alive=False
 except PermissionError:alive=None
results=read(ev/'comparison.json')
print(json.dumps({'observed_at':now,'gpus':gpus,'training':{'chars':chars,'budget':budget,'tokens':train.get('tokens'), 'steps':train.get('steps'),'chars_sec':chars/wall if wall else None,'tokens_sec':train.get('tokens',0)/wall if wall else None,'elapsed':elapsed,'eta':elapsed*(budget-chars)/chars if elapsed and chars and not complete else None,'complete':complete,'process_alive':alive,'progress_age':now-(a/'progress.json').stat().st_mtime if (a/'progress.json').exists() else None,'loss':train.get('last_train_loss'),'bpb':m.get('validation',{}).get('overall',{}).get('bits_per_byte')},'evaluation':{'complete':bool(read(ev/'done.json')),'finished':len(list(ev.glob('old-seed*.json')))+len(list(ev.glob('hardened-seed*.json'))),'total':8,'rows':results if isinstance(results,list) else []}}))
'''

def collect(config):
    host=config['host']
    if host.startswith('-') or any(c.isspace() for c in host):
        raise ValueError('invalid SSH host alias')
    script=REMOTE.replace('CONFIG',repr(json.dumps(config)))
    try:
        result=subprocess.run(['/usr/bin/ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','-o','StrictHostKeyChecking=yes','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=1',host,'python3 -c '+shlex.quote(script)],capture_output=True,text=True,timeout=18)
        if result.returncode:
            error=result.stderr.lower()
            auth=any(s in error for s in ['permission denied','authenticate','authentication','tailscale','host key verification failed'])
            return {'ok':False,'status':'認証・ホスト鍵の確認待ち' if auth else 'SSH接続失敗','observed_at':time.time()}
        data=json.loads(result.stdout)
        data.update(ok=True,status='接続中')
        return data
    except subprocess.TimeoutExpired:
        return {'ok':False,'status':'接続タイムアウト','observed_at':time.time()}
    except (OSError,ValueError):
        return {'ok':False,'status':'設定または応答エラー','observed_at':time.time()}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once',action='store_true',required=True)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    try:
        lock = args.config.with_suffix('.lock').open('a')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result=collect(json.loads(args.config.read_text()))
        except BlockingIOError:
            result={'ok':False,'status':'前の取得処理が継続中','observed_at':time.time()}
    except (OSError,ValueError,KeyError):result={'ok':False,'status':'設定ファイルを確認してください','observed_at':time.time()}
    print(json.dumps(result,ensure_ascii=False))
    raise SystemExit(0 if result['ok'] else 1)
