"""Launch independent fixed-weight extraction shards without changing sources."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--scratch',type=Path,required=True)
    parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--gpus',default='0,1,2,3')
    args=parser.parse_args()
    args.scratch.mkdir(parents=True,exist_ok=False)
    records=[]
    gpus=args.gpus.split(',')
    for index,gpu in enumerate(gpus):
        command=[sys.executable,str(Path(__file__).with_name('extract.py')),
                 '--runs',str(args.runs),'--output',str(args.scratch/'parts'),
                 '--shards',str(len(gpus)),'--shard',str(index)]
        with (args.scratch/f'shard{index}.log').open('w') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                                   env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu},start_new_session=True)
        records.append({'pid':child.pid,'gpu':gpu,'command':command})
    (args.scratch/'launch.json').write_text(json.dumps({'started':time.time(),'jobs':records},indent=2)+'\n')
    print(json.dumps(records))
