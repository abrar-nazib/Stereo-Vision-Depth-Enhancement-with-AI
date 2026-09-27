"""A02 native-image baseline audit. Never resize images or ground truth."""
import argparse
import csv
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from model import ROOT, load_stereo
from run import pfm, metrics, write_json
from core.liteanystereo import LiteAnyStereo
from core.liteanystereov2 import LiteAnyStereoV2
from core.utils.utils import InputPadder


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--model',choices=['las1','las2s','las2m','lightstereo'],required=True)
    args=p.parse_args()
    torch.set_num_threads(4)
    records=json.loads((args.run/'manifest.json').read_text())
    out=args.run/'native_audit_v2';out.mkdir(exist_ok=True)
    if args.model=='las1':
        model=LiteAnyStereo(fnet_pretrained=False)
        checkpoint=ROOT/'models/stereo/liteanystereo/LiteAnyStereo.pth'
    elif args.model.startswith('las2'):
        size=args.model[-1]
        model=LiteAnyStereoV2(model_size=size,fnet_pretrained=False)
        checkpoint=ROOT/f'models/stereo/liteanystereo/LAS2_{size.upper()}.pth'
    else:
        from lightstereo_adapter import build_lightstereo
        model=build_lightstereo()
        checkpoint=ROOT/'models/stereo/lightstereo/LightStereo-S-SceneFlow.ckpt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if 'state_dict' in state:state=state['state_dict']
    if 'model_state' in state:state=state['model_state']
    state={k.removeprefix('module.'):v for k,v in state.items()}
    model.load_state_dict(state,strict=True)
    model.cuda().eval()
    def predict(l,r):
        if args.model=='lightstereo':
            mean=l.new_tensor([.485,.456,.406])[None,:,None,None]
            std=l.new_tensor([.229,.224,.225])[None,:,None,None]
            return model({'left':(l/255-mean)/std,'right':(r/255-mean)/std})['disp_pred']
        return model(l,r,test_mode=True)
    rows=[]
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for i,record in enumerate(records):
            images=[cv2.cvtColor(cv2.imread(record[k]),cv2.COLOR_BGR2RGB) for k in ('left','right')]
            l,r=[torch.from_numpy(x).permute(2,0,1)[None].float().cuda() for x in images]
            gt=torch.from_numpy(pfm(record['disparity']))[None,None]
            padder=InputPadder(l.shape,divis_by=32)
            if args.model=='lightstereo':
                # Official LightStereo evaluation pads top/right.
                padder._pad=[0,(-l.shape[-1])%32,(-l.shape[-2])%32,0]
            l,r=padder.pad(l,r)
            # Match official FP32 evaluation before considering AMP speed.
            if i==0:
                for _ in range(10):predict(l,r)
            torch.cuda.synchronize();start=time.perf_counter()
            pred=predict(l,r)
            torch.cuda.synchronize();latency=(time.perf_counter()-start)*1000
            pred=padder.unpad(pred).cpu()
            assert pred.shape==gt.shape
            row=dict(index=i,height=gt.shape[-2],width=gt.shape[-1],latency_ms=latency,**metrics(pred,gt))
            row['valid_fraction']=row['valid_pixels']/gt.numel()
            rows.append(row)
            if (i+1)%20==0:print(args.model,i+1,'EPE',np.mean([x['epe'] for x in rows]),flush=True)
    with (out/f'{args.model}_per_image.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    summary={k:float(np.mean([x[k] for x in rows])) for k in rows[0] if k!='index'}
    summary.update(model=args.model,checkpoint=str(checkpoint),precision='FP32',resize=False,padding='replicate symmetric to multiple of 32, removed before scoring',
                   metric_mask='finite GT, 0 < disparity < 192 native pixels',gpu=torch.cuda.get_device_name(),
                   peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,parameters=sum(x.numel() for x in model.parameters()),
                   fps=1000/summary['latency_ms'],median_latency_ms=float(np.median([x['latency_ms'] for x in rows])),
                   p95_latency_ms=float(np.percentile([x['latency_ms'] for x in rows],95)))
    if args.model=='lightstereo':summary['padding']='replicate top/right to multiple of 32, removed before scoring'
    write_json(out/f'{args.model}.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
