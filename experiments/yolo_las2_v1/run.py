"""Versioned 100-pair Driving capacity study; native resolution by default."""
import argparse
import csv
import hashlib
import json
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from model import JointModel, load_stereo, ROOT
from core.utils.utils import InputPadder


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def pfm(path):
    with open(path, 'rb') as f:
        assert f.readline().strip() == b'Pf'
        w, h = map(int, f.readline().split())
        scale = float(f.readline())
        data = np.fromfile(f, '<f4' if scale < 0 else '>f4').reshape(h, w)
    return np.flipud(data).copy() * abs(scale)


def metrics(pred, gt):
    valid = torch.isfinite(gt) & (gt > 0) & (gt < 192)
    err = (pred.float() - gt).abs()[valid]
    truth = gt[valid]
    result = dict(epe=err.mean().item(), rmse=err.square().mean().sqrt().item(),
                  median=err.median().item(), valid_pixels=err.numel())
    for threshold in (.5, 1, 2, 3):
        result[f'bad_{threshold:g}'] = (err > threshold).float().mean().item() * 100
    result['d1'] = ((err > 3) & (err > .05 * truth)).float().mean().item() * 100
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--height', type=int, default=384)
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--legacy-resize', action='store_true', help='Superseded A01 protocol, explicit opt-in only')
    p.add_argument('--data', type=Path, default=Path('/media/abrar/AbrarSSD/Datasets/sceneflow_driving'))
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    assert torch.cuda.is_available()
    prefix='A01_legacy' if args.legacy_resize else 'A02_native'
    out = ROOT / 'experiments/yolo_las2_v1/runs' / (prefix+datetime.now(timezone.utc).strftime('_%Y%m%dT%H%M%SZ'))
    out.mkdir(parents=True, exist_ok=False)
    print(f'RUN_DIRECTORY={out}', flush=True)
    config = dict(vars(args), data=str(args.data), architecture='A01-v1.0', protocol='100-pair overfit, no held-out evaluation',
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                  batch_size=1, lr=1e-4, optimizer='AdamW', weight_decay=1e-4,
                  loss='smooth_l1(full) + 0.3*smooth_l1(coarse)', max_disparity=192,
                  precision='float16 autocast with GradScaler', resize='RGB bilinear; disparity nearest scaled by width ratio',
                  aggregation='macro average over images', semantic='frozen; no semantic GT, no mIoU claim')
    config['source_sha256'] = {f: hashlib.sha256((Path(__file__).parent/f).read_bytes()).hexdigest() for f in ('model.py','run.py')}
    config['resize']='legacy 384x640' if args.legacy_resize else 'NONE: native full images, symmetric replicate padding to /32'
    config['protocol_version']='A01-legacy' if args.legacy_resize else 'A02-native-v1.0'
    if not args.legacy_resize:
        config['height']=None;config['width']=None
    config['upstream_commit'] = subprocess.check_output(['git','-C',str(ROOT/'external_models/LiteAnyStereo'),'rev-parse','HEAD'], text=True).strip()
    config['checkpoint_sha256'] = {str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in
        [ROOT/'models/stereo/liteanystereo/LAS2_S.pth',ROOT/'models/segmentation/yolo26s-sem-cityscapes.pt']}
    write_json(out/'config.json', config)
    groups = sorted((args.data/'frames_finalpass').glob('*/*/*/left'))
    records = []
    for i, group in enumerate(groups):
        paths = sorted(group.glob('*.png'))
        n = 100//len(groups) + (i < 100%len(groups))
        for index in np.linspace(0, len(paths)-1, n, dtype=int):
            left = paths[index]
            rel = left.relative_to(args.data/'frames_finalpass')
            records.append(dict(left=str(left),right=str(left.parent.parent/'right'/left.name),
                                disparity=str((args.data/'disparity'/rel).with_suffix('.pfm')), sequence=str(group.relative_to(args.data))))
    assert len(records) == 100 and len({r['left'] for r in records}) == 100
    write_json(out/'manifest.json', records)
    samples = []
    for record in records:
        imgs = [cv2.cvtColor(cv2.imread(record[k]),cv2.COLOR_BGR2RGB) for k in ('left','right')]
        width = imgs[0].shape[1]
        tensors = [torch.from_numpy(cv2.resize(x,(args.width,args.height)) if args.legacy_resize else x).permute(2,0,1).float()[None] for x in imgs]
        gt = pfm(record['disparity'])
        if args.legacy_resize:
            gt=cv2.resize(gt,(args.width,args.height),interpolation=cv2.INTER_NEAREST)*args.width/width
        samples.append((*tensors,torch.from_numpy(gt)[None,None]))
    def forward(model,l,r,joint=False,semantic=True):
        padder=InputPadder(l.shape,divis_by=32)
        l,r=padder.pad(l,r)
        if joint:
            full,coarse,logits=model(l,r,semantic=semantic)
            return padder.unpad(full),padder.unpad(coarse),logits
        return padder.unpad(model(l,r,test_mode=True))
    def evaluate(model, name, joint=False):
        model.eval()
        rows=[]
        with torch.inference_mode():
            for i, (l,r,g) in enumerate(samples):
                with torch.autocast('cuda',dtype=torch.float16):
                    output=forward(model,l.cuda(),r.cuda(),joint)
                pred=output[0] if joint else output
                rows.append(dict(index=i,**metrics(pred.cpu(),g)))
        with (out/f'{name}_per_image.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
        summary={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='index'}
        write_json(out/f'{name}.json',summary)
        print(name,summary,flush=True)
        return summary
    baseline=load_stereo().cuda()
    evaluate(baseline,'baseline_original_las2s')
    baseline.eval()
    timings=[]
    l,r,_=[x.cuda() for x in samples[0]]
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        for i in range(60):
            torch.cuda.synchronize();t=time.perf_counter()
            forward(baseline,l,r)
            torch.cuda.synchronize()
            if i>=10:timings.append((time.perf_counter()-t)*1000)
    write_json(out/'baseline_performance.json',dict(mean_ms=float(np.mean(timings)),median_ms=float(np.median(timings)),
               p95_ms=float(np.percentile(timings,95)),fps=1000/np.mean(timings),peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
               scope='disparity only; GPU-resident input'))
    del baseline
    torch.cuda.empty_cache()
    model=JointModel().cuda()
    config['parameters_total']=sum(p.numel() for p in model.parameters())
    config['parameters_trainable']=sum(p.numel() for p in model.parameters() if p.requires_grad)
    write_json(out/'config.json',config)
    frozen_before={k:v.detach().cpu().clone() for k,v in model.semantic.state_dict().items()}
    initial=evaluate(model,'hybrid_initial',True)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda')
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    with (out/'train.csv').open('w',buffering=1) as f:
        writer=csv.DictWriter(f,fieldnames=['step','loss','epe','elapsed_seconds']);writer.writeheader()
        order=[]
        for step in range(1,args.steps+1):
            if not order:
                order=list(range(100));random.shuffle(order)
            l,r,g=[x.cuda() for x in samples[order.pop()]]
            model.train();opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                pred,coarse,_=forward(model,l,r,joint=True,semantic=False)
                mask=torch.isfinite(g)&(g>0)&(g<192)
                loss=F.smooth_l1_loss(pred[mask],g[mask])+.3*F.smooth_l1_loss(coarse[mask],g[mask])
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            scaler.scale(loss).backward();scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
            scaler.step(opt);scaler.update()
            row=dict(step=step,loss=loss.item(),epe=(pred[mask]-g[mask]).abs().mean().item(),elapsed_seconds=time.perf_counter()-start)
            writer.writerow(row)
            if step%50==0: print(row,flush=True)
    training=dict(seconds=time.perf_counter()-start,peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
                  peak_reserved_mb=torch.cuda.max_memory_reserved()/2**20)
    final=evaluate(model,'hybrid_final',True)
    unchanged=all(torch.equal(v.cpu(),frozen_before[k]) for k,v in model.semantic.state_dict().items())
    assert unchanged, 'Frozen semantic model changed'
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),config=config,step=args.steps),out/'final.pth')
    model.eval();l,r,_=[x.cuda() for x in samples[0]]
    torch.cuda.reset_peak_memory_stats()
    timings=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        for i in range(60):
            torch.cuda.synchronize();t=time.perf_counter()
            padder=InputPadder(l.shape,divis_by=32)
            lp,rp=padder.pad(l,r)
            pred,_,logits=model(lp,rp)
            labels=padder.unpad(F.interpolate(logits.float(),size=lp.shape[-2:],mode='bilinear',align_corners=False)).argmax(1)
            torch.cuda.synchronize()
            if i>=10:timings.append((time.perf_counter()-t)*1000)
    performance=dict(mean_ms=float(np.mean(timings)),median_ms=float(np.median(timings)),p95_ms=float(np.percentile(timings,95)),
                     fps=1000/np.mean(timings),peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
                     scope='joint forward plus full-resolution semantic argmax; GPU-resident input; no capture or point cloud')
    write_json(out/'results.json',dict(initial=initial,final=final,training=training,performance=performance,semantic_weights_and_buffers_unchanged=unchanged))
    print('COMPLETE',out,performance,flush=True)


if __name__=='__main__':main()
