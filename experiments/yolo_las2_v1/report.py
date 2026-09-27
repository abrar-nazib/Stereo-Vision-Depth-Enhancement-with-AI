"""Generate the reviewable comparison from saved measurements, never constants."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True)
    args=p.parse_args()
    load=lambda name:json.loads((args.run/name).read_text())
    config=load('config.json');result=load('results.json')
    baseline=load('baseline_original_las2s.json')
    speed=load('baseline_performance.json')
    rows=[]
    for name in ('lightstereo','las1','las2s','las2m'):
        path=args.audit/f'{name}.json'
        if path.exists():
            d=json.loads(path.read_text());rows.append((name+' FP32',d,d['latency_ms']))
    rows.extend([('Original LAS2-S FP16',baseline,speed['mean_ms']),
                 ('Hybrid before training FP16',result['initial'],None),
                 ('Hybrid after 500 updates FP16',result['final'],result['performance']['mean_ms'])])
    text=['# A02 native v1.0: 100-pair Driving capacity study','',
          'Full 540x960 images, padding only. Errors are in native-image pixels. '
          'Same 100 stereo pairs for training and evaluation; this measures fitting capacity, not generalization.',
          '', '| Model | EPE | RMSE | bad-0.5 % | bad-1 % | bad-2 % | bad-3 % | D1 % | Mean ms |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,d,ms in rows:
        values=' | '.join(f'{d[k]:.3f}' for k in ('epe','rmse','bad_0.5','bad_1','bad_2','bad_3','d1'))
        text.append(f'| {name} | {values} | '+(f'{ms:.2f}' if ms else 'not measured')+' |')
    delta=result['final']['epe']-baseline['epe']
    text += ['',f'Hybrid minus original LAS2-S FP16 EPE: {delta:+.3f} px (negative is better).',
             '', '## Interpretation', '',
             'The trained hybrid has seen these 100 pairs; the original checkpoints were not fine-tuned in this run. '
             'This is not evidence of generalization superiority. Original checkpoint training may also include Driving.',
             'Five passes are a short feasibility budget, not a converged ablation. '
             'The original model timing produces disparity only; hybrid timing also produces full-resolution semantic labels. '
             'FP32 audit latencies and FP16 training-run latencies are separate comparisons.',
             '', '## Configuration and verification', '',
             f'- Architecture: {config["architecture"]}; shared frozen YOLO26s-sem, four trainable adapters, retained LAS2-S FPN/head.',
             f'- Parameters: {config["parameters_total"]:,} total, {config["parameters_trainable"]:,} trainable.',
             f'- GPU: {config["gpu"]}; batch 1; AdamW LR 1e-4; 500 updates; seed 42.',
             f'- Training time: {result["training"]["seconds"]:.2f} s; peak allocated GPU memory: {result["training"]["peak_allocated_mb"]:.2f} MiB.',
             f'- Frozen semantic weights and buffers unchanged: {result["semantic_weights_and_buffers_unchanged"]}.',
             '- Separate forward equivalence check against original YOLO semantic output passed at 128x256 (FP32, max absolute difference 6.68e-6).',
             '- No semantic ground truth or mIoU measurement. Class predictions are semantic categories, not separate object instances.',
             '- Metric mask: finite GT, 0 < d < 192; D1 requires >3 px AND >5%; macro-average across images.',
             '- Exact pairs and checkpoint/source fingerprints: manifest.json and config.json.',
             '- A01 resize results are superseded; they must not be used as the native benchmark.',
             '', '![Training loss](training_curve.png)', '']
    (args.run/'REPORT.md').write_text('\n'.join(text))
    with (args.run/'train.csv').open() as f:data=list(csv.DictReader(f))
    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot([int(x['step']) for x in data],[float(x['loss']) for x in data],alpha=.5,label='Per-step loss')
    ax.set(xlabel='Optimizer update',ylabel='Training loss',title='A02 native: YOLO26s-sem + LAS2-S (100-pair fit)')
    ax.grid(alpha=.2);fig.tight_layout();fig.savefig(args.run/'training_curve.png',dpi=150);plt.close(fig)
    # Fingerprint the independently evaluated baseline artifacts as well.
    paths=list(args.audit.glob('*.json'))+list(args.audit.glob('*.csv'))
    (args.run/'audit_artifacts.json').write_text(json.dumps({str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths},indent=2)+'\n')
    print(args.run/'REPORT.md')


if __name__=='__main__':main()
