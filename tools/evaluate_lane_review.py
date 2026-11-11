#!/usr/bin/env python3
"""Sparse assistant-review benchmark. Does not certify unlabelled video frames."""
import argparse,json
from pathlib import Path
import numpy as np

def compare(ref,pred):
 p=np.asarray(pred['points']);r=np.asarray(ref['points']);p=p[np.argsort(p[:,1])];r=r[np.argsort(r[:,1])]
 ys=np.arange(r[0,1],r[-1,1]+.1,2);inside=(ys>=p[0,1])&(ys<=p[-1,1])
 if inside.mean()<.7:return None
 return np.abs(np.interp(ys[inside],r[:,1],r[:,0])-np.interp(ys[inside],p[:,1],p[:,0]))

def evaluate(labels,predictions,split):
 lookup={(r['sequence'],r['frame']):r for r in predictions};result={};details=[]
 for seq in sorted({r['sequence'] for r in labels['records']}):
  tp=fp=fn=style_correct=style_outputs=style_total=0;errors=[]
  for r in labels['records']:
   if r['sequence']!=seq or (split!='all' and r['split']!=split):continue
   output=lookup.get((seq,r['frame']),{'lanes':[]});lanes=output['lanes'];ignore=r.get('ignore',[])
   for side in ['left','right']:
    if side in ignore:continue
    gt=next((l for l in r['lanes'] if l['side']==side),None);pr=next((l for l in lanes if l['side']==side),None);err=None
    if gt is not None:style_total+=gt['style']!='unknown'
    if pr is not None and gt is not None:err=compare(gt,pr)
    match=err is not None and np.median(err)<=.012*r['width']
    tp+=match;fp+=(pr is not None and not match);fn+=(gt is not None and not match)
    if match:
     errors.extend((err/r['width']).tolist())
     if pr.get('style','unknown')!='unknown':style_outputs+=1;style_correct+=pr['style']==gt['style']
    elif pr is not None and pr.get('style','unknown')!='unknown':style_outputs+=1
    details.append(dict(sequence=seq,frame=r['frame'],side=side,match=bool(match),case_pass=bool(match or (gt is None and pr is None)),reference=None if gt is None else gt['style'],prediction=None if pr is None else pr.get('style'),median_error_px=None if err is None else float(np.median(err))))
  precision=tp/max(1,tp+fp);recall=tp/max(1,tp+fn);sty_prec=style_correct/max(1,style_outputs);coverage=style_correct/max(1,style_total);med=float(np.median(errors)) if errors else 1.;p90=float(np.quantile(errors,.9)) if errors else 1.
  result[seq]=dict(tp=int(tp),fp=int(fp),fn=int(fn),precision=precision,recall=recall,median_relative_error=med,p90_relative_error=p90,style_precision=sty_prec,correct_style_coverage=coverage,geometry_pass=precision>=.95 and recall>=.9 and med<=.006 and p90<=.015,style_pass=sty_prec>=.95 and coverage>=.7)
 return dict(label_version=labels['version'],split=split,sequences=result,details=details,note='Assistant-authored sparse review labels; not official or independent ground truth. All-frame correctness is not measured.')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--labels',required=True);p.add_argument('--predictions',required=True);p.add_argument('--output',required=True);p.add_argument('--require-all-pass',action='store_true');p.add_argument('--split',default='validation',choices=['validation','tuning','all']);a=p.parse_args()
 out=evaluate(json.loads(Path(a.labels).read_text()),json.loads(Path(a.predictions).read_text()),a.split);Path(a.output).write_text(json.dumps(out,indent=2,default=lambda v:v.item()));print(json.dumps(out['sequences'],indent=2,default=lambda v:v.item()))

 if a.require_all_pass and not all(v['geometry_pass'] and v['style_pass'] for v in out['sequences'].values()):
  raise SystemExit('Accuracy gates not met; full metrics were saved to '+a.output)
