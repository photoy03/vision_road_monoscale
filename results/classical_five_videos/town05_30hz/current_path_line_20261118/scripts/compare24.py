#!/usr/bin/env python3
"""Plan 24: side-by-side captures (same script moment, one tile per drawing variant) and the lit-share table.
Writes ../captures/<script>_t<time>_<group>.jpg and ../metrics/lit_summary.json; prints a table.
Groups: styles (none, fill, edges, centre, dots: full length) and lengths (centre 24/10/6 m, edges 24/6 m)."""
import json
from pathlib import Path
import cv2,numpy as np
D=Path(__file__).resolve().parent.parent;H=D/'hud';C=D/'captures';C.mkdir(exist_ok=True)
GROUPS={'styles':['none','fill','edges','centre','dots'],'lengths':['fill','centre','centre_10m','centre_6m','edges','edges_6m'],'dots':['fill','dots','dots_8m','dots_6m','none','centre_10m']}
LABEL={'none':'current path not drawn (style none)','fill':'filled band (plan 23)','edges':'(A) two edge lines','centre':'(B) centre line','dots':'(C) centre dots',
       'centre_10m':'(B) centre line, 10 m','centre_6m':'(B) centre line, 6 m','edges_6m':'(A) edge lines, 6 m','dots_8m':'(C) centre dots, 8 m','dots_6m':'(C) centre dots, 6 m'}
MOMENTS=[('yellow',3.),('wobble',5.),('change',4.),('dashed',4.),('change',7.),('yellow',6.)]
for script,t in MOMENTS:
    for g,vars_ in GROUPS.items():
        tiles=[]
        for v in vars_:
            f=H/v/f'{script}_t{t:04.1f}.jpg'
            if not f.exists():continue
            img=cv2.imread(str(f));img[:42]=0
            cv2.putText(img,LABEL[v],(10,30),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
            cv2.rectangle(img,(0,0),(img.shape[1]-1,img.shape[0]-1),(80,80,80),2);tiles.append(img)
        if not tiles:continue
        cols=3;rows=[tiles[i:i+cols] for i in range(0,len(tiles),cols)]
        rows=[np.hstack(r+[np.zeros_like(tiles[0])]*(cols-len(r))) for r in rows]
        grid=np.vstack(rows);band=np.zeros((50,grid.shape[1],3),np.uint8)
        cv2.putText(band,f'{script}  t={t:.1f}s',(12,36),cv2.FONT_HERSHEY_SIMPLEX,1.1,(255,255,255),2)
        cv2.imwrite(str(C/f'{script}_t{t:04.1f}_{g}.jpg'),cv2.resize(np.vstack([band,grid]),None,fx=.75,fy=.75),[cv2.IMWRITE_JPEG_QUALITY,85])
summary={}
for f in sorted((D/'metrics').glob('lit_*.json')):
    if f.name=='lit_summary.json':continue
    d=json.load(open(f));v=f.stem[4:]
    summary[v]={k:x['lit_mean'] for k,x in d['scripts'].items()}
    summary[v]['mean']=round(float(np.mean([x['lit_mean'] for x in d['scripts'].values()])),4)
    print(f"{v:12s}",'  '.join(f"{k} {100*x:5.2f}%" for k,x in summary[v].items()))
json.dump(summary,open(D/'metrics'/'lit_summary.json','w'),indent=1)
