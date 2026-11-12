"""Sheet of preview frames with confirmed markings (solid) and candidates (thin)."""
import sys,gzip,json,cv2,numpy as np
from pathlib import Path
sys.path.insert(0,'tools');sys.path.insert(0,'src/vision_road')
import maploop_offline_compare as M
run,cache,out=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3];secs=[float(x) for x in sys.argv[4:]]
pk=Path('src/vision_road');cfg=M.settings(pk);proj=M.projector(pk,cfg)
rows=[json.loads(l) for l in gzip.open(run/'predictions.jsonl.gz','rt')];t0=rows[0]['stamp']
COL=dict(stop_line=(0,0,255),crosswalk=(255,0,255),marking=(0,200,255))
tiles=[]
for sec in secs:
    i=min(range(len(rows)),key=lambda k:abs(rows[k]['stamp']-t0-sec));r=rows[i]
    img=cv2.imread(str(cache/'preview'/f'{i:05d}.png'))
    def poly(pts,color,th,closed):
        pts=np.asarray(pts,float)
        if closed:pts=np.vstack([pts,pts[:1]])
        # densify so the projection follows the ground
        d=[];[d.extend(np.linspace(a,b,10)) for a,b in zip(pts[:-1],pts[1:])];d=np.array(d)
        px,vis=proj.image_points(d)
        if vis.sum()<2:return
        cv2.polylines(img,[np.rint(px[vis]*.5).astype(np.int32)],False,color,th)
    for m in r['marking_candidates']:
        pts=[[m['x'],m['left']],[m['x'],m['right']]] if m['kind']=='stop_line' else [[m['x0'],m['y0']],[m['x0'],m['y1']],[m['x1'],m['y1']],[m['x1'],m['y0']]]
        poly(pts,COL[m['kind']],1,m['kind']!='stop_line')
    for m in r['markings']:poly(m['points'],COL[m['kind']],4,m['kind']!='stop_line')
    cv2.putText(img,'%.1fs #%d %s'%(sec,i,','.join(m['kind'][0] for m in r['markings'])),(8,24),0,.7,(255,255,255),2)
    tiles.append(img)
while len(tiles)%3:tiles.append(np.zeros_like(tiles[0]))
sheet=np.vstack([np.hstack(tiles[k:k+3]) for k in range(0,len(tiles),3)])
cv2.imwrite(out,cv2.resize(sheet,None,fx=.6,fy=.6),[cv2.IMWRITE_JPEG_QUALITY,85])
