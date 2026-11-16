"""Per-pixel colour gate of carla.py binary() on a BEV row band around a lateral position.
usage: yellow_probe.py BEV.png X_M Y_FROM Y_TO   -> prints per column: BGR, HLS, g-b, white?, yellow?"""
import sys,cv2,numpy as np
MPP=.07;FMAX=35.;LMAX=6.
bev=cv2.imread(sys.argv[1]);x=float(sys.argv[2]);y0,y1=float(sys.argv[3]),float(sys.argv[4])
r=round((FMAX-x)/MPP);band=bev[r-3:r+4].astype(np.float32).mean(0)
hls=cv2.cvtColor(bev,cv2.COLOR_BGR2HLS)[r-3:r+4].astype(np.float32).mean(0)
for y in np.arange(y0,y1-1e-6,-MPP):
    c=round((LMAX-y)/MPP);B,G,R=band[c];h,l,s=hls[c]
    yel=(16<=h<=42)&(s>20)&((G-B)>18)&((G-B)>.09*G)&((G-B)>1.25*(R-G))
    print(f'y {y:5.2f} BGR {B:5.0f}{G:5.0f}{R:5.0f}  H {h:4.0f} L {l:4.0f} S {s:4.0f} g-b {G-B:5.1f} r-g {R-G:5.1f} white {bool(s<120 and l>0)} yellow {bool(yel)}')
