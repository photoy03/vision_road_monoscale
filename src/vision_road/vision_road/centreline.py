"""Lane centrelines from two boundaries (base_link, x forward, y left), for a path planner to follow.

The middle is taken across the lane, perpendicular to the boundaries, not at equal x: in a curve
the two boundaries at one x are not opposite each other.  With one boundary the centre is that
boundary moved half a lane width across it, marked estimated."""
import math
import numpy as np


def normals(points):
    """Unit left normals of a polyline sorted by x."""
    d=np.gradient(points,axis=0);n=np.linalg.norm(d,axis=1);n[n==0]=1.;t=d/n[:,None]
    return np.column_stack([-t[:,1],t[:,0]])


def between(left,right,step=.1):
    """Centre points and widths between a left and a right boundary (each (n,2), sorted by x).

    For every left boundary point, its foot on the right boundary (nearest point of the right line
    densified to `step`); the centre is the middle of the two, the width their distance.  A foot at
    a right line end that is not perpendicular (the left point lies past that end) is dropped.
    Returns (centres sorted by x, widths), empty when the lines do not face each other."""
    x=np.arange(right[0,0],right[-1,0]+1e-9,step)
    if len(x)<2:return np.empty((0,2)),np.empty(0)
    Q=np.column_stack([x,np.interp(x,right[:,0],right[:,1])]);T=np.gradient(Q,axis=0);T/=np.linalg.norm(T,axis=1)[:,None]
    # The foot lies near the same x (within the width times the heading): a window, not all of Q.
    base=np.clip(np.rint((left[:,0]-x[0])/step).astype(int),0,len(x)-1)
    window=np.clip(base[:,None]+np.arange(-60,61),0,len(x)-1);D=left[:,None,:]-Q[window]
    foot=window[np.arange(len(left)),np.argmin(D[...,0]**2+D[...,1]**2,axis=1)]
    d=left-Q[foot];along=np.abs(d[:,0]*T[foot,0]+d[:,1]*T[foot,1])
    keep=along<=.6*step  # an inner foot is within half a step of perpendicular
    if not keep.any():return np.empty((0,2)),np.empty(0)
    c=(left[keep]+Q[foot[keep]])/2;w=np.hypot(*d[keep].T)
    order=np.argsort(c[:,0],kind='stable');return c[order],w[order]


def shifted(boundary,width,side):
    """Centre from one boundary: half a lane width across it, into the lane.  side: which boundary
    of the lane this is ('left' boundary: the lane lies to its right)."""
    n=normals(boundary);sign=-1. if side=='left' else 1.
    c=boundary+sign*.5*width*n
    return c[np.argsort(c[:,0],kind='stable')]


def joined(left,right,width,step=.1):
    """Centre of a lane with both boundaries: measured (between) where they face each other, and past
    that part, the boundary that goes on moved across by the width measured at that end, so the
    line covers the reach of both.  Returns (centre, measured range (lo, hi) or None, widths): with
    fewer than 2 facing points, the longer boundary moved by `width` (measured range None)."""
    c,w=between(left,right,step)
    if len(c)<2:
        one,side=(left,'left') if np.ptp(left[:,0])>=np.ptp(right[:,0]) else (right,'right')
        return shifted(one,width,side),None,w
    lo,hi=float(c[0,0]),float(c[-1,0]);parts=[c]
    # Each end: the boundary reaching farther past the facing part (one only, so the pieces do not overlap).
    first=min(((left,'left'),(right,'right')),key=lambda b:b[0][0,0])
    if first[0][0,0]<lo-step:
        piece=shifted(first[0],float(w[0]),first[1]);parts.insert(0,piece[piece[:,0]<lo])
    last=max(((left,'left'),(right,'right')),key=lambda b:b[0][-1,0])
    if last[0][-1,0]>hi+step:
        piece=shifted(last[0],float(w[-1]),last[1]);parts.append(piece[piece[:,0]>hi])
    out=np.vstack([q for q in parts if len(q)])
    return out[np.argsort(out[:,0],kind='stable')],(lo,hi),w


def at_reference(centre,ref,span):
    """Lateral offset (y, + left), heading (rad, + to the left) and curvature (1/m, + turning left) of
    the centreline at x = ref, or at its first point when it starts farther: a quadratic in (x - ref)
    through the centre points within span past it (a straight line when they cover under 6 m).  Also the offset at x = 0 (the rear axle) along
    the heading, an extrapolation outside the view.  None when it covers less than 1 m there (a shown boundary is at least that long)."""
    x0=max(float(ref),float(centre[0,0]))
    part=centre[(centre[:,0]>=x0)&(centre[:,0]<=x0+span)]
    if len(part)<2 or part[-1,0]-part[0,0]<1.:return None
    u=part[:,0]-x0;degree=2 if u[-1]>=6. else 1
    coef=np.polyfit(u,part[:,1],degree)
    if degree==1:coef=np.r_[0.,coef]
    q,b,y0=coef  # highest power first
    curvature=2*q/(1+b*b)**1.5
    return dict(ref_x_m=round(x0,3),offset_m=float(y0),heading_rad=float(math.atan(b)),curvature_1pm=float(curvature),
                offset_at_axle_m=float(y0-x0*b))
