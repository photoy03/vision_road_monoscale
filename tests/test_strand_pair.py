"""Double line seen as one pair of rails: far pairs decide which strand the near single paint is."""
from dataclasses import replace
import numpy as np
from vision_road.core import Settings
from vision_road.carla import CarlaDetector

C=Settings(forward_min=3,forward_max=35,lateral_min=-6,lateral_max=6,mpp=.05,min_span_m=.4)
YELLOW=(40,200,220);WHITE=(220,220,220);ROAD=(100,100,100)
K=1/60.;GAP=.27;MID=1.75


def canvas():
    h=round((C.forward_max-C.forward_min)/C.mpp);w=round((C.lateral_max-C.lateral_min)/C.mpp)
    return np.full((h,w,3),ROAD,np.uint8)


def arc(im,offset,start,end,color=YELLOW,width=.12,k=K):
    for x in np.arange(start,end,C.mpp/2):
        y=offset+k/2*x*x;r=round((C.forward_max-x)/C.mpp)
        c0=round((C.lateral_max-y-width/2)/C.mpp);c1=round((C.lateral_max-y+width/2)/C.mpp)
        if 0<=r<im.shape[0]:im[r,max(0,c0):c1+1]=color


def left(im,cfg):
    d=CarlaDetector(cfg);valid=np.full(im.shape[:2],255,np.uint8);d.detect(d.binary(im,valid))
    b={b.side:b for b in d.boundary_candidates}['left'].points
    return d,lambda x:float(np.interp(x,b[:,0],b[:,1]))-K/2*x*x


def scene(near_outer):
    """Both strands from 10 m on; nearer only one of them, which one depends on near_outer."""
    im=canvas();arc(im,MID+GAP/2,10,30);arc(im,MID-GAP/2,10,30)
    arc(im,MID+GAP/2 if near_outer else MID-GAP/2,4,9.6)
    arc(im,-1.75,4,30,WHITE)
    return im


def test_near_single_strand_is_labelled_by_the_far_pairs():
    cfg=replace(C,carla_strand_pair=True)
    for outer in (True,False):
        d,y=left(scene(outer),cfg)
        assert abs(y(6.)-MID)<.04 and abs(y(20.)-MID)<.04
        assert abs(d.strand_gap['left'][0]-GAP)<.04


def test_a_third_strand_does_not_break_the_pair():
    # A repair mark beside the far pair: the old split (all strands within 0.55 m) fails and
    # the line moves half a gap onto the outer strand; the pair vote keeps the centre.
    im=scene(True);arc(im,MID+GAP/2+.35,18,24)
    _,y=left(im,C)
    assert abs(y(6.)-MID)>.10
    _,y=left(im,replace(C,carla_strand_pair=True))
    assert abs(y(6.)-MID)<.04 and abs(y(20.)-MID)<.04


def test_remembered_pair_centre_labels_a_lone_strand():
    """Frame 1 sees both strands; 2 m further on only the inner one, with no pair to vote.
    The centre remembered in odometry coordinates puts the boundary back between the
    strands; without odometry nothing is remembered."""
    cfg=replace(C,carla_strand_pair=True,carla_strand_memory_m=25.)
    pair=canvas();arc(pair,MID+GAP/2,4,30,k=0);arc(pair,MID-GAP/2,4,30,k=0);arc(pair,-1.75,4,30,WHITE,k=0)
    lone=canvas();arc(lone,MID-GAP/2,4,30,k=0);arc(lone,-1.75,4,30,WHITE,k=0)
    valid=np.full(pair.shape[:2],255,np.uint8)
    d=CarlaDetector(cfg);d.pose=(0.,0.,0.,0.);d.detect(d.binary(pair,valid))
    assert d.strand_memory['left']
    d.pose=(2.,2.,0.,0.);d.last_spines={}  # no previous-frame boundary to lean on
    d.detect(d.binary(lone,valid))
    b={b.side:b for b in d.boundary_candidates}['left'].points
    assert abs(float(np.interp(6.,b[:,0],b[:,1]))-MID)<.04
    d.pose=None;d.last_spines={};d.detect(d.binary(lone,valid))
    assert 'left' not in d.strand_memory
    b={b.side:b for b in d.boundary_candidates}['left'].points
    assert abs(float(np.interp(6.,b[:,0],b[:,1]))-(MID-GAP/2))<.04  # one strand alone, no memory: left on it
