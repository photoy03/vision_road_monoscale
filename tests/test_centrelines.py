"""Plan 20: lane centrelines (carla_centre_enabled) and the deferred outputs beside the own lines."""
from dataclasses import replace
import numpy as np
from vision_road.core import Settings
from vision_road.carla import CarlaDetector
from vision_road.temporal import TemporalDetector
from vision_road import centreline as C
from test_outer_lines import scene


def arc(radius,offset,start=.03,end=.3):
    """A left-turning arc around (0, radius), `offset` to the left of the radius-`radius` path."""
    th=np.linspace(start,end,300);r=radius-offset
    return np.column_stack([r*np.sin(th),radius-r*np.cos(th)])


def test_centre_is_taken_across_the_lane_in_a_curve():
    # A 20 m radius turn (a junction corner), 46 degrees of heading at the far end.
    left,right,middle=arc(20.,1.75,.03,.8),arc(20.,-1.75,.03,.8),arc(20.,0.,.03,.8)
    centre,width=C.between(left,right)
    assert np.allclose(width,3.5,atol=1e-3)
    assert np.max(np.abs(np.interp(centre[:,0],middle[:,0],middle[:,1])-centre[:,1]))<1e-4
    # At one x the two boundaries are not opposite each other: their plain average is off the middle.
    x=np.linspace(5,12,30);naive=(np.interp(x,left[:,0],left[:,1])+np.interp(x,right[:,0],right[:,1]))/2
    assert np.max(np.abs(naive-np.interp(x,middle[:,0],middle[:,1])))>.05


def test_reference_values_of_a_curve_and_one_boundary_moved_across():
    middle=arc(100.,0.,0.,.35)
    ref=C.at_reference(middle,5.,15.)
    assert ref['ref_x_m']==5. and abs(ref['curvature_1pm']-.01)<.001
    assert ref['offset_m']>0 and ref['heading_rad']>0
    # Carried back to 0 m along the heading, a straight line: it misses the curve by about k x^2 / 2.
    assert abs(ref['offset_at_axle_m']+.01*5**2/2)<.01
    moved=C.shifted(arc(100.,1.75,0.,.35),3.5,'left')
    assert np.max(np.abs(np.interp(moved[:,0],middle[:,0],middle[:,1])-moved[:,1]))<1e-3


def test_joined_centre_covers_both_boundaries_reach():
    x=np.arange(5.5,10.4,.1);left=np.column_stack([x,np.full_like(x,1.75)])
    x2=np.arange(9.,30.,.1);right=np.column_stack([x2,np.full_like(x2,-1.65)])
    centre,span,_=C.joined(left,right,3.4)
    assert span is not None and abs(span[0]-9.)<.2 and abs(span[1]-10.3)<.2
    assert centre[0,0]<=5.6 and centre[-1,0]>=29.8 and np.max(np.abs(centre[:,1]-.05))<1e-6
    assert np.all(np.diff(centre[:,0])>=0)


def run(c,im,frames=6,defer=False):
    d=TemporalDetector(c);d.detector=CarlaDetector(c);valid=np.full(im.shape[:2],255,np.uint8)
    for i in range(frames):
        d.process(im,valid,1+i*.05,(0.,0.,0.),defer=defer)
        if defer:d.finish()
    return d


def test_centrelines_are_off_by_default():
    assert Settings().carla_centre_enabled is False
    c,im=scene();d=run(replace(c,carla_outer_enabled=True),im)
    assert d.centre_lanes==[]


def test_three_lanes_with_their_middles_widths_and_flags():
    c,im=scene();d=run(replace(c,carla_outer_enabled=True,carla_centre_enabled=True),im)
    lanes={l['name']:l for l in d.centre_lanes}
    assert set(lanes)=={'ego','left','right'}
    for name,y,width in (('ego',.05,3.4),('left',3.5,3.5),('right',-3.4,3.5)):
        lane=lanes[name];p=np.asarray(lane['points']);p=p[(p[:,0]>=6)&(p[:,0]<=25)]
        assert not lane['estimated'] and abs(lane['width_m']-width)<.12
        assert np.max(np.abs(p[:,1]-y))<.1 and abs(lane['offset_m']-y)<.1 and abs(lane['heading_rad'])<.01
    assert lanes['left']['opposite_direction'] and not lanes['right']['opposite_direction']
    # The own right line is white dashed: one may move into the right lane, not over the yellow line.
    assert lanes['right']['crossable'] and not lanes['left']['crossable']
    assert lanes['right']['edge_unknown'] and lanes['right']['lane_like']
    assert lanes['ego']['left_boundary']['yellow'] and lanes['ego']['right_boundary']['style']=='dashed'


def test_own_lane_with_one_boundary_is_estimated_and_neighbours_need_both():
    c,im=scene(right_own=False);d=run(replace(c,carla_outer_enabled=True,carla_centre_enabled=True),im)
    lanes={l['name']:l for l in d.centre_lanes}
    assert lanes['ego']['estimated'] and lanes['ego']['right_boundary'] is None
    assert 'right' not in lanes


def test_deferred_outputs_are_the_same():
    c,im=scene();c=replace(c,carla_outer_enabled=True,carla_centre_enabled=True)
    a=run(c,im);b=run(c,im,defer=True)
    assert [l.side for l in a.outer_lanes]==[l.side for l in b.outer_lanes]
    for x,y in zip(a.outer_lanes,b.outer_lanes):assert np.array_equal(x.points,y.points)
    for x,y in zip(a.centre_lanes,b.centre_lanes):assert x['name']==y['name'] and np.array_equal(x['points'],y['points'])
    # A deferred frame left unfinished is finished by the next process() call.
    d=TemporalDetector(c);d.detector=CarlaDetector(c);valid=np.full(im.shape[:2],255,np.uint8)
    for i in range(6):d.process(im,valid,1+i*.05,(0.,0.,0.),defer=True)
    d.finish()
    assert [l.side for l in d.outer_lanes]==[l.side for l in a.outer_lanes]


def test_side_topics_send_every_frame_and_markers_at_the_debug_rate():
    import json
    from builtin_interfaces.msg import Time
    from vision_road.side_topics import SideTopics
    class Publisher:
        def __init__(self):self.sent=[]
        def publish(self,msg):self.sent.append(msg)
    class Node:
        def __init__(self):self.pubs={}
        def create_publisher(self,kind,topic,depth):self.pubs[topic]=Publisher();return self.pubs[topic]
    c,im=scene();d=run(replace(c,carla_outer_enabled=True,carla_centre_enabled=True),im)
    node=Node();topics=SideTopics(node,'base_link',5.,.025,True,True)
    for i in range(7):topics.publish(Time(sec=1,nanosec=i*33333333),1+i/30,d.outer_lanes,d.centre_lanes)  # 0.2 s at 30 Hz
    assert len(node.pubs['/road/lane_outer'].sent)==7 and len(node.pubs['/road/lane_centres'].sent)==7
    assert len(node.pubs['/road/lane_outer_markers'].sent)==2 and len(node.pubs['/road/lane_centre_markers'].sent)==2
    centres=json.loads(node.pubs['/road/lane_centres'].sent[-1].data)['lanes']
    assert {l['name'] for l in centres}=={'ego','left','right'}
    ego=next(l for l in centres if l['name']=='ego')
    for key in ('points','both_m','width_m','offset_m','heading_rad','curvature_1pm','ref_x_m','offset_at_axle_m','estimated','left_boundary','right_boundary','opposite_direction'):
        assert key in ego
    right=next(l for l in centres if l['name']=='right');assert right['crossable'] and right['edge_unknown'] and right['lane_like']
    outer=json.loads(node.pubs['/road/lane_outer'].sent[-1].data)['lanes']
    assert {l['side'] for l in outer}=={'left_outer','right_outer'}
