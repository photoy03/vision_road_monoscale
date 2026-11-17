"""Topics beside the two own lines: the neighbouring lanes' outer lines and the lane centrelines.

road_node publishes them, or, with image_in_worker, the perception worker does, right after it has sent
road_node the frame's main result (so road_node's own outputs are not held up by them).

/road/lane_outer (std_msgs/String, JSON, every frame, base_link): per outer line side (left_outer,
right_outer), base_side, points [x, y], style solid|dashed|unknown, style_confidence, yellow,
opposite_direction (the own line of that side is yellow: the lane beyond it is oncoming), gap_m (distance
from the own line along its normal), observed_m (forward range of its paint this frame; [] when carried
over), paint_m, hits, reprojected.
/road/lane_outer_markers: one LINE_STRIP per outer line (ns outer/<side>, magenta; color.r halved when
carried over, yellow paint: color.g=1), at the debug image rate.

/road/lane_centres (std_msgs/String, JSON, every frame, base_link: x forward, y left), per lane: name
ego|left|right, index 0|-1|+1, points [x, y] (the middle across the lane), both_m (forward range with both
boundaries facing each other; [] when estimated), width_m (this frame; the running width when estimated),
tracked_width_m, estimated (no facing boundaries: one moved half a width across it), ref_x_m (where the
next three are given: carla_centre_ref_m ahead, or the centreline's first point when it starts later),
offset_m (y of the centreline there, + left), heading_rad (its direction against the vehicle's, + to the
left), curvature_1pm (+ turning left), offset_at_axle_m (offset_m carried back to x = 0 along the heading:
outside the view), left_boundary / right_boundary (line, style, yellow, reprojected; null when missing),
opposite_direction, and for the neighbouring lanes crossable (the line shared with the own lane is white
dashed), edge_unknown (its outer edge is a white solid line: a road edge with a shoulder may lie beyond,
or a lane), width_ratio (to the own lane) and lane_like (ratio 0.85-1.15 and at least 2.8 m wide).
/road/lane_centre_markers: LINE_STRIP per lane (ns centre/<name>, orange; estimated: half alpha), at the
debug image rate."""
import json
import math
import numpy as np
from .display_filter import simplify


class SideTopics:
    def __init__(self,node,frame,debug_hz,error,outer,centre):
        from std_msgs.msg import String
        from visualization_msgs.msg import MarkerArray
        self.frame=frame;self.debug_hz=debug_hz;self.error=error;self.last={}
        self.outer=(node.create_publisher(MarkerArray,'/road/lane_outer_markers',10),node.create_publisher(String,'/road/lane_outer',10)) if outer else None
        self.centre=(node.create_publisher(MarkerArray,'/road/lane_centre_markers',10),node.create_publisher(String,'/road/lane_centres',10)) if centre else None

    def _markers_due(self,kind,t):
        if self.debug_hz>0 and t-self.last.get(kind,-math.inf)<1/self.debug_hz-1e-8:return False
        self.last[kind]=t;return True

    def publish(self,stamp,t,outer,centres):
        """stamp: builtin_interfaces/Time of the frame, t its seconds; outer: Lanes with .outer; centres: dicts."""
        from std_msgs.msg import String,Header
        from visualization_msgs.msg import Marker,MarkerArray
        from geometry_msgs.msg import Point
        seconds=stamp.sec+stamp.nanosec*1e-9
        def strip(header,ns,ident,points,r,g,b,a,width):
            m=Marker();m.header=header;m.ns=ns;m.id=ident;m.type=Marker.LINE_STRIP;m.action=Marker.ADD;m.pose.orientation.w=1.
            m.scale.x=width;m.color.r=r;m.color.g=g;m.color.b=b;m.color.a=a;m.lifetime.nanosec=300000000
            m.points=[Point(x=float(p[0]),y=float(p[1]),z=.02) for p in points];return m
        for kind,lanes,pubs in (('outer',outer,self.outer),('centre',centres,self.centre)):
            if pubs is None:continue
            due=self._markers_due(kind,t);items=[]
            msg=MarkerArray();clear=Marker();clear.header=Header(stamp=stamp,frame_id=self.frame);clear.action=Marker.DELETEALL;msg.markers.append(clear)
            for ident,lane in enumerate(lanes):
                if kind=='outer':
                    info=getattr(lane,'outer',{}) or {};points=np.round(simplify(lane.points,self.error),3)
                    items.append(dict(side=lane.side,base_side=lane.side[:-6],points=points.tolist(),style=lane.style,
                                      style_confidence=round(float(lane.style_confidence),3),yellow=bool(info.get('yellow')),
                                      opposite_direction=bool(info.get('opposite')),gap_m=info.get('gap_m'),observed_m=info.get('observed',[]),
                                      paint_m=info.get('paint_m'),hits=int(lane.hits),reprojected=bool(lane.reprojected)))
                    if due:
                        m=strip(clear.header,'outer/'+lane.side,ident,points,.5 if lane.reprojected else 1.,1. if info.get('yellow') else 0.,1.,.9,.05)
                        m.text=lane.style+(' yellow' if info.get('yellow') else '')+(' opposite' if info.get('opposite') else '');msg.markers.append(m)
                else:
                    points=np.round(simplify(np.asarray(lane['points'],float),self.error),3);items.append(dict(lane,points=points.tolist()))
                    if due:msg.markers.append(strip(clear.header,'centre/'+lane['name'],ident,points,1.,.55,0.,.45 if lane['estimated'] else .9,.04))
            if due:pubs[0].publish(msg)
            pubs[1].publish(String(data=json.dumps(dict(stamp=seconds,frame_id=self.frame,lanes=items),separators=(',',':'))))
