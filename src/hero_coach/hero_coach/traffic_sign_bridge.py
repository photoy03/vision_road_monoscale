"""YOLO detections -> the coach's traffic light and sign topics (plan 29).

In:  /yolo/detections (std_msgs/String JSON from src/yolo_signs: stamp, image_width, detections [{class_name,
     confidence, xyxy}]).
Out: /hero/traffic_light (hero_hud/SignalIndicator) and /hero/sign (hero_hud/SignMarker, KIND_SPEED + value,
     SOURCE_FRONT), the topics hero_coach's coach_node reads (in simulation carla_traffic fills them; the coach does
     not ask which).  lane_guidance reads the same topics.
What is shown comes from traffic_signs.SignTracker (2 of the last 3 messages, light off 0.5 s after the last light,
sign kept 3 s; with several lights the large, central one wins).  While something is confirmed it is sent again with
every detection message (the coach drops what is older than signal_timeout_sec, 1 s); when it goes off,
STATE_NONE / KIND_UNKNOWN is sent once.  Parameters: input_topic, signal_topic, sign_topic and sign.<SignSettings
field> (e.g. sign.select_centre_sigma)."""
import json

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String

from .traffic_signs import SignTracker, SignSettings

STATES = {'red': 1, 'yellow': 2, 'green': 3}  # SignalIndicator.STATE_RED / YELLOW / GREEN


class TrafficSignBridge(Node):
    def __init__(self):
        super().__init__('traffic_sign_bridge')
        p = lambda name, default: self.declare_parameter(name, default).value
        settings = SignSettings()
        for name in settings.__dataclass_fields__:
            setattr(settings, name, type(getattr(settings, name))(p('sign.'+name, getattr(settings, name))))
        self.tracker = SignTracker(settings)
        from hero_hud.msg import SignalIndicator, SignMarker
        self.SignalIndicator, self.SignMarker = SignalIndicator, SignMarker
        self.signal_pub = self.create_publisher(SignalIndicator, p('signal_topic', '/hero/traffic_light'), 10)
        self.sign_pub = self.create_publisher(SignMarker, p('sign_topic', '/hero/sign'), 10)
        self.create_subscription(String, p('input_topic', '/yolo/detections'), self.on_detections, 10)
        self.shown = (None, None)
        self.get_logger().info('Traffic sign bridge: '+str(settings))

    def on_detections(self, msg):
        data = json.loads(msg.data);stamp = data.get('stamp', {})
        t = float(stamp.get('sec', 0))+float(stamp.get('nanosec', 0))*1e-9
        self.tracker.update(t, data.get('detections', []), data.get('image_width'))
        light, sign = self.tracker.state(t)
        for out in messages(self.SignalIndicator, self.SignMarker, light, sign, self.shown, self.tracker):
            (self.signal_pub if isinstance(out, self.SignalIndicator) else self.sign_pub).publish(out)
        self.shown = (light, sign)


def messages(SignalIndicator, SignMarker, light, sign, shown, tracker):
    """The messages to send for this detection message: the confirmed light and sign (again each time), and
    STATE_NONE / KIND_UNKNOWN once when one goes off."""
    out = []
    if light is not None or shown[0] is not None:
        m = SignalIndicator();m.state = STATES[light] if light is not None else SignalIndicator.STATE_NONE;out.append(m)
    if sign is not None or shown[1] is not None:
        m = SignMarker();m.source = SignMarker.SOURCE_FRONT
        if sign is not None:m.kind = SignMarker.KIND_SPEED;m.value = float(sign);m.confidence = float(tracker.sign_confidence)
        else:m.kind = SignMarker.KIND_UNKNOWN
        out.append(m)
    return out


def main():
    rclpy.init();node = TrafficSignBridge()
    try:rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):pass  # Ctrl+C or a launch shutdown
    finally:
        node.destroy_node();rclpy.try_shutdown()
