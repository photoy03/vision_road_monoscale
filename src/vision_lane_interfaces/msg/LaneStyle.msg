# Pattern classification of an existing lane candidate; not a guarantee it is road paint.
uint8 UNKNOWN=0
uint8 SOLID=1
uint8 DASHED=2
string side                        # left/right; matches MarkingSegment.family
uint8 style
float32 confidence                 # style confidence; 0 when UNKNOWN
float32 observed_span_m             # contiguous camera-visible span along the curve
float32 paint_fraction              # fraction supported by measured paint candidates
uint32 visible_dash_count
bool reprojected                    # true when this is carried from a previous observation
builtin_interfaces/Time source_stamp # last actual lane observation; array header is current time
builtin_interfaces/Time style_source_stamp # last positive style evidence; may precede current geometry
