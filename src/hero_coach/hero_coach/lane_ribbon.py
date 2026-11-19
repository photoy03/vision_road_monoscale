"""The lane guidance ribbon as coach_node's ideal when it has no parking segment (plan 29).  Pure Python, no ROS."""


def lane_ideal(plan_active, ribbon, ribbon_age, max_age=.5):
    """The ribbon (hero_hud TrajectoryRibbon, base_link, from lane_guidance) to show as the scene's ideal, or None.

    plan_active: coach_node has a parking plan with a current segment (that one wins, it is the drive's own plan).
    ribbon_age: seconds since the ribbon arrived (None: none yet).  A ribbon with fewer than 2 points means
    lane_guidance shows nothing now."""
    if plan_active or ribbon is None or ribbon_age is None or not 0. <= ribbon_age <= max_age:return None
    return ribbon if len(ribbon.points) >= 2 else None
