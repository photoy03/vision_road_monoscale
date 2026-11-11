from dataclasses import replace
import numpy as np
from vision_road.core import Settings,Lane
from vision_road.path_tracking import PathTracking


def line(y):return np.column_stack([np.linspace(4,22,40),np.full(40,y)])
def candidate(y,score):return score,[dict(lane=Lane('right',line(y),.9))]
def chosen_y(result):return float(np.median(result[1][0]['lane'].points[:,1]))


def test_existing_track_wins_small_score_fluctuations_in_both_directions():
    for alternative in [-.7,-2.7]:
        tracker=PathTracking(Settings())
        result=tracker.select('right',[candidate(-1.7,10),candidate(alternative,11)],(line(-1.7),1.),1)
        assert chosen_y(result)==-1.7


def test_strong_new_candidate_must_be_consistent_before_switching():
    tracker=PathTracking(Settings());pool=[candidate(-1.7,10),candidate(-2.7,16)]
    for frame in [1,2]:assert chosen_y(tracker.select('right',pool,(line(-1.7),1.),frame))==-1.7
    assert chosen_y(tracker.select('right',pool,(line(-1.7),1.),3))==-2.7


def test_missing_incumbent_does_not_create_paint_and_can_be_reacquired():
    tracker=PathTracking(Settings())
    for frame in [1,2]:assert tracker.select('right',[candidate(-2.7,16)],(line(-1.7),1.),frame) is None
    assert chosen_y(tracker.select('right',[candidate(-2.7,16)],(line(-1.7),1.),3))==-2.7
    assert tracker.select('right',[],(line(-1.7),1.),4) is None


def test_same_frame_ranking_retries_do_not_confirm_switch_early():
    tracker=PathTracking(Settings());pool=[candidate(-2.7,16)]
    for _ in range(4):assert tracker.select('right',pool,(line(-1.7),1.),1) is None
    assert tracker.challenges['right'].count==1


def test_unstable_challenger_resets_confirmation():
    tracker=PathTracking(Settings())
    for frame,y in [(1,-2.7),(2,-3.3),(3,-2.7),(4,-3.3)]:assert tracker.select('right',[candidate(y,16)],(line(-1.7),1.),frame) is None
    assert tracker.challenges['right'].count==1


def test_unregistered_or_expired_prior_does_not_lock_old_coordinates():
    tracker=PathTracking(Settings());assert chosen_y(tracker.select('right',[candidate(-3,16)],None,1))==-3
    tracker.select('right',[candidate(-3,16)],(line(-1.7),1.),2);tracker.warp(None)
    assert not tracker.challenges
    tracker.reset();assert not tracker.challenges and not tracker.decisions


def test_disjoint_dashes_are_not_misclassified_as_lateral_switches():
    tracker=PathTracking(Settings())
    old=np.column_stack([np.linspace(4,7,20),np.full(20,-1.7)])
    new=np.column_stack([np.linspace(12,15,20),np.full(20,-1.7)])
    result=tracker.select('right',[(10,[dict(lane=Lane('right',new,.9))])],(old,1.),1)
    assert result is not None
    assert tracker.decisions['right']['action']=='no_common_extent'
