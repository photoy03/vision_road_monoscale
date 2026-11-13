"""HUD display filter: confirmation, smoothing, outlier gate, hold, motion, restart, fixed sampling and styles."""
import numpy as np
from vision_road.display_filter import DisplayFilter, FilterSettings, fit, evaluate, shift, relative

DT=1/30


def line(y,start=4.,end=30.,c=0.):
    x=np.linspace(start,end,6 if c==0 else 27)  # curves come densely sampled, as road_node simplifies to 2.5 cm
    return np.column_stack([x,y+c*x*x])


def feed(f,t,y=None,predicted=False,motion=(0.,0.,0.),**kw):
    f.update(t,[] if y is None else [('left',line(y,**kw),predicted)],motion)
    return {side:(p,pred,style) for side,p,pred,style in f.output()}


def at10(out):return float(np.interp(10.,out['left'][0][:,0],out['left'][0][:,1]))


def test_fit_recovers_a_quadratic_from_a_sparse_polyline():
    x=np.linspace(4,30,5);stations,near,far=fit(np.column_stack([x,1.+.02*x+.001*x*x]))
    assert near==4 and abs(far-30)<1e-6
    assert np.allclose(evaluate(stations,np.array([10.])),1.+.2+.1,atol=.01)


def test_side_is_shown_only_after_m_of_n_observed_frames():
    f=DisplayFilter()
    shown=[bool(feed(f,i*DT,1.75)) for i in range(6)]
    assert shown==[False]*4+[True,True]
    # Short or far-only pieces do not count toward confirmation.
    f=DisplayFilter()
    assert not any(feed(f,i*DT,1.75,end=8.) for i in range(10))
    assert not any(feed(f,i*DT,1.75,start=25.) for i in range(10))


def test_noise_is_suppressed_on_a_straight_line():
    rng=np.random.default_rng(0);f=DisplayFilter();out=[]
    for i in range(300):
        o=feed(f,i*DT,1.75+rng.normal(0,.03))
        if o:out.append(at10(o))
    out=np.array(out[30:])
    assert np.std(np.diff(out))<.01 and abs(out.mean()-1.75)<.01


def test_curve_entry_is_followed_without_large_lag():
    f=DisplayFilter();lag=[]
    for i in range(150):
        c=min(i,60)/60*.004  # curvature ramps in over 2 s, then holds
        o=feed(f,i*DT,1.75,c=c)
        if o and i>=10:lag.append(abs(at10(o)-(1.75+c*100)))
    assert np.mean(lag)<.15 and lag[-1]<.01


def test_single_jump_is_ignored_but_a_sustained_one_is_taken():
    f=DisplayFilter()
    for i in range(20):feed(f,i*DT,1.75)
    assert abs(at10(feed(f,20*DT,2.4))-1.75)<.02
    assert abs(at10(feed(f,21*DT,1.75))-1.75)<.02
    for i in range(22,24):assert abs(at10(feed(f,i*DT,2.4))-1.75)<.02
    assert abs(at10(feed(f,24*DT,2.4))-2.4)<.02


def test_lost_side_is_held_as_predicted_then_deleted():
    f=DisplayFilter()
    for i in range(20):feed(f,i*DT,1.75)
    t0=19*DT;held=[]
    for i in range(1,40):
        o=feed(f,t0+i*DT)
        held.append(bool(o))
        if o:assert o['left'][1] and abs(at10(o)-1.75)<.01
    assert all(held[:29]) and not any(held[31:])
    # Coming back needs confirmation again.
    t1=t0+40*DT
    assert not feed(f,t1,1.75)


def test_output_has_fixed_spacing_and_stops_three_metres_past_the_observation():
    f=DisplayFilter()
    for i in range(20):o=feed(f,i*DT,1.75,end=12.)
    x=o['left'][0][:,0]
    assert x[0]==3. and np.allclose(np.diff(x),.5) and abs(x[-1]-15.)<.01
    for i in range(20,60):o=feed(f,i*DT,1.75,end=34.)
    assert abs(o['left'][0][-1,0]-20.)<1e-6


def test_style_switches_only_after_it_is_stable():
    f=DisplayFilter(FilterSettings())
    for i in range(20):f.update_style(i*DT,'left','solid',1.);feed(f,i*DT,1.75)
    assert f.output()[0][3]=='solid'
    t=20*DT
    # A flicker shorter than 0.5 s, or low-confidence dashed, does not switch.
    for i in range(10):f.update_style(t+i*DT,'left','dashed',.9)
    f.update_style(t+10*DT,'left','solid',1.)
    for i in range(11,40):f.update_style(t+i*DT,'left','dashed',.3)
    assert f.output()[0][3]=='solid'
    for i in range(40,56):f.update_style(t+i*DT,'left','dashed',.9)
    assert f.output()[0][3]=='dashed'


def test_style_is_remembered_across_a_short_loss():
    f=DisplayFilter()
    for i in range(30):f.update_style(i*DT,'left','dashed',.9);feed(f,i*DT,1.75)
    t=30*DT
    for i in range(150):feed(f,t+i*DT)        # 5 s without the line: dropped, style kept
    t+=150*DT
    for i in range(10):o=feed(f,t+i*DT,1.75)  # back, with only UNKNOWN style evidence
    assert o['left'][2]=='dashed'
    for i in range(10,400):feed(f,t+i*DT)     # gone longer than hold + memory: forgotten
    t+=400*DT
    for i in range(10):o=feed(f,t+i*DT,1.75)
    assert o['left'][2] is None


def test_shift_moves_a_curve_into_the_next_base_link():
    flat=np.array([1.75]*3)
    assert np.allclose(shift(flat,(1.,0.,0.)),1.75)
    assert np.allclose(shift(flat,(0.,.5,0.)),1.25)
    turned=shift(np.zeros(3),(0.,0.,.1))
    assert abs(evaluate(turned,np.array([10.]))[0]+np.tan(.1)*10)<.02
    assert np.allclose(relative((1.,2.,np.pi/2),(1.,3.,np.pi/2)),(1.,0.,0.),atol=1e-9)


def test_held_line_follows_the_vehicle():
    f=DisplayFilter()
    for i in range(20):feed(f,i*DT,1.75)
    # The vehicle drifts left 5 cm per frame for 20 frames without seeing the line.
    for i in range(20,40):o=feed(f,i*DT,motion=(.5,.05,0.))
    assert o['left'][1] and abs(at10(o)-.75)<.02


def test_without_odometry_a_lost_side_is_kept_only_briefly():
    f=DisplayFilter()
    for i in range(20):feed(f,i*DT,1.75)
    held=[bool(feed(f,(20+i)*DT,motion=None)) for i in range(15)]
    assert all(held[:5]) and not any(held[8:])  # 0.2 s at 30 Hz


def test_rejected_jump_is_drawn_as_predicted():
    f=DisplayFilter()
    for i in range(20):feed(f,i*DT,1.75)
    o=feed(f,20*DT,2.4)
    assert o['left'][1] and abs(at10(o)-1.75)<.02
    o=feed(f,21*DT,1.75)
    assert not o['left'][1]


def test_replay_restart_starts_over():
    f=DisplayFilter()
    for i in range(300):f.update_style(i*DT,'left','dashed',.9);feed(f,i*DT,1.75)
    out=[feed(f,i*DT,2.5) for i in range(10)]
    assert not out[0] and out[-1] and abs(at10(out[-1])-2.5)<.02
    assert out[-1]['left'][2] is None


def test_unstable_candidates_do_not_keep_the_line_past_hold():
    # Review reproduction: candidates alternating 0.9 m apart for 4 s never settle.
    for motion,hold in (((0.,0.,0.),1.),(None,.2)):
        f=DisplayFilter()
        for i in range(20):feed(f,i*DT,1.75)
        t0=19*DT;shown=[]
        for i in range(1,121):
            t=t0+i*DT
            o=feed(f,t,2.4 if i%2 else 3.3,motion=motion)
            assert len(f.sides['left'].pending)<=3
            shown.append((t-t0,bool(o)))
        # Shown (as the kept line) until the hold runs out, then gone and not re-confirmed.
        assert all(s for dt,s in shown if dt<=hold-1e-6)
        assert not any(s for dt,s in shown if dt>hold+DT)


def test_sustained_move_is_taken_within_the_short_hold():
    # Without odometry the hold is 0.2 s; the two rejected frames before a move is taken fit in it.
    f=DisplayFilter()
    for i in range(20):feed(f,i*DT,1.75)
    out=[feed(f,(20+i)*DT,2.4,motion=None) for i in range(4)]
    assert all(out) and abs(at10(out[2])-2.4)<.02 and not out[2]['left'][1]
