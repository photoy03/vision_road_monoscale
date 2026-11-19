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


MEMORY=dict(near_m=0.,memory_m=10.)


def dashed_drive(f,frames,speed=.3,seed=0):
    """A straight right line at y=-1.8 painted 3 m on / 3 m off, seen from 5.5 m.  The boundary
    road_node publishes spans the gaps and its heading is off by up to 10 mrad from frame to
    frame; the paint fragments are exact.  Returns the displayed lateral value at 0 m per frame."""
    rng=np.random.default_rng(seed);y0=[]
    for i in range(frames):
        travelled=i*speed;dashes=[]
        for k in range(30):
            a,b=k*6.-travelled,k*6.+3.-travelled
            if b>5.5 and a<30:dashes.append((max(a,5.5),min(b,30.)))
        start=dashes[0][0];err=rng.uniform(-.01,.01)
        x=np.linspace(start,30.,12);boundary=np.column_stack([x,-1.8+err*(x-15.)])
        paint=[('right',np.array([[a,-1.8],[b,-1.8]])) for a,b in dashes]
        f.update(i*DT,[('right',boundary,False)],(speed,0.,0.) if i else (0.,0.,0.),paint)
        out={side:p for side,p,_,_ in f.output()}
        if 'right' in out:y0.append(float(np.interp(0.,out['right'][:,0],out['right'][:,1])))
    return np.array(y0)


def test_remembered_paint_holds_the_near_end_of_a_dashed_line():
    plain=dashed_drive(DisplayFilter(FilterSettings(near_m=0.)),150)[60:]
    kept=dashed_drive(DisplayFilter(FilterSettings(**MEMORY)),150)[60:]
    # Without memory the end at the rear axle swings with the far fragments' heading.
    # (The remembered points are the boundary's, so they carry some of its error: 7 -> 2.5 cm at most.)
    assert np.abs(kept+1.8).max()<.03<.06<np.abs(plain+1.8).max()
    assert np.std(np.diff(kept))<.2*np.std(np.diff(plain))


def test_only_paint_seen_this_frame_is_remembered():
    f=DisplayFilter(FilterSettings(**MEMORY));line=np.column_stack([np.linspace(5.5,30,12),np.full(12,-1.8)])
    for i in range(8):f.update(i*DT,[('right',line,False)],(0.,0.,0.),[('right',np.array([[5.5,-1.8],[6.5,-1.8]]))])
    m=f.sides['right'].memory
    assert len(m) and m[:,0].min()>=5.5-1e-6 and m[:,0].max()<=6.6
    # Stray paint well off the boundary does not count as seen.
    f=DisplayFilter(FilterSettings(**MEMORY))
    for i in range(8):f.update(i*DT,[('right',line,False)],(0.,0.,0.),[('right',np.array([[5.5,-3.],[9.,-3.]]))])
    assert not len(f.sides['right'].memory)


def test_memory_is_dropped_by_travel_restart_lost_odometry_and_off_by_default():
    f=DisplayFilter(FilterSettings(**MEMORY));line=np.column_stack([np.linspace(5.5,30,12),np.full(12,-1.8)])
    paint=[('right',line)]
    for i in range(8):f.update(i*DT,[('right',line,False)],(0.,0.,0.),paint)
    assert len(f.sides['right'].memory)
    for i in range(8,8+40):f.update(i*DT,[],(.3,0.,0.))  # 12 m without seeing anything new
    assert not len(f.sides['right'].memory)
    for i in range(50,58):f.update(i*DT,[('right',line,False)],(0.,0.,0.),paint)
    f.update(57*DT,[('right',line,False)],None,paint)
    assert not len(f.sides['right'].memory)
    for i in range(58,66):f.update(i*DT,[('right',line,False)],(0.,0.,0.),paint)
    f.update(1.,[('right',line,False)],(0.,0.,0.),paint)  # the replay restarted
    assert len(f.sides["right"].memory)==17  # one frame: 5.5-9.5 m every 0.25 m
    f=DisplayFilter()
    for i in range(8):f.update(i*DT,[('right',line,False)],(0.,0.,0.),paint)
    assert not len(f.sides['right'].memory)


def test_a_line_switch_forgets_the_old_line():
    f=DisplayFilter(FilterSettings(**MEMORY));x=np.linspace(5.5,30,12)
    for i in range(30):
        f.update(i*DT,[('right',np.column_stack([x,np.full(12,-1.8)]),False)],(.3,0.,0.),[('right',np.array([[5.5,-1.8],[9.,-1.8]]))])
    for i in range(30,36):
        f.update(i*DT,[('right',np.column_stack([x+5,np.full(12,-3.6)]),False)],(.3,0.,0.),[('right',np.array([[10.5,-3.6],[14.,-3.6]]))])
    p={side:q for side,q,_,_ in f.output()}['right']
    # The new line is drawn flat down to the rear axle, not tilted toward the old one.
    assert np.abs(p[:,1]+3.6).max()<.03


def test_cubic_display_follows_a_straight_then_curving_line_and_is_off_by_default():
    x=np.arange(3.,25.,.5);y=-1.8+np.where(x>11,.012*(x-11)**2,0.);line=np.column_stack([x,y])
    def miss(stations):
        near=(x>=6)&(x<=20);return float(np.max(np.abs(evaluate(stations,x[near])-y[near])))
    quadratic=fit(line)[0];cubic=fit(line,cubic=.05)[0]
    assert len(quadratic)==3 and len(cubic)==4
    assert miss(quadratic)>.08 and miss(cubic)<.05
    # A straight or evenly curving line keeps its quadratic shape.
    arc=np.column_stack([x,-1.8+.004*x*x])
    assert np.allclose(evaluate(fit(arc,cubic=.05)[0],x),evaluate(fit(arc)[0],x),atol=.01)
    # Motion keeps a cubic a cubic.
    moved=shift(cubic,(1.,0.,.02))
    assert len(moved)==4 and abs(evaluate(moved,np.array([9.]))[0]-evaluate(cubic,np.array([10.]))[0])<.3


ENDS=dict(gate_span=True,gate_per_10m=1.,end_from_seen=True,start_from_seen=True,outside_quadratic=True,near_quadratic_m=6.)


def test_line_end_settings_are_off_by_default():
    s=FilterSettings()
    assert not (s.gate_span or s.end_from_seen or s.start_from_seen or s.outside_quadratic) and s.near_quadratic_m==0.


def test_an_observation_that_only_matches_at_10m_is_set_aside():
    """A diagonal joined to the line (as at a stop line) agrees at 10 m but not at 14 m."""
    diagonal=np.array([[7.9,1.75],[10.,1.75],[14.3,3.3]])
    shown={}
    for ends in (False,True):
        f=DisplayFilter(FilterSettings(near_m=0.,**(ENDS if ends else {})))
        for i in range(30):feed(f,i*DT,1.75,start=4.,end=14.)
        f.update(30*DT,[('left',diagonal,False)],(0.,0.,0.))
        out={side:(p,pred) for side,p,pred,_ in f.output()}
        shown[ends]=out['left']
    p,pred=shown[False];assert not pred and float(np.interp(14.,p[:,0],p[:,1]))>2.2  # taken: the far end bends away
    p,pred=shown[True];assert pred and abs(float(np.interp(14.,p[:,0],p[:,1]))-1.75)<.05  # kept line, predicted colour


def test_one_long_frame_does_not_lengthen_the_drawn_line():
    ends={}
    for on in (False,True):
        f=DisplayFilter(FilterSettings(near_m=0.,**(ENDS if on else {})))
        for i in range(30):feed(f,i*DT,1.75,start=5.,end=11.)
        feed(f,30*DT,1.75,start=5.,end=35.)
        out=feed(f,31*DT,1.75,start=5.,end=11.)
        ends[on]=float(out['left'][0][-1,0])
    assert ends[False]>=19.5 and ends[True]<=11.+3.+1e-6


def test_a_new_line_starts_near_its_first_seen_paint():
    """Seen from 9 m only: drawn from 6 m (extend_m below it), not swung down to 0 m."""
    starts={}
    for on in (False,True):
        f=DisplayFilter(FilterSettings(near_m=0.,**(ENDS if on else {})))
        for i in range(8):out=feed(f,i*DT,-1.6,start=9.,end=20.)
        starts[on]=float(out['left'][0][0,0])
    assert starts[False]==0. and abs(starts[True]-6.)<1e-6


def test_outside_the_seen_span_the_line_has_no_third_order_bend():
    """A line seen 4-12 m that curves away only at its end: past 12 m it goes on with the end's slope and
    a fading curvature instead of the cubic's growing bend."""
    x=np.linspace(4.,12.,33);y=1.75+.002*(x-4.)**3
    drawn={}
    for on in (False,True):
        f=DisplayFilter(FilterSettings(near_m=0.,cubic_m=.05,**(ENDS if on else {})))
        for i in range(30):f.update(i*DT,[('left',np.column_stack([x,y]),False)],(0.,0.,0.))
        drawn[on]=[q for side,q,_,_ in f.output() if side=='left'][0]
    p=drawn[True];assert p[-1,0]<=15.+1e-6
    bend=np.diff(p[:,1],2)[p[1:-1,0]>12.]  # second differences past the seen end (0.5 m steps)
    inside=np.diff(p[:,1],2)[(p[1:-1,0]>10.)&(p[1:-1,0]<=12.)]
    assert np.all(np.diff(bend)<=1e-9) and bend[0]<=inside[-1]+1e-9  # the curvature only fades past 12 m
    q=drawn[False];k=np.searchsorted(q[:,0],p[-1,0])
    assert q[k,1]>p[-1,1]  # the cubic bends further away


def test_trim_cuts_a_far_end_that_runs_off_the_kept_line():
    # Plan 32 (trim_m): an observation is cut at its first point beyond the limit and the rest is taken.
    f=DisplayFilter(FilterSettings(trim_m=.15,trim_per_10m=.15,end_from_seen=True))
    for i in range(10):f.update(i*DT,[('left',line(1.75),False)],(0.,0.,0.))
    track=f.sides['left']
    hooked=np.array([[4.,1.75],[9.,1.75],[14.,1.76],[20.,2.3],[26.,2.9]])
    cut=track.trimmed(hooked);assert len(cut)==3 and cut[-1,0]==14.
    assert len(track.trimmed(line(1.80)))==6  # within 0.15 m: whole
    f.update(10*DT,[('left',hooked,False)],(0.,0.,0.))
    assert not track.predicted and track.seen[-1][1]==14.  # taken, as far as the cut


def test_max_accel_draws_a_step_smoothly_but_gates_on_the_kept_line():
    # Plan 32 (max_accel_m): the kept line taken again 0.2 m away is drawn without a jump (second difference at 10 m
    # within the limit), reaches it without overshoot, and the gate still compares with the kept line.
    a=.005;f=DisplayFilter(FilterSettings(max_accel_m=a,min_cutoff_hz=100.,beta=0.))
    for i in range(10):feed(f,i*DT,1.75)
    ys=[]
    for i in range(10,70):ys.append(at10(feed(f,i*DT,1.95)))
    d2=np.abs(np.diff(ys,2));track=f.sides['left']
    assert d2.max()<=a+1e-3 and max(ys)<=1.95+1e-3 and abs(ys[-1]-1.95)<.01
    assert abs(evaluate(track.value,np.array([10.]))[0]-1.95)<.01  # the kept line is not held back
    # Off by default: the same step is drawn at once (the kept line follows quickly with this filter).
    g=DisplayFilter(FilterSettings(min_cutoff_hz=100.,beta=0.))
    for i in range(10):feed(g,i*DT,1.75)
    assert abs(at10(feed(g,10*DT,1.95))-1.95)<.02
