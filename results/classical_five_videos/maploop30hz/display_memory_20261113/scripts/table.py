import json,sys
runs=[(a.split('=')[0],json.load(open(a.split('=')[1]))) for a in sys.argv[1:]]
for seg in runs[0][1]:
    if seg.split('/')[0] in ('junction',):continue
    for name,r in runs:
        v=r[seg];g=lambda k:v.get(k) or {}
        print('%-16s %-7s ev %3d mem %s | h36 ev %5s/%5s oth %5s/%5s | y3 ev %5s oth %5s | h03 all %5s/%5s | y0 all %5s/%5s'%(seg,name,v['events'],v['memory'],
          g('disp_head_event').get('rms'),g('disp_head_event').get('p95'),g('disp_head_other').get('rms'),g('disp_head_other').get('p95'),
          g('disp_y3_event').get('p95'),g('disp_y3_other').get('p95'),g('disp_head03_all').get('rms'),g('disp_head03_all').get('p95'),g('disp_y0_all').get('rms'),g('disp_y0_all').get('p95')))
