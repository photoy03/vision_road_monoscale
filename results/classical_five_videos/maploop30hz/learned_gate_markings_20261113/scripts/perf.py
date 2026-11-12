import json,sys,numpy as np
for f in sys.argv[1:]:
    d=json.load(open(f));r=d['records']
    lat=np.array([(x['finish_ns']-x['arrival_ns'])/1e6 for x in r]);ph={};[ph.setdefault(k,[]).append(v) for x in r for k,v in x['phases'].items()]
    print(f.split('/')[-2],'frames',d['frames'],'mean %.1f p95 %.1f over33 %d'%(lat.mean(),np.percentile(lat,95),(lat>33.3).sum()),{k:round(float(np.mean(v)),2) for k,v in ph.items()})
