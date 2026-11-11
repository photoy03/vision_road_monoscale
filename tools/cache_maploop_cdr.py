#!/usr/bin/env python3
"""Compress one camera topic, verify every byte against the read-only SSD bag."""
import argparse,hashlib,json,sqlite3,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src/vision_road'))
from vision_road.zstd_cdr import Codec


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bag',type=Path,default=Path('/media/jaehun/38F7-677B/rosbag/maploop_30hz/maploop_30hz_0.db3'));ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    con=sqlite3.connect('file:'+str(a.bag.resolve())+'?mode=ro&immutable=1',uri=True)
    topic=con.execute('select id from topics where name=?',('/sensing/camera/dashcam/image_raw',)).fetchone()[0]
    entries=[];codec=None;start=time.perf_counter();temporary=a.output.with_suffix(a.output.suffix+'.partial')
    with temporary.open('wb') as file:
        file.write(b'VRCDRZ01')
        for i,(data,) in enumerate(con.execute('select data from messages where topic_id=? order by timestamp',(topic,))):
            if codec is None:codec=Codec(len(data))
            compressed=codec.compress(data)
            assert codec.decompress(compressed,len(data))==data,'Lossless round-trip failed'
            entries.append(dict(offset=file.tell(),compressed_size=len(compressed),raw_size=len(data),sha256=hashlib.sha256(data).hexdigest()))
            file.write(compressed)
            if i%150==0:print(i,round(time.perf_counter()-start,1),round(file.tell()/2**20,1),'MiB',flush=True)
    con.close();temporary.replace(a.output)
    manifest=dict(version=1,source=str(a.bag),source_size=a.bag.stat().st_size,source_mtime_ns=a.bag.stat().st_mtime_ns,max_raw_size=codec.capacity,entries=entries,frames=len(entries),lossless_round_trip_verified=True,elapsed_sec=time.perf_counter()-start)
    a.output.with_suffix(a.output.suffix+'.json').write_text(json.dumps(manifest,separators=(',',':')));print('done',len(entries),a.output.stat().st_size,flush=True)
if __name__=='__main__':main()
