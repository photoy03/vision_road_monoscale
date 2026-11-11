#!/usr/bin/env python3
"""Fetch only KITTI left color frames and OXTS from the public ZIP, with CRC checks."""
import argparse
from collections import OrderedDict
import io
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import time
import urllib.request
import zipfile
import zlib


BASE='https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/'


class RemoteZip(io.RawIOBase):
    def __init__(self,url):
        self.url=url; self.position=0; self.block_size=4*1024*1024; self.cache=OrderedDict()
        with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=30) as response:
            self.size=int(response.headers['Content-Length']); self.etag=response.headers.get('ETag')
        self.downloaded=0;self.requests=0;self.last_progress=0.
        self.pool=ThreadPoolExecutor(max_workers=4);self.futures={};self.plan=[];self.plan_positions={}
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        if self.position<0:raise ValueError('Negative remote seek')
        return self.position
    def prefetch(self,files):
        blocks=set()
        for item in files:
            start=item.header_offset
            end=min(self.size-1,start+item.compress_size+65536)
            blocks.update(range(start//self.block_size,end//self.block_size+1))
        self.plan=sorted(blocks);self.plan_positions={value:i for i,value in enumerate(self.plan)}
        if self.plan:self._schedule(self.plan[0])
    def _schedule(self,index):
        position=self.plan_positions.get(index)
        if position is None:return
        for value in self.plan[position:position+8]:
            if value not in self.cache and value not in self.futures:
                self.futures[value]=self.pool.submit(self._fetch,value)
    def _fetch(self,index):
        start=index*self.block_size;end=min(self.size,start+self.block_size)-1
        headers={'Range':f'bytes={start}-{end}'}
        if self.etag:headers['If-Match']=self.etag
        for attempt in range(3):
            try:
                with urllib.request.urlopen(urllib.request.Request(self.url,headers=headers),timeout=60) as response:
                    if response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):
                        raise ValueError('Server did not honor the requested public byte range')
                    data=response.read()
                if len(data)!=end-start+1:raise ValueError('Incomplete HTTP range')
                return data
            except Exception:
                if attempt==2:raise
                time.sleep(.5*(attempt+1))
    def _block(self,index):
        if index in self.cache:
            self.cache.move_to_end(index);self._schedule(index);return self.cache[index]
        self._schedule(index)
        future=self.futures.pop(index,None)
        data=future.result() if future is not None else self._fetch(index)
        self.cache[index]=data;self.downloaded+=len(data);self.requests+=1
        while len(self.cache)>16:self.cache.popitem(last=False)
        if time.monotonic()-self.last_progress>5:
            print(f'Downloaded {self.downloaded/1024**2:.1f} MiB in {self.requests} ranges',flush=True);self.last_progress=time.monotonic()
        return data
    def read(self,size=-1):
        if size is None or size<0:size=self.size-self.position
        size=min(size,max(0,self.size-self.position));chunks=[]
        while size:
            index=self.position//self.block_size;offset=self.position%self.block_size
            data=self._block(index);take=min(size,len(data)-offset)
            if take<=0:break
            chunks.append(data[offset:offset+take]);self.position+=take;size-=take
        return b''.join(chunks)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence',required=True)
    parser.add_argument('--root',required=True,help='KITTI root containing date directories')
    args=parser.parse_args()
    if not re.fullmatch(r'2011_\d{2}_\d{2}_drive_\d{4}',args.sequence):parser.error('Invalid KITTI sequence name')
    root=Path(args.root).expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
    date=args.sequence[:10];prefix=date+'/'+args.sequence+'_sync/'
    url=BASE+args.sequence+'/'+args.sequence+'_sync.zip'
    remote=RemoteZip(url);print(f'Archive {url} ({remote.size/1024**3:.2f} GiB); only required sensors will be fetched',flush=True)
    files=[]
    with zipfile.ZipFile(remote) as archive:
        for info in archive.infolist():
            name=info.filename
            if not name.startswith(prefix) or info.is_dir():continue
            relative=name[len(prefix):]
            selected=(re.fullmatch(r'image_02/data/\d{10}\.png',relative) or re.fullmatch(r'oxts/data/\d{10}\.txt',relative)
                      or relative in ('image_02/timestamps.txt','oxts/timestamps.txt','oxts/dataformat.txt'))
            if selected:files.append(info)
        files.sort(key=lambda item:item.header_offset)
        remote.prefetch([item for item in files if not (root/item.filename).exists()])
        if not files:raise ValueError('Expected left-camera/OXTS entries were not found')
        for number,info in enumerate(files,1):
            target=(root/info.filename).resolve()
            if not target.is_relative_to(root):raise ValueError('Unsafe ZIP path')
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if target.stat().st_size!=info.file_size or zlib.crc32(target.read_bytes())&0xffffffff!=info.CRC:
                    raise ValueError('Existing file differs; refusing to overwrite '+str(target))
                continue
            part=target.with_name(target.name+'.part')
            with archive.open(info) as source, part.open('wb') as output:
                while True:
                    chunk=source.read(1024*1024)
                    if not chunk:break
                    output.write(chunk)
            part.rename(target)
            if number%100==0:print(f'Extracted {number}/{len(files)} required files',flush=True)
    remote.pool.shutdown(wait=True)
    manifest={'source_url':url,'etag':remote.etag,'archive_bytes':remote.size,'downloaded_range_bytes':remote.downloaded,
              'range_requests':remote.requests,'selected_files':len(files),'image_frames':sum('/image_02/data/' in f.filename for f in files),
              'oxts_packets':sum('/oxts/data/' in f.filename for f in files),'integrity':'ZIP CRC32 verified by Python zipfile for each downloaded entry',
              'note':'Original timestamps and PNGs; no video transcoding, no lidar or other camera download.'}
    output=root/date/(args.sequence+'_sync')/'download_manifest.json'
    output.write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':main()
