#!/usr/bin/env python3
"""Serve the local lane videos on loopback with byte-range seeking support."""
import argparse,functools,hashlib,json,mimetypes,os,re,subprocess,sys,time,urllib.request,webbrowser
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path

class Handler(SimpleHTTPRequestHandler):
    protocol_version='HTTP/1.1'
    def list_directory(self,path):
        self.send_error(403,'Directory listing disabled');return None
    def do_GET(self):
        if self.path=='/health':
            body=json.dumps({'app':'vision-road-results','root_id':self.server.root_id}).encode()
            self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body);return
        return super().do_GET()
    def send_head(self):
        self.byte_range=None
        path=Path(self.translate_path(self.path))
        root=Path(self.directory).resolve()
        if not path.resolve().is_relative_to(root):
            self.send_error(403);return None
        if path.is_dir():path=path/'index.html'
        if not path.is_file():self.send_error(404);return None
        size=path.stat().st_size;start,end=0,size-1;partial=False
        requested=self.headers.get('Range')
        if requested:
            match=re.fullmatch(r'bytes=(\d*)-(\d*)',requested.strip())
            if match:
                a,b=match.groups()
                if a:start=int(a);end=min(int(b),end) if b else end
                elif b:start=max(0,size-int(b))
                if start>end or start>=size:
                    self.send_response(416);self.send_header('Content-Range',f'bytes */{size}');self.send_header('Content-Length','0');self.end_headers();return None
                partial=True
        stream=path.open('rb');stream.seek(start)
        self.send_response(206 if partial else 200)
        self.send_header('Content-Type',mimetypes.guess_type(path)[0] or 'application/octet-stream')
        self.send_header('Content-Length',str(end-start+1));self.send_header('Accept-Ranges','bytes');self.send_header('Cache-Control','no-cache')
        if partial:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
        self.end_headers();self.byte_range=end-start+1;return stream
    def copyfile(self,source,outputfile):
        remaining=self.byte_range
        try:
            while remaining:
                block=source.read(min(256*1024,remaining))
                if not block:break
                outputfile.write(block);remaining-=len(block)
        except (BrokenPipeError,ConnectionResetError):pass

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--serve',action='store_true');parser.add_argument('--print-url',action='store_true');parser.add_argument('--port',type=int,default=18765);parser.add_argument('--directory',type=Path,default=Path(__file__).resolve().parents[1]/'results/classical_five_videos');a=parser.parse_args()
    root=a.directory.resolve();root_id=hashlib.sha256(str(root).encode()).hexdigest()[:16];base=f'http://127.0.0.1:{a.port}'
    if a.serve:
        server=ThreadingHTTPServer(('127.0.0.1',a.port),functools.partial(Handler,directory=str(root)));server.root_id=root_id
        print(base+'/index.html',flush=True);server.serve_forever();return
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def ready():
        try:
            with opener.open(base+'/health',timeout=.5) as r:return json.load(r)=={'app':'vision-road-results','root_id':root_id}
        except Exception:return False
    if not ready():
        with (root/'playback_server.log').open('a') as log:
            subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--serve','--port',str(a.port),'--directory',str(root)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
        for _ in range(40):
            if ready():break
            time.sleep(.1)
        else:raise SystemExit('Local viewer did not start; see playback_server.log')
    url=base+'/index.html';print(url,flush=True)
    if not a.print_url:webbrowser.open(url,new=2)
if __name__=='__main__':main()
