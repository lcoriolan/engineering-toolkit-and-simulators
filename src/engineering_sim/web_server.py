"""Loopback-only host for the actual browser numerical kernels and central controller."""
import json
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from .execution import browser_request
WEB=Path(__file__).with_name('web')
ASSETS={'/':('index.html','text/html'),'/lbm.js':('lbm.js','text/javascript'),'/lbm3d.js':('lbm3d.js','text/javascript'),'/simulation-engine.js':('simulation-engine.js','text/javascript')}
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def reply(self,value,status=200):
        data=json.dumps(value,allow_nan=False).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    def do_GET(self):
        entry=ASSETS.get(self.path)
        if not entry:self.reply({'error':'Not found'},404);return
        data=(WEB/entry[0]).read_bytes();self.send_response(200);self.send_header('Content-Type',entry[1]);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    def do_POST(self):
        try:
            if self.path!='/api/simulation/browser':self.reply({'error':'Not found'},404);return
            host=self.headers.get('Host','');allowed={f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}
            if host not in allowed or self.headers.get('Origin') not in (None,'http://'+host):raise ValueError('Same-origin request required.')
            if self.headers.get('Content-Type','').split(';')[0]!='application/json':raise ValueError('JSON required.')
            n=int(self.headers.get('Content-Length','0'))
            if not 0<n<=16384:raise ValueError('Request size limit exceeded.')
            self.reply(browser_request(json.loads(self.rfile.read(n))))
        except (ValueError,KeyError,TypeError) as e:self.reply({'error':str(e)},400)
def serve(port=8765):
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    print(f'Browser flow simulator: http://localhost:{server.server_port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
