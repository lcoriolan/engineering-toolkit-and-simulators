"""Exercise the real local transport, static assets and numerical-control endpoint."""
import json,threading,urllib.request,urllib.error
from http.server import ThreadingHTTPServer
import pytest
from engineering_sim.web_server import Handler

@pytest.fixture
def server():
    srv=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    t=threading.Thread(target=srv.serve_forever,daemon=True);t.start()
    yield 'http://127.0.0.1:'+str(srv.server_port)
    srv.shutdown();srv.server_close();t.join()

def post(url,payload,**headers):
    req=urllib.request.Request(url+'/api/simulation/browser',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json',**headers})
    with urllib.request.urlopen(req,timeout=5) as response:return json.load(response)

def test_real_http_lifecycle(server):
    grant=post(server,dict(action='create',operation='flow.lbm2d',cells=16))
    state=post(server,dict(action='complete',id=grant['id'],token=grant['permit']['token'],result=dict(finite=True,steps=0)))
    assert state['state']=='running'
    post(server,dict(action='pause',id=grant['id']))
    assert post(server,dict(action='grant',id=grant['id'],steps=1))['permit'] is None
    assert post(server,dict(action='stop',id=grant['id']))['state']=='stopped'

def test_assets_and_unavailable_paths(server):
    for path in ('/','/lbm.js','/lbm3d.js','/simulation-engine.js'):
        with urllib.request.urlopen(server+path) as response:assert len(response.read())>100
    with pytest.raises(urllib.error.HTTPError) as error:urllib.request.urlopen(server+'/../execution.py')
    assert error.value.code==404

def test_cross_origin_request_rejected(server):
    with pytest.raises(urllib.error.HTTPError) as error:post(server,{'action':'create'},Origin='https://example.org')
    assert error.value.code==400
