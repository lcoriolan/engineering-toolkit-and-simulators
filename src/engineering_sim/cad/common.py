"""Minimal portable support for the exact CAD converter."""
import hashlib,importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write_json(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
class Model:
    def __init__(self,path):
        path=Path(path);spec=importlib.util.spec_from_file_location('converted_'+sha(path)[:16],path)
        self.module=importlib.util.module_from_spec(spec);exec(compile(path.read_text(),str(path),'exec'),self.module.__dict__)
    def normalize(self,raw=None):
        p=self.module.defaults();p.update(raw or {});errors=self.module.validate(p)
        if errors:raise ValueError(str(errors))
        return p
    def recipe(self):return dict(parameters=self.normalize())
