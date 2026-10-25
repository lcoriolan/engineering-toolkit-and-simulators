"""Command-line interface for calculations, simulations and CAD operations."""
import argparse,json,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('list');run=sub.add_parser('run');run.add_argument('operator');run.add_argument('--input',type=Path);run.add_argument('--output',type=Path)
    demo=sub.add_parser('demo');demo.add_argument('--device',default='cpu');demo.add_argument('--output')
    serve=sub.add_parser('serve');serve.add_argument('--port',type=int,default=8765)
    cad=sub.add_parser('cad-import');cad.add_argument('source',type=Path);cad.add_argument('destination',type=Path);cad.add_argument('--name',default='converted_model')
    build=sub.add_parser('cad-build');build.add_argument('input',type=Path);build.add_argument('destination',type=Path)
    args=p.parse_args()
    try:
        if args.command=='serve':
            from .web_server import serve
            serve(args.port);return
        if args.command=='cad-build':
            from .cad.builder import build
            print(json.dumps(build(json.loads(args.input.read_text()),args.destination),indent=2));return
        if args.command=='cad-import':
            from .cad.importer import convert
            convert(args.source,args.destination,args.name);return
        from .api import OPERATORS,run,plain
        if args.command=='list':value={'operators':sorted(OPERATORS),'science':'Use operator science with payload containing op and args; mechanics and science operator catalogs are available through that dispatcher.'}
        elif args.command=='demo':value=run('world',dict(device=args.device,output=args.output))
        else:
            value=run(args.operator,json.loads(args.input.read_text()) if args.input else json.load(sys.stdin))
        text=json.dumps(plain(value),indent=2,allow_nan=False)+'\n'
        if args.command=='run' and args.output:args.output.write_text(text)
        else:print(text,end='')
    except Exception as e:
        print(json.dumps({'ok':False,'error':type(e).__name__+': '+str(e)}),file=sys.stderr);raise SystemExit(1)
