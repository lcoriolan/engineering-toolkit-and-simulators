"""Import the installed package from a neutral directory; report optional CAD boundaries."""
import importlib,importlib.util,json,pkgutil
import engineering_sim
names=[m.name for m in pkgutil.walk_packages(engineering_sim.__path__,engineering_sim.__name__+'.')]
results=[]
for name in names:
    if name.endswith('.portable_bridge'):
        results.append(dict(module=name,status='flat-export-only bridge; exercised by portable dispatcher tests'));continue
    try:importlib.import_module(name);results.append(dict(module=name,status='imported'))
    except ModuleNotFoundError as exc:
        if exc.name in ('FreeCAD','Part'):
            results.append(dict(module=name,status='requires separate FreeCAD interpreter',dependency=exc.name))
        else:results.append(dict(module=name,status='failed',error_type=type(exc).__name__,dependency=exc.name))
    except Exception as exc:results.append(dict(module=name,status='failed',error_type=type(exc).__name__))
report=dict(modules=len(names),failed=sum(r['status']=='failed' for r in results),results=results)
print(json.dumps(report,indent=2))
if report['failed']:raise SystemExit(1)
