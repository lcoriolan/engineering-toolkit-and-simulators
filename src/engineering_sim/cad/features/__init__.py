"""Generic FreeCAD features; parameters and published dimensions are supplied by the caller."""
import importlib
FEATURES=('air_scoop','body_tube','camera_dome','chamfer','control_fin','edf_duct','fillet','fin_set','flange','gusset','hole_array','import_step','inner_duct','motor_mount','nose_cone','pad','plate','pocket','servo_bay','slot')
def apply(name, shape, parameters, context=None):
    if name not in FEATURES: raise ValueError('Unknown feature '+str(name))
    return importlib.import_module(__name__+'.'+name).apply(shape,parameters,context or {})
