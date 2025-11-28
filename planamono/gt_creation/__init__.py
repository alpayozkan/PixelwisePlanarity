"""
Planar segmentation algorithms and post-processing.
"""

import pkgutil
import inspect

# Expose child modules
__all__ = [module.name for module in pkgutil.iter_modules(__path__)]

def __getattr__(name):
    # 1) Access to submodules (segmentation.plan2seg etc.)
    if name in __all__:
        module = __import__(f"{__name__}.{name}", fromlist=[name])
        globals()[name] = module
        return module

    # 2) Access to functions/classes inside submodules
    for _, module_name, _ in pkgutil.iter_modules(__path__):
        module = __import__(f"{__name__}.{module_name}", fromlist=[module_name])
        if hasattr(module, name):
            attr = getattr(module, name)
            globals()[name] = attr  # cache attribute
            return attr

    raise AttributeError(f"module {__name__} has no attribute {name}")
