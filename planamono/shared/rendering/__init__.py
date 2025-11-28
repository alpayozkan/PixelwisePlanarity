import pkgutil

__all__ = [module.name for module in pkgutil.iter_modules(__path__)]

def __getattr__(name):
    if name in __all__:
        module = __import__(f"{__name__}.{name}", fromlist=[name])
        globals()[name] = module
        return module
    raise AttributeError(name)
