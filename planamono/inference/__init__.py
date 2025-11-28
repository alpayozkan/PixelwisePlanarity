__all__ = ["planarity", "segmentation"]

def __getattr__(name):
    if name in __all__:
        module = __import__(f"{__name__}.{name}", fromlist=[name])
        globals()[name] = module  # cache for next time
        return module
    raise AttributeError(name)