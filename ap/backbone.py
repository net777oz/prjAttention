from importlib import import_module

_REG = {}  # name -> (module_path, symbol)

def register(name: str, module_path: str, symbol: str = "build_model"):
    if name in _REG:
        raise KeyError(f"Backbone already registered: {name}")
    _REG[name] = (module_path, symbol)

def build_backbone(name: str, **kwargs):
    if name not in _REG:
        raise KeyError(f"Unknown backbone '{name}'. Available: {sorted(_REG)}")
    mod_path, sym = _REG[name]
    mod = import_module(mod_path)
    fn = getattr(mod, sym)
    return fn(**kwargs)

def available():
    return sorted(_REG)
