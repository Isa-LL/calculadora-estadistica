import site, os

site_dir = site.getsitepackages()[0]

# 1. sitecustomize.py - runs automatically on every Python start
sitecustomize = os.path.join(site_dir, 'sitecustomize.py')
with open(sitecustomize, 'w') as f:
    f.write("""import builtins

builtins.unicode = str
builtins.basestring = str
builtins.long = int

# Fix str(value, encoding) called with encoding=None (Python 2 pattern)
_orig_str = builtins.str
class _compat_str(_orig_str):
    def __new__(cls, obj=b'', encoding=None, errors='strict'):
        if isinstance(obj, bytes) and encoding is not None:
            return _orig_str.__new__(cls, obj, encoding, errors)
        return _orig_str.__new__(cls, obj)

builtins.str = _compat_str
""")
print(f"Created: {sitecustomize}")

# 2. cookiecutter_extensions.py - required by flet build template
ext_path = os.path.join(site_dir, 'cookiecutter_extensions.py')
with open(ext_path, 'w') as f:
    f.write("""from jinja2.ext import Extension

class FletExtension(Extension):
    def __init__(self, environment):
        super().__init__(environment)
        environment.filters["to_cmake_list"] = lambda v: ";".join(v) if v else ""
        environment.filters["to_android_dependencies"] = lambda v: "" if not v else "\\n".join(
            ["    implementation '" + d + "'" for d in v]
        )
        environment.filters["to_plist_array"] = lambda v: "" if not v else "\\n".join(
            ["    <string>" + i + "</string>" for i in v]
        )
        environment.globals["get_pyproject"] = lambda *a, **kw: {}
        environment.filters["get_pyproject"] = lambda v, *a, **kw: {}
        environment.globals["to_toml"] = lambda v: ""
        environment.filters["to_toml"] = lambda v: ""
""")
print(f"Created: {ext_path}")
print("All patches applied successfully!")
