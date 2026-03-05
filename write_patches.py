import site, os

site_dir = site.getsitepackages()[0]

# 1. sitecustomize.py - runs automatically on every Python start
sitecustomize = os.path.join(site_dir, 'sitecustomize.py')
with open(sitecustomize, 'w') as f:
    f.write("import builtins\n")
    f.write("builtins.unicode = str\n")
    f.write("builtins.basestring = str\n")
    f.write("builtins.long = int\n")
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
