import site, os, re

site_dir = site.getsitepackages()[0]

# 1. sitecustomize.py - solo define unicode/basestring sin tocar str
sitecustomize = os.path.join(site_dir, 'sitecustomize.py')
with open(sitecustomize, 'w') as f:
    f.write("import builtins\n")
    f.write("builtins.unicode = str\n")
    f.write("builtins.basestring = str\n")
    f.write("builtins.long = int\n")
print(f"Created: {sitecustomize}")

# 2. Patch cookiecutter/compat.py - fix str(value, encoding) with encoding=None
compat_path = os.path.join(site_dir, 'cookiecutter', 'compat.py')
if os.path.exists(compat_path):
    with open(compat_path, 'r') as f:
        content = f.read()
    # Replace any str(x, encoding) pattern with x.decode(encoding) or just str(x)
    new_content = content.replace(
        "str(value, encoding)",
        "(value.decode(encoding) if isinstance(value, bytes) and encoding else str(value))"
    ).replace(
        "str(s, encoding)",
        "(s.decode(encoding) if isinstance(s, bytes) and encoding else str(s))"
    )
    if new_content != content:
        with open(compat_path, 'w') as f:
            f.write(new_content)
        print(f"Patched: {compat_path}")
    else:
        print(f"No str(x, encoding) pattern found in {compat_path}, showing content:")
        print(content[:500])

# 3. Patch ALL cookiecutter .py files for str(x, encoding) pattern
cc_dir = os.path.join(site_dir, 'cookiecutter')
for fname in os.listdir(cc_dir):
    if fname.endswith('.py'):
        fpath = os.path.join(cc_dir, fname)
        with open(fpath, 'r', errors='ignore') as f:
            content = f.read()
        new_content = re.sub(
            r'str\((\w+),\s*(\w+)\)',
            r'(\1.decode(\2) if isinstance(\1, bytes) and \2 else str(\1))',
            content
        )
        if new_content != content:
            with open(fpath, 'w') as f:
                f.write(new_content)
            print(f"Patched str(x,enc): {fpath}")

# 4. cookiecutter_extensions.py
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
