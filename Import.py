import subprocess
import sys

def install_package(package):
    """Install a package."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def import_package(package):
    """Import (or install) a package."""

    try:
        return __import__(package)
    except ImportError:
        install_package(package)
        return __import__(package)

def do_import(package, *parts):
    module = import_package(package)

    if parts is None:
        return module
    
    if isinstance(parts, str):
        parts = [parts]
    
    return [getattr(module, part) for part in parts]

