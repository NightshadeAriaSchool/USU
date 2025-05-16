import subprocess
import sys

def install_package(package):
    """Install a package via pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def import_package(package):
    """Try to import a package; install it if not present."""
    try:
        return __import__(package)
    except ImportError:
        install_package(package)
        return __import__(package)

def do_import(package, *parts):
    """Dynamically import a module or objects from it.

    Args:
        package (str): Name of the package/module.
        *parts (str): Specific objects to import from the module.

    Returns:
        module or list of objects from the module.
    """
    module = import_package(package)

    if not parts:
        return module

    # Single item? Return just the object, not a list.
    if len(parts) == 1:
        return getattr(module, parts[0])

    return [getattr(module, part) for part in parts]