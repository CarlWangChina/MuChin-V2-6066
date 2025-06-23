import importlib.util

def is_package_installed(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None