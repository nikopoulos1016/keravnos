import os, sys

# handle DLL loading (Windows only)
curr_dir = os.path.dirname(__file__)
lib_dir = os.path.join(curr_dir, 'lib')
if os.name == 'nt' and os.path.isdir(lib_dir):
  os.add_dll_directory(lib_dir)

# import public interface from api.py
from .api import Transformer
  