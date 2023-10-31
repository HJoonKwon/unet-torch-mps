"""
%run -i start.py
Forked from https://github.com/daeyun/dshin
"""
import sys

import ipykernel
from IPython import display as IPy_display

from os import path


def __extend_sys_path(prepend_paths=(), append_paths=()):
    pathset = set(sys.path)
    for p in prepend_paths:
        fullpath = path.realpath(path.expanduser(p))
        if fullpath not in pathset:
            sys.path.insert(0, fullpath)
    for p in append_paths:
        fullpath = path.realpath(path.expanduser(p))
        if fullpath not in pathset:
            sys.path.append(fullpath)
    sys.path = [p for p in sys.path if p]


__extend_sys_path(
    prepend_paths=[],
    append_paths=[path.realpath(path.join(__file__, '../../python'))],
)

# Default imports.
import re
import time
import random
import collections
import argparse
import json
import glob
import gzip
import pickle
import shutil
import logging
import copy
import pprint
import os
import importlib
import typing
import hashlib
import warnings
from pathlib import Path
from typing import Sequence, Tuple, Union, Mapping, MutableMapping, List, Any, ByteString, AnyStr, Dict

from IPython import get_ipython

ipython = get_ipython()

ipython.run_line_magic('reload_ext', 'autoreload')
ipython.run_line_magic('autoreload', '2')

from IPython.core.magic import (register_line_magic)
from IPython import Application
from IPython.core.interactiveshell import InteractiveShell


@register_line_magic
def restart(line):
    app = Application.instance()
    app.kernel.do_shutdown(True)


def import_optional(module_name, alias=None, is_verbose=False):
    if alias is None:
        alias = module_name
    try:
        # globals()[alias] = __import__(module_name)
        globals()[alias] = importlib.import_module(module_name)
        ipython.run_line_magic('aimport', '-{}'.format(module_name))

    except ImportError:
        if is_verbose:
            print('Optional dependency {} could not be imported.'.format(module_name), file=sys.stderr)


import_optional('tqdm')
import_optional('torch')
import_optional('cv2')
import_optional('imageio')
import_optional('matplotlib.pyplot', alias='pt')
import_optional('matplotlib', alias='mpl')
import_optional('numpy', alias='np')
import_optional('numpy.linalg', alias='la')