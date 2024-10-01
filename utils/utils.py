import json
import logging
import importlib.metadata
import importlib.util
import traceback

import torch
import torch.distributed as dist


logger = logging.getLogger(__name__)


def load_json(src):
    with open(src, encoding="utf-8") as frd:
        return json.load(frd)
    
    
def save_json(obj, tgt, indent=None):
    with open(tgt, "w") as fwt:
        json.dump(obj, fwt, ensure_ascii=False, indent=indent)


def is_package_available(name):
    return importlib.util.find_spec(name) is not None


def get_version(name):
    """get version of a package"""

    try:
        return importlib.metadata.version(name)
    except:
        return None


def is_rank_zero():
    return not dist.is_initialized() or dist.get_rank() == 0


def is_dist_initialized():
    return dist.is_initialized()
