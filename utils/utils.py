import json


def load_json(src):
    with open(src, encoding="utf-8") as frd:
        return json.load(frd)
    
    
def save_json(obj, tgt, indent=None):
    with open(tgt, "w") as fwt:
        json.dump(obj, fwt, ensure_ascii=False, indent=indent)
