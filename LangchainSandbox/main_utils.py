import json


def as_json(dictionary: dict):
    return json.dump(dictionary, indent=4, ensure_ascii=True)
