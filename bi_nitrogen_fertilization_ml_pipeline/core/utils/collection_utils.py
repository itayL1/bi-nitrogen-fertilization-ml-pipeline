from typing import Callable


def filter_dict(
    dict_: dict,
    entry_predicate: Callable[[object, object], bool],
) -> dict:
    return {
        key: val
        for key, val in dict_.items()
        if entry_predicate(key, val)
    }


def set_new_dict_entry(
    dict_: dict, key, val,
) -> None:
    assert key not in dict_,\
        f"the provided key '{key}' already present in the provided dict"
    dict_[key] = val
