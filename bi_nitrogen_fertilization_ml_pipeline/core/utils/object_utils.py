import functools


def nested_getattr(obj, attr, *args):
    def _getattr(obj_, attr_):
        return getattr(obj_, attr_, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def nested_setattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(nested_getattr(obj, pre) if pre else obj, post, val)
