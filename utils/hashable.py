import hashlib


def hashable(cls):
    def __hash__(self):
        return hashlib.md5(repr(self).encode("utf-8")).hexdigest()

    def hash_extra(self, extra=""):
        full_str = repr(self) + extra
        return hashlib.md5(full_str.encode("utf-8")).hexdigest()

    cls.__hash__ = __hash__
    cls.hash_extra = hash_extra
    return cls
