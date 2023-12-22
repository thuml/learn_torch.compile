
# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __setitem__(self, key, value):
    nonlocal __class__
    super().__setitem__(key, value)
    super().__setattr__(key, value)
    return None

def transformed___setitem__(self, key, value):
    L = {"self": self, "key": key, "value": value}
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __setitem__(self, key, value)

#============ end of __setitem__ ============#
