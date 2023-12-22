
def __guard_0_for_resume_in_scaled_compute_loss(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (hasattr(L['___stack0'], '_dynamo_dynamic_indices') == False) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139651424778944))) \
        and (___compile_config_hash() == '019f57046a524d537e1c4be4975f6c7b') \
        and (___check_tensors(L['___stack0'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_7*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_7(*args, **kwargs):
    pass

def __transformed_code_0_for_resume_in_scaled_compute_loss(___stack0):
    pred = None; self = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    return __compiled_fn_7(___stack0)[0]


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_6_6(___stack0):
    return ___stack0 / 1000.0

def transformed___resume_at_6_6(___stack0):
    L = {"___stack0": ___stack0}
    if __guard_0_for_resume_in_scaled_compute_loss(L):
        return __transformed_code_0_for_resume_in_scaled_compute_loss(___stack0)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_6_6(___stack0)

#============ end of __resume_at_6_6 ============#

def __guard_0_for_scaled_compute_loss(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (hasattr(L['pred'], '_dynamo_dynamic_indices') == False) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139651424778944))) \
        and (___compile_config_hash() == '019f57046a524d537e1c4be4975f6c7b') \
        and (not ___needs_nopython()) \
        and (___check_tensors(L['pred'], tensor_check_names=tensor_check_names))

def __transformed_code_0_for_scaled_compute_loss(self, pred):
    return __resume_at_6_6(reduce_to_scalar_loss(pred))


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def scaled_compute_loss(self, pred):
    return reduce_to_scalar_loss(pred) / 1000.0

def transformed_scaled_compute_loss(self, pred):
    L = {"self": self, "pred": pred}
    if __guard_0_for_scaled_compute_loss(L):
        return __transformed_code_0_for_scaled_compute_loss(self, pred)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return scaled_compute_loss(self, pred)

#============ end of scaled_compute_loss ============#
