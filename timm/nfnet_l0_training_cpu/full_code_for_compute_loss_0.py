
def __guard_0_for_compute_loss(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (hasattr(L['pred'], '_dynamo_dynamic_indices') == False) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140491160173248))) \
        and (___compile_config_hash() == '784a8fc242bcfb5c87f54c5f0b812226') \
        and (not ___needs_nopython()) \
        and (___check_tensors(L['pred'], tensor_check_names=tensor_check_names))

def __transformed_code_0_for_compute_loss(self, pred):
    return reduce_to_scalar_loss(pred)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def compute_loss(self, pred):
    return reduce_to_scalar_loss(pred)

def transformed_compute_loss(self, pred):
    L = {"self": self, "pred": pred}
    if __guard_0_for_compute_loss(L):
        return __transformed_code_0_for_compute_loss(self, pred)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return compute_loss(self, pred)

#============ end of compute_loss ============#
