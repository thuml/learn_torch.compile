
# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_20_2(___stack0):
    return None
    mod.zero_grad(True)
    return None

def transformed___resume_at_20_2(___stack0):
    L = {"___stack0": ___stack0}
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_20_2(___stack0)

#============ end of __resume_at_20_2 ============#

def __guard_0_for_optimizer_zero_grad(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_type_id(L['self'], 17993008)) \
        and (___check_type_id(L['self'].optimizer, 85784752)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140136145107648))) \
        and (___compile_config_hash() == '035c3b2ca3ad5d2047352e5f69a8747e') \
        and (not ___needs_nopython())

def __transformed_code_0_for_optimizer_zero_grad(self, mod):
    return __resume_at_20_2(self.optimizer.zero_grad(True))


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def optimizer_zero_grad(self, mod):
    if self.optimizer is not None:
        self.optimizer.zero_grad(True)
        return None
    mod.zero_grad(True)
    return None

def transformed_optimizer_zero_grad(self, mod):
    L = {"self": self, "mod": mod}
    if __guard_0_for_optimizer_zero_grad(L):
        return __transformed_code_0_for_optimizer_zero_grad(self, mod)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return optimizer_zero_grad(self, mod)

#============ end of optimizer_zero_grad ============#
