
# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_34_2(___stack0):
    mod.zero_grad(True)
    return None

def transformed___resume_at_34_2(___stack0):
    L = {"___stack0": ___stack0}
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_34_2(___stack0)

#============ end of __resume_at_34_2 ============#

def __guard_0_for_optimizer_zero_grad(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['mod'], 140657084546784)) \
        and (L['mod'].training == True) \
        and (___check_type_id(L['self'], 41127664)) \
        and (___check_obj_id(L['self'].optimizer, 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140653856701120))) \
        and (___compile_config_hash() == 'd16bd24b7da1206ec662b6f86a0149e6') \
        and (not ___needs_nopython())

def __transformed_code_0_for_optimizer_zero_grad(self, mod):
    return __resume_at_34_2(mod.zero_grad(True))


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
