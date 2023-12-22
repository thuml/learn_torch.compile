
# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_22_1(___stack0, ___stack1):
    with ___stack0() as __temp_15:
        return ___stack1
    return None

def transformed___resume_at_22_1(___stack0, ___stack1):
    L = {"___stack0": ___stack0, "___stack1": ___stack1}
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_22_1(___stack0, ___stack1)

#============ end of __resume_at_22_1 ============#

def __guard_0_for_forward_pass(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['mod'], 140449776628928)) \
        and (L['mod'].training == False) \
        and (___check_type_id(L['self'], 157130688)) \
        and (___check_type_id(L['inputs'], 7638432)) \
        and (set(L['inputs'].keys()) == {'labels', 'input_ids'}) \
        and (___check_obj_id(L['self'].autocast, 25844800)) \
        and (hasattr(L['inputs']['input_ids'], '_dynamo_dynamic_indices') == False) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140446506655248))) \
        and (___compile_config_hash() == 'af3dff8b9274e598a0fa60276a5578a9') \
        and (not ___needs_nopython()) \
        and (___check_tensors(L['inputs']['input_ids'], tensor_check_names=tensor_check_names))

def __transformed_code_0_for_forward_pass(self, mod, inputs, collect_outputs):
    ___context_manager_0_0 = __import_contextlib.nullcontext()
    ___context_manager_0_0.__enter__()
    try:
        __temp_4 = mod(*(), **{'input_ids': inputs['input_ids'], 'labels':
            inputs['labels']})
    finally:
        ___context_manager_0_0.__exit__(None, None, None)
    return __resume_at_22_1(__import_contextlib.nullcontext, __temp_4)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def forward_pass(self, mod, inputs, collect_outputs):
    with self.autocast() as __temp_18:
        __temp_20 = {}
        __temp_20.update(inputs)
        return mod(*(), **__temp_20)
    return None

def transformed_forward_pass(self, mod, inputs, collect_outputs):
    L = {"self": self, "mod": mod, "inputs": inputs, "collect_outputs": collect_outputs}
    if __guard_0_for_forward_pass(L):
        return __transformed_code_0_for_forward_pass(self, mod, inputs, collect_outputs)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return forward_pass(self, mod, inputs, collect_outputs)

#============ end of forward_pass ============#
