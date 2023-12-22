
def __guard_0_for___resume_at_12_49(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_type_id(L['key'], 7605632)) \
        and (L['key'] == 'encoder_last_hidden_state') \
        and (___check_type_id(L['self'], 146540416)) \
        and (L['self'].encoder_last_hidden_state is L['value']) \
        and (___check_obj_id(L['__class__'], 125960464)) \
        and (___check_obj_id(L['self'].past_key_values, 7628576)) \
        and (___check_obj_id(L['self'].cross_attentions, 7628576)) \
        and (hasattr(L['self'].last_hidden_state, '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['self'].decoder_attentions, 7628576)) \
        and (___check_obj_id(L['self'].encoder_attentions, 7628576)) \
        and (___check_obj_id(L['self'].decoder_hidden_states, 7628576)) \
        and (___check_obj_id(L['self'].encoder_hidden_states, 7628576)) \
        and (hasattr(L['self'].encoder_last_hidden_state, '_dynamo_dynamic_indices') == False) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139712823664144))) \
        and (___compile_config_hash() == 'ef53a1e5e652702a41f4e7cf661cd065') \
        and (not ___needs_nopython()) \
        and (___check_tensors(L['self'].last_hidden_state, L['self'].encoder_last_hidden_state, tensor_check_names=tensor_check_names))

def __helper_outer_function():
    # this is a helper function to help compilers generate bytecode to read capture variables from closures, rather than reading values from global scope. The value of these variables does not matter, and will be determined in runtime.
    __class__ = None
    def __transformed_code_0_for___resume_at_12_49(___stack0, self, key, value):
        nonlocal __class__
        import importlib
        def __resume_at_16_50(___stack0):
            nonlocal __class__
            return None
        return __resume_at_16_50(importlib.import_module('builtins').super(
            __class__, self).__setattr__(key, value))


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_12_49(___stack0, self, key, value):
    nonlocal __class__
    super(__class__, self).__setattr__(key, value)
    return None

def transformed___resume_at_12_49(___stack0, self, key, value):
    L = {"___stack0": ___stack0, "self": self, "key": key, "value": value}
    if __guard_0_for___resume_at_12_49(L):
        return __transformed_code_0_for___resume_at_12_49(___stack0, self, key, value)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_12_49(___stack0, self, key, value)

#============ end of __resume_at_12_49 ============#
