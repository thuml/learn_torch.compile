
def __guard_0_for___resume_at_12_31(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_type_id(L['key'], 7605632)) \
        and (L['key'] == 'past_key_values') \
        and (___check_type_id(L['self'], 137555808)) \
        and (___check_type_id(L['value'], 7617760)) \
        and (len(L['value']) == 6) \
        and (___check_type_id(L['value'][0], 7617760)) \
        and (len(L['value'][0]) == 4) \
        and (___check_type_id(L['value'][1], 7617760)) \
        and (len(L['value'][1]) == 4) \
        and (___check_type_id(L['value'][2], 7617760)) \
        and (len(L['value'][2]) == 4) \
        and (___check_type_id(L['value'][3], 7617760)) \
        and (len(L['value'][3]) == 4) \
        and (___check_type_id(L['value'][4], 7617760)) \
        and (len(L['value'][4]) == 4) \
        and (___check_type_id(L['value'][5], 7617760)) \
        and (len(L['value'][5]) == 4) \
        and (___check_obj_id(L['__class__'], 117108432)) \
        and (L['self'].past_key_values[0][0] is L['value'][0][0]) \
        and (L['self'].past_key_values[0][1] is L['value'][0][1]) \
        and (L['self'].past_key_values[0][2] is L['value'][0][2]) \
        and (L['self'].past_key_values[0][3] is L['value'][0][3]) \
        and (L['self'].past_key_values[1][0] is L['value'][1][0]) \
        and (L['self'].past_key_values[1][1] is L['value'][1][1]) \
        and (L['self'].past_key_values[1][2] is L['value'][1][2]) \
        and (L['self'].past_key_values[1][3] is L['value'][1][3]) \
        and (L['self'].past_key_values[2][0] is L['value'][2][0]) \
        and (L['self'].past_key_values[2][1] is L['value'][2][1]) \
        and (L['self'].past_key_values[2][2] is L['value'][2][2]) \
        and (L['self'].past_key_values[2][3] is L['value'][2][3]) \
        and (L['self'].past_key_values[3][0] is L['value'][3][0]) \
        and (L['self'].past_key_values[3][1] is L['value'][3][1]) \
        and (L['self'].past_key_values[3][2] is L['value'][3][2]) \
        and (L['self'].past_key_values[3][3] is L['value'][3][3]) \
        and (L['self'].past_key_values[4][0] is L['value'][4][0]) \
        and (L['self'].past_key_values[4][1] is L['value'][4][1]) \
        and (L['self'].past_key_values[4][2] is L['value'][4][2]) \
        and (L['self'].past_key_values[4][3] is L['value'][4][3]) \
        and (L['self'].past_key_values[5][0] is L['value'][5][0]) \
        and (L['self'].past_key_values[5][1] is L['value'][5][1]) \
        and (L['self'].past_key_values[5][2] is L['value'][5][2]) \
        and (L['self'].past_key_values[5][3] is L['value'][5][3]) \
        and (___check_obj_id(L['self'].attentions, 7628576)) \
        and (___check_obj_id(L['self'].hidden_states, 7628576)) \
        and (___check_type_id(L['self'].past_key_values, 7617760)) \
        and (len(L['self'].past_key_values) == 6) \
        and (___check_obj_id(L['self'].cross_attentions, 7628576)) \
        and (hasattr(L['self'].last_hidden_state, '_dynamo_dynamic_indices') == False) \
        and (___check_type_id(L['self'].past_key_values[0], 7617760)) \
        and (len(L['self'].past_key_values[0]) == 4) \
        and (___check_type_id(L['self'].past_key_values[1], 7617760)) \
        and (len(L['self'].past_key_values[1]) == 4) \
        and (___check_type_id(L['self'].past_key_values[2], 7617760)) \
        and (len(L['self'].past_key_values[2]) == 4) \
        and (___check_type_id(L['self'].past_key_values[3], 7617760)) \
        and (len(L['self'].past_key_values[3]) == 4) \
        and (___check_type_id(L['self'].past_key_values[4], 7617760)) \
        and (len(L['self'].past_key_values[4]) == 4) \
        and (___check_type_id(L['self'].past_key_values[5], 7617760)) \
        and (len(L['self'].past_key_values[5]) == 4) \
        and (hasattr(L['self'].past_key_values[0][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['self'].past_key_values[0][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['self'].past_key_values[0][2], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['self'].past_key_values[0][3], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['self'].past_key_values[1][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['self'].past_key_values[1][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['self'].past_key_values[1][2], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['self'].past_key_values[1][3], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['self'].past_key_values[2][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['self'].past_key_values[2][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['self'].past_key_values[2][2], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['self'].past_key_values[2][3], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['self'].past_key_values[3][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['self'].past_key_values[3][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['self'].past_key_values[3][2], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['self'].past_key_values[3][3], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['self'].past_key_values[4][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['self'].past_key_values[4][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['self'].past_key_values[4][2], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['self'].past_key_values[4][3], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['self'].past_key_values[5][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['self'].past_key_values[5][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['self'].past_key_values[5][2], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['self'].past_key_values[5][3], '_dynamo_dynamic_indices') == False) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139628730179088))) \
        and (___compile_config_hash() == '72eb15e504422d051de1548d9fa37e9d') \
        and (not ___needs_nopython()) \
        and (___check_tensors(L['self'].last_hidden_state, L['self'].past_key_values[0][0], L['self'].past_key_values[0][1], L['self'].past_key_values[0][2], L['self'].past_key_values[0][3], L['self'].past_key_values[1][0], L['self'].past_key_values[1][1], L['self'].past_key_values[1][2], L['self'].past_key_values[1][3], L['self'].past_key_values[2][0], L['self'].past_key_values[2][1], L['self'].past_key_values[2][2], L['self'].past_key_values[2][3], L['self'].past_key_values[3][0], L['self'].past_key_values[3][1], L['self'].past_key_values[3][2], L['self'].past_key_values[3][3], L['self'].past_key_values[4][0], L['self'].past_key_values[4][1], L['self'].past_key_values[4][2], L['self'].past_key_values[4][3], L['self'].past_key_values[5][0], L['self'].past_key_values[5][1], L['self'].past_key_values[5][2], L['self'].past_key_values[5][3], tensor_check_names=tensor_check_names))

def __helper_outer_function():
    # this is a helper function to help compilers generate bytecode to read capture variables from closures, rather than reading values from global scope. The value of these variables does not matter, and will be determined in runtime.
    __class__ = None
    def __transformed_code_0_for___resume_at_12_31(___stack0, self, key, value):
        nonlocal __class__
        import importlib
        def __resume_at_16_32(___stack0):
            nonlocal __class__
            return None
        return __resume_at_16_32(importlib.import_module('builtins').super(
            __class__, self).__setattr__(key, value))


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_12_31(___stack0, self, key, value):
    nonlocal __class__
    super(__class__, self).__setattr__(key, value)
    return None

def transformed___resume_at_12_31(___stack0, self, key, value):
    L = {"___stack0": ___stack0, "self": self, "key": key, "value": value}
    if __guard_0_for___resume_at_12_31(L):
        return __transformed_code_0_for___resume_at_12_31(___stack0, self, key, value)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_12_31(___stack0, self, key, value)

#============ end of __resume_at_12_31 ============#
