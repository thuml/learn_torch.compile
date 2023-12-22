
# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_14_21(self, hidden_states, attentions, cross_attentions):
    self.hidden_states = hidden_states
    self.attentions = attentions
    self.cross_attentions = cross_attentions
    self.__post_init__()
    return None

def transformed___resume_at_14_21(self, hidden_states, attentions, cross_attentions):
    L = {"self": self, "hidden_states": hidden_states, "attentions": attentions, "cross_attentions": cross_attentions}
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_14_21(self, hidden_states, attentions, cross_attentions)

#============ end of __resume_at_14_21 ============#

def __guard_0_for_resume_in___init__(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_type_id(L['self'], 140452336)) \
        and (___check_type_id(L['past_key_values'], 7617760)) \
        and (len(L['past_key_values']) == 12) \
        and (___check_obj_id(L['self'].attentions, 7628576)) \
        and (___check_type_id(L['past_key_values'][0], 7617760)) \
        and (len(L['past_key_values'][0]) == 2) \
        and (___check_type_id(L['past_key_values'][1], 7617760)) \
        and (len(L['past_key_values'][1]) == 2) \
        and (___check_type_id(L['past_key_values'][2], 7617760)) \
        and (len(L['past_key_values'][2]) == 2) \
        and (___check_type_id(L['past_key_values'][3], 7617760)) \
        and (len(L['past_key_values'][3]) == 2) \
        and (___check_type_id(L['past_key_values'][4], 7617760)) \
        and (len(L['past_key_values'][4]) == 2) \
        and (___check_type_id(L['past_key_values'][5], 7617760)) \
        and (len(L['past_key_values'][5]) == 2) \
        and (___check_type_id(L['past_key_values'][6], 7617760)) \
        and (len(L['past_key_values'][6]) == 2) \
        and (___check_type_id(L['past_key_values'][7], 7617760)) \
        and (len(L['past_key_values'][7]) == 2) \
        and (___check_type_id(L['past_key_values'][8], 7617760)) \
        and (len(L['past_key_values'][8]) == 2) \
        and (___check_type_id(L['past_key_values'][9], 7617760)) \
        and (len(L['past_key_values'][9]) == 2) \
        and (___check_obj_id(L['self'].hidden_states, 7628576)) \
        and (___check_type_id(L['past_key_values'][10], 7617760)) \
        and (len(L['past_key_values'][10]) == 2) \
        and (___check_type_id(L['past_key_values'][11], 7617760)) \
        and (len(L['past_key_values'][11]) == 2) \
        and (___check_obj_id(L['self'].past_key_values, 7628576)) \
        and (hasattr(L['past_key_values'][0][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['past_key_values'][0][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['past_key_values'][1][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['past_key_values'][1][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['past_key_values'][2][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['past_key_values'][2][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['past_key_values'][3][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['past_key_values'][3][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['past_key_values'][4][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['past_key_values'][4][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['past_key_values'][5][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['past_key_values'][5][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['past_key_values'][6][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['past_key_values'][6][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['past_key_values'][7][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['past_key_values'][7][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['past_key_values'][8][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['past_key_values'][8][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['past_key_values'][9][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['past_key_values'][9][1], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['self'].cross_attentions, 7628576)) \
        and (hasattr(L['past_key_values'][10][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['past_key_values'][10][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['past_key_values'][11][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['past_key_values'][11][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['self'].last_hidden_state, '_dynamo_dynamic_indices') == False) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139844393524752))) \
        and (___compile_config_hash() == 'c159a66ee23ff013ad86792a2f0986b0') \
        and (not ___needs_nopython()) \
        and (___check_tensors(L['past_key_values'][0][0], L['past_key_values'][0][1], L['past_key_values'][1][0], L['past_key_values'][1][1], L['past_key_values'][2][0], L['past_key_values'][2][1], L['past_key_values'][3][0], L['past_key_values'][3][1], L['past_key_values'][4][0], L['past_key_values'][4][1], L['past_key_values'][5][0], L['past_key_values'][5][1], L['past_key_values'][6][0], L['past_key_values'][6][1], L['past_key_values'][7][0], L['past_key_values'][7][1], L['past_key_values'][8][0], L['past_key_values'][8][1], L['past_key_values'][9][0], L['past_key_values'][9][1], L['past_key_values'][10][0], L['past_key_values'][10][1], L['past_key_values'][11][0], L['past_key_values'][11][1], L['self'].last_hidden_state, tensor_check_names=tensor_check_names))

def __transformed_code_0_for_resume_in___init__(self, past_key_values, hidden_states, attentions, cross_attentions):
    last_hidden_state = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    self.past_key_values = past_key_values
    return __resume_at_14_21(self, hidden_states, attentions, cross_attentions)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_6_20(self, past_key_values, hidden_states, attentions, cross_attentions):
    self.past_key_values = past_key_values
    self.hidden_states = hidden_states
    self.attentions = attentions
    self.cross_attentions = cross_attentions
    self.__post_init__()
    return None

def transformed___resume_at_6_20(self, past_key_values, hidden_states, attentions, cross_attentions):
    L = {"self": self, "past_key_values": past_key_values, "hidden_states": hidden_states, "attentions": attentions, "cross_attentions": cross_attentions}
    if __guard_0_for_resume_in___init__(L):
        return __transformed_code_0_for_resume_in___init__(self, past_key_values, hidden_states, attentions, cross_attentions)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_6_20(self, past_key_values, hidden_states, attentions, cross_attentions)

#============ end of __resume_at_6_20 ============#

def __guard_0_for___init__(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_type_id(L['self'], 140452336)) \
        and (___check_obj_id(L['self'].attentions, 7628576)) \
        and (hasattr(L['last_hidden_state'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['self'].hidden_states, 7628576)) \
        and (___check_obj_id(L['self'].past_key_values, 7628576)) \
        and (___check_obj_id(L['self'].cross_attentions, 7628576)) \
        and (___check_obj_id(L['self'].last_hidden_state, 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139844393524752))) \
        and (___compile_config_hash() == 'c159a66ee23ff013ad86792a2f0986b0') \
        and (not ___needs_nopython()) \
        and (___check_tensors(L['last_hidden_state'], tensor_check_names=tensor_check_names))

def __transformed_code_0_for___init__(self, last_hidden_state, past_key_values, hidden_states, attentions, cross_attentions):
    self.last_hidden_state = last_hidden_state
    return __resume_at_6_20(self, past_key_values, hidden_states, attentions,
        cross_attentions)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __init__(self, last_hidden_state, past_key_values, hidden_states, attentions, cross_attentions):
    self.last_hidden_state = last_hidden_state
    self.past_key_values = past_key_values
    self.hidden_states = hidden_states
    self.attentions = attentions
    self.cross_attentions = cross_attentions
    self.__post_init__()
    return None

def transformed___init__(self, last_hidden_state, past_key_values, hidden_states, attentions, cross_attentions):
    L = {"self": self, "last_hidden_state": last_hidden_state, "past_key_values": past_key_values, "hidden_states": hidden_states, "attentions": attentions, "cross_attentions": cross_attentions}
    if __guard_0_for___init__(L):
        return __transformed_code_0_for___init__(self, last_hidden_state, past_key_values, hidden_states, attentions, cross_attentions)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __init__(self, last_hidden_state, past_key_values, hidden_states, attentions, cross_attentions)

#============ end of __init__ ============#
