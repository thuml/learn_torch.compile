
# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_6_30(self, past_key_values, hidden_states, attentions, cross_attentions):
    self.past_key_values = past_key_values
    self.hidden_states = hidden_states
    self.attentions = attentions
    self.cross_attentions = cross_attentions
    self.__post_init__()
    return None

def transformed___resume_at_6_30(self, past_key_values, hidden_states, attentions, cross_attentions):
    L = {"self": self, "past_key_values": past_key_values, "hidden_states": hidden_states, "attentions": attentions, "cross_attentions": cross_attentions}
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_6_30(self, past_key_values, hidden_states, attentions, cross_attentions)

#============ end of __resume_at_6_30 ============#

def __guard_1_for___init__(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_type_id(L['self'], 132368144)) \
        and (___check_obj_id(L['self'].attentions, 7628576)) \
        and (hasattr(L['last_hidden_state'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['self'].hidden_states, 7628576)) \
        and (___check_obj_id(L['self'].past_key_values, 7628576)) \
        and (___check_obj_id(L['self'].cross_attentions, 7628576)) \
        and (___check_obj_id(L['self'].last_hidden_state, 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139886168940048))) \
        and (___compile_config_hash() == '07b52bff0c86f8e311705687144327df') \
        and (not ___needs_nopython()) \
        and (___check_tensors(L['last_hidden_state'], tensor_check_names=tensor_check_names))

def __transformed_code_1_for___init__(self, last_hidden_state, past_key_values, hidden_states, attentions, cross_attentions):
    self.last_hidden_state = last_hidden_state
    return __resume_at_6_30(self, past_key_values, hidden_states, attentions,
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
    if __guard_1_for___init__(L):
        return __transformed_code_1_for___init__(self, last_hidden_state, past_key_values, hidden_states, attentions, cross_attentions)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __init__(self, last_hidden_state, past_key_values, hidden_states, attentions, cross_attentions)

#============ end of __init__ ============#
