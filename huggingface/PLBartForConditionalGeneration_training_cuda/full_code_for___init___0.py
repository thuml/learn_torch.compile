
# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_6_15(self, hidden_states, attentions):
    self.hidden_states = hidden_states
    self.attentions = attentions
    self.__post_init__()
    return None

def transformed___resume_at_6_15(self, hidden_states, attentions):
    L = {"self": self, "hidden_states": hidden_states, "attentions": attentions}
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_6_15(self, hidden_states, attentions)

#============ end of __resume_at_6_15 ============#

def __guard_0_for___init__(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_type_id(L['self'], 132585360)) \
        and (___check_obj_id(L['self'].attentions, 7628576)) \
        and (hasattr(L['last_hidden_state'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['self'].hidden_states, 7628576)) \
        and (___check_obj_id(L['self'].last_hidden_state, 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140382235860496))) \
        and (___compile_config_hash() == 'f0354ca969ec95d221ebd09c87295c79') \
        and (not ___needs_nopython()) \
        and (___check_tensors(L['last_hidden_state'], tensor_check_names=tensor_check_names))

def __transformed_code_0_for___init__(self, last_hidden_state, hidden_states, attentions):
    self.last_hidden_state = last_hidden_state
    return __resume_at_6_15(self, hidden_states, attentions)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __init__(self, last_hidden_state, hidden_states, attentions):
    self.last_hidden_state = last_hidden_state
    self.hidden_states = hidden_states
    self.attentions = attentions
    self.__post_init__()
    return None

def transformed___init__(self, last_hidden_state, hidden_states, attentions):
    L = {"self": self, "last_hidden_state": last_hidden_state, "hidden_states": hidden_states, "attentions": attentions}
    if __guard_0_for___init__(L):
        return __transformed_code_0_for___init__(self, last_hidden_state, hidden_states, attentions)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __init__(self, last_hidden_state, hidden_states, attentions)

#============ end of __init__ ============#