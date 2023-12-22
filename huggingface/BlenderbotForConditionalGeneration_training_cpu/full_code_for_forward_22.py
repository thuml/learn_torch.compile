
def __guard_29_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 140276879975392)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['use_cache'], 7677632)) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['attention_mask'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['past_key_value'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (hasattr(L['encoder_hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['encoder_attention_mask'], 7628576)) \
        and (___check_obj_id(L['cross_attn_layer_head_mask'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140273611202064))) \
        and (___compile_config_hash() == 'b861ac09b147974d65ae8b0b2b040ad0') \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[4], 7677632)) \
        and (___check_tensors(L['hidden_states'], L['attention_mask'], L['encoder_hidden_states'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_39*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_39(*args, **kwargs):
    pass

def __transformed_code_29_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    cross_attn_past_key_value = None; cross_attn_present_key_value = None; cross_attn_weights = None; outputs = None; present_key_value = None; residual = None; self_attn_past_key_value = None; self_attn_weights = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_39(hidden_states, attention_mask,
        encoder_hidden_states)
    return graph_out_0[0],


def __guard_28_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 140276879981968)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['use_cache'], 7677632)) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['attention_mask'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['past_key_value'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (hasattr(L['encoder_hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['encoder_attention_mask'], 7628576)) \
        and (___check_obj_id(L['cross_attn_layer_head_mask'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140273611202064))) \
        and (___compile_config_hash() == 'b861ac09b147974d65ae8b0b2b040ad0') \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[4], 7677632)) \
        and (___check_tensors(L['hidden_states'], L['attention_mask'], L['encoder_hidden_states'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_38*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_38(*args, **kwargs):
    pass

def __transformed_code_28_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    cross_attn_past_key_value = None; cross_attn_present_key_value = None; cross_attn_weights = None; outputs = None; present_key_value = None; residual = None; self_attn_past_key_value = None; self_attn_weights = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_38(hidden_states, attention_mask,
        encoder_hidden_states)
    return graph_out_0[0],


def __guard_27_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 140276879978032)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['use_cache'], 7677632)) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['attention_mask'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['past_key_value'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (hasattr(L['encoder_hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['encoder_attention_mask'], 7628576)) \
        and (___check_obj_id(L['cross_attn_layer_head_mask'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140273611202064))) \
        and (___compile_config_hash() == 'b861ac09b147974d65ae8b0b2b040ad0') \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[4], 7677632)) \
        and (___check_tensors(L['hidden_states'], L['attention_mask'], L['encoder_hidden_states'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_37*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_37(*args, **kwargs):
    pass

def __transformed_code_27_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    cross_attn_past_key_value = None; cross_attn_present_key_value = None; cross_attn_weights = None; outputs = None; present_key_value = None; residual = None; self_attn_past_key_value = None; self_attn_weights = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_37(hidden_states, attention_mask,
        encoder_hidden_states)
    return graph_out_0[0],


def __guard_26_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 140276879974624)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['use_cache'], 7677632)) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['attention_mask'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['past_key_value'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (hasattr(L['encoder_hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['encoder_attention_mask'], 7628576)) \
        and (___check_obj_id(L['cross_attn_layer_head_mask'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140273611202064))) \
        and (___compile_config_hash() == 'b861ac09b147974d65ae8b0b2b040ad0') \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[4], 7677632)) \
        and (___check_tensors(L['hidden_states'], L['attention_mask'], L['encoder_hidden_states'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_36*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_36(*args, **kwargs):
    pass

def __transformed_code_26_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    cross_attn_past_key_value = None; cross_attn_present_key_value = None; cross_attn_weights = None; outputs = None; present_key_value = None; residual = None; self_attn_past_key_value = None; self_attn_weights = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_36(hidden_states, attention_mask,
        encoder_hidden_states)
    return graph_out_0[0],


def __guard_25_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 140276879970880)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['use_cache'], 7677632)) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['attention_mask'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['past_key_value'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (hasattr(L['encoder_hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['encoder_attention_mask'], 7628576)) \
        and (___check_obj_id(L['cross_attn_layer_head_mask'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140273611202064))) \
        and (___compile_config_hash() == 'b861ac09b147974d65ae8b0b2b040ad0') \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[4], 7677632)) \
        and (___check_tensors(L['hidden_states'], L['attention_mask'], L['encoder_hidden_states'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_35*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_35(*args, **kwargs):
    pass

def __transformed_code_25_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    cross_attn_past_key_value = None; cross_attn_present_key_value = None; cross_attn_weights = None; outputs = None; present_key_value = None; residual = None; self_attn_past_key_value = None; self_attn_weights = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_35(hidden_states, attention_mask,
        encoder_hidden_states)
    return graph_out_0[0],


def __guard_24_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 140276879967136)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['use_cache'], 7677632)) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['attention_mask'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['past_key_value'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (hasattr(L['encoder_hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['encoder_attention_mask'], 7628576)) \
        and (___check_obj_id(L['cross_attn_layer_head_mask'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140273611202064))) \
        and (___compile_config_hash() == 'b861ac09b147974d65ae8b0b2b040ad0') \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[4], 7677632)) \
        and (___check_tensors(L['hidden_states'], L['attention_mask'], L['encoder_hidden_states'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_34*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_34(*args, **kwargs):
    pass

def __transformed_code_24_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    cross_attn_past_key_value = None; cross_attn_present_key_value = None; cross_attn_weights = None; outputs = None; present_key_value = None; residual = None; self_attn_past_key_value = None; self_attn_weights = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_34(hidden_states, attention_mask,
        encoder_hidden_states)
    return graph_out_0[0],


def __guard_23_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 140276881456768)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['use_cache'], 7677632)) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['attention_mask'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['past_key_value'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (hasattr(L['encoder_hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['encoder_attention_mask'], 7628576)) \
        and (___check_obj_id(L['cross_attn_layer_head_mask'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140273611202064))) \
        and (___compile_config_hash() == 'b861ac09b147974d65ae8b0b2b040ad0') \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[4], 7677632)) \
        and (___check_tensors(L['hidden_states'], L['attention_mask'], L['encoder_hidden_states'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_33*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_33(*args, **kwargs):
    pass

def __transformed_code_23_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    cross_attn_past_key_value = None; cross_attn_present_key_value = None; cross_attn_weights = None; outputs = None; present_key_value = None; residual = None; self_attn_past_key_value = None; self_attn_weights = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_33(hidden_states, attention_mask,
        encoder_hidden_states)
    return graph_out_0[0],


def __guard_22_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 140276881455232)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['use_cache'], 7677632)) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['attention_mask'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['past_key_value'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (hasattr(L['encoder_hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['encoder_attention_mask'], 7628576)) \
        and (___check_obj_id(L['cross_attn_layer_head_mask'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140273611202064))) \
        and (___compile_config_hash() == 'b861ac09b147974d65ae8b0b2b040ad0') \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[4], 7677632)) \
        and (___check_tensors(L['hidden_states'], L['attention_mask'], L['encoder_hidden_states'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_32*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_32(*args, **kwargs):
    pass

def __transformed_code_22_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    cross_attn_past_key_value = None; cross_attn_present_key_value = None; cross_attn_weights = None; outputs = None; present_key_value = None; residual = None; self_attn_past_key_value = None; self_attn_weights = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_32(hidden_states, attention_mask,
        encoder_hidden_states)
    return graph_out_0[0],


def __guard_21_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 140276881451872)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['use_cache'], 7677632)) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['attention_mask'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['past_key_value'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (hasattr(L['encoder_hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['encoder_attention_mask'], 7628576)) \
        and (___check_obj_id(L['cross_attn_layer_head_mask'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140273611202064))) \
        and (___compile_config_hash() == 'b861ac09b147974d65ae8b0b2b040ad0') \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[4], 7677632)) \
        and (___check_tensors(L['hidden_states'], L['attention_mask'], L['encoder_hidden_states'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_31*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_31(*args, **kwargs):
    pass

def __transformed_code_21_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    cross_attn_past_key_value = None; cross_attn_present_key_value = None; cross_attn_weights = None; outputs = None; present_key_value = None; residual = None; self_attn_past_key_value = None; self_attn_weights = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_31(hidden_states, attention_mask,
        encoder_hidden_states)
    return graph_out_0[0],


def __guard_20_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 140276881451392)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['use_cache'], 7677632)) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['attention_mask'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['past_key_value'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (hasattr(L['encoder_hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['encoder_attention_mask'], 7628576)) \
        and (___check_obj_id(L['cross_attn_layer_head_mask'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140273611202064))) \
        and (___compile_config_hash() == 'b861ac09b147974d65ae8b0b2b040ad0') \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[4], 7677632)) \
        and (___check_tensors(L['hidden_states'], L['attention_mask'], L['encoder_hidden_states'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_30*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_30(*args, **kwargs):
    pass

def __transformed_code_20_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    cross_attn_past_key_value = None; cross_attn_present_key_value = None; cross_attn_weights = None; outputs = None; present_key_value = None; residual = None; self_attn_past_key_value = None; self_attn_weights = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_30(hidden_states, attention_mask,
        encoder_hidden_states)
    return graph_out_0[0],


def __guard_19_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 140276881450720)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['use_cache'], 7677632)) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['attention_mask'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['past_key_value'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (hasattr(L['encoder_hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['encoder_attention_mask'], 7628576)) \
        and (___check_obj_id(L['cross_attn_layer_head_mask'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140273611202064))) \
        and (___compile_config_hash() == 'b861ac09b147974d65ae8b0b2b040ad0') \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[4], 7677632)) \
        and (___check_tensors(L['hidden_states'], L['attention_mask'], L['encoder_hidden_states'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_29*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_29(*args, **kwargs):
    pass

def __transformed_code_19_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    cross_attn_past_key_value = None; cross_attn_present_key_value = None; cross_attn_weights = None; outputs = None; present_key_value = None; residual = None; self_attn_past_key_value = None; self_attn_weights = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_29(hidden_states, attention_mask,
        encoder_hidden_states)
    return graph_out_0[0],


def __guard_18_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 140276759025536)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['use_cache'], 7677632)) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['attention_mask'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['past_key_value'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (hasattr(L['encoder_hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['encoder_attention_mask'], 7628576)) \
        and (___check_obj_id(L['cross_attn_layer_head_mask'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140273611202064))) \
        and (___compile_config_hash() == 'b861ac09b147974d65ae8b0b2b040ad0') \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[4], 7677632)) \
        and (___check_tensors(L['hidden_states'], L['attention_mask'], L['encoder_hidden_states'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_28*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_28(*args, **kwargs):
    pass

def __transformed_code_18_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    cross_attn_past_key_value = None; cross_attn_present_key_value = None; cross_attn_weights = None; outputs = None; present_key_value = None; residual = None; self_attn_past_key_value = None; self_attn_weights = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_28(hidden_states, attention_mask,
        encoder_hidden_states)
    return graph_out_0[0],


def __guard_17_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 140276759024720)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['use_cache'], 7677632)) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['attention_mask'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['past_key_value'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (hasattr(L['encoder_hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['encoder_attention_mask'], 7628576)) \
        and (___check_obj_id(L['cross_attn_layer_head_mask'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140273611202064))) \
        and (___compile_config_hash() == 'b861ac09b147974d65ae8b0b2b040ad0') \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[4], 7677632)) \
        and (___check_tensors(L['hidden_states'], L['attention_mask'], L['encoder_hidden_states'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_27*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_27(*args, **kwargs):
    pass

def __transformed_code_17_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    cross_attn_past_key_value = None; cross_attn_present_key_value = None; cross_attn_weights = None; outputs = None; present_key_value = None; residual = None; self_attn_past_key_value = None; self_attn_weights = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_27(hidden_states, attention_mask,
        encoder_hidden_states)
    return graph_out_0[0],


def __guard_16_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 140276759023904)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['use_cache'], 7677632)) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['attention_mask'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['past_key_value'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (hasattr(L['encoder_hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['encoder_attention_mask'], 7628576)) \
        and (___check_obj_id(L['cross_attn_layer_head_mask'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140273611202064))) \
        and (___compile_config_hash() == 'b861ac09b147974d65ae8b0b2b040ad0') \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[4], 7677632)) \
        and (___check_tensors(L['hidden_states'], L['attention_mask'], L['encoder_hidden_states'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_26*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_26(*args, **kwargs):
    pass

def __transformed_code_16_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    cross_attn_past_key_value = None; cross_attn_present_key_value = None; cross_attn_weights = None; outputs = None; present_key_value = None; residual = None; self_attn_past_key_value = None; self_attn_weights = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_26(hidden_states, attention_mask,
        encoder_hidden_states)
    return graph_out_0[0],


def __guard_15_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 140276759023088)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['use_cache'], 7677632)) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['attention_mask'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['past_key_value'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (hasattr(L['encoder_hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['encoder_attention_mask'], 7628576)) \
        and (___check_obj_id(L['cross_attn_layer_head_mask'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140273611202064))) \
        and (___compile_config_hash() == 'b861ac09b147974d65ae8b0b2b040ad0') \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[4], 7677632)) \
        and (___check_tensors(L['hidden_states'], L['attention_mask'], L['encoder_hidden_states'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_25*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_25(*args, **kwargs):
    pass

def __transformed_code_15_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    cross_attn_past_key_value = None; cross_attn_present_key_value = None; cross_attn_weights = None; outputs = None; present_key_value = None; residual = None; self_attn_past_key_value = None; self_attn_weights = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_25(hidden_states, attention_mask,
        encoder_hidden_states)
    return graph_out_0[0],


def __guard_14_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 140276879850608)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['use_cache'], 7677632)) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['attention_mask'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['past_key_value'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (hasattr(L['encoder_hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['encoder_attention_mask'], 7628576)) \
        and (___check_obj_id(L['cross_attn_layer_head_mask'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140273611202064))) \
        and (___compile_config_hash() == 'b861ac09b147974d65ae8b0b2b040ad0') \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[4], 7677632)) \
        and (___check_tensors(L['hidden_states'], L['attention_mask'], L['encoder_hidden_states'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_24*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_24(*args, **kwargs):
    pass

def __transformed_code_14_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    cross_attn_past_key_value = None; cross_attn_present_key_value = None; cross_attn_weights = None; outputs = None; present_key_value = None; residual = None; self_attn_past_key_value = None; self_attn_weights = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_24(hidden_states, attention_mask,
        encoder_hidden_states)
    return graph_out_0[0],


def __guard_13_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 140276879850752)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['use_cache'], 7677632)) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['attention_mask'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['past_key_value'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (hasattr(L['encoder_hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['encoder_attention_mask'], 7628576)) \
        and (___check_obj_id(L['cross_attn_layer_head_mask'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140273611202064))) \
        and (___compile_config_hash() == 'b861ac09b147974d65ae8b0b2b040ad0') \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[4], 7677632)) \
        and (___check_tensors(L['hidden_states'], L['attention_mask'], L['encoder_hidden_states'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_23*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_23(*args, **kwargs):
    pass

def __transformed_code_13_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    cross_attn_past_key_value = None; cross_attn_present_key_value = None; cross_attn_weights = None; outputs = None; present_key_value = None; residual = None; self_attn_past_key_value = None; self_attn_weights = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_23(hidden_states, attention_mask,
        encoder_hidden_states)
    return graph_out_0[0],


def __guard_12_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 140276879850704)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['use_cache'], 7677632)) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['attention_mask'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['past_key_value'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (hasattr(L['encoder_hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['encoder_attention_mask'], 7628576)) \
        and (___check_obj_id(L['cross_attn_layer_head_mask'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140273611202064))) \
        and (___compile_config_hash() == 'b861ac09b147974d65ae8b0b2b040ad0') \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[4], 7677632)) \
        and (___check_tensors(L['hidden_states'], L['attention_mask'], L['encoder_hidden_states'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_22*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_22(*args, **kwargs):
    pass

def __transformed_code_12_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    cross_attn_past_key_value = None; cross_attn_present_key_value = None; cross_attn_weights = None; outputs = None; present_key_value = None; residual = None; self_attn_past_key_value = None; self_attn_weights = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_22(hidden_states, attention_mask,
        encoder_hidden_states)
    return graph_out_0[0],


def __guard_11_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 140276879849936)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['use_cache'], 7677632)) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['attention_mask'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['past_key_value'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (hasattr(L['encoder_hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['encoder_attention_mask'], 7628576)) \
        and (___check_obj_id(L['cross_attn_layer_head_mask'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140273611202064))) \
        and (___compile_config_hash() == 'b861ac09b147974d65ae8b0b2b040ad0') \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[4], 7677632)) \
        and (___check_tensors(L['hidden_states'], L['attention_mask'], L['encoder_hidden_states'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_21*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_21(*args, **kwargs):
    pass

def __transformed_code_11_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    cross_attn_past_key_value = None; cross_attn_present_key_value = None; cross_attn_weights = None; outputs = None; present_key_value = None; residual = None; self_attn_past_key_value = None; self_attn_weights = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_21(hidden_states, attention_mask,
        encoder_hidden_states)
    return graph_out_0[0],


def __guard_10_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 140276879849120)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['use_cache'], 7677632)) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['attention_mask'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['past_key_value'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (hasattr(L['encoder_hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['encoder_attention_mask'], 7628576)) \
        and (___check_obj_id(L['cross_attn_layer_head_mask'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140273611202064))) \
        and (___compile_config_hash() == 'b861ac09b147974d65ae8b0b2b040ad0') \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[4], 7677632)) \
        and (___check_tensors(L['hidden_states'], L['attention_mask'], L['encoder_hidden_states'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_20*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_20(*args, **kwargs):
    pass

def __transformed_code_10_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    cross_attn_past_key_value = None; cross_attn_present_key_value = None; cross_attn_weights = None; outputs = None; present_key_value = None; residual = None; self_attn_past_key_value = None; self_attn_weights = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_20(hidden_states, attention_mask,
        encoder_hidden_states)
    return graph_out_0[0],


def __guard_9_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 140276879848112)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['use_cache'], 7677632)) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['attention_mask'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['past_key_value'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (hasattr(L['encoder_hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['encoder_attention_mask'], 7628576)) \
        and (___check_obj_id(L['cross_attn_layer_head_mask'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140273611202064))) \
        and (___compile_config_hash() == 'b861ac09b147974d65ae8b0b2b040ad0') \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[4], 7677632)) \
        and (___check_tensors(L['hidden_states'], L['attention_mask'], L['encoder_hidden_states'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_19*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_19(*args, **kwargs):
    pass

def __transformed_code_9_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    cross_attn_past_key_value = None; cross_attn_present_key_value = None; cross_attn_weights = None; outputs = None; present_key_value = None; residual = None; self_attn_past_key_value = None; self_attn_weights = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_19(hidden_states, attention_mask,
        encoder_hidden_states)
    return graph_out_0[0],


def __guard_8_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 140276879850560)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['use_cache'], 7677632)) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['attention_mask'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['past_key_value'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (hasattr(L['encoder_hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['encoder_attention_mask'], 7628576)) \
        and (___check_obj_id(L['cross_attn_layer_head_mask'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140273611202064))) \
        and (___compile_config_hash() == 'b861ac09b147974d65ae8b0b2b040ad0') \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[4], 7677632)) \
        and (___check_tensors(L['hidden_states'], L['attention_mask'], L['encoder_hidden_states'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_18*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_18(*args, **kwargs):
    pass

def __transformed_code_8_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    cross_attn_past_key_value = None; cross_attn_present_key_value = None; cross_attn_weights = None; outputs = None; present_key_value = None; residual = None; self_attn_past_key_value = None; self_attn_weights = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_18(hidden_states, attention_mask,
        encoder_hidden_states)
    return graph_out_0[0],


def __guard_7_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 140276879848304)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['use_cache'], 7677632)) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['attention_mask'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['past_key_value'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (hasattr(L['encoder_hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['encoder_attention_mask'], 7628576)) \
        and (___check_obj_id(L['cross_attn_layer_head_mask'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140273611202064))) \
        and (___compile_config_hash() == 'b861ac09b147974d65ae8b0b2b040ad0') \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[4], 7677632)) \
        and (___check_tensors(L['hidden_states'], L['attention_mask'], L['encoder_hidden_states'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_17*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_17(*args, **kwargs):
    pass

def __transformed_code_7_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    cross_attn_past_key_value = None; cross_attn_present_key_value = None; cross_attn_weights = None; outputs = None; present_key_value = None; residual = None; self_attn_past_key_value = None; self_attn_weights = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_17(hidden_states, attention_mask,
        encoder_hidden_states)
    return graph_out_0[0],


def __guard_6_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 140276879847680)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['use_cache'], 7677632)) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['attention_mask'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['past_key_value'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (hasattr(L['encoder_hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['encoder_attention_mask'], 7628576)) \
        and (___check_obj_id(L['cross_attn_layer_head_mask'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140273611202064))) \
        and (___compile_config_hash() == 'b861ac09b147974d65ae8b0b2b040ad0') \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].self_attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].encoder_attn.forward.__defaults__[4], 7677632)) \
        and (___check_tensors(L['hidden_states'], L['attention_mask'], L['encoder_hidden_states'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_16*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_16(*args, **kwargs):
    pass

def __transformed_code_6_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    cross_attn_past_key_value = None; cross_attn_present_key_value = None; cross_attn_weights = None; outputs = None; present_key_value = None; residual = None; self_attn_past_key_value = None; self_attn_weights = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_16(hidden_states, attention_mask,
        encoder_hidden_states)
    return graph_out_0[0],


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    residual = hidden_states
    hidden_states = self.self_attn_layer_norm(hidden_states)
    if past_key_value is not None:
        self_attn_past_key_value = past_key_value[slice(None, 2)]
    else:
        self_attn_past_key_value = None
    __temp_1780 = self.self_attn(hidden_states=hidden_states, past_key_value=
        self_attn_past_key_value, attention_mask=attention_mask,
        layer_head_mask=layer_head_mask, output_attentions=output_attentions)
    hidden_states = __temp_1780[0]
    self_attn_weights = __temp_1780[1]
    present_key_value = __temp_1780[2]
    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout,
        training=self.training)
    hidden_states = residual + hidden_states
    cross_attn_present_key_value = None
    cross_attn_weights = None
    if encoder_hidden_states is not None:
        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        if past_key_value is not None:
            cross_attn_past_key_value = past_key_value[slice(-2, None)]
        else:
            cross_attn_past_key_value = None
        __temp_1783 = self.encoder_attn(hidden_states=hidden_states,
            key_value_states=encoder_hidden_states, attention_mask=
            encoder_attention_mask, layer_head_mask=cross_attn_layer_head_mask,
            past_key_value=cross_attn_past_key_value, output_attentions=
            output_attentions)
        hidden_states = __temp_1783[0]
        cross_attn_weights = __temp_1783[1]
        cross_attn_present_key_value = __temp_1783[2]
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout,
            training=self.training)
        hidden_states = residual + hidden_states
        present_key_value = present_key_value + cross_attn_present_key_value
    residual = hidden_states
    hidden_states = self.final_layer_norm(hidden_states)
    hidden_states = self.activation_fn(self.fc1(hidden_states))
    hidden_states = nn.functional.dropout(hidden_states, p=self.
        activation_dropout, training=self.training)
    hidden_states = self.fc2(hidden_states)
    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout,
        training=self.training)
    hidden_states = residual + hidden_states
    outputs = hidden_states,
    if output_attentions:
        outputs += self_attn_weights, cross_attn_weights
    if use_cache:
        outputs += present_key_value,
    return outputs

def transformed_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    L = {"self": self, "hidden_states": hidden_states, "attention_mask": attention_mask, "encoder_hidden_states": encoder_hidden_states, "encoder_attention_mask": encoder_attention_mask, "layer_head_mask": layer_head_mask, "cross_attn_layer_head_mask": cross_attn_layer_head_mask, "past_key_value": past_key_value, "output_attentions": output_attentions, "use_cache": use_cache}
    if __guard_29_for_forward(L):
        return __transformed_code_29_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)
    if __guard_28_for_forward(L):
        return __transformed_code_28_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)
    if __guard_27_for_forward(L):
        return __transformed_code_27_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)
    if __guard_26_for_forward(L):
        return __transformed_code_26_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)
    if __guard_25_for_forward(L):
        return __transformed_code_25_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)
    if __guard_24_for_forward(L):
        return __transformed_code_24_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)
    if __guard_23_for_forward(L):
        return __transformed_code_23_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)
    if __guard_22_for_forward(L):
        return __transformed_code_22_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)
    if __guard_21_for_forward(L):
        return __transformed_code_21_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)
    if __guard_20_for_forward(L):
        return __transformed_code_20_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)
    if __guard_19_for_forward(L):
        return __transformed_code_19_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)
    if __guard_18_for_forward(L):
        return __transformed_code_18_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)
    if __guard_17_for_forward(L):
        return __transformed_code_17_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)
    if __guard_16_for_forward(L):
        return __transformed_code_16_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)
    if __guard_15_for_forward(L):
        return __transformed_code_15_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)
    if __guard_14_for_forward(L):
        return __transformed_code_14_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)
    if __guard_13_for_forward(L):
        return __transformed_code_13_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)
    if __guard_12_for_forward(L):
        return __transformed_code_12_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)
    if __guard_11_for_forward(L):
        return __transformed_code_11_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)
    if __guard_10_for_forward(L):
        return __transformed_code_10_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)
    if __guard_9_for_forward(L):
        return __transformed_code_9_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)
    if __guard_8_for_forward(L):
        return __transformed_code_8_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)
    if __guard_7_for_forward(L):
        return __transformed_code_7_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)
    if __guard_6_for_forward(L):
        return __transformed_code_6_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)

#============ end of forward ============#
