
def __guard_27_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 140346451017392)) \
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
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140343385792016))) \
        and (___compile_config_hash() == '9f64cb439de8127be15121eda92b9482') \
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
        and (___check_obj_id(L['self'], 140346451005344)) \
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
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140343385792016))) \
        and (___compile_config_hash() == '9f64cb439de8127be15121eda92b9482') \
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
        and (___check_obj_id(L['self'], 140346451015280)) \
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
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140343385792016))) \
        and (___compile_config_hash() == '9f64cb439de8127be15121eda92b9482') \
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
        and (___check_obj_id(L['self'], 140346451015376)) \
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
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140343385792016))) \
        and (___compile_config_hash() == '9f64cb439de8127be15121eda92b9482') \
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
        and (___check_obj_id(L['self'], 140346451013504)) \
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
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140343385792016))) \
        and (___compile_config_hash() == '9f64cb439de8127be15121eda92b9482') \
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
        and (___check_obj_id(L['self'], 140346451012832)) \
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
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140343385792016))) \
        and (___compile_config_hash() == '9f64cb439de8127be15121eda92b9482') \
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
        and (___check_obj_id(L['self'], 140346451012256)) \
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
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140343385792016))) \
        and (___compile_config_hash() == '9f64cb439de8127be15121eda92b9482') \
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
        and (___check_obj_id(L['self'], 140346451010864)) \
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
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140343385792016))) \
        and (___compile_config_hash() == '9f64cb439de8127be15121eda92b9482') \
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
        and (___check_obj_id(L['self'], 140346451010048)) \
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
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140343385792016))) \
        and (___compile_config_hash() == '9f64cb439de8127be15121eda92b9482') \
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
        and (___check_obj_id(L['self'], 140346451005728)) \
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
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140343385792016))) \
        and (___compile_config_hash() == '9f64cb439de8127be15121eda92b9482') \
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
        and (___check_obj_id(L['self'], 140346451009856)) \
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
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140343385792016))) \
        and (___compile_config_hash() == '9f64cb439de8127be15121eda92b9482') \
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
        and (___check_obj_id(L['self'], 140346451008176)) \
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
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140343385792016))) \
        and (___compile_config_hash() == '9f64cb439de8127be15121eda92b9482') \
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


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    residual = hidden_states
    hidden_states = self.self_attn_layer_norm(hidden_states)
    if past_key_value is not None:
        self_attn_past_key_value = past_key_value[slice(None, 2)]
    else:
        self_attn_past_key_value = None
    __temp_952 = self.self_attn(hidden_states=hidden_states, past_key_value=
        self_attn_past_key_value, attention_mask=attention_mask,
        layer_head_mask=layer_head_mask, output_attentions=output_attentions)
    hidden_states = __temp_952[0]
    self_attn_weights = __temp_952[1]
    present_key_value = __temp_952[2]
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
        __temp_955 = self.encoder_attn(hidden_states=hidden_states,
            key_value_states=encoder_hidden_states, attention_mask=
            encoder_attention_mask, layer_head_mask=cross_attn_layer_head_mask,
            past_key_value=cross_attn_past_key_value, output_attentions=
            output_attentions)
        hidden_states = __temp_955[0]
        cross_attn_weights = __temp_955[1]
        cross_attn_present_key_value = __temp_955[2]
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
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)

#============ end of forward ============#
