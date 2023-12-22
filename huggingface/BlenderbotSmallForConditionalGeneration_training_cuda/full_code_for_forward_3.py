
def __guard_10_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 139946272256400)) \
        and (L['self'].training == True) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['attention_mask'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139943001890320))) \
        and (___compile_config_hash() == 'e372bf5d906916bac23c6fd5dd0b3288') \
        and (___check_type_id(G['torch'].float16, 139948263380736)) \
        and (G['torch'].float16 == torch.float16) \
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
        and (___check_tensors(L['hidden_states'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_15*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_15(*args, **kwargs):
    pass

def __transformed_code_10_for_forward(self, hidden_states, attention_mask, layer_head_mask, output_attentions):
    _ = None; attn_weights = None; clamp_value = None; outputs = None; residual = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_15(hidden_states)
    return graph_out_0[0],


def __guard_9_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 139946271634192)) \
        and (L['self'].training == True) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['attention_mask'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139943001890320))) \
        and (___compile_config_hash() == 'e372bf5d906916bac23c6fd5dd0b3288') \
        and (___check_type_id(G['torch'].float16, 139948263380736)) \
        and (G['torch'].float16 == torch.float16) \
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
        and (___check_tensors(L['hidden_states'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_14*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_14(*args, **kwargs):
    pass

def __transformed_code_9_for_forward(self, hidden_states, attention_mask, layer_head_mask, output_attentions):
    _ = None; attn_weights = None; clamp_value = None; outputs = None; residual = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_14(hidden_states)
    return graph_out_0[0],


def __guard_8_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 139946271635200)) \
        and (L['self'].training == True) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['attention_mask'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139943001890320))) \
        and (___compile_config_hash() == 'e372bf5d906916bac23c6fd5dd0b3288') \
        and (___check_type_id(G['torch'].float16, 139948263380736)) \
        and (G['torch'].float16 == torch.float16) \
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
        and (___check_tensors(L['hidden_states'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_13*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_13(*args, **kwargs):
    pass

def __transformed_code_8_for_forward(self, hidden_states, attention_mask, layer_head_mask, output_attentions):
    _ = None; attn_weights = None; clamp_value = None; outputs = None; residual = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_13(hidden_states)
    return graph_out_0[0],


def __guard_7_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 139946272255824)) \
        and (L['self'].training == True) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['attention_mask'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139943001890320))) \
        and (___compile_config_hash() == 'e372bf5d906916bac23c6fd5dd0b3288') \
        and (___check_type_id(G['torch'].float16, 139948263380736)) \
        and (G['torch'].float16 == torch.float16) \
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
        and (___check_tensors(L['hidden_states'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_12*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_12(*args, **kwargs):
    pass

def __transformed_code_7_for_forward(self, hidden_states, attention_mask, layer_head_mask, output_attentions):
    _ = None; attn_weights = None; clamp_value = None; outputs = None; residual = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_12(hidden_states)
    return graph_out_0[0],


def __guard_6_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 139946271563568)) \
        and (L['self'].training == True) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['attention_mask'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139943001890320))) \
        and (___compile_config_hash() == 'e372bf5d906916bac23c6fd5dd0b3288') \
        and (___check_type_id(G['torch'].float16, 139948263380736)) \
        and (G['torch'].float16 == torch.float16) \
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
        and (___check_tensors(L['hidden_states'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_11*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_11(*args, **kwargs):
    pass

def __transformed_code_6_for_forward(self, hidden_states, attention_mask, layer_head_mask, output_attentions):
    _ = None; attn_weights = None; clamp_value = None; outputs = None; residual = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_11(hidden_states)
    return graph_out_0[0],


def __guard_5_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 139946271561600)) \
        and (L['self'].training == True) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['attention_mask'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139943001890320))) \
        and (___compile_config_hash() == 'e372bf5d906916bac23c6fd5dd0b3288') \
        and (___check_type_id(G['torch'].float16, 139948263380736)) \
        and (G['torch'].float16 == torch.float16) \
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
        and (___check_tensors(L['hidden_states'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_10*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_10(*args, **kwargs):
    pass

def __transformed_code_5_for_forward(self, hidden_states, attention_mask, layer_head_mask, output_attentions):
    _ = None; attn_weights = None; clamp_value = None; outputs = None; residual = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_10(hidden_states)
    return graph_out_0[0],


def __guard_4_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 139946272255440)) \
        and (L['self'].training == True) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['attention_mask'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139943001890320))) \
        and (___compile_config_hash() == 'e372bf5d906916bac23c6fd5dd0b3288') \
        and (___check_type_id(G['torch'].float16, 139948263380736)) \
        and (G['torch'].float16 == torch.float16) \
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
        and (___check_tensors(L['hidden_states'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_9*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_9(*args, **kwargs):
    pass

def __transformed_code_4_for_forward(self, hidden_states, attention_mask, layer_head_mask, output_attentions):
    _ = None; attn_weights = None; clamp_value = None; outputs = None; residual = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_9(hidden_states)
    return graph_out_0[0],


def __guard_3_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 139946272254144)) \
        and (L['self'].training == True) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['attention_mask'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139943001890320))) \
        and (___compile_config_hash() == 'e372bf5d906916bac23c6fd5dd0b3288') \
        and (___check_type_id(G['torch'].float16, 139948263380736)) \
        and (G['torch'].float16 == torch.float16) \
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
        and (___check_tensors(L['hidden_states'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_8*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_8(*args, **kwargs):
    pass

def __transformed_code_3_for_forward(self, hidden_states, attention_mask, layer_head_mask, output_attentions):
    _ = None; attn_weights = None; clamp_value = None; outputs = None; residual = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_8(hidden_states)
    return graph_out_0[0],


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def forward(self, hidden_states, attention_mask, layer_head_mask, output_attentions):
    residual = hidden_states
    __temp_228 = self.self_attn(hidden_states=hidden_states, attention_mask=
        attention_mask, layer_head_mask=layer_head_mask, output_attentions=
        output_attentions)
    hidden_states = __temp_228[0]
    attn_weights = __temp_228[1]
    _ = __temp_228[2]
    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout,
        training=self.training)
    hidden_states = residual + hidden_states
    hidden_states = self.self_attn_layer_norm(hidden_states)
    residual = hidden_states
    hidden_states = self.activation_fn(self.fc1(hidden_states))
    hidden_states = nn.functional.dropout(hidden_states, p=self.
        activation_dropout, training=self.training)
    hidden_states = self.fc2(hidden_states)
    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout,
        training=self.training)
    hidden_states = residual + hidden_states
    hidden_states = self.final_layer_norm(hidden_states)
    if hidden_states.dtype == torch.float16:
        if not torch.isinf(hidden_states).any():
            if torch.isnan(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value,
                    max=clamp_value)
        else:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=
                clamp_value)
    outputs = hidden_states,
    if output_attentions:
        outputs += attn_weights,
    return outputs

def transformed_forward(self, hidden_states, attention_mask, layer_head_mask, output_attentions):
    L = {"self": self, "hidden_states": hidden_states, "attention_mask": attention_mask, "layer_head_mask": layer_head_mask, "output_attentions": output_attentions}
    if __guard_10_for_forward(L):
        return __transformed_code_10_for_forward(self, hidden_states, attention_mask, layer_head_mask, output_attentions)
    if __guard_9_for_forward(L):
        return __transformed_code_9_for_forward(self, hidden_states, attention_mask, layer_head_mask, output_attentions)
    if __guard_8_for_forward(L):
        return __transformed_code_8_for_forward(self, hidden_states, attention_mask, layer_head_mask, output_attentions)
    if __guard_7_for_forward(L):
        return __transformed_code_7_for_forward(self, hidden_states, attention_mask, layer_head_mask, output_attentions)
    if __guard_6_for_forward(L):
        return __transformed_code_6_for_forward(self, hidden_states, attention_mask, layer_head_mask, output_attentions)
    if __guard_5_for_forward(L):
        return __transformed_code_5_for_forward(self, hidden_states, attention_mask, layer_head_mask, output_attentions)
    if __guard_4_for_forward(L):
        return __transformed_code_4_for_forward(self, hidden_states, attention_mask, layer_head_mask, output_attentions)
    if __guard_3_for_forward(L):
        return __transformed_code_3_for_forward(self, hidden_states, attention_mask, layer_head_mask, output_attentions)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return forward(self, hidden_states, attention_mask, layer_head_mask, output_attentions)

#============ end of forward ============#
