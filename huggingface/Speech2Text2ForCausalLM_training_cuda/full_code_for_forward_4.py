
def __guard_7_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 140067223121504)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['use_cache'], 7677664)) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['attention_mask'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['past_key_value'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (___check_obj_id(L['encoder_hidden_states'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140063927393808))) \
        and (___compile_config_hash() == 'd9da5cdf5912d92e85cf14993755c996') \
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
        and (___check_tensors(L['hidden_states'], L['attention_mask'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_13*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_13(*args, **kwargs):
    pass

def __transformed_code_7_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    cross_attn_past_key_value = None; cross_attn_present_key_value = None; cross_attn_weights = None; outputs = None; present_key_value = None; residual = None; self_attn_past_key_value = None; self_attn_weights = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_13(hidden_states, attention_mask)
    return graph_out_0[0], (graph_out_0[1], graph_out_0[2])


def __guard_6_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 140067223119056)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['use_cache'], 7677664)) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['attention_mask'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['past_key_value'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (___check_obj_id(L['encoder_hidden_states'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140063927393808))) \
        and (___compile_config_hash() == 'd9da5cdf5912d92e85cf14993755c996') \
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
        and (___check_tensors(L['hidden_states'], L['attention_mask'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_12*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_12(*args, **kwargs):
    pass

def __transformed_code_6_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    cross_attn_past_key_value = None; cross_attn_present_key_value = None; cross_attn_weights = None; outputs = None; present_key_value = None; residual = None; self_attn_past_key_value = None; self_attn_weights = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_12(hidden_states, attention_mask)
    return graph_out_0[0], (graph_out_0[1], graph_out_0[2])


def __guard_5_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 140067223113776)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['use_cache'], 7677664)) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['attention_mask'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['past_key_value'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (___check_obj_id(L['encoder_hidden_states'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140063927393808))) \
        and (___compile_config_hash() == 'd9da5cdf5912d92e85cf14993755c996') \
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
        and (___check_tensors(L['hidden_states'], L['attention_mask'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_11*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_11(*args, **kwargs):
    pass

def __transformed_code_5_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    cross_attn_past_key_value = None; cross_attn_present_key_value = None; cross_attn_weights = None; outputs = None; present_key_value = None; residual = None; self_attn_past_key_value = None; self_attn_weights = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_11(hidden_states, attention_mask)
    return graph_out_0[0], (graph_out_0[1], graph_out_0[2])


def __guard_4_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 140067222882624)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['use_cache'], 7677664)) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['attention_mask'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['past_key_value'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (___check_obj_id(L['encoder_hidden_states'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140063927393808))) \
        and (___compile_config_hash() == 'd9da5cdf5912d92e85cf14993755c996') \
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
        and (___check_tensors(L['hidden_states'], L['attention_mask'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_10*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_10(*args, **kwargs):
    pass

def __transformed_code_4_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    cross_attn_past_key_value = None; cross_attn_present_key_value = None; cross_attn_weights = None; outputs = None; present_key_value = None; residual = None; self_attn_past_key_value = None; self_attn_weights = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_10(hidden_states, attention_mask)
    return graph_out_0[0], (graph_out_0[1], graph_out_0[2])


def __guard_3_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 140067222883680)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['use_cache'], 7677664)) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['attention_mask'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['past_key_value'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (___check_obj_id(L['encoder_hidden_states'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140063927393808))) \
        and (___compile_config_hash() == 'd9da5cdf5912d92e85cf14993755c996') \
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
        and (___check_tensors(L['hidden_states'], L['attention_mask'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_9*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_9(*args, **kwargs):
    pass

def __transformed_code_3_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    cross_attn_past_key_value = None; cross_attn_present_key_value = None; cross_attn_weights = None; outputs = None; present_key_value = None; residual = None; self_attn_past_key_value = None; self_attn_weights = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_9(hidden_states, attention_mask)
    return graph_out_0[0], (graph_out_0[1], graph_out_0[2])


def __guard_2_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 140067222883584)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['use_cache'], 7677664)) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['attention_mask'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['past_key_value'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (___check_obj_id(L['encoder_hidden_states'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140063927393808))) \
        and (___compile_config_hash() == 'd9da5cdf5912d92e85cf14993755c996') \
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
        and (___check_tensors(L['hidden_states'], L['attention_mask'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_8*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_8(*args, **kwargs):
    pass

def __transformed_code_2_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    cross_attn_past_key_value = None; cross_attn_present_key_value = None; cross_attn_weights = None; outputs = None; present_key_value = None; residual = None; self_attn_past_key_value = None; self_attn_weights = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_8(hidden_states, attention_mask)
    return graph_out_0[0], (graph_out_0[1], graph_out_0[2])


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    residual = hidden_states
    if past_key_value is not None:
        self_attn_past_key_value = past_key_value[slice(None, 2)]
    else:
        self_attn_past_key_value = None
    __temp_217 = self.self_attn(hidden_states=hidden_states, past_key_value=
        self_attn_past_key_value, attention_mask=attention_mask,
        layer_head_mask=layer_head_mask, output_attentions=output_attentions)
    hidden_states = __temp_217[0]
    self_attn_weights = __temp_217[1]
    present_key_value = __temp_217[2]
    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout,
        training=self.training)
    hidden_states = residual + hidden_states
    hidden_states = self.self_attn_layer_norm(hidden_states)
    cross_attn_present_key_value = None
    cross_attn_weights = None
    if encoder_hidden_states is not None:
        residual = hidden_states
        if past_key_value is not None:
            cross_attn_past_key_value = past_key_value[slice(-2, None)]
        else:
            cross_attn_past_key_value = None
        __temp_220 = self.encoder_attn(hidden_states=hidden_states,
            key_value_states=encoder_hidden_states, attention_mask=
            encoder_attention_mask, layer_head_mask=cross_attn_layer_head_mask,
            past_key_value=cross_attn_past_key_value, output_attentions=
            output_attentions)
        hidden_states = __temp_220[0]
        cross_attn_weights = __temp_220[1]
        cross_attn_present_key_value = __temp_220[2]
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout,
            training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        present_key_value = present_key_value + cross_attn_present_key_value
    residual = hidden_states
    hidden_states = self.activation_fn(self.fc1(hidden_states))
    hidden_states = nn.functional.dropout(hidden_states, p=self.
        activation_dropout, training=self.training)
    hidden_states = self.fc2(hidden_states)
    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout,
        training=self.training)
    hidden_states = residual + hidden_states
    hidden_states = self.final_layer_norm(hidden_states)
    outputs = hidden_states,
    if output_attentions:
        outputs += self_attn_weights, cross_attn_weights
    if use_cache:
        outputs += present_key_value,
    return outputs

def transformed_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
    L = {"self": self, "hidden_states": hidden_states, "attention_mask": attention_mask, "encoder_hidden_states": encoder_hidden_states, "encoder_attention_mask": encoder_attention_mask, "layer_head_mask": layer_head_mask, "cross_attn_layer_head_mask": cross_attn_layer_head_mask, "past_key_value": past_key_value, "output_attentions": output_attentions, "use_cache": use_cache}
    if __guard_7_for_forward(L):
        return __transformed_code_7_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)
    if __guard_6_for_forward(L):
        return __transformed_code_6_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)
    if __guard_5_for_forward(L):
        return __transformed_code_5_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)
    if __guard_4_for_forward(L):
        return __transformed_code_4_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)
    if __guard_3_for_forward(L):
        return __transformed_code_3_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)
    if __guard_2_for_forward(L):
        return __transformed_code_2_for_forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask, cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache)

#============ end of forward ============#
