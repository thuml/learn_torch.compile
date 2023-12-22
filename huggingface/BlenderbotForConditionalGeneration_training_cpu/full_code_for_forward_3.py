
def __guard_4_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 140276879847632)) \
        and (L['self'].training == True) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['attention_mask'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140273611202064))) \
        and (___compile_config_hash() == 'b861ac09b147974d65ae8b0b2b040ad0') \
        and (___check_type_id(G['torch'].float16, 140278872815360)) \
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
        and (___check_obj_id(L['self'], 140276879847392)) \
        and (L['self'].training == True) \
        and (hasattr(L['hidden_states'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['attention_mask'], 7628576)) \
        and (___check_obj_id(L['layer_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140273611202064))) \
        and (___compile_config_hash() == 'b861ac09b147974d65ae8b0b2b040ad0') \
        and (___check_type_id(G['torch'].float16, 140278872815360)) \
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
    hidden_states = self.self_attn_layer_norm(hidden_states)
    __temp_217 = self.self_attn(hidden_states=hidden_states, attention_mask=
        attention_mask, layer_head_mask=layer_head_mask, output_attentions=
        output_attentions)
    hidden_states = __temp_217[0]
    attn_weights = __temp_217[1]
    _ = __temp_217[2]
    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout,
        training=self.training)
    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = self.final_layer_norm(hidden_states)
    hidden_states = self.activation_fn(self.fc1(hidden_states))
    hidden_states = nn.functional.dropout(hidden_states, p=self.
        activation_dropout, training=self.training)
    hidden_states = self.fc2(hidden_states)
    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout,
        training=self.training)
    hidden_states = residual + hidden_states
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
    if __guard_4_for_forward(L):
        return __transformed_code_4_for_forward(self, hidden_states, attention_mask, layer_head_mask, output_attentions)
    if __guard_3_for_forward(L):
        return __transformed_code_3_for_forward(self, hidden_states, attention_mask, layer_head_mask, output_attentions)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return forward(self, hidden_states, attention_mask, layer_head_mask, output_attentions)

#============ end of forward ============#
