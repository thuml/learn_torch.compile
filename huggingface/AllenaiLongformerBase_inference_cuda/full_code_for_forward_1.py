
# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_346_4(___stack0, self, return_dict):
    'Failed to decompile.'

def transformed___resume_at_346_4(___stack0, self, return_dict):
    L = {"___stack0": ___stack0, "self": self, "return_dict": return_dict}
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_346_4(___stack0, self, return_dict)

#============ end of __resume_at_346_4 ============#

def __guard_1_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 139642392460544)) \
        and (L['self'].training == False) \
        and (___check_obj_id(L['head_mask'], 7628576)) \
        and (___check_type_id(L['input_ids'], 98528224)) \
        and (hasattr(L['input_ids'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['return_dict'], 7677664)) \
        and (___check_obj_id(L['position_ids'], 7628576)) \
        and (___check_obj_id(L['inputs_embeds'], 7628576)) \
        and (___check_obj_id(L['attention_mask'], 7628576)) \
        and (___check_obj_id(L['token_type_ids'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7628576)) \
        and (___check_obj_id(L['output_hidden_states'], 7628576)) \
        and (___check_obj_id(L['global_attention_mask'], 7628576)) \
        and (___check_obj_id(L['self'].get_extended_attention_mask.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].get_extended_attention_mask.__defaults__[1], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139639130168848))) \
        and (___compile_config_hash() == 'd23842bdb1f875b062b4abc655654038') \
        and (not ___needs_nopython()) \
        and (___check_type_id(G['torch'].long, 139644390237952)) \
        and (G['torch'].long == torch.int64) \
        and (___check_type_id(G['__import_transformers_dot_modeling_utils'].XLA_USE_BF16, 7605632)) \
        and (G['__import_transformers_dot_modeling_utils'].XLA_USE_BF16 == '0') \
        and (___check_type_id(G['__import_transformers_dot_modeling_utils'].XLA_DOWNCAST_BF16, 7605632)) \
        and (G['__import_transformers_dot_modeling_utils'].XLA_DOWNCAST_BF16 == '0') \
        and (___check_type_id(G['__import_transformers_dot_modeling_utils'].ENV_VARS_TRUE_VALUES, 7622752)) \
        and (G['__import_transformers_dot_modeling_utils'].ENV_VARS_TRUE_VALUES == {'ON', 'YES', 'TRUE', '1'}) \
        and (___check_obj_id(G['__import_transformers_dot_utils_dot_import_utils']._torch_available, 7677664)) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_obj_id(G['__import_transformers_dot_utils_dot_import_utils']._torch_fx_available, 7677664)) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_obj_id(L['self'].embeddings.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].embeddings.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].embeddings.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].embeddings.forward.__defaults__[3], 7628576)) \
        and (___check_tensors(L['input_ids'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_3*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_3(*args, **kwargs):
    pass

def __transformed_code_1_for_forward(self, input_ids, attention_mask, global_attention_mask, head_mask, token_type_ids, position_ids, inputs_embeds, output_attentions, output_hidden_states, return_dict):
    device = None; embedding_output = None; encoder_outputs = None; extended_attention_mask = None; input_shape = None; padding_len = None; pooled_output = None; sequence_output = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_3(input_ids)
    return __resume_at_346_4(self.encoder(graph_out_0[0], attention_mask=
        graph_out_0[1], head_mask=head_mask, padding_len=0, output_attentions=
        self.config.output_attentions, output_hidden_states=self.config.
        output_hidden_states, return_dict=return_dict), self, return_dict)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def forward(self, input_ids, attention_mask, global_attention_mask, head_mask, token_type_ids, position_ids, inputs_embeds, output_attentions, output_hidden_states, return_dict):
    'Failed to decompile.'

def transformed_forward(self, input_ids, attention_mask, global_attention_mask, head_mask, token_type_ids, position_ids, inputs_embeds, output_attentions, output_hidden_states, return_dict):
    L = {"self": self, "input_ids": input_ids, "attention_mask": attention_mask, "global_attention_mask": global_attention_mask, "head_mask": head_mask, "token_type_ids": token_type_ids, "position_ids": position_ids, "inputs_embeds": inputs_embeds, "output_attentions": output_attentions, "output_hidden_states": output_hidden_states, "return_dict": return_dict}
    if __guard_1_for_forward(L):
        return __transformed_code_1_for_forward(self, input_ids, attention_mask, global_attention_mask, head_mask, token_type_ids, position_ids, inputs_embeds, output_attentions, output_hidden_states, return_dict)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return forward(self, input_ids, attention_mask, global_attention_mask, head_mask, token_type_ids, position_ids, inputs_embeds, output_attentions, output_hidden_states, return_dict)

#============ end of forward ============#
