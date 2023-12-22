
def __guard_0_for_resume_in_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 140063927089664)) \
        and (L['self'].training == True) \
        and (hasattr(L['labels'], '_dynamo_dynamic_indices') == False) \
        and (___check_type_id(L['___stack0'], 148926288)) \
        and (___check_obj_id(L['return_dict'], 7677664)) \
        and (___check_obj_id(L['___stack0'].attentions, 7628576)) \
        and (___check_obj_id(L['___stack0'].hidden_states, 7628576)) \
        and (___check_type_id(L['___stack0'].past_key_values, 7617760)) \
        and (len(L['___stack0'].past_key_values) == 6) \
        and (___check_obj_id(L['___stack0'].cross_attentions, 7628576)) \
        and (hasattr(L['___stack0'].last_hidden_state, '_dynamo_dynamic_indices') == False) \
        and (___check_type_id(L['___stack0'].past_key_values[0], 7617760)) \
        and (len(L['___stack0'].past_key_values[0]) == 2) \
        and (___check_type_id(L['___stack0'].past_key_values[1], 7617760)) \
        and (len(L['___stack0'].past_key_values[1]) == 2) \
        and (___check_type_id(L['___stack0'].past_key_values[2], 7617760)) \
        and (len(L['___stack0'].past_key_values[2]) == 2) \
        and (___check_type_id(L['___stack0'].past_key_values[3], 7617760)) \
        and (len(L['___stack0'].past_key_values[3]) == 2) \
        and (___check_type_id(L['___stack0'].past_key_values[4], 7617760)) \
        and (len(L['___stack0'].past_key_values[4]) == 2) \
        and (___check_type_id(L['___stack0'].past_key_values[5], 7617760)) \
        and (len(L['___stack0'].past_key_values[5]) == 2) \
        and (hasattr(L['___stack0'].past_key_values[0][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack0'].past_key_values[0][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack0'].past_key_values[1][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack0'].past_key_values[1][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack0'].past_key_values[2][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack0'].past_key_values[2][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack0'].past_key_values[3][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack0'].past_key_values[3][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack0'].past_key_values[4][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack0'].past_key_values[4][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack0'].past_key_values[5][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['___stack0'].past_key_values[5][1], '_dynamo_dynamic_indices') == False) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140063927393808))) \
        and (___compile_config_hash() == 'd9da5cdf5912d92e85cf14993755c996') \
        and (___check_tensors(L['labels'], L['___stack0'].last_hidden_state, L['___stack0'].past_key_values[0][0], L['___stack0'].past_key_values[0][1], L['___stack0'].past_key_values[1][0], L['___stack0'].past_key_values[1][1], L['___stack0'].past_key_values[2][0], L['___stack0'].past_key_values[2][1], L['___stack0'].past_key_values[3][0], L['___stack0'].past_key_values[3][1], L['___stack0'].past_key_values[4][0], L['___stack0'].past_key_values[4][1], L['___stack0'].past_key_values[5][0], L['___stack0'].past_key_values[5][1], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_20*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_20(*args, **kwargs):
    pass

def __transformed_code_0_for_resume_in_forward(___stack0, self, labels, return_dict):
    attention_mask = None; cross_attn_head_mask = None; encoder_attention_mask = None; encoder_hidden_states = None; head_mask = None; input_ids = None; inputs_embeds = None; logits = None; loss = None; loss_fct = None; output = None; output_attentions = None; output_hidden_states = None; outputs = None; past_key_values = None; use_cache = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_20(___stack0.last_hidden_state, labels)
    import importlib
    return importlib.import_module('transformers.modeling_outputs'
        ).CausalLMOutputWithCrossAttentions(loss=graph_out_0[0], logits=
        graph_out_0[1], past_key_values=___stack0.past_key_values,
        hidden_states=___stack0.hidden_states, attentions=___stack0.attentions,
        cross_attentions=___stack0.cross_attentions)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_94_5(___stack0, self, labels, return_dict):
    'Failed to decompile.'

def transformed___resume_at_94_5(___stack0, self, labels, return_dict):
    L = {"___stack0": ___stack0, "self": self, "labels": labels, "return_dict": return_dict}
    if __guard_0_for_resume_in_forward(L):
        return __transformed_code_0_for_resume_in_forward(___stack0, self, labels, return_dict)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_94_5(___stack0, self, labels, return_dict)

#============ end of __resume_at_94_5 ============#

def __guard_0_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 140063927089664)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['head_mask'], 7628576)) \
        and (hasattr(L['input_ids'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['use_cache'], 7628576)) \
        and (___check_obj_id(L['return_dict'], 7628576)) \
        and (___check_obj_id(L['inputs_embeds'], 7628576)) \
        and (___check_obj_id(L['attention_mask'], 7628576)) \
        and (___check_obj_id(L['past_key_values'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7628576)) \
        and (___check_obj_id(L['cross_attn_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_hidden_states'], 7628576)) \
        and (___check_obj_id(L['encoder_hidden_states'], 7628576)) \
        and (___check_obj_id(L['encoder_attention_mask'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140063927393808))) \
        and (___compile_config_hash() == 'd9da5cdf5912d92e85cf14993755c996') \
        and (not ___needs_nopython()) \
        and (___check_tensors(L['input_ids'], tensor_check_names=tensor_check_names))

def __transformed_code_0_for_forward(self, input_ids, attention_mask, encoder_hidden_states, encoder_attention_mask, head_mask, cross_attn_head_mask, past_key_values, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict):
    logits = None; loss = None; loss_fct = None; output = None; outputs = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    return_dict = self.config.use_return_dict
    return __resume_at_94_5(self.model.decoder(input_ids=input_ids,
        attention_mask=attention_mask, encoder_hidden_states=
        encoder_hidden_states, encoder_attention_mask=encoder_attention_mask,
        head_mask=head_mask, cross_attn_head_mask=cross_attn_head_mask,
        past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache
        =use_cache, output_attentions=self.config.output_attentions,
        output_hidden_states=self.config.output_hidden_states, return_dict=self
        .config.use_return_dict), self, labels, return_dict)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def forward(self, input_ids, attention_mask, encoder_hidden_states, encoder_attention_mask, head_mask, cross_attn_head_mask, past_key_values, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict):
    'Failed to decompile.'

def transformed_forward(self, input_ids, attention_mask, encoder_hidden_states, encoder_attention_mask, head_mask, cross_attn_head_mask, past_key_values, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict):
    L = {"self": self, "input_ids": input_ids, "attention_mask": attention_mask, "encoder_hidden_states": encoder_hidden_states, "encoder_attention_mask": encoder_attention_mask, "head_mask": head_mask, "cross_attn_head_mask": cross_attn_head_mask, "past_key_values": past_key_values, "inputs_embeds": inputs_embeds, "labels": labels, "use_cache": use_cache, "output_attentions": output_attentions, "output_hidden_states": output_hidden_states, "return_dict": return_dict}
    if __guard_0_for_forward(L):
        return __transformed_code_0_for_forward(self, input_ids, attention_mask, encoder_hidden_states, encoder_attention_mask, head_mask, cross_attn_head_mask, past_key_values, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return forward(self, input_ids, attention_mask, encoder_hidden_states, encoder_attention_mask, head_mask, cross_attn_head_mask, past_key_values, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict)

#============ end of forward ============#
