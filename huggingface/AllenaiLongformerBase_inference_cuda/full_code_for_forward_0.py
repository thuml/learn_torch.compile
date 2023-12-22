
def __guard_0_for_resume_in_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 139642392454448)) \
        and (L['self'].training == False) \
        and (hasattr(L['labels'], '_dynamo_dynamic_indices') == False) \
        and (___check_type_id(L['___stack0'], 168900864)) \
        and (___check_obj_id(L['return_dict'], 7677664)) \
        and (___check_obj_id(L['___stack0'].attentions, 7628576)) \
        and (___check_obj_id(L['___stack0'].hidden_states, 7628576)) \
        and (___check_obj_id(L['___stack0'].pooler_output, 7628576)) \
        and (___check_obj_id(L['___stack0'].global_attentions, 7628576)) \
        and (hasattr(L['___stack0'].last_hidden_state, '_dynamo_dynamic_indices') == False) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139639130168848))) \
        and (___compile_config_hash() == 'd23842bdb1f875b062b4abc655654038') \
        and (___check_obj_id(G['gelu'], 139639222992752)) \
        and (G['gelu'].training == True) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_tensors(L['labels'], L['___stack0'].last_hidden_state, tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_8*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_8(*args, **kwargs):
    pass

def __transformed_code_0_for_resume_in_forward(___stack0, self, labels, return_dict):
    attention_mask = None; global_attention_mask = None; head_mask = None; input_ids = None; inputs_embeds = None; loss_fct = None; masked_lm_loss = None; output = None; output_attentions = None; output_hidden_states = None; outputs = None; position_ids = None; prediction_scores = None; sequence_output = None; token_type_ids = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_8(___stack0.last_hidden_state, labels)
    import importlib
    return importlib.import_module(
        'transformers.models.longformer.modeling_longformer'
        ).LongformerMaskedLMOutput(loss=graph_out_0[0], logits=graph_out_0[1],
        hidden_states=___stack0.hidden_states, attentions=___stack0.attentions,
        global_attentions=___stack0.global_attentions)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_48_2(___stack0, self, labels, return_dict):
    return_dict = self.config.use_return_dict
    outputs = self.longformer(input_ids, attention_mask=attention_mask,
        global_attention_mask=global_attention_mask, head_mask=head_mask,
        token_type_ids=token_type_ids, position_ids=position_ids, inputs_embeds
        =inputs_embeds, output_attentions=output_attentions,
        output_hidden_states=output_hidden_states, return_dict=return_dict)
    sequence_output = outputs[0]
    prediction_scores = self.lm_head(sequence_output)
    masked_lm_loss = None
    if labels is not None:
        loss_fct = CrossEntropyLoss()
        labels = labels.to(prediction_scores.device)
        masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.
            vocab_size), labels.view(-1))
    if not return_dict:
        output = (prediction_scores,) + outputs[slice(2, None)]
        if masked_lm_loss is not None:
            return (masked_lm_loss,) + output
        return output
    return LongformerMaskedLMOutput(loss=masked_lm_loss, logits=
        prediction_scores, hidden_states=outputs.hidden_states, attentions=
        outputs.attentions, global_attentions=outputs.global_attentions)

def transformed___resume_at_48_2(___stack0, self, labels, return_dict):
    L = {"___stack0": ___stack0, "self": self, "labels": labels, "return_dict": return_dict}
    if __guard_0_for_resume_in_forward(L):
        return __transformed_code_0_for_resume_in_forward(___stack0, self, labels, return_dict)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_48_2(___stack0, self, labels, return_dict)

#============ end of __resume_at_48_2 ============#

def __guard_0_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 139642392454448)) \
        and (L['self'].training == False) \
        and (___check_obj_id(L['head_mask'], 7628576)) \
        and (hasattr(L['input_ids'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['return_dict'], 7628576)) \
        and (___check_obj_id(L['position_ids'], 7628576)) \
        and (___check_obj_id(L['inputs_embeds'], 7628576)) \
        and (___check_obj_id(L['attention_mask'], 7628576)) \
        and (___check_obj_id(L['token_type_ids'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7628576)) \
        and (___check_obj_id(L['output_hidden_states'], 7628576)) \
        and (___check_obj_id(L['global_attention_mask'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139639130168848))) \
        and (___compile_config_hash() == 'd23842bdb1f875b062b4abc655654038') \
        and (not ___needs_nopython()) \
        and (___check_tensors(L['input_ids'], tensor_check_names=tensor_check_names))

def __transformed_code_0_for_forward(self, input_ids, attention_mask, global_attention_mask, head_mask, token_type_ids, position_ids, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict):
    loss_fct = None; masked_lm_loss = None; output = None; outputs = None; prediction_scores = None; sequence_output = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    return_dict = self.config.use_return_dict
    return __resume_at_48_2(self.longformer(input_ids, attention_mask=
        attention_mask, global_attention_mask=global_attention_mask, head_mask=
        head_mask, token_type_ids=token_type_ids, position_ids=position_ids,
        inputs_embeds=inputs_embeds, output_attentions=output_attentions,
        output_hidden_states=output_hidden_states, return_dict=self.config.
        use_return_dict), self, labels, return_dict)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def forward(self, input_ids, attention_mask, global_attention_mask, head_mask, token_type_ids, position_ids, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict):
    'Failed to decompile.'

def transformed_forward(self, input_ids, attention_mask, global_attention_mask, head_mask, token_type_ids, position_ids, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict):
    L = {"self": self, "input_ids": input_ids, "attention_mask": attention_mask, "global_attention_mask": global_attention_mask, "head_mask": head_mask, "token_type_ids": token_type_ids, "position_ids": position_ids, "inputs_embeds": inputs_embeds, "labels": labels, "output_attentions": output_attentions, "output_hidden_states": output_hidden_states, "return_dict": return_dict}
    if __guard_0_for_forward(L):
        return __transformed_code_0_for_forward(self, input_ids, attention_mask, global_attention_mask, head_mask, token_type_ids, position_ids, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return forward(self, input_ids, attention_mask, global_attention_mask, head_mask, token_type_ids, position_ids, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)

#============ end of forward ============#
