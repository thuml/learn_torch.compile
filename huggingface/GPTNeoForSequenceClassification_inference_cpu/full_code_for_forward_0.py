
def __guard_0_for_resume_in_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 140286246402656)) \
        and (L['self'].training == False) \
        and (hasattr(L['labels'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['return_dict'], 7677664)) \
        and (hasattr(L['pooled_logits'], '_dynamo_dynamic_indices') == False) \
        and (___check_type_id(L['transformer_outputs'], 157671024)) \
        and (___check_obj_id(L['transformer_outputs'].attentions, 7628576)) \
        and (___check_obj_id(L['transformer_outputs'].hidden_states, 7628576)) \
        and (___check_type_id(L['transformer_outputs'].past_key_values, 7617760)) \
        and (len(L['transformer_outputs'].past_key_values) == 24) \
        and (hasattr(L['transformer_outputs'].last_hidden_state, '_dynamo_dynamic_indices') == False) \
        and (___check_type_id(L['transformer_outputs'].past_key_values[0], 7617760)) \
        and (len(L['transformer_outputs'].past_key_values[0]) == 2) \
        and (___check_type_id(L['transformer_outputs'].past_key_values[1], 7617760)) \
        and (len(L['transformer_outputs'].past_key_values[1]) == 2) \
        and (___check_type_id(L['transformer_outputs'].past_key_values[2], 7617760)) \
        and (len(L['transformer_outputs'].past_key_values[2]) == 2) \
        and (___check_type_id(L['transformer_outputs'].past_key_values[3], 7617760)) \
        and (len(L['transformer_outputs'].past_key_values[3]) == 2) \
        and (___check_type_id(L['transformer_outputs'].past_key_values[4], 7617760)) \
        and (len(L['transformer_outputs'].past_key_values[4]) == 2) \
        and (___check_type_id(L['transformer_outputs'].past_key_values[5], 7617760)) \
        and (len(L['transformer_outputs'].past_key_values[5]) == 2) \
        and (___check_type_id(L['transformer_outputs'].past_key_values[6], 7617760)) \
        and (len(L['transformer_outputs'].past_key_values[6]) == 2) \
        and (___check_type_id(L['transformer_outputs'].past_key_values[7], 7617760)) \
        and (len(L['transformer_outputs'].past_key_values[7]) == 2) \
        and (___check_type_id(L['transformer_outputs'].past_key_values[8], 7617760)) \
        and (len(L['transformer_outputs'].past_key_values[8]) == 2) \
        and (___check_type_id(L['transformer_outputs'].past_key_values[9], 7617760)) \
        and (len(L['transformer_outputs'].past_key_values[9]) == 2) \
        and (___check_type_id(L['transformer_outputs'].past_key_values[10], 7617760)) \
        and (len(L['transformer_outputs'].past_key_values[10]) == 2) \
        and (___check_type_id(L['transformer_outputs'].past_key_values[11], 7617760)) \
        and (len(L['transformer_outputs'].past_key_values[11]) == 2) \
        and (___check_type_id(L['transformer_outputs'].past_key_values[12], 7617760)) \
        and (len(L['transformer_outputs'].past_key_values[12]) == 2) \
        and (___check_type_id(L['transformer_outputs'].past_key_values[13], 7617760)) \
        and (len(L['transformer_outputs'].past_key_values[13]) == 2) \
        and (___check_type_id(L['transformer_outputs'].past_key_values[14], 7617760)) \
        and (len(L['transformer_outputs'].past_key_values[14]) == 2) \
        and (___check_type_id(L['transformer_outputs'].past_key_values[15], 7617760)) \
        and (len(L['transformer_outputs'].past_key_values[15]) == 2) \
        and (___check_type_id(L['transformer_outputs'].past_key_values[16], 7617760)) \
        and (len(L['transformer_outputs'].past_key_values[16]) == 2) \
        and (___check_type_id(L['transformer_outputs'].past_key_values[17], 7617760)) \
        and (len(L['transformer_outputs'].past_key_values[17]) == 2) \
        and (___check_type_id(L['transformer_outputs'].past_key_values[18], 7617760)) \
        and (len(L['transformer_outputs'].past_key_values[18]) == 2) \
        and (___check_type_id(L['transformer_outputs'].past_key_values[19], 7617760)) \
        and (len(L['transformer_outputs'].past_key_values[19]) == 2) \
        and (___check_type_id(L['transformer_outputs'].past_key_values[20], 7617760)) \
        and (len(L['transformer_outputs'].past_key_values[20]) == 2) \
        and (___check_type_id(L['transformer_outputs'].past_key_values[21], 7617760)) \
        and (len(L['transformer_outputs'].past_key_values[21]) == 2) \
        and (___check_type_id(L['transformer_outputs'].past_key_values[22], 7617760)) \
        and (len(L['transformer_outputs'].past_key_values[22]) == 2) \
        and (___check_type_id(L['transformer_outputs'].past_key_values[23], 7617760)) \
        and (len(L['transformer_outputs'].past_key_values[23]) == 2) \
        and (hasattr(L['transformer_outputs'].past_key_values[0][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[0][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[1][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[1][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[2][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[2][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[3][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[3][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[4][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[4][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[5][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[5][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[6][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[6][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[7][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[7][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[8][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[8][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[9][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[9][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[10][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[10][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[11][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[11][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[12][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[12][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[13][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[13][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[14][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[14][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[15][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[15][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[16][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[16][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[17][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[17][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[18][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[18][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[19][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[19][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[20][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[20][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[21][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[21][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[22][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[22][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[23][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['transformer_outputs'].past_key_values[23][1], '_dynamo_dynamic_indices') == False) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140282968858128))) \
        and (___compile_config_hash() == 'f7d79c5506973e8904041a594c8f5fe8') \
        and (___check_tensors(L['labels'], L['pooled_logits'], L['transformer_outputs'].last_hidden_state, L['transformer_outputs'].past_key_values[0][0], L['transformer_outputs'].past_key_values[0][1], L['transformer_outputs'].past_key_values[1][0], L['transformer_outputs'].past_key_values[1][1], L['transformer_outputs'].past_key_values[2][0], L['transformer_outputs'].past_key_values[2][1], L['transformer_outputs'].past_key_values[3][0], L['transformer_outputs'].past_key_values[3][1], L['transformer_outputs'].past_key_values[4][0], L['transformer_outputs'].past_key_values[4][1], L['transformer_outputs'].past_key_values[5][0], L['transformer_outputs'].past_key_values[5][1], L['transformer_outputs'].past_key_values[6][0], L['transformer_outputs'].past_key_values[6][1], L['transformer_outputs'].past_key_values[7][0], L['transformer_outputs'].past_key_values[7][1], L['transformer_outputs'].past_key_values[8][0], L['transformer_outputs'].past_key_values[8][1], L['transformer_outputs'].past_key_values[9][0], L['transformer_outputs'].past_key_values[9][1], L['transformer_outputs'].past_key_values[10][0], L['transformer_outputs'].past_key_values[10][1], L['transformer_outputs'].past_key_values[11][0], L['transformer_outputs'].past_key_values[11][1], L['transformer_outputs'].past_key_values[12][0], L['transformer_outputs'].past_key_values[12][1], L['transformer_outputs'].past_key_values[13][0], L['transformer_outputs'].past_key_values[13][1], L['transformer_outputs'].past_key_values[14][0], L['transformer_outputs'].past_key_values[14][1], L['transformer_outputs'].past_key_values[15][0], L['transformer_outputs'].past_key_values[15][1], L['transformer_outputs'].past_key_values[16][0], L['transformer_outputs'].past_key_values[16][1], L['transformer_outputs'].past_key_values[17][0], L['transformer_outputs'].past_key_values[17][1], L['transformer_outputs'].past_key_values[18][0], L['transformer_outputs'].past_key_values[18][1], L['transformer_outputs'].past_key_values[19][0], L['transformer_outputs'].past_key_values[19][1], L['transformer_outputs'].past_key_values[20][0], L['transformer_outputs'].past_key_values[20][1], L['transformer_outputs'].past_key_values[21][0], L['transformer_outputs'].past_key_values[21][1], L['transformer_outputs'].past_key_values[22][0], L['transformer_outputs'].past_key_values[22][1], L['transformer_outputs'].past_key_values[23][0], L['transformer_outputs'].past_key_values[23][1], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_4*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_4(*args, **kwargs):
    pass

def __transformed_code_0_for_resume_in_forward(self, labels, return_dict, transformer_outputs, pooled_logits, loss):
    attention_mask = None; batch_size = None; head_mask = None; hidden_states = None; input_ids = None; inputs_embeds = None; logits = None; loss_fct = None; output = None; output_attentions = None; output_hidden_states = None; past_key_values = None; position_ids = None; sequence_length = None; sequence_lengths = None; token_type_ids = None; use_cache = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_4(pooled_logits, labels)
    import importlib
    return importlib.import_module('transformers.modeling_outputs'
        ).SequenceClassifierOutputWithPast(loss=graph_out_0[0], logits=
        pooled_logits, past_key_values=transformer_outputs.past_key_values,
        hidden_states=transformer_outputs.hidden_states, attentions=
        transformer_outputs.attentions)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_344_3(self, labels, return_dict, transformer_outputs, pooled_logits, loss):
    return_dict = self.config.use_return_dict
    transformer_outputs = self.transformer(input_ids, past_key_values=
        past_key_values, attention_mask=attention_mask, token_type_ids=
        token_type_ids, position_ids=position_ids, head_mask=head_mask,
        inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=
        output_attentions, output_hidden_states=output_hidden_states,
        return_dict=return_dict)
    hidden_states = transformer_outputs[0]
    logits = self.score(hidden_states)
    if input_ids is not None:
        batch_size = input_ids.shape[slice(None, 2)][0]
        sequence_length = input_ids.shape[slice(None, 2)][1]
    else:
        batch_size = inputs_embeds.shape[slice(None, 2)][0]
        sequence_length = inputs_embeds.shape[slice(None, 2)][1]
    if self.config.pad_token_id is None:
        if batch_size != 1:
            raise ValueError(
                'Cannot handle batch sizes > 1 if no padding token is defined.')
    if self.config.pad_token_id is None:
        sequence_lengths = -1
        pooled_logits = logits[torch.arange(batch_size, device=logits.device),
            sequence_lengths]
    elif input_ids is not None:
        sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).long(
            ).argmax(-1) - 1).to(logits.device)
        pooled_logits = logits[torch.arange(batch_size, device=logits.device),
            sequence_lengths]
    else:
        sequence_lengths = -1
        logger.warning(str(self.__class__.__name__) +
            ' will not detect padding tokens in `inputs_embeds`. Results may be unexpected if using padding tokens in conjunction with `inputs_embeds.`'
            )
        pooled_logits = logits[torch.arange(batch_size, device=logits.device),
            sequence_lengths]
    loss = None
    if labels is not None:
        if self.config.problem_type is None:
            if self.num_labels == 1:
                self.config.problem_type = 'regression'
            elif self.num_labels > 1:
                if not labels.dtype == torch.long:
                    if labels.dtype == torch.int:
                        self.config.problem_type = 'single_label_classification'
                    else:
                        self.config.problem_type = 'multi_label_classification'
                else:
                    self.config.problem_type = 'single_label_classification'
            else:
                self.config.problem_type = 'multi_label_classification'
        if self.config.problem_type == 'regression':
            loss_fct = MSELoss()
            if self.num_labels == 1:
                loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
            else:
                loss = loss_fct(pooled_logits, labels)
        elif self.config.problem_type == 'single_label_classification':
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.
                view(-1))
        elif self.config.problem_type == 'multi_label_classification':
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(pooled_logits, labels)
    if not return_dict:
        output = (pooled_logits,) + transformer_outputs[slice(1, None)]
        if loss is not None:
            return (loss,) + output
        return output
    return SequenceClassifierOutputWithPast(loss=loss, logits=pooled_logits,
        past_key_values=transformer_outputs.past_key_values, hidden_states=
        transformer_outputs.hidden_states, attentions=transformer_outputs.
        attentions)

def transformed___resume_at_344_3(self, labels, return_dict, transformer_outputs, pooled_logits, loss):
    L = {"self": self, "labels": labels, "return_dict": return_dict, "transformer_outputs": transformer_outputs, "pooled_logits": pooled_logits, "loss": loss}
    if __guard_0_for_resume_in_forward(L):
        return __transformed_code_0_for_resume_in_forward(self, labels, return_dict, transformer_outputs, pooled_logits, loss)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_344_3(self, labels, return_dict, transformer_outputs, pooled_logits, loss)

#============ end of __resume_at_344_3 ============#

def __guard_0_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 140286246402656)) \
        and (L['self'].training == False) \
        and (___check_type_id(L['labels'], 101685024)) \
        and (hasattr(L['labels'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['head_mask'], 7628576)) \
        and (___check_type_id(L['input_ids'], 101685024)) \
        and (hasattr(L['input_ids'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['use_cache'], 7628576)) \
        and (___check_obj_id(L['return_dict'], 7628576)) \
        and (___check_obj_id(L['position_ids'], 7628576)) \
        and (___check_obj_id(L['inputs_embeds'], 7628576)) \
        and (___check_obj_id(L['attention_mask'], 7628576)) \
        and (___check_obj_id(L['token_type_ids'], 7628576)) \
        and (___check_obj_id(L['past_key_values'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7628576)) \
        and (___check_obj_id(L['output_hidden_states'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140282968858128))) \
        and (___compile_config_hash() == 'f7d79c5506973e8904041a594c8f5fe8') \
        and (not ___needs_nopython()) \
        and (___check_type_id(G['torch'].long, 140288230364928)) \
        and (G['torch'].long == torch.int64) \
        and (___check_type_id(G['torch'].float32, 140288230364928)) \
        and (G['torch'].float32 == torch.float32) \
        and (___check_type_id(G['__import_transformers_dot_activations'].math.pi, 7644160)) \
        and (G['__import_transformers_dot_activations'].math.pi == 3.141592653589793) \
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
        and (___check_obj_id(L['self'].transformer.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].transformer.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['self'].transformer.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['self'].transformer.forward.__defaults__[6], 7628576)) \
        and (___check_obj_id(L['self'].transformer.forward.__defaults__[7], 7628576)) \
        and (___check_obj_id(L['self'].transformer.forward.__defaults__[8], 7628576)) \
        and (___check_obj_id(L['self'].transformer.forward.__defaults__[9], 7628576)) \
        and (___check_obj_id(L['self'].transformer.forward.__defaults__[10], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[0].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[0].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[0].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[0].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[0].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[1].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[1].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[1].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[1].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[1].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[2].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[2].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[2].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[2].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[2].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[3].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[3].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[3].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[3].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[3].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[4].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[4].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[4].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[4].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[4].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[5].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[5].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[5].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[5].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[5].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[6].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[6].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[6].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[6].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[6].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[7].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[7].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[7].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[7].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[7].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[8].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[8].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[8].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[8].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[8].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[9].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[9].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[9].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[9].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[9].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.get_head_mask.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[10].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[10].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[10].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[10].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[10].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[11].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[11].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[11].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[11].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[11].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[12].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[12].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[12].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[12].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[12].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[13].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[13].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[13].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[13].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[13].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[14].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[14].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[14].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[14].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[14].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[15].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[15].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[15].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[15].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[15].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[16].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[16].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[16].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[16].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[16].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[17].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[17].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[17].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[17].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[17].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[18].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[18].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[18].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[18].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[18].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[19].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[19].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[19].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[19].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[19].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[20].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[20].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[20].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[20].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[20].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[21].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[21].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[21].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[21].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[21].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[22].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[22].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[22].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[22].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[22].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[23].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[23].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[23].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[23].forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[23].forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[0].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[0].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[0].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[0].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[0].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[1].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[1].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[1].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[1].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[1].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[2].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[2].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[2].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[2].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[2].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[3].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[3].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[3].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[3].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[3].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[4].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[4].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[4].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[4].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[4].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[5].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[5].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[5].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[5].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[5].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[6].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[6].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[6].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[6].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[6].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[7].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[7].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[7].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[7].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[7].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[8].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[8].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[8].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[8].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[8].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[9].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[9].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[9].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[9].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[9].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[10].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[10].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[10].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[10].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[10].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[11].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[11].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[11].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[11].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[11].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[12].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[12].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[12].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[12].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[12].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[13].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[13].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[13].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[13].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[13].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[14].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[14].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[14].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[14].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[14].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[15].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[15].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[15].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[15].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[15].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[16].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[16].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[16].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[16].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[16].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[17].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[17].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[17].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[17].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[17].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[18].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[18].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[18].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[18].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[18].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[19].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[19].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[19].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[19].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[19].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[20].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[20].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[20].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[20].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[20].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[21].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[21].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[21].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[21].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[21].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[22].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[22].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[22].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[22].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[22].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[23].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[23].attn.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[23].attn.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[23].attn.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[23].attn.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[0].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[0].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[1].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[1].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[2].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[2].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[3].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[3].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[4].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[4].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[5].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[5].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[6].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[6].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[7].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[7].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[8].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[8].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[9].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[9].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[10].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[10].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[11].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[11].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[12].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[12].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[13].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[13].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[14].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[14].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[15].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[15].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[16].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[16].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[17].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[17].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[18].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[18].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[19].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[19].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[20].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[20].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[21].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[21].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[22].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[22].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[23].attn.attention._attn.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[23].attn.attention._attn.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[0].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[0].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[0].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[0].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[0].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[1].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[1].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[1].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[1].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[1].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[2].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[2].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[2].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[2].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[2].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[3].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[3].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[3].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[3].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[3].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[4].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[4].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[4].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[4].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[4].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[5].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[5].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[5].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[5].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[5].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[6].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[6].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[6].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[6].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[6].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[7].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[7].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[7].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[7].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[7].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[8].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[8].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[8].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[8].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[8].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[9].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[9].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[9].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[9].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[9].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[10].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[10].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[10].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[10].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[10].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[11].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[11].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[11].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[11].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[11].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[12].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[12].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[12].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[12].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[12].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[13].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[13].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[13].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[13].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[13].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[14].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[14].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[14].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[14].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[14].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[15].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[15].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[15].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[15].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[15].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[16].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[16].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[16].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[16].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[16].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[17].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[17].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[17].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[17].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[17].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[18].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[18].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[18].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[18].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[18].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[19].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[19].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[19].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[19].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[19].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[20].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[20].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[20].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[20].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[20].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[21].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[21].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[21].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[21].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[21].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[22].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[22].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[22].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[22].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[22].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[23].attn.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[23].attn.attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[23].attn.attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].transformer.h[23].attn.attention.forward.__defaults__[3], 7677632)) \
        and (___check_obj_id(L['self'].transformer.h[23].attn.attention.forward.__defaults__[4], 7677632)) \
        and (___check_tensors(L['labels'], L['input_ids'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_2*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_2(*args, **kwargs):
    pass

def __transformed_code_0_for_forward(self, input_ids, past_key_values, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict):
    batch_size = None; hidden_states = None; logits = None; loss_fct = None; output = None; sequence_length = None; sequence_lengths = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_2(input_ids)
    import importlib
    loss = None
    pooled_logits = graph_out_0[49]
    transformer_outputs = importlib.import_module('transformers.modeling_outputs'
        ).BaseModelOutputWithPast(last_hidden_state=graph_out_0[0],
        past_key_values=((graph_out_0[1], graph_out_0[2]), (graph_out_0[3],
        graph_out_0[4]), (graph_out_0[5], graph_out_0[6]), (graph_out_0[7],
        graph_out_0[8]), (graph_out_0[9], graph_out_0[10]), (graph_out_0[11],
        graph_out_0[12]), (graph_out_0[13], graph_out_0[14]), (graph_out_0[15],
        graph_out_0[16]), (graph_out_0[17], graph_out_0[18]), (graph_out_0[19],
        graph_out_0[20]), (graph_out_0[21], graph_out_0[22]), (graph_out_0[23],
        graph_out_0[24]), (graph_out_0[25], graph_out_0[26]), (graph_out_0[27],
        graph_out_0[28]), (graph_out_0[29], graph_out_0[30]), (graph_out_0[31],
        graph_out_0[32]), (graph_out_0[33], graph_out_0[34]), (graph_out_0[35],
        graph_out_0[36]), (graph_out_0[37], graph_out_0[38]), (graph_out_0[39],
        graph_out_0[40]), (graph_out_0[41], graph_out_0[42]), (graph_out_0[43],
        graph_out_0[44]), (graph_out_0[45], graph_out_0[46]), (graph_out_0[47],
        graph_out_0[48])), hidden_states=None, attentions=None)
    return_dict = self.config.use_return_dict
    self.config.problem_type = 'single_label_classification'
    return __resume_at_344_3(self, labels, return_dict, transformer_outputs,
        pooled_logits, loss)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def forward(self, input_ids, past_key_values, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict):
    'Failed to decompile.'

def transformed_forward(self, input_ids, past_key_values, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict):
    L = {"self": self, "input_ids": input_ids, "past_key_values": past_key_values, "attention_mask": attention_mask, "token_type_ids": token_type_ids, "position_ids": position_ids, "head_mask": head_mask, "inputs_embeds": inputs_embeds, "labels": labels, "use_cache": use_cache, "output_attentions": output_attentions, "output_hidden_states": output_hidden_states, "return_dict": return_dict}
    if __guard_0_for_forward(L):
        return __transformed_code_0_for_forward(self, input_ids, past_key_values, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return forward(self, input_ids, past_key_values, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict)

#============ end of forward ============#
