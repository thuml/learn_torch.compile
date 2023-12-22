
def __guard_0_for_resume_in_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 140270499417488)) \
        and (L['self'].training == True) \
        and (hasattr(L['labels'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['logits'], '_dynamo_dynamic_indices') == False) \
        and (___check_type_id(L['outputs'], 128921808)) \
        and (___check_obj_id(L['return_dict'], 7677664)) \
        and (___check_obj_id(L['outputs'].attentions, 7628576)) \
        and (___check_obj_id(L['outputs'].hidden_states, 7628576)) \
        and (hasattr(L['outputs'].pooler_output, '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['outputs'].past_key_values, 7628576)) \
        and (___check_obj_id(L['outputs'].cross_attentions, 7628576)) \
        and (hasattr(L['outputs'].last_hidden_state, '_dynamo_dynamic_indices') == False) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140267302739472))) \
        and (___compile_config_hash() == '1e91b99c9698a2c5597469468e1c031b') \
        and (___check_tensors(L['labels'], L['logits'], L['outputs'].pooler_output, L['outputs'].last_hidden_state, tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_7*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_7(*args, **kwargs):
    pass

def __transformed_code_0_for_resume_in_forward(self, labels, return_dict, outputs, logits, loss):
    attention_mask = None; bbox = None; head_mask = None; input_ids = None; inputs_embeds = None; loss_fct = None; output = None; output_attentions = None; output_hidden_states = None; pooled_output = None; position_ids = None; token_type_ids = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_7(logits, labels)
    import importlib
    return importlib.import_module('transformers.modeling_outputs'
        ).SequenceClassifierOutput(loss=graph_out_0[0], logits=logits,
        hidden_states=outputs.hidden_states, attentions=outputs.attentions)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_164_6(self, labels, return_dict, outputs, logits, loss):
    return_dict = self.config.use_return_dict
    outputs = self.layoutlm(input_ids=input_ids, bbox=bbox, attention_mask=
        attention_mask, token_type_ids=token_type_ids, position_ids=
        position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
        output_attentions=output_attentions, output_hidden_states=
        output_hidden_states, return_dict=return_dict)
    pooled_output = outputs[1]
    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output)
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
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                loss = loss_fct(logits, labels)
        elif self.config.problem_type == 'single_label_classification':
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        elif self.config.problem_type == 'multi_label_classification':
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
    if not return_dict:
        output = (logits,) + outputs[slice(2, None)]
        if loss is not None:
            return (loss,) + output
        return output
    return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=
        outputs.hidden_states, attentions=outputs.attentions)

def transformed___resume_at_164_6(self, labels, return_dict, outputs, logits, loss):
    L = {"self": self, "labels": labels, "return_dict": return_dict, "outputs": outputs, "logits": logits, "loss": loss}
    if __guard_0_for_resume_in_forward(L):
        return __transformed_code_0_for_resume_in_forward(self, labels, return_dict, outputs, logits, loss)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_164_6(self, labels, return_dict, outputs, logits, loss)

#============ end of __resume_at_164_6 ============#

def __guard_0_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['bbox'], 7628576)) \
        and (___check_obj_id(L['self'], 140270499417488)) \
        and (L['self'].training == True) \
        and (___check_type_id(L['labels'], 72906416)) \
        and (hasattr(L['labels'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['head_mask'], 7628576)) \
        and (___check_type_id(L['input_ids'], 72906416)) \
        and (hasattr(L['input_ids'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['return_dict'], 7628576)) \
        and (___check_obj_id(L['position_ids'], 7628576)) \
        and (___check_obj_id(L['inputs_embeds'], 7628576)) \
        and (___check_obj_id(L['attention_mask'], 7628576)) \
        and (___check_obj_id(L['token_type_ids'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7628576)) \
        and (___check_obj_id(L['output_hidden_states'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140267302739472))) \
        and (___compile_config_hash() == '1e91b99c9698a2c5597469468e1c031b') \
        and (not ___needs_nopython()) \
        and (___check_type_id(G['torch'].long, 140272564279040)) \
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
        and (___check_obj_id(L['self'].layoutlm.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.forward.__defaults__[6], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.forward.__defaults__[7], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.forward.__defaults__[8], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.forward.__defaults__[9], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.forward.__defaults__[10], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.forward.__defaults__[11], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.forward.__defaults__[5], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.forward.__defaults__[6], 7677632)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.forward.__defaults__[7], 7677632)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.forward.__defaults__[8], 7677664)) \
        and (___check_obj_id(L['self'].layoutlm.embeddings.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.embeddings.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.embeddings.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.embeddings.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.embeddings.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[0].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[0].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[0].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[0].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[0].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[0].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[1].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[1].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[1].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[1].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[1].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[1].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[2].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[2].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[2].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[2].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[2].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[2].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[3].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[3].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[3].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[3].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[3].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[3].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[4].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[4].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[4].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[4].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[4].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[4].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[5].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[5].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[5].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[5].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[5].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[5].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[6].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[6].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[6].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[6].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[6].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[6].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[7].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[7].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[7].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[7].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[7].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[7].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[8].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[8].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[8].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[8].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[8].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[8].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[9].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[9].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[9].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[9].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[9].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[9].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[10].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[10].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[10].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[10].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[10].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[10].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[11].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[11].forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[11].forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[11].forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[11].forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[11].forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[0].attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[0].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[0].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[0].attention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[0].attention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[0].attention.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[1].attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[1].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[1].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[1].attention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[1].attention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[1].attention.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[2].attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[2].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[2].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[2].attention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[2].attention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[2].attention.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[3].attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[3].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[3].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[3].attention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[3].attention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[3].attention.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[4].attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[4].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[4].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[4].attention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[4].attention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[4].attention.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[5].attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[5].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[5].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[5].attention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[5].attention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[5].attention.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[6].attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[6].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[6].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[6].attention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[6].attention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[6].attention.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[7].attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[7].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[7].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[7].attention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[7].attention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[7].attention.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[8].attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[8].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[8].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[8].attention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[8].attention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[8].attention.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[9].attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[9].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[9].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[9].attention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[9].attention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[9].attention.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[10].attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[10].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[10].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[10].attention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[10].attention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[10].attention.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[11].attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[11].attention.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[11].attention.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[11].attention.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[11].attention.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[11].attention.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[0].attention.self.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[0].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[0].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[0].attention.self.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[0].attention.self.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[0].attention.self.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[1].attention.self.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[1].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[1].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[1].attention.self.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[1].attention.self.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[1].attention.self.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[2].attention.self.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[2].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[2].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[2].attention.self.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[2].attention.self.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[2].attention.self.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[3].attention.self.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[3].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[3].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[3].attention.self.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[3].attention.self.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[3].attention.self.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[4].attention.self.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[4].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[4].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[4].attention.self.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[4].attention.self.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[4].attention.self.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[5].attention.self.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[5].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[5].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[5].attention.self.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[5].attention.self.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[5].attention.self.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[6].attention.self.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[6].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[6].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[6].attention.self.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[6].attention.self.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[6].attention.self.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[7].attention.self.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[7].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[7].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[7].attention.self.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[7].attention.self.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[7].attention.self.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[8].attention.self.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[8].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[8].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[8].attention.self.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[8].attention.self.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[8].attention.self.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[9].attention.self.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[9].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[9].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[9].attention.self.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[9].attention.self.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[9].attention.self.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[10].attention.self.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[10].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[10].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[10].attention.self.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[10].attention.self.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[10].attention.self.forward.__defaults__[5], 7677632)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[11].attention.self.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[11].attention.self.forward.__defaults__[1], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[11].attention.self.forward.__defaults__[2], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[11].attention.self.forward.__defaults__[3], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[11].attention.self.forward.__defaults__[4], 7628576)) \
        and (___check_obj_id(L['self'].layoutlm.encoder.layer[11].attention.self.forward.__defaults__[5], 7677632)) \
        and (___check_tensors(L['labels'], L['input_ids'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_5*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_5(*args, **kwargs):
    pass

def __transformed_code_0_for_forward(self, input_ids, bbox, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict):
    loss_fct = None; output = None; pooled_output = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_5(input_ids)
    import importlib
    loss = None
    logits = graph_out_0[2]
    outputs = importlib.import_module('transformers.modeling_outputs'
        ).BaseModelOutputWithPoolingAndCrossAttentions(last_hidden_state=
        graph_out_0[0], pooler_output=graph_out_0[1], hidden_states=None,
        past_key_values=None, attentions=None, cross_attentions=None)
    return_dict = self.config.use_return_dict
    self.config.problem_type = 'single_label_classification'
    return __resume_at_164_6(self, labels, return_dict, outputs, logits, loss)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def forward(self, input_ids, bbox, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict):
    'Failed to decompile.'

def transformed_forward(self, input_ids, bbox, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict):
    L = {"self": self, "input_ids": input_ids, "bbox": bbox, "attention_mask": attention_mask, "token_type_ids": token_type_ids, "position_ids": position_ids, "head_mask": head_mask, "inputs_embeds": inputs_embeds, "labels": labels, "output_attentions": output_attentions, "output_hidden_states": output_hidden_states, "return_dict": return_dict}
    if __guard_0_for_forward(L):
        return __transformed_code_0_for_forward(self, input_ids, bbox, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return forward(self, input_ids, bbox, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)

#============ end of forward ============#
