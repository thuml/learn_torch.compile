def __transformed_code_0_for_forward(self, input_ids, bbox, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict):
    loss_fct = None; output = None; pooled_output = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function

    graph_out_0 = __compiled_fn_2(input_ids)
    import importlib
    loss = None
    logits = graph_out_0[2]
    outputs = importlib.import_module('transformers.modeling_outputs'
        ).BaseModelOutputWithPoolingAndCrossAttentions(last_hidden_state=
        graph_out_0[0], pooler_output=graph_out_0[1], hidden_states=None,
        past_key_values=None, attentions=None, cross_attentions=None)
    return_dict = self.config.use_return_dict
    self.config.problem_type = 'single_label_classification'
    return __resume_at_164_3(self, labels, return_dict, outputs, logits, loss)
