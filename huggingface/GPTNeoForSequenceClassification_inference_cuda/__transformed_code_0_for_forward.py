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
