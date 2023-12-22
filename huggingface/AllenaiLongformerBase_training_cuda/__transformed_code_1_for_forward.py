def __transformed_code_1_for_forward(self, input_ids, attention_mask, global_attention_mask, head_mask, token_type_ids, position_ids, inputs_embeds, output_attentions, output_hidden_states, return_dict):
    device = None; embedding_output = None; encoder_outputs = None; extended_attention_mask = None; input_shape = None; padding_len = None; pooled_output = None; sequence_output = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function

    graph_out_0 = __compiled_fn_6(input_ids)
    return __resume_at_346_7(self.encoder(graph_out_0[0], attention_mask=
        graph_out_0[1], head_mask=head_mask, padding_len=0, output_attentions=
        self.config.output_attentions, output_hidden_states=self.config.
        output_hidden_states, return_dict=return_dict), self, return_dict)
