def __transformed_code_0_for_forward_pass(self, mod, inputs, collect_outputs):
    graph_out_0 = __compiled_fn_0(inputs['labels'], inputs['input_ids'])
    import importlib
    return importlib.import_module('transformers.modeling_outputs'
        ).Seq2SeqLMOutput(loss=graph_out_0[0], logits=graph_out_0[1],
        past_key_values=None, decoder_hidden_states=None, decoder_attentions=
        None, cross_attentions=None, encoder_last_hidden_state=graph_out_0[2],
        encoder_hidden_states=None, encoder_attentions=None)
