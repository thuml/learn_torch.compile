def __transformed_code_0_for_resume_in_forward(self, labels, return_dict, transformer_outputs, pooled_logits, loss):
    attention_mask = None; batch_size = None; head_mask = None; hidden_states = None; input_ids = None; inputs_embeds = None; logits = None; loss_fct = None; output = None; output_attentions = None; output_hidden_states = None; past_key_values = None; position_ids = None; sequence_length = None; sequence_lengths = None; token_type_ids = None; use_cache = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function

    graph_out_0 = __compiled_fn_4(pooled_logits, labels)
    import importlib
    return importlib.import_module('transformers.modeling_outputs'
        ).SequenceClassifierOutputWithPast(loss=graph_out_0[0], logits=
        pooled_logits, past_key_values=transformer_outputs.past_key_values,
        hidden_states=transformer_outputs.hidden_states, attentions=
        transformer_outputs.attentions)
