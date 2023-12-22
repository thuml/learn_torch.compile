def __transformed_code_1_for_resume_in_forward(___stack0, self, labels, return_dict):
    attention_mask = None; cross_attn_head_mask = None; decoder_attention_mask = None; decoder_head_mask = None; decoder_input_ids = None; decoder_inputs_embeds = None; encoder_outputs = None; head_mask = None; input_ids = None; inputs_embeds = None; lm_logits = None; loss_fct = None; masked_lm_loss = None; output = None; output_attentions = None; output_hidden_states = None; outputs = None; past_key_values = None; use_cache = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function

    graph_out_0 = __compiled_fn_53(___stack0.last_hidden_state, labels)
    import importlib
    return importlib.import_module('transformers.modeling_outputs'
        ).Seq2SeqLMOutput(loss=graph_out_0[0], logits=graph_out_0[1],
        past_key_values=___stack0.past_key_values, decoder_hidden_states=
        ___stack0.decoder_hidden_states, decoder_attentions=___stack0.
        decoder_attentions, cross_attentions=___stack0.cross_attentions,
        encoder_last_hidden_state=___stack0.encoder_last_hidden_state,
        encoder_hidden_states=___stack0.encoder_hidden_states,
        encoder_attentions=___stack0.encoder_attentions)
