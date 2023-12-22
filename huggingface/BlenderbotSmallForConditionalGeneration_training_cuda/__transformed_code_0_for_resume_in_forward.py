def __transformed_code_0_for_resume_in_forward(___stack0, self, attention_mask, decoder_input_ids, decoder_attention_mask, decoder_head_mask, cross_attn_head_mask, past_key_values, decoder_inputs_embeds, use_cache, output_attentions, output_hidden_states, return_dict):
    decoder_outputs = None; head_mask = None; input_ids = None; inputs_embeds = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function

    encoder_outputs = ___stack0
    return __resume_at_226_19(self.decoder(input_ids=decoder_input_ids,
        attention_mask=decoder_attention_mask, encoder_hidden_states=___stack0.
        last_hidden_state, encoder_attention_mask=attention_mask, head_mask=
        decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask,
        past_key_values=past_key_values, inputs_embeds=decoder_inputs_embeds,
        use_cache=use_cache, output_attentions=output_attentions,
        output_hidden_states=output_hidden_states, return_dict=return_dict),
        return_dict, encoder_outputs)
