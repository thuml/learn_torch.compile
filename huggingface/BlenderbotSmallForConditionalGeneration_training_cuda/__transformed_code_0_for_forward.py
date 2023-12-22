def __transformed_code_0_for_forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, head_mask, decoder_head_mask, cross_attn_head_mask, encoder_outputs, past_key_values, inputs_embeds, decoder_inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict):
    lm_logits = None; loss_fct = None; masked_lm_loss = None; output = None; outputs = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function

    return_dict = self.config.use_return_dict
    return __resume_at_120_5(self.model(input_ids, attention_mask=
        attention_mask, decoder_input_ids=decoder_input_ids, encoder_outputs=
        encoder_outputs, decoder_attention_mask=decoder_attention_mask,
        head_mask=head_mask, decoder_head_mask=decoder_head_mask,
        cross_attn_head_mask=cross_attn_head_mask, past_key_values=
        past_key_values, inputs_embeds=inputs_embeds, decoder_inputs_embeds=
        decoder_inputs_embeds, use_cache=False, output_attentions=
        output_attentions, output_hidden_states=output_hidden_states,
        return_dict=self.config.use_return_dict), self, labels, return_dict)
