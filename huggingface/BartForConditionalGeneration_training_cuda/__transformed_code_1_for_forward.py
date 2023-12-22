def __transformed_code_1_for_forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, head_mask, decoder_head_mask, cross_attn_head_mask, encoder_outputs, past_key_values, inputs_embeds, decoder_inputs_embeds, use_cache, output_attentions, output_hidden_states, return_dict):
    decoder_outputs = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function

    output_hidden_states = self.config.output_hidden_states
    output_attentions = self.config.output_attentions
    return __resume_at_162_7(self.encoder(input_ids=input_ids, attention_mask=
        attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds,
        output_attentions=self.config.output_attentions, output_hidden_states=
        self.config.output_hidden_states, return_dict=return_dict), self,
        attention_mask, decoder_input_ids, decoder_attention_mask,
        decoder_head_mask, cross_attn_head_mask, past_key_values,
        decoder_inputs_embeds, use_cache, output_attentions,
        output_hidden_states, return_dict)
