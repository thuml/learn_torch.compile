def __transformed_code_0_for_forward(self, input_ids, attention_mask, head_mask, past_key_values, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict):
    logits = None; loss = None; loss_fct = None; output = None; outputs = None; shift_labels = None; shift_logits = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function

    return_dict = self.config.use_return_dict
    return __resume_at_88_5(self.model.decoder(input_ids=input_ids,
        attention_mask=attention_mask, head_mask=head_mask, past_key_values=
        past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
        output_attentions=self.config.output_attentions, output_hidden_states=
        self.config.output_hidden_states, return_dict=self.config.
        use_return_dict), self, labels, return_dict)
