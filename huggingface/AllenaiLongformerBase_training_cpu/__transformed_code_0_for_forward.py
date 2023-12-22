def __transformed_code_0_for_forward(self, input_ids, attention_mask, global_attention_mask, head_mask, token_type_ids, position_ids, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict):
    loss_fct = None; masked_lm_loss = None; output = None; outputs = None; prediction_scores = None; sequence_output = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function

    return_dict = self.config.use_return_dict
    return __resume_at_48_5(self.longformer(input_ids, attention_mask=
        attention_mask, global_attention_mask=global_attention_mask, head_mask=
        head_mask, token_type_ids=token_type_ids, position_ids=position_ids,
        inputs_embeds=inputs_embeds, output_attentions=output_attentions,
        output_hidden_states=output_hidden_states, return_dict=self.config.
        use_return_dict), self, labels, return_dict)
