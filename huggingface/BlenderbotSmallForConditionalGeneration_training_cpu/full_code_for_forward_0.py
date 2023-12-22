
def __guard_1_for_resume_in_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 139889434003392)) \
        and (L['self'].training == True) \
        and (hasattr(L['labels'], '_dynamo_dynamic_indices') == False) \
        and (___check_type_id(L['___stack0'], 132373056)) \
        and (___check_obj_id(L['return_dict'], 7677664)) \
        and (___check_obj_id(L['___stack0'].past_key_values, 7628576)) \
        and (___check_obj_id(L['___stack0'].cross_attentions, 7628576)) \
        and (hasattr(L['___stack0'].last_hidden_state, '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['___stack0'].decoder_attentions, 7628576)) \
        and (___check_obj_id(L['___stack0'].encoder_attentions, 7628576)) \
        and (___check_obj_id(L['___stack0'].decoder_hidden_states, 7628576)) \
        and (___check_obj_id(L['___stack0'].encoder_hidden_states, 7628576)) \
        and (hasattr(L['___stack0'].encoder_last_hidden_state, '_dynamo_dynamic_indices') == False) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139886168940048))) \
        and (___compile_config_hash() == '07b52bff0c86f8e311705687144327df') \
        and (___check_tensors(L['labels'], L['___stack0'].last_hidden_state, L['___stack0'].encoder_last_hidden_state, tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_43*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_43(*args, **kwargs):
    pass

def __transformed_code_1_for_resume_in_forward(___stack0, self, labels, return_dict):
    attention_mask = None; cross_attn_head_mask = None; decoder_attention_mask = None; decoder_head_mask = None; decoder_input_ids = None; decoder_inputs_embeds = None; encoder_outputs = None; head_mask = None; input_ids = None; inputs_embeds = None; lm_logits = None; loss_fct = None; masked_lm_loss = None; output = None; output_attentions = None; output_hidden_states = None; outputs = None; past_key_values = None; use_cache = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_43(___stack0.last_hidden_state, labels)
    import importlib
    return importlib.import_module('transformers.modeling_outputs'
        ).Seq2SeqLMOutput(loss=graph_out_0[0], logits=graph_out_0[1],
        past_key_values=___stack0.past_key_values, decoder_hidden_states=
        ___stack0.decoder_hidden_states, decoder_attentions=___stack0.
        decoder_attentions, cross_attentions=___stack0.cross_attentions,
        encoder_last_hidden_state=___stack0.encoder_last_hidden_state,
        encoder_hidden_states=___stack0.encoder_hidden_states,
        encoder_attentions=___stack0.encoder_attentions)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_120_5(___stack0, self, labels, return_dict):
    return_dict = self.config.use_return_dict
    if labels is not None:
        if use_cache:
            logger.warning(
                'The `use_cache` argument is changed to `False` since `labels` is provided.'
                )
            use_cache = False
        else:
            use_cache = False
        if decoder_input_ids is None:
            if decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.
                    pad_token_id, self.config.decoder_start_token_id)
    outputs = self.model(input_ids, attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids, encoder_outputs=encoder_outputs,
        decoder_attention_mask=decoder_attention_mask, head_mask=head_mask,
        decoder_head_mask=decoder_head_mask, cross_attn_head_mask=
        cross_attn_head_mask, past_key_values=past_key_values, inputs_embeds=
        inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds, use_cache=
        use_cache, output_attentions=output_attentions, output_hidden_states=
        output_hidden_states, return_dict=return_dict)
    lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
    masked_lm_loss = None
    if labels is not None:
        loss_fct = CrossEntropyLoss()
        masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size),
            labels.view(-1))
    if not return_dict:
        output = (lm_logits,) + outputs[slice(1, None)]
        if masked_lm_loss is not None:
            return (masked_lm_loss,) + output
        return output
    return Seq2SeqLMOutput(loss=masked_lm_loss, logits=lm_logits,
        past_key_values=outputs.past_key_values, decoder_hidden_states=outputs.
        decoder_hidden_states, decoder_attentions=outputs.decoder_attentions,
        cross_attentions=outputs.cross_attentions, encoder_last_hidden_state=
        outputs.encoder_last_hidden_state, encoder_hidden_states=outputs.
        encoder_hidden_states, encoder_attentions=outputs.encoder_attentions)

def transformed___resume_at_120_5(___stack0, self, labels, return_dict):
    L = {"___stack0": ___stack0, "self": self, "labels": labels, "return_dict": return_dict}
    if __guard_1_for_resume_in_forward(L):
        return __transformed_code_1_for_resume_in_forward(___stack0, self, labels, return_dict)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_120_5(___stack0, self, labels, return_dict)

#============ end of __resume_at_120_5 ============#

def __guard_0_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 139889434003392)) \
        and (L['self'].training == True) \
        and (hasattr(L['labels'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['head_mask'], 7628576)) \
        and (hasattr(L['input_ids'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['use_cache'], 7628576)) \
        and (___check_obj_id(L['return_dict'], 7628576)) \
        and (___check_obj_id(L['inputs_embeds'], 7628576)) \
        and (___check_obj_id(L['attention_mask'], 7628576)) \
        and (___check_obj_id(L['encoder_outputs'], 7628576)) \
        and (___check_obj_id(L['past_key_values'], 7628576)) \
        and (___check_obj_id(L['decoder_head_mask'], 7628576)) \
        and (hasattr(L['decoder_input_ids'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['output_attentions'], 7628576)) \
        and (___check_obj_id(L['cross_attn_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_hidden_states'], 7628576)) \
        and (___check_obj_id(L['decoder_inputs_embeds'], 7628576)) \
        and (___check_obj_id(L['decoder_attention_mask'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139886168940048))) \
        and (___compile_config_hash() == '07b52bff0c86f8e311705687144327df') \
        and (not ___needs_nopython()) \
        and (___check_tensors(L['labels'], L['input_ids'], L['decoder_input_ids'], tensor_check_names=tensor_check_names))

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


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, head_mask, decoder_head_mask, cross_attn_head_mask, encoder_outputs, past_key_values, inputs_embeds, decoder_inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict):
    'Failed to decompile.'

def transformed_forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, head_mask, decoder_head_mask, cross_attn_head_mask, encoder_outputs, past_key_values, inputs_embeds, decoder_inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict):
    L = {"self": self, "input_ids": input_ids, "attention_mask": attention_mask, "decoder_input_ids": decoder_input_ids, "decoder_attention_mask": decoder_attention_mask, "head_mask": head_mask, "decoder_head_mask": decoder_head_mask, "cross_attn_head_mask": cross_attn_head_mask, "encoder_outputs": encoder_outputs, "past_key_values": past_key_values, "inputs_embeds": inputs_embeds, "decoder_inputs_embeds": decoder_inputs_embeds, "labels": labels, "use_cache": use_cache, "output_attentions": output_attentions, "output_hidden_states": output_hidden_states, "return_dict": return_dict}
    if __guard_0_for_forward(L):
        return __transformed_code_0_for_forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, head_mask, decoder_head_mask, cross_attn_head_mask, encoder_outputs, past_key_values, inputs_embeds, decoder_inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, head_mask, decoder_head_mask, cross_attn_head_mask, encoder_outputs, past_key_values, inputs_embeds, decoder_inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict)

#============ end of forward ============#
