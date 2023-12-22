
# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_226_23(___stack0, return_dict, encoder_outputs):
    'Failed to decompile.'

def transformed___resume_at_226_23(___stack0, return_dict, encoder_outputs):
    L = {"___stack0": ___stack0, "return_dict": return_dict, "encoder_outputs": encoder_outputs}
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_226_23(___stack0, return_dict, encoder_outputs)

#============ end of __resume_at_226_23 ============#

def __guard_0_for_resume_in_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 140346494400736)) \
        and (L['self'].training == True) \
        and (___check_type_id(L['___stack0'], 134668928)) \
        and (___check_obj_id(L['use_cache'], 7677632)) \
        and (___check_obj_id(L['return_dict'], 7677664)) \
        and (___check_obj_id(L['attention_mask'], 7628576)) \
        and (___check_obj_id(L['past_key_values'], 7628576)) \
        and (___check_obj_id(L['decoder_head_mask'], 7628576)) \
        and (hasattr(L['decoder_input_ids'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['output_attentions'], 7677632)) \
        and (___check_obj_id(L['___stack0'].attentions, 7628576)) \
        and (___check_obj_id(L['cross_attn_head_mask'], 7628576)) \
        and (___check_obj_id(L['output_hidden_states'], 7677632)) \
        and (___check_obj_id(L['decoder_inputs_embeds'], 7628576)) \
        and (___check_obj_id(L['decoder_attention_mask'], 7628576)) \
        and (___check_obj_id(L['___stack0'].hidden_states, 7628576)) \
        and (hasattr(L['___stack0'].last_hidden_state, '_dynamo_dynamic_indices') == False) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140343385792016))) \
        and (___compile_config_hash() == '9f64cb439de8127be15121eda92b9482') \
        and (not ___needs_nopython()) \
        and (___check_tensors(L['decoder_input_ids'], L['___stack0'].last_hidden_state, tensor_check_names=tensor_check_names))

def __transformed_code_0_for_resume_in_forward(___stack0, self, attention_mask, decoder_input_ids, decoder_attention_mask, decoder_head_mask, cross_attn_head_mask, past_key_values, decoder_inputs_embeds, use_cache, output_attentions, output_hidden_states, return_dict):
    decoder_outputs = None; head_mask = None; input_ids = None; inputs_embeds = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    encoder_outputs = ___stack0
    return __resume_at_226_23(self.decoder(input_ids=decoder_input_ids,
        attention_mask=decoder_attention_mask, encoder_hidden_states=___stack0.
        last_hidden_state, encoder_attention_mask=attention_mask, head_mask=
        decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask,
        past_key_values=past_key_values, inputs_embeds=decoder_inputs_embeds,
        use_cache=use_cache, output_attentions=output_attentions,
        output_hidden_states=output_hidden_states, return_dict=return_dict),
        return_dict, encoder_outputs)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_110_6(___stack0, self, attention_mask, decoder_input_ids, decoder_attention_mask, decoder_head_mask, cross_attn_head_mask, past_key_values, decoder_inputs_embeds, use_cache, output_attentions, output_hidden_states, return_dict):
    'Failed to decompile.'

def transformed___resume_at_110_6(___stack0, self, attention_mask, decoder_input_ids, decoder_attention_mask, decoder_head_mask, cross_attn_head_mask, past_key_values, decoder_inputs_embeds, use_cache, output_attentions, output_hidden_states, return_dict):
    L = {"___stack0": ___stack0, "self": self, "attention_mask": attention_mask, "decoder_input_ids": decoder_input_ids, "decoder_attention_mask": decoder_attention_mask, "decoder_head_mask": decoder_head_mask, "cross_attn_head_mask": cross_attn_head_mask, "past_key_values": past_key_values, "decoder_inputs_embeds": decoder_inputs_embeds, "use_cache": use_cache, "output_attentions": output_attentions, "output_hidden_states": output_hidden_states, "return_dict": return_dict}
    if __guard_0_for_resume_in_forward(L):
        return __transformed_code_0_for_resume_in_forward(___stack0, self, attention_mask, decoder_input_ids, decoder_attention_mask, decoder_head_mask, cross_attn_head_mask, past_key_values, decoder_inputs_embeds, use_cache, output_attentions, output_hidden_states, return_dict)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_110_6(___stack0, self, attention_mask, decoder_input_ids, decoder_attention_mask, decoder_head_mask, cross_attn_head_mask, past_key_values, decoder_inputs_embeds, use_cache, output_attentions, output_hidden_states, return_dict)

#============ end of __resume_at_110_6 ============#

def __guard_1_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 140346494400736)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['head_mask'], 7628576)) \
        and (hasattr(L['input_ids'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['use_cache'], 7677632)) \
        and (___check_obj_id(L['return_dict'], 7677664)) \
        and (___check_obj_id(L['inputs_embeds'], 7628576)) \
        and (___check_obj_id(L['attention_mask'], 7628576)) \
        and (___check_obj_id(L['encoder_outputs'], 7628576)) \
        and (___check_obj_id(L['output_attentions'], 7628576)) \
        and (___check_obj_id(L['output_hidden_states'], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140343385792016))) \
        and (___compile_config_hash() == '9f64cb439de8127be15121eda92b9482') \
        and (not ___needs_nopython()) \
        and (___check_tensors(L['input_ids'], tensor_check_names=tensor_check_names))

def __transformed_code_1_for_forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, head_mask, decoder_head_mask, cross_attn_head_mask, encoder_outputs, past_key_values, inputs_embeds, decoder_inputs_embeds, use_cache, output_attentions, output_hidden_states, return_dict):
    decoder_outputs = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    output_hidden_states = self.config.output_hidden_states
    output_attentions = self.config.output_attentions
    return __resume_at_110_6(self.encoder(input_ids=input_ids, attention_mask=
        attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds,
        output_attentions=self.config.output_attentions, output_hidden_states=
        self.config.output_hidden_states, return_dict=return_dict), self,
        attention_mask, decoder_input_ids, decoder_attention_mask,
        decoder_head_mask, cross_attn_head_mask, past_key_values,
        decoder_inputs_embeds, use_cache, output_attentions,
        output_hidden_states, return_dict)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, head_mask, decoder_head_mask, cross_attn_head_mask, encoder_outputs, past_key_values, inputs_embeds, decoder_inputs_embeds, use_cache, output_attentions, output_hidden_states, return_dict):
    'Failed to decompile.'

def transformed_forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, head_mask, decoder_head_mask, cross_attn_head_mask, encoder_outputs, past_key_values, inputs_embeds, decoder_inputs_embeds, use_cache, output_attentions, output_hidden_states, return_dict):
    L = {"self": self, "input_ids": input_ids, "attention_mask": attention_mask, "decoder_input_ids": decoder_input_ids, "decoder_attention_mask": decoder_attention_mask, "head_mask": head_mask, "decoder_head_mask": decoder_head_mask, "cross_attn_head_mask": cross_attn_head_mask, "encoder_outputs": encoder_outputs, "past_key_values": past_key_values, "inputs_embeds": inputs_embeds, "decoder_inputs_embeds": decoder_inputs_embeds, "use_cache": use_cache, "output_attentions": output_attentions, "output_hidden_states": output_hidden_states, "return_dict": return_dict}
    if __guard_1_for_forward(L):
        return __transformed_code_1_for_forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, head_mask, decoder_head_mask, cross_attn_head_mask, encoder_outputs, past_key_values, inputs_embeds, decoder_inputs_embeds, use_cache, output_attentions, output_hidden_states, return_dict)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, head_mask, decoder_head_mask, cross_attn_head_mask, encoder_outputs, past_key_values, inputs_embeds, decoder_inputs_embeds, use_cache, output_attentions, output_hidden_states, return_dict)

#============ end of forward ============#
