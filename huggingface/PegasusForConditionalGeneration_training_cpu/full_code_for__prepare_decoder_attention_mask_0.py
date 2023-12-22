
def __guard_0_for__prepare_decoder_attention_mask(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_type_id(L['input_shape'], 139718085177056)) \
        and (len(L['input_shape']) == 2) \
        and (___check_type_id(L['inputs_embeds'], 90531504)) \
        and (hasattr(L['inputs_embeds'], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['attention_mask'], 7628576)) \
        and (___check_type_id(L['input_shape'][0], 7640416)) \
        and (L['input_shape'][0] == 1) \
        and (___check_type_id(L['input_shape'][1], 7640416)) \
        and (L['input_shape'][1] == 128) \
        and (___check_type_id(L['past_key_values_length'], 7640416)) \
        and (L['past_key_values_length'] == 0) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139712823664144))) \
        and (___compile_config_hash() == 'ef53a1e5e652702a41f4e7cf661cd065') \
        and (___check_type_id(G['_make_causal_mask'].__defaults__[0], 7640416)) \
        and (G['_make_causal_mask'].__defaults__[0] == 0) \
        and (___check_tensors(L['inputs_embeds'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_24*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_24(*args, **kwargs):
    pass

def __transformed_code_0_for__prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
    combined_attention_mask = None; expanded_attn_mask = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    return __compiled_fn_24()[0]


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(input_shape, inputs_embeds.
            dtype, device=inputs_embeds.device, past_key_values_length=
            past_key_values_length)
    if attention_mask is not None:
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype,
            tgt_len=input_shape[-1]).to(inputs_embeds.device)
        if combined_attention_mask is None:
            combined_attention_mask = expanded_attn_mask
        else:
            combined_attention_mask = expanded_attn_mask + combined_attention_mask
    return combined_attention_mask

def transformed__prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
    L = {"self": self, "attention_mask": attention_mask, "input_shape": input_shape, "inputs_embeds": inputs_embeds, "past_key_values_length": past_key_values_length}
    if __guard_0_for__prepare_decoder_attention_mask(L):
        return __transformed_code_0_for__prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length)

#============ end of _prepare_decoder_attention_mask ============#
