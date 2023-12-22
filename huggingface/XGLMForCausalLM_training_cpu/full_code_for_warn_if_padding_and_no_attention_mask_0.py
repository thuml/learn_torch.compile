
def __guard_0_for_warn_if_padding_and_no_attention_mask(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (hasattr(L['input_ids'], '_dynamo_dynamic_indices') == False) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140087129579024))) \
        and (___compile_config_hash() == '2e7255621b8c647cf36d14f32441f8ea') \
        and (___check_obj_id(G['__import_transformers_dot_utils_dot_import_utils']._torch_available, 7677664)) \
        and (___check_obj_id(G['__import_transformers_dot_utils_dot_import_utils']._torch_fx_available, 7677664)) \
        and (___check_tensors(L['input_ids'], tensor_check_names=tensor_check_names))

def __transformed_code_0_for_warn_if_padding_and_no_attention_mask(self, input_ids, attention_mask):
    warn_string = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    return None


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def warn_if_padding_and_no_attention_mask(self, input_ids, attention_mask):
    if not is_torch_fx_proxy(input_ids):
        if not torch.jit.is_tracing():
            if is_torchdynamo_compiling():
                return None
        else:
            return None
    else:
        return None
    if not attention_mask is not None:
        if self.config.pad_token_id is None:
            return None
    else:
        return None
    if self.config.pad_token_id in input_ids[slice(None, None), [-1, 0]]:
        warn_string = (
            'We strongly recommend passing in an `attention_mask` since your input_ids may be padded. See https://huggingface.co/docs/transformers/troubleshooting#incorrect-output-when-padding-tokens-arent-masked.'
            )
        if self.config.bos_token_id is not None:
            if not self.config.bos_token_id == self.config.pad_token_id:
                if self.config.eos_token_id is not None:
                    if not self.config.eos_token_id == self.config.pad_token_id:
                        if self.config.sep_token_id is not None:
                            if (self.config.sep_token_id == self.config.
                                pad_token_id):
                                warn_string += (
                                    """
    You may ignore this warning if your `pad_token_id` ("""
                                     + str(self.config.pad_token_id) +
                                    ') is identical to the `bos_token_id` (' +
                                    str(self.config.bos_token_id) +
                                    '), `eos_token_id` (' + str(self.config.
                                    eos_token_id) +
                                    '), or the `sep_token_id` (' + str(self.
                                    config.sep_token_id) +
                                    '), and your input is not padded.')
                    else:
                        warn_string += (
                            '\nYou may ignore this warning if your `pad_token_id` ('
                             + str(self.config.pad_token_id) +
                            ') is identical to the `bos_token_id` (' + str(self
                            .config.bos_token_id) + '), `eos_token_id` (' + str
                            (self.config.eos_token_id) +
                            '), or the `sep_token_id` (' + str(self.config.
                            sep_token_id) + '), and your input is not padded.')
                elif self.config.sep_token_id is not None:
                    if self.config.sep_token_id == self.config.pad_token_id:
                        warn_string += (
                            '\nYou may ignore this warning if your `pad_token_id` ('
                             + str(self.config.pad_token_id) +
                            ') is identical to the `bos_token_id` (' + str(self
                            .config.bos_token_id) + '), `eos_token_id` (' + str
                            (self.config.eos_token_id) +
                            '), or the `sep_token_id` (' + str(self.config.
                            sep_token_id) + '), and your input is not padded.')
            else:
                warn_string += (
                    '\nYou may ignore this warning if your `pad_token_id` (' +
                    str(self.config.pad_token_id) +
                    ') is identical to the `bos_token_id` (' + str(self.config.
                    bos_token_id) + '), `eos_token_id` (' + str(self.config.
                    eos_token_id) + '), or the `sep_token_id` (' + str(self.
                    config.sep_token_id) + '), and your input is not padded.')
        elif self.config.eos_token_id is not None:
            if not self.config.eos_token_id == self.config.pad_token_id:
                if self.config.sep_token_id is not None:
                    if self.config.sep_token_id == self.config.pad_token_id:
                        warn_string += (
                            '\nYou may ignore this warning if your `pad_token_id` ('
                             + str(self.config.pad_token_id) +
                            ') is identical to the `bos_token_id` (' + str(self
                            .config.bos_token_id) + '), `eos_token_id` (' + str
                            (self.config.eos_token_id) +
                            '), or the `sep_token_id` (' + str(self.config.
                            sep_token_id) + '), and your input is not padded.')
            else:
                warn_string += (
                    '\nYou may ignore this warning if your `pad_token_id` (' +
                    str(self.config.pad_token_id) +
                    ') is identical to the `bos_token_id` (' + str(self.config.
                    bos_token_id) + '), `eos_token_id` (' + str(self.config.
                    eos_token_id) + '), or the `sep_token_id` (' + str(self.
                    config.sep_token_id) + '), and your input is not padded.')
        elif self.config.sep_token_id is not None:
            if self.config.sep_token_id == self.config.pad_token_id:
                warn_string += (
                    '\nYou may ignore this warning if your `pad_token_id` (' +
                    str(self.config.pad_token_id) +
                    ') is identical to the `bos_token_id` (' + str(self.config.
                    bos_token_id) + '), `eos_token_id` (' + str(self.config.
                    eos_token_id) + '), or the `sep_token_id` (' + str(self.
                    config.sep_token_id) + '), and your input is not padded.')
        logger.warning_once(warn_string)
        return None
    return None

def transformed_warn_if_padding_and_no_attention_mask(self, input_ids, attention_mask):
    L = {"self": self, "input_ids": input_ids, "attention_mask": attention_mask}
    if __guard_0_for_warn_if_padding_and_no_attention_mask(L):
        return __transformed_code_0_for_warn_if_padding_and_no_attention_mask(self, input_ids, attention_mask)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return warn_if_padding_and_no_attention_mask(self, input_ids, attention_mask)

#============ end of warn_if_padding_and_no_attention_mask ============#
