
def guard_0(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (hasattr(L['attention_mask'], '_dynamo_dynamic_indices') == False) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139639130168848))) \
        and (___compile_config_hash() == 'd23842bdb1f875b062b4abc655654038') \
        and (not ___needs_nopython()) \
        and (___check_tensors(L['attention_mask'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_5*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_5(*args, **kwargs):
    pass

def transformed_code_0(self, hidden_states, attention_mask, head_mask, padding_len, output_attentions, output_hidden_states, return_dict):
    graph_out_0 = __compiled_fn_5(attention_mask)
    is_index_global_attn = graph_out_0[2]
    is_index_masked = graph_out_0[1]
    def __resume_at_30_6(___stack0, self, hidden_states, attention_mask,
        head_mask, output_hidden_states, return_dict, is_index_masked,
        is_index_global_attn):
        nonlocal is_global_attn, output_attentions, padding_len
        is_global_attn = ___stack0
        if output_hidden_states:
            all_hidden_states = ()
        else:
            all_hidden_states = None
        if output_attentions:
            all_attentions = ()
        else:
            all_attentions = None
        if output_attentions:
            if is_global_attn:
                all_global_attentions = ()
            else:
                all_global_attentions = None
        else:
            all_global_attentions = None
        if head_mask is not None:
            if not head_mask.size()[0] == len(self.layer):
                raise AssertionError('The head_mask should be specified for ' +
                    str(len(self.layer)) + ' layers, but it is for ' + str(
                    head_mask.size()[0]) + '.')
        for __temp_125 in iter(enumerate(self.layer)):
            idx = __temp_125[0]
            layer_module = __temp_125[1]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if self.gradient_checkpointing:
                if self.training:
                    def create_custom_forward(module):
                        nonlocal is_global_attn, output_attentions
                        def custom_forward(*inputs):
                            nonlocal is_global_attn, module, output_attentions
                            __temp_127 = []
                            __temp_127.extend(inputs)
                            __temp_127.append(is_global_attn)
                            __temp_127.append(output_attentions)
                            return module(*tuple(__temp_127))
                        return custom_forward
                    __temp_129 = create_custom_forward(layer_module)
                    if head_mask is not None:
                        layer_outputs = torch.utils.checkpoint.checkpoint(
                            __temp_129, hidden_states, attention_mask,
                            head_mask[idx], is_index_masked, is_index_global_attn)
                    else:
                        layer_outputs = torch.utils.checkpoint.checkpoint(
                            __temp_129, hidden_states, attention_mask, None,
                            is_index_masked, is_index_global_attn)
                elif head_mask is not None:
                    layer_outputs = layer_module(hidden_states, attention_mask=
                        attention_mask, layer_head_mask=head_mask[idx],
                        is_index_masked=is_index_masked, is_index_global_attn=
                        is_index_global_attn, is_global_attn=is_global_attn,
                        output_attentions=output_attentions)
                else:
                    layer_outputs = layer_module(hidden_states, attention_mask=
                        attention_mask, layer_head_mask=None, is_index_masked=
                        is_index_masked, is_index_global_attn=
                        is_index_global_attn, is_global_attn=is_global_attn,
                        output_attentions=output_attentions)
            elif head_mask is not None:
                layer_outputs = layer_module(hidden_states, attention_mask=
                    attention_mask, layer_head_mask=head_mask[idx],
                    is_index_masked=is_index_masked, is_index_global_attn=
                    is_index_global_attn, is_global_attn=is_global_attn,
                    output_attentions=output_attentions)
            else:
                layer_outputs = layer_module(hidden_states, attention_mask=
                    attention_mask, layer_head_mask=None, is_index_masked=
                    is_index_masked, is_index_global_attn=is_index_global_attn,
                    is_global_attn=is_global_attn, output_attentions=
                    output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1].transpose(1,
                    2),)
                if is_global_attn:
                    all_global_attentions = all_global_attentions + (layer_outputs
                        [2].transpose(2, 3),)
            continue
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        hidden_states = hidden_states[slice(None, None), slice(None, 
            hidden_states.shape[1] - padding_len)]
        if output_hidden_states:
            """original function name <listcomp> is illegal, use a temp name."""
            def __temp_138(comp_arg_0):
                nonlocal padding_len
                __temp_139 = []
                for __temp_140 in comp_arg_0:
                    state = __temp_140
                    __temp_139.append(state[slice(None, None), slice(None, 
                        state.shape[1] - padding_len)])
                    continue
                return __temp_139
            all_hidden_states = tuple(__temp_138(iter(all_hidden_states)))
        if output_attentions:
            """original function name <listcomp> is illegal, use a temp name."""
            def __temp_143(comp_arg_0):
                nonlocal padding_len
                __temp_144 = []
                for __temp_145 in comp_arg_0:
                    state = __temp_145
                    __temp_144.append(state[slice(None, None), slice(None, None
                        ), slice(None, state.shape[2] - padding_len), slice(
                        None, None)])
                    continue
                return __temp_144
            all_attentions = tuple(__temp_143(iter(all_attentions)))
        if not return_dict:
            """original function name <genexpr> is illegal, use a temp name."""
            def __temp_148(comp_arg_0):
                for __temp_149 in comp_arg_0:
                    v = __temp_149
                    if not v is not None:
                        continue
                    yield v
                    continue
                return None
            return tuple(__temp_148(iter((hidden_states, all_hidden_states,
                all_attentions, all_global_attentions))))
        return LongformerBaseModelOutput(last_hidden_state=hidden_states,
            hidden_states=all_hidden_states, attentions=all_attentions,
            global_attentions=all_global_attentions)
    return __resume_at_30_6(graph_out_0[0].item(), self, hidden_states,
        attention_mask, head_mask, output_hidden_states, return_dict,
        is_index_masked, is_index_global_attn)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def forward(self, hidden_states, attention_mask, head_mask, padding_len, output_attentions, output_hidden_states, return_dict):
    is_index_masked = attention_mask < 0
    is_index_global_attn = attention_mask > 0
    is_global_attn = is_index_global_attn.flatten().any().item()
    if output_hidden_states:
        all_hidden_states = ()
    else:
        all_hidden_states = None
    if output_attentions:
        all_attentions = ()
    else:
        all_attentions = None
    if output_attentions:
        if is_global_attn:
            all_global_attentions = ()
        else:
            all_global_attentions = None
    else:
        all_global_attentions = None
    if head_mask is not None:
        if not head_mask.size()[0] == len(self.layer):
            raise AssertionError('The head_mask should be specified for ' + str
                (len(self.layer)) + ' layers, but it is for ' + str(head_mask.
                size()[0]) + '.')
    for __temp_163 in iter(enumerate(self.layer)):
        idx = __temp_163[0]
        layer_module = __temp_163[1]
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if self.gradient_checkpointing:
            if self.training:
                def create_custom_forward(module):
                    nonlocal is_global_attn, output_attentions
                    def custom_forward(*inputs):
                        nonlocal is_global_attn, module, output_attentions
                        __temp_165 = []
                        __temp_165.extend(inputs)
                        __temp_165.append(is_global_attn)
                        __temp_165.append(output_attentions)
                        return module(*tuple(__temp_165))
                    return custom_forward
                __temp_167 = create_custom_forward(layer_module)
                if head_mask is not None:
                    layer_outputs = torch.utils.checkpoint.checkpoint(__temp_167,
                        hidden_states, attention_mask, head_mask[idx],
                        is_index_masked, is_index_global_attn)
                else:
                    layer_outputs = torch.utils.checkpoint.checkpoint(__temp_167,
                        hidden_states, attention_mask, None, is_index_masked,
                        is_index_global_attn)
            elif head_mask is not None:
                layer_outputs = layer_module(hidden_states, attention_mask=
                    attention_mask, layer_head_mask=head_mask[idx],
                    is_index_masked=is_index_masked, is_index_global_attn=
                    is_index_global_attn, is_global_attn=is_global_attn,
                    output_attentions=output_attentions)
            else:
                layer_outputs = layer_module(hidden_states, attention_mask=
                    attention_mask, layer_head_mask=None, is_index_masked=
                    is_index_masked, is_index_global_attn=is_index_global_attn,
                    is_global_attn=is_global_attn, output_attentions=
                    output_attentions)
        elif head_mask is not None:
            layer_outputs = layer_module(hidden_states, attention_mask=
                attention_mask, layer_head_mask=head_mask[idx], is_index_masked
                =is_index_masked, is_index_global_attn=is_index_global_attn,
                is_global_attn=is_global_attn, output_attentions=output_attentions)
        else:
            layer_outputs = layer_module(hidden_states, attention_mask=
                attention_mask, layer_head_mask=None, is_index_masked=
                is_index_masked, is_index_global_attn=is_index_global_attn,
                is_global_attn=is_global_attn, output_attentions=output_attentions)
        hidden_states = layer_outputs[0]
        if output_attentions:
            all_attentions = all_attentions + (layer_outputs[1].transpose(1, 2),)
            if is_global_attn:
                all_global_attentions = all_global_attentions + (layer_outputs[
                    2].transpose(2, 3),)
        continue
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)
    hidden_states = hidden_states[slice(None, None), slice(None, hidden_states.
        shape[1] - padding_len)]
    if output_hidden_states:
        """original function name <listcomp> is illegal, use a temp name."""
        def __temp_176(comp_arg_0):
            nonlocal padding_len
            __temp_177 = []
            for __temp_178 in comp_arg_0:
                state = __temp_178
                __temp_177.append(state[slice(None, None), slice(None, state.
                    shape[1] - padding_len)])
                continue
            return __temp_177
        all_hidden_states = tuple(__temp_176(iter(all_hidden_states)))
    if output_attentions:
        """original function name <listcomp> is illegal, use a temp name."""
        def __temp_181(comp_arg_0):
            nonlocal padding_len
            __temp_182 = []
            for __temp_183 in comp_arg_0:
                state = __temp_183
                __temp_182.append(state[slice(None, None), slice(None, None),
                    slice(None, state.shape[2] - padding_len), slice(None, None)])
                continue
            return __temp_182
        all_attentions = tuple(__temp_181(iter(all_attentions)))
    if not return_dict:
        """original function name <genexpr> is illegal, use a temp name."""
        def __temp_186(comp_arg_0):
            for __temp_187 in comp_arg_0:
                v = __temp_187
                if not v is not None:
                    continue
                yield v
                continue
            return None
        return tuple(__temp_186(iter((hidden_states, all_hidden_states,
            all_attentions, all_global_attentions))))
    return LongformerBaseModelOutput(last_hidden_state=hidden_states,
        hidden_states=all_hidden_states, attentions=all_attentions,
        global_attentions=all_global_attentions)

def transformed_forward(self, hidden_states, attention_mask, head_mask, padding_len, output_attentions, output_hidden_states, return_dict):
    L = {"self": self, "hidden_states": hidden_states, "attention_mask": attention_mask, "head_mask": head_mask, "padding_len": padding_len, "output_attentions": output_attentions, "output_hidden_states": output_hidden_states, "return_dict": return_dict}
    if guard_0(L):
        return transformed_code_0(self, hidden_states, attention_mask, head_mask, padding_len, output_attentions, output_hidden_states, return_dict)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return forward(self, hidden_states, attention_mask, head_mask, padding_len, output_attentions, output_hidden_states, return_dict)

#============ end of forward ============#
