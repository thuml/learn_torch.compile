def __transformed_code_0_for_forward_pass(self, mod, inputs, collect_outputs):
    graph_out_0 = __compiled_fn_0(inputs[0])
    getattr(getattr(mod.stages, '2').blocks, '3').attn.attention_bias_cache.clear()
    getattr(getattr(mod.stages, '2').blocks, '3').attn.attention_bias_cache.update(
        {'cuda:0': graph_out_0[14]})
    getattr(getattr(mod.stages, '2').blocks, '2').attn.attention_bias_cache.clear()
    getattr(getattr(mod.stages, '2').blocks, '2').attn.attention_bias_cache.update(
        {'cuda:0': graph_out_0[13]})
    getattr(getattr(mod.stages, '2').blocks, '1').attn.attention_bias_cache.clear()
    getattr(getattr(mod.stages, '2').blocks, '1').attn.attention_bias_cache.update(
        {'cuda:0': graph_out_0[12]})
    getattr(getattr(mod.stages, '2').blocks, '0').attn.attention_bias_cache.clear()
    getattr(getattr(mod.stages, '2').blocks, '0').attn.attention_bias_cache.update(
        {'cuda:0': graph_out_0[11]})
    getattr(mod.stages, '2').downsample.attn_downsample.attention_bias_cache.clear(
        )
    getattr(mod.stages, '2'
        ).downsample.attn_downsample.attention_bias_cache.update({'cuda:0':
        graph_out_0[10]})
    getattr(getattr(mod.stages, '1').blocks, '3').attn.attention_bias_cache.clear()
    getattr(getattr(mod.stages, '1').blocks, '3').attn.attention_bias_cache.update(
        {'cuda:0': graph_out_0[9]})
    getattr(getattr(mod.stages, '1').blocks, '2').attn.attention_bias_cache.clear()
    getattr(getattr(mod.stages, '1').blocks, '2').attn.attention_bias_cache.update(
        {'cuda:0': graph_out_0[8]})
    getattr(getattr(mod.stages, '1').blocks, '1').attn.attention_bias_cache.clear()
    getattr(getattr(mod.stages, '1').blocks, '1').attn.attention_bias_cache.update(
        {'cuda:0': graph_out_0[7]})
    getattr(getattr(mod.stages, '1').blocks, '0').attn.attention_bias_cache.clear()
    getattr(getattr(mod.stages, '1').blocks, '0').attn.attention_bias_cache.update(
        {'cuda:0': graph_out_0[6]})
    getattr(mod.stages, '1').downsample.attn_downsample.attention_bias_cache.clear(
        )
    getattr(mod.stages, '1'
        ).downsample.attn_downsample.attention_bias_cache.update({'cuda:0':
        graph_out_0[5]})
    getattr(getattr(mod.stages, '0').blocks, '3').attn.attention_bias_cache.clear()
    getattr(getattr(mod.stages, '0').blocks, '3').attn.attention_bias_cache.update(
        {'cuda:0': graph_out_0[4]})
    getattr(getattr(mod.stages, '0').blocks, '2').attn.attention_bias_cache.clear()
    getattr(getattr(mod.stages, '0').blocks, '2').attn.attention_bias_cache.update(
        {'cuda:0': graph_out_0[3]})
    getattr(getattr(mod.stages, '0').blocks, '1').attn.attention_bias_cache.clear()
    getattr(getattr(mod.stages, '0').blocks, '1').attn.attention_bias_cache.update(
        {'cuda:0': graph_out_0[2]})
    getattr(getattr(mod.stages, '0').blocks, '0').attn.attention_bias_cache.clear()
    getattr(getattr(mod.stages, '0').blocks, '0').attn.attention_bias_cache.update(
        {'cuda:0': graph_out_0[1]})
    return graph_out_0[0]
