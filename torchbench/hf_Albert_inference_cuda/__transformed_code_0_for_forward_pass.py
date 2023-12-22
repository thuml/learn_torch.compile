def __transformed_code_0_for_forward_pass(self, mod, inputs, collect_outputs):
    graph_out_0 = __compiled_fn_0(inputs[0])
    import importlib
    return importlib.import_module('transformers.modeling_outputs').MaskedLMOutput(
        loss=None, logits=graph_out_0[0], hidden_states=None, attentions=None)
