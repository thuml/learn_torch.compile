def __transformed_code_0_for_forward_pass(self, mod, inputs, collect_outputs):
    graph_out_0 = __compiled_fn_0(inputs['input_ids'], inputs['labels'])
    import importlib
    return importlib.import_module('transformers.modeling_outputs'
        ).CausalLMOutputWithCrossAttentions(loss=graph_out_0[0], logits=
        graph_out_0[1], past_key_values=((graph_out_0[2], graph_out_0[3]), (
        graph_out_0[4], graph_out_0[5]), (graph_out_0[6], graph_out_0[7]), (
        graph_out_0[8], graph_out_0[9]), (graph_out_0[10], graph_out_0[11]), (
        graph_out_0[12], graph_out_0[13])), hidden_states=None, attentions=None,
        cross_attentions=None)
