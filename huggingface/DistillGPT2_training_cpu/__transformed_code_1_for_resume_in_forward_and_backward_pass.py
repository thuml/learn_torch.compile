def __transformed_code_1_for_resume_in_forward_and_backward_pass(___stack0, self, mod, collect_outputs, cloned_inputs):
    inputs = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function

    graph_out_0 = __compiled_fn_3(cloned_inputs['input_ids'], cloned_inputs[
        'labels'])
    import importlib
    loss = graph_out_0[0]
    pred = importlib.import_module('transformers.modeling_outputs'
        ).CausalLMOutputWithCrossAttentions(loss=graph_out_0[0], logits=
        graph_out_0[1], past_key_values=((graph_out_0[2], graph_out_0[3]), (
        graph_out_0[4], graph_out_0[5]), (graph_out_0[6], graph_out_0[7]), (
        graph_out_0[8], graph_out_0[9]), (graph_out_0[10], graph_out_0[11]), (
        graph_out_0[12], graph_out_0[13])), hidden_states=None, attentions=None,
        cross_attentions=None)
    return __resume_at_100_4(graph_out_0[0].backward(), self, mod,
        collect_outputs, cloned_inputs, pred, loss)
