def __transformed_code_1_for_resume_in_forward_and_backward_pass(___stack0, self, mod, collect_outputs, cloned_inputs):
    inputs = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function

    graph_out_0 = __compiled_fn_3(cloned_inputs['input_ids'], cloned_inputs[
        'labels'])
    import importlib
    loss = graph_out_0[0]
    pred = importlib.import_module('transformers.modeling_outputs').MaskedLMOutput(
        loss=graph_out_0[0], logits=graph_out_0[1], hidden_states=None,
        attentions=None)
    return __resume_at_100_4(graph_out_0[0].backward(), self, mod,
        collect_outputs, cloned_inputs, pred, loss)
