def __transformed_code_1_for_resume_in_forward_and_backward_pass(___stack0, self, mod, collect_outputs, cloned_inputs):
    inputs = None; loss = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function

    graph_out_0 = __compiled_fn_3(cloned_inputs[0])
    import importlib
    __temp_9 = importlib.import_module('transformers.modeling_outputs'
        ).CausalLMOutputWithCrossAttentions(loss=None, logits=graph_out_0[0],
        past_key_values=((graph_out_0[1], graph_out_0[2]), (graph_out_0[3],
        graph_out_0[4]), (graph_out_0[5], graph_out_0[6]), (graph_out_0[7],
        graph_out_0[8]), (graph_out_0[9], graph_out_0[10]), (graph_out_0[11],
        graph_out_0[12]), (graph_out_0[13], graph_out_0[14]), (graph_out_0[15],
        graph_out_0[16]), (graph_out_0[17], graph_out_0[18]), (graph_out_0[19],
        graph_out_0[20]), (graph_out_0[21], graph_out_0[22]), (graph_out_0[23],
        graph_out_0[24])), hidden_states=None, attentions=None,
        cross_attentions=None)
    pred = __temp_9
    ___context_manager_0_4 = __import_contextlib.nullcontext()
    ___context_manager_0_4.__enter__()
    try:
        __temp_12 = self.compute_loss(__temp_9)
    finally:
        ___context_manager_0_4.__exit__(None, None, None)
    return __resume_at_48_5(__import_contextlib.nullcontext, __temp_12, self,
        mod, collect_outputs, cloned_inputs, pred)
