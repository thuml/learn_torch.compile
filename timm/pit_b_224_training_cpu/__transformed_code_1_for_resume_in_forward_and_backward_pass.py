def __transformed_code_1_for_resume_in_forward_and_backward_pass(___stack0, self, mod, collect_outputs, cloned_inputs):
    inputs = None; loss = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function

    graph_out_0 = __compiled_fn_3(cloned_inputs[0])
    pred = graph_out_0[0]
    ___context_manager_0_4 = __import_contextlib.nullcontext()
    ___context_manager_0_4.__enter__()
    try:
        __temp_10 = self.compute_loss(graph_out_0[0])
    finally:
        ___context_manager_0_4.__exit__(None, None, None)
    return __resume_at_66_5(__import_contextlib.nullcontext, __temp_10, self,
        mod, collect_outputs, cloned_inputs, pred)
