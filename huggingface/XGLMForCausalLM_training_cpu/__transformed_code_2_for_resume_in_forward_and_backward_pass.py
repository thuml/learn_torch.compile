def __transformed_code_2_for_resume_in_forward_and_backward_pass(___stack0, ___stack1, self, mod, collect_outputs, cloned_inputs):
    inputs = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function

    loss = ___stack1.loss
    pred = ___stack1
    return __resume_at_144_39(___stack1.loss.backward(), self, mod,
        collect_outputs, cloned_inputs, pred, loss)
