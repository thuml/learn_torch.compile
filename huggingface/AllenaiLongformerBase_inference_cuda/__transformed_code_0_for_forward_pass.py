def __transformed_code_0_for_forward_pass(self, mod, inputs, collect_outputs):
    ___context_manager_0_0 = __import_contextlib.nullcontext()
    ___context_manager_0_0.__enter__()
    try:
        __temp_4 = mod(*(), **{'input_ids': inputs['input_ids'], 'labels':
            inputs['labels']})
    finally:
        ___context_manager_0_0.__exit__(None, None, None)
    return __resume_at_22_1(__import_contextlib.nullcontext, __temp_4)
