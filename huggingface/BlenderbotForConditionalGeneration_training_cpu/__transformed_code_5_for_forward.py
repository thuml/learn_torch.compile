def __helper_outer_function():
    # this is a helper function to help compilers generate bytecode to read capture variables from closures, rather than reading values from global scope. The value of these variables does not matter, and will be determined in runtime.
    __class__ = None
    def __transformed_code_5_for_forward(self, input_ids_shape, past_key_values_length):
        bsz = None; positions = None; seq_len = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    
        nonlocal __class__
        return __compiled_fn_15()[0]
