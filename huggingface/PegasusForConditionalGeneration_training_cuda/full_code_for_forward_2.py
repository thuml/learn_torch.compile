
def __guard_15_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 140346451011728)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['__class__'], 146834384)) \
        and (___check_type_id(L['input_ids_shape'], 140348647300832)) \
        and (len(L['input_ids_shape']) == 2) \
        and (___check_type_id(L['input_ids_shape'][0], 7640416)) \
        and (L['input_ids_shape'][0] == 1) \
        and (___check_type_id(L['input_ids_shape'][1], 7640416)) \
        and (L['input_ids_shape'][1] == 128) \
        and (___check_type_id(L['past_key_values_length'], 7640416)) \
        and (L['past_key_values_length'] == 0) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140343385792016))) \
        and (___compile_config_hash() == '9f64cb439de8127be15121eda92b9482') \
        and (___check_type_id(G['torch'].long, 140348647294720)) \
        and (G['torch'].long == torch.int64)

# Note: please refer to the graph code in __compiled_fn_25*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_25(*args, **kwargs):
    pass

def __helper_outer_function():
    # this is a helper function to help compilers generate bytecode to read capture variables from closures, rather than reading values from global scope. The value of these variables does not matter, and will be determined in runtime.
    __class__ = None
    def __transformed_code_15_for_forward(self, input_ids_shape, past_key_values_length):
        bsz = None; positions = None; seq_len = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
        nonlocal __class__
        return __compiled_fn_25()[0]


def __guard_2_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 140346451002992)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['__class__'], 146834384)) \
        and (___check_type_id(L['input_ids_shape'], 140348647300832)) \
        and (len(L['input_ids_shape']) == 2) \
        and (___check_type_id(L['input_ids_shape'][0], 7640416)) \
        and (L['input_ids_shape'][0] == 1) \
        and (___check_type_id(L['input_ids_shape'][1], 7640416)) \
        and (L['input_ids_shape'][1] == 128) \
        and (___check_type_id(L['past_key_values_length'], 7640416)) \
        and (L['past_key_values_length'] == 0) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140343385792016))) \
        and (___compile_config_hash() == '9f64cb439de8127be15121eda92b9482') \
        and (___check_type_id(G['torch'].long, 140348647294720)) \
        and (G['torch'].long == torch.int64)

# Note: please refer to the graph code in __compiled_fn_7*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_7(*args, **kwargs):
    pass

def __helper_outer_function():
    # this is a helper function to help compilers generate bytecode to read capture variables from closures, rather than reading values from global scope. The value of these variables does not matter, and will be determined in runtime.
    __class__ = None
    def __transformed_code_2_for_forward(self, input_ids_shape, past_key_values_length):
        bsz = None; positions = None; seq_len = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
        nonlocal __class__
        return __compiled_fn_7()[0]


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def forward(self, input_ids_shape, past_key_values_length):
    nonlocal __class__
    bsz = input_ids_shape[slice(None, 2)][0]
    seq_len = input_ids_shape[slice(None, 2)][1]
    positions = torch.arange(past_key_values_length, past_key_values_length +
        seq_len, dtype=torch.long, device=self.weight.device)
    return super().forward(positions)

def transformed_forward(self, input_ids_shape, past_key_values_length):
    L = {"self": self, "input_ids_shape": input_ids_shape, "past_key_values_length": past_key_values_length}
    if __guard_15_for_forward(L):
        return __transformed_code_15_for_forward(self, input_ids_shape, past_key_values_length)
    if __guard_2_for_forward(L):
        return __transformed_code_2_for_forward(self, input_ids_shape, past_key_values_length)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return forward(self, input_ids_shape, past_key_values_length)

#============ end of forward ============#
