
def __guard_1_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 140482411815040)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['__class__'], 166269776)) \
        and (___check_type_id(L['input_ids'], 96413136)) \
        and (hasattr(L['input_ids'], '_dynamo_dynamic_indices') == False) \
        and (___check_type_id(L['past_key_values_length'], 7640416)) \
        and (L['past_key_values_length'] == 0) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140479138422288))) \
        and (___compile_config_hash() == 'f8fafcff2517b8f4e8d6c21b5d7481ce') \
        and (___check_type_id(G['torch'].long, 140484400146176)) \
        and (G['torch'].long == torch.int64) \
        and (___check_tensors(L['input_ids'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_6*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_6(*args, **kwargs):
    pass

def __helper_outer_function():
    # this is a helper function to help compilers generate bytecode to read capture variables from closures, rather than reading values from global scope. The value of these variables does not matter, and will be determined in runtime.
    __class__ = None
    def __transformed_code_1_for_forward(self, input_ids, past_key_values_length):
        bsz = None; positions = None; seq_len = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
        nonlocal __class__
        return __compiled_fn_6()[0]


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def forward(self, input_ids, past_key_values_length):
    nonlocal __class__
    bsz = input_ids.shape[slice(None, 2)][0]
    seq_len = input_ids.shape[slice(None, 2)][1]
    positions = torch.arange(past_key_values_length, past_key_values_length +
        seq_len, dtype=torch.long, device=self.weight.device).expand(bsz, -1)
    return super().forward(positions + self.offset)

def transformed_forward(self, input_ids, past_key_values_length):
    L = {"self": self, "input_ids": input_ids, "past_key_values_length": past_key_values_length}
    if __guard_1_for_forward(L):
        return __transformed_code_1_for_forward(self, input_ids, past_key_values_length)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return forward(self, input_ids, past_key_values_length)

#============ end of forward ============#
