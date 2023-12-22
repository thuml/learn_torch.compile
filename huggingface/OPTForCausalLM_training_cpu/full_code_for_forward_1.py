
def __guard_1_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 139747409368288)) \
        and (L['self'].training == True) \
        and (___check_obj_id(L['__class__'], 159555696)) \
        and (hasattr(L['attention_mask'], '_dynamo_dynamic_indices') == False) \
        and (___check_type_id(L['past_key_values_length'], 7640416)) \
        and (L['past_key_values_length'] == 0) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139744179592720))) \
        and (___compile_config_hash() == 'dbbdebae9d5a596724d86ea5aadf9793') \
        and (___check_tensors(L['attention_mask'], tensor_check_names=tensor_check_names))

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
    def __transformed_code_1_for_forward(self, attention_mask, past_key_values_length):
        positions = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
        nonlocal __class__
        return __compiled_fn_7(attention_mask)[0]


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def forward(self, attention_mask, past_key_values_length):
    nonlocal __class__
    attention_mask = attention_mask.long()
    positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) *
        attention_mask).long() - 1
    positions = positions[slice(None, None), slice(past_key_values_length, None)]
    return super().forward(positions + self.offset)

def transformed_forward(self, attention_mask, past_key_values_length):
    L = {"self": self, "attention_mask": attention_mask, "past_key_values_length": past_key_values_length}
    if __guard_1_for_forward(L):
        return __transformed_code_1_for_forward(self, attention_mask, past_key_values_length)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return forward(self, attention_mask, past_key_values_length)

#============ end of forward ============#
