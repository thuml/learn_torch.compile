
def __guard_1_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 140067222895968)) \
        and (L['self'].training == True) \
        and (___check_type_id(L['input_ids'], 92927168)) \
        and (hasattr(L['input_ids'], '_dynamo_dynamic_indices') == False) \
        and (___check_type_id(L['past_key_values_length'], 7640416)) \
        and (L['past_key_values_length'] == 0) \
        and (___check_type_id(L['self'].create_position_ids_from_input_ids.__defaults__[0], 7640416)) \
        and (L['self'].create_position_ids_from_input_ids.__defaults__[0] == 0) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140063927393808))) \
        and (___compile_config_hash() == 'd9da5cdf5912d92e85cf14993755c996') \
        and (___check_tensors(L['input_ids'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_7*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_7(*args, **kwargs):
    pass

def __transformed_code_1_for_forward(self, input_ids, past_key_values_length):
    bsz = None; max_pos = None; position_ids = None; seq_len = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    return __compiled_fn_7(input_ids)[0]


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def forward(self, input_ids, past_key_values_length):
    __temp_126 = input_ids.size()
    bsz = __temp_126[0]
    seq_len = __temp_126[1]
    position_ids = self.create_position_ids_from_input_ids(input_ids, self.
        padding_idx, past_key_values_length).to(input_ids.device)
    max_pos = self.padding_idx + 1 + seq_len
    if max_pos > self.weights.size(0):
        self.make_weights(max_pos + self.offset, self.embedding_dim, self.
            padding_idx)
        return self.weights.index_select(0, position_ids.view(-1)).view(bsz,
            seq_len, -1).detach()
    else:
        return self.weights.index_select(0, position_ids.view(-1)).view(bsz,
            seq_len, -1).detach()

def transformed_forward(self, input_ids, past_key_values_length):
    L = {"self": self, "input_ids": input_ids, "past_key_values_length": past_key_values_length}
    if __guard_1_for_forward(L):
        return __transformed_code_1_for_forward(self, input_ids, past_key_values_length)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return forward(self, input_ids, past_key_values_length)

#============ end of forward ============#
