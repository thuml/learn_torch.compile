
def __guard_1_for_forward(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['self'], 139745391015152)) \
        and (L['self'].training == True) \
        and (hasattr(L['position_ids'], '_dynamo_dynamic_indices') == False) \
        and (___check_type_id(L['past_key_values_length'], 7640416)) \
        and (L['past_key_values_length'] == 0) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139742126890512))) \
        and (___compile_config_hash() == 'e66c45e2da697ecf4e02762b48b3dc0f') \
        and (___check_tensors(L['position_ids'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_7*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_7(*args, **kwargs):
    pass

def __transformed_code_1_for_forward(self, position_ids, past_key_values_length):
    bsz = None; max_pos = None; seq_len = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    return __compiled_fn_7(position_ids)[0]


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def forward(self, position_ids, past_key_values_length):
    __temp_153 = position_ids.size()
    bsz = __temp_153[0]
    seq_len = __temp_153[1]
    position_ids += self.offset
    max_pos = 2 + seq_len + past_key_values_length
    if max_pos > self.weights.size(0):
        self.make_weights(max_pos, self.embedding_dim, self.padding_idx)
        return self.weights.index_select(0, position_ids.view(-1)).view(bsz,
            seq_len, self.weights.shape[-1]).detach()
    else:
        return self.weights.index_select(0, position_ids.view(-1)).view(bsz,
            seq_len, self.weights.shape[-1]).detach()

def transformed_forward(self, position_ids, past_key_values_length):
    L = {"self": self, "position_ids": position_ids, "past_key_values_length": past_key_values_length}
    if __guard_1_for_forward(L):
        return __transformed_code_1_for_forward(self, position_ids, past_key_values_length)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return forward(self, position_ids, past_key_values_length)

#============ end of forward ============#
