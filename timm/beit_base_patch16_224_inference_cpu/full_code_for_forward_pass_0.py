
def __guard_0_for_forward_pass(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['mod'], 139670732963456)) \
        and (L['mod'].training == False) \
        and (___check_type_id(L['self'], 135021392)) \
        and (___check_type_id(L['inputs'], 7642176)) \
        and (len(L['inputs']) == 1) \
        and (___check_type_id(L['inputs'][0], 80079216)) \
        and (hasattr(L['inputs'][0], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['self'].autocast, 18406528)) \
        and (___check_obj_id(L['mod'].forward_head.__defaults__[0], 7677632)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139667595606720))) \
        and (___compile_config_hash() == '1d8dee009d905e1cbfaf1374ca4cb50f') \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_obj_id(L['mod'].blocks[0].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].blocks[1].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].blocks[2].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].blocks[3].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].blocks[4].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].blocks[5].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].blocks[6].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].blocks[7].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].blocks[8].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].blocks[9].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].blocks[10].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].blocks[11].forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].blocks[0].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].blocks[1].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].blocks[2].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].blocks[3].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].blocks[4].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].blocks[5].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].blocks[6].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].blocks[7].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].blocks[8].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].blocks[9].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].blocks[10].attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].blocks[11].attn.forward.__defaults__[0], 7628576)) \
        and (___check_tensors(L['inputs'][0], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_0*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_0(*args, **kwargs):
    pass

def __transformed_code_0_for_forward_pass(self, mod, inputs, collect_outputs):
    return __compiled_fn_0(inputs[0])[0]


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def forward_pass(self, mod, inputs, collect_outputs):
    with self.autocast() as __temp_6:
        return mod(*inputs)
    return None

def transformed_forward_pass(self, mod, inputs, collect_outputs):
    L = {"self": self, "mod": mod, "inputs": inputs, "collect_outputs": collect_outputs}
    if __guard_0_for_forward_pass(L):
        return __transformed_code_0_for_forward_pass(self, mod, inputs, collect_outputs)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return forward_pass(self, mod, inputs, collect_outputs)

#============ end of forward_pass ============#
