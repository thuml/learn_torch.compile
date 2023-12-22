
def __guard_0_for_forward_pass(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['mod'], 140387328442528)) \
        and (L['mod'].training == False) \
        and (___check_type_id(L['self'], 20651536)) \
        and (___check_type_id(L['inputs'], 7642176)) \
        and (len(L['inputs']) == 1) \
        and (___check_type_id(L['inputs'][0], 81395792)) \
        and (hasattr(L['inputs'][0], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['self'].autocast, 19725024)) \
        and (___check_obj_id(L['mod'].forward_head.__defaults__[0], 7677632)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140384066730688))) \
        and (___compile_config_hash() == 'edcffe9cba5bba4eec926f9c79308605') \
        and (___check_obj_id(G['__import_timm_dot_layers_dot_format'].Format.NHWC, 140384157025104)) \
        and (___check_obj_id(G['__import_timm_dot_layers_dot_patch_embed'].Format.NCHW, 140384157024992)) \
        and (___check_type_id(G['__import_timm_dot_layers_dot_drop'].drop_path.__defaults__[0], 7644160)) \
        and (G['__import_timm_dot_layers_dot_drop'].drop_path.__defaults__[0] == 0.0) \
        and (___check_obj_id(G['__import_timm_dot_layers_dot_drop'].drop_path.__defaults__[1], 7677632)) \
        and (___check_obj_id(G['__import_timm_dot_layers_dot_drop'].drop_path.__defaults__[2], 7677664)) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_obj_id(L['mod'].head.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].layers, '0').blocks, '0').attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(getattr(getattr(L['mod'].layers, '0').blocks, '1').attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(getattr(getattr(L['mod'].layers, '1').blocks, '0').attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(getattr(getattr(L['mod'].layers, '1').blocks, '1').attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(getattr(getattr(L['mod'].layers, '2').blocks, '0').attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(getattr(getattr(L['mod'].layers, '2').blocks, '1').attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(getattr(getattr(L['mod'].layers, '2').blocks, '2').attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(getattr(getattr(L['mod'].layers, '2').blocks, '3').attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(getattr(getattr(L['mod'].layers, '2').blocks, '4').attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(getattr(getattr(L['mod'].layers, '2').blocks, '5').attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(getattr(getattr(L['mod'].layers, '2').blocks, '6').attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(getattr(getattr(L['mod'].layers, '2').blocks, '7').attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(getattr(getattr(L['mod'].layers, '2').blocks, '8').attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(getattr(getattr(L['mod'].layers, '2').blocks, '9').attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(getattr(getattr(L['mod'].layers, '3').blocks, '0').attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(getattr(getattr(L['mod'].layers, '3').blocks, '1').attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(getattr(getattr(L['mod'].layers, '2').blocks, '10').attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(getattr(getattr(L['mod'].layers, '2').blocks, '11').attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(getattr(getattr(L['mod'].layers, '2').blocks, '12').attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(getattr(getattr(L['mod'].layers, '2').blocks, '13').attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(getattr(getattr(L['mod'].layers, '2').blocks, '14').attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(getattr(getattr(L['mod'].layers, '2').blocks, '15').attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(getattr(getattr(L['mod'].layers, '2').blocks, '16').attn.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(getattr(getattr(L['mod'].layers, '2').blocks, '17').attn.forward.__defaults__[0], 7628576)) \
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
