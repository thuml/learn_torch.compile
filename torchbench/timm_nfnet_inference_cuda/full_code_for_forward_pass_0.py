
def __guard_0_for_forward_pass(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['mod'], 139804379043184)) \
        and (L['mod'].training == False) \
        and (___check_type_id(L['self'], 118404624)) \
        and (___check_type_id(L['inputs'], 7642176)) \
        and (len(L['inputs']) == 1) \
        and (hasattr(L['inputs'][0], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['self'].autocast, 35646144)) \
        and (___check_obj_id(L['mod'].forward_head.__defaults__[0], 7677632)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139801454944480))) \
        and (___compile_config_hash() == '915a5e4444ffa380e9bad59f14a7af38') \
        and (___check_type_id(G['__import_timm_dot_layers_dot_std_conv'].pad_same.__defaults__[0], 7617760)) \
        and (len(G['__import_timm_dot_layers_dot_std_conv'].pad_same.__defaults__[0]) == 2) \
        and (___check_type_id(G['__import_timm_dot_layers_dot_std_conv'].pad_same.__defaults__[1], 7640416)) \
        and (G['__import_timm_dot_layers_dot_std_conv'].pad_same.__defaults__[1] == 0) \
        and (___check_type_id(G['__import_timm_dot_layers_dot_std_conv'].pad_same.__defaults__[0][0], 7640416)) \
        and (G['__import_timm_dot_layers_dot_std_conv'].pad_same.__defaults__[0][0] == 1) \
        and (___check_type_id(G['__import_timm_dot_layers_dot_std_conv'].pad_same.__defaults__[0][1], 7640416)) \
        and (G['__import_timm_dot_layers_dot_std_conv'].pad_same.__defaults__[0][1] == 1) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_obj_id(L['mod'].head.forward.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].final_act.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].stem.act2.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].stem.act3.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(L['mod'].stem.act4.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '0'), '0').act1.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '0'), '0').act2.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '0'), '0').act3.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '1'), '0').act1.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '1'), '0').act2.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '1'), '0').act3.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '1'), '1').act1.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '1'), '1').act2.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '1'), '1').act3.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '2'), '0').act1.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '2'), '0').act2.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '2'), '0').act3.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '2'), '1').act1.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '2'), '1').act2.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '2'), '1').act3.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '2'), '2').act1.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '2'), '2').act2.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '2'), '2').act3.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '2'), '3').act1.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '2'), '3').act2.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '2'), '3').act3.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '2'), '4').act1.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '2'), '4').act2.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '2'), '4').act3.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '2'), '5').act1.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '2'), '5').act2.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '2'), '5').act3.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '3'), '0').act1.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '3'), '0').act2.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '3'), '0').act3.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '3'), '1').act1.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '3'), '1').act2.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '3'), '1').act3.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '3'), '2').act1.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '3'), '2').act2.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '3'), '2').act3.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '0'), '0').act2b.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '1'), '0').act2b.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '1'), '1').act2b.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '2'), '0').act2b.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '2'), '1').act2b.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '2'), '2').act2b.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '2'), '3').act2b.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '2'), '4').act2b.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '2'), '5').act2b.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '3'), '0').act2b.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '3'), '1').act2b.act_fn.__defaults__[0], 7677632)) \
        and (___check_obj_id(getattr(getattr(L['mod'].stages, '3'), '2').act2b.act_fn.__defaults__[0], 7677632)) \
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
