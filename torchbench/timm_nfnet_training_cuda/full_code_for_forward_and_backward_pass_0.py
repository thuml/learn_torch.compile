
# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_102_7(___stack0, mod, collect_outputs, cloned_inputs, pred, loss):
    'Failed to decompile.'

def transformed___resume_at_102_7(___stack0, mod, collect_outputs, cloned_inputs, pred, loss):
    L = {"___stack0": ___stack0, "mod": mod, "collect_outputs": collect_outputs, "cloned_inputs": cloned_inputs, "pred": pred, "loss": loss}
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_102_7(___stack0, mod, collect_outputs, cloned_inputs, pred, loss)

#============ end of __resume_at_102_7 ============#

def __guard_3_for_resume_in_forward_and_backward_pass(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_type_id(L['self'], 94674336)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139859616489696))) \
        and (___compile_config_hash() == '0487ace406d876273bd209654917c9ce') \
        and (not ___needs_nopython())

def __transformed_code_3_for_resume_in_forward_and_backward_pass(___stack0, self, mod, collect_outputs, cloned_inputs, pred, loss):
    inputs = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    return __resume_at_102_7(self.optimizer_step(), mod, collect_outputs,
        cloned_inputs, pred, loss)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_138_6(___stack0, self, mod, collect_outputs, cloned_inputs, pred, loss):
    'Failed to decompile.'

def transformed___resume_at_138_6(___stack0, self, mod, collect_outputs, cloned_inputs, pred, loss):
    L = {"___stack0": ___stack0, "self": self, "mod": mod, "collect_outputs": collect_outputs, "cloned_inputs": cloned_inputs, "pred": pred, "loss": loss}
    if __guard_3_for_resume_in_forward_and_backward_pass(L):
        return __transformed_code_3_for_resume_in_forward_and_backward_pass(___stack0, self, mod, collect_outputs, cloned_inputs, pred, loss)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_138_6(___stack0, self, mod, collect_outputs, cloned_inputs, pred, loss)

#============ end of __resume_at_138_6 ============#

def __guard_2_for_resume_in_forward_and_backward_pass(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_type_id(L['self'], 94674336)) \
        and (___check_obj_id(L['___stack0'], 11918016)) \
        and (hasattr(L['___stack1'], '_dynamo_dynamic_indices') == False) \
        and (___check_type_id(L['self'].grad_scaler, 128592976)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139859616489696))) \
        and (___compile_config_hash() == '0487ace406d876273bd209654917c9ce') \
        and (not ___needs_nopython()) \
        and (___check_tensors(L['___stack1'], tensor_check_names=tensor_check_names))

def __transformed_code_2_for_resume_in_forward_and_backward_pass(___stack0, ___stack1, self, mod, collect_outputs, cloned_inputs, pred):
    inputs = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    loss = ___stack1
    return __resume_at_138_6(___stack1.backward(), self, mod, collect_outputs,
        cloned_inputs, pred, loss)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_48_5(___stack0, ___stack1, self, mod, collect_outputs, cloned_inputs, pred):
    with ___stack0() as __temp_24:
        loss = ___stack1
    self.grad_scaler.scale(loss).backward()
    self.optimizer_step()
    if collect_outputs:
        return collect_results(mod, pred, loss, cloned_inputs)
    return None

def transformed___resume_at_48_5(___stack0, ___stack1, self, mod, collect_outputs, cloned_inputs, pred):
    L = {"___stack0": ___stack0, "___stack1": ___stack1, "self": self, "mod": mod, "collect_outputs": collect_outputs, "cloned_inputs": cloned_inputs, "pred": pred}
    if __guard_2_for_resume_in_forward_and_backward_pass(L):
        return __transformed_code_2_for_resume_in_forward_and_backward_pass(___stack0, ___stack1, self, mod, collect_outputs, cloned_inputs, pred)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_48_5(___stack0, ___stack1, self, mod, collect_outputs, cloned_inputs, pred)

#============ end of __resume_at_48_5 ============#

def __guard_1_for_resume_in_forward_and_backward_pass(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['mod'], 139862540550000)) \
        and (L['mod'].training == False) \
        and (___check_type_id(L['self'], 94674336)) \
        and (___check_type_id(L['cloned_inputs'], 7642176)) \
        and (len(L['cloned_inputs']) == 1) \
        and (___check_obj_id(L['self'].autocast, 11918016)) \
        and (hasattr(L['cloned_inputs'][0], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['mod'].forward_head.__defaults__[0], 7677632)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139859616489696))) \
        and (___compile_config_hash() == '0487ace406d876273bd209654917c9ce') \
        and (not ___needs_nopython()) \
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
        and (___check_tensors(L['cloned_inputs'][0], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_3*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_3(*args, **kwargs):
    pass

def __transformed_code_1_for_resume_in_forward_and_backward_pass(___stack0, self, mod, collect_outputs, cloned_inputs):
    inputs = None; loss = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    graph_out_0 = __compiled_fn_3(cloned_inputs[0])
    pred = graph_out_0[0]
    ___context_manager_0_4 = __import_contextlib.nullcontext()
    ___context_manager_0_4.__enter__()
    try:
        __temp_10 = self.compute_loss(graph_out_0[0])
    finally:
        ___context_manager_0_4.__exit__(None, None, None)
    return __resume_at_48_5(__import_contextlib.nullcontext, __temp_10, self,
        mod, collect_outputs, cloned_inputs, pred)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_20_1(___stack0, self, mod, collect_outputs, cloned_inputs):
    with self.autocast() as __temp_31:
        pred = mod(*cloned_inputs)
        loss = self.compute_loss(pred)
    self.grad_scaler.scale(loss).backward()
    self.optimizer_step()
    if collect_outputs:
        return collect_results(mod, pred, loss, cloned_inputs)
    return None

def transformed___resume_at_20_1(___stack0, self, mod, collect_outputs, cloned_inputs):
    L = {"___stack0": ___stack0, "self": self, "mod": mod, "collect_outputs": collect_outputs, "cloned_inputs": cloned_inputs}
    if __guard_1_for_resume_in_forward_and_backward_pass(L):
        return __transformed_code_1_for_resume_in_forward_and_backward_pass(___stack0, self, mod, collect_outputs, cloned_inputs)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_20_1(___stack0, self, mod, collect_outputs, cloned_inputs)

#============ end of __resume_at_20_1 ============#

def __guard_0_for_resume_in_forward_and_backward_pass(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['mod'], 139862540550000)) \
        and (L['mod'].training == False) \
        and (___check_type_id(L['self'], 94674336)) \
        and (___check_type_id(L['___stack0'], 7642176)) \
        and (len(L['___stack0']) == 1) \
        and (hasattr(L['___stack0'][0], '_dynamo_dynamic_indices') == False) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139859616489696))) \
        and (___compile_config_hash() == '0487ace406d876273bd209654917c9ce') \
        and (not ___needs_nopython()) \
        and (___check_tensors(L['___stack0'][0], tensor_check_names=tensor_check_names))

def __transformed_code_0_for_resume_in_forward_and_backward_pass(___stack0, self, mod, collect_outputs):
    inputs = None; loss = None; pred = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    cloned_inputs = ___stack0
    return __resume_at_20_1(self.optimizer_zero_grad(mod), self, mod,
        collect_outputs, cloned_inputs)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_6_0(___stack0, self, mod, collect_outputs):
    cloned_inputs = ___stack0
    self.optimizer_zero_grad(mod)
    with self.autocast() as __temp_41:
        pred = mod(*cloned_inputs)
        loss = self.compute_loss(pred)
    self.grad_scaler.scale(loss).backward()
    self.optimizer_step()
    if collect_outputs:
        return collect_results(mod, pred, loss, cloned_inputs)
    return None

def transformed___resume_at_6_0(___stack0, self, mod, collect_outputs):
    L = {"___stack0": ___stack0, "self": self, "mod": mod, "collect_outputs": collect_outputs}
    if __guard_0_for_resume_in_forward_and_backward_pass(L):
        return __transformed_code_0_for_resume_in_forward_and_backward_pass(___stack0, self, mod, collect_outputs)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_6_0(___stack0, self, mod, collect_outputs)

#============ end of __resume_at_6_0 ============#

def __guard_0_for_forward_and_backward_pass(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_type_id(L['inputs'], 7642176)) \
        and (len(L['inputs']) == 1) \
        and (hasattr(L['inputs'][0], '_dynamo_dynamic_indices') == False) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139859616489696))) \
        and (___compile_config_hash() == '0487ace406d876273bd209654917c9ce') \
        and (not ___needs_nopython()) \
        and (___check_tensors(L['inputs'][0], tensor_check_names=tensor_check_names))

def __transformed_code_0_for_forward_and_backward_pass(self, mod, inputs, collect_outputs):
    cloned_inputs = None; loss = None; pred = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    return __resume_at_6_0(clone_inputs(inputs), self, mod, collect_outputs)


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def forward_and_backward_pass(self, mod, inputs, collect_outputs):
    cloned_inputs = clone_inputs(inputs)
    self.optimizer_zero_grad(mod)
    with self.autocast() as __temp_52:
        pred = mod(*cloned_inputs)
        loss = self.compute_loss(pred)
    self.grad_scaler.scale(loss).backward()
    self.optimizer_step()
    if collect_outputs:
        return collect_results(mod, pred, loss, cloned_inputs)
    return None

def transformed_forward_and_backward_pass(self, mod, inputs, collect_outputs):
    L = {"self": self, "mod": mod, "inputs": inputs, "collect_outputs": collect_outputs}
    if __guard_0_for_forward_and_backward_pass(L):
        return __transformed_code_0_for_forward_and_backward_pass(self, mod, inputs, collect_outputs)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return forward_and_backward_pass(self, mod, inputs, collect_outputs)

#============ end of forward_and_backward_pass ============#
