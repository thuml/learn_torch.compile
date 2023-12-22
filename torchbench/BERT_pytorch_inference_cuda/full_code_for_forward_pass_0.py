
def __guard_0_for_forward_pass(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_obj_id(L['mod'], 140281430703456)) \
        and (L['mod'].training == False) \
        and (___check_type_id(L['self'], 102451232)) \
        and (___check_type_id(L['inputs'], 7642176)) \
        and (len(L['inputs']) == 2) \
        and (hasattr(L['inputs'][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['inputs'][1], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['self'].autocast, 19692320)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(140278461030624))) \
        and (___compile_config_hash() == '754d567bd8c856980a0c36eca862ea06') \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_forward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks, 7489504)) \
        and (set(G['__import_torch_dot_nn_dot_modules_dot_module']._global_backward_pre_hooks.keys()) == set()) \
        and (___check_type_id(G['__import_torchbenchmark_dot_models_dot_BERT_pytorch_dot_bert_pytorch_dot_model_dot_attention_dot_single'].torch.float16, 140283623737088)) \
        and (G['__import_torchbenchmark_dot_models_dot_BERT_pytorch_dot_bert_pytorch_dot_model_dot_attention_dot_single'].torch.float16 == torch.float16) \
        and (___check_obj_id(L['mod'].transformer_blocks[0].lambda_module.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer_blocks[1].lambda_module.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer_blocks[2].lambda_module.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer_blocks[3].lambda_module.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer_blocks[4].lambda_module.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer_blocks[5].lambda_module.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer_blocks[6].lambda_module.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer_blocks[7].lambda_module.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer_blocks[8].lambda_module.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer_blocks[9].lambda_module.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer_blocks[10].lambda_module.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer_blocks[11].lambda_module.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer_blocks[0].lambda_module.attention.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer_blocks[1].lambda_module.attention.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer_blocks[2].lambda_module.attention.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer_blocks[3].lambda_module.attention.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer_blocks[4].lambda_module.attention.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer_blocks[5].lambda_module.attention.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer_blocks[6].lambda_module.attention.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer_blocks[7].lambda_module.attention.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer_blocks[8].lambda_module.attention.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer_blocks[9].lambda_module.attention.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer_blocks[10].lambda_module.attention.attention.forward.__defaults__[0], 7628576)) \
        and (___check_obj_id(L['mod'].transformer_blocks[11].lambda_module.attention.attention.forward.__defaults__[0], 7628576)) \
        and (___check_tensors(L['inputs'][0], L['inputs'][1], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_0*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_0(*args, **kwargs):
    pass

def __transformed_code_0_for_forward_pass(self, mod, inputs, collect_outputs):
    graph_out_0 = __compiled_fn_0(inputs[0], inputs[1])
    mod.transformer_blocks[11].lambda_module.mask = graph_out_0[1]
    mod.transformer_blocks[10].lambda_module.mask = graph_out_0[1]
    mod.transformer_blocks[9].lambda_module.mask = graph_out_0[1]
    mod.transformer_blocks[8].lambda_module.mask = graph_out_0[1]
    mod.transformer_blocks[7].lambda_module.mask = graph_out_0[1]
    mod.transformer_blocks[6].lambda_module.mask = graph_out_0[1]
    mod.transformer_blocks[5].lambda_module.mask = graph_out_0[1]
    mod.transformer_blocks[4].lambda_module.mask = graph_out_0[1]
    mod.transformer_blocks[3].lambda_module.mask = graph_out_0[1]
    mod.transformer_blocks[2].lambda_module.mask = graph_out_0[1]
    mod.transformer_blocks[1].lambda_module.mask = graph_out_0[1]
    mod.transformer_blocks[0].lambda_module.mask = graph_out_0[1]
    return graph_out_0[0]


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
