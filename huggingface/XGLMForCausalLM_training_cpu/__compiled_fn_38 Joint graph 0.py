from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[256008, 1024]"; primals_2: "f32[1, 128, 1024]"; primals_3: "i64[1, 128]"; tangents_1: "f32[]"; tangents_2: "f32[1, 128, 256008]"; 

    primals_1, primals_2, primals_3, tangents_1, tangents_2, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:828, code: logits = self.lm_head(outputs[0])
    permute: "f32[1024, 256008]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
    view: "f32[128, 1024]" = torch.ops.aten.view.default(primals_2, [128, 1024]);  primals_2 = None
    mm: "f32[128, 256008]" = torch.ops.aten.mm.default(view, permute)
    view_1: "f32[1, 128, 256008]" = torch.ops.aten.view.default(mm, [1, 128, 256008]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:833, code: shift_labels = labels.new_zeros(labels.shape)
    full: "i64[1, 128]" = torch.ops.aten.full.default([1, 128], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:834, code: shift_labels[:, :-1] = labels[:, 1:].clone()
    slice_1: "i64[1, 128]" = torch.ops.aten.slice.Tensor(primals_3, 0, 0, 9223372036854775807);  primals_3 = None
    slice_2: "i64[1, 127]" = torch.ops.aten.slice.Tensor(slice_1, 1, 1, 9223372036854775807);  slice_1 = None
    clone: "i64[1, 127]" = torch.ops.aten.clone.default(slice_2);  slice_2 = None
    slice_3: "i64[1, 128]" = torch.ops.aten.slice.Tensor(full, 0, 0, 9223372036854775807)
    slice_4: "i64[1, 127]" = torch.ops.aten.slice.Tensor(slice_3, 1, 0, -1);  slice_3 = None
    copy: "i64[1, 127]" = torch.ops.aten.copy.default(slice_4, clone);  slice_4 = clone = None
    slice_5: "i64[1, 128]" = torch.ops.aten.slice.Tensor(full, 0, 0, 9223372036854775807)
    slice_scatter: "i64[1, 128]" = torch.ops.aten.slice_scatter.default(slice_5, copy, 1, 0, -1);  slice_5 = copy = None
    slice_scatter_1: "i64[1, 128]" = torch.ops.aten.slice_scatter.default(full, slice_scatter, 0, 0, 9223372036854775807);  full = slice_scatter = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:835, code: shift_labels[:, -1] = self.config.pad_token_id
    _tensor_constant0 = self._tensor_constant0
    lift_fresh_copy: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
    slice_9: "i64[1, 128]" = torch.ops.aten.slice.Tensor(slice_scatter_1, 0, 0, 9223372036854775807)
    select_1: "i64[1]" = torch.ops.aten.select.int(slice_9, 1, -1);  slice_9 = None
    copy_1: "i64[1]" = torch.ops.aten.copy.default(select_1, lift_fresh_copy);  select_1 = lift_fresh_copy = None
    slice_10: "i64[1, 128]" = torch.ops.aten.slice.Tensor(slice_scatter_1, 0, 0, 9223372036854775807)
    select_scatter: "i64[1, 128]" = torch.ops.aten.select_scatter.default(slice_10, copy_1, 1, -1);  slice_10 = copy_1 = None
    slice_scatter_2: "i64[1, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_1, select_scatter, 0, 0, 9223372036854775807);  slice_scatter_1 = select_scatter = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:838, code: loss = loss_fct(logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
    view_2: "f32[128, 256008]" = torch.ops.aten.view.default(view_1, [-1, 256008])
    amax: "f32[128, 1]" = torch.ops.aten.amax.default(view_2, [1], True)
    sub: "f32[128, 256008]" = torch.ops.aten.sub.Tensor(view_2, amax);  view_2 = amax = None
    exp: "f32[128, 256008]" = torch.ops.aten.exp.default(sub)
    sum_1: "f32[128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
    log: "f32[128, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
    sub_1: "f32[128, 256008]" = torch.ops.aten.sub.Tensor(sub, log);  sub = log = None
    alias: "f32[128, 256008]" = torch.ops.aten.alias.default(sub_1)
    view_4: "i64[128]" = torch.ops.aten.view.default(slice_scatter_2, [-1]);  slice_scatter_2 = None
    ne: "b8[128]" = torch.ops.aten.ne.Scalar(view_4, -100)
    scalar_tensor: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'))
    where: "i64[128]" = torch.ops.aten.where.self(ne, view_4, scalar_tensor);  ne = scalar_tensor = None
    unsqueeze: "i64[128, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
    gather: "f32[128, 1]" = torch.ops.aten.gather.default(sub_1, 1, unsqueeze);  sub_1 = unsqueeze = None
    squeeze: "f32[128]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[128]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
    ne_1: "b8[128]" = torch.ops.aten.ne.Scalar(view_4, -100)
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_1: "f32[128]" = torch.ops.aten.where.self(ne_1, neg, scalar_tensor_1);  ne_1 = neg = scalar_tensor_1 = None
    ne_2: "b8[128]" = torch.ops.aten.ne.Scalar(view_4, -100)
    sum_2: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
    sum_3: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
    div: "f32[]" = torch.ops.aten.div.Tensor(sum_3, convert_element_type);  sum_3 = None
    div_1: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type);  tangents_1 = convert_element_type = None
    unsqueeze_1: "i64[128, 1]" = torch.ops.aten.unsqueeze.default(view_4, 1);  view_4 = None
    ne_3: "b8[128, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_1, -100)
    scalar_tensor_2: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'))
    where_2: "i64[128, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_1, scalar_tensor_2);  ne_3 = scalar_tensor_2 = None
    full_1: "f32[128, 256008]" = torch.ops.aten.full.default([128, 256008], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    scatter: "f32[128, 256008]" = torch.ops.aten.scatter.value(full_1, 1, where_2, -1.0);  full_1 = where_2 = None
    ne_4: "b8[128, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_1, -100);  unsqueeze_1 = None
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_3: "f32[128, 1]" = torch.ops.aten.where.self(ne_4, div_1, scalar_tensor_3);  ne_4 = div_1 = scalar_tensor_3 = None
    mul: "f32[128, 256008]" = torch.ops.aten.mul.Tensor(scatter, where_3);  scatter = where_3 = None
    alias_1: "f32[128, 256008]" = torch.ops.aten.alias.default(alias);  alias = None
    exp_1: "f32[128, 256008]" = torch.ops.aten.exp.default(alias_1);  alias_1 = None
    sum_4: "f32[128, 1]" = torch.ops.aten.sum.dim_IntList(mul, [1], True)
    mul_1: "f32[128, 256008]" = torch.ops.aten.mul.Tensor(exp_1, sum_4);  exp_1 = sum_4 = None
    sub_2: "f32[128, 256008]" = torch.ops.aten.sub.Tensor(mul, mul_1);  mul = mul_1 = None
    view_5: "f32[1, 128, 256008]" = torch.ops.aten.view.default(sub_2, [1, 128, 256008]);  sub_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:838, code: loss = loss_fct(logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
    add: "f32[1, 128, 256008]" = torch.ops.aten.add.Tensor(tangents_2, view_5);  tangents_2 = view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:828, code: logits = self.lm_head(outputs[0])
    view_6: "f32[128, 256008]" = torch.ops.aten.view.default(add, [128, 256008]);  add = None
    permute_1: "f32[256008, 128]" = torch.ops.aten.permute.default(view_6, [1, 0])
    mm_1: "f32[256008, 1024]" = torch.ops.aten.mm.default(permute_1, view);  permute_1 = view = None
    permute_2: "f32[1024, 256008]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    permute_3: "f32[256008, 1024]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    mm_2: "f32[128, 1024]" = torch.ops.aten.mm.default(view_6, permute_3);  view_6 = permute_3 = None
    view_7: "f32[1, 128, 1024]" = torch.ops.aten.view.default(mm_2, [1, 128, 1024]);  mm_2 = None
    permute_4: "f32[256008, 1024]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    return pytree.tree_unflatten([div, view_1, permute_4, view_7, None], self._out_spec)
    