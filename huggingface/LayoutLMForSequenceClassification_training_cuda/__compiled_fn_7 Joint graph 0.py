from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[1, 2]"; primals_2: "i64[1]"; tangents_1: "f32[]"; 

    primals_1, primals_2, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:1105, code: loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    view: "f32[1, 2]" = torch.ops.aten.view.default(primals_1, [-1, 2]);  primals_1 = None
    view_1: "i64[1]" = torch.ops.aten.view.default(primals_2, [-1]);  primals_2 = None
    amax: "f32[1, 1]" = torch.ops.aten.amax.default(view, [1], True)
    sub: "f32[1, 2]" = torch.ops.aten.sub.Tensor(view, amax);  view = amax = None
    exp: "f32[1, 2]" = torch.ops.aten.exp.default(sub)
    sum_1: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
    log: "f32[1, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
    sub_1: "f32[1, 2]" = torch.ops.aten.sub.Tensor(sub, log);  sub = log = None
    alias: "f32[1, 2]" = torch.ops.aten.alias.default(sub_1)
    ne: "b8[1]" = torch.ops.aten.ne.Scalar(view_1, -100)
    scalar_tensor: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0))
    where: "i64[1]" = torch.ops.aten.where.self(ne, view_1, scalar_tensor);  ne = scalar_tensor = None
    unsqueeze: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
    gather: "f32[1, 1]" = torch.ops.aten.gather.default(sub_1, 1, unsqueeze);  sub_1 = unsqueeze = None
    squeeze: "f32[1]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[1]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
    ne_1: "b8[1]" = torch.ops.aten.ne.Scalar(view_1, -100)
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_1: "f32[1]" = torch.ops.aten.where.self(ne_1, neg, scalar_tensor_1);  ne_1 = neg = scalar_tensor_1 = None
    ne_2: "b8[1]" = torch.ops.aten.ne.Scalar(view_1, -100)
    sum_2: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
    sum_3: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
    div: "f32[]" = torch.ops.aten.div.Tensor(sum_3, convert_element_type);  sum_3 = None
    div_1: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type);  tangents_1 = convert_element_type = None
    unsqueeze_1: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(view_1, 1);  view_1 = None
    ne_3: "b8[1, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_1, -100)
    scalar_tensor_2: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0))
    where_2: "i64[1, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_1, scalar_tensor_2);  ne_3 = scalar_tensor_2 = None
    full: "f32[1, 2]" = torch.ops.aten.full.default([1, 2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    scatter: "f32[1, 2]" = torch.ops.aten.scatter.value(full, 1, where_2, -1.0);  full = where_2 = None
    ne_4: "b8[1, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_1, -100);  unsqueeze_1 = None
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_3: "f32[1, 1]" = torch.ops.aten.where.self(ne_4, div_1, scalar_tensor_3);  ne_4 = div_1 = scalar_tensor_3 = None
    mul: "f32[1, 2]" = torch.ops.aten.mul.Tensor(scatter, where_3);  scatter = where_3 = None
    alias_1: "f32[1, 2]" = torch.ops.aten.alias.default(alias);  alias = None
    exp_1: "f32[1, 2]" = torch.ops.aten.exp.default(alias_1);  alias_1 = None
    sum_4: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(mul, [1], True)
    mul_1: "f32[1, 2]" = torch.ops.aten.mul.Tensor(exp_1, sum_4);  exp_1 = sum_4 = None
    sub_2: "f32[1, 2]" = torch.ops.aten.sub.Tensor(mul, mul_1);  mul = mul_1 = None
    view_2: "f32[1, 2]" = torch.ops.aten.view.default(sub_2, [1, 2]);  sub_2 = None
    return pytree.tree_unflatten([div, view_2, None], self._out_spec)
    