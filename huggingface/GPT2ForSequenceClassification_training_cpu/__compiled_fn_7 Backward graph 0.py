from __future__ import annotations



def forward(self, primals_2: "i64[1]", sub_1: "f32[1, 2]", ne: "b8[1]", tangents_1: "f32[]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1477, code: loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
    view_1: "i64[1]" = torch.ops.aten.view.default(primals_2, [-1]);  primals_2 = None
    alias: "f32[1, 2]" = torch.ops.aten.alias.default(sub_1);  sub_1 = None
    full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sum_2: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
    div_1: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type);  tangents_1 = convert_element_type = None
    unsqueeze_1: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(view_1, 1);  view_1 = None
    ne_3: "b8[1, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_1, -100)
    where_2: "i64[1, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_1, full_default);  unsqueeze_1 = full_default = None
    full_default_3: "f32[1, 2]" = torch.ops.aten.full.default([1, 2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    scatter: "f32[1, 2]" = torch.ops.aten.scatter.value(full_default_3, 1, where_2, -1.0);  full_default_3 = where_2 = None
    where_3: "f32[1, 1]" = torch.ops.aten.where.self(ne_3, div_1, full_default_1);  ne_3 = div_1 = full_default_1 = None
    mul: "f32[1, 2]" = torch.ops.aten.mul.Tensor(scatter, where_3);  scatter = where_3 = None
    alias_1: "f32[1, 2]" = torch.ops.aten.alias.default(alias);  alias = None
    exp_1: "f32[1, 2]" = torch.ops.aten.exp.default(alias_1);  alias_1 = None
    sum_4: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(mul, [1], True)
    mul_1: "f32[1, 2]" = torch.ops.aten.mul.Tensor(exp_1, sum_4);  exp_1 = sum_4 = None
    sub_2: "f32[1, 2]" = torch.ops.aten.sub.Tensor(mul, mul_1);  mul = mul_1 = None
    return [sub_2, None]
    