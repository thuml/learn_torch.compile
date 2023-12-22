from __future__ import annotations



def forward(self, arg0_1: "f32[1, 2]", arg1_1: "i64[1]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:921, code: loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
    view: "f32[1, 2]" = torch.ops.aten.view.default(arg0_1, [-1, 2]);  arg0_1 = None
    view_1: "i64[1]" = torch.ops.aten.view.default(arg1_1, [-1]);  arg1_1 = None
    amax: "f32[1, 1]" = torch.ops.aten.amax.default(view, [1], True)
    sub: "f32[1, 2]" = torch.ops.aten.sub.Tensor(view, amax);  view = amax = None
    exp: "f32[1, 2]" = torch.ops.aten.exp.default(sub)
    sum_1: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
    log: "f32[1, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
    sub_1: "f32[1, 2]" = torch.ops.aten.sub.Tensor(sub, log);  sub = log = None
    ne: "b8[1]" = torch.ops.aten.ne.Scalar(view_1, -100)
    scalar_tensor: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'))
    where: "i64[1]" = torch.ops.aten.where.self(ne, view_1, scalar_tensor);  ne = scalar_tensor = None
    unsqueeze: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
    gather: "f32[1, 1]" = torch.ops.aten.gather.default(sub_1, 1, unsqueeze);  sub_1 = unsqueeze = None
    squeeze: "f32[1]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[1]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
    ne_1: "b8[1]" = torch.ops.aten.ne.Scalar(view_1, -100)
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_1: "f32[1]" = torch.ops.aten.where.self(ne_1, neg, scalar_tensor_1);  ne_1 = neg = scalar_tensor_1 = None
    ne_2: "b8[1]" = torch.ops.aten.ne.Scalar(view_1, -100);  view_1 = None
    sum_2: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
    sum_3: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
    div: "f32[]" = torch.ops.aten.div.Tensor(sum_3, convert_element_type);  sum_3 = convert_element_type = None
    return (div,)
    