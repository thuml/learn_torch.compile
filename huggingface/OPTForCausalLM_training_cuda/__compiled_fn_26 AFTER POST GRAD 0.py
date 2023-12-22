from __future__ import annotations



def forward(self, primals_1: "f32[50272, 768]", primals_2: "f32[1, 2048, 768]", primals_3: "i64[1, 2048]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:956, code: logits = self.lm_head(outputs[0]).contiguous()
    permute: "f32[768, 50272]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
    view: "f32[2048, 768]" = torch.ops.aten.reshape.default(primals_2, [2048, 768]);  primals_2 = None
    mm: "f32[2048, 50272]" = torch.ops.aten.mm.default(view, permute)
    view_1: "f32[1, 2048, 50272]" = torch.ops.aten.reshape.default(mm, [1, 2048, 50272]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:963, code: shift_logits = logits[..., :-1, :].contiguous()
    slice_1: "f32[1, 2047, 50272]" = torch.ops.aten.slice.Tensor(view_1, 1, 0, -1)
    slice_2: "f32[1, 2047, 50272]" = torch.ops.aten.slice.Tensor(slice_1, 2, 0, 9223372036854775807);  slice_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:964, code: shift_labels = labels[..., 1:].contiguous()
    slice_3: "i64[1, 2047]" = torch.ops.aten.slice.Tensor(primals_3, 1, 1, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:967, code: loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
    view_2: "f32[2047, 50272]" = torch.ops.aten.reshape.default(slice_2, [-1, 50272]);  slice_2 = None
    view_3: "i64[2047]" = torch.ops.aten.reshape.default(slice_3, [-1]);  slice_3 = None
    amax: "f32[2047, 1]" = torch.ops.aten.amax.default(view_2, [1], True)
    sub: "f32[2047, 50272]" = torch.ops.aten.sub.Tensor(view_2, amax);  view_2 = amax = None
    exp: "f32[2047, 50272]" = torch.ops.aten.exp.default(sub)
    sum_1: "f32[2047, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
    log: "f32[2047, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
    sub_1: "f32[2047, 50272]" = torch.ops.aten.sub.Tensor(sub, log);  sub = log = None
    ne: "b8[2047]" = torch.ops.aten.ne.Scalar(view_3, -100)
    full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where: "i64[2047]" = torch.ops.aten.where.self(ne, view_3, full_default);  view_3 = full_default = None
    unsqueeze: "i64[2047, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
    gather: "f32[2047, 1]" = torch.ops.aten.gather.default(sub_1, 1, unsqueeze);  unsqueeze = None
    squeeze: "f32[2047]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[2047]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
    full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where_1: "f32[2047]" = torch.ops.aten.where.self(ne, neg, full_default_1);  neg = full_default_1 = None
    sum_2: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
    sum_3: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
    div: "f32[]" = torch.ops.aten.div.Tensor(sum_3, convert_element_type);  sum_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:956, code: logits = self.lm_head(outputs[0]).contiguous()
    permute_3: "f32[50272, 768]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    return [div, view_1, primals_3, view, sub_1, convert_element_type, permute_3]
    