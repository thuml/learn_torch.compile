from __future__ import annotations



def forward(self, primals_1: "f32[256008, 1024]", primals_2: "f32[1, 128, 1024]", primals_3: "i64[1, 128]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:828, code: logits = self.lm_head(outputs[0])
    permute: "f32[1024, 256008]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
    view: "f32[128, 1024]" = torch.ops.aten.reshape.default(primals_2, [128, 1024]);  primals_2 = None
    mm: "f32[128, 256008]" = torch.ops.aten.mm.default(view, permute)
    view_1: "f32[1, 128, 256008]" = torch.ops.aten.reshape.default(mm, [1, 128, 256008]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:833, code: shift_labels = labels.new_zeros(labels.shape)
    full: "i64[1, 128]" = torch.ops.aten.full.default([1, 128], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:834, code: shift_labels[:, :-1] = labels[:, 1:].clone()
    slice_2: "i64[1, 127]" = torch.ops.aten.slice.Tensor(primals_3, 1, 1, 9223372036854775807);  primals_3 = None
    clone: "i64[1, 127]" = torch.ops.aten.clone.default(slice_2);  slice_2 = None
    slice_4: "i64[1, 127]" = torch.ops.aten.slice.Tensor(full, 1, 0, -1)
    copy: "i64[1, 127]" = torch.ops.aten.copy.default(slice_4, clone);  slice_4 = clone = None
    slice_scatter: "i64[1, 128]" = torch.ops.aten.slice_scatter.default(full, copy, 1, 0, -1);  full = copy = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:835, code: shift_labels[:, -1] = self.config.pad_token_id
    full_default: "i64[]" = torch.ops.aten.full.default([], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    select_1: "i64[1]" = torch.ops.aten.select.int(slice_scatter, 1, -1)
    copy_1: "i64[1]" = torch.ops.aten.copy.default(select_1, full_default);  select_1 = full_default = None
    select_scatter: "i64[1, 128]" = torch.ops.aten.select_scatter.default(slice_scatter, copy_1, 1, -1);  slice_scatter = copy_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:838, code: loss = loss_fct(logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
    view_2: "f32[128, 256008]" = torch.ops.aten.reshape.default(view_1, [-1, 256008])
    amax: "f32[128, 1]" = torch.ops.aten.amax.default(view_2, [1], True)
    sub: "f32[128, 256008]" = torch.ops.aten.sub.Tensor(view_2, amax);  view_2 = amax = None
    exp: "f32[128, 256008]" = torch.ops.aten.exp.default(sub)
    sum_1: "f32[128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
    log: "f32[128, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
    sub_1: "f32[128, 256008]" = torch.ops.aten.sub.Tensor(sub, log);  sub = log = None
    view_4: "i64[128]" = torch.ops.aten.reshape.default(select_scatter, [-1])
    ne: "b8[128]" = torch.ops.aten.ne.Scalar(view_4, -100)
    full_default_1: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where: "i64[128]" = torch.ops.aten.where.self(ne, view_4, full_default_1);  view_4 = full_default_1 = None
    unsqueeze: "i64[128, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
    gather: "f32[128, 1]" = torch.ops.aten.gather.default(sub_1, 1, unsqueeze);  unsqueeze = None
    squeeze: "f32[128]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[128]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
    full_default_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where_1: "f32[128]" = torch.ops.aten.where.self(ne, neg, full_default_2);  neg = full_default_2 = None
    sum_2: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
    sum_3: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
    div: "f32[]" = torch.ops.aten.div.Tensor(sum_3, convert_element_type);  sum_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:828, code: logits = self.lm_head(outputs[0])
    permute_3: "f32[256008, 1024]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    return [div, view_1, view, select_scatter, sub_1, convert_element_type, permute_3]
    