from __future__ import annotations



def forward(self, primals_3: "i64[1, 128]", view: "f32[128, 256]", sub_1: "f32[128, 10000]", convert_element_type: "f32[]", permute_3: "f32[10000, 256]", tangents_1: "f32[]", tangents_2: "f32[1, 128, 10000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:943, code: loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
    view_3: "i64[128]" = torch.ops.aten.reshape.default(primals_3, [-1]);  primals_3 = None
    full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    div_1: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type);  tangents_1 = convert_element_type = None
    unsqueeze_1: "i64[128, 1]" = torch.ops.aten.unsqueeze.default(view_3, 1);  view_3 = None
    ne_3: "b8[128, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_1, -100)
    where_2: "i64[128, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_1, full_default);  unsqueeze_1 = full_default = None
    full_default_3: "f32[128, 10000]" = torch.ops.aten.full.default([128, 10000], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    scatter: "f32[128, 10000]" = torch.ops.aten.scatter.value(full_default_3, 1, where_2, -1.0);  full_default_3 = where_2 = None
    where_3: "f32[128, 1]" = torch.ops.aten.where.self(ne_3, div_1, full_default_1);  ne_3 = div_1 = full_default_1 = None
    mul: "f32[128, 10000]" = torch.ops.aten.mul.Tensor(scatter, where_3);  scatter = where_3 = None
    exp_1: "f32[128, 10000]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
    sum_4: "f32[128, 1]" = torch.ops.aten.sum.dim_IntList(mul, [1], True)
    mul_1: "f32[128, 10000]" = torch.ops.aten.mul.Tensor(exp_1, sum_4);  exp_1 = sum_4 = None
    sub_2: "f32[128, 10000]" = torch.ops.aten.sub.Tensor(mul, mul_1);  mul = mul_1 = None
    view_4: "f32[1, 128, 10000]" = torch.ops.aten.reshape.default(sub_2, [1, 128, 10000]);  sub_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:943, code: loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
    add: "f32[1, 128, 10000]" = torch.ops.aten.add.Tensor(tangents_2, view_4);  tangents_2 = view_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:938, code: logits = self.lm_head(outputs[0])
    view_5: "f32[128, 10000]" = torch.ops.aten.reshape.default(add, [128, 10000]);  add = None
    permute_1: "f32[10000, 128]" = torch.ops.aten.permute.default(view_5, [1, 0])
    mm_1: "f32[10000, 256]" = torch.ops.aten.mm.default(permute_1, view);  permute_1 = view = None
    permute_2: "f32[256, 10000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    mm_2: "f32[128, 256]" = torch.ops.aten.mm.default(view_5, permute_3);  view_5 = permute_3 = None
    view_6: "f32[1, 128, 256]" = torch.ops.aten.reshape.default(mm_2, [1, 128, 256]);  mm_2 = None
    permute_4: "f32[10000, 256]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    return [permute_4, view_6, None]
    