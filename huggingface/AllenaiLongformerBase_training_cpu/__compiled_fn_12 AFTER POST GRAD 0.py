from __future__ import annotations



def forward(self, primals_3: "f32[768]", primals_8: "i64[1, 1024]", view: "f32[1024, 768]", addmm: "f32[1024, 768]", mul_3: "f32[1, 1024, 768]", view_2: "f32[1024, 768]", sub_2: "f32[1024, 50265]", convert_element_type: "f32[]", permute_2: "f32[50265, 768]", div_2: "f32[1, 1024, 1]", permute_6: "f32[768, 768]", tangents_1: "f32[]", tangents_2: "f32[1, 1024, 50265]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1397, code: x = self.dense(features)
    view_1: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(addmm, [1, 1024, 768]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_1: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1, 0.7071067811865476)
    erf: "f32[1, 1024, 768]" = torch.ops.aten.erf.default(mul_1);  mul_1 = None
    add: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1864, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    view_5: "i64[1024]" = torch.ops.aten.reshape.default(primals_8, [-1]);  primals_8 = None
    full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    div_1: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type);  tangents_1 = convert_element_type = None
    unsqueeze_1: "i64[1024, 1]" = torch.ops.aten.unsqueeze.default(view_5, 1);  view_5 = None
    ne_3: "b8[1024, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_1, -100)
    where_2: "i64[1024, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_1, full_default);  unsqueeze_1 = full_default = None
    full_default_3: "f32[1024, 50265]" = torch.ops.aten.full.default([1024, 50265], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    scatter: "f32[1024, 50265]" = torch.ops.aten.scatter.value(full_default_3, 1, where_2, -1.0);  full_default_3 = where_2 = None
    where_3: "f32[1024, 1]" = torch.ops.aten.where.self(ne_3, div_1, full_default_1);  ne_3 = div_1 = full_default_1 = None
    mul_5: "f32[1024, 50265]" = torch.ops.aten.mul.Tensor(scatter, where_3);  scatter = where_3 = None
    exp_1: "f32[1024, 50265]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
    sum_4: "f32[1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_5, [1], True)
    mul_6: "f32[1024, 50265]" = torch.ops.aten.mul.Tensor(exp_1, sum_4);  exp_1 = sum_4 = None
    sub_3: "f32[1024, 50265]" = torch.ops.aten.sub.Tensor(mul_5, mul_6);  mul_5 = mul_6 = None
    view_6: "f32[1, 1024, 50265]" = torch.ops.aten.reshape.default(sub_3, [1, 1024, 50265]);  sub_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1864, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    add_3: "f32[1, 1024, 50265]" = torch.ops.aten.add.Tensor(tangents_2, view_6);  tangents_2 = view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1402, code: x = self.decoder(x)
    view_7: "f32[1024, 50265]" = torch.ops.aten.reshape.default(add_3, [1024, 50265]);  add_3 = None
    mm: "f32[1024, 768]" = torch.ops.aten.mm.default(view_7, permute_2);  permute_2 = None
    permute_3: "f32[50265, 1024]" = torch.ops.aten.permute.default(view_7, [1, 0])
    mm_1: "f32[50265, 768]" = torch.ops.aten.mm.default(permute_3, view_2);  permute_3 = view_2 = None
    permute_4: "f32[768, 50265]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_5: "f32[1, 50265]" = torch.ops.aten.sum.dim_IntList(view_7, [0], True);  view_7 = None
    view_8: "f32[50265]" = torch.ops.aten.reshape.default(sum_5, [50265]);  sum_5 = None
    permute_5: "f32[50265, 768]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
    view_9: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(mm, [1, 1024, 768]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1399, code: x = self.layer_norm(x)
    mul_8: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_9, primals_3);  primals_3 = None
    mul_9: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_8, 768)
    sum_6: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_8, [2], True)
    mul_10: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_8, mul_3);  mul_8 = None
    sum_7: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_10, [2], True);  mul_10 = None
    mul_11: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_3, sum_7);  sum_7 = None
    sub_5: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_9, sum_6);  mul_9 = sum_6 = None
    sub_6: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_5, mul_11);  sub_5 = mul_11 = None
    mul_12: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_2, sub_6);  div_2 = sub_6 = None
    mul_13: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_9, mul_3);  mul_3 = None
    sum_8: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_13, [0, 1]);  mul_13 = None
    sum_9: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_9, [0, 1]);  view_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_15: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add, 0.5);  add = None
    mul_16: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1, view_1)
    mul_17: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_16, -0.5);  mul_16 = None
    exp_2: "f32[1, 1024, 768]" = torch.ops.aten.exp.default(mul_17);  mul_17 = None
    mul_18: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(exp_2, 0.3989422804014327);  exp_2 = None
    mul_19: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1, mul_18);  view_1 = mul_18 = None
    add_5: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_15, mul_19);  mul_15 = mul_19 = None
    mul_20: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_12, add_5);  mul_12 = add_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1397, code: x = self.dense(features)
    view_10: "f32[1024, 768]" = torch.ops.aten.reshape.default(mul_20, [1024, 768]);  mul_20 = None
    mm_2: "f32[1024, 768]" = torch.ops.aten.mm.default(view_10, permute_6);  permute_6 = None
    permute_7: "f32[768, 1024]" = torch.ops.aten.permute.default(view_10, [1, 0])
    mm_3: "f32[768, 768]" = torch.ops.aten.mm.default(permute_7, view);  permute_7 = view = None
    permute_8: "f32[768, 768]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_10: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_10, [0], True);  view_10 = None
    view_11: "f32[768]" = torch.ops.aten.reshape.default(sum_10, [768]);  sum_10 = None
    permute_9: "f32[768, 768]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    view_12: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(mm_2, [1, 1024, 768]);  mm_2 = None
    return [permute_9, view_11, sum_8, sum_9, permute_5, view_8, view_12, None]
    