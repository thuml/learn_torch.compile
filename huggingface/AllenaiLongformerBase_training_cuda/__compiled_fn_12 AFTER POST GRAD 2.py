from __future__ import annotations



def forward(self, primals_4: "f32[768]", primals_6: "i64[1, 1024]", full_default: "i64[1, 1024]", add: "i64[1, 1024]", mul_2: "f32[1, 1024, 768]", getitem_3: "b8[1, 1024, 768]", div: "f32[1, 1024, 1]", tangents_1: "f32[1, 1024, 768]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:476, code: embeddings = self.dropout(embeddings)
    convert_element_type_3: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_3, torch.float32);  getitem_3 = None
    mul_4: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.1111111111111112);  convert_element_type_3 = None
    mul_5: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(tangents_1, mul_4);  tangents_1 = mul_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:475, code: embeddings = self.LayerNorm(embeddings)
    mul_7: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_5, primals_4);  primals_4 = None
    mul_8: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_7, 768)
    sum_1: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_7, [2], True)
    mul_9: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_7, mul_2);  mul_7 = None
    sum_2: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_9, [2], True);  mul_9 = None
    mul_10: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2, sum_2);  sum_2 = None
    sub_3: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_8, sum_1);  mul_8 = sum_1 = None
    sub_4: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_3, mul_10);  sub_3 = mul_10 = None
    mul_11: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div, sub_4);  div = sub_4 = None
    mul_12: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_5, mul_2);  mul_2 = None
    sum_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_12, [0, 1]);  mul_12 = None
    sum_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_5, [0, 1]);  mul_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:472, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    full_default_2: "b8[1, 1024, 1]" = torch.ops.aten.full.default([1, 1024, 1], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    full_default_3: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where: "f32[1, 1024, 768]" = torch.ops.aten.where.self(full_default_2, full_default_3, mul_11);  full_default_2 = None
    full_default_4: "f32[1, 768]" = torch.ops.aten.full.default([1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put: "f32[1, 768]" = torch.ops.prims._unsafe_index_put_.default(full_default_4, [full_default], where, True);  full_default_4 = full_default = where = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:471, code: position_embeddings = self.position_embeddings(position_ids)
    eq_1: "b8[1, 1024]" = torch.ops.aten.eq.Scalar(add, 1)
    unsqueeze_3: "b8[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
    where_1: "f32[1, 1024, 768]" = torch.ops.aten.where.self(unsqueeze_3, full_default_3, mul_11);  unsqueeze_3 = None
    full_default_6: "f32[4098, 768]" = torch.ops.aten.full.default([4098, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put_1: "f32[4098, 768]" = torch.ops.prims._unsafe_index_put_.default(full_default_6, [add], where_1, True);  full_default_6 = add = where_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:470, code: inputs_embeds = self.word_embeddings(input_ids)
    eq_2: "b8[1, 1024]" = torch.ops.aten.eq.Scalar(primals_6, 1)
    unsqueeze_4: "b8[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_2, -1);  eq_2 = None
    where_2: "f32[1, 1024, 768]" = torch.ops.aten.where.self(unsqueeze_4, full_default_3, mul_11);  unsqueeze_4 = full_default_3 = mul_11 = None
    full_default_8: "f32[50265, 768]" = torch.ops.aten.full.default([50265, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put_2: "f32[50265, 768]" = torch.ops.prims._unsafe_index_put_.default(full_default_8, [primals_6], where_2, True);  full_default_8 = primals_6 = where_2 = None
    return [_unsafe_index_put_2, _unsafe_index_put_1, _unsafe_index_put, sum_3, sum_4, None]
    