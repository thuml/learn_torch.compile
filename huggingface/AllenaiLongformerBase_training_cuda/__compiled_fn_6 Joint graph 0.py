from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[50265, 768]"; primals_2: "f32[4098, 768]"; primals_3: "f32[1, 768]"; primals_4: "f32[768]"; primals_5: "f32[768]"; primals_6: "i64[1, 1024]"; tangents_1: "f32[1, 1024, 768]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1720, code: attention_mask = torch.ones(input_shape, device=device)
    full: "f32[1, 1024]" = torch.ops.aten.full.default([1, 1024], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1722, code: token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
    full_1: "i64[1, 1024]" = torch.ops.aten.full.default([1, 1024], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:916, code: extended_attention_mask = attention_mask[:, None, None, :]
    slice_1: "f32[1, 1024]" = torch.ops.aten.slice.Tensor(full, 0, 0, 9223372036854775807);  full = None
    unsqueeze: "f32[1, 1, 1024]" = torch.ops.aten.unsqueeze.default(slice_1, 1);  slice_1 = None
    unsqueeze_1: "f32[1, 1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
    slice_2: "f32[1, 1, 1, 1024]" = torch.ops.aten.slice.Tensor(unsqueeze_1, 3, 0, 9223372036854775807);  unsqueeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:928, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    sub: "f32[1, 1, 1, 1024]" = torch.ops.aten.sub.Tensor(1.0, slice_2);  slice_2 = None
    mul: "f32[1, 1, 1, 1024]" = torch.ops.aten.mul.Tensor(sub, -3.4028234663852886e+38);  sub = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1739, code: extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)[
    slice_3: "f32[1, 1, 1, 1024]" = torch.ops.aten.slice.Tensor(mul, 0, 0, 9223372036854775807);  mul = None
    select: "f32[1, 1, 1024]" = torch.ops.aten.select.int(slice_3, 1, 0);  slice_3 = None
    select_1: "f32[1, 1024]" = torch.ops.aten.select.int(select, 1, 0);  select = None
    slice_4: "f32[1, 1024]" = torch.ops.aten.slice.Tensor(select_1, 1, 0, 9223372036854775807);  select_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:428, code: mask = input_ids.ne(padding_idx).int()
    ne: "b8[1, 1024]" = torch.ops.aten.ne.Scalar(primals_6, 1)
    convert_element_type: "i32[1, 1024]" = torch.ops.prims.convert_element_type.default(ne, torch.int32);  ne = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:429, code: incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
    cumsum: "i64[1, 1024]" = torch.ops.aten.cumsum.default(convert_element_type, 1)
    convert_element_type_1: "i32[1, 1024]" = torch.ops.prims.convert_element_type.default(cumsum, torch.int32);  cumsum = None
    mul_1: "i32[1, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_1, convert_element_type);  convert_element_type_1 = convert_element_type = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:430, code: return incremental_indices.long() + padding_idx
    convert_element_type_2: "i64[1, 1024]" = torch.ops.prims.convert_element_type.default(mul_1, torch.int64);  mul_1 = None
    add: "i64[1, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_2, 1);  convert_element_type_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:470, code: inputs_embeds = self.word_embeddings(input_ids)
    embedding: "f32[1, 1024, 768]" = torch.ops.aten.embedding.default(primals_1, primals_6, 1);  primals_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:471, code: position_embeddings = self.position_embeddings(position_ids)
    embedding_1: "f32[1, 1024, 768]" = torch.ops.aten.embedding.default(primals_2, add, 1);  primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:472, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    embedding_2: "f32[1, 1024, 768]" = torch.ops.aten.embedding.default(primals_3, full_1);  primals_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:474, code: embeddings = inputs_embeds + position_embeddings + token_type_embeddings
    add_1: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
    add_2: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_1, embedding_2);  add_1 = embedding_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:475, code: embeddings = self.LayerNorm(embeddings)
    var_mean = torch.ops.aten.var_mean.correction(add_2, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 1024, 1]" = var_mean[0]
    getitem_1: "f32[1, 1024, 1]" = var_mean[1];  var_mean = None
    add_3: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
    sub_1: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_2, getitem_1)
    mul_2: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = None
    mul_3: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2, primals_4);  mul_2 = None
    add_4: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_3, primals_5);  mul_3 = primals_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:476, code: embeddings = self.dropout(embeddings)
    native_dropout = torch.ops.aten.native_dropout.default(add_4, 0.1, True);  add_4 = None
    getitem_2: "f32[1, 1024, 768]" = native_dropout[0]
    getitem_3: "b8[1, 1024, 768]" = native_dropout[1];  native_dropout = None
    convert_element_type_3: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_3, torch.float32);  getitem_3 = None
    mul_4: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.1111111111111112);  convert_element_type_3 = None
    mul_5: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(tangents_1, mul_4);  tangents_1 = mul_4 = None
    clone: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_5, memory_format = torch.contiguous_format);  mul_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:475, code: embeddings = self.LayerNorm(embeddings)
    sub_2: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_2, getitem_1);  add_2 = getitem_1 = None
    mul_6: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt);  sub_2 = None
    mul_7: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(clone, primals_4);  primals_4 = None
    mul_8: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_7, 768)
    sum_1: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_7, [2], True)
    mul_9: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_7, mul_6);  mul_7 = None
    sum_2: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_9, [2], True);  mul_9 = None
    mul_10: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_6, sum_2);  sum_2 = None
    sub_3: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_8, sum_1);  mul_8 = sum_1 = None
    sub_4: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_3, mul_10);  sub_3 = mul_10 = None
    div: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
    mul_11: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div, sub_4);  div = sub_4 = None
    mul_12: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(clone, mul_6);  mul_6 = None
    sum_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_12, [0, 1]);  mul_12 = None
    sum_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone, [0, 1]);  clone = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:472, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    eq: "b8[1, 1024]" = torch.ops.aten.eq.Scalar(full_1, -1)
    unsqueeze_2: "b8[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where: "f32[1, 1024, 768]" = torch.ops.aten.where.self(unsqueeze_2, scalar_tensor, mul_11);  unsqueeze_2 = scalar_tensor = None
    full_2: "f32[1, 768]" = torch.ops.aten.full.default([1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put: "f32[1, 768]" = torch.ops.aten._unsafe_index_put.default(full_2, [full_1], where, True);  full_2 = full_1 = where = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:471, code: position_embeddings = self.position_embeddings(position_ids)
    eq_1: "b8[1, 1024]" = torch.ops.aten.eq.Scalar(add, 1)
    unsqueeze_3: "b8[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_1: "f32[1, 1024, 768]" = torch.ops.aten.where.self(unsqueeze_3, scalar_tensor_1, mul_11);  unsqueeze_3 = scalar_tensor_1 = None
    full_3: "f32[4098, 768]" = torch.ops.aten.full.default([4098, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put_1: "f32[4098, 768]" = torch.ops.aten._unsafe_index_put.default(full_3, [add], where_1, True);  full_3 = add = where_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:470, code: inputs_embeds = self.word_embeddings(input_ids)
    eq_2: "b8[1, 1024]" = torch.ops.aten.eq.Scalar(primals_6, 1)
    unsqueeze_4: "b8[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_2, -1);  eq_2 = None
    scalar_tensor_2: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_2: "f32[1, 1024, 768]" = torch.ops.aten.where.self(unsqueeze_4, scalar_tensor_2, mul_11);  unsqueeze_4 = scalar_tensor_2 = mul_11 = None
    full_4: "f32[50265, 768]" = torch.ops.aten.full.default([50265, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put_2: "f32[50265, 768]" = torch.ops.aten._unsafe_index_put.default(full_4, [primals_6], where_2, True);  full_4 = primals_6 = where_2 = None
    return pytree.tree_unflatten([getitem_2, slice_4, _unsafe_index_put_2, _unsafe_index_put_1, _unsafe_index_put, sum_3, sum_4, None], self._out_spec)
    