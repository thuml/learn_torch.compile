from __future__ import annotations



def forward(self, primals_1: "f32[50265, 768]", primals_2: "f32[4098, 768]", primals_3: "f32[1, 768]", primals_4: "f32[768]", primals_5: "f32[768]", primals_6: "i64[1, 1024]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1722, code: token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
    full_default: "i64[1, 1024]" = torch.ops.aten.full.default([1, 1024], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1739, code: extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)[
    full_default_1: "f32[1, 1024]" = torch.ops.aten.full.default([1, 1024], -0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
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
    embedding_2: "f32[1, 1024, 768]" = torch.ops.aten.embedding.default(primals_3, full_default);  primals_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:474, code: embeddings = inputs_embeds + position_embeddings + token_type_embeddings
    add_1: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
    add_2: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_1, embedding_2);  add_1 = embedding_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:475, code: embeddings = self.LayerNorm(embeddings)
    var_mean = torch.ops.aten.var_mean.correction(add_2, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 1024, 1]" = var_mean[0]
    getitem_1: "f32[1, 1024, 1]" = var_mean[1];  var_mean = None
    add_3: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
    sub_1: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_2, getitem_1);  add_2 = getitem_1 = None
    mul_2: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = None
    mul_3: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2, primals_4)
    add_4: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_3, primals_5);  mul_3 = primals_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:476, code: embeddings = self.dropout(embeddings)
    native_dropout = torch.ops.aten.native_dropout.default(add_4, 0.1, True);  add_4 = None
    getitem_2: "f32[1, 1024, 768]" = native_dropout[0]
    getitem_3: "b8[1, 1024, 768]" = native_dropout[1];  native_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:475, code: embeddings = self.LayerNorm(embeddings)
    div: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
    return [getitem_2, full_default_1, primals_4, primals_6, full_default, add, mul_2, getitem_3, div]
    