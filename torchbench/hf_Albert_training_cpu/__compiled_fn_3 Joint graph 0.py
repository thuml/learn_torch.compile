from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[30000, 128]"; primals_2: "f32[2, 128]"; primals_3: "f32[512, 128]"; primals_4: "f32[128]"; primals_5: "f32[128]"; primals_6: "f32[768, 128]"; primals_7: "f32[768]"; primals_8: "f32[768, 768]"; primals_9: "f32[768]"; primals_10: "f32[768, 768]"; primals_11: "f32[768]"; primals_12: "f32[768, 768]"; primals_13: "f32[768]"; primals_14: "f32[768, 768]"; primals_15: "f32[768]"; primals_16: "f32[768]"; primals_17: "f32[768]"; primals_18: "f32[3072, 768]"; primals_19: "f32[3072]"; primals_20: "f32[768, 3072]"; primals_21: "f32[768]"; primals_22: "f32[768]"; primals_23: "f32[768]"; primals_24: "f32[128, 768]"; primals_25: "f32[128]"; primals_26: "f32[128]"; primals_27: "f32[128]"; primals_28: "f32[30000, 128]"; primals_29: "f32[30000]"; primals_30: "i64[1, 512]"; primals_31: "i64[1, 512]"; primals_32: "i64[4, 512]"; tangents_1: "f32[4, 512, 30000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:715, code: attention_mask = torch.ones(input_shape, device=device)
    full: "f32[4, 512]" = torch.ops.aten.full.default([4, 512], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:718, code: buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
    slice_1: "i64[1, 512]" = torch.ops.aten.slice.Tensor(primals_30, 0, 0, 9223372036854775807);  primals_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:719, code: buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
    expand: "i64[4, 512]" = torch.ops.aten.expand.default(slice_1, [4, 512]);  slice_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:724, code: extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    unsqueeze: "f32[4, 1, 512]" = torch.ops.aten.unsqueeze.default(full, 1);  full = None
    unsqueeze_1: "f32[4, 1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:726, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min
    sub: "f32[4, 1, 1, 512]" = torch.ops.aten.sub.Tensor(1.0, unsqueeze_1);  unsqueeze_1 = None
    mul: "f32[4, 1, 1, 512]" = torch.ops.aten.mul.Tensor(sub, -3.4028234663852886e+38);  sub = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:236, code: position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
    slice_2: "i64[1, 512]" = torch.ops.aten.slice.Tensor(primals_31, 0, 0, 9223372036854775807);  primals_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:250, code: inputs_embeds = self.word_embeddings(input_ids)
    embedding: "f32[4, 512, 128]" = torch.ops.aten.embedding.default(primals_1, primals_32, 0);  primals_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:251, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    embedding_1: "f32[4, 512, 128]" = torch.ops.aten.embedding.default(primals_2, expand);  primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:253, code: embeddings = inputs_embeds + token_type_embeddings
    add: "f32[4, 512, 128]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:255, code: position_embeddings = self.position_embeddings(position_ids)
    embedding_2: "f32[1, 512, 128]" = torch.ops.aten.embedding.default(primals_3, slice_2);  primals_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:256, code: embeddings += position_embeddings
    add_1: "f32[4, 512, 128]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:257, code: embeddings = self.LayerNorm(embeddings)
    var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
    getitem: "f32[4, 512, 1]" = var_mean[0]
    getitem_1: "f32[4, 512, 1]" = var_mean[1];  var_mean = None
    add_2: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
    rsqrt: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
    sub_1: "f32[4, 512, 128]" = torch.ops.aten.sub.Tensor(add_1, getitem_1)
    mul_1: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = None
    mul_2: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(mul_1, primals_4);  mul_1 = None
    add_3: "f32[4, 512, 128]" = torch.ops.aten.add.Tensor(mul_2, primals_5);  mul_2 = primals_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:258, code: embeddings = self.dropout(embeddings)
    clone: "f32[4, 512, 128]" = torch.ops.aten.clone.default(add_3);  add_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:467, code: hidden_states = self.embedding_hidden_mapping_in(hidden_states)
    view: "f32[2048, 128]" = torch.ops.aten.view.default(clone, [2048, 128]);  clone = None
    permute: "f32[128, 768]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
    addmm: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_7, view, permute);  primals_7 = None
    view_1: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm, [4, 512, 768]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_2: "f32[2048, 768]" = torch.ops.aten.view.default(view_1, [2048, 768])
    permute_1: "f32[768, 768]" = torch.ops.aten.permute.default(primals_8, [1, 0])
    addmm_1: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_9, view_2, permute_1)
    view_3: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_1, [4, 512, 768]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_4: "f32[2048, 768]" = torch.ops.aten.view.default(view_1, [2048, 768])
    permute_2: "f32[768, 768]" = torch.ops.aten.permute.default(primals_10, [1, 0])
    addmm_2: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_11, view_4, permute_2)
    view_5: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_2, [4, 512, 768]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_6: "f32[2048, 768]" = torch.ops.aten.view.default(view_1, [2048, 768])
    permute_3: "f32[768, 768]" = torch.ops.aten.permute.default(primals_12, [1, 0])
    addmm_3: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_13, view_6, permute_3)
    view_7: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_3, [4, 512, 768]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_8: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_3, [4, 512, 12, 64]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_4: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_9: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_5, [4, 512, 12, 64]);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_5: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_10: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_7, [4, 512, 12, 64]);  view_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_6: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_7: "f32[4, 12, 64, 512]" = torch.ops.aten.permute.default(permute_5, [0, 1, 3, 2]);  permute_5 = None
    expand_1: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_4, [4, 12, 512, 64]);  permute_4 = None
    clone_1: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
    view_11: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_1, [48, 512, 64]);  clone_1 = None
    expand_2: "f32[4, 12, 64, 512]" = torch.ops.aten.expand.default(permute_7, [4, 12, 64, 512]);  permute_7 = None
    clone_2: "f32[4, 12, 64, 512]" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
    view_12: "f32[48, 64, 512]" = torch.ops.aten.view.default(clone_2, [48, 64, 512]);  clone_2 = None
    bmm: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_11, view_12)
    view_13: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm, [4, 12, 512, 512]);  bmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_13, 8.0);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:336, code: attention_scores = attention_scores + attention_mask
    add_4: "f32[4, 12, 512, 512]" = torch.ops.aten.add.Tensor(div, mul);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax: "f32[4, 12, 512, 1]" = torch.ops.aten.amax.default(add_4, [-1], True)
    sub_2: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_4, amax);  add_4 = amax = None
    exp: "f32[4, 12, 512, 512]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
    sum_1: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div_1: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(div_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    clone_3: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_1);  div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_3: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_3, [4, 12, 512, 512]);  clone_3 = None
    view_14: "f32[48, 512, 512]" = torch.ops.aten.view.default(expand_3, [48, 512, 512]);  expand_3 = None
    expand_4: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_6, [4, 12, 512, 64]);  permute_6 = None
    clone_4: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
    view_15: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_4, [48, 512, 64]);  clone_4 = None
    bmm_1: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_14, view_15)
    view_16: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_1, [4, 12, 512, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    permute_8: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
    clone_5: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
    view_17: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_5, [4, 512, 768]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_18: "f32[2048, 768]" = torch.ops.aten.view.default(view_17, [2048, 768]);  view_17 = None
    permute_9: "f32[768, 768]" = torch.ops.aten.permute.default(primals_14, [1, 0])
    addmm_4: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_15, view_18, permute_9)
    view_19: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_4, [4, 512, 768]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:369, code: projected_context_layer_dropout = self.output_dropout(projected_context_layer)
    clone_6: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_19);  view_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_5: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_1, clone_6);  view_1 = clone_6 = None
    var_mean_1 = torch.ops.aten.var_mean.correction(add_5, [2], correction = 0, keepdim = True)
    getitem_2: "f32[4, 512, 1]" = var_mean_1[0]
    getitem_3: "f32[4, 512, 1]" = var_mean_1[1];  var_mean_1 = None
    add_6: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
    rsqrt_1: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_3: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_5, getitem_3)
    mul_3: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = None
    mul_4: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_3, primals_16);  mul_3 = None
    add_7: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_4, primals_17);  mul_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_20: "f32[2048, 768]" = torch.ops.aten.view.default(add_7, [2048, 768])
    permute_10: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_18, [1, 0])
    addmm_5: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_19, view_20, permute_10)
    view_21: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_5, [4, 512, 3072]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_5: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_21, 0.5)
    pow_1: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_21, 3.0)
    mul_6: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
    add_8: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(view_21, mul_6);  mul_6 = None
    mul_7: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_8, 0.7978845608028654);  add_8 = None
    tanh: "f32[4, 512, 3072]" = torch.ops.aten.tanh.default(mul_7);  mul_7 = None
    alias_1: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(tanh)
    add_9: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
    mul_8: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_5, add_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_22: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_8, [2048, 3072]);  mul_8 = None
    permute_11: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_20, [1, 0])
    addmm_6: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_21, view_22, permute_11)
    view_23: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_6, [4, 512, 768]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_10: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_23, add_7);  view_23 = add_7 = None
    var_mean_2 = torch.ops.aten.var_mean.correction(add_10, [2], correction = 0, keepdim = True)
    getitem_4: "f32[4, 512, 1]" = var_mean_2[0]
    getitem_5: "f32[4, 512, 1]" = var_mean_2[1];  var_mean_2 = None
    add_11: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
    rsqrt_2: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_4: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_10, getitem_5)
    mul_9: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = None
    mul_10: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_9, primals_22);  mul_9 = None
    add_12: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_10, primals_23);  mul_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_24: "f32[2048, 768]" = torch.ops.aten.view.default(add_12, [2048, 768])
    permute_12: "f32[768, 768]" = torch.ops.aten.permute.default(primals_8, [1, 0])
    addmm_7: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_9, view_24, permute_12)
    view_25: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_7, [4, 512, 768]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_26: "f32[2048, 768]" = torch.ops.aten.view.default(add_12, [2048, 768])
    permute_13: "f32[768, 768]" = torch.ops.aten.permute.default(primals_10, [1, 0])
    addmm_8: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_11, view_26, permute_13)
    view_27: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_8, [4, 512, 768]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_28: "f32[2048, 768]" = torch.ops.aten.view.default(add_12, [2048, 768])
    permute_14: "f32[768, 768]" = torch.ops.aten.permute.default(primals_12, [1, 0])
    addmm_9: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_13, view_28, permute_14)
    view_29: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_9, [4, 512, 768]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_30: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_25, [4, 512, 12, 64]);  view_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_15: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_31: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_27, [4, 512, 12, 64]);  view_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_16: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_32: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_29, [4, 512, 12, 64]);  view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_17: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_18: "f32[4, 12, 64, 512]" = torch.ops.aten.permute.default(permute_16, [0, 1, 3, 2]);  permute_16 = None
    expand_5: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_15, [4, 12, 512, 64]);  permute_15 = None
    clone_7: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
    view_33: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_7, [48, 512, 64]);  clone_7 = None
    expand_6: "f32[4, 12, 64, 512]" = torch.ops.aten.expand.default(permute_18, [4, 12, 64, 512]);  permute_18 = None
    clone_8: "f32[4, 12, 64, 512]" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
    view_34: "f32[48, 64, 512]" = torch.ops.aten.view.default(clone_8, [48, 64, 512]);  clone_8 = None
    bmm_2: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_33, view_34)
    view_35: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_2, [4, 12, 512, 512]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_2: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_35, 8.0);  view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:336, code: attention_scores = attention_scores + attention_mask
    add_13: "f32[4, 12, 512, 512]" = torch.ops.aten.add.Tensor(div_2, mul);  div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_1: "f32[4, 12, 512, 1]" = torch.ops.aten.amax.default(add_13, [-1], True)
    sub_5: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_13, amax_1);  add_13 = amax_1 = None
    exp_1: "f32[4, 12, 512, 512]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
    sum_2: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_3: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    alias_2: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(div_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    clone_9: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_3);  div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_7: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_9, [4, 12, 512, 512]);  clone_9 = None
    view_36: "f32[48, 512, 512]" = torch.ops.aten.view.default(expand_7, [48, 512, 512]);  expand_7 = None
    expand_8: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_17, [4, 12, 512, 64]);  permute_17 = None
    clone_10: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
    view_37: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_10, [48, 512, 64]);  clone_10 = None
    bmm_3: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_36, view_37)
    view_38: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_3, [4, 12, 512, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    permute_19: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
    clone_11: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None
    view_39: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_11, [4, 512, 768]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_40: "f32[2048, 768]" = torch.ops.aten.view.default(view_39, [2048, 768]);  view_39 = None
    permute_20: "f32[768, 768]" = torch.ops.aten.permute.default(primals_14, [1, 0])
    addmm_10: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_15, view_40, permute_20)
    view_41: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_10, [4, 512, 768]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:369, code: projected_context_layer_dropout = self.output_dropout(projected_context_layer)
    clone_12: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_41);  view_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_14: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_12, clone_12);  add_12 = clone_12 = None
    var_mean_3 = torch.ops.aten.var_mean.correction(add_14, [2], correction = 0, keepdim = True)
    getitem_6: "f32[4, 512, 1]" = var_mean_3[0]
    getitem_7: "f32[4, 512, 1]" = var_mean_3[1];  var_mean_3 = None
    add_15: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
    rsqrt_3: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
    sub_6: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_14, getitem_7)
    mul_11: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_3);  sub_6 = None
    mul_12: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_11, primals_16);  mul_11 = None
    add_16: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_12, primals_17);  mul_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_42: "f32[2048, 768]" = torch.ops.aten.view.default(add_16, [2048, 768])
    permute_21: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_18, [1, 0])
    addmm_11: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_19, view_42, permute_21)
    view_43: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_11, [4, 512, 3072]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_13: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_43, 0.5)
    pow_2: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_43, 3.0)
    mul_14: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
    add_17: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(view_43, mul_14);  mul_14 = None
    mul_15: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_17, 0.7978845608028654);  add_17 = None
    tanh_1: "f32[4, 512, 3072]" = torch.ops.aten.tanh.default(mul_15);  mul_15 = None
    alias_3: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(tanh_1)
    add_18: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
    mul_16: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_13, add_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_44: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_16, [2048, 3072]);  mul_16 = None
    permute_22: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_20, [1, 0])
    addmm_12: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_21, view_44, permute_22)
    view_45: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_12, [4, 512, 768]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_19: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_45, add_16);  view_45 = add_16 = None
    var_mean_4 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
    getitem_8: "f32[4, 512, 1]" = var_mean_4[0]
    getitem_9: "f32[4, 512, 1]" = var_mean_4[1];  var_mean_4 = None
    add_20: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
    rsqrt_4: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
    sub_7: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_19, getitem_9)
    mul_17: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_4);  sub_7 = None
    mul_18: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_17, primals_22);  mul_17 = None
    add_21: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_18, primals_23);  mul_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_46: "f32[2048, 768]" = torch.ops.aten.view.default(add_21, [2048, 768])
    permute_23: "f32[768, 768]" = torch.ops.aten.permute.default(primals_8, [1, 0])
    addmm_13: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_9, view_46, permute_23)
    view_47: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_13, [4, 512, 768]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_48: "f32[2048, 768]" = torch.ops.aten.view.default(add_21, [2048, 768])
    permute_24: "f32[768, 768]" = torch.ops.aten.permute.default(primals_10, [1, 0])
    addmm_14: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_11, view_48, permute_24)
    view_49: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_14, [4, 512, 768]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_50: "f32[2048, 768]" = torch.ops.aten.view.default(add_21, [2048, 768])
    permute_25: "f32[768, 768]" = torch.ops.aten.permute.default(primals_12, [1, 0])
    addmm_15: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_13, view_50, permute_25)
    view_51: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_15, [4, 512, 768]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_52: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_47, [4, 512, 12, 64]);  view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_26: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_53: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_49, [4, 512, 12, 64]);  view_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_27: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_54: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_51, [4, 512, 12, 64]);  view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_28: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_29: "f32[4, 12, 64, 512]" = torch.ops.aten.permute.default(permute_27, [0, 1, 3, 2]);  permute_27 = None
    expand_9: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_26, [4, 12, 512, 64]);  permute_26 = None
    clone_13: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
    view_55: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_13, [48, 512, 64]);  clone_13 = None
    expand_10: "f32[4, 12, 64, 512]" = torch.ops.aten.expand.default(permute_29, [4, 12, 64, 512]);  permute_29 = None
    clone_14: "f32[4, 12, 64, 512]" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
    view_56: "f32[48, 64, 512]" = torch.ops.aten.view.default(clone_14, [48, 64, 512]);  clone_14 = None
    bmm_4: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_55, view_56)
    view_57: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_4, [4, 12, 512, 512]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_4: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_57, 8.0);  view_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:336, code: attention_scores = attention_scores + attention_mask
    add_22: "f32[4, 12, 512, 512]" = torch.ops.aten.add.Tensor(div_4, mul);  div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_2: "f32[4, 12, 512, 1]" = torch.ops.aten.amax.default(add_22, [-1], True)
    sub_8: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_22, amax_2);  add_22 = amax_2 = None
    exp_2: "f32[4, 12, 512, 512]" = torch.ops.aten.exp.default(sub_8);  sub_8 = None
    sum_3: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_5: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    alias_4: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(div_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    clone_15: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_5);  div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_11: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_15, [4, 12, 512, 512]);  clone_15 = None
    view_58: "f32[48, 512, 512]" = torch.ops.aten.view.default(expand_11, [48, 512, 512]);  expand_11 = None
    expand_12: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_28, [4, 12, 512, 64]);  permute_28 = None
    clone_16: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
    view_59: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_16, [48, 512, 64]);  clone_16 = None
    bmm_5: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_58, view_59)
    view_60: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_5, [4, 12, 512, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    permute_30: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_60, [0, 2, 1, 3]);  view_60 = None
    clone_17: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
    view_61: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_17, [4, 512, 768]);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_62: "f32[2048, 768]" = torch.ops.aten.view.default(view_61, [2048, 768]);  view_61 = None
    permute_31: "f32[768, 768]" = torch.ops.aten.permute.default(primals_14, [1, 0])
    addmm_16: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_15, view_62, permute_31)
    view_63: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_16, [4, 512, 768]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:369, code: projected_context_layer_dropout = self.output_dropout(projected_context_layer)
    clone_18: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_63);  view_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_23: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_21, clone_18);  add_21 = clone_18 = None
    var_mean_5 = torch.ops.aten.var_mean.correction(add_23, [2], correction = 0, keepdim = True)
    getitem_10: "f32[4, 512, 1]" = var_mean_5[0]
    getitem_11: "f32[4, 512, 1]" = var_mean_5[1];  var_mean_5 = None
    add_24: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
    rsqrt_5: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
    sub_9: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_23, getitem_11)
    mul_19: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_5);  sub_9 = None
    mul_20: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_19, primals_16);  mul_19 = None
    add_25: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_20, primals_17);  mul_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_64: "f32[2048, 768]" = torch.ops.aten.view.default(add_25, [2048, 768])
    permute_32: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_18, [1, 0])
    addmm_17: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_19, view_64, permute_32)
    view_65: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_17, [4, 512, 3072]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_21: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_65, 0.5)
    pow_3: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_65, 3.0)
    mul_22: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
    add_26: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(view_65, mul_22);  mul_22 = None
    mul_23: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_26, 0.7978845608028654);  add_26 = None
    tanh_2: "f32[4, 512, 3072]" = torch.ops.aten.tanh.default(mul_23);  mul_23 = None
    alias_5: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(tanh_2)
    add_27: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
    mul_24: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_21, add_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_66: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_24, [2048, 3072]);  mul_24 = None
    permute_33: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_20, [1, 0])
    addmm_18: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_21, view_66, permute_33)
    view_67: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_18, [4, 512, 768]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_28: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_67, add_25);  view_67 = add_25 = None
    var_mean_6 = torch.ops.aten.var_mean.correction(add_28, [2], correction = 0, keepdim = True)
    getitem_12: "f32[4, 512, 1]" = var_mean_6[0]
    getitem_13: "f32[4, 512, 1]" = var_mean_6[1];  var_mean_6 = None
    add_29: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
    rsqrt_6: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
    sub_10: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_28, getitem_13)
    mul_25: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_6);  sub_10 = None
    mul_26: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_25, primals_22);  mul_25 = None
    add_30: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_26, primals_23);  mul_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_68: "f32[2048, 768]" = torch.ops.aten.view.default(add_30, [2048, 768])
    permute_34: "f32[768, 768]" = torch.ops.aten.permute.default(primals_8, [1, 0])
    addmm_19: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_9, view_68, permute_34)
    view_69: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_19, [4, 512, 768]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_70: "f32[2048, 768]" = torch.ops.aten.view.default(add_30, [2048, 768])
    permute_35: "f32[768, 768]" = torch.ops.aten.permute.default(primals_10, [1, 0])
    addmm_20: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_11, view_70, permute_35)
    view_71: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_20, [4, 512, 768]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_72: "f32[2048, 768]" = torch.ops.aten.view.default(add_30, [2048, 768])
    permute_36: "f32[768, 768]" = torch.ops.aten.permute.default(primals_12, [1, 0])
    addmm_21: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_13, view_72, permute_36)
    view_73: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_21, [4, 512, 768]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_74: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_69, [4, 512, 12, 64]);  view_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_37: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_75: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_71, [4, 512, 12, 64]);  view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_38: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_76: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_73, [4, 512, 12, 64]);  view_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_39: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_40: "f32[4, 12, 64, 512]" = torch.ops.aten.permute.default(permute_38, [0, 1, 3, 2]);  permute_38 = None
    expand_13: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_37, [4, 12, 512, 64]);  permute_37 = None
    clone_19: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
    view_77: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_19, [48, 512, 64]);  clone_19 = None
    expand_14: "f32[4, 12, 64, 512]" = torch.ops.aten.expand.default(permute_40, [4, 12, 64, 512]);  permute_40 = None
    clone_20: "f32[4, 12, 64, 512]" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
    view_78: "f32[48, 64, 512]" = torch.ops.aten.view.default(clone_20, [48, 64, 512]);  clone_20 = None
    bmm_6: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_77, view_78)
    view_79: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_6, [4, 12, 512, 512]);  bmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_6: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_79, 8.0);  view_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:336, code: attention_scores = attention_scores + attention_mask
    add_31: "f32[4, 12, 512, 512]" = torch.ops.aten.add.Tensor(div_6, mul);  div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_3: "f32[4, 12, 512, 1]" = torch.ops.aten.amax.default(add_31, [-1], True)
    sub_11: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_31, amax_3);  add_31 = amax_3 = None
    exp_3: "f32[4, 12, 512, 512]" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
    sum_4: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_7: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    alias_6: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(div_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    clone_21: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_7);  div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_15: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_21, [4, 12, 512, 512]);  clone_21 = None
    view_80: "f32[48, 512, 512]" = torch.ops.aten.view.default(expand_15, [48, 512, 512]);  expand_15 = None
    expand_16: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_39, [4, 12, 512, 64]);  permute_39 = None
    clone_22: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
    view_81: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_22, [48, 512, 64]);  clone_22 = None
    bmm_7: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_80, view_81)
    view_82: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_7, [4, 12, 512, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    permute_41: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
    clone_23: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_41, memory_format = torch.contiguous_format);  permute_41 = None
    view_83: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_23, [4, 512, 768]);  clone_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_84: "f32[2048, 768]" = torch.ops.aten.view.default(view_83, [2048, 768]);  view_83 = None
    permute_42: "f32[768, 768]" = torch.ops.aten.permute.default(primals_14, [1, 0])
    addmm_22: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_15, view_84, permute_42)
    view_85: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_22, [4, 512, 768]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:369, code: projected_context_layer_dropout = self.output_dropout(projected_context_layer)
    clone_24: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_85);  view_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_32: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_30, clone_24);  add_30 = clone_24 = None
    var_mean_7 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
    getitem_14: "f32[4, 512, 1]" = var_mean_7[0]
    getitem_15: "f32[4, 512, 1]" = var_mean_7[1];  var_mean_7 = None
    add_33: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
    rsqrt_7: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
    sub_12: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_32, getitem_15)
    mul_27: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_7);  sub_12 = None
    mul_28: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_27, primals_16);  mul_27 = None
    add_34: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_28, primals_17);  mul_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_86: "f32[2048, 768]" = torch.ops.aten.view.default(add_34, [2048, 768])
    permute_43: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_18, [1, 0])
    addmm_23: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_19, view_86, permute_43)
    view_87: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_23, [4, 512, 3072]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_29: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_87, 0.5)
    pow_4: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_87, 3.0)
    mul_30: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
    add_35: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(view_87, mul_30);  mul_30 = None
    mul_31: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_35, 0.7978845608028654);  add_35 = None
    tanh_3: "f32[4, 512, 3072]" = torch.ops.aten.tanh.default(mul_31);  mul_31 = None
    alias_7: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(tanh_3)
    add_36: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
    mul_32: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_29, add_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_88: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_32, [2048, 3072]);  mul_32 = None
    permute_44: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_20, [1, 0])
    addmm_24: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_21, view_88, permute_44)
    view_89: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_24, [4, 512, 768]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_37: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_89, add_34);  view_89 = add_34 = None
    var_mean_8 = torch.ops.aten.var_mean.correction(add_37, [2], correction = 0, keepdim = True)
    getitem_16: "f32[4, 512, 1]" = var_mean_8[0]
    getitem_17: "f32[4, 512, 1]" = var_mean_8[1];  var_mean_8 = None
    add_38: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
    rsqrt_8: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
    sub_13: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_37, getitem_17)
    mul_33: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_8);  sub_13 = None
    mul_34: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_33, primals_22);  mul_33 = None
    add_39: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_34, primals_23);  mul_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_90: "f32[2048, 768]" = torch.ops.aten.view.default(add_39, [2048, 768])
    permute_45: "f32[768, 768]" = torch.ops.aten.permute.default(primals_8, [1, 0])
    addmm_25: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_9, view_90, permute_45)
    view_91: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_25, [4, 512, 768]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_92: "f32[2048, 768]" = torch.ops.aten.view.default(add_39, [2048, 768])
    permute_46: "f32[768, 768]" = torch.ops.aten.permute.default(primals_10, [1, 0])
    addmm_26: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_11, view_92, permute_46)
    view_93: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_26, [4, 512, 768]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_94: "f32[2048, 768]" = torch.ops.aten.view.default(add_39, [2048, 768])
    permute_47: "f32[768, 768]" = torch.ops.aten.permute.default(primals_12, [1, 0])
    addmm_27: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_13, view_94, permute_47)
    view_95: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_27, [4, 512, 768]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_96: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_91, [4, 512, 12, 64]);  view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_48: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_97: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_93, [4, 512, 12, 64]);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_49: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_97, [0, 2, 1, 3]);  view_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_98: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_95, [4, 512, 12, 64]);  view_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_50: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_51: "f32[4, 12, 64, 512]" = torch.ops.aten.permute.default(permute_49, [0, 1, 3, 2]);  permute_49 = None
    expand_17: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_48, [4, 12, 512, 64]);  permute_48 = None
    clone_25: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
    view_99: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_25, [48, 512, 64]);  clone_25 = None
    expand_18: "f32[4, 12, 64, 512]" = torch.ops.aten.expand.default(permute_51, [4, 12, 64, 512]);  permute_51 = None
    clone_26: "f32[4, 12, 64, 512]" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
    view_100: "f32[48, 64, 512]" = torch.ops.aten.view.default(clone_26, [48, 64, 512]);  clone_26 = None
    bmm_8: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_99, view_100)
    view_101: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_8, [4, 12, 512, 512]);  bmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_8: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_101, 8.0);  view_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:336, code: attention_scores = attention_scores + attention_mask
    add_40: "f32[4, 12, 512, 512]" = torch.ops.aten.add.Tensor(div_8, mul);  div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_4: "f32[4, 12, 512, 1]" = torch.ops.aten.amax.default(add_40, [-1], True)
    sub_14: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_40, amax_4);  add_40 = amax_4 = None
    exp_4: "f32[4, 12, 512, 512]" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
    sum_5: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_9: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    alias_8: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(div_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    clone_27: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_9);  div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_19: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_27, [4, 12, 512, 512]);  clone_27 = None
    view_102: "f32[48, 512, 512]" = torch.ops.aten.view.default(expand_19, [48, 512, 512]);  expand_19 = None
    expand_20: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_50, [4, 12, 512, 64]);  permute_50 = None
    clone_28: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
    view_103: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_28, [48, 512, 64]);  clone_28 = None
    bmm_9: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_102, view_103)
    view_104: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_9, [4, 12, 512, 64]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    permute_52: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None
    clone_29: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None
    view_105: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_29, [4, 512, 768]);  clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_106: "f32[2048, 768]" = torch.ops.aten.view.default(view_105, [2048, 768]);  view_105 = None
    permute_53: "f32[768, 768]" = torch.ops.aten.permute.default(primals_14, [1, 0])
    addmm_28: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_15, view_106, permute_53)
    view_107: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_28, [4, 512, 768]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:369, code: projected_context_layer_dropout = self.output_dropout(projected_context_layer)
    clone_30: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_107);  view_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_41: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_39, clone_30);  add_39 = clone_30 = None
    var_mean_9 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
    getitem_18: "f32[4, 512, 1]" = var_mean_9[0]
    getitem_19: "f32[4, 512, 1]" = var_mean_9[1];  var_mean_9 = None
    add_42: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
    rsqrt_9: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_15: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_41, getitem_19)
    mul_35: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_9);  sub_15 = None
    mul_36: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_35, primals_16);  mul_35 = None
    add_43: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_36, primals_17);  mul_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_108: "f32[2048, 768]" = torch.ops.aten.view.default(add_43, [2048, 768])
    permute_54: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_18, [1, 0])
    addmm_29: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_19, view_108, permute_54)
    view_109: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_29, [4, 512, 3072]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_37: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_109, 0.5)
    pow_5: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_109, 3.0)
    mul_38: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
    add_44: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(view_109, mul_38);  mul_38 = None
    mul_39: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_44, 0.7978845608028654);  add_44 = None
    tanh_4: "f32[4, 512, 3072]" = torch.ops.aten.tanh.default(mul_39);  mul_39 = None
    alias_9: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(tanh_4)
    add_45: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
    mul_40: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_37, add_45)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_110: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_40, [2048, 3072]);  mul_40 = None
    permute_55: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_20, [1, 0])
    addmm_30: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_21, view_110, permute_55)
    view_111: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_30, [4, 512, 768]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_46: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_111, add_43);  view_111 = add_43 = None
    var_mean_10 = torch.ops.aten.var_mean.correction(add_46, [2], correction = 0, keepdim = True)
    getitem_20: "f32[4, 512, 1]" = var_mean_10[0]
    getitem_21: "f32[4, 512, 1]" = var_mean_10[1];  var_mean_10 = None
    add_47: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
    rsqrt_10: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
    sub_16: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_46, getitem_21)
    mul_41: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_10);  sub_16 = None
    mul_42: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_41, primals_22);  mul_41 = None
    add_48: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_42, primals_23);  mul_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_112: "f32[2048, 768]" = torch.ops.aten.view.default(add_48, [2048, 768])
    permute_56: "f32[768, 768]" = torch.ops.aten.permute.default(primals_8, [1, 0])
    addmm_31: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_9, view_112, permute_56)
    view_113: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_31, [4, 512, 768]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_114: "f32[2048, 768]" = torch.ops.aten.view.default(add_48, [2048, 768])
    permute_57: "f32[768, 768]" = torch.ops.aten.permute.default(primals_10, [1, 0])
    addmm_32: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_11, view_114, permute_57)
    view_115: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_32, [4, 512, 768]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_116: "f32[2048, 768]" = torch.ops.aten.view.default(add_48, [2048, 768])
    permute_58: "f32[768, 768]" = torch.ops.aten.permute.default(primals_12, [1, 0])
    addmm_33: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_13, view_116, permute_58)
    view_117: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_33, [4, 512, 768]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_118: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_113, [4, 512, 12, 64]);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_59: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_119: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_115, [4, 512, 12, 64]);  view_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_60: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_119, [0, 2, 1, 3]);  view_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_120: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_117, [4, 512, 12, 64]);  view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_61: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_62: "f32[4, 12, 64, 512]" = torch.ops.aten.permute.default(permute_60, [0, 1, 3, 2]);  permute_60 = None
    expand_21: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_59, [4, 12, 512, 64]);  permute_59 = None
    clone_31: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
    view_121: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_31, [48, 512, 64]);  clone_31 = None
    expand_22: "f32[4, 12, 64, 512]" = torch.ops.aten.expand.default(permute_62, [4, 12, 64, 512]);  permute_62 = None
    clone_32: "f32[4, 12, 64, 512]" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
    view_122: "f32[48, 64, 512]" = torch.ops.aten.view.default(clone_32, [48, 64, 512]);  clone_32 = None
    bmm_10: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_121, view_122)
    view_123: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_10, [4, 12, 512, 512]);  bmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_10: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_123, 8.0);  view_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:336, code: attention_scores = attention_scores + attention_mask
    add_49: "f32[4, 12, 512, 512]" = torch.ops.aten.add.Tensor(div_10, mul);  div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_5: "f32[4, 12, 512, 1]" = torch.ops.aten.amax.default(add_49, [-1], True)
    sub_17: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_49, amax_5);  add_49 = amax_5 = None
    exp_5: "f32[4, 12, 512, 512]" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
    sum_6: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_11: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    alias_10: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(div_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    clone_33: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_11);  div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_23: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_33, [4, 12, 512, 512]);  clone_33 = None
    view_124: "f32[48, 512, 512]" = torch.ops.aten.view.default(expand_23, [48, 512, 512]);  expand_23 = None
    expand_24: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_61, [4, 12, 512, 64]);  permute_61 = None
    clone_34: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
    view_125: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_34, [48, 512, 64]);  clone_34 = None
    bmm_11: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_124, view_125)
    view_126: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_11, [4, 12, 512, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    permute_63: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
    clone_35: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format);  permute_63 = None
    view_127: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_35, [4, 512, 768]);  clone_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_128: "f32[2048, 768]" = torch.ops.aten.view.default(view_127, [2048, 768]);  view_127 = None
    permute_64: "f32[768, 768]" = torch.ops.aten.permute.default(primals_14, [1, 0])
    addmm_34: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_15, view_128, permute_64)
    view_129: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_34, [4, 512, 768]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:369, code: projected_context_layer_dropout = self.output_dropout(projected_context_layer)
    clone_36: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_129);  view_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_50: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_48, clone_36);  add_48 = clone_36 = None
    var_mean_11 = torch.ops.aten.var_mean.correction(add_50, [2], correction = 0, keepdim = True)
    getitem_22: "f32[4, 512, 1]" = var_mean_11[0]
    getitem_23: "f32[4, 512, 1]" = var_mean_11[1];  var_mean_11 = None
    add_51: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
    rsqrt_11: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
    sub_18: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_50, getitem_23)
    mul_43: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_11);  sub_18 = None
    mul_44: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_43, primals_16);  mul_43 = None
    add_52: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_44, primals_17);  mul_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_130: "f32[2048, 768]" = torch.ops.aten.view.default(add_52, [2048, 768])
    permute_65: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_18, [1, 0])
    addmm_35: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_19, view_130, permute_65)
    view_131: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_35, [4, 512, 3072]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_45: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_131, 0.5)
    pow_6: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_131, 3.0)
    mul_46: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
    add_53: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(view_131, mul_46);  mul_46 = None
    mul_47: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_53, 0.7978845608028654);  add_53 = None
    tanh_5: "f32[4, 512, 3072]" = torch.ops.aten.tanh.default(mul_47);  mul_47 = None
    alias_11: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(tanh_5)
    add_54: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
    mul_48: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_45, add_54)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_132: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_48, [2048, 3072]);  mul_48 = None
    permute_66: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_20, [1, 0])
    addmm_36: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_21, view_132, permute_66)
    view_133: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_36, [4, 512, 768]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_55: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_133, add_52);  view_133 = add_52 = None
    var_mean_12 = torch.ops.aten.var_mean.correction(add_55, [2], correction = 0, keepdim = True)
    getitem_24: "f32[4, 512, 1]" = var_mean_12[0]
    getitem_25: "f32[4, 512, 1]" = var_mean_12[1];  var_mean_12 = None
    add_56: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
    rsqrt_12: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    sub_19: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_55, getitem_25)
    mul_49: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_12);  sub_19 = None
    mul_50: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_49, primals_22);  mul_49 = None
    add_57: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_50, primals_23);  mul_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_134: "f32[2048, 768]" = torch.ops.aten.view.default(add_57, [2048, 768])
    permute_67: "f32[768, 768]" = torch.ops.aten.permute.default(primals_8, [1, 0])
    addmm_37: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_9, view_134, permute_67)
    view_135: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_37, [4, 512, 768]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_136: "f32[2048, 768]" = torch.ops.aten.view.default(add_57, [2048, 768])
    permute_68: "f32[768, 768]" = torch.ops.aten.permute.default(primals_10, [1, 0])
    addmm_38: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_11, view_136, permute_68)
    view_137: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_38, [4, 512, 768]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_138: "f32[2048, 768]" = torch.ops.aten.view.default(add_57, [2048, 768])
    permute_69: "f32[768, 768]" = torch.ops.aten.permute.default(primals_12, [1, 0])
    addmm_39: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_13, view_138, permute_69)
    view_139: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_39, [4, 512, 768]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_140: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_135, [4, 512, 12, 64]);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_70: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_141: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_137, [4, 512, 12, 64]);  view_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_71: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_141, [0, 2, 1, 3]);  view_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_142: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_139, [4, 512, 12, 64]);  view_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_72: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_142, [0, 2, 1, 3]);  view_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_73: "f32[4, 12, 64, 512]" = torch.ops.aten.permute.default(permute_71, [0, 1, 3, 2]);  permute_71 = None
    expand_25: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_70, [4, 12, 512, 64]);  permute_70 = None
    clone_37: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
    view_143: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_37, [48, 512, 64]);  clone_37 = None
    expand_26: "f32[4, 12, 64, 512]" = torch.ops.aten.expand.default(permute_73, [4, 12, 64, 512]);  permute_73 = None
    clone_38: "f32[4, 12, 64, 512]" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
    view_144: "f32[48, 64, 512]" = torch.ops.aten.view.default(clone_38, [48, 64, 512]);  clone_38 = None
    bmm_12: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_143, view_144)
    view_145: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_12, [4, 12, 512, 512]);  bmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_12: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_145, 8.0);  view_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:336, code: attention_scores = attention_scores + attention_mask
    add_58: "f32[4, 12, 512, 512]" = torch.ops.aten.add.Tensor(div_12, mul);  div_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_6: "f32[4, 12, 512, 1]" = torch.ops.aten.amax.default(add_58, [-1], True)
    sub_20: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_58, amax_6);  add_58 = amax_6 = None
    exp_6: "f32[4, 12, 512, 512]" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
    sum_7: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_13: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    alias_12: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(div_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    clone_39: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_13);  div_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_27: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_39, [4, 12, 512, 512]);  clone_39 = None
    view_146: "f32[48, 512, 512]" = torch.ops.aten.view.default(expand_27, [48, 512, 512]);  expand_27 = None
    expand_28: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_72, [4, 12, 512, 64]);  permute_72 = None
    clone_40: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
    view_147: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_40, [48, 512, 64]);  clone_40 = None
    bmm_13: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_146, view_147)
    view_148: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_13, [4, 12, 512, 64]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    permute_74: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
    clone_41: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_74, memory_format = torch.contiguous_format);  permute_74 = None
    view_149: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_41, [4, 512, 768]);  clone_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_150: "f32[2048, 768]" = torch.ops.aten.view.default(view_149, [2048, 768]);  view_149 = None
    permute_75: "f32[768, 768]" = torch.ops.aten.permute.default(primals_14, [1, 0])
    addmm_40: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_15, view_150, permute_75)
    view_151: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_40, [4, 512, 768]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:369, code: projected_context_layer_dropout = self.output_dropout(projected_context_layer)
    clone_42: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_151);  view_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_59: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_57, clone_42);  add_57 = clone_42 = None
    var_mean_13 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
    getitem_26: "f32[4, 512, 1]" = var_mean_13[0]
    getitem_27: "f32[4, 512, 1]" = var_mean_13[1];  var_mean_13 = None
    add_60: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
    rsqrt_13: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    sub_21: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_59, getitem_27)
    mul_51: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_13);  sub_21 = None
    mul_52: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_51, primals_16);  mul_51 = None
    add_61: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_52, primals_17);  mul_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_152: "f32[2048, 768]" = torch.ops.aten.view.default(add_61, [2048, 768])
    permute_76: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_18, [1, 0])
    addmm_41: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_19, view_152, permute_76)
    view_153: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_41, [4, 512, 3072]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_53: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_153, 0.5)
    pow_7: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_153, 3.0)
    mul_54: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
    add_62: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(view_153, mul_54);  mul_54 = None
    mul_55: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_62, 0.7978845608028654);  add_62 = None
    tanh_6: "f32[4, 512, 3072]" = torch.ops.aten.tanh.default(mul_55);  mul_55 = None
    alias_13: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(tanh_6)
    add_63: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
    mul_56: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_53, add_63)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_154: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_56, [2048, 3072]);  mul_56 = None
    permute_77: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_20, [1, 0])
    addmm_42: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_21, view_154, permute_77)
    view_155: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_42, [4, 512, 768]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_64: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_155, add_61);  view_155 = add_61 = None
    var_mean_14 = torch.ops.aten.var_mean.correction(add_64, [2], correction = 0, keepdim = True)
    getitem_28: "f32[4, 512, 1]" = var_mean_14[0]
    getitem_29: "f32[4, 512, 1]" = var_mean_14[1];  var_mean_14 = None
    add_65: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
    rsqrt_14: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
    sub_22: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_64, getitem_29)
    mul_57: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_14);  sub_22 = None
    mul_58: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_57, primals_22);  mul_57 = None
    add_66: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_58, primals_23);  mul_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_156: "f32[2048, 768]" = torch.ops.aten.view.default(add_66, [2048, 768])
    permute_78: "f32[768, 768]" = torch.ops.aten.permute.default(primals_8, [1, 0])
    addmm_43: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_9, view_156, permute_78)
    view_157: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_43, [4, 512, 768]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_158: "f32[2048, 768]" = torch.ops.aten.view.default(add_66, [2048, 768])
    permute_79: "f32[768, 768]" = torch.ops.aten.permute.default(primals_10, [1, 0])
    addmm_44: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_11, view_158, permute_79)
    view_159: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_44, [4, 512, 768]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_160: "f32[2048, 768]" = torch.ops.aten.view.default(add_66, [2048, 768])
    permute_80: "f32[768, 768]" = torch.ops.aten.permute.default(primals_12, [1, 0])
    addmm_45: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_13, view_160, permute_80)
    view_161: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_45, [4, 512, 768]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_162: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_157, [4, 512, 12, 64]);  view_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_81: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_163: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_159, [4, 512, 12, 64]);  view_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_82: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_163, [0, 2, 1, 3]);  view_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_164: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_161, [4, 512, 12, 64]);  view_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_83: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_164, [0, 2, 1, 3]);  view_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_84: "f32[4, 12, 64, 512]" = torch.ops.aten.permute.default(permute_82, [0, 1, 3, 2]);  permute_82 = None
    expand_29: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_81, [4, 12, 512, 64]);  permute_81 = None
    clone_43: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
    view_165: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_43, [48, 512, 64]);  clone_43 = None
    expand_30: "f32[4, 12, 64, 512]" = torch.ops.aten.expand.default(permute_84, [4, 12, 64, 512]);  permute_84 = None
    clone_44: "f32[4, 12, 64, 512]" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
    view_166: "f32[48, 64, 512]" = torch.ops.aten.view.default(clone_44, [48, 64, 512]);  clone_44 = None
    bmm_14: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_165, view_166)
    view_167: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_14, [4, 12, 512, 512]);  bmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_14: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_167, 8.0);  view_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:336, code: attention_scores = attention_scores + attention_mask
    add_67: "f32[4, 12, 512, 512]" = torch.ops.aten.add.Tensor(div_14, mul);  div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_7: "f32[4, 12, 512, 1]" = torch.ops.aten.amax.default(add_67, [-1], True)
    sub_23: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_67, amax_7);  add_67 = amax_7 = None
    exp_7: "f32[4, 12, 512, 512]" = torch.ops.aten.exp.default(sub_23);  sub_23 = None
    sum_8: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_15: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    alias_14: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(div_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    clone_45: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_15);  div_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_31: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_45, [4, 12, 512, 512]);  clone_45 = None
    view_168: "f32[48, 512, 512]" = torch.ops.aten.view.default(expand_31, [48, 512, 512]);  expand_31 = None
    expand_32: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_83, [4, 12, 512, 64]);  permute_83 = None
    clone_46: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
    view_169: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_46, [48, 512, 64]);  clone_46 = None
    bmm_15: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_168, view_169)
    view_170: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_15, [4, 12, 512, 64]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    permute_85: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_170, [0, 2, 1, 3]);  view_170 = None
    clone_47: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_85, memory_format = torch.contiguous_format);  permute_85 = None
    view_171: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_47, [4, 512, 768]);  clone_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_172: "f32[2048, 768]" = torch.ops.aten.view.default(view_171, [2048, 768]);  view_171 = None
    permute_86: "f32[768, 768]" = torch.ops.aten.permute.default(primals_14, [1, 0])
    addmm_46: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_15, view_172, permute_86)
    view_173: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_46, [4, 512, 768]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:369, code: projected_context_layer_dropout = self.output_dropout(projected_context_layer)
    clone_48: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_173);  view_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_68: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_66, clone_48);  add_66 = clone_48 = None
    var_mean_15 = torch.ops.aten.var_mean.correction(add_68, [2], correction = 0, keepdim = True)
    getitem_30: "f32[4, 512, 1]" = var_mean_15[0]
    getitem_31: "f32[4, 512, 1]" = var_mean_15[1];  var_mean_15 = None
    add_69: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
    rsqrt_15: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
    sub_24: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_68, getitem_31)
    mul_59: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_15);  sub_24 = None
    mul_60: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_59, primals_16);  mul_59 = None
    add_70: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_60, primals_17);  mul_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_174: "f32[2048, 768]" = torch.ops.aten.view.default(add_70, [2048, 768])
    permute_87: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_18, [1, 0])
    addmm_47: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_19, view_174, permute_87)
    view_175: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_47, [4, 512, 3072]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_61: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_175, 0.5)
    pow_8: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_175, 3.0)
    mul_62: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
    add_71: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(view_175, mul_62);  mul_62 = None
    mul_63: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_71, 0.7978845608028654);  add_71 = None
    tanh_7: "f32[4, 512, 3072]" = torch.ops.aten.tanh.default(mul_63);  mul_63 = None
    alias_15: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(tanh_7)
    add_72: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
    mul_64: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_61, add_72)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_176: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_64, [2048, 3072]);  mul_64 = None
    permute_88: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_20, [1, 0])
    addmm_48: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_21, view_176, permute_88)
    view_177: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_48, [4, 512, 768]);  addmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_73: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_177, add_70);  view_177 = add_70 = None
    var_mean_16 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
    getitem_32: "f32[4, 512, 1]" = var_mean_16[0]
    getitem_33: "f32[4, 512, 1]" = var_mean_16[1];  var_mean_16 = None
    add_74: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
    rsqrt_16: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    sub_25: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_73, getitem_33)
    mul_65: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_16);  sub_25 = None
    mul_66: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_65, primals_22);  mul_65 = None
    add_75: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_66, primals_23);  mul_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_178: "f32[2048, 768]" = torch.ops.aten.view.default(add_75, [2048, 768])
    permute_89: "f32[768, 768]" = torch.ops.aten.permute.default(primals_8, [1, 0])
    addmm_49: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_9, view_178, permute_89)
    view_179: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_49, [4, 512, 768]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_180: "f32[2048, 768]" = torch.ops.aten.view.default(add_75, [2048, 768])
    permute_90: "f32[768, 768]" = torch.ops.aten.permute.default(primals_10, [1, 0])
    addmm_50: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_11, view_180, permute_90)
    view_181: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_50, [4, 512, 768]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_182: "f32[2048, 768]" = torch.ops.aten.view.default(add_75, [2048, 768])
    permute_91: "f32[768, 768]" = torch.ops.aten.permute.default(primals_12, [1, 0])
    addmm_51: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_13, view_182, permute_91)
    view_183: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_51, [4, 512, 768]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_184: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_179, [4, 512, 12, 64]);  view_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_92: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_185: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_181, [4, 512, 12, 64]);  view_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_93: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_186: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_183, [4, 512, 12, 64]);  view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_94: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_95: "f32[4, 12, 64, 512]" = torch.ops.aten.permute.default(permute_93, [0, 1, 3, 2]);  permute_93 = None
    expand_33: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_92, [4, 12, 512, 64]);  permute_92 = None
    clone_49: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
    view_187: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_49, [48, 512, 64]);  clone_49 = None
    expand_34: "f32[4, 12, 64, 512]" = torch.ops.aten.expand.default(permute_95, [4, 12, 64, 512]);  permute_95 = None
    clone_50: "f32[4, 12, 64, 512]" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
    view_188: "f32[48, 64, 512]" = torch.ops.aten.view.default(clone_50, [48, 64, 512]);  clone_50 = None
    bmm_16: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_187, view_188)
    view_189: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_16, [4, 12, 512, 512]);  bmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_16: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_189, 8.0);  view_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:336, code: attention_scores = attention_scores + attention_mask
    add_76: "f32[4, 12, 512, 512]" = torch.ops.aten.add.Tensor(div_16, mul);  div_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_8: "f32[4, 12, 512, 1]" = torch.ops.aten.amax.default(add_76, [-1], True)
    sub_26: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_76, amax_8);  add_76 = amax_8 = None
    exp_8: "f32[4, 12, 512, 512]" = torch.ops.aten.exp.default(sub_26);  sub_26 = None
    sum_9: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_17: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    alias_16: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(div_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    clone_51: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_17);  div_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_35: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_51, [4, 12, 512, 512]);  clone_51 = None
    view_190: "f32[48, 512, 512]" = torch.ops.aten.view.default(expand_35, [48, 512, 512]);  expand_35 = None
    expand_36: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_94, [4, 12, 512, 64]);  permute_94 = None
    clone_52: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
    view_191: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_52, [48, 512, 64]);  clone_52 = None
    bmm_17: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_190, view_191)
    view_192: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_17, [4, 12, 512, 64]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    permute_96: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None
    clone_53: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_96, memory_format = torch.contiguous_format);  permute_96 = None
    view_193: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_53, [4, 512, 768]);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_194: "f32[2048, 768]" = torch.ops.aten.view.default(view_193, [2048, 768]);  view_193 = None
    permute_97: "f32[768, 768]" = torch.ops.aten.permute.default(primals_14, [1, 0])
    addmm_52: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_15, view_194, permute_97)
    view_195: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_52, [4, 512, 768]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:369, code: projected_context_layer_dropout = self.output_dropout(projected_context_layer)
    clone_54: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_195);  view_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_77: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_75, clone_54);  add_75 = clone_54 = None
    var_mean_17 = torch.ops.aten.var_mean.correction(add_77, [2], correction = 0, keepdim = True)
    getitem_34: "f32[4, 512, 1]" = var_mean_17[0]
    getitem_35: "f32[4, 512, 1]" = var_mean_17[1];  var_mean_17 = None
    add_78: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
    rsqrt_17: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    sub_27: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_77, getitem_35)
    mul_67: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_17);  sub_27 = None
    mul_68: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_67, primals_16);  mul_67 = None
    add_79: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_68, primals_17);  mul_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_196: "f32[2048, 768]" = torch.ops.aten.view.default(add_79, [2048, 768])
    permute_98: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_18, [1, 0])
    addmm_53: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_19, view_196, permute_98)
    view_197: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_53, [4, 512, 3072]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_69: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_197, 0.5)
    pow_9: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_197, 3.0)
    mul_70: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
    add_80: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(view_197, mul_70);  mul_70 = None
    mul_71: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_80, 0.7978845608028654);  add_80 = None
    tanh_8: "f32[4, 512, 3072]" = torch.ops.aten.tanh.default(mul_71);  mul_71 = None
    alias_17: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(tanh_8)
    add_81: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
    mul_72: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_69, add_81)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_198: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_72, [2048, 3072]);  mul_72 = None
    permute_99: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_20, [1, 0])
    addmm_54: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_21, view_198, permute_99)
    view_199: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_54, [4, 512, 768]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_82: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_199, add_79);  view_199 = add_79 = None
    var_mean_18 = torch.ops.aten.var_mean.correction(add_82, [2], correction = 0, keepdim = True)
    getitem_36: "f32[4, 512, 1]" = var_mean_18[0]
    getitem_37: "f32[4, 512, 1]" = var_mean_18[1];  var_mean_18 = None
    add_83: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
    rsqrt_18: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
    sub_28: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_82, getitem_37)
    mul_73: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_18);  sub_28 = None
    mul_74: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_73, primals_22);  mul_73 = None
    add_84: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_74, primals_23);  mul_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_200: "f32[2048, 768]" = torch.ops.aten.view.default(add_84, [2048, 768])
    permute_100: "f32[768, 768]" = torch.ops.aten.permute.default(primals_8, [1, 0])
    addmm_55: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_9, view_200, permute_100)
    view_201: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_55, [4, 512, 768]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_202: "f32[2048, 768]" = torch.ops.aten.view.default(add_84, [2048, 768])
    permute_101: "f32[768, 768]" = torch.ops.aten.permute.default(primals_10, [1, 0])
    addmm_56: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_11, view_202, permute_101)
    view_203: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_56, [4, 512, 768]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_204: "f32[2048, 768]" = torch.ops.aten.view.default(add_84, [2048, 768])
    permute_102: "f32[768, 768]" = torch.ops.aten.permute.default(primals_12, [1, 0])
    addmm_57: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_13, view_204, permute_102)
    view_205: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_57, [4, 512, 768]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_206: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_201, [4, 512, 12, 64]);  view_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_103: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_207: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_203, [4, 512, 12, 64]);  view_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_104: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_207, [0, 2, 1, 3]);  view_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_208: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_205, [4, 512, 12, 64]);  view_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_105: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_106: "f32[4, 12, 64, 512]" = torch.ops.aten.permute.default(permute_104, [0, 1, 3, 2]);  permute_104 = None
    expand_37: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_103, [4, 12, 512, 64]);  permute_103 = None
    clone_55: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
    view_209: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_55, [48, 512, 64]);  clone_55 = None
    expand_38: "f32[4, 12, 64, 512]" = torch.ops.aten.expand.default(permute_106, [4, 12, 64, 512]);  permute_106 = None
    clone_56: "f32[4, 12, 64, 512]" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
    view_210: "f32[48, 64, 512]" = torch.ops.aten.view.default(clone_56, [48, 64, 512]);  clone_56 = None
    bmm_18: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_209, view_210)
    view_211: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_18, [4, 12, 512, 512]);  bmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_18: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_211, 8.0);  view_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:336, code: attention_scores = attention_scores + attention_mask
    add_85: "f32[4, 12, 512, 512]" = torch.ops.aten.add.Tensor(div_18, mul);  div_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_9: "f32[4, 12, 512, 1]" = torch.ops.aten.amax.default(add_85, [-1], True)
    sub_29: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_85, amax_9);  add_85 = amax_9 = None
    exp_9: "f32[4, 12, 512, 512]" = torch.ops.aten.exp.default(sub_29);  sub_29 = None
    sum_10: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_19: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    alias_18: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(div_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    clone_57: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_19);  div_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_39: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_57, [4, 12, 512, 512]);  clone_57 = None
    view_212: "f32[48, 512, 512]" = torch.ops.aten.view.default(expand_39, [48, 512, 512]);  expand_39 = None
    expand_40: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_105, [4, 12, 512, 64]);  permute_105 = None
    clone_58: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
    view_213: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_58, [48, 512, 64]);  clone_58 = None
    bmm_19: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_212, view_213)
    view_214: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_19, [4, 12, 512, 64]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    permute_107: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_214, [0, 2, 1, 3]);  view_214 = None
    clone_59: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_107, memory_format = torch.contiguous_format);  permute_107 = None
    view_215: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_59, [4, 512, 768]);  clone_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_216: "f32[2048, 768]" = torch.ops.aten.view.default(view_215, [2048, 768]);  view_215 = None
    permute_108: "f32[768, 768]" = torch.ops.aten.permute.default(primals_14, [1, 0])
    addmm_58: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_15, view_216, permute_108)
    view_217: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_58, [4, 512, 768]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:369, code: projected_context_layer_dropout = self.output_dropout(projected_context_layer)
    clone_60: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_217);  view_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_86: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_84, clone_60);  add_84 = clone_60 = None
    var_mean_19 = torch.ops.aten.var_mean.correction(add_86, [2], correction = 0, keepdim = True)
    getitem_38: "f32[4, 512, 1]" = var_mean_19[0]
    getitem_39: "f32[4, 512, 1]" = var_mean_19[1];  var_mean_19 = None
    add_87: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
    rsqrt_19: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
    sub_30: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_86, getitem_39)
    mul_75: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_19);  sub_30 = None
    mul_76: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_75, primals_16);  mul_75 = None
    add_88: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_76, primals_17);  mul_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_218: "f32[2048, 768]" = torch.ops.aten.view.default(add_88, [2048, 768])
    permute_109: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_18, [1, 0])
    addmm_59: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_19, view_218, permute_109)
    view_219: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_59, [4, 512, 3072]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_77: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_219, 0.5)
    pow_10: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_219, 3.0)
    mul_78: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
    add_89: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(view_219, mul_78);  mul_78 = None
    mul_79: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_89, 0.7978845608028654);  add_89 = None
    tanh_9: "f32[4, 512, 3072]" = torch.ops.aten.tanh.default(mul_79);  mul_79 = None
    alias_19: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(tanh_9)
    add_90: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
    mul_80: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_77, add_90)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_220: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_80, [2048, 3072]);  mul_80 = None
    permute_110: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_20, [1, 0])
    addmm_60: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_21, view_220, permute_110)
    view_221: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_60, [4, 512, 768]);  addmm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_91: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_221, add_88);  view_221 = add_88 = None
    var_mean_20 = torch.ops.aten.var_mean.correction(add_91, [2], correction = 0, keepdim = True)
    getitem_40: "f32[4, 512, 1]" = var_mean_20[0]
    getitem_41: "f32[4, 512, 1]" = var_mean_20[1];  var_mean_20 = None
    add_92: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
    rsqrt_20: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
    sub_31: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_91, getitem_41)
    mul_81: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_20);  sub_31 = None
    mul_82: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_81, primals_22);  mul_81 = None
    add_93: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_82, primals_23);  mul_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_222: "f32[2048, 768]" = torch.ops.aten.view.default(add_93, [2048, 768])
    permute_111: "f32[768, 768]" = torch.ops.aten.permute.default(primals_8, [1, 0])
    addmm_61: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_9, view_222, permute_111)
    view_223: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_61, [4, 512, 768]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_224: "f32[2048, 768]" = torch.ops.aten.view.default(add_93, [2048, 768])
    permute_112: "f32[768, 768]" = torch.ops.aten.permute.default(primals_10, [1, 0])
    addmm_62: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_11, view_224, permute_112)
    view_225: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_62, [4, 512, 768]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_226: "f32[2048, 768]" = torch.ops.aten.view.default(add_93, [2048, 768])
    permute_113: "f32[768, 768]" = torch.ops.aten.permute.default(primals_12, [1, 0])
    addmm_63: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_13, view_226, permute_113)
    view_227: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_63, [4, 512, 768]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_228: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_223, [4, 512, 12, 64]);  view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_114: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_229: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_225, [4, 512, 12, 64]);  view_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_115: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_229, [0, 2, 1, 3]);  view_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_230: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_227, [4, 512, 12, 64]);  view_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_116: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_230, [0, 2, 1, 3]);  view_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_117: "f32[4, 12, 64, 512]" = torch.ops.aten.permute.default(permute_115, [0, 1, 3, 2]);  permute_115 = None
    expand_41: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_114, [4, 12, 512, 64]);  permute_114 = None
    clone_61: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
    view_231: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_61, [48, 512, 64]);  clone_61 = None
    expand_42: "f32[4, 12, 64, 512]" = torch.ops.aten.expand.default(permute_117, [4, 12, 64, 512]);  permute_117 = None
    clone_62: "f32[4, 12, 64, 512]" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
    view_232: "f32[48, 64, 512]" = torch.ops.aten.view.default(clone_62, [48, 64, 512]);  clone_62 = None
    bmm_20: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_231, view_232)
    view_233: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_20, [4, 12, 512, 512]);  bmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_20: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_233, 8.0);  view_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:336, code: attention_scores = attention_scores + attention_mask
    add_94: "f32[4, 12, 512, 512]" = torch.ops.aten.add.Tensor(div_20, mul);  div_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_10: "f32[4, 12, 512, 1]" = torch.ops.aten.amax.default(add_94, [-1], True)
    sub_32: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_94, amax_10);  add_94 = amax_10 = None
    exp_10: "f32[4, 12, 512, 512]" = torch.ops.aten.exp.default(sub_32);  sub_32 = None
    sum_11: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_21: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    alias_20: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(div_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    clone_63: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_21);  div_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_43: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_63, [4, 12, 512, 512]);  clone_63 = None
    view_234: "f32[48, 512, 512]" = torch.ops.aten.view.default(expand_43, [48, 512, 512]);  expand_43 = None
    expand_44: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_116, [4, 12, 512, 64]);  permute_116 = None
    clone_64: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
    view_235: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_64, [48, 512, 64]);  clone_64 = None
    bmm_21: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_234, view_235)
    view_236: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_21, [4, 12, 512, 64]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    permute_118: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_236, [0, 2, 1, 3]);  view_236 = None
    clone_65: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_118, memory_format = torch.contiguous_format);  permute_118 = None
    view_237: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_65, [4, 512, 768]);  clone_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_238: "f32[2048, 768]" = torch.ops.aten.view.default(view_237, [2048, 768]);  view_237 = None
    permute_119: "f32[768, 768]" = torch.ops.aten.permute.default(primals_14, [1, 0])
    addmm_64: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_15, view_238, permute_119)
    view_239: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_64, [4, 512, 768]);  addmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:369, code: projected_context_layer_dropout = self.output_dropout(projected_context_layer)
    clone_66: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_239);  view_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_95: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_93, clone_66);  add_93 = clone_66 = None
    var_mean_21 = torch.ops.aten.var_mean.correction(add_95, [2], correction = 0, keepdim = True)
    getitem_42: "f32[4, 512, 1]" = var_mean_21[0]
    getitem_43: "f32[4, 512, 1]" = var_mean_21[1];  var_mean_21 = None
    add_96: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
    rsqrt_21: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
    sub_33: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_95, getitem_43)
    mul_83: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_21);  sub_33 = None
    mul_84: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_83, primals_16);  mul_83 = None
    add_97: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_84, primals_17);  mul_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_240: "f32[2048, 768]" = torch.ops.aten.view.default(add_97, [2048, 768])
    permute_120: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_18, [1, 0])
    addmm_65: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_19, view_240, permute_120)
    view_241: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_65, [4, 512, 3072]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_85: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_241, 0.5)
    pow_11: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_241, 3.0)
    mul_86: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
    add_98: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(view_241, mul_86);  mul_86 = None
    mul_87: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_98, 0.7978845608028654);  add_98 = None
    tanh_10: "f32[4, 512, 3072]" = torch.ops.aten.tanh.default(mul_87);  mul_87 = None
    alias_21: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(tanh_10)
    add_99: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
    mul_88: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_85, add_99)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_242: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_88, [2048, 3072]);  mul_88 = None
    permute_121: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_20, [1, 0])
    addmm_66: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_21, view_242, permute_121)
    view_243: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_66, [4, 512, 768]);  addmm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_100: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_243, add_97);  view_243 = add_97 = None
    var_mean_22 = torch.ops.aten.var_mean.correction(add_100, [2], correction = 0, keepdim = True)
    getitem_44: "f32[4, 512, 1]" = var_mean_22[0]
    getitem_45: "f32[4, 512, 1]" = var_mean_22[1];  var_mean_22 = None
    add_101: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
    rsqrt_22: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
    sub_34: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_100, getitem_45)
    mul_89: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_22);  sub_34 = None
    mul_90: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_89, primals_22);  mul_89 = None
    add_102: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_90, primals_23);  mul_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_244: "f32[2048, 768]" = torch.ops.aten.view.default(add_102, [2048, 768])
    permute_122: "f32[768, 768]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
    addmm_67: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_9, view_244, permute_122);  primals_9 = None
    view_245: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_67, [4, 512, 768]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_246: "f32[2048, 768]" = torch.ops.aten.view.default(add_102, [2048, 768])
    permute_123: "f32[768, 768]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
    addmm_68: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_11, view_246, permute_123);  primals_11 = None
    view_247: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_68, [4, 512, 768]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_248: "f32[2048, 768]" = torch.ops.aten.view.default(add_102, [2048, 768])
    permute_124: "f32[768, 768]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
    addmm_69: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_13, view_248, permute_124);  primals_13 = None
    view_249: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_69, [4, 512, 768]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_250: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_245, [4, 512, 12, 64]);  view_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_125: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_251: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_247, [4, 512, 12, 64]);  view_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_126: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_251, [0, 2, 1, 3]);  view_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_252: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_249, [4, 512, 12, 64]);  view_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_127: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_128: "f32[4, 12, 64, 512]" = torch.ops.aten.permute.default(permute_126, [0, 1, 3, 2]);  permute_126 = None
    expand_45: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_125, [4, 12, 512, 64]);  permute_125 = None
    clone_67: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
    view_253: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_67, [48, 512, 64]);  clone_67 = None
    expand_46: "f32[4, 12, 64, 512]" = torch.ops.aten.expand.default(permute_128, [4, 12, 64, 512]);  permute_128 = None
    clone_68: "f32[4, 12, 64, 512]" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
    view_254: "f32[48, 64, 512]" = torch.ops.aten.view.default(clone_68, [48, 64, 512]);  clone_68 = None
    bmm_22: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_253, view_254)
    view_255: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_22, [4, 12, 512, 512]);  bmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_22: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_255, 8.0);  view_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:336, code: attention_scores = attention_scores + attention_mask
    add_103: "f32[4, 12, 512, 512]" = torch.ops.aten.add.Tensor(div_22, mul);  div_22 = mul = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_11: "f32[4, 12, 512, 1]" = torch.ops.aten.amax.default(add_103, [-1], True)
    sub_35: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_103, amax_11);  add_103 = amax_11 = None
    exp_11: "f32[4, 12, 512, 512]" = torch.ops.aten.exp.default(sub_35);  sub_35 = None
    sum_12: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_23: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    alias_22: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(div_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    clone_69: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_23);  div_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_47: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_69, [4, 12, 512, 512]);  clone_69 = None
    view_256: "f32[48, 512, 512]" = torch.ops.aten.view.default(expand_47, [48, 512, 512]);  expand_47 = None
    expand_48: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_127, [4, 12, 512, 64]);  permute_127 = None
    clone_70: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
    view_257: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_70, [48, 512, 64]);  clone_70 = None
    bmm_23: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_256, view_257)
    view_258: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_23, [4, 12, 512, 64]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    permute_129: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_258, [0, 2, 1, 3]);  view_258 = None
    clone_71: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None
    view_259: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_71, [4, 512, 768]);  clone_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_260: "f32[2048, 768]" = torch.ops.aten.view.default(view_259, [2048, 768]);  view_259 = None
    permute_130: "f32[768, 768]" = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None
    addmm_70: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_15, view_260, permute_130);  primals_15 = None
    view_261: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_70, [4, 512, 768]);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:369, code: projected_context_layer_dropout = self.output_dropout(projected_context_layer)
    clone_72: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_261);  view_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_104: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_102, clone_72);  add_102 = clone_72 = None
    var_mean_23 = torch.ops.aten.var_mean.correction(add_104, [2], correction = 0, keepdim = True)
    getitem_46: "f32[4, 512, 1]" = var_mean_23[0]
    getitem_47: "f32[4, 512, 1]" = var_mean_23[1];  var_mean_23 = None
    add_105: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
    rsqrt_23: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
    sub_36: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_104, getitem_47)
    mul_91: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_23);  sub_36 = None
    mul_92: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_91, primals_16);  mul_91 = None
    add_106: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_92, primals_17);  mul_92 = primals_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_262: "f32[2048, 768]" = torch.ops.aten.view.default(add_106, [2048, 768])
    permute_131: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
    addmm_71: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_19, view_262, permute_131);  primals_19 = None
    view_263: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_71, [4, 512, 3072]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_93: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_263, 0.5)
    pow_12: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_263, 3.0)
    mul_94: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
    add_107: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(view_263, mul_94);  mul_94 = None
    mul_95: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_107, 0.7978845608028654);  add_107 = None
    tanh_11: "f32[4, 512, 3072]" = torch.ops.aten.tanh.default(mul_95);  mul_95 = None
    alias_23: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(tanh_11)
    add_108: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
    mul_96: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_93, add_108)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_264: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_96, [2048, 3072]);  mul_96 = None
    permute_132: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_20, [1, 0]);  primals_20 = None
    addmm_72: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_21, view_264, permute_132);  primals_21 = None
    view_265: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_72, [4, 512, 768]);  addmm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_109: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_265, add_106);  view_265 = add_106 = None
    var_mean_24 = torch.ops.aten.var_mean.correction(add_109, [2], correction = 0, keepdim = True)
    getitem_48: "f32[4, 512, 1]" = var_mean_24[0]
    getitem_49: "f32[4, 512, 1]" = var_mean_24[1];  var_mean_24 = None
    add_110: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
    rsqrt_24: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
    sub_37: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_109, getitem_49)
    mul_97: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_24);  sub_37 = None
    mul_98: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_97, primals_22);  mul_97 = None
    add_111: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_98, primals_23);  mul_98 = primals_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:880, code: hidden_states = self.dense(hidden_states)
    view_266: "f32[2048, 768]" = torch.ops.aten.view.default(add_111, [2048, 768]);  add_111 = None
    permute_133: "f32[768, 128]" = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
    addmm_73: "f32[2048, 128]" = torch.ops.aten.addmm.default(primals_25, view_266, permute_133);  primals_25 = None
    view_267: "f32[4, 512, 128]" = torch.ops.aten.view.default(addmm_73, [4, 512, 128]);  addmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_99: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(view_267, 0.5)
    pow_13: "f32[4, 512, 128]" = torch.ops.aten.pow.Tensor_Scalar(view_267, 3.0)
    mul_100: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(pow_13, 0.044715);  pow_13 = None
    add_112: "f32[4, 512, 128]" = torch.ops.aten.add.Tensor(view_267, mul_100);  mul_100 = None
    mul_101: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(add_112, 0.7978845608028654);  add_112 = None
    tanh_12: "f32[4, 512, 128]" = torch.ops.aten.tanh.default(mul_101);  mul_101 = None
    alias_24: "f32[4, 512, 128]" = torch.ops.aten.alias.default(tanh_12)
    add_113: "f32[4, 512, 128]" = torch.ops.aten.add.Tensor(tanh_12, 1.0);  tanh_12 = None
    mul_102: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(mul_99, add_113)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:882, code: hidden_states = self.LayerNorm(hidden_states)
    var_mean_25 = torch.ops.aten.var_mean.correction(mul_102, [2], correction = 0, keepdim = True)
    getitem_50: "f32[4, 512, 1]" = var_mean_25[0]
    getitem_51: "f32[4, 512, 1]" = var_mean_25[1];  var_mean_25 = None
    add_114: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
    rsqrt_25: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
    sub_38: "f32[4, 512, 128]" = torch.ops.aten.sub.Tensor(mul_102, getitem_51)
    mul_103: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_25);  sub_38 = None
    mul_104: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(mul_103, primals_26);  mul_103 = None
    add_115: "f32[4, 512, 128]" = torch.ops.aten.add.Tensor(mul_104, primals_27);  mul_104 = primals_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:883, code: hidden_states = self.decoder(hidden_states)
    view_268: "f32[2048, 128]" = torch.ops.aten.view.default(add_115, [2048, 128]);  add_115 = None
    permute_134: "f32[128, 30000]" = torch.ops.aten.permute.default(primals_28, [1, 0]);  primals_28 = None
    addmm_74: "f32[2048, 30000]" = torch.ops.aten.addmm.default(primals_29, view_268, permute_134);  primals_29 = None
    view_269: "f32[4, 512, 30000]" = torch.ops.aten.view.default(addmm_74, [4, 512, 30000]);  addmm_74 = None
    view_270: "f32[2048, 30000]" = torch.ops.aten.view.default(tangents_1, [2048, 30000]);  tangents_1 = None
    permute_135: "f32[30000, 128]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
    mm: "f32[2048, 128]" = torch.ops.aten.mm.default(view_270, permute_135);  permute_135 = None
    permute_136: "f32[30000, 2048]" = torch.ops.aten.permute.default(view_270, [1, 0])
    mm_1: "f32[30000, 128]" = torch.ops.aten.mm.default(permute_136, view_268);  permute_136 = view_268 = None
    permute_137: "f32[128, 30000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_13: "f32[1, 30000]" = torch.ops.aten.sum.dim_IntList(view_270, [0], True);  view_270 = None
    view_271: "f32[30000]" = torch.ops.aten.view.default(sum_13, [30000]);  sum_13 = None
    permute_138: "f32[30000, 128]" = torch.ops.aten.permute.default(permute_137, [1, 0]);  permute_137 = None
    view_272: "f32[4, 512, 128]" = torch.ops.aten.view.default(mm, [4, 512, 128]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:882, code: hidden_states = self.LayerNorm(hidden_states)
    sub_39: "f32[4, 512, 128]" = torch.ops.aten.sub.Tensor(mul_102, getitem_51);  mul_102 = getitem_51 = None
    mul_105: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_25);  sub_39 = None
    mul_106: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(view_272, primals_26);  primals_26 = None
    mul_107: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(mul_106, 128)
    sum_14: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_106, [2], True)
    mul_108: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(mul_106, mul_105);  mul_106 = None
    sum_15: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_108, [2], True);  mul_108 = None
    mul_109: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(mul_105, sum_15);  sum_15 = None
    sub_40: "f32[4, 512, 128]" = torch.ops.aten.sub.Tensor(mul_107, sum_14);  mul_107 = sum_14 = None
    sub_41: "f32[4, 512, 128]" = torch.ops.aten.sub.Tensor(sub_40, mul_109);  sub_40 = mul_109 = None
    div_24: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 128);  rsqrt_25 = None
    mul_110: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(div_24, sub_41);  div_24 = sub_41 = None
    mul_111: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(view_272, mul_105);  mul_105 = None
    sum_16: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_111, [0, 1]);  mul_111 = None
    sum_17: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_272, [0, 1]);  view_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_112: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(mul_110, mul_99);  mul_99 = None
    mul_113: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(mul_110, add_113);  mul_110 = add_113 = None
    alias_25: "f32[4, 512, 128]" = torch.ops.aten.alias.default(alias_24);  alias_24 = None
    mul_114: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(alias_25, alias_25);  alias_25 = None
    sub_42: "f32[4, 512, 128]" = torch.ops.aten.sub.Tensor(1, mul_114);  mul_114 = None
    mul_115: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(mul_112, sub_42);  mul_112 = sub_42 = None
    mul_116: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(mul_115, 0.7978845608028654);  mul_115 = None
    mul_117: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(mul_116, 0.044715)
    pow_14: "f32[4, 512, 128]" = torch.ops.aten.pow.Tensor_Scalar(view_267, 2.0);  view_267 = None
    mul_118: "f32[4, 512, 128]" = torch.ops.aten.mul.Scalar(pow_14, 3.0);  pow_14 = None
    mul_119: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(mul_117, mul_118);  mul_117 = mul_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_116: "f32[4, 512, 128]" = torch.ops.aten.add.Tensor(mul_116, mul_119);  mul_116 = mul_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_120: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(mul_113, 0.5);  mul_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_117: "f32[4, 512, 128]" = torch.ops.aten.add.Tensor(add_116, mul_120);  add_116 = mul_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:880, code: hidden_states = self.dense(hidden_states)
    view_273: "f32[2048, 128]" = torch.ops.aten.view.default(add_117, [2048, 128]);  add_117 = None
    permute_139: "f32[128, 768]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
    mm_2: "f32[2048, 768]" = torch.ops.aten.mm.default(view_273, permute_139);  permute_139 = None
    permute_140: "f32[128, 2048]" = torch.ops.aten.permute.default(view_273, [1, 0])
    mm_3: "f32[128, 768]" = torch.ops.aten.mm.default(permute_140, view_266);  permute_140 = view_266 = None
    permute_141: "f32[768, 128]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_18: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_273, [0], True);  view_273 = None
    view_274: "f32[128]" = torch.ops.aten.view.default(sum_18, [128]);  sum_18 = None
    permute_142: "f32[128, 768]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
    view_275: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_2, [4, 512, 768]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    sub_43: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_109, getitem_49);  add_109 = getitem_49 = None
    mul_121: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_24);  sub_43 = None
    mul_122: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_275, primals_22)
    mul_123: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_122, 768)
    sum_19: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_122, [2], True)
    mul_124: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_122, mul_121);  mul_122 = None
    sum_20: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_124, [2], True);  mul_124 = None
    mul_125: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_121, sum_20);  sum_20 = None
    sub_44: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_123, sum_19);  mul_123 = sum_19 = None
    sub_45: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_44, mul_125);  sub_44 = mul_125 = None
    div_25: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None
    mul_126: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_25, sub_45);  div_25 = sub_45 = None
    mul_127: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_275, mul_121);  mul_121 = None
    sum_21: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_127, [0, 1]);  mul_127 = None
    sum_22: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_275, [0, 1]);  view_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_276: "f32[2048, 768]" = torch.ops.aten.view.default(mul_126, [2048, 768])
    permute_143: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
    mm_4: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_276, permute_143);  permute_143 = None
    permute_144: "f32[768, 2048]" = torch.ops.aten.permute.default(view_276, [1, 0])
    mm_5: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_144, view_264);  permute_144 = view_264 = None
    permute_145: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_23: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_276, [0], True);  view_276 = None
    view_277: "f32[768]" = torch.ops.aten.view.default(sum_23, [768]);  sum_23 = None
    permute_146: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
    view_278: "f32[4, 512, 3072]" = torch.ops.aten.view.default(mm_4, [4, 512, 3072]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_128: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_278, mul_93);  mul_93 = None
    mul_129: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_278, add_108);  view_278 = add_108 = None
    alias_26: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(alias_23);  alias_23 = None
    mul_130: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(alias_26, alias_26);  alias_26 = None
    sub_46: "f32[4, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_130);  mul_130 = None
    mul_131: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_128, sub_46);  mul_128 = sub_46 = None
    mul_132: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_131, 0.7978845608028654);  mul_131 = None
    mul_133: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_132, 0.044715)
    pow_15: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_263, 2.0);  view_263 = None
    mul_134: "f32[4, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_15, 3.0);  pow_15 = None
    mul_135: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_133, mul_134);  mul_133 = mul_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_118: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_132, mul_135);  mul_132 = mul_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_136: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_129, 0.5);  mul_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_119: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(add_118, mul_136);  add_118 = mul_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_279: "f32[2048, 3072]" = torch.ops.aten.view.default(add_119, [2048, 3072]);  add_119 = None
    permute_147: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
    mm_6: "f32[2048, 768]" = torch.ops.aten.mm.default(view_279, permute_147);  permute_147 = None
    permute_148: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_279, [1, 0])
    mm_7: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_148, view_262);  permute_148 = view_262 = None
    permute_149: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_24: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_279, [0], True);  view_279 = None
    view_280: "f32[3072]" = torch.ops.aten.view.default(sum_24, [3072]);  sum_24 = None
    permute_150: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_149, [1, 0]);  permute_149 = None
    view_281: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_6, [4, 512, 768]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_120: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_126, view_281);  mul_126 = view_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    sub_47: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_104, getitem_47);  add_104 = getitem_47 = None
    mul_137: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_23);  sub_47 = None
    mul_138: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_120, primals_16)
    mul_139: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_138, 768)
    sum_25: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_138, [2], True)
    mul_140: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_138, mul_137);  mul_138 = None
    sum_26: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_140, [2], True);  mul_140 = None
    mul_141: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_137, sum_26);  sum_26 = None
    sub_48: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_139, sum_25);  mul_139 = sum_25 = None
    sub_49: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_48, mul_141);  sub_48 = mul_141 = None
    div_26: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
    mul_142: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_26, sub_49);  div_26 = sub_49 = None
    mul_143: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_120, mul_137);  mul_137 = None
    sum_27: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_143, [0, 1]);  mul_143 = None
    sum_28: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_120, [0, 1]);  add_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_282: "f32[2048, 768]" = torch.ops.aten.view.default(mul_142, [2048, 768])
    permute_151: "f32[768, 768]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
    mm_8: "f32[2048, 768]" = torch.ops.aten.mm.default(view_282, permute_151);  permute_151 = None
    permute_152: "f32[768, 2048]" = torch.ops.aten.permute.default(view_282, [1, 0])
    mm_9: "f32[768, 768]" = torch.ops.aten.mm.default(permute_152, view_260);  permute_152 = view_260 = None
    permute_153: "f32[768, 768]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_29: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_282, [0], True);  view_282 = None
    view_283: "f32[768]" = torch.ops.aten.view.default(sum_29, [768]);  sum_29 = None
    permute_154: "f32[768, 768]" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
    view_284: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_8, [4, 512, 768]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    view_285: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_284, [4, 512, 12, 64]);  view_284 = None
    permute_155: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_285, [0, 2, 1, 3]);  view_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    clone_73: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_155, memory_format = torch.contiguous_format);  permute_155 = None
    view_286: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_73, [48, 512, 64]);  clone_73 = None
    permute_156: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_256, [0, 2, 1]);  view_256 = None
    bmm_24: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_156, view_286);  permute_156 = None
    permute_157: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_257, [0, 2, 1]);  view_257 = None
    bmm_25: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_286, permute_157);  view_286 = permute_157 = None
    view_287: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_24, [4, 12, 512, 64]);  bmm_24 = None
    view_288: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_25, [4, 12, 512, 512]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_27: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(alias_22);  alias_22 = None
    mul_144: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_288, alias_27);  view_288 = None
    sum_30: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_144, [-1], True)
    mul_145: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_27, sum_30);  alias_27 = sum_30 = None
    sub_50: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_27: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_50, 8.0);  sub_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_289: "f32[48, 512, 512]" = torch.ops.aten.view.default(div_27, [48, 512, 512]);  div_27 = None
    permute_158: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_253, [0, 2, 1]);  view_253 = None
    bmm_26: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_158, view_289);  permute_158 = None
    permute_159: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None
    bmm_27: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_289, permute_159);  view_289 = permute_159 = None
    view_290: "f32[4, 12, 64, 512]" = torch.ops.aten.view.default(bmm_26, [4, 12, 64, 512]);  bmm_26 = None
    view_291: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_27, [4, 12, 512, 64]);  bmm_27 = None
    permute_160: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_290, [0, 1, 3, 2]);  view_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_161: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_287, [0, 2, 1, 3]);  view_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_74: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
    view_292: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_74, [4, 512, 768]);  clone_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_162: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(permute_160, [0, 2, 1, 3]);  permute_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_293: "f32[4, 512, 768]" = torch.ops.aten.view.default(permute_162, [4, 512, 768]);  permute_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_163: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_291, [0, 2, 1, 3]);  view_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_75: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_163, memory_format = torch.contiguous_format);  permute_163 = None
    view_294: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_75, [4, 512, 768]);  clone_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_295: "f32[2048, 768]" = torch.ops.aten.view.default(view_292, [2048, 768]);  view_292 = None
    permute_164: "f32[768, 768]" = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
    mm_10: "f32[2048, 768]" = torch.ops.aten.mm.default(view_295, permute_164);  permute_164 = None
    permute_165: "f32[768, 2048]" = torch.ops.aten.permute.default(view_295, [1, 0])
    mm_11: "f32[768, 768]" = torch.ops.aten.mm.default(permute_165, view_248);  permute_165 = view_248 = None
    permute_166: "f32[768, 768]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_31: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_295, [0], True);  view_295 = None
    view_296: "f32[768]" = torch.ops.aten.view.default(sum_31, [768]);  sum_31 = None
    permute_167: "f32[768, 768]" = torch.ops.aten.permute.default(permute_166, [1, 0]);  permute_166 = None
    view_297: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_10, [4, 512, 768]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_121: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_142, view_297);  mul_142 = view_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    clone_76: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_293, memory_format = torch.contiguous_format);  view_293 = None
    view_298: "f32[2048, 768]" = torch.ops.aten.view.default(clone_76, [2048, 768]);  clone_76 = None
    permute_168: "f32[768, 768]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
    mm_12: "f32[2048, 768]" = torch.ops.aten.mm.default(view_298, permute_168);  permute_168 = None
    permute_169: "f32[768, 2048]" = torch.ops.aten.permute.default(view_298, [1, 0])
    mm_13: "f32[768, 768]" = torch.ops.aten.mm.default(permute_169, view_246);  permute_169 = view_246 = None
    permute_170: "f32[768, 768]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_32: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_298, [0], True);  view_298 = None
    view_299: "f32[768]" = torch.ops.aten.view.default(sum_32, [768]);  sum_32 = None
    permute_171: "f32[768, 768]" = torch.ops.aten.permute.default(permute_170, [1, 0]);  permute_170 = None
    view_300: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_12, [4, 512, 768]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_122: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_121, view_300);  add_121 = view_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_301: "f32[2048, 768]" = torch.ops.aten.view.default(view_294, [2048, 768]);  view_294 = None
    permute_172: "f32[768, 768]" = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
    mm_14: "f32[2048, 768]" = torch.ops.aten.mm.default(view_301, permute_172);  permute_172 = None
    permute_173: "f32[768, 2048]" = torch.ops.aten.permute.default(view_301, [1, 0])
    mm_15: "f32[768, 768]" = torch.ops.aten.mm.default(permute_173, view_244);  permute_173 = view_244 = None
    permute_174: "f32[768, 768]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_33: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_301, [0], True);  view_301 = None
    view_302: "f32[768]" = torch.ops.aten.view.default(sum_33, [768]);  sum_33 = None
    permute_175: "f32[768, 768]" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
    view_303: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_14, [4, 512, 768]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_123: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_122, view_303);  add_122 = view_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    sub_51: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_100, getitem_45);  add_100 = getitem_45 = None
    mul_146: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_22);  sub_51 = None
    mul_147: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_123, primals_22)
    mul_148: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_147, 768)
    sum_34: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_147, [2], True)
    mul_149: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_147, mul_146);  mul_147 = None
    sum_35: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_149, [2], True);  mul_149 = None
    mul_150: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_146, sum_35);  sum_35 = None
    sub_52: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_148, sum_34);  mul_148 = sum_34 = None
    sub_53: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_52, mul_150);  sub_52 = mul_150 = None
    div_28: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
    mul_151: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_28, sub_53);  div_28 = sub_53 = None
    mul_152: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_123, mul_146);  mul_146 = None
    sum_36: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_152, [0, 1]);  mul_152 = None
    sum_37: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_123, [0, 1]);  add_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_124: "f32[768]" = torch.ops.aten.add.Tensor(sum_21, sum_36);  sum_21 = sum_36 = None
    add_125: "f32[768]" = torch.ops.aten.add.Tensor(sum_22, sum_37);  sum_22 = sum_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_304: "f32[2048, 768]" = torch.ops.aten.view.default(mul_151, [2048, 768])
    permute_176: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
    mm_16: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_304, permute_176);  permute_176 = None
    permute_177: "f32[768, 2048]" = torch.ops.aten.permute.default(view_304, [1, 0])
    mm_17: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_177, view_242);  permute_177 = view_242 = None
    permute_178: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_38: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_304, [0], True);  view_304 = None
    view_305: "f32[768]" = torch.ops.aten.view.default(sum_38, [768]);  sum_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_126: "f32[768]" = torch.ops.aten.add.Tensor(view_277, view_305);  view_277 = view_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    permute_179: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_127: "f32[768, 3072]" = torch.ops.aten.add.Tensor(permute_146, permute_179);  permute_146 = permute_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_306: "f32[4, 512, 3072]" = torch.ops.aten.view.default(mm_16, [4, 512, 3072]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_153: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_306, mul_85);  mul_85 = None
    mul_154: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_306, add_99);  view_306 = add_99 = None
    alias_28: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(alias_21);  alias_21 = None
    mul_155: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(alias_28, alias_28);  alias_28 = None
    sub_54: "f32[4, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_155);  mul_155 = None
    mul_156: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_153, sub_54);  mul_153 = sub_54 = None
    mul_157: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_156, 0.7978845608028654);  mul_156 = None
    mul_158: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_157, 0.044715)
    pow_16: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_241, 2.0);  view_241 = None
    mul_159: "f32[4, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_16, 3.0);  pow_16 = None
    mul_160: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_158, mul_159);  mul_158 = mul_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_128: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_157, mul_160);  mul_157 = mul_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_161: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_154, 0.5);  mul_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_129: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(add_128, mul_161);  add_128 = mul_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_307: "f32[2048, 3072]" = torch.ops.aten.view.default(add_129, [2048, 3072]);  add_129 = None
    permute_180: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
    mm_18: "f32[2048, 768]" = torch.ops.aten.mm.default(view_307, permute_180);  permute_180 = None
    permute_181: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_307, [1, 0])
    mm_19: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_181, view_240);  permute_181 = view_240 = None
    permute_182: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_39: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_307, [0], True);  view_307 = None
    view_308: "f32[3072]" = torch.ops.aten.view.default(sum_39, [3072]);  sum_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_130: "f32[3072]" = torch.ops.aten.add.Tensor(view_280, view_308);  view_280 = view_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    permute_183: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_182, [1, 0]);  permute_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_131: "f32[3072, 768]" = torch.ops.aten.add.Tensor(permute_150, permute_183);  permute_150 = permute_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_309: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_18, [4, 512, 768]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_132: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_151, view_309);  mul_151 = view_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    sub_55: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_95, getitem_43);  add_95 = getitem_43 = None
    mul_162: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_21);  sub_55 = None
    mul_163: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_132, primals_16)
    mul_164: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_163, 768)
    sum_40: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_163, [2], True)
    mul_165: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_163, mul_162);  mul_163 = None
    sum_41: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_165, [2], True);  mul_165 = None
    mul_166: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_162, sum_41);  sum_41 = None
    sub_56: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_164, sum_40);  mul_164 = sum_40 = None
    sub_57: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_56, mul_166);  sub_56 = mul_166 = None
    div_29: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
    mul_167: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_29, sub_57);  div_29 = sub_57 = None
    mul_168: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_132, mul_162);  mul_162 = None
    sum_42: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_168, [0, 1]);  mul_168 = None
    sum_43: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_132, [0, 1]);  add_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_133: "f32[768]" = torch.ops.aten.add.Tensor(sum_27, sum_42);  sum_27 = sum_42 = None
    add_134: "f32[768]" = torch.ops.aten.add.Tensor(sum_28, sum_43);  sum_28 = sum_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_310: "f32[2048, 768]" = torch.ops.aten.view.default(mul_167, [2048, 768])
    permute_184: "f32[768, 768]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
    mm_20: "f32[2048, 768]" = torch.ops.aten.mm.default(view_310, permute_184);  permute_184 = None
    permute_185: "f32[768, 2048]" = torch.ops.aten.permute.default(view_310, [1, 0])
    mm_21: "f32[768, 768]" = torch.ops.aten.mm.default(permute_185, view_238);  permute_185 = view_238 = None
    permute_186: "f32[768, 768]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_44: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_310, [0], True);  view_310 = None
    view_311: "f32[768]" = torch.ops.aten.view.default(sum_44, [768]);  sum_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_135: "f32[768]" = torch.ops.aten.add.Tensor(view_283, view_311);  view_283 = view_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    permute_187: "f32[768, 768]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_136: "f32[768, 768]" = torch.ops.aten.add.Tensor(permute_154, permute_187);  permute_154 = permute_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_312: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_20, [4, 512, 768]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    view_313: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_312, [4, 512, 12, 64]);  view_312 = None
    permute_188: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_313, [0, 2, 1, 3]);  view_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    clone_77: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_188, memory_format = torch.contiguous_format);  permute_188 = None
    view_314: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_77, [48, 512, 64]);  clone_77 = None
    permute_189: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_234, [0, 2, 1]);  view_234 = None
    bmm_28: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_189, view_314);  permute_189 = None
    permute_190: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_235, [0, 2, 1]);  view_235 = None
    bmm_29: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_314, permute_190);  view_314 = permute_190 = None
    view_315: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_28, [4, 12, 512, 64]);  bmm_28 = None
    view_316: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_29, [4, 12, 512, 512]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_29: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(alias_20);  alias_20 = None
    mul_169: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_316, alias_29);  view_316 = None
    sum_45: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_169, [-1], True)
    mul_170: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_29, sum_45);  alias_29 = sum_45 = None
    sub_58: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_169, mul_170);  mul_169 = mul_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_30: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_58, 8.0);  sub_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_317: "f32[48, 512, 512]" = torch.ops.aten.view.default(div_30, [48, 512, 512]);  div_30 = None
    permute_191: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_231, [0, 2, 1]);  view_231 = None
    bmm_30: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_191, view_317);  permute_191 = None
    permute_192: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_232, [0, 2, 1]);  view_232 = None
    bmm_31: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_317, permute_192);  view_317 = permute_192 = None
    view_318: "f32[4, 12, 64, 512]" = torch.ops.aten.view.default(bmm_30, [4, 12, 64, 512]);  bmm_30 = None
    view_319: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_31, [4, 12, 512, 64]);  bmm_31 = None
    permute_193: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_318, [0, 1, 3, 2]);  view_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_194: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_315, [0, 2, 1, 3]);  view_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_78: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
    view_320: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_78, [4, 512, 768]);  clone_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_195: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(permute_193, [0, 2, 1, 3]);  permute_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_321: "f32[4, 512, 768]" = torch.ops.aten.view.default(permute_195, [4, 512, 768]);  permute_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_196: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_319, [0, 2, 1, 3]);  view_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_79: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_196, memory_format = torch.contiguous_format);  permute_196 = None
    view_322: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_79, [4, 512, 768]);  clone_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_323: "f32[2048, 768]" = torch.ops.aten.view.default(view_320, [2048, 768]);  view_320 = None
    permute_197: "f32[768, 768]" = torch.ops.aten.permute.default(permute_113, [1, 0]);  permute_113 = None
    mm_22: "f32[2048, 768]" = torch.ops.aten.mm.default(view_323, permute_197);  permute_197 = None
    permute_198: "f32[768, 2048]" = torch.ops.aten.permute.default(view_323, [1, 0])
    mm_23: "f32[768, 768]" = torch.ops.aten.mm.default(permute_198, view_226);  permute_198 = view_226 = None
    permute_199: "f32[768, 768]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_46: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_323, [0], True);  view_323 = None
    view_324: "f32[768]" = torch.ops.aten.view.default(sum_46, [768]);  sum_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_137: "f32[768]" = torch.ops.aten.add.Tensor(view_296, view_324);  view_296 = view_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    permute_200: "f32[768, 768]" = torch.ops.aten.permute.default(permute_199, [1, 0]);  permute_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_138: "f32[768, 768]" = torch.ops.aten.add.Tensor(permute_167, permute_200);  permute_167 = permute_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_325: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_22, [4, 512, 768]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_139: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_167, view_325);  mul_167 = view_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    clone_80: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_321, memory_format = torch.contiguous_format);  view_321 = None
    view_326: "f32[2048, 768]" = torch.ops.aten.view.default(clone_80, [2048, 768]);  clone_80 = None
    permute_201: "f32[768, 768]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
    mm_24: "f32[2048, 768]" = torch.ops.aten.mm.default(view_326, permute_201);  permute_201 = None
    permute_202: "f32[768, 2048]" = torch.ops.aten.permute.default(view_326, [1, 0])
    mm_25: "f32[768, 768]" = torch.ops.aten.mm.default(permute_202, view_224);  permute_202 = view_224 = None
    permute_203: "f32[768, 768]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_47: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_326, [0], True);  view_326 = None
    view_327: "f32[768]" = torch.ops.aten.view.default(sum_47, [768]);  sum_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_140: "f32[768]" = torch.ops.aten.add.Tensor(view_299, view_327);  view_299 = view_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    permute_204: "f32[768, 768]" = torch.ops.aten.permute.default(permute_203, [1, 0]);  permute_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_141: "f32[768, 768]" = torch.ops.aten.add.Tensor(permute_171, permute_204);  permute_171 = permute_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_328: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_24, [4, 512, 768]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_142: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_139, view_328);  add_139 = view_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_329: "f32[2048, 768]" = torch.ops.aten.view.default(view_322, [2048, 768]);  view_322 = None
    permute_205: "f32[768, 768]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
    mm_26: "f32[2048, 768]" = torch.ops.aten.mm.default(view_329, permute_205);  permute_205 = None
    permute_206: "f32[768, 2048]" = torch.ops.aten.permute.default(view_329, [1, 0])
    mm_27: "f32[768, 768]" = torch.ops.aten.mm.default(permute_206, view_222);  permute_206 = view_222 = None
    permute_207: "f32[768, 768]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_48: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_329, [0], True);  view_329 = None
    view_330: "f32[768]" = torch.ops.aten.view.default(sum_48, [768]);  sum_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_143: "f32[768]" = torch.ops.aten.add.Tensor(view_302, view_330);  view_302 = view_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    permute_208: "f32[768, 768]" = torch.ops.aten.permute.default(permute_207, [1, 0]);  permute_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_144: "f32[768, 768]" = torch.ops.aten.add.Tensor(permute_175, permute_208);  permute_175 = permute_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_331: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_26, [4, 512, 768]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_145: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_142, view_331);  add_142 = view_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    sub_59: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_91, getitem_41);  add_91 = getitem_41 = None
    mul_171: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_20);  sub_59 = None
    mul_172: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_145, primals_22)
    mul_173: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_172, 768)
    sum_49: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_172, [2], True)
    mul_174: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_172, mul_171);  mul_172 = None
    sum_50: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_174, [2], True);  mul_174 = None
    mul_175: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_171, sum_50);  sum_50 = None
    sub_60: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_173, sum_49);  mul_173 = sum_49 = None
    sub_61: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_60, mul_175);  sub_60 = mul_175 = None
    div_31: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
    mul_176: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_31, sub_61);  div_31 = sub_61 = None
    mul_177: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_145, mul_171);  mul_171 = None
    sum_51: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_177, [0, 1]);  mul_177 = None
    sum_52: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_145, [0, 1]);  add_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_146: "f32[768]" = torch.ops.aten.add.Tensor(add_124, sum_51);  add_124 = sum_51 = None
    add_147: "f32[768]" = torch.ops.aten.add.Tensor(add_125, sum_52);  add_125 = sum_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_332: "f32[2048, 768]" = torch.ops.aten.view.default(mul_176, [2048, 768])
    permute_209: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
    mm_28: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_332, permute_209);  permute_209 = None
    permute_210: "f32[768, 2048]" = torch.ops.aten.permute.default(view_332, [1, 0])
    mm_29: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_210, view_220);  permute_210 = view_220 = None
    permute_211: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_53: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_332, [0], True);  view_332 = None
    view_333: "f32[768]" = torch.ops.aten.view.default(sum_53, [768]);  sum_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_148: "f32[768]" = torch.ops.aten.add.Tensor(add_126, view_333);  add_126 = view_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    permute_212: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_149: "f32[768, 3072]" = torch.ops.aten.add.Tensor(add_127, permute_212);  add_127 = permute_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_334: "f32[4, 512, 3072]" = torch.ops.aten.view.default(mm_28, [4, 512, 3072]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_178: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_334, mul_77);  mul_77 = None
    mul_179: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_334, add_90);  view_334 = add_90 = None
    alias_30: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(alias_19);  alias_19 = None
    mul_180: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(alias_30, alias_30);  alias_30 = None
    sub_62: "f32[4, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_180);  mul_180 = None
    mul_181: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_178, sub_62);  mul_178 = sub_62 = None
    mul_182: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_181, 0.7978845608028654);  mul_181 = None
    mul_183: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_182, 0.044715)
    pow_17: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_219, 2.0);  view_219 = None
    mul_184: "f32[4, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_17, 3.0);  pow_17 = None
    mul_185: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_183, mul_184);  mul_183 = mul_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_150: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_182, mul_185);  mul_182 = mul_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_186: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_179, 0.5);  mul_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_151: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(add_150, mul_186);  add_150 = mul_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_335: "f32[2048, 3072]" = torch.ops.aten.view.default(add_151, [2048, 3072]);  add_151 = None
    permute_213: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
    mm_30: "f32[2048, 768]" = torch.ops.aten.mm.default(view_335, permute_213);  permute_213 = None
    permute_214: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_335, [1, 0])
    mm_31: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_214, view_218);  permute_214 = view_218 = None
    permute_215: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_54: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_335, [0], True);  view_335 = None
    view_336: "f32[3072]" = torch.ops.aten.view.default(sum_54, [3072]);  sum_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_152: "f32[3072]" = torch.ops.aten.add.Tensor(add_130, view_336);  add_130 = view_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    permute_216: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_215, [1, 0]);  permute_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_153: "f32[3072, 768]" = torch.ops.aten.add.Tensor(add_131, permute_216);  add_131 = permute_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_337: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_30, [4, 512, 768]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_154: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_176, view_337);  mul_176 = view_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    sub_63: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_86, getitem_39);  add_86 = getitem_39 = None
    mul_187: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_19);  sub_63 = None
    mul_188: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_154, primals_16)
    mul_189: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_188, 768)
    sum_55: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_188, [2], True)
    mul_190: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_188, mul_187);  mul_188 = None
    sum_56: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_190, [2], True);  mul_190 = None
    mul_191: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_187, sum_56);  sum_56 = None
    sub_64: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_189, sum_55);  mul_189 = sum_55 = None
    sub_65: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_64, mul_191);  sub_64 = mul_191 = None
    div_32: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
    mul_192: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_32, sub_65);  div_32 = sub_65 = None
    mul_193: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_154, mul_187);  mul_187 = None
    sum_57: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_193, [0, 1]);  mul_193 = None
    sum_58: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_154, [0, 1]);  add_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_155: "f32[768]" = torch.ops.aten.add.Tensor(add_133, sum_57);  add_133 = sum_57 = None
    add_156: "f32[768]" = torch.ops.aten.add.Tensor(add_134, sum_58);  add_134 = sum_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_338: "f32[2048, 768]" = torch.ops.aten.view.default(mul_192, [2048, 768])
    permute_217: "f32[768, 768]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
    mm_32: "f32[2048, 768]" = torch.ops.aten.mm.default(view_338, permute_217);  permute_217 = None
    permute_218: "f32[768, 2048]" = torch.ops.aten.permute.default(view_338, [1, 0])
    mm_33: "f32[768, 768]" = torch.ops.aten.mm.default(permute_218, view_216);  permute_218 = view_216 = None
    permute_219: "f32[768, 768]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_59: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_338, [0], True);  view_338 = None
    view_339: "f32[768]" = torch.ops.aten.view.default(sum_59, [768]);  sum_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_157: "f32[768]" = torch.ops.aten.add.Tensor(add_135, view_339);  add_135 = view_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    permute_220: "f32[768, 768]" = torch.ops.aten.permute.default(permute_219, [1, 0]);  permute_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_158: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_136, permute_220);  add_136 = permute_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_340: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_32, [4, 512, 768]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    view_341: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_340, [4, 512, 12, 64]);  view_340 = None
    permute_221: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_341, [0, 2, 1, 3]);  view_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    clone_81: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_221, memory_format = torch.contiguous_format);  permute_221 = None
    view_342: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_81, [48, 512, 64]);  clone_81 = None
    permute_222: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_212, [0, 2, 1]);  view_212 = None
    bmm_32: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_222, view_342);  permute_222 = None
    permute_223: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_213, [0, 2, 1]);  view_213 = None
    bmm_33: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_342, permute_223);  view_342 = permute_223 = None
    view_343: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_32, [4, 12, 512, 64]);  bmm_32 = None
    view_344: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_33, [4, 12, 512, 512]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_31: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(alias_18);  alias_18 = None
    mul_194: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_344, alias_31);  view_344 = None
    sum_60: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_194, [-1], True)
    mul_195: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_31, sum_60);  alias_31 = sum_60 = None
    sub_66: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_194, mul_195);  mul_194 = mul_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_33: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_66, 8.0);  sub_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_345: "f32[48, 512, 512]" = torch.ops.aten.view.default(div_33, [48, 512, 512]);  div_33 = None
    permute_224: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_209, [0, 2, 1]);  view_209 = None
    bmm_34: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_224, view_345);  permute_224 = None
    permute_225: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None
    bmm_35: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_345, permute_225);  view_345 = permute_225 = None
    view_346: "f32[4, 12, 64, 512]" = torch.ops.aten.view.default(bmm_34, [4, 12, 64, 512]);  bmm_34 = None
    view_347: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_35, [4, 12, 512, 64]);  bmm_35 = None
    permute_226: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_346, [0, 1, 3, 2]);  view_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_227: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_343, [0, 2, 1, 3]);  view_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_82: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
    view_348: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_82, [4, 512, 768]);  clone_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_228: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(permute_226, [0, 2, 1, 3]);  permute_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_349: "f32[4, 512, 768]" = torch.ops.aten.view.default(permute_228, [4, 512, 768]);  permute_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_229: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_347, [0, 2, 1, 3]);  view_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_83: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_229, memory_format = torch.contiguous_format);  permute_229 = None
    view_350: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_83, [4, 512, 768]);  clone_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_351: "f32[2048, 768]" = torch.ops.aten.view.default(view_348, [2048, 768]);  view_348 = None
    permute_230: "f32[768, 768]" = torch.ops.aten.permute.default(permute_102, [1, 0]);  permute_102 = None
    mm_34: "f32[2048, 768]" = torch.ops.aten.mm.default(view_351, permute_230);  permute_230 = None
    permute_231: "f32[768, 2048]" = torch.ops.aten.permute.default(view_351, [1, 0])
    mm_35: "f32[768, 768]" = torch.ops.aten.mm.default(permute_231, view_204);  permute_231 = view_204 = None
    permute_232: "f32[768, 768]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_61: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_351, [0], True);  view_351 = None
    view_352: "f32[768]" = torch.ops.aten.view.default(sum_61, [768]);  sum_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_159: "f32[768]" = torch.ops.aten.add.Tensor(add_137, view_352);  add_137 = view_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    permute_233: "f32[768, 768]" = torch.ops.aten.permute.default(permute_232, [1, 0]);  permute_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_160: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_138, permute_233);  add_138 = permute_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_353: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_34, [4, 512, 768]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_161: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_192, view_353);  mul_192 = view_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    clone_84: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_349, memory_format = torch.contiguous_format);  view_349 = None
    view_354: "f32[2048, 768]" = torch.ops.aten.view.default(clone_84, [2048, 768]);  clone_84 = None
    permute_234: "f32[768, 768]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
    mm_36: "f32[2048, 768]" = torch.ops.aten.mm.default(view_354, permute_234);  permute_234 = None
    permute_235: "f32[768, 2048]" = torch.ops.aten.permute.default(view_354, [1, 0])
    mm_37: "f32[768, 768]" = torch.ops.aten.mm.default(permute_235, view_202);  permute_235 = view_202 = None
    permute_236: "f32[768, 768]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_62: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_354, [0], True);  view_354 = None
    view_355: "f32[768]" = torch.ops.aten.view.default(sum_62, [768]);  sum_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_162: "f32[768]" = torch.ops.aten.add.Tensor(add_140, view_355);  add_140 = view_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    permute_237: "f32[768, 768]" = torch.ops.aten.permute.default(permute_236, [1, 0]);  permute_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_163: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_141, permute_237);  add_141 = permute_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_356: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_36, [4, 512, 768]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_164: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_161, view_356);  add_161 = view_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_357: "f32[2048, 768]" = torch.ops.aten.view.default(view_350, [2048, 768]);  view_350 = None
    permute_238: "f32[768, 768]" = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
    mm_38: "f32[2048, 768]" = torch.ops.aten.mm.default(view_357, permute_238);  permute_238 = None
    permute_239: "f32[768, 2048]" = torch.ops.aten.permute.default(view_357, [1, 0])
    mm_39: "f32[768, 768]" = torch.ops.aten.mm.default(permute_239, view_200);  permute_239 = view_200 = None
    permute_240: "f32[768, 768]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_63: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_357, [0], True);  view_357 = None
    view_358: "f32[768]" = torch.ops.aten.view.default(sum_63, [768]);  sum_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_165: "f32[768]" = torch.ops.aten.add.Tensor(add_143, view_358);  add_143 = view_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    permute_241: "f32[768, 768]" = torch.ops.aten.permute.default(permute_240, [1, 0]);  permute_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_166: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_144, permute_241);  add_144 = permute_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_359: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_38, [4, 512, 768]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_167: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_164, view_359);  add_164 = view_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    sub_67: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_82, getitem_37);  add_82 = getitem_37 = None
    mul_196: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_18);  sub_67 = None
    mul_197: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_167, primals_22)
    mul_198: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_197, 768)
    sum_64: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_197, [2], True)
    mul_199: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_197, mul_196);  mul_197 = None
    sum_65: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_199, [2], True);  mul_199 = None
    mul_200: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_196, sum_65);  sum_65 = None
    sub_68: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_198, sum_64);  mul_198 = sum_64 = None
    sub_69: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_68, mul_200);  sub_68 = mul_200 = None
    div_34: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
    mul_201: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_34, sub_69);  div_34 = sub_69 = None
    mul_202: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_167, mul_196);  mul_196 = None
    sum_66: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_202, [0, 1]);  mul_202 = None
    sum_67: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_167, [0, 1]);  add_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_168: "f32[768]" = torch.ops.aten.add.Tensor(add_146, sum_66);  add_146 = sum_66 = None
    add_169: "f32[768]" = torch.ops.aten.add.Tensor(add_147, sum_67);  add_147 = sum_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_360: "f32[2048, 768]" = torch.ops.aten.view.default(mul_201, [2048, 768])
    permute_242: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
    mm_40: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_360, permute_242);  permute_242 = None
    permute_243: "f32[768, 2048]" = torch.ops.aten.permute.default(view_360, [1, 0])
    mm_41: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_243, view_198);  permute_243 = view_198 = None
    permute_244: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_68: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_360, [0], True);  view_360 = None
    view_361: "f32[768]" = torch.ops.aten.view.default(sum_68, [768]);  sum_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_170: "f32[768]" = torch.ops.aten.add.Tensor(add_148, view_361);  add_148 = view_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    permute_245: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_244, [1, 0]);  permute_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_171: "f32[768, 3072]" = torch.ops.aten.add.Tensor(add_149, permute_245);  add_149 = permute_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_362: "f32[4, 512, 3072]" = torch.ops.aten.view.default(mm_40, [4, 512, 3072]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_203: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_362, mul_69);  mul_69 = None
    mul_204: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_362, add_81);  view_362 = add_81 = None
    alias_32: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(alias_17);  alias_17 = None
    mul_205: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(alias_32, alias_32);  alias_32 = None
    sub_70: "f32[4, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_205);  mul_205 = None
    mul_206: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_203, sub_70);  mul_203 = sub_70 = None
    mul_207: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_206, 0.7978845608028654);  mul_206 = None
    mul_208: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_207, 0.044715)
    pow_18: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_197, 2.0);  view_197 = None
    mul_209: "f32[4, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_18, 3.0);  pow_18 = None
    mul_210: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_208, mul_209);  mul_208 = mul_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_172: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_207, mul_210);  mul_207 = mul_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_211: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_204, 0.5);  mul_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_173: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(add_172, mul_211);  add_172 = mul_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_363: "f32[2048, 3072]" = torch.ops.aten.view.default(add_173, [2048, 3072]);  add_173 = None
    permute_246: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
    mm_42: "f32[2048, 768]" = torch.ops.aten.mm.default(view_363, permute_246);  permute_246 = None
    permute_247: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_363, [1, 0])
    mm_43: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_247, view_196);  permute_247 = view_196 = None
    permute_248: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_69: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_363, [0], True);  view_363 = None
    view_364: "f32[3072]" = torch.ops.aten.view.default(sum_69, [3072]);  sum_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_174: "f32[3072]" = torch.ops.aten.add.Tensor(add_152, view_364);  add_152 = view_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    permute_249: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_248, [1, 0]);  permute_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_175: "f32[3072, 768]" = torch.ops.aten.add.Tensor(add_153, permute_249);  add_153 = permute_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_365: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_42, [4, 512, 768]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_176: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_201, view_365);  mul_201 = view_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    sub_71: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_77, getitem_35);  add_77 = getitem_35 = None
    mul_212: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_17);  sub_71 = None
    mul_213: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_176, primals_16)
    mul_214: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_213, 768)
    sum_70: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_213, [2], True)
    mul_215: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_213, mul_212);  mul_213 = None
    sum_71: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_215, [2], True);  mul_215 = None
    mul_216: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_212, sum_71);  sum_71 = None
    sub_72: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_214, sum_70);  mul_214 = sum_70 = None
    sub_73: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_72, mul_216);  sub_72 = mul_216 = None
    div_35: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
    mul_217: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_35, sub_73);  div_35 = sub_73 = None
    mul_218: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_176, mul_212);  mul_212 = None
    sum_72: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_218, [0, 1]);  mul_218 = None
    sum_73: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_176, [0, 1]);  add_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_177: "f32[768]" = torch.ops.aten.add.Tensor(add_155, sum_72);  add_155 = sum_72 = None
    add_178: "f32[768]" = torch.ops.aten.add.Tensor(add_156, sum_73);  add_156 = sum_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_366: "f32[2048, 768]" = torch.ops.aten.view.default(mul_217, [2048, 768])
    permute_250: "f32[768, 768]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
    mm_44: "f32[2048, 768]" = torch.ops.aten.mm.default(view_366, permute_250);  permute_250 = None
    permute_251: "f32[768, 2048]" = torch.ops.aten.permute.default(view_366, [1, 0])
    mm_45: "f32[768, 768]" = torch.ops.aten.mm.default(permute_251, view_194);  permute_251 = view_194 = None
    permute_252: "f32[768, 768]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_74: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_366, [0], True);  view_366 = None
    view_367: "f32[768]" = torch.ops.aten.view.default(sum_74, [768]);  sum_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_179: "f32[768]" = torch.ops.aten.add.Tensor(add_157, view_367);  add_157 = view_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    permute_253: "f32[768, 768]" = torch.ops.aten.permute.default(permute_252, [1, 0]);  permute_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_180: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_158, permute_253);  add_158 = permute_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_368: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_44, [4, 512, 768]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    view_369: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_368, [4, 512, 12, 64]);  view_368 = None
    permute_254: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_369, [0, 2, 1, 3]);  view_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    clone_85: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_254, memory_format = torch.contiguous_format);  permute_254 = None
    view_370: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_85, [48, 512, 64]);  clone_85 = None
    permute_255: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_190, [0, 2, 1]);  view_190 = None
    bmm_36: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_255, view_370);  permute_255 = None
    permute_256: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_191, [0, 2, 1]);  view_191 = None
    bmm_37: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_370, permute_256);  view_370 = permute_256 = None
    view_371: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_36, [4, 12, 512, 64]);  bmm_36 = None
    view_372: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_37, [4, 12, 512, 512]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_33: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(alias_16);  alias_16 = None
    mul_219: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_372, alias_33);  view_372 = None
    sum_75: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_219, [-1], True)
    mul_220: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_33, sum_75);  alias_33 = sum_75 = None
    sub_74: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_219, mul_220);  mul_219 = mul_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_36: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_74, 8.0);  sub_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_373: "f32[48, 512, 512]" = torch.ops.aten.view.default(div_36, [48, 512, 512]);  div_36 = None
    permute_257: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_187, [0, 2, 1]);  view_187 = None
    bmm_38: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_257, view_373);  permute_257 = None
    permute_258: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None
    bmm_39: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_373, permute_258);  view_373 = permute_258 = None
    view_374: "f32[4, 12, 64, 512]" = torch.ops.aten.view.default(bmm_38, [4, 12, 64, 512]);  bmm_38 = None
    view_375: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_39, [4, 12, 512, 64]);  bmm_39 = None
    permute_259: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_374, [0, 1, 3, 2]);  view_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_260: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_371, [0, 2, 1, 3]);  view_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_86: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None
    view_376: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_86, [4, 512, 768]);  clone_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_261: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(permute_259, [0, 2, 1, 3]);  permute_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_377: "f32[4, 512, 768]" = torch.ops.aten.view.default(permute_261, [4, 512, 768]);  permute_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_262: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_375, [0, 2, 1, 3]);  view_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_87: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_262, memory_format = torch.contiguous_format);  permute_262 = None
    view_378: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_87, [4, 512, 768]);  clone_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_379: "f32[2048, 768]" = torch.ops.aten.view.default(view_376, [2048, 768]);  view_376 = None
    permute_263: "f32[768, 768]" = torch.ops.aten.permute.default(permute_91, [1, 0]);  permute_91 = None
    mm_46: "f32[2048, 768]" = torch.ops.aten.mm.default(view_379, permute_263);  permute_263 = None
    permute_264: "f32[768, 2048]" = torch.ops.aten.permute.default(view_379, [1, 0])
    mm_47: "f32[768, 768]" = torch.ops.aten.mm.default(permute_264, view_182);  permute_264 = view_182 = None
    permute_265: "f32[768, 768]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_76: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_379, [0], True);  view_379 = None
    view_380: "f32[768]" = torch.ops.aten.view.default(sum_76, [768]);  sum_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_181: "f32[768]" = torch.ops.aten.add.Tensor(add_159, view_380);  add_159 = view_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    permute_266: "f32[768, 768]" = torch.ops.aten.permute.default(permute_265, [1, 0]);  permute_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_182: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_160, permute_266);  add_160 = permute_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_381: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_46, [4, 512, 768]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_183: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_217, view_381);  mul_217 = view_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    clone_88: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_377, memory_format = torch.contiguous_format);  view_377 = None
    view_382: "f32[2048, 768]" = torch.ops.aten.view.default(clone_88, [2048, 768]);  clone_88 = None
    permute_267: "f32[768, 768]" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
    mm_48: "f32[2048, 768]" = torch.ops.aten.mm.default(view_382, permute_267);  permute_267 = None
    permute_268: "f32[768, 2048]" = torch.ops.aten.permute.default(view_382, [1, 0])
    mm_49: "f32[768, 768]" = torch.ops.aten.mm.default(permute_268, view_180);  permute_268 = view_180 = None
    permute_269: "f32[768, 768]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_77: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_382, [0], True);  view_382 = None
    view_383: "f32[768]" = torch.ops.aten.view.default(sum_77, [768]);  sum_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_184: "f32[768]" = torch.ops.aten.add.Tensor(add_162, view_383);  add_162 = view_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    permute_270: "f32[768, 768]" = torch.ops.aten.permute.default(permute_269, [1, 0]);  permute_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_185: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_163, permute_270);  add_163 = permute_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_384: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_48, [4, 512, 768]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_186: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_183, view_384);  add_183 = view_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_385: "f32[2048, 768]" = torch.ops.aten.view.default(view_378, [2048, 768]);  view_378 = None
    permute_271: "f32[768, 768]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
    mm_50: "f32[2048, 768]" = torch.ops.aten.mm.default(view_385, permute_271);  permute_271 = None
    permute_272: "f32[768, 2048]" = torch.ops.aten.permute.default(view_385, [1, 0])
    mm_51: "f32[768, 768]" = torch.ops.aten.mm.default(permute_272, view_178);  permute_272 = view_178 = None
    permute_273: "f32[768, 768]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_78: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_385, [0], True);  view_385 = None
    view_386: "f32[768]" = torch.ops.aten.view.default(sum_78, [768]);  sum_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_187: "f32[768]" = torch.ops.aten.add.Tensor(add_165, view_386);  add_165 = view_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    permute_274: "f32[768, 768]" = torch.ops.aten.permute.default(permute_273, [1, 0]);  permute_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_188: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_166, permute_274);  add_166 = permute_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_387: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_50, [4, 512, 768]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_189: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_186, view_387);  add_186 = view_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    sub_75: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_73, getitem_33);  add_73 = getitem_33 = None
    mul_221: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_16);  sub_75 = None
    mul_222: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_189, primals_22)
    mul_223: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_222, 768)
    sum_79: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_222, [2], True)
    mul_224: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_222, mul_221);  mul_222 = None
    sum_80: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_224, [2], True);  mul_224 = None
    mul_225: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_221, sum_80);  sum_80 = None
    sub_76: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_223, sum_79);  mul_223 = sum_79 = None
    sub_77: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_76, mul_225);  sub_76 = mul_225 = None
    div_37: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
    mul_226: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_37, sub_77);  div_37 = sub_77 = None
    mul_227: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_189, mul_221);  mul_221 = None
    sum_81: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_227, [0, 1]);  mul_227 = None
    sum_82: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_189, [0, 1]);  add_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_190: "f32[768]" = torch.ops.aten.add.Tensor(add_168, sum_81);  add_168 = sum_81 = None
    add_191: "f32[768]" = torch.ops.aten.add.Tensor(add_169, sum_82);  add_169 = sum_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_388: "f32[2048, 768]" = torch.ops.aten.view.default(mul_226, [2048, 768])
    permute_275: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
    mm_52: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_388, permute_275);  permute_275 = None
    permute_276: "f32[768, 2048]" = torch.ops.aten.permute.default(view_388, [1, 0])
    mm_53: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_276, view_176);  permute_276 = view_176 = None
    permute_277: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_83: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_388, [0], True);  view_388 = None
    view_389: "f32[768]" = torch.ops.aten.view.default(sum_83, [768]);  sum_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_192: "f32[768]" = torch.ops.aten.add.Tensor(add_170, view_389);  add_170 = view_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    permute_278: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_277, [1, 0]);  permute_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_193: "f32[768, 3072]" = torch.ops.aten.add.Tensor(add_171, permute_278);  add_171 = permute_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_390: "f32[4, 512, 3072]" = torch.ops.aten.view.default(mm_52, [4, 512, 3072]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_228: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_390, mul_61);  mul_61 = None
    mul_229: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_390, add_72);  view_390 = add_72 = None
    alias_34: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(alias_15);  alias_15 = None
    mul_230: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(alias_34, alias_34);  alias_34 = None
    sub_78: "f32[4, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_230);  mul_230 = None
    mul_231: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_228, sub_78);  mul_228 = sub_78 = None
    mul_232: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_231, 0.7978845608028654);  mul_231 = None
    mul_233: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_232, 0.044715)
    pow_19: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_175, 2.0);  view_175 = None
    mul_234: "f32[4, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_19, 3.0);  pow_19 = None
    mul_235: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_233, mul_234);  mul_233 = mul_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_194: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_232, mul_235);  mul_232 = mul_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_236: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_229, 0.5);  mul_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_195: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(add_194, mul_236);  add_194 = mul_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_391: "f32[2048, 3072]" = torch.ops.aten.view.default(add_195, [2048, 3072]);  add_195 = None
    permute_279: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
    mm_54: "f32[2048, 768]" = torch.ops.aten.mm.default(view_391, permute_279);  permute_279 = None
    permute_280: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_391, [1, 0])
    mm_55: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_280, view_174);  permute_280 = view_174 = None
    permute_281: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_84: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_391, [0], True);  view_391 = None
    view_392: "f32[3072]" = torch.ops.aten.view.default(sum_84, [3072]);  sum_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_196: "f32[3072]" = torch.ops.aten.add.Tensor(add_174, view_392);  add_174 = view_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    permute_282: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_281, [1, 0]);  permute_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_197: "f32[3072, 768]" = torch.ops.aten.add.Tensor(add_175, permute_282);  add_175 = permute_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_393: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_54, [4, 512, 768]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_198: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_226, view_393);  mul_226 = view_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    sub_79: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_68, getitem_31);  add_68 = getitem_31 = None
    mul_237: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_15);  sub_79 = None
    mul_238: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_198, primals_16)
    mul_239: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_238, 768)
    sum_85: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_238, [2], True)
    mul_240: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_238, mul_237);  mul_238 = None
    sum_86: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_240, [2], True);  mul_240 = None
    mul_241: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_237, sum_86);  sum_86 = None
    sub_80: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_239, sum_85);  mul_239 = sum_85 = None
    sub_81: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_80, mul_241);  sub_80 = mul_241 = None
    div_38: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
    mul_242: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_38, sub_81);  div_38 = sub_81 = None
    mul_243: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_198, mul_237);  mul_237 = None
    sum_87: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_243, [0, 1]);  mul_243 = None
    sum_88: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_198, [0, 1]);  add_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_199: "f32[768]" = torch.ops.aten.add.Tensor(add_177, sum_87);  add_177 = sum_87 = None
    add_200: "f32[768]" = torch.ops.aten.add.Tensor(add_178, sum_88);  add_178 = sum_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_394: "f32[2048, 768]" = torch.ops.aten.view.default(mul_242, [2048, 768])
    permute_283: "f32[768, 768]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
    mm_56: "f32[2048, 768]" = torch.ops.aten.mm.default(view_394, permute_283);  permute_283 = None
    permute_284: "f32[768, 2048]" = torch.ops.aten.permute.default(view_394, [1, 0])
    mm_57: "f32[768, 768]" = torch.ops.aten.mm.default(permute_284, view_172);  permute_284 = view_172 = None
    permute_285: "f32[768, 768]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_89: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_394, [0], True);  view_394 = None
    view_395: "f32[768]" = torch.ops.aten.view.default(sum_89, [768]);  sum_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_201: "f32[768]" = torch.ops.aten.add.Tensor(add_179, view_395);  add_179 = view_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    permute_286: "f32[768, 768]" = torch.ops.aten.permute.default(permute_285, [1, 0]);  permute_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_202: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_180, permute_286);  add_180 = permute_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_396: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_56, [4, 512, 768]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    view_397: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_396, [4, 512, 12, 64]);  view_396 = None
    permute_287: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_397, [0, 2, 1, 3]);  view_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    clone_89: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_287, memory_format = torch.contiguous_format);  permute_287 = None
    view_398: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_89, [48, 512, 64]);  clone_89 = None
    permute_288: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_168, [0, 2, 1]);  view_168 = None
    bmm_40: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_288, view_398);  permute_288 = None
    permute_289: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_169, [0, 2, 1]);  view_169 = None
    bmm_41: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_398, permute_289);  view_398 = permute_289 = None
    view_399: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_40, [4, 12, 512, 64]);  bmm_40 = None
    view_400: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_41, [4, 12, 512, 512]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_35: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(alias_14);  alias_14 = None
    mul_244: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_400, alias_35);  view_400 = None
    sum_90: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_244, [-1], True)
    mul_245: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_35, sum_90);  alias_35 = sum_90 = None
    sub_82: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_244, mul_245);  mul_244 = mul_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_39: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_82, 8.0);  sub_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_401: "f32[48, 512, 512]" = torch.ops.aten.view.default(div_39, [48, 512, 512]);  div_39 = None
    permute_290: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_165, [0, 2, 1]);  view_165 = None
    bmm_42: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_290, view_401);  permute_290 = None
    permute_291: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_166, [0, 2, 1]);  view_166 = None
    bmm_43: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_401, permute_291);  view_401 = permute_291 = None
    view_402: "f32[4, 12, 64, 512]" = torch.ops.aten.view.default(bmm_42, [4, 12, 64, 512]);  bmm_42 = None
    view_403: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_43, [4, 12, 512, 64]);  bmm_43 = None
    permute_292: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_402, [0, 1, 3, 2]);  view_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_293: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_399, [0, 2, 1, 3]);  view_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_90: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_293, memory_format = torch.contiguous_format);  permute_293 = None
    view_404: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_90, [4, 512, 768]);  clone_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_294: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(permute_292, [0, 2, 1, 3]);  permute_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_405: "f32[4, 512, 768]" = torch.ops.aten.view.default(permute_294, [4, 512, 768]);  permute_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_295: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_403, [0, 2, 1, 3]);  view_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_91: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_295, memory_format = torch.contiguous_format);  permute_295 = None
    view_406: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_91, [4, 512, 768]);  clone_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_407: "f32[2048, 768]" = torch.ops.aten.view.default(view_404, [2048, 768]);  view_404 = None
    permute_296: "f32[768, 768]" = torch.ops.aten.permute.default(permute_80, [1, 0]);  permute_80 = None
    mm_58: "f32[2048, 768]" = torch.ops.aten.mm.default(view_407, permute_296);  permute_296 = None
    permute_297: "f32[768, 2048]" = torch.ops.aten.permute.default(view_407, [1, 0])
    mm_59: "f32[768, 768]" = torch.ops.aten.mm.default(permute_297, view_160);  permute_297 = view_160 = None
    permute_298: "f32[768, 768]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_91: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_407, [0], True);  view_407 = None
    view_408: "f32[768]" = torch.ops.aten.view.default(sum_91, [768]);  sum_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_203: "f32[768]" = torch.ops.aten.add.Tensor(add_181, view_408);  add_181 = view_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    permute_299: "f32[768, 768]" = torch.ops.aten.permute.default(permute_298, [1, 0]);  permute_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_204: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_182, permute_299);  add_182 = permute_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_409: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_58, [4, 512, 768]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_205: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_242, view_409);  mul_242 = view_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    clone_92: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_405, memory_format = torch.contiguous_format);  view_405 = None
    view_410: "f32[2048, 768]" = torch.ops.aten.view.default(clone_92, [2048, 768]);  clone_92 = None
    permute_300: "f32[768, 768]" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
    mm_60: "f32[2048, 768]" = torch.ops.aten.mm.default(view_410, permute_300);  permute_300 = None
    permute_301: "f32[768, 2048]" = torch.ops.aten.permute.default(view_410, [1, 0])
    mm_61: "f32[768, 768]" = torch.ops.aten.mm.default(permute_301, view_158);  permute_301 = view_158 = None
    permute_302: "f32[768, 768]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_92: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_410, [0], True);  view_410 = None
    view_411: "f32[768]" = torch.ops.aten.view.default(sum_92, [768]);  sum_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_206: "f32[768]" = torch.ops.aten.add.Tensor(add_184, view_411);  add_184 = view_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    permute_303: "f32[768, 768]" = torch.ops.aten.permute.default(permute_302, [1, 0]);  permute_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_207: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_185, permute_303);  add_185 = permute_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_412: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_60, [4, 512, 768]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_208: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_205, view_412);  add_205 = view_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_413: "f32[2048, 768]" = torch.ops.aten.view.default(view_406, [2048, 768]);  view_406 = None
    permute_304: "f32[768, 768]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
    mm_62: "f32[2048, 768]" = torch.ops.aten.mm.default(view_413, permute_304);  permute_304 = None
    permute_305: "f32[768, 2048]" = torch.ops.aten.permute.default(view_413, [1, 0])
    mm_63: "f32[768, 768]" = torch.ops.aten.mm.default(permute_305, view_156);  permute_305 = view_156 = None
    permute_306: "f32[768, 768]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_93: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_413, [0], True);  view_413 = None
    view_414: "f32[768]" = torch.ops.aten.view.default(sum_93, [768]);  sum_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_209: "f32[768]" = torch.ops.aten.add.Tensor(add_187, view_414);  add_187 = view_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    permute_307: "f32[768, 768]" = torch.ops.aten.permute.default(permute_306, [1, 0]);  permute_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_210: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_188, permute_307);  add_188 = permute_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_415: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_62, [4, 512, 768]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_211: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_208, view_415);  add_208 = view_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    sub_83: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_64, getitem_29);  add_64 = getitem_29 = None
    mul_246: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_14);  sub_83 = None
    mul_247: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_211, primals_22)
    mul_248: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_247, 768)
    sum_94: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_247, [2], True)
    mul_249: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_247, mul_246);  mul_247 = None
    sum_95: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_249, [2], True);  mul_249 = None
    mul_250: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_246, sum_95);  sum_95 = None
    sub_84: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_248, sum_94);  mul_248 = sum_94 = None
    sub_85: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_84, mul_250);  sub_84 = mul_250 = None
    div_40: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
    mul_251: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_40, sub_85);  div_40 = sub_85 = None
    mul_252: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_211, mul_246);  mul_246 = None
    sum_96: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_252, [0, 1]);  mul_252 = None
    sum_97: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_211, [0, 1]);  add_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_212: "f32[768]" = torch.ops.aten.add.Tensor(add_190, sum_96);  add_190 = sum_96 = None
    add_213: "f32[768]" = torch.ops.aten.add.Tensor(add_191, sum_97);  add_191 = sum_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_416: "f32[2048, 768]" = torch.ops.aten.view.default(mul_251, [2048, 768])
    permute_308: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
    mm_64: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_416, permute_308);  permute_308 = None
    permute_309: "f32[768, 2048]" = torch.ops.aten.permute.default(view_416, [1, 0])
    mm_65: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_309, view_154);  permute_309 = view_154 = None
    permute_310: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_98: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_416, [0], True);  view_416 = None
    view_417: "f32[768]" = torch.ops.aten.view.default(sum_98, [768]);  sum_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_214: "f32[768]" = torch.ops.aten.add.Tensor(add_192, view_417);  add_192 = view_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    permute_311: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_310, [1, 0]);  permute_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_215: "f32[768, 3072]" = torch.ops.aten.add.Tensor(add_193, permute_311);  add_193 = permute_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_418: "f32[4, 512, 3072]" = torch.ops.aten.view.default(mm_64, [4, 512, 3072]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_253: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_418, mul_53);  mul_53 = None
    mul_254: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_418, add_63);  view_418 = add_63 = None
    alias_36: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(alias_13);  alias_13 = None
    mul_255: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(alias_36, alias_36);  alias_36 = None
    sub_86: "f32[4, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_255);  mul_255 = None
    mul_256: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_253, sub_86);  mul_253 = sub_86 = None
    mul_257: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_256, 0.7978845608028654);  mul_256 = None
    mul_258: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_257, 0.044715)
    pow_20: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_153, 2.0);  view_153 = None
    mul_259: "f32[4, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_20, 3.0);  pow_20 = None
    mul_260: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_258, mul_259);  mul_258 = mul_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_216: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_257, mul_260);  mul_257 = mul_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_261: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_254, 0.5);  mul_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_217: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(add_216, mul_261);  add_216 = mul_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_419: "f32[2048, 3072]" = torch.ops.aten.view.default(add_217, [2048, 3072]);  add_217 = None
    permute_312: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
    mm_66: "f32[2048, 768]" = torch.ops.aten.mm.default(view_419, permute_312);  permute_312 = None
    permute_313: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_419, [1, 0])
    mm_67: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_313, view_152);  permute_313 = view_152 = None
    permute_314: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_99: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_419, [0], True);  view_419 = None
    view_420: "f32[3072]" = torch.ops.aten.view.default(sum_99, [3072]);  sum_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_218: "f32[3072]" = torch.ops.aten.add.Tensor(add_196, view_420);  add_196 = view_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    permute_315: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_314, [1, 0]);  permute_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_219: "f32[3072, 768]" = torch.ops.aten.add.Tensor(add_197, permute_315);  add_197 = permute_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_421: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_66, [4, 512, 768]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_220: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_251, view_421);  mul_251 = view_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    sub_87: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_59, getitem_27);  add_59 = getitem_27 = None
    mul_262: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_13);  sub_87 = None
    mul_263: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_220, primals_16)
    mul_264: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_263, 768)
    sum_100: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_263, [2], True)
    mul_265: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_263, mul_262);  mul_263 = None
    sum_101: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_265, [2], True);  mul_265 = None
    mul_266: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_262, sum_101);  sum_101 = None
    sub_88: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_264, sum_100);  mul_264 = sum_100 = None
    sub_89: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_88, mul_266);  sub_88 = mul_266 = None
    div_41: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
    mul_267: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_41, sub_89);  div_41 = sub_89 = None
    mul_268: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_220, mul_262);  mul_262 = None
    sum_102: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_268, [0, 1]);  mul_268 = None
    sum_103: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_220, [0, 1]);  add_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_221: "f32[768]" = torch.ops.aten.add.Tensor(add_199, sum_102);  add_199 = sum_102 = None
    add_222: "f32[768]" = torch.ops.aten.add.Tensor(add_200, sum_103);  add_200 = sum_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_422: "f32[2048, 768]" = torch.ops.aten.view.default(mul_267, [2048, 768])
    permute_316: "f32[768, 768]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
    mm_68: "f32[2048, 768]" = torch.ops.aten.mm.default(view_422, permute_316);  permute_316 = None
    permute_317: "f32[768, 2048]" = torch.ops.aten.permute.default(view_422, [1, 0])
    mm_69: "f32[768, 768]" = torch.ops.aten.mm.default(permute_317, view_150);  permute_317 = view_150 = None
    permute_318: "f32[768, 768]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_104: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_422, [0], True);  view_422 = None
    view_423: "f32[768]" = torch.ops.aten.view.default(sum_104, [768]);  sum_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_223: "f32[768]" = torch.ops.aten.add.Tensor(add_201, view_423);  add_201 = view_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    permute_319: "f32[768, 768]" = torch.ops.aten.permute.default(permute_318, [1, 0]);  permute_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_224: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_202, permute_319);  add_202 = permute_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_424: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_68, [4, 512, 768]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    view_425: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_424, [4, 512, 12, 64]);  view_424 = None
    permute_320: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_425, [0, 2, 1, 3]);  view_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    clone_93: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_320, memory_format = torch.contiguous_format);  permute_320 = None
    view_426: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_93, [48, 512, 64]);  clone_93 = None
    permute_321: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_146, [0, 2, 1]);  view_146 = None
    bmm_44: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_321, view_426);  permute_321 = None
    permute_322: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_147, [0, 2, 1]);  view_147 = None
    bmm_45: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_426, permute_322);  view_426 = permute_322 = None
    view_427: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_44, [4, 12, 512, 64]);  bmm_44 = None
    view_428: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_45, [4, 12, 512, 512]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_37: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    mul_269: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_428, alias_37);  view_428 = None
    sum_105: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_269, [-1], True)
    mul_270: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_37, sum_105);  alias_37 = sum_105 = None
    sub_90: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_269, mul_270);  mul_269 = mul_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_42: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_90, 8.0);  sub_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_429: "f32[48, 512, 512]" = torch.ops.aten.view.default(div_42, [48, 512, 512]);  div_42 = None
    permute_323: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_143, [0, 2, 1]);  view_143 = None
    bmm_46: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_323, view_429);  permute_323 = None
    permute_324: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_144, [0, 2, 1]);  view_144 = None
    bmm_47: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_429, permute_324);  view_429 = permute_324 = None
    view_430: "f32[4, 12, 64, 512]" = torch.ops.aten.view.default(bmm_46, [4, 12, 64, 512]);  bmm_46 = None
    view_431: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_47, [4, 12, 512, 64]);  bmm_47 = None
    permute_325: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_430, [0, 1, 3, 2]);  view_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_326: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_427, [0, 2, 1, 3]);  view_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_94: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_326, memory_format = torch.contiguous_format);  permute_326 = None
    view_432: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_94, [4, 512, 768]);  clone_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_327: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(permute_325, [0, 2, 1, 3]);  permute_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_433: "f32[4, 512, 768]" = torch.ops.aten.view.default(permute_327, [4, 512, 768]);  permute_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_328: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_431, [0, 2, 1, 3]);  view_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_95: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_328, memory_format = torch.contiguous_format);  permute_328 = None
    view_434: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_95, [4, 512, 768]);  clone_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_435: "f32[2048, 768]" = torch.ops.aten.view.default(view_432, [2048, 768]);  view_432 = None
    permute_329: "f32[768, 768]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
    mm_70: "f32[2048, 768]" = torch.ops.aten.mm.default(view_435, permute_329);  permute_329 = None
    permute_330: "f32[768, 2048]" = torch.ops.aten.permute.default(view_435, [1, 0])
    mm_71: "f32[768, 768]" = torch.ops.aten.mm.default(permute_330, view_138);  permute_330 = view_138 = None
    permute_331: "f32[768, 768]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_106: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_435, [0], True);  view_435 = None
    view_436: "f32[768]" = torch.ops.aten.view.default(sum_106, [768]);  sum_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_225: "f32[768]" = torch.ops.aten.add.Tensor(add_203, view_436);  add_203 = view_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    permute_332: "f32[768, 768]" = torch.ops.aten.permute.default(permute_331, [1, 0]);  permute_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_226: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_204, permute_332);  add_204 = permute_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_437: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_70, [4, 512, 768]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_227: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_267, view_437);  mul_267 = view_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    clone_96: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_433, memory_format = torch.contiguous_format);  view_433 = None
    view_438: "f32[2048, 768]" = torch.ops.aten.view.default(clone_96, [2048, 768]);  clone_96 = None
    permute_333: "f32[768, 768]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
    mm_72: "f32[2048, 768]" = torch.ops.aten.mm.default(view_438, permute_333);  permute_333 = None
    permute_334: "f32[768, 2048]" = torch.ops.aten.permute.default(view_438, [1, 0])
    mm_73: "f32[768, 768]" = torch.ops.aten.mm.default(permute_334, view_136);  permute_334 = view_136 = None
    permute_335: "f32[768, 768]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_107: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_438, [0], True);  view_438 = None
    view_439: "f32[768]" = torch.ops.aten.view.default(sum_107, [768]);  sum_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_228: "f32[768]" = torch.ops.aten.add.Tensor(add_206, view_439);  add_206 = view_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    permute_336: "f32[768, 768]" = torch.ops.aten.permute.default(permute_335, [1, 0]);  permute_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_229: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_207, permute_336);  add_207 = permute_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_440: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_72, [4, 512, 768]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_230: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_227, view_440);  add_227 = view_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_441: "f32[2048, 768]" = torch.ops.aten.view.default(view_434, [2048, 768]);  view_434 = None
    permute_337: "f32[768, 768]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
    mm_74: "f32[2048, 768]" = torch.ops.aten.mm.default(view_441, permute_337);  permute_337 = None
    permute_338: "f32[768, 2048]" = torch.ops.aten.permute.default(view_441, [1, 0])
    mm_75: "f32[768, 768]" = torch.ops.aten.mm.default(permute_338, view_134);  permute_338 = view_134 = None
    permute_339: "f32[768, 768]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_108: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_441, [0], True);  view_441 = None
    view_442: "f32[768]" = torch.ops.aten.view.default(sum_108, [768]);  sum_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_231: "f32[768]" = torch.ops.aten.add.Tensor(add_209, view_442);  add_209 = view_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    permute_340: "f32[768, 768]" = torch.ops.aten.permute.default(permute_339, [1, 0]);  permute_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_232: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_210, permute_340);  add_210 = permute_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_443: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_74, [4, 512, 768]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_233: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_230, view_443);  add_230 = view_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    sub_91: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_55, getitem_25);  add_55 = getitem_25 = None
    mul_271: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_91, rsqrt_12);  sub_91 = None
    mul_272: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_233, primals_22)
    mul_273: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_272, 768)
    sum_109: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_272, [2], True)
    mul_274: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_272, mul_271);  mul_272 = None
    sum_110: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_274, [2], True);  mul_274 = None
    mul_275: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_271, sum_110);  sum_110 = None
    sub_92: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_273, sum_109);  mul_273 = sum_109 = None
    sub_93: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_92, mul_275);  sub_92 = mul_275 = None
    div_43: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
    mul_276: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_43, sub_93);  div_43 = sub_93 = None
    mul_277: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_233, mul_271);  mul_271 = None
    sum_111: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_277, [0, 1]);  mul_277 = None
    sum_112: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_233, [0, 1]);  add_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_234: "f32[768]" = torch.ops.aten.add.Tensor(add_212, sum_111);  add_212 = sum_111 = None
    add_235: "f32[768]" = torch.ops.aten.add.Tensor(add_213, sum_112);  add_213 = sum_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_444: "f32[2048, 768]" = torch.ops.aten.view.default(mul_276, [2048, 768])
    permute_341: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    mm_76: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_444, permute_341);  permute_341 = None
    permute_342: "f32[768, 2048]" = torch.ops.aten.permute.default(view_444, [1, 0])
    mm_77: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_342, view_132);  permute_342 = view_132 = None
    permute_343: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_113: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_444, [0], True);  view_444 = None
    view_445: "f32[768]" = torch.ops.aten.view.default(sum_113, [768]);  sum_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_236: "f32[768]" = torch.ops.aten.add.Tensor(add_214, view_445);  add_214 = view_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    permute_344: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_343, [1, 0]);  permute_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_237: "f32[768, 3072]" = torch.ops.aten.add.Tensor(add_215, permute_344);  add_215 = permute_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_446: "f32[4, 512, 3072]" = torch.ops.aten.view.default(mm_76, [4, 512, 3072]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_278: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_446, mul_45);  mul_45 = None
    mul_279: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_446, add_54);  view_446 = add_54 = None
    alias_38: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    mul_280: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(alias_38, alias_38);  alias_38 = None
    sub_94: "f32[4, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_280);  mul_280 = None
    mul_281: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_278, sub_94);  mul_278 = sub_94 = None
    mul_282: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_281, 0.7978845608028654);  mul_281 = None
    mul_283: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_282, 0.044715)
    pow_21: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_131, 2.0);  view_131 = None
    mul_284: "f32[4, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_21, 3.0);  pow_21 = None
    mul_285: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_283, mul_284);  mul_283 = mul_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_238: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_282, mul_285);  mul_282 = mul_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_286: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_279, 0.5);  mul_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_239: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(add_238, mul_286);  add_238 = mul_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_447: "f32[2048, 3072]" = torch.ops.aten.view.default(add_239, [2048, 3072]);  add_239 = None
    permute_345: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    mm_78: "f32[2048, 768]" = torch.ops.aten.mm.default(view_447, permute_345);  permute_345 = None
    permute_346: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_447, [1, 0])
    mm_79: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_346, view_130);  permute_346 = view_130 = None
    permute_347: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_114: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_447, [0], True);  view_447 = None
    view_448: "f32[3072]" = torch.ops.aten.view.default(sum_114, [3072]);  sum_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_240: "f32[3072]" = torch.ops.aten.add.Tensor(add_218, view_448);  add_218 = view_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    permute_348: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_347, [1, 0]);  permute_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_241: "f32[3072, 768]" = torch.ops.aten.add.Tensor(add_219, permute_348);  add_219 = permute_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_449: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_78, [4, 512, 768]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_242: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_276, view_449);  mul_276 = view_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    sub_95: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_50, getitem_23);  add_50 = getitem_23 = None
    mul_287: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_95, rsqrt_11);  sub_95 = None
    mul_288: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_242, primals_16)
    mul_289: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_288, 768)
    sum_115: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_288, [2], True)
    mul_290: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_288, mul_287);  mul_288 = None
    sum_116: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_290, [2], True);  mul_290 = None
    mul_291: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_287, sum_116);  sum_116 = None
    sub_96: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_289, sum_115);  mul_289 = sum_115 = None
    sub_97: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_96, mul_291);  sub_96 = mul_291 = None
    div_44: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
    mul_292: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_44, sub_97);  div_44 = sub_97 = None
    mul_293: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_242, mul_287);  mul_287 = None
    sum_117: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_293, [0, 1]);  mul_293 = None
    sum_118: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_242, [0, 1]);  add_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_243: "f32[768]" = torch.ops.aten.add.Tensor(add_221, sum_117);  add_221 = sum_117 = None
    add_244: "f32[768]" = torch.ops.aten.add.Tensor(add_222, sum_118);  add_222 = sum_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_450: "f32[2048, 768]" = torch.ops.aten.view.default(mul_292, [2048, 768])
    permute_349: "f32[768, 768]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
    mm_80: "f32[2048, 768]" = torch.ops.aten.mm.default(view_450, permute_349);  permute_349 = None
    permute_350: "f32[768, 2048]" = torch.ops.aten.permute.default(view_450, [1, 0])
    mm_81: "f32[768, 768]" = torch.ops.aten.mm.default(permute_350, view_128);  permute_350 = view_128 = None
    permute_351: "f32[768, 768]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_119: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_450, [0], True);  view_450 = None
    view_451: "f32[768]" = torch.ops.aten.view.default(sum_119, [768]);  sum_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_245: "f32[768]" = torch.ops.aten.add.Tensor(add_223, view_451);  add_223 = view_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    permute_352: "f32[768, 768]" = torch.ops.aten.permute.default(permute_351, [1, 0]);  permute_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_246: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_224, permute_352);  add_224 = permute_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_452: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_80, [4, 512, 768]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    view_453: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_452, [4, 512, 12, 64]);  view_452 = None
    permute_353: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_453, [0, 2, 1, 3]);  view_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    clone_97: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_353, memory_format = torch.contiguous_format);  permute_353 = None
    view_454: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_97, [48, 512, 64]);  clone_97 = None
    permute_354: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_124, [0, 2, 1]);  view_124 = None
    bmm_48: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_354, view_454);  permute_354 = None
    permute_355: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_125, [0, 2, 1]);  view_125 = None
    bmm_49: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_454, permute_355);  view_454 = permute_355 = None
    view_455: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_48, [4, 12, 512, 64]);  bmm_48 = None
    view_456: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_49, [4, 12, 512, 512]);  bmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_39: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    mul_294: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_456, alias_39);  view_456 = None
    sum_120: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_294, [-1], True)
    mul_295: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_39, sum_120);  alias_39 = sum_120 = None
    sub_98: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_294, mul_295);  mul_294 = mul_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_45: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_98, 8.0);  sub_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_457: "f32[48, 512, 512]" = torch.ops.aten.view.default(div_45, [48, 512, 512]);  div_45 = None
    permute_356: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_121, [0, 2, 1]);  view_121 = None
    bmm_50: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_356, view_457);  permute_356 = None
    permute_357: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None
    bmm_51: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_457, permute_357);  view_457 = permute_357 = None
    view_458: "f32[4, 12, 64, 512]" = torch.ops.aten.view.default(bmm_50, [4, 12, 64, 512]);  bmm_50 = None
    view_459: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_51, [4, 12, 512, 64]);  bmm_51 = None
    permute_358: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_458, [0, 1, 3, 2]);  view_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_359: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_455, [0, 2, 1, 3]);  view_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_98: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_359, memory_format = torch.contiguous_format);  permute_359 = None
    view_460: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_98, [4, 512, 768]);  clone_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_360: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(permute_358, [0, 2, 1, 3]);  permute_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_461: "f32[4, 512, 768]" = torch.ops.aten.view.default(permute_360, [4, 512, 768]);  permute_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_361: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_459, [0, 2, 1, 3]);  view_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_99: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_361, memory_format = torch.contiguous_format);  permute_361 = None
    view_462: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_99, [4, 512, 768]);  clone_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_463: "f32[2048, 768]" = torch.ops.aten.view.default(view_460, [2048, 768]);  view_460 = None
    permute_362: "f32[768, 768]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
    mm_82: "f32[2048, 768]" = torch.ops.aten.mm.default(view_463, permute_362);  permute_362 = None
    permute_363: "f32[768, 2048]" = torch.ops.aten.permute.default(view_463, [1, 0])
    mm_83: "f32[768, 768]" = torch.ops.aten.mm.default(permute_363, view_116);  permute_363 = view_116 = None
    permute_364: "f32[768, 768]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_121: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_463, [0], True);  view_463 = None
    view_464: "f32[768]" = torch.ops.aten.view.default(sum_121, [768]);  sum_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_247: "f32[768]" = torch.ops.aten.add.Tensor(add_225, view_464);  add_225 = view_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    permute_365: "f32[768, 768]" = torch.ops.aten.permute.default(permute_364, [1, 0]);  permute_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_248: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_226, permute_365);  add_226 = permute_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_465: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_82, [4, 512, 768]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_249: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_292, view_465);  mul_292 = view_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    clone_100: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_461, memory_format = torch.contiguous_format);  view_461 = None
    view_466: "f32[2048, 768]" = torch.ops.aten.view.default(clone_100, [2048, 768]);  clone_100 = None
    permute_366: "f32[768, 768]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
    mm_84: "f32[2048, 768]" = torch.ops.aten.mm.default(view_466, permute_366);  permute_366 = None
    permute_367: "f32[768, 2048]" = torch.ops.aten.permute.default(view_466, [1, 0])
    mm_85: "f32[768, 768]" = torch.ops.aten.mm.default(permute_367, view_114);  permute_367 = view_114 = None
    permute_368: "f32[768, 768]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_122: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_466, [0], True);  view_466 = None
    view_467: "f32[768]" = torch.ops.aten.view.default(sum_122, [768]);  sum_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_250: "f32[768]" = torch.ops.aten.add.Tensor(add_228, view_467);  add_228 = view_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    permute_369: "f32[768, 768]" = torch.ops.aten.permute.default(permute_368, [1, 0]);  permute_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_251: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_229, permute_369);  add_229 = permute_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_468: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_84, [4, 512, 768]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_252: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_249, view_468);  add_249 = view_468 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_469: "f32[2048, 768]" = torch.ops.aten.view.default(view_462, [2048, 768]);  view_462 = None
    permute_370: "f32[768, 768]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    mm_86: "f32[2048, 768]" = torch.ops.aten.mm.default(view_469, permute_370);  permute_370 = None
    permute_371: "f32[768, 2048]" = torch.ops.aten.permute.default(view_469, [1, 0])
    mm_87: "f32[768, 768]" = torch.ops.aten.mm.default(permute_371, view_112);  permute_371 = view_112 = None
    permute_372: "f32[768, 768]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
    sum_123: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_469, [0], True);  view_469 = None
    view_470: "f32[768]" = torch.ops.aten.view.default(sum_123, [768]);  sum_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_253: "f32[768]" = torch.ops.aten.add.Tensor(add_231, view_470);  add_231 = view_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    permute_373: "f32[768, 768]" = torch.ops.aten.permute.default(permute_372, [1, 0]);  permute_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_254: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_232, permute_373);  add_232 = permute_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_471: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_86, [4, 512, 768]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_255: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_252, view_471);  add_252 = view_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    sub_99: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_46, getitem_21);  add_46 = getitem_21 = None
    mul_296: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_99, rsqrt_10);  sub_99 = None
    mul_297: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_255, primals_22)
    mul_298: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_297, 768)
    sum_124: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_297, [2], True)
    mul_299: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_297, mul_296);  mul_297 = None
    sum_125: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_299, [2], True);  mul_299 = None
    mul_300: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_296, sum_125);  sum_125 = None
    sub_100: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_298, sum_124);  mul_298 = sum_124 = None
    sub_101: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_100, mul_300);  sub_100 = mul_300 = None
    div_46: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
    mul_301: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_46, sub_101);  div_46 = sub_101 = None
    mul_302: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_255, mul_296);  mul_296 = None
    sum_126: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_302, [0, 1]);  mul_302 = None
    sum_127: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_255, [0, 1]);  add_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_256: "f32[768]" = torch.ops.aten.add.Tensor(add_234, sum_126);  add_234 = sum_126 = None
    add_257: "f32[768]" = torch.ops.aten.add.Tensor(add_235, sum_127);  add_235 = sum_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_472: "f32[2048, 768]" = torch.ops.aten.view.default(mul_301, [2048, 768])
    permute_374: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    mm_88: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_472, permute_374);  permute_374 = None
    permute_375: "f32[768, 2048]" = torch.ops.aten.permute.default(view_472, [1, 0])
    mm_89: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_375, view_110);  permute_375 = view_110 = None
    permute_376: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    sum_128: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_472, [0], True);  view_472 = None
    view_473: "f32[768]" = torch.ops.aten.view.default(sum_128, [768]);  sum_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_258: "f32[768]" = torch.ops.aten.add.Tensor(add_236, view_473);  add_236 = view_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    permute_377: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_376, [1, 0]);  permute_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_259: "f32[768, 3072]" = torch.ops.aten.add.Tensor(add_237, permute_377);  add_237 = permute_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_474: "f32[4, 512, 3072]" = torch.ops.aten.view.default(mm_88, [4, 512, 3072]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_303: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_474, mul_37);  mul_37 = None
    mul_304: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_474, add_45);  view_474 = add_45 = None
    alias_40: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    mul_305: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(alias_40, alias_40);  alias_40 = None
    sub_102: "f32[4, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_305);  mul_305 = None
    mul_306: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_303, sub_102);  mul_303 = sub_102 = None
    mul_307: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_306, 0.7978845608028654);  mul_306 = None
    mul_308: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_307, 0.044715)
    pow_22: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_109, 2.0);  view_109 = None
    mul_309: "f32[4, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_22, 3.0);  pow_22 = None
    mul_310: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_308, mul_309);  mul_308 = mul_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_260: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_307, mul_310);  mul_307 = mul_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_311: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_304, 0.5);  mul_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_261: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(add_260, mul_311);  add_260 = mul_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_475: "f32[2048, 3072]" = torch.ops.aten.view.default(add_261, [2048, 3072]);  add_261 = None
    permute_378: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    mm_90: "f32[2048, 768]" = torch.ops.aten.mm.default(view_475, permute_378);  permute_378 = None
    permute_379: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_475, [1, 0])
    mm_91: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_379, view_108);  permute_379 = view_108 = None
    permute_380: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_129: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_475, [0], True);  view_475 = None
    view_476: "f32[3072]" = torch.ops.aten.view.default(sum_129, [3072]);  sum_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_262: "f32[3072]" = torch.ops.aten.add.Tensor(add_240, view_476);  add_240 = view_476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    permute_381: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_380, [1, 0]);  permute_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_263: "f32[3072, 768]" = torch.ops.aten.add.Tensor(add_241, permute_381);  add_241 = permute_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_477: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_90, [4, 512, 768]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_264: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_301, view_477);  mul_301 = view_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    sub_103: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_41, getitem_19);  add_41 = getitem_19 = None
    mul_312: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_103, rsqrt_9);  sub_103 = None
    mul_313: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_264, primals_16)
    mul_314: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_313, 768)
    sum_130: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_313, [2], True)
    mul_315: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_313, mul_312);  mul_313 = None
    sum_131: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_315, [2], True);  mul_315 = None
    mul_316: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_312, sum_131);  sum_131 = None
    sub_104: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_314, sum_130);  mul_314 = sum_130 = None
    sub_105: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_104, mul_316);  sub_104 = mul_316 = None
    div_47: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
    mul_317: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_47, sub_105);  div_47 = sub_105 = None
    mul_318: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_264, mul_312);  mul_312 = None
    sum_132: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_318, [0, 1]);  mul_318 = None
    sum_133: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_264, [0, 1]);  add_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_265: "f32[768]" = torch.ops.aten.add.Tensor(add_243, sum_132);  add_243 = sum_132 = None
    add_266: "f32[768]" = torch.ops.aten.add.Tensor(add_244, sum_133);  add_244 = sum_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_478: "f32[2048, 768]" = torch.ops.aten.view.default(mul_317, [2048, 768])
    permute_382: "f32[768, 768]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
    mm_92: "f32[2048, 768]" = torch.ops.aten.mm.default(view_478, permute_382);  permute_382 = None
    permute_383: "f32[768, 2048]" = torch.ops.aten.permute.default(view_478, [1, 0])
    mm_93: "f32[768, 768]" = torch.ops.aten.mm.default(permute_383, view_106);  permute_383 = view_106 = None
    permute_384: "f32[768, 768]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_134: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_478, [0], True);  view_478 = None
    view_479: "f32[768]" = torch.ops.aten.view.default(sum_134, [768]);  sum_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_267: "f32[768]" = torch.ops.aten.add.Tensor(add_245, view_479);  add_245 = view_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    permute_385: "f32[768, 768]" = torch.ops.aten.permute.default(permute_384, [1, 0]);  permute_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_268: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_246, permute_385);  add_246 = permute_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_480: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_92, [4, 512, 768]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    view_481: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_480, [4, 512, 12, 64]);  view_480 = None
    permute_386: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_481, [0, 2, 1, 3]);  view_481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    clone_101: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_386, memory_format = torch.contiguous_format);  permute_386 = None
    view_482: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_101, [48, 512, 64]);  clone_101 = None
    permute_387: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_102, [0, 2, 1]);  view_102 = None
    bmm_52: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_387, view_482);  permute_387 = None
    permute_388: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_103, [0, 2, 1]);  view_103 = None
    bmm_53: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_482, permute_388);  view_482 = permute_388 = None
    view_483: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_52, [4, 12, 512, 64]);  bmm_52 = None
    view_484: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_53, [4, 12, 512, 512]);  bmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_41: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    mul_319: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_484, alias_41);  view_484 = None
    sum_135: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_319, [-1], True)
    mul_320: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_41, sum_135);  alias_41 = sum_135 = None
    sub_106: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_319, mul_320);  mul_319 = mul_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_48: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_106, 8.0);  sub_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_485: "f32[48, 512, 512]" = torch.ops.aten.view.default(div_48, [48, 512, 512]);  div_48 = None
    permute_389: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_99, [0, 2, 1]);  view_99 = None
    bmm_54: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_389, view_485);  permute_389 = None
    permute_390: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_100, [0, 2, 1]);  view_100 = None
    bmm_55: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_485, permute_390);  view_485 = permute_390 = None
    view_486: "f32[4, 12, 64, 512]" = torch.ops.aten.view.default(bmm_54, [4, 12, 64, 512]);  bmm_54 = None
    view_487: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_55, [4, 12, 512, 64]);  bmm_55 = None
    permute_391: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_486, [0, 1, 3, 2]);  view_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_392: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_483, [0, 2, 1, 3]);  view_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_102: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_392, memory_format = torch.contiguous_format);  permute_392 = None
    view_488: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_102, [4, 512, 768]);  clone_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_393: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(permute_391, [0, 2, 1, 3]);  permute_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_489: "f32[4, 512, 768]" = torch.ops.aten.view.default(permute_393, [4, 512, 768]);  permute_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_394: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_487, [0, 2, 1, 3]);  view_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_103: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_394, memory_format = torch.contiguous_format);  permute_394 = None
    view_490: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_103, [4, 512, 768]);  clone_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_491: "f32[2048, 768]" = torch.ops.aten.view.default(view_488, [2048, 768]);  view_488 = None
    permute_395: "f32[768, 768]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
    mm_94: "f32[2048, 768]" = torch.ops.aten.mm.default(view_491, permute_395);  permute_395 = None
    permute_396: "f32[768, 2048]" = torch.ops.aten.permute.default(view_491, [1, 0])
    mm_95: "f32[768, 768]" = torch.ops.aten.mm.default(permute_396, view_94);  permute_396 = view_94 = None
    permute_397: "f32[768, 768]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_136: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_491, [0], True);  view_491 = None
    view_492: "f32[768]" = torch.ops.aten.view.default(sum_136, [768]);  sum_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_269: "f32[768]" = torch.ops.aten.add.Tensor(add_247, view_492);  add_247 = view_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    permute_398: "f32[768, 768]" = torch.ops.aten.permute.default(permute_397, [1, 0]);  permute_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_270: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_248, permute_398);  add_248 = permute_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_493: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_94, [4, 512, 768]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_271: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_317, view_493);  mul_317 = view_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    clone_104: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_489, memory_format = torch.contiguous_format);  view_489 = None
    view_494: "f32[2048, 768]" = torch.ops.aten.view.default(clone_104, [2048, 768]);  clone_104 = None
    permute_399: "f32[768, 768]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
    mm_96: "f32[2048, 768]" = torch.ops.aten.mm.default(view_494, permute_399);  permute_399 = None
    permute_400: "f32[768, 2048]" = torch.ops.aten.permute.default(view_494, [1, 0])
    mm_97: "f32[768, 768]" = torch.ops.aten.mm.default(permute_400, view_92);  permute_400 = view_92 = None
    permute_401: "f32[768, 768]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_137: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_494, [0], True);  view_494 = None
    view_495: "f32[768]" = torch.ops.aten.view.default(sum_137, [768]);  sum_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_272: "f32[768]" = torch.ops.aten.add.Tensor(add_250, view_495);  add_250 = view_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    permute_402: "f32[768, 768]" = torch.ops.aten.permute.default(permute_401, [1, 0]);  permute_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_273: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_251, permute_402);  add_251 = permute_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_496: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_96, [4, 512, 768]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_274: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_271, view_496);  add_271 = view_496 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_497: "f32[2048, 768]" = torch.ops.aten.view.default(view_490, [2048, 768]);  view_490 = None
    permute_403: "f32[768, 768]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
    mm_98: "f32[2048, 768]" = torch.ops.aten.mm.default(view_497, permute_403);  permute_403 = None
    permute_404: "f32[768, 2048]" = torch.ops.aten.permute.default(view_497, [1, 0])
    mm_99: "f32[768, 768]" = torch.ops.aten.mm.default(permute_404, view_90);  permute_404 = view_90 = None
    permute_405: "f32[768, 768]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    sum_138: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_497, [0], True);  view_497 = None
    view_498: "f32[768]" = torch.ops.aten.view.default(sum_138, [768]);  sum_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_275: "f32[768]" = torch.ops.aten.add.Tensor(add_253, view_498);  add_253 = view_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    permute_406: "f32[768, 768]" = torch.ops.aten.permute.default(permute_405, [1, 0]);  permute_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_276: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_254, permute_406);  add_254 = permute_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_499: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_98, [4, 512, 768]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_277: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_274, view_499);  add_274 = view_499 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    sub_107: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_37, getitem_17);  add_37 = getitem_17 = None
    mul_321: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_107, rsqrt_8);  sub_107 = None
    mul_322: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_277, primals_22)
    mul_323: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_322, 768)
    sum_139: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_322, [2], True)
    mul_324: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_322, mul_321);  mul_322 = None
    sum_140: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_324, [2], True);  mul_324 = None
    mul_325: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_321, sum_140);  sum_140 = None
    sub_108: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_323, sum_139);  mul_323 = sum_139 = None
    sub_109: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_108, mul_325);  sub_108 = mul_325 = None
    div_49: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
    mul_326: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_49, sub_109);  div_49 = sub_109 = None
    mul_327: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_277, mul_321);  mul_321 = None
    sum_141: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_327, [0, 1]);  mul_327 = None
    sum_142: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_277, [0, 1]);  add_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_278: "f32[768]" = torch.ops.aten.add.Tensor(add_256, sum_141);  add_256 = sum_141 = None
    add_279: "f32[768]" = torch.ops.aten.add.Tensor(add_257, sum_142);  add_257 = sum_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_500: "f32[2048, 768]" = torch.ops.aten.view.default(mul_326, [2048, 768])
    permute_407: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
    mm_100: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_500, permute_407);  permute_407 = None
    permute_408: "f32[768, 2048]" = torch.ops.aten.permute.default(view_500, [1, 0])
    mm_101: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_408, view_88);  permute_408 = view_88 = None
    permute_409: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_101, [1, 0]);  mm_101 = None
    sum_143: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_500, [0], True);  view_500 = None
    view_501: "f32[768]" = torch.ops.aten.view.default(sum_143, [768]);  sum_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_280: "f32[768]" = torch.ops.aten.add.Tensor(add_258, view_501);  add_258 = view_501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    permute_410: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_409, [1, 0]);  permute_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_281: "f32[768, 3072]" = torch.ops.aten.add.Tensor(add_259, permute_410);  add_259 = permute_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_502: "f32[4, 512, 3072]" = torch.ops.aten.view.default(mm_100, [4, 512, 3072]);  mm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_328: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_502, mul_29);  mul_29 = None
    mul_329: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_502, add_36);  view_502 = add_36 = None
    alias_42: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    mul_330: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(alias_42, alias_42);  alias_42 = None
    sub_110: "f32[4, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_330);  mul_330 = None
    mul_331: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_328, sub_110);  mul_328 = sub_110 = None
    mul_332: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_331, 0.7978845608028654);  mul_331 = None
    mul_333: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_332, 0.044715)
    pow_23: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_87, 2.0);  view_87 = None
    mul_334: "f32[4, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_23, 3.0);  pow_23 = None
    mul_335: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_333, mul_334);  mul_333 = mul_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_282: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_332, mul_335);  mul_332 = mul_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_336: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_329, 0.5);  mul_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_283: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(add_282, mul_336);  add_282 = mul_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_503: "f32[2048, 3072]" = torch.ops.aten.view.default(add_283, [2048, 3072]);  add_283 = None
    permute_411: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
    mm_102: "f32[2048, 768]" = torch.ops.aten.mm.default(view_503, permute_411);  permute_411 = None
    permute_412: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_503, [1, 0])
    mm_103: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_412, view_86);  permute_412 = view_86 = None
    permute_413: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
    sum_144: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_503, [0], True);  view_503 = None
    view_504: "f32[3072]" = torch.ops.aten.view.default(sum_144, [3072]);  sum_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_284: "f32[3072]" = torch.ops.aten.add.Tensor(add_262, view_504);  add_262 = view_504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    permute_414: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_413, [1, 0]);  permute_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_285: "f32[3072, 768]" = torch.ops.aten.add.Tensor(add_263, permute_414);  add_263 = permute_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_505: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_102, [4, 512, 768]);  mm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_286: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_326, view_505);  mul_326 = view_505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    sub_111: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_32, getitem_15);  add_32 = getitem_15 = None
    mul_337: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_111, rsqrt_7);  sub_111 = None
    mul_338: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_286, primals_16)
    mul_339: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_338, 768)
    sum_145: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_338, [2], True)
    mul_340: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_338, mul_337);  mul_338 = None
    sum_146: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_340, [2], True);  mul_340 = None
    mul_341: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_337, sum_146);  sum_146 = None
    sub_112: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_339, sum_145);  mul_339 = sum_145 = None
    sub_113: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_112, mul_341);  sub_112 = mul_341 = None
    div_50: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
    mul_342: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_50, sub_113);  div_50 = sub_113 = None
    mul_343: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_286, mul_337);  mul_337 = None
    sum_147: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_343, [0, 1]);  mul_343 = None
    sum_148: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_286, [0, 1]);  add_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_287: "f32[768]" = torch.ops.aten.add.Tensor(add_265, sum_147);  add_265 = sum_147 = None
    add_288: "f32[768]" = torch.ops.aten.add.Tensor(add_266, sum_148);  add_266 = sum_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_506: "f32[2048, 768]" = torch.ops.aten.view.default(mul_342, [2048, 768])
    permute_415: "f32[768, 768]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    mm_104: "f32[2048, 768]" = torch.ops.aten.mm.default(view_506, permute_415);  permute_415 = None
    permute_416: "f32[768, 2048]" = torch.ops.aten.permute.default(view_506, [1, 0])
    mm_105: "f32[768, 768]" = torch.ops.aten.mm.default(permute_416, view_84);  permute_416 = view_84 = None
    permute_417: "f32[768, 768]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
    sum_149: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_506, [0], True);  view_506 = None
    view_507: "f32[768]" = torch.ops.aten.view.default(sum_149, [768]);  sum_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_289: "f32[768]" = torch.ops.aten.add.Tensor(add_267, view_507);  add_267 = view_507 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    permute_418: "f32[768, 768]" = torch.ops.aten.permute.default(permute_417, [1, 0]);  permute_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_290: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_268, permute_418);  add_268 = permute_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_508: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_104, [4, 512, 768]);  mm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    view_509: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_508, [4, 512, 12, 64]);  view_508 = None
    permute_419: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_509, [0, 2, 1, 3]);  view_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    clone_105: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_419, memory_format = torch.contiguous_format);  permute_419 = None
    view_510: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_105, [48, 512, 64]);  clone_105 = None
    permute_420: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_80, [0, 2, 1]);  view_80 = None
    bmm_56: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_420, view_510);  permute_420 = None
    permute_421: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_81, [0, 2, 1]);  view_81 = None
    bmm_57: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_510, permute_421);  view_510 = permute_421 = None
    view_511: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_56, [4, 12, 512, 64]);  bmm_56 = None
    view_512: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_57, [4, 12, 512, 512]);  bmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_43: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    mul_344: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_512, alias_43);  view_512 = None
    sum_150: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_344, [-1], True)
    mul_345: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_43, sum_150);  alias_43 = sum_150 = None
    sub_114: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_344, mul_345);  mul_344 = mul_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_51: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_114, 8.0);  sub_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_513: "f32[48, 512, 512]" = torch.ops.aten.view.default(div_51, [48, 512, 512]);  div_51 = None
    permute_422: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_77, [0, 2, 1]);  view_77 = None
    bmm_58: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_422, view_513);  permute_422 = None
    permute_423: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
    bmm_59: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_513, permute_423);  view_513 = permute_423 = None
    view_514: "f32[4, 12, 64, 512]" = torch.ops.aten.view.default(bmm_58, [4, 12, 64, 512]);  bmm_58 = None
    view_515: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_59, [4, 12, 512, 64]);  bmm_59 = None
    permute_424: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_514, [0, 1, 3, 2]);  view_514 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_425: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_511, [0, 2, 1, 3]);  view_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_106: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_425, memory_format = torch.contiguous_format);  permute_425 = None
    view_516: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_106, [4, 512, 768]);  clone_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_426: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(permute_424, [0, 2, 1, 3]);  permute_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_517: "f32[4, 512, 768]" = torch.ops.aten.view.default(permute_426, [4, 512, 768]);  permute_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_427: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_515, [0, 2, 1, 3]);  view_515 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_107: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_427, memory_format = torch.contiguous_format);  permute_427 = None
    view_518: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_107, [4, 512, 768]);  clone_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_519: "f32[2048, 768]" = torch.ops.aten.view.default(view_516, [2048, 768]);  view_516 = None
    permute_428: "f32[768, 768]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
    mm_106: "f32[2048, 768]" = torch.ops.aten.mm.default(view_519, permute_428);  permute_428 = None
    permute_429: "f32[768, 2048]" = torch.ops.aten.permute.default(view_519, [1, 0])
    mm_107: "f32[768, 768]" = torch.ops.aten.mm.default(permute_429, view_72);  permute_429 = view_72 = None
    permute_430: "f32[768, 768]" = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
    sum_151: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_519, [0], True);  view_519 = None
    view_520: "f32[768]" = torch.ops.aten.view.default(sum_151, [768]);  sum_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_291: "f32[768]" = torch.ops.aten.add.Tensor(add_269, view_520);  add_269 = view_520 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    permute_431: "f32[768, 768]" = torch.ops.aten.permute.default(permute_430, [1, 0]);  permute_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_292: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_270, permute_431);  add_270 = permute_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_521: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_106, [4, 512, 768]);  mm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_293: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_342, view_521);  mul_342 = view_521 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    clone_108: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_517, memory_format = torch.contiguous_format);  view_517 = None
    view_522: "f32[2048, 768]" = torch.ops.aten.view.default(clone_108, [2048, 768]);  clone_108 = None
    permute_432: "f32[768, 768]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
    mm_108: "f32[2048, 768]" = torch.ops.aten.mm.default(view_522, permute_432);  permute_432 = None
    permute_433: "f32[768, 2048]" = torch.ops.aten.permute.default(view_522, [1, 0])
    mm_109: "f32[768, 768]" = torch.ops.aten.mm.default(permute_433, view_70);  permute_433 = view_70 = None
    permute_434: "f32[768, 768]" = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
    sum_152: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_522, [0], True);  view_522 = None
    view_523: "f32[768]" = torch.ops.aten.view.default(sum_152, [768]);  sum_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_294: "f32[768]" = torch.ops.aten.add.Tensor(add_272, view_523);  add_272 = view_523 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    permute_435: "f32[768, 768]" = torch.ops.aten.permute.default(permute_434, [1, 0]);  permute_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_295: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_273, permute_435);  add_273 = permute_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_524: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_108, [4, 512, 768]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_296: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_293, view_524);  add_293 = view_524 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_525: "f32[2048, 768]" = torch.ops.aten.view.default(view_518, [2048, 768]);  view_518 = None
    permute_436: "f32[768, 768]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    mm_110: "f32[2048, 768]" = torch.ops.aten.mm.default(view_525, permute_436);  permute_436 = None
    permute_437: "f32[768, 2048]" = torch.ops.aten.permute.default(view_525, [1, 0])
    mm_111: "f32[768, 768]" = torch.ops.aten.mm.default(permute_437, view_68);  permute_437 = view_68 = None
    permute_438: "f32[768, 768]" = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
    sum_153: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_525, [0], True);  view_525 = None
    view_526: "f32[768]" = torch.ops.aten.view.default(sum_153, [768]);  sum_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_297: "f32[768]" = torch.ops.aten.add.Tensor(add_275, view_526);  add_275 = view_526 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    permute_439: "f32[768, 768]" = torch.ops.aten.permute.default(permute_438, [1, 0]);  permute_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_298: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_276, permute_439);  add_276 = permute_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_527: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_110, [4, 512, 768]);  mm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_299: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_296, view_527);  add_296 = view_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    sub_115: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_28, getitem_13);  add_28 = getitem_13 = None
    mul_346: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_115, rsqrt_6);  sub_115 = None
    mul_347: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_299, primals_22)
    mul_348: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_347, 768)
    sum_154: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_347, [2], True)
    mul_349: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_347, mul_346);  mul_347 = None
    sum_155: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_349, [2], True);  mul_349 = None
    mul_350: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_346, sum_155);  sum_155 = None
    sub_116: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_348, sum_154);  mul_348 = sum_154 = None
    sub_117: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_116, mul_350);  sub_116 = mul_350 = None
    div_52: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
    mul_351: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_52, sub_117);  div_52 = sub_117 = None
    mul_352: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_299, mul_346);  mul_346 = None
    sum_156: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_352, [0, 1]);  mul_352 = None
    sum_157: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_299, [0, 1]);  add_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_300: "f32[768]" = torch.ops.aten.add.Tensor(add_278, sum_156);  add_278 = sum_156 = None
    add_301: "f32[768]" = torch.ops.aten.add.Tensor(add_279, sum_157);  add_279 = sum_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_528: "f32[2048, 768]" = torch.ops.aten.view.default(mul_351, [2048, 768])
    permute_440: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    mm_112: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_528, permute_440);  permute_440 = None
    permute_441: "f32[768, 2048]" = torch.ops.aten.permute.default(view_528, [1, 0])
    mm_113: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_441, view_66);  permute_441 = view_66 = None
    permute_442: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_113, [1, 0]);  mm_113 = None
    sum_158: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_528, [0], True);  view_528 = None
    view_529: "f32[768]" = torch.ops.aten.view.default(sum_158, [768]);  sum_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_302: "f32[768]" = torch.ops.aten.add.Tensor(add_280, view_529);  add_280 = view_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    permute_443: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_442, [1, 0]);  permute_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_303: "f32[768, 3072]" = torch.ops.aten.add.Tensor(add_281, permute_443);  add_281 = permute_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_530: "f32[4, 512, 3072]" = torch.ops.aten.view.default(mm_112, [4, 512, 3072]);  mm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_353: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_530, mul_21);  mul_21 = None
    mul_354: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_530, add_27);  view_530 = add_27 = None
    alias_44: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    mul_355: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(alias_44, alias_44);  alias_44 = None
    sub_118: "f32[4, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_355);  mul_355 = None
    mul_356: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_353, sub_118);  mul_353 = sub_118 = None
    mul_357: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_356, 0.7978845608028654);  mul_356 = None
    mul_358: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_357, 0.044715)
    pow_24: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_65, 2.0);  view_65 = None
    mul_359: "f32[4, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_24, 3.0);  pow_24 = None
    mul_360: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_358, mul_359);  mul_358 = mul_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_304: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_357, mul_360);  mul_357 = mul_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_361: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_354, 0.5);  mul_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_305: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(add_304, mul_361);  add_304 = mul_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_531: "f32[2048, 3072]" = torch.ops.aten.view.default(add_305, [2048, 3072]);  add_305 = None
    permute_444: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    mm_114: "f32[2048, 768]" = torch.ops.aten.mm.default(view_531, permute_444);  permute_444 = None
    permute_445: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_531, [1, 0])
    mm_115: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_445, view_64);  permute_445 = view_64 = None
    permute_446: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
    sum_159: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_531, [0], True);  view_531 = None
    view_532: "f32[3072]" = torch.ops.aten.view.default(sum_159, [3072]);  sum_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_306: "f32[3072]" = torch.ops.aten.add.Tensor(add_284, view_532);  add_284 = view_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    permute_447: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_446, [1, 0]);  permute_446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_307: "f32[3072, 768]" = torch.ops.aten.add.Tensor(add_285, permute_447);  add_285 = permute_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_533: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_114, [4, 512, 768]);  mm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_308: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_351, view_533);  mul_351 = view_533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    sub_119: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_23, getitem_11);  add_23 = getitem_11 = None
    mul_362: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_119, rsqrt_5);  sub_119 = None
    mul_363: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_308, primals_16)
    mul_364: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_363, 768)
    sum_160: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_363, [2], True)
    mul_365: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_363, mul_362);  mul_363 = None
    sum_161: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_365, [2], True);  mul_365 = None
    mul_366: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_362, sum_161);  sum_161 = None
    sub_120: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_364, sum_160);  mul_364 = sum_160 = None
    sub_121: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_120, mul_366);  sub_120 = mul_366 = None
    div_53: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
    mul_367: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_53, sub_121);  div_53 = sub_121 = None
    mul_368: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_308, mul_362);  mul_362 = None
    sum_162: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_368, [0, 1]);  mul_368 = None
    sum_163: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_308, [0, 1]);  add_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_309: "f32[768]" = torch.ops.aten.add.Tensor(add_287, sum_162);  add_287 = sum_162 = None
    add_310: "f32[768]" = torch.ops.aten.add.Tensor(add_288, sum_163);  add_288 = sum_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_534: "f32[2048, 768]" = torch.ops.aten.view.default(mul_367, [2048, 768])
    permute_448: "f32[768, 768]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
    mm_116: "f32[2048, 768]" = torch.ops.aten.mm.default(view_534, permute_448);  permute_448 = None
    permute_449: "f32[768, 2048]" = torch.ops.aten.permute.default(view_534, [1, 0])
    mm_117: "f32[768, 768]" = torch.ops.aten.mm.default(permute_449, view_62);  permute_449 = view_62 = None
    permute_450: "f32[768, 768]" = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
    sum_164: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_534, [0], True);  view_534 = None
    view_535: "f32[768]" = torch.ops.aten.view.default(sum_164, [768]);  sum_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_311: "f32[768]" = torch.ops.aten.add.Tensor(add_289, view_535);  add_289 = view_535 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    permute_451: "f32[768, 768]" = torch.ops.aten.permute.default(permute_450, [1, 0]);  permute_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_312: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_290, permute_451);  add_290 = permute_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_536: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_116, [4, 512, 768]);  mm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    view_537: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_536, [4, 512, 12, 64]);  view_536 = None
    permute_452: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_537, [0, 2, 1, 3]);  view_537 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    clone_109: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_452, memory_format = torch.contiguous_format);  permute_452 = None
    view_538: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_109, [48, 512, 64]);  clone_109 = None
    permute_453: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_58, [0, 2, 1]);  view_58 = None
    bmm_60: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_453, view_538);  permute_453 = None
    permute_454: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_59, [0, 2, 1]);  view_59 = None
    bmm_61: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_538, permute_454);  view_538 = permute_454 = None
    view_539: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_60, [4, 12, 512, 64]);  bmm_60 = None
    view_540: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_61, [4, 12, 512, 512]);  bmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_45: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    mul_369: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_540, alias_45);  view_540 = None
    sum_165: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_369, [-1], True)
    mul_370: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_45, sum_165);  alias_45 = sum_165 = None
    sub_122: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_369, mul_370);  mul_369 = mul_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_54: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_122, 8.0);  sub_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_541: "f32[48, 512, 512]" = torch.ops.aten.view.default(div_54, [48, 512, 512]);  div_54 = None
    permute_455: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_55, [0, 2, 1]);  view_55 = None
    bmm_62: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_455, view_541);  permute_455 = None
    permute_456: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None
    bmm_63: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_541, permute_456);  view_541 = permute_456 = None
    view_542: "f32[4, 12, 64, 512]" = torch.ops.aten.view.default(bmm_62, [4, 12, 64, 512]);  bmm_62 = None
    view_543: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_63, [4, 12, 512, 64]);  bmm_63 = None
    permute_457: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_542, [0, 1, 3, 2]);  view_542 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_458: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_539, [0, 2, 1, 3]);  view_539 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_110: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_458, memory_format = torch.contiguous_format);  permute_458 = None
    view_544: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_110, [4, 512, 768]);  clone_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_459: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(permute_457, [0, 2, 1, 3]);  permute_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_545: "f32[4, 512, 768]" = torch.ops.aten.view.default(permute_459, [4, 512, 768]);  permute_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_460: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_543, [0, 2, 1, 3]);  view_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_111: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_460, memory_format = torch.contiguous_format);  permute_460 = None
    view_546: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_111, [4, 512, 768]);  clone_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_547: "f32[2048, 768]" = torch.ops.aten.view.default(view_544, [2048, 768]);  view_544 = None
    permute_461: "f32[768, 768]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
    mm_118: "f32[2048, 768]" = torch.ops.aten.mm.default(view_547, permute_461);  permute_461 = None
    permute_462: "f32[768, 2048]" = torch.ops.aten.permute.default(view_547, [1, 0])
    mm_119: "f32[768, 768]" = torch.ops.aten.mm.default(permute_462, view_50);  permute_462 = view_50 = None
    permute_463: "f32[768, 768]" = torch.ops.aten.permute.default(mm_119, [1, 0]);  mm_119 = None
    sum_166: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_547, [0], True);  view_547 = None
    view_548: "f32[768]" = torch.ops.aten.view.default(sum_166, [768]);  sum_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_313: "f32[768]" = torch.ops.aten.add.Tensor(add_291, view_548);  add_291 = view_548 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    permute_464: "f32[768, 768]" = torch.ops.aten.permute.default(permute_463, [1, 0]);  permute_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_314: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_292, permute_464);  add_292 = permute_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_549: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_118, [4, 512, 768]);  mm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_315: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_367, view_549);  mul_367 = view_549 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    clone_112: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_545, memory_format = torch.contiguous_format);  view_545 = None
    view_550: "f32[2048, 768]" = torch.ops.aten.view.default(clone_112, [2048, 768]);  clone_112 = None
    permute_465: "f32[768, 768]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
    mm_120: "f32[2048, 768]" = torch.ops.aten.mm.default(view_550, permute_465);  permute_465 = None
    permute_466: "f32[768, 2048]" = torch.ops.aten.permute.default(view_550, [1, 0])
    mm_121: "f32[768, 768]" = torch.ops.aten.mm.default(permute_466, view_48);  permute_466 = view_48 = None
    permute_467: "f32[768, 768]" = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
    sum_167: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_550, [0], True);  view_550 = None
    view_551: "f32[768]" = torch.ops.aten.view.default(sum_167, [768]);  sum_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_316: "f32[768]" = torch.ops.aten.add.Tensor(add_294, view_551);  add_294 = view_551 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    permute_468: "f32[768, 768]" = torch.ops.aten.permute.default(permute_467, [1, 0]);  permute_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_317: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_295, permute_468);  add_295 = permute_468 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_552: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_120, [4, 512, 768]);  mm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_318: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_315, view_552);  add_315 = view_552 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_553: "f32[2048, 768]" = torch.ops.aten.view.default(view_546, [2048, 768]);  view_546 = None
    permute_469: "f32[768, 768]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    mm_122: "f32[2048, 768]" = torch.ops.aten.mm.default(view_553, permute_469);  permute_469 = None
    permute_470: "f32[768, 2048]" = torch.ops.aten.permute.default(view_553, [1, 0])
    mm_123: "f32[768, 768]" = torch.ops.aten.mm.default(permute_470, view_46);  permute_470 = view_46 = None
    permute_471: "f32[768, 768]" = torch.ops.aten.permute.default(mm_123, [1, 0]);  mm_123 = None
    sum_168: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_553, [0], True);  view_553 = None
    view_554: "f32[768]" = torch.ops.aten.view.default(sum_168, [768]);  sum_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_319: "f32[768]" = torch.ops.aten.add.Tensor(add_297, view_554);  add_297 = view_554 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    permute_472: "f32[768, 768]" = torch.ops.aten.permute.default(permute_471, [1, 0]);  permute_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_320: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_298, permute_472);  add_298 = permute_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_555: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_122, [4, 512, 768]);  mm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_321: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_318, view_555);  add_318 = view_555 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    sub_123: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_19, getitem_9);  add_19 = getitem_9 = None
    mul_371: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_123, rsqrt_4);  sub_123 = None
    mul_372: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_321, primals_22)
    mul_373: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_372, 768)
    sum_169: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_372, [2], True)
    mul_374: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_372, mul_371);  mul_372 = None
    sum_170: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_374, [2], True);  mul_374 = None
    mul_375: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_371, sum_170);  sum_170 = None
    sub_124: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_373, sum_169);  mul_373 = sum_169 = None
    sub_125: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_124, mul_375);  sub_124 = mul_375 = None
    div_55: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
    mul_376: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_55, sub_125);  div_55 = sub_125 = None
    mul_377: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_321, mul_371);  mul_371 = None
    sum_171: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_377, [0, 1]);  mul_377 = None
    sum_172: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_321, [0, 1]);  add_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_322: "f32[768]" = torch.ops.aten.add.Tensor(add_300, sum_171);  add_300 = sum_171 = None
    add_323: "f32[768]" = torch.ops.aten.add.Tensor(add_301, sum_172);  add_301 = sum_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_556: "f32[2048, 768]" = torch.ops.aten.view.default(mul_376, [2048, 768])
    permute_473: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    mm_124: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_556, permute_473);  permute_473 = None
    permute_474: "f32[768, 2048]" = torch.ops.aten.permute.default(view_556, [1, 0])
    mm_125: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_474, view_44);  permute_474 = view_44 = None
    permute_475: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_125, [1, 0]);  mm_125 = None
    sum_173: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_556, [0], True);  view_556 = None
    view_557: "f32[768]" = torch.ops.aten.view.default(sum_173, [768]);  sum_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_324: "f32[768]" = torch.ops.aten.add.Tensor(add_302, view_557);  add_302 = view_557 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    permute_476: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_475, [1, 0]);  permute_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_325: "f32[768, 3072]" = torch.ops.aten.add.Tensor(add_303, permute_476);  add_303 = permute_476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_558: "f32[4, 512, 3072]" = torch.ops.aten.view.default(mm_124, [4, 512, 3072]);  mm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_378: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_558, mul_13);  mul_13 = None
    mul_379: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_558, add_18);  view_558 = add_18 = None
    alias_46: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    mul_380: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(alias_46, alias_46);  alias_46 = None
    sub_126: "f32[4, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_380);  mul_380 = None
    mul_381: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_378, sub_126);  mul_378 = sub_126 = None
    mul_382: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_381, 0.7978845608028654);  mul_381 = None
    mul_383: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_382, 0.044715)
    pow_25: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_43, 2.0);  view_43 = None
    mul_384: "f32[4, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_25, 3.0);  pow_25 = None
    mul_385: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_383, mul_384);  mul_383 = mul_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_326: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_382, mul_385);  mul_382 = mul_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_386: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_379, 0.5);  mul_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_327: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(add_326, mul_386);  add_326 = mul_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_559: "f32[2048, 3072]" = torch.ops.aten.view.default(add_327, [2048, 3072]);  add_327 = None
    permute_477: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    mm_126: "f32[2048, 768]" = torch.ops.aten.mm.default(view_559, permute_477);  permute_477 = None
    permute_478: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_559, [1, 0])
    mm_127: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_478, view_42);  permute_478 = view_42 = None
    permute_479: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_127, [1, 0]);  mm_127 = None
    sum_174: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_559, [0], True);  view_559 = None
    view_560: "f32[3072]" = torch.ops.aten.view.default(sum_174, [3072]);  sum_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_328: "f32[3072]" = torch.ops.aten.add.Tensor(add_306, view_560);  add_306 = view_560 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    permute_480: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_479, [1, 0]);  permute_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_329: "f32[3072, 768]" = torch.ops.aten.add.Tensor(add_307, permute_480);  add_307 = permute_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_561: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_126, [4, 512, 768]);  mm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_330: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_376, view_561);  mul_376 = view_561 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    sub_127: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_14, getitem_7);  add_14 = getitem_7 = None
    mul_387: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_127, rsqrt_3);  sub_127 = None
    mul_388: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_330, primals_16)
    mul_389: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_388, 768)
    sum_175: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_388, [2], True)
    mul_390: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_388, mul_387);  mul_388 = None
    sum_176: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_390, [2], True);  mul_390 = None
    mul_391: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_387, sum_176);  sum_176 = None
    sub_128: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_389, sum_175);  mul_389 = sum_175 = None
    sub_129: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_128, mul_391);  sub_128 = mul_391 = None
    div_56: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
    mul_392: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_56, sub_129);  div_56 = sub_129 = None
    mul_393: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_330, mul_387);  mul_387 = None
    sum_177: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_393, [0, 1]);  mul_393 = None
    sum_178: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_330, [0, 1]);  add_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_331: "f32[768]" = torch.ops.aten.add.Tensor(add_309, sum_177);  add_309 = sum_177 = None
    add_332: "f32[768]" = torch.ops.aten.add.Tensor(add_310, sum_178);  add_310 = sum_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_562: "f32[2048, 768]" = torch.ops.aten.view.default(mul_392, [2048, 768])
    permute_481: "f32[768, 768]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    mm_128: "f32[2048, 768]" = torch.ops.aten.mm.default(view_562, permute_481);  permute_481 = None
    permute_482: "f32[768, 2048]" = torch.ops.aten.permute.default(view_562, [1, 0])
    mm_129: "f32[768, 768]" = torch.ops.aten.mm.default(permute_482, view_40);  permute_482 = view_40 = None
    permute_483: "f32[768, 768]" = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
    sum_179: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_562, [0], True);  view_562 = None
    view_563: "f32[768]" = torch.ops.aten.view.default(sum_179, [768]);  sum_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_333: "f32[768]" = torch.ops.aten.add.Tensor(add_311, view_563);  add_311 = view_563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    permute_484: "f32[768, 768]" = torch.ops.aten.permute.default(permute_483, [1, 0]);  permute_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_334: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_312, permute_484);  add_312 = permute_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_564: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_128, [4, 512, 768]);  mm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    view_565: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_564, [4, 512, 12, 64]);  view_564 = None
    permute_485: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_565, [0, 2, 1, 3]);  view_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    clone_113: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_485, memory_format = torch.contiguous_format);  permute_485 = None
    view_566: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_113, [48, 512, 64]);  clone_113 = None
    permute_486: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_36, [0, 2, 1]);  view_36 = None
    bmm_64: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_486, view_566);  permute_486 = None
    permute_487: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_37, [0, 2, 1]);  view_37 = None
    bmm_65: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_566, permute_487);  view_566 = permute_487 = None
    view_567: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_64, [4, 12, 512, 64]);  bmm_64 = None
    view_568: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_65, [4, 12, 512, 512]);  bmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_47: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    mul_394: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_568, alias_47);  view_568 = None
    sum_180: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_394, [-1], True)
    mul_395: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_47, sum_180);  alias_47 = sum_180 = None
    sub_130: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_394, mul_395);  mul_394 = mul_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_57: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_130, 8.0);  sub_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_569: "f32[48, 512, 512]" = torch.ops.aten.view.default(div_57, [48, 512, 512]);  div_57 = None
    permute_488: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_33, [0, 2, 1]);  view_33 = None
    bmm_66: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_488, view_569);  permute_488 = None
    permute_489: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
    bmm_67: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_569, permute_489);  view_569 = permute_489 = None
    view_570: "f32[4, 12, 64, 512]" = torch.ops.aten.view.default(bmm_66, [4, 12, 64, 512]);  bmm_66 = None
    view_571: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_67, [4, 12, 512, 64]);  bmm_67 = None
    permute_490: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_570, [0, 1, 3, 2]);  view_570 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_491: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_567, [0, 2, 1, 3]);  view_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_114: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_491, memory_format = torch.contiguous_format);  permute_491 = None
    view_572: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_114, [4, 512, 768]);  clone_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_492: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(permute_490, [0, 2, 1, 3]);  permute_490 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_573: "f32[4, 512, 768]" = torch.ops.aten.view.default(permute_492, [4, 512, 768]);  permute_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_493: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_571, [0, 2, 1, 3]);  view_571 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_115: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_493, memory_format = torch.contiguous_format);  permute_493 = None
    view_574: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_115, [4, 512, 768]);  clone_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_575: "f32[2048, 768]" = torch.ops.aten.view.default(view_572, [2048, 768]);  view_572 = None
    permute_494: "f32[768, 768]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
    mm_130: "f32[2048, 768]" = torch.ops.aten.mm.default(view_575, permute_494);  permute_494 = None
    permute_495: "f32[768, 2048]" = torch.ops.aten.permute.default(view_575, [1, 0])
    mm_131: "f32[768, 768]" = torch.ops.aten.mm.default(permute_495, view_28);  permute_495 = view_28 = None
    permute_496: "f32[768, 768]" = torch.ops.aten.permute.default(mm_131, [1, 0]);  mm_131 = None
    sum_181: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_575, [0], True);  view_575 = None
    view_576: "f32[768]" = torch.ops.aten.view.default(sum_181, [768]);  sum_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_335: "f32[768]" = torch.ops.aten.add.Tensor(add_313, view_576);  add_313 = view_576 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    permute_497: "f32[768, 768]" = torch.ops.aten.permute.default(permute_496, [1, 0]);  permute_496 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_336: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_314, permute_497);  add_314 = permute_497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_577: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_130, [4, 512, 768]);  mm_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_337: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_392, view_577);  mul_392 = view_577 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    clone_116: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_573, memory_format = torch.contiguous_format);  view_573 = None
    view_578: "f32[2048, 768]" = torch.ops.aten.view.default(clone_116, [2048, 768]);  clone_116 = None
    permute_498: "f32[768, 768]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    mm_132: "f32[2048, 768]" = torch.ops.aten.mm.default(view_578, permute_498);  permute_498 = None
    permute_499: "f32[768, 2048]" = torch.ops.aten.permute.default(view_578, [1, 0])
    mm_133: "f32[768, 768]" = torch.ops.aten.mm.default(permute_499, view_26);  permute_499 = view_26 = None
    permute_500: "f32[768, 768]" = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
    sum_182: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_578, [0], True);  view_578 = None
    view_579: "f32[768]" = torch.ops.aten.view.default(sum_182, [768]);  sum_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_338: "f32[768]" = torch.ops.aten.add.Tensor(add_316, view_579);  add_316 = view_579 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    permute_501: "f32[768, 768]" = torch.ops.aten.permute.default(permute_500, [1, 0]);  permute_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_339: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_317, permute_501);  add_317 = permute_501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_580: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_132, [4, 512, 768]);  mm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_340: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_337, view_580);  add_337 = view_580 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_581: "f32[2048, 768]" = torch.ops.aten.view.default(view_574, [2048, 768]);  view_574 = None
    permute_502: "f32[768, 768]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    mm_134: "f32[2048, 768]" = torch.ops.aten.mm.default(view_581, permute_502);  permute_502 = None
    permute_503: "f32[768, 2048]" = torch.ops.aten.permute.default(view_581, [1, 0])
    mm_135: "f32[768, 768]" = torch.ops.aten.mm.default(permute_503, view_24);  permute_503 = view_24 = None
    permute_504: "f32[768, 768]" = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
    sum_183: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_581, [0], True);  view_581 = None
    view_582: "f32[768]" = torch.ops.aten.view.default(sum_183, [768]);  sum_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_341: "f32[768]" = torch.ops.aten.add.Tensor(add_319, view_582);  add_319 = view_582 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    permute_505: "f32[768, 768]" = torch.ops.aten.permute.default(permute_504, [1, 0]);  permute_504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_342: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_320, permute_505);  add_320 = permute_505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_583: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_134, [4, 512, 768]);  mm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_343: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_340, view_583);  add_340 = view_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    sub_131: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_10, getitem_5);  add_10 = getitem_5 = None
    mul_396: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_131, rsqrt_2);  sub_131 = None
    mul_397: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_343, primals_22);  primals_22 = None
    mul_398: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_397, 768)
    sum_184: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_397, [2], True)
    mul_399: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_397, mul_396);  mul_397 = None
    sum_185: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_399, [2], True);  mul_399 = None
    mul_400: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_396, sum_185);  sum_185 = None
    sub_132: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_398, sum_184);  mul_398 = sum_184 = None
    sub_133: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_132, mul_400);  sub_132 = mul_400 = None
    div_58: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
    mul_401: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_58, sub_133);  div_58 = sub_133 = None
    mul_402: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_343, mul_396);  mul_396 = None
    sum_186: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_402, [0, 1]);  mul_402 = None
    sum_187: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_343, [0, 1]);  add_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_344: "f32[768]" = torch.ops.aten.add.Tensor(add_322, sum_186);  add_322 = sum_186 = None
    add_345: "f32[768]" = torch.ops.aten.add.Tensor(add_323, sum_187);  add_323 = sum_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_584: "f32[2048, 768]" = torch.ops.aten.view.default(mul_401, [2048, 768])
    permute_506: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    mm_136: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_584, permute_506);  permute_506 = None
    permute_507: "f32[768, 2048]" = torch.ops.aten.permute.default(view_584, [1, 0])
    mm_137: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_507, view_22);  permute_507 = view_22 = None
    permute_508: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_137, [1, 0]);  mm_137 = None
    sum_188: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_584, [0], True);  view_584 = None
    view_585: "f32[768]" = torch.ops.aten.view.default(sum_188, [768]);  sum_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_346: "f32[768]" = torch.ops.aten.add.Tensor(add_324, view_585);  add_324 = view_585 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    permute_509: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_508, [1, 0]);  permute_508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    add_347: "f32[768, 3072]" = torch.ops.aten.add.Tensor(add_325, permute_509);  add_325 = permute_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_586: "f32[4, 512, 3072]" = torch.ops.aten.view.default(mm_136, [4, 512, 3072]);  mm_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_403: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_586, mul_5);  mul_5 = None
    mul_404: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_586, add_9);  view_586 = add_9 = None
    alias_48: "f32[4, 512, 3072]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    mul_405: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(alias_48, alias_48);  alias_48 = None
    sub_134: "f32[4, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_405);  mul_405 = None
    mul_406: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_403, sub_134);  mul_403 = sub_134 = None
    mul_407: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_406, 0.7978845608028654);  mul_406 = None
    mul_408: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_407, 0.044715)
    pow_26: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_21, 2.0);  view_21 = None
    mul_409: "f32[4, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_26, 3.0);  pow_26 = None
    mul_410: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_408, mul_409);  mul_408 = mul_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_348: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_407, mul_410);  mul_407 = mul_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_411: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_404, 0.5);  mul_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_349: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(add_348, mul_411);  add_348 = mul_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_587: "f32[2048, 3072]" = torch.ops.aten.view.default(add_349, [2048, 3072]);  add_349 = None
    permute_510: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    mm_138: "f32[2048, 768]" = torch.ops.aten.mm.default(view_587, permute_510);  permute_510 = None
    permute_511: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_587, [1, 0])
    mm_139: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_511, view_20);  permute_511 = view_20 = None
    permute_512: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
    sum_189: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_587, [0], True);  view_587 = None
    view_588: "f32[3072]" = torch.ops.aten.view.default(sum_189, [3072]);  sum_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_350: "f32[3072]" = torch.ops.aten.add.Tensor(add_328, view_588);  add_328 = view_588 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    permute_513: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_512, [1, 0]);  permute_512 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_351: "f32[3072, 768]" = torch.ops.aten.add.Tensor(add_329, permute_513);  add_329 = permute_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_589: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_138, [4, 512, 768]);  mm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    add_352: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_401, view_589);  mul_401 = view_589 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    sub_135: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_5, getitem_3);  add_5 = getitem_3 = None
    mul_412: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_135, rsqrt_1);  sub_135 = None
    mul_413: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_352, primals_16);  primals_16 = None
    mul_414: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_413, 768)
    sum_190: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_413, [2], True)
    mul_415: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_413, mul_412);  mul_413 = None
    sum_191: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_415, [2], True);  mul_415 = None
    mul_416: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_412, sum_191);  sum_191 = None
    sub_136: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_414, sum_190);  mul_414 = sum_190 = None
    sub_137: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_136, mul_416);  sub_136 = mul_416 = None
    div_59: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
    mul_417: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_59, sub_137);  div_59 = sub_137 = None
    mul_418: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_352, mul_412);  mul_412 = None
    sum_192: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_418, [0, 1]);  mul_418 = None
    sum_193: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_352, [0, 1]);  add_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:370, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
    add_353: "f32[768]" = torch.ops.aten.add.Tensor(add_331, sum_192);  add_331 = sum_192 = None
    add_354: "f32[768]" = torch.ops.aten.add.Tensor(add_332, sum_193);  add_332 = sum_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_590: "f32[2048, 768]" = torch.ops.aten.view.default(mul_417, [2048, 768])
    permute_514: "f32[768, 768]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    mm_140: "f32[2048, 768]" = torch.ops.aten.mm.default(view_590, permute_514);  permute_514 = None
    permute_515: "f32[768, 2048]" = torch.ops.aten.permute.default(view_590, [1, 0])
    mm_141: "f32[768, 768]" = torch.ops.aten.mm.default(permute_515, view_18);  permute_515 = view_18 = None
    permute_516: "f32[768, 768]" = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
    sum_194: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_590, [0], True);  view_590 = None
    view_591: "f32[768]" = torch.ops.aten.view.default(sum_194, [768]);  sum_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_355: "f32[768]" = torch.ops.aten.add.Tensor(add_333, view_591);  add_333 = view_591 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    permute_517: "f32[768, 768]" = torch.ops.aten.permute.default(permute_516, [1, 0]);  permute_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    add_356: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_334, permute_517);  add_334 = permute_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_592: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_140, [4, 512, 768]);  mm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    view_593: "f32[4, 512, 12, 64]" = torch.ops.aten.view.default(view_592, [4, 512, 12, 64]);  view_592 = None
    permute_518: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_593, [0, 2, 1, 3]);  view_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    clone_117: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_518, memory_format = torch.contiguous_format);  permute_518 = None
    view_594: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_117, [48, 512, 64]);  clone_117 = None
    permute_519: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_14, [0, 2, 1]);  view_14 = None
    bmm_68: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_519, view_594);  permute_519 = None
    permute_520: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_15, [0, 2, 1]);  view_15 = None
    bmm_69: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_594, permute_520);  view_594 = permute_520 = None
    view_595: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_68, [4, 12, 512, 64]);  bmm_68 = None
    view_596: "f32[4, 12, 512, 512]" = torch.ops.aten.view.default(bmm_69, [4, 12, 512, 512]);  bmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_49: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(alias);  alias = None
    mul_419: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_596, alias_49);  view_596 = None
    sum_195: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_419, [-1], True)
    mul_420: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_49, sum_195);  alias_49 = sum_195 = None
    sub_138: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_419, mul_420);  mul_419 = mul_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:332, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_60: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_138, 8.0);  sub_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:331, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_597: "f32[48, 512, 512]" = torch.ops.aten.view.default(div_60, [48, 512, 512]);  div_60 = None
    permute_521: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
    bmm_70: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_521, view_597);  permute_521 = None
    permute_522: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
    bmm_71: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_597, permute_522);  view_597 = permute_522 = None
    view_598: "f32[4, 12, 64, 512]" = torch.ops.aten.view.default(bmm_70, [4, 12, 64, 512]);  bmm_70 = None
    view_599: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_71, [4, 12, 512, 64]);  bmm_71 = None
    permute_523: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_598, [0, 1, 3, 2]);  view_598 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_524: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_595, [0, 2, 1, 3]);  view_595 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_118: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_524, memory_format = torch.contiguous_format);  permute_524 = None
    view_600: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_118, [4, 512, 768]);  clone_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_525: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(permute_523, [0, 2, 1, 3]);  permute_523 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    view_601: "f32[4, 512, 768]" = torch.ops.aten.view.default(permute_525, [4, 512, 768]);  permute_525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:295, code: return x.permute(0, 2, 1, 3)
    permute_526: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_599, [0, 2, 1, 3]);  view_599 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:294, code: x = x.view(new_x_shape)
    clone_119: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_526, memory_format = torch.contiguous_format);  permute_526 = None
    view_602: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_119, [4, 512, 768]);  clone_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_603: "f32[2048, 768]" = torch.ops.aten.view.default(view_600, [2048, 768]);  view_600 = None
    permute_527: "f32[768, 768]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    mm_142: "f32[2048, 768]" = torch.ops.aten.mm.default(view_603, permute_527);  permute_527 = None
    permute_528: "f32[768, 2048]" = torch.ops.aten.permute.default(view_603, [1, 0])
    mm_143: "f32[768, 768]" = torch.ops.aten.mm.default(permute_528, view_6);  permute_528 = view_6 = None
    permute_529: "f32[768, 768]" = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
    sum_196: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_603, [0], True);  view_603 = None
    view_604: "f32[768]" = torch.ops.aten.view.default(sum_196, [768]);  sum_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_357: "f32[768]" = torch.ops.aten.add.Tensor(add_335, view_604);  add_335 = view_604 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    permute_530: "f32[768, 768]" = torch.ops.aten.permute.default(permute_529, [1, 0]);  permute_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_358: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_336, permute_530);  add_336 = permute_530 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_605: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_142, [4, 512, 768]);  mm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    add_359: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_417, view_605);  mul_417 = view_605 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    clone_120: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_601, memory_format = torch.contiguous_format);  view_601 = None
    view_606: "f32[2048, 768]" = torch.ops.aten.view.default(clone_120, [2048, 768]);  clone_120 = None
    permute_531: "f32[768, 768]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    mm_144: "f32[2048, 768]" = torch.ops.aten.mm.default(view_606, permute_531);  permute_531 = None
    permute_532: "f32[768, 2048]" = torch.ops.aten.permute.default(view_606, [1, 0])
    mm_145: "f32[768, 768]" = torch.ops.aten.mm.default(permute_532, view_4);  permute_532 = view_4 = None
    permute_533: "f32[768, 768]" = torch.ops.aten.permute.default(mm_145, [1, 0]);  mm_145 = None
    sum_197: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_606, [0], True);  view_606 = None
    view_607: "f32[768]" = torch.ops.aten.view.default(sum_197, [768]);  sum_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_360: "f32[768]" = torch.ops.aten.add.Tensor(add_338, view_607);  add_338 = view_607 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    permute_534: "f32[768, 768]" = torch.ops.aten.permute.default(permute_533, [1, 0]);  permute_533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_361: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_339, permute_534);  add_339 = permute_534 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_608: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_144, [4, 512, 768]);  mm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    add_362: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_359, view_608);  add_359 = view_608 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_609: "f32[2048, 768]" = torch.ops.aten.view.default(view_602, [2048, 768]);  view_602 = None
    permute_535: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    mm_146: "f32[2048, 768]" = torch.ops.aten.mm.default(view_609, permute_535);  permute_535 = None
    permute_536: "f32[768, 2048]" = torch.ops.aten.permute.default(view_609, [1, 0])
    mm_147: "f32[768, 768]" = torch.ops.aten.mm.default(permute_536, view_2);  permute_536 = view_2 = None
    permute_537: "f32[768, 768]" = torch.ops.aten.permute.default(mm_147, [1, 0]);  mm_147 = None
    sum_198: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_609, [0], True);  view_609 = None
    view_610: "f32[768]" = torch.ops.aten.view.default(sum_198, [768]);  sum_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_363: "f32[768]" = torch.ops.aten.add.Tensor(add_341, view_610);  add_341 = view_610 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    permute_538: "f32[768, 768]" = torch.ops.aten.permute.default(permute_537, [1, 0]);  permute_537 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_364: "f32[768, 768]" = torch.ops.aten.add.Tensor(add_342, permute_538);  add_342 = permute_538 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_611: "f32[4, 512, 768]" = torch.ops.aten.view.default(mm_146, [4, 512, 768]);  mm_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    add_365: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_362, view_611);  add_362 = view_611 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:467, code: hidden_states = self.embedding_hidden_mapping_in(hidden_states)
    view_612: "f32[2048, 768]" = torch.ops.aten.view.default(add_365, [2048, 768]);  add_365 = None
    permute_539: "f32[768, 128]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    mm_148: "f32[2048, 128]" = torch.ops.aten.mm.default(view_612, permute_539);  permute_539 = None
    permute_540: "f32[768, 2048]" = torch.ops.aten.permute.default(view_612, [1, 0])
    mm_149: "f32[768, 128]" = torch.ops.aten.mm.default(permute_540, view);  permute_540 = view = None
    permute_541: "f32[128, 768]" = torch.ops.aten.permute.default(mm_149, [1, 0]);  mm_149 = None
    sum_199: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_612, [0], True);  view_612 = None
    view_613: "f32[768]" = torch.ops.aten.view.default(sum_199, [768]);  sum_199 = None
    permute_542: "f32[768, 128]" = torch.ops.aten.permute.default(permute_541, [1, 0]);  permute_541 = None
    view_614: "f32[4, 512, 128]" = torch.ops.aten.view.default(mm_148, [4, 512, 128]);  mm_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:257, code: embeddings = self.LayerNorm(embeddings)
    sub_139: "f32[4, 512, 128]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
    mul_421: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(sub_139, rsqrt);  sub_139 = None
    mul_422: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(view_614, primals_4);  primals_4 = None
    mul_423: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(mul_422, 128)
    sum_200: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_422, [2], True)
    mul_424: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(mul_422, mul_421);  mul_422 = None
    sum_201: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_424, [2], True);  mul_424 = None
    mul_425: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(mul_421, sum_201);  sum_201 = None
    sub_140: "f32[4, 512, 128]" = torch.ops.aten.sub.Tensor(mul_423, sum_200);  mul_423 = sum_200 = None
    sub_141: "f32[4, 512, 128]" = torch.ops.aten.sub.Tensor(sub_140, mul_425);  sub_140 = mul_425 = None
    div_61: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 128);  rsqrt = None
    mul_426: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(div_61, sub_141);  div_61 = sub_141 = None
    mul_427: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(view_614, mul_421);  mul_421 = None
    sum_202: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_427, [0, 1]);  mul_427 = None
    sum_203: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_614, [0, 1]);  view_614 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:256, code: embeddings += position_embeddings
    sum_204: "f32[1, 512, 128]" = torch.ops.aten.sum.dim_IntList(mul_426, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:255, code: position_embeddings = self.position_embeddings(position_ids)
    eq: "b8[1, 512]" = torch.ops.aten.eq.Scalar(slice_2, -1)
    unsqueeze_2: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where: "f32[1, 512, 128]" = torch.ops.aten.where.self(unsqueeze_2, scalar_tensor, sum_204);  unsqueeze_2 = scalar_tensor = sum_204 = None
    full_1: "f32[512, 128]" = torch.ops.aten.full.default([512, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put: "f32[512, 128]" = torch.ops.aten._unsafe_index_put.default(full_1, [slice_2], where, True);  full_1 = slice_2 = where = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:251, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    eq_1: "b8[4, 512]" = torch.ops.aten.eq.Scalar(expand, -1)
    unsqueeze_3: "b8[4, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_1: "f32[4, 512, 128]" = torch.ops.aten.where.self(unsqueeze_3, scalar_tensor_1, mul_426);  unsqueeze_3 = scalar_tensor_1 = None
    full_2: "f32[2, 128]" = torch.ops.aten.full.default([2, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_1: "f32[2, 128]" = torch.ops.aten._unsafe_index_put.default(full_2, [expand], where_1, True);  full_2 = expand = where_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:250, code: inputs_embeds = self.word_embeddings(input_ids)
    eq_2: "b8[4, 512]" = torch.ops.aten.eq.Scalar(primals_32, 0)
    unsqueeze_4: "b8[4, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_2, -1);  eq_2 = None
    scalar_tensor_2: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_2: "f32[4, 512, 128]" = torch.ops.aten.where.self(unsqueeze_4, scalar_tensor_2, mul_426);  unsqueeze_4 = scalar_tensor_2 = mul_426 = None
    full_3: "f32[30000, 128]" = torch.ops.aten.full.default([30000, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_2: "f32[30000, 128]" = torch.ops.aten._unsafe_index_put.default(full_3, [primals_32], where_2, True);  full_3 = primals_32 = where_2 = None
    return pytree.tree_unflatten([view_269, _unsafe_index_put_2, _unsafe_index_put_1, _unsafe_index_put, sum_202, sum_203, permute_542, view_613, add_364, add_363, add_361, add_360, add_358, add_357, add_356, add_355, add_353, add_354, add_351, add_350, add_347, add_346, add_344, add_345, permute_142, view_274, sum_16, sum_17, permute_138, view_271, None, None, None], self._out_spec)
    