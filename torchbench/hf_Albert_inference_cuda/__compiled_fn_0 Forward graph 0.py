from __future__ import annotations



def forward(self, arg0_1: "f32[30000, 128]", arg1_1: "f32[2, 128]", arg2_1: "f32[512, 128]", arg3_1: "f32[128]", arg4_1: "f32[128]", arg5_1: "f32[768, 128]", arg6_1: "f32[768]", arg7_1: "f32[768, 768]", arg8_1: "f32[768]", arg9_1: "f32[768, 768]", arg10_1: "f32[768]", arg11_1: "f32[768, 768]", arg12_1: "f32[768]", arg13_1: "f32[768, 768]", arg14_1: "f32[768]", arg15_1: "f32[768]", arg16_1: "f32[768]", arg17_1: "f32[3072, 768]", arg18_1: "f32[3072]", arg19_1: "f32[768, 3072]", arg20_1: "f32[768]", arg21_1: "f32[768]", arg22_1: "f32[768]", arg23_1: "f32[128, 768]", arg24_1: "f32[128]", arg25_1: "f32[128]", arg26_1: "f32[128]", arg27_1: "f32[30000, 128]", arg28_1: "f32[30000]", arg29_1: "i64[1, 512]", arg30_1: "i64[1, 512]", arg31_1: "i64[4, 512]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:715, code: attention_mask = torch.ones(input_shape, device=device)
    full: "f32[4, 512]" = torch.ops.aten.full.default([4, 512], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:718, code: buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
    slice_1: "i64[1, 512]" = torch.ops.aten.slice.Tensor(arg29_1, 0, 0, 9223372036854775807);  arg29_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:719, code: buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
    expand: "i64[4, 512]" = torch.ops.aten.expand.default(slice_1, [4, 512]);  slice_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:724, code: extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    unsqueeze: "f32[4, 1, 512]" = torch.ops.aten.unsqueeze.default(full, 1);  full = None
    unsqueeze_1: "f32[4, 1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:726, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min
    sub: "f32[4, 1, 1, 512]" = torch.ops.aten.sub.Tensor(1.0, unsqueeze_1);  unsqueeze_1 = None
    mul: "f32[4, 1, 1, 512]" = torch.ops.aten.mul.Tensor(sub, -3.4028234663852886e+38);  sub = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:236, code: position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
    slice_2: "i64[1, 512]" = torch.ops.aten.slice.Tensor(arg30_1, 0, 0, 9223372036854775807);  arg30_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:250, code: inputs_embeds = self.word_embeddings(input_ids)
    embedding: "f32[4, 512, 128]" = torch.ops.aten.embedding.default(arg0_1, arg31_1, 0);  arg0_1 = arg31_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:251, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    embedding_1: "f32[4, 512, 128]" = torch.ops.aten.embedding.default(arg1_1, expand);  arg1_1 = expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:253, code: embeddings = inputs_embeds + token_type_embeddings
    add: "f32[4, 512, 128]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:255, code: position_embeddings = self.position_embeddings(position_ids)
    embedding_2: "f32[1, 512, 128]" = torch.ops.aten.embedding.default(arg2_1, slice_2);  arg2_1 = slice_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:256, code: embeddings += position_embeddings
    add_1: "f32[4, 512, 128]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:257, code: embeddings = self.LayerNorm(embeddings)
    var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
    getitem: "f32[4, 512, 1]" = var_mean[0]
    getitem_1: "f32[4, 512, 1]" = var_mean[1];  var_mean = None
    add_2: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
    rsqrt: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
    sub_1: "f32[4, 512, 128]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
    mul_1: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = rsqrt = None
    mul_2: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
    add_3: "f32[4, 512, 128]" = torch.ops.aten.add.Tensor(mul_2, arg4_1);  mul_2 = arg4_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:258, code: embeddings = self.dropout(embeddings)
    clone: "f32[4, 512, 128]" = torch.ops.aten.clone.default(add_3);  add_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:467, code: hidden_states = self.embedding_hidden_mapping_in(hidden_states)
    view: "f32[2048, 128]" = torch.ops.aten.view.default(clone, [2048, 128]);  clone = None
    permute: "f32[128, 768]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
    addmm: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg6_1, view, permute);  arg6_1 = view = permute = None
    view_1: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm, [4, 512, 768]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_2: "f32[2048, 768]" = torch.ops.aten.view.default(view_1, [2048, 768])
    permute_1: "f32[768, 768]" = torch.ops.aten.permute.default(arg7_1, [1, 0])
    addmm_1: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg8_1, view_2, permute_1);  view_2 = permute_1 = None
    view_3: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_1, [4, 512, 768]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_4: "f32[2048, 768]" = torch.ops.aten.view.default(view_1, [2048, 768])
    permute_2: "f32[768, 768]" = torch.ops.aten.permute.default(arg9_1, [1, 0])
    addmm_2: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg10_1, view_4, permute_2);  view_4 = permute_2 = None
    view_5: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_2, [4, 512, 768]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_6: "f32[2048, 768]" = torch.ops.aten.view.default(view_1, [2048, 768])
    permute_3: "f32[768, 768]" = torch.ops.aten.permute.default(arg11_1, [1, 0])
    addmm_3: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg12_1, view_6, permute_3);  view_6 = permute_3 = None
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
    bmm: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_11, view_12);  view_11 = view_12 = None
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    clone_3: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_1);  div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_3: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_3, [4, 12, 512, 512]);  clone_3 = None
    view_14: "f32[48, 512, 512]" = torch.ops.aten.view.default(expand_3, [48, 512, 512]);  expand_3 = None
    expand_4: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_6, [4, 12, 512, 64]);  permute_6 = None
    clone_4: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
    view_15: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_4, [48, 512, 64]);  clone_4 = None
    bmm_1: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_14, view_15);  view_14 = view_15 = None
    view_16: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_1, [4, 12, 512, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    permute_8: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
    clone_5: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
    view_17: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_5, [4, 512, 768]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_18: "f32[2048, 768]" = torch.ops.aten.view.default(view_17, [2048, 768]);  view_17 = None
    permute_9: "f32[768, 768]" = torch.ops.aten.permute.default(arg13_1, [1, 0])
    addmm_4: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg14_1, view_18, permute_9);  view_18 = permute_9 = None
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
    sub_3: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_5, getitem_3);  add_5 = getitem_3 = None
    mul_3: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = rsqrt_1 = None
    mul_4: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_3, arg15_1);  mul_3 = None
    add_7: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_4, arg16_1);  mul_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_20: "f32[2048, 768]" = torch.ops.aten.view.default(add_7, [2048, 768])
    permute_10: "f32[768, 3072]" = torch.ops.aten.permute.default(arg17_1, [1, 0])
    addmm_5: "f32[2048, 3072]" = torch.ops.aten.addmm.default(arg18_1, view_20, permute_10);  view_20 = permute_10 = None
    view_21: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_5, [4, 512, 3072]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_5: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_21, 0.5)
    pow_1: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_21, 3.0)
    mul_6: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
    add_8: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(view_21, mul_6);  view_21 = mul_6 = None
    mul_7: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_8, 0.7978845608028654);  add_8 = None
    tanh: "f32[4, 512, 3072]" = torch.ops.aten.tanh.default(mul_7);  mul_7 = None
    add_9: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
    mul_8: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_5, add_9);  mul_5 = add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_22: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_8, [2048, 3072]);  mul_8 = None
    permute_11: "f32[3072, 768]" = torch.ops.aten.permute.default(arg19_1, [1, 0])
    addmm_6: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg20_1, view_22, permute_11);  view_22 = permute_11 = None
    view_23: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_6, [4, 512, 768]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_10: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_23, add_7);  view_23 = add_7 = None
    var_mean_2 = torch.ops.aten.var_mean.correction(add_10, [2], correction = 0, keepdim = True)
    getitem_4: "f32[4, 512, 1]" = var_mean_2[0]
    getitem_5: "f32[4, 512, 1]" = var_mean_2[1];  var_mean_2 = None
    add_11: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
    rsqrt_2: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_4: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_10, getitem_5);  add_10 = getitem_5 = None
    mul_9: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = rsqrt_2 = None
    mul_10: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_9, arg21_1);  mul_9 = None
    add_12: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_10, arg22_1);  mul_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_24: "f32[2048, 768]" = torch.ops.aten.view.default(add_12, [2048, 768])
    permute_12: "f32[768, 768]" = torch.ops.aten.permute.default(arg7_1, [1, 0])
    addmm_7: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg8_1, view_24, permute_12);  view_24 = permute_12 = None
    view_25: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_7, [4, 512, 768]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_26: "f32[2048, 768]" = torch.ops.aten.view.default(add_12, [2048, 768])
    permute_13: "f32[768, 768]" = torch.ops.aten.permute.default(arg9_1, [1, 0])
    addmm_8: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg10_1, view_26, permute_13);  view_26 = permute_13 = None
    view_27: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_8, [4, 512, 768]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_28: "f32[2048, 768]" = torch.ops.aten.view.default(add_12, [2048, 768])
    permute_14: "f32[768, 768]" = torch.ops.aten.permute.default(arg11_1, [1, 0])
    addmm_9: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg12_1, view_28, permute_14);  view_28 = permute_14 = None
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
    bmm_2: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_33, view_34);  view_33 = view_34 = None
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    clone_9: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_3);  div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_7: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_9, [4, 12, 512, 512]);  clone_9 = None
    view_36: "f32[48, 512, 512]" = torch.ops.aten.view.default(expand_7, [48, 512, 512]);  expand_7 = None
    expand_8: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_17, [4, 12, 512, 64]);  permute_17 = None
    clone_10: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
    view_37: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_10, [48, 512, 64]);  clone_10 = None
    bmm_3: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_36, view_37);  view_36 = view_37 = None
    view_38: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_3, [4, 12, 512, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    permute_19: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
    clone_11: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None
    view_39: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_11, [4, 512, 768]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_40: "f32[2048, 768]" = torch.ops.aten.view.default(view_39, [2048, 768]);  view_39 = None
    permute_20: "f32[768, 768]" = torch.ops.aten.permute.default(arg13_1, [1, 0])
    addmm_10: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg14_1, view_40, permute_20);  view_40 = permute_20 = None
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
    sub_6: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_14, getitem_7);  add_14 = getitem_7 = None
    mul_11: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_3);  sub_6 = rsqrt_3 = None
    mul_12: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_11, arg15_1);  mul_11 = None
    add_16: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_12, arg16_1);  mul_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_42: "f32[2048, 768]" = torch.ops.aten.view.default(add_16, [2048, 768])
    permute_21: "f32[768, 3072]" = torch.ops.aten.permute.default(arg17_1, [1, 0])
    addmm_11: "f32[2048, 3072]" = torch.ops.aten.addmm.default(arg18_1, view_42, permute_21);  view_42 = permute_21 = None
    view_43: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_11, [4, 512, 3072]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_13: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_43, 0.5)
    pow_2: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_43, 3.0)
    mul_14: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
    add_17: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(view_43, mul_14);  view_43 = mul_14 = None
    mul_15: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_17, 0.7978845608028654);  add_17 = None
    tanh_1: "f32[4, 512, 3072]" = torch.ops.aten.tanh.default(mul_15);  mul_15 = None
    add_18: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
    mul_16: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_13, add_18);  mul_13 = add_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_44: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_16, [2048, 3072]);  mul_16 = None
    permute_22: "f32[3072, 768]" = torch.ops.aten.permute.default(arg19_1, [1, 0])
    addmm_12: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg20_1, view_44, permute_22);  view_44 = permute_22 = None
    view_45: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_12, [4, 512, 768]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_19: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_45, add_16);  view_45 = add_16 = None
    var_mean_4 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
    getitem_8: "f32[4, 512, 1]" = var_mean_4[0]
    getitem_9: "f32[4, 512, 1]" = var_mean_4[1];  var_mean_4 = None
    add_20: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
    rsqrt_4: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
    sub_7: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_19, getitem_9);  add_19 = getitem_9 = None
    mul_17: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_4);  sub_7 = rsqrt_4 = None
    mul_18: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_17, arg21_1);  mul_17 = None
    add_21: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_18, arg22_1);  mul_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_46: "f32[2048, 768]" = torch.ops.aten.view.default(add_21, [2048, 768])
    permute_23: "f32[768, 768]" = torch.ops.aten.permute.default(arg7_1, [1, 0])
    addmm_13: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg8_1, view_46, permute_23);  view_46 = permute_23 = None
    view_47: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_13, [4, 512, 768]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_48: "f32[2048, 768]" = torch.ops.aten.view.default(add_21, [2048, 768])
    permute_24: "f32[768, 768]" = torch.ops.aten.permute.default(arg9_1, [1, 0])
    addmm_14: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg10_1, view_48, permute_24);  view_48 = permute_24 = None
    view_49: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_14, [4, 512, 768]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_50: "f32[2048, 768]" = torch.ops.aten.view.default(add_21, [2048, 768])
    permute_25: "f32[768, 768]" = torch.ops.aten.permute.default(arg11_1, [1, 0])
    addmm_15: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg12_1, view_50, permute_25);  view_50 = permute_25 = None
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
    bmm_4: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_55, view_56);  view_55 = view_56 = None
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    clone_15: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_5);  div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_11: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_15, [4, 12, 512, 512]);  clone_15 = None
    view_58: "f32[48, 512, 512]" = torch.ops.aten.view.default(expand_11, [48, 512, 512]);  expand_11 = None
    expand_12: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_28, [4, 12, 512, 64]);  permute_28 = None
    clone_16: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
    view_59: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_16, [48, 512, 64]);  clone_16 = None
    bmm_5: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_58, view_59);  view_58 = view_59 = None
    view_60: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_5, [4, 12, 512, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    permute_30: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_60, [0, 2, 1, 3]);  view_60 = None
    clone_17: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
    view_61: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_17, [4, 512, 768]);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_62: "f32[2048, 768]" = torch.ops.aten.view.default(view_61, [2048, 768]);  view_61 = None
    permute_31: "f32[768, 768]" = torch.ops.aten.permute.default(arg13_1, [1, 0])
    addmm_16: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg14_1, view_62, permute_31);  view_62 = permute_31 = None
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
    sub_9: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_23, getitem_11);  add_23 = getitem_11 = None
    mul_19: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_5);  sub_9 = rsqrt_5 = None
    mul_20: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_19, arg15_1);  mul_19 = None
    add_25: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_20, arg16_1);  mul_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_64: "f32[2048, 768]" = torch.ops.aten.view.default(add_25, [2048, 768])
    permute_32: "f32[768, 3072]" = torch.ops.aten.permute.default(arg17_1, [1, 0])
    addmm_17: "f32[2048, 3072]" = torch.ops.aten.addmm.default(arg18_1, view_64, permute_32);  view_64 = permute_32 = None
    view_65: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_17, [4, 512, 3072]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_21: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_65, 0.5)
    pow_3: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_65, 3.0)
    mul_22: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
    add_26: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(view_65, mul_22);  view_65 = mul_22 = None
    mul_23: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_26, 0.7978845608028654);  add_26 = None
    tanh_2: "f32[4, 512, 3072]" = torch.ops.aten.tanh.default(mul_23);  mul_23 = None
    add_27: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
    mul_24: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_21, add_27);  mul_21 = add_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_66: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_24, [2048, 3072]);  mul_24 = None
    permute_33: "f32[3072, 768]" = torch.ops.aten.permute.default(arg19_1, [1, 0])
    addmm_18: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg20_1, view_66, permute_33);  view_66 = permute_33 = None
    view_67: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_18, [4, 512, 768]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_28: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_67, add_25);  view_67 = add_25 = None
    var_mean_6 = torch.ops.aten.var_mean.correction(add_28, [2], correction = 0, keepdim = True)
    getitem_12: "f32[4, 512, 1]" = var_mean_6[0]
    getitem_13: "f32[4, 512, 1]" = var_mean_6[1];  var_mean_6 = None
    add_29: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
    rsqrt_6: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
    sub_10: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_28, getitem_13);  add_28 = getitem_13 = None
    mul_25: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_6);  sub_10 = rsqrt_6 = None
    mul_26: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_25, arg21_1);  mul_25 = None
    add_30: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_26, arg22_1);  mul_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_68: "f32[2048, 768]" = torch.ops.aten.view.default(add_30, [2048, 768])
    permute_34: "f32[768, 768]" = torch.ops.aten.permute.default(arg7_1, [1, 0])
    addmm_19: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg8_1, view_68, permute_34);  view_68 = permute_34 = None
    view_69: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_19, [4, 512, 768]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_70: "f32[2048, 768]" = torch.ops.aten.view.default(add_30, [2048, 768])
    permute_35: "f32[768, 768]" = torch.ops.aten.permute.default(arg9_1, [1, 0])
    addmm_20: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg10_1, view_70, permute_35);  view_70 = permute_35 = None
    view_71: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_20, [4, 512, 768]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_72: "f32[2048, 768]" = torch.ops.aten.view.default(add_30, [2048, 768])
    permute_36: "f32[768, 768]" = torch.ops.aten.permute.default(arg11_1, [1, 0])
    addmm_21: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg12_1, view_72, permute_36);  view_72 = permute_36 = None
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
    bmm_6: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_77, view_78);  view_77 = view_78 = None
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    clone_21: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_7);  div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_15: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_21, [4, 12, 512, 512]);  clone_21 = None
    view_80: "f32[48, 512, 512]" = torch.ops.aten.view.default(expand_15, [48, 512, 512]);  expand_15 = None
    expand_16: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_39, [4, 12, 512, 64]);  permute_39 = None
    clone_22: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
    view_81: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_22, [48, 512, 64]);  clone_22 = None
    bmm_7: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_80, view_81);  view_80 = view_81 = None
    view_82: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_7, [4, 12, 512, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    permute_41: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
    clone_23: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_41, memory_format = torch.contiguous_format);  permute_41 = None
    view_83: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_23, [4, 512, 768]);  clone_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_84: "f32[2048, 768]" = torch.ops.aten.view.default(view_83, [2048, 768]);  view_83 = None
    permute_42: "f32[768, 768]" = torch.ops.aten.permute.default(arg13_1, [1, 0])
    addmm_22: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg14_1, view_84, permute_42);  view_84 = permute_42 = None
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
    sub_12: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_32, getitem_15);  add_32 = getitem_15 = None
    mul_27: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_7);  sub_12 = rsqrt_7 = None
    mul_28: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_27, arg15_1);  mul_27 = None
    add_34: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_28, arg16_1);  mul_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_86: "f32[2048, 768]" = torch.ops.aten.view.default(add_34, [2048, 768])
    permute_43: "f32[768, 3072]" = torch.ops.aten.permute.default(arg17_1, [1, 0])
    addmm_23: "f32[2048, 3072]" = torch.ops.aten.addmm.default(arg18_1, view_86, permute_43);  view_86 = permute_43 = None
    view_87: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_23, [4, 512, 3072]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_29: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_87, 0.5)
    pow_4: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_87, 3.0)
    mul_30: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
    add_35: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(view_87, mul_30);  view_87 = mul_30 = None
    mul_31: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_35, 0.7978845608028654);  add_35 = None
    tanh_3: "f32[4, 512, 3072]" = torch.ops.aten.tanh.default(mul_31);  mul_31 = None
    add_36: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
    mul_32: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_29, add_36);  mul_29 = add_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_88: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_32, [2048, 3072]);  mul_32 = None
    permute_44: "f32[3072, 768]" = torch.ops.aten.permute.default(arg19_1, [1, 0])
    addmm_24: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg20_1, view_88, permute_44);  view_88 = permute_44 = None
    view_89: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_24, [4, 512, 768]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_37: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_89, add_34);  view_89 = add_34 = None
    var_mean_8 = torch.ops.aten.var_mean.correction(add_37, [2], correction = 0, keepdim = True)
    getitem_16: "f32[4, 512, 1]" = var_mean_8[0]
    getitem_17: "f32[4, 512, 1]" = var_mean_8[1];  var_mean_8 = None
    add_38: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
    rsqrt_8: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
    sub_13: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_37, getitem_17);  add_37 = getitem_17 = None
    mul_33: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_8);  sub_13 = rsqrt_8 = None
    mul_34: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_33, arg21_1);  mul_33 = None
    add_39: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_34, arg22_1);  mul_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_90: "f32[2048, 768]" = torch.ops.aten.view.default(add_39, [2048, 768])
    permute_45: "f32[768, 768]" = torch.ops.aten.permute.default(arg7_1, [1, 0])
    addmm_25: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg8_1, view_90, permute_45);  view_90 = permute_45 = None
    view_91: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_25, [4, 512, 768]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_92: "f32[2048, 768]" = torch.ops.aten.view.default(add_39, [2048, 768])
    permute_46: "f32[768, 768]" = torch.ops.aten.permute.default(arg9_1, [1, 0])
    addmm_26: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg10_1, view_92, permute_46);  view_92 = permute_46 = None
    view_93: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_26, [4, 512, 768]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_94: "f32[2048, 768]" = torch.ops.aten.view.default(add_39, [2048, 768])
    permute_47: "f32[768, 768]" = torch.ops.aten.permute.default(arg11_1, [1, 0])
    addmm_27: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg12_1, view_94, permute_47);  view_94 = permute_47 = None
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
    bmm_8: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_99, view_100);  view_99 = view_100 = None
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    clone_27: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_9);  div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_19: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_27, [4, 12, 512, 512]);  clone_27 = None
    view_102: "f32[48, 512, 512]" = torch.ops.aten.view.default(expand_19, [48, 512, 512]);  expand_19 = None
    expand_20: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_50, [4, 12, 512, 64]);  permute_50 = None
    clone_28: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
    view_103: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_28, [48, 512, 64]);  clone_28 = None
    bmm_9: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_102, view_103);  view_102 = view_103 = None
    view_104: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_9, [4, 12, 512, 64]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    permute_52: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None
    clone_29: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None
    view_105: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_29, [4, 512, 768]);  clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_106: "f32[2048, 768]" = torch.ops.aten.view.default(view_105, [2048, 768]);  view_105 = None
    permute_53: "f32[768, 768]" = torch.ops.aten.permute.default(arg13_1, [1, 0])
    addmm_28: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg14_1, view_106, permute_53);  view_106 = permute_53 = None
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
    sub_15: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_41, getitem_19);  add_41 = getitem_19 = None
    mul_35: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_9);  sub_15 = rsqrt_9 = None
    mul_36: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_35, arg15_1);  mul_35 = None
    add_43: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_36, arg16_1);  mul_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_108: "f32[2048, 768]" = torch.ops.aten.view.default(add_43, [2048, 768])
    permute_54: "f32[768, 3072]" = torch.ops.aten.permute.default(arg17_1, [1, 0])
    addmm_29: "f32[2048, 3072]" = torch.ops.aten.addmm.default(arg18_1, view_108, permute_54);  view_108 = permute_54 = None
    view_109: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_29, [4, 512, 3072]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_37: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_109, 0.5)
    pow_5: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_109, 3.0)
    mul_38: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
    add_44: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(view_109, mul_38);  view_109 = mul_38 = None
    mul_39: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_44, 0.7978845608028654);  add_44 = None
    tanh_4: "f32[4, 512, 3072]" = torch.ops.aten.tanh.default(mul_39);  mul_39 = None
    add_45: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
    mul_40: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_37, add_45);  mul_37 = add_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_110: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_40, [2048, 3072]);  mul_40 = None
    permute_55: "f32[3072, 768]" = torch.ops.aten.permute.default(arg19_1, [1, 0])
    addmm_30: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg20_1, view_110, permute_55);  view_110 = permute_55 = None
    view_111: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_30, [4, 512, 768]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_46: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_111, add_43);  view_111 = add_43 = None
    var_mean_10 = torch.ops.aten.var_mean.correction(add_46, [2], correction = 0, keepdim = True)
    getitem_20: "f32[4, 512, 1]" = var_mean_10[0]
    getitem_21: "f32[4, 512, 1]" = var_mean_10[1];  var_mean_10 = None
    add_47: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
    rsqrt_10: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
    sub_16: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_46, getitem_21);  add_46 = getitem_21 = None
    mul_41: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_10);  sub_16 = rsqrt_10 = None
    mul_42: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_41, arg21_1);  mul_41 = None
    add_48: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_42, arg22_1);  mul_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_112: "f32[2048, 768]" = torch.ops.aten.view.default(add_48, [2048, 768])
    permute_56: "f32[768, 768]" = torch.ops.aten.permute.default(arg7_1, [1, 0])
    addmm_31: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg8_1, view_112, permute_56);  view_112 = permute_56 = None
    view_113: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_31, [4, 512, 768]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_114: "f32[2048, 768]" = torch.ops.aten.view.default(add_48, [2048, 768])
    permute_57: "f32[768, 768]" = torch.ops.aten.permute.default(arg9_1, [1, 0])
    addmm_32: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg10_1, view_114, permute_57);  view_114 = permute_57 = None
    view_115: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_32, [4, 512, 768]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_116: "f32[2048, 768]" = torch.ops.aten.view.default(add_48, [2048, 768])
    permute_58: "f32[768, 768]" = torch.ops.aten.permute.default(arg11_1, [1, 0])
    addmm_33: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg12_1, view_116, permute_58);  view_116 = permute_58 = None
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
    bmm_10: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_121, view_122);  view_121 = view_122 = None
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    clone_33: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_11);  div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_23: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_33, [4, 12, 512, 512]);  clone_33 = None
    view_124: "f32[48, 512, 512]" = torch.ops.aten.view.default(expand_23, [48, 512, 512]);  expand_23 = None
    expand_24: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_61, [4, 12, 512, 64]);  permute_61 = None
    clone_34: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
    view_125: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_34, [48, 512, 64]);  clone_34 = None
    bmm_11: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_124, view_125);  view_124 = view_125 = None
    view_126: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_11, [4, 12, 512, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    permute_63: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
    clone_35: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format);  permute_63 = None
    view_127: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_35, [4, 512, 768]);  clone_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_128: "f32[2048, 768]" = torch.ops.aten.view.default(view_127, [2048, 768]);  view_127 = None
    permute_64: "f32[768, 768]" = torch.ops.aten.permute.default(arg13_1, [1, 0])
    addmm_34: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg14_1, view_128, permute_64);  view_128 = permute_64 = None
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
    sub_18: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_50, getitem_23);  add_50 = getitem_23 = None
    mul_43: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_11);  sub_18 = rsqrt_11 = None
    mul_44: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_43, arg15_1);  mul_43 = None
    add_52: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_44, arg16_1);  mul_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_130: "f32[2048, 768]" = torch.ops.aten.view.default(add_52, [2048, 768])
    permute_65: "f32[768, 3072]" = torch.ops.aten.permute.default(arg17_1, [1, 0])
    addmm_35: "f32[2048, 3072]" = torch.ops.aten.addmm.default(arg18_1, view_130, permute_65);  view_130 = permute_65 = None
    view_131: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_35, [4, 512, 3072]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_45: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_131, 0.5)
    pow_6: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_131, 3.0)
    mul_46: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
    add_53: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(view_131, mul_46);  view_131 = mul_46 = None
    mul_47: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_53, 0.7978845608028654);  add_53 = None
    tanh_5: "f32[4, 512, 3072]" = torch.ops.aten.tanh.default(mul_47);  mul_47 = None
    add_54: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
    mul_48: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_45, add_54);  mul_45 = add_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_132: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_48, [2048, 3072]);  mul_48 = None
    permute_66: "f32[3072, 768]" = torch.ops.aten.permute.default(arg19_1, [1, 0])
    addmm_36: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg20_1, view_132, permute_66);  view_132 = permute_66 = None
    view_133: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_36, [4, 512, 768]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_55: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_133, add_52);  view_133 = add_52 = None
    var_mean_12 = torch.ops.aten.var_mean.correction(add_55, [2], correction = 0, keepdim = True)
    getitem_24: "f32[4, 512, 1]" = var_mean_12[0]
    getitem_25: "f32[4, 512, 1]" = var_mean_12[1];  var_mean_12 = None
    add_56: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
    rsqrt_12: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    sub_19: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_55, getitem_25);  add_55 = getitem_25 = None
    mul_49: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_12);  sub_19 = rsqrt_12 = None
    mul_50: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_49, arg21_1);  mul_49 = None
    add_57: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_50, arg22_1);  mul_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_134: "f32[2048, 768]" = torch.ops.aten.view.default(add_57, [2048, 768])
    permute_67: "f32[768, 768]" = torch.ops.aten.permute.default(arg7_1, [1, 0])
    addmm_37: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg8_1, view_134, permute_67);  view_134 = permute_67 = None
    view_135: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_37, [4, 512, 768]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_136: "f32[2048, 768]" = torch.ops.aten.view.default(add_57, [2048, 768])
    permute_68: "f32[768, 768]" = torch.ops.aten.permute.default(arg9_1, [1, 0])
    addmm_38: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg10_1, view_136, permute_68);  view_136 = permute_68 = None
    view_137: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_38, [4, 512, 768]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_138: "f32[2048, 768]" = torch.ops.aten.view.default(add_57, [2048, 768])
    permute_69: "f32[768, 768]" = torch.ops.aten.permute.default(arg11_1, [1, 0])
    addmm_39: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg12_1, view_138, permute_69);  view_138 = permute_69 = None
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
    bmm_12: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_143, view_144);  view_143 = view_144 = None
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    clone_39: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_13);  div_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_27: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_39, [4, 12, 512, 512]);  clone_39 = None
    view_146: "f32[48, 512, 512]" = torch.ops.aten.view.default(expand_27, [48, 512, 512]);  expand_27 = None
    expand_28: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_72, [4, 12, 512, 64]);  permute_72 = None
    clone_40: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
    view_147: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_40, [48, 512, 64]);  clone_40 = None
    bmm_13: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_146, view_147);  view_146 = view_147 = None
    view_148: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_13, [4, 12, 512, 64]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    permute_74: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
    clone_41: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_74, memory_format = torch.contiguous_format);  permute_74 = None
    view_149: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_41, [4, 512, 768]);  clone_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_150: "f32[2048, 768]" = torch.ops.aten.view.default(view_149, [2048, 768]);  view_149 = None
    permute_75: "f32[768, 768]" = torch.ops.aten.permute.default(arg13_1, [1, 0])
    addmm_40: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg14_1, view_150, permute_75);  view_150 = permute_75 = None
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
    sub_21: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_59, getitem_27);  add_59 = getitem_27 = None
    mul_51: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_13);  sub_21 = rsqrt_13 = None
    mul_52: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_51, arg15_1);  mul_51 = None
    add_61: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_52, arg16_1);  mul_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_152: "f32[2048, 768]" = torch.ops.aten.view.default(add_61, [2048, 768])
    permute_76: "f32[768, 3072]" = torch.ops.aten.permute.default(arg17_1, [1, 0])
    addmm_41: "f32[2048, 3072]" = torch.ops.aten.addmm.default(arg18_1, view_152, permute_76);  view_152 = permute_76 = None
    view_153: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_41, [4, 512, 3072]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_53: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_153, 0.5)
    pow_7: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_153, 3.0)
    mul_54: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
    add_62: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(view_153, mul_54);  view_153 = mul_54 = None
    mul_55: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_62, 0.7978845608028654);  add_62 = None
    tanh_6: "f32[4, 512, 3072]" = torch.ops.aten.tanh.default(mul_55);  mul_55 = None
    add_63: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
    mul_56: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_53, add_63);  mul_53 = add_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_154: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_56, [2048, 3072]);  mul_56 = None
    permute_77: "f32[3072, 768]" = torch.ops.aten.permute.default(arg19_1, [1, 0])
    addmm_42: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg20_1, view_154, permute_77);  view_154 = permute_77 = None
    view_155: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_42, [4, 512, 768]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_64: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_155, add_61);  view_155 = add_61 = None
    var_mean_14 = torch.ops.aten.var_mean.correction(add_64, [2], correction = 0, keepdim = True)
    getitem_28: "f32[4, 512, 1]" = var_mean_14[0]
    getitem_29: "f32[4, 512, 1]" = var_mean_14[1];  var_mean_14 = None
    add_65: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
    rsqrt_14: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
    sub_22: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_64, getitem_29);  add_64 = getitem_29 = None
    mul_57: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_14);  sub_22 = rsqrt_14 = None
    mul_58: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_57, arg21_1);  mul_57 = None
    add_66: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_58, arg22_1);  mul_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_156: "f32[2048, 768]" = torch.ops.aten.view.default(add_66, [2048, 768])
    permute_78: "f32[768, 768]" = torch.ops.aten.permute.default(arg7_1, [1, 0])
    addmm_43: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg8_1, view_156, permute_78);  view_156 = permute_78 = None
    view_157: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_43, [4, 512, 768]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_158: "f32[2048, 768]" = torch.ops.aten.view.default(add_66, [2048, 768])
    permute_79: "f32[768, 768]" = torch.ops.aten.permute.default(arg9_1, [1, 0])
    addmm_44: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg10_1, view_158, permute_79);  view_158 = permute_79 = None
    view_159: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_44, [4, 512, 768]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_160: "f32[2048, 768]" = torch.ops.aten.view.default(add_66, [2048, 768])
    permute_80: "f32[768, 768]" = torch.ops.aten.permute.default(arg11_1, [1, 0])
    addmm_45: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg12_1, view_160, permute_80);  view_160 = permute_80 = None
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
    bmm_14: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_165, view_166);  view_165 = view_166 = None
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    clone_45: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_15);  div_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_31: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_45, [4, 12, 512, 512]);  clone_45 = None
    view_168: "f32[48, 512, 512]" = torch.ops.aten.view.default(expand_31, [48, 512, 512]);  expand_31 = None
    expand_32: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_83, [4, 12, 512, 64]);  permute_83 = None
    clone_46: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
    view_169: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_46, [48, 512, 64]);  clone_46 = None
    bmm_15: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_168, view_169);  view_168 = view_169 = None
    view_170: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_15, [4, 12, 512, 64]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    permute_85: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_170, [0, 2, 1, 3]);  view_170 = None
    clone_47: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_85, memory_format = torch.contiguous_format);  permute_85 = None
    view_171: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_47, [4, 512, 768]);  clone_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_172: "f32[2048, 768]" = torch.ops.aten.view.default(view_171, [2048, 768]);  view_171 = None
    permute_86: "f32[768, 768]" = torch.ops.aten.permute.default(arg13_1, [1, 0])
    addmm_46: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg14_1, view_172, permute_86);  view_172 = permute_86 = None
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
    sub_24: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_68, getitem_31);  add_68 = getitem_31 = None
    mul_59: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_15);  sub_24 = rsqrt_15 = None
    mul_60: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_59, arg15_1);  mul_59 = None
    add_70: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_60, arg16_1);  mul_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_174: "f32[2048, 768]" = torch.ops.aten.view.default(add_70, [2048, 768])
    permute_87: "f32[768, 3072]" = torch.ops.aten.permute.default(arg17_1, [1, 0])
    addmm_47: "f32[2048, 3072]" = torch.ops.aten.addmm.default(arg18_1, view_174, permute_87);  view_174 = permute_87 = None
    view_175: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_47, [4, 512, 3072]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_61: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_175, 0.5)
    pow_8: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_175, 3.0)
    mul_62: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
    add_71: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(view_175, mul_62);  view_175 = mul_62 = None
    mul_63: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_71, 0.7978845608028654);  add_71 = None
    tanh_7: "f32[4, 512, 3072]" = torch.ops.aten.tanh.default(mul_63);  mul_63 = None
    add_72: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
    mul_64: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_61, add_72);  mul_61 = add_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_176: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_64, [2048, 3072]);  mul_64 = None
    permute_88: "f32[3072, 768]" = torch.ops.aten.permute.default(arg19_1, [1, 0])
    addmm_48: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg20_1, view_176, permute_88);  view_176 = permute_88 = None
    view_177: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_48, [4, 512, 768]);  addmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_73: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_177, add_70);  view_177 = add_70 = None
    var_mean_16 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
    getitem_32: "f32[4, 512, 1]" = var_mean_16[0]
    getitem_33: "f32[4, 512, 1]" = var_mean_16[1];  var_mean_16 = None
    add_74: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
    rsqrt_16: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    sub_25: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_73, getitem_33);  add_73 = getitem_33 = None
    mul_65: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_16);  sub_25 = rsqrt_16 = None
    mul_66: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_65, arg21_1);  mul_65 = None
    add_75: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_66, arg22_1);  mul_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_178: "f32[2048, 768]" = torch.ops.aten.view.default(add_75, [2048, 768])
    permute_89: "f32[768, 768]" = torch.ops.aten.permute.default(arg7_1, [1, 0])
    addmm_49: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg8_1, view_178, permute_89);  view_178 = permute_89 = None
    view_179: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_49, [4, 512, 768]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_180: "f32[2048, 768]" = torch.ops.aten.view.default(add_75, [2048, 768])
    permute_90: "f32[768, 768]" = torch.ops.aten.permute.default(arg9_1, [1, 0])
    addmm_50: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg10_1, view_180, permute_90);  view_180 = permute_90 = None
    view_181: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_50, [4, 512, 768]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_182: "f32[2048, 768]" = torch.ops.aten.view.default(add_75, [2048, 768])
    permute_91: "f32[768, 768]" = torch.ops.aten.permute.default(arg11_1, [1, 0])
    addmm_51: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg12_1, view_182, permute_91);  view_182 = permute_91 = None
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
    bmm_16: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_187, view_188);  view_187 = view_188 = None
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    clone_51: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_17);  div_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_35: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_51, [4, 12, 512, 512]);  clone_51 = None
    view_190: "f32[48, 512, 512]" = torch.ops.aten.view.default(expand_35, [48, 512, 512]);  expand_35 = None
    expand_36: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_94, [4, 12, 512, 64]);  permute_94 = None
    clone_52: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
    view_191: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_52, [48, 512, 64]);  clone_52 = None
    bmm_17: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_190, view_191);  view_190 = view_191 = None
    view_192: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_17, [4, 12, 512, 64]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    permute_96: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None
    clone_53: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_96, memory_format = torch.contiguous_format);  permute_96 = None
    view_193: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_53, [4, 512, 768]);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_194: "f32[2048, 768]" = torch.ops.aten.view.default(view_193, [2048, 768]);  view_193 = None
    permute_97: "f32[768, 768]" = torch.ops.aten.permute.default(arg13_1, [1, 0])
    addmm_52: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg14_1, view_194, permute_97);  view_194 = permute_97 = None
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
    sub_27: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_77, getitem_35);  add_77 = getitem_35 = None
    mul_67: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_17);  sub_27 = rsqrt_17 = None
    mul_68: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_67, arg15_1);  mul_67 = None
    add_79: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_68, arg16_1);  mul_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_196: "f32[2048, 768]" = torch.ops.aten.view.default(add_79, [2048, 768])
    permute_98: "f32[768, 3072]" = torch.ops.aten.permute.default(arg17_1, [1, 0])
    addmm_53: "f32[2048, 3072]" = torch.ops.aten.addmm.default(arg18_1, view_196, permute_98);  view_196 = permute_98 = None
    view_197: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_53, [4, 512, 3072]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_69: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_197, 0.5)
    pow_9: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_197, 3.0)
    mul_70: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
    add_80: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(view_197, mul_70);  view_197 = mul_70 = None
    mul_71: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_80, 0.7978845608028654);  add_80 = None
    tanh_8: "f32[4, 512, 3072]" = torch.ops.aten.tanh.default(mul_71);  mul_71 = None
    add_81: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
    mul_72: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_69, add_81);  mul_69 = add_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_198: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_72, [2048, 3072]);  mul_72 = None
    permute_99: "f32[3072, 768]" = torch.ops.aten.permute.default(arg19_1, [1, 0])
    addmm_54: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg20_1, view_198, permute_99);  view_198 = permute_99 = None
    view_199: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_54, [4, 512, 768]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_82: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_199, add_79);  view_199 = add_79 = None
    var_mean_18 = torch.ops.aten.var_mean.correction(add_82, [2], correction = 0, keepdim = True)
    getitem_36: "f32[4, 512, 1]" = var_mean_18[0]
    getitem_37: "f32[4, 512, 1]" = var_mean_18[1];  var_mean_18 = None
    add_83: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
    rsqrt_18: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
    sub_28: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_82, getitem_37);  add_82 = getitem_37 = None
    mul_73: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_18);  sub_28 = rsqrt_18 = None
    mul_74: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_73, arg21_1);  mul_73 = None
    add_84: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_74, arg22_1);  mul_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_200: "f32[2048, 768]" = torch.ops.aten.view.default(add_84, [2048, 768])
    permute_100: "f32[768, 768]" = torch.ops.aten.permute.default(arg7_1, [1, 0])
    addmm_55: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg8_1, view_200, permute_100);  view_200 = permute_100 = None
    view_201: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_55, [4, 512, 768]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_202: "f32[2048, 768]" = torch.ops.aten.view.default(add_84, [2048, 768])
    permute_101: "f32[768, 768]" = torch.ops.aten.permute.default(arg9_1, [1, 0])
    addmm_56: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg10_1, view_202, permute_101);  view_202 = permute_101 = None
    view_203: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_56, [4, 512, 768]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_204: "f32[2048, 768]" = torch.ops.aten.view.default(add_84, [2048, 768])
    permute_102: "f32[768, 768]" = torch.ops.aten.permute.default(arg11_1, [1, 0])
    addmm_57: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg12_1, view_204, permute_102);  view_204 = permute_102 = None
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
    bmm_18: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_209, view_210);  view_209 = view_210 = None
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    clone_57: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_19);  div_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_39: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_57, [4, 12, 512, 512]);  clone_57 = None
    view_212: "f32[48, 512, 512]" = torch.ops.aten.view.default(expand_39, [48, 512, 512]);  expand_39 = None
    expand_40: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_105, [4, 12, 512, 64]);  permute_105 = None
    clone_58: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
    view_213: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_58, [48, 512, 64]);  clone_58 = None
    bmm_19: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_212, view_213);  view_212 = view_213 = None
    view_214: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_19, [4, 12, 512, 64]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    permute_107: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_214, [0, 2, 1, 3]);  view_214 = None
    clone_59: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_107, memory_format = torch.contiguous_format);  permute_107 = None
    view_215: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_59, [4, 512, 768]);  clone_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_216: "f32[2048, 768]" = torch.ops.aten.view.default(view_215, [2048, 768]);  view_215 = None
    permute_108: "f32[768, 768]" = torch.ops.aten.permute.default(arg13_1, [1, 0])
    addmm_58: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg14_1, view_216, permute_108);  view_216 = permute_108 = None
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
    sub_30: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_86, getitem_39);  add_86 = getitem_39 = None
    mul_75: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_19);  sub_30 = rsqrt_19 = None
    mul_76: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_75, arg15_1);  mul_75 = None
    add_88: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_76, arg16_1);  mul_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_218: "f32[2048, 768]" = torch.ops.aten.view.default(add_88, [2048, 768])
    permute_109: "f32[768, 3072]" = torch.ops.aten.permute.default(arg17_1, [1, 0])
    addmm_59: "f32[2048, 3072]" = torch.ops.aten.addmm.default(arg18_1, view_218, permute_109);  view_218 = permute_109 = None
    view_219: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_59, [4, 512, 3072]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_77: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_219, 0.5)
    pow_10: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_219, 3.0)
    mul_78: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
    add_89: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(view_219, mul_78);  view_219 = mul_78 = None
    mul_79: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_89, 0.7978845608028654);  add_89 = None
    tanh_9: "f32[4, 512, 3072]" = torch.ops.aten.tanh.default(mul_79);  mul_79 = None
    add_90: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
    mul_80: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_77, add_90);  mul_77 = add_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_220: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_80, [2048, 3072]);  mul_80 = None
    permute_110: "f32[3072, 768]" = torch.ops.aten.permute.default(arg19_1, [1, 0])
    addmm_60: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg20_1, view_220, permute_110);  view_220 = permute_110 = None
    view_221: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_60, [4, 512, 768]);  addmm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_91: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_221, add_88);  view_221 = add_88 = None
    var_mean_20 = torch.ops.aten.var_mean.correction(add_91, [2], correction = 0, keepdim = True)
    getitem_40: "f32[4, 512, 1]" = var_mean_20[0]
    getitem_41: "f32[4, 512, 1]" = var_mean_20[1];  var_mean_20 = None
    add_92: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
    rsqrt_20: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
    sub_31: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_91, getitem_41);  add_91 = getitem_41 = None
    mul_81: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_20);  sub_31 = rsqrt_20 = None
    mul_82: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_81, arg21_1);  mul_81 = None
    add_93: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_82, arg22_1);  mul_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_222: "f32[2048, 768]" = torch.ops.aten.view.default(add_93, [2048, 768])
    permute_111: "f32[768, 768]" = torch.ops.aten.permute.default(arg7_1, [1, 0])
    addmm_61: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg8_1, view_222, permute_111);  view_222 = permute_111 = None
    view_223: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_61, [4, 512, 768]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_224: "f32[2048, 768]" = torch.ops.aten.view.default(add_93, [2048, 768])
    permute_112: "f32[768, 768]" = torch.ops.aten.permute.default(arg9_1, [1, 0])
    addmm_62: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg10_1, view_224, permute_112);  view_224 = permute_112 = None
    view_225: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_62, [4, 512, 768]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_226: "f32[2048, 768]" = torch.ops.aten.view.default(add_93, [2048, 768])
    permute_113: "f32[768, 768]" = torch.ops.aten.permute.default(arg11_1, [1, 0])
    addmm_63: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg12_1, view_226, permute_113);  view_226 = permute_113 = None
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
    bmm_20: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_231, view_232);  view_231 = view_232 = None
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    clone_63: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_21);  div_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_43: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_63, [4, 12, 512, 512]);  clone_63 = None
    view_234: "f32[48, 512, 512]" = torch.ops.aten.view.default(expand_43, [48, 512, 512]);  expand_43 = None
    expand_44: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_116, [4, 12, 512, 64]);  permute_116 = None
    clone_64: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
    view_235: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_64, [48, 512, 64]);  clone_64 = None
    bmm_21: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_234, view_235);  view_234 = view_235 = None
    view_236: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_21, [4, 12, 512, 64]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    permute_118: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_236, [0, 2, 1, 3]);  view_236 = None
    clone_65: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_118, memory_format = torch.contiguous_format);  permute_118 = None
    view_237: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_65, [4, 512, 768]);  clone_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_238: "f32[2048, 768]" = torch.ops.aten.view.default(view_237, [2048, 768]);  view_237 = None
    permute_119: "f32[768, 768]" = torch.ops.aten.permute.default(arg13_1, [1, 0])
    addmm_64: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg14_1, view_238, permute_119);  view_238 = permute_119 = None
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
    sub_33: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_95, getitem_43);  add_95 = getitem_43 = None
    mul_83: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_21);  sub_33 = rsqrt_21 = None
    mul_84: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_83, arg15_1);  mul_83 = None
    add_97: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_84, arg16_1);  mul_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_240: "f32[2048, 768]" = torch.ops.aten.view.default(add_97, [2048, 768])
    permute_120: "f32[768, 3072]" = torch.ops.aten.permute.default(arg17_1, [1, 0])
    addmm_65: "f32[2048, 3072]" = torch.ops.aten.addmm.default(arg18_1, view_240, permute_120);  view_240 = permute_120 = None
    view_241: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_65, [4, 512, 3072]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_85: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_241, 0.5)
    pow_11: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_241, 3.0)
    mul_86: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
    add_98: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(view_241, mul_86);  view_241 = mul_86 = None
    mul_87: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_98, 0.7978845608028654);  add_98 = None
    tanh_10: "f32[4, 512, 3072]" = torch.ops.aten.tanh.default(mul_87);  mul_87 = None
    add_99: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
    mul_88: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_85, add_99);  mul_85 = add_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_242: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_88, [2048, 3072]);  mul_88 = None
    permute_121: "f32[3072, 768]" = torch.ops.aten.permute.default(arg19_1, [1, 0])
    addmm_66: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg20_1, view_242, permute_121);  view_242 = permute_121 = None
    view_243: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_66, [4, 512, 768]);  addmm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_100: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_243, add_97);  view_243 = add_97 = None
    var_mean_22 = torch.ops.aten.var_mean.correction(add_100, [2], correction = 0, keepdim = True)
    getitem_44: "f32[4, 512, 1]" = var_mean_22[0]
    getitem_45: "f32[4, 512, 1]" = var_mean_22[1];  var_mean_22 = None
    add_101: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
    rsqrt_22: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
    sub_34: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_100, getitem_45);  add_100 = getitem_45 = None
    mul_89: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_22);  sub_34 = rsqrt_22 = None
    mul_90: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_89, arg21_1);  mul_89 = None
    add_102: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_90, arg22_1);  mul_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:322, code: mixed_query_layer = self.query(hidden_states)
    view_244: "f32[2048, 768]" = torch.ops.aten.view.default(add_102, [2048, 768])
    permute_122: "f32[768, 768]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
    addmm_67: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg8_1, view_244, permute_122);  arg8_1 = view_244 = permute_122 = None
    view_245: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_67, [4, 512, 768]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:323, code: mixed_key_layer = self.key(hidden_states)
    view_246: "f32[2048, 768]" = torch.ops.aten.view.default(add_102, [2048, 768])
    permute_123: "f32[768, 768]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
    addmm_68: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg10_1, view_246, permute_123);  arg10_1 = view_246 = permute_123 = None
    view_247: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_68, [4, 512, 768]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:324, code: mixed_value_layer = self.value(hidden_states)
    view_248: "f32[2048, 768]" = torch.ops.aten.view.default(add_102, [2048, 768])
    permute_124: "f32[768, 768]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
    addmm_69: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg12_1, view_248, permute_124);  arg12_1 = view_248 = permute_124 = None
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
    bmm_22: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_253, view_254);  view_253 = view_254 = None
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:359, code: attention_probs = self.attention_dropout(attention_probs)
    clone_69: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_23);  div_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_47: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_69, [4, 12, 512, 512]);  clone_69 = None
    view_256: "f32[48, 512, 512]" = torch.ops.aten.view.default(expand_47, [48, 512, 512]);  expand_47 = None
    expand_48: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_127, [4, 12, 512, 64]);  permute_127 = None
    clone_70: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
    view_257: "f32[48, 512, 64]" = torch.ops.aten.view.default(clone_70, [48, 512, 64]);  clone_70 = None
    bmm_23: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_256, view_257);  view_256 = view_257 = None
    view_258: "f32[4, 12, 512, 64]" = torch.ops.aten.view.default(bmm_23, [4, 12, 512, 64]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:366, code: context_layer = context_layer.transpose(2, 1).flatten(2)
    permute_129: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_258, [0, 2, 1, 3]);  view_258 = None
    clone_71: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None
    view_259: "f32[4, 512, 768]" = torch.ops.aten.view.default(clone_71, [4, 512, 768]);  clone_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:368, code: projected_context_layer = self.dense(context_layer)
    view_260: "f32[2048, 768]" = torch.ops.aten.view.default(view_259, [2048, 768]);  view_259 = None
    permute_130: "f32[768, 768]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
    addmm_70: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg14_1, view_260, permute_130);  arg14_1 = view_260 = permute_130 = None
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
    sub_36: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_104, getitem_47);  add_104 = getitem_47 = None
    mul_91: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_23);  sub_36 = rsqrt_23 = None
    mul_92: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_91, arg15_1);  mul_91 = arg15_1 = None
    add_106: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_92, arg16_1);  mul_92 = arg16_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:409, code: ffn_output = self.ffn(attention_output)
    view_262: "f32[2048, 768]" = torch.ops.aten.view.default(add_106, [2048, 768])
    permute_131: "f32[768, 3072]" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
    addmm_71: "f32[2048, 3072]" = torch.ops.aten.addmm.default(arg18_1, view_262, permute_131);  arg18_1 = view_262 = permute_131 = None
    view_263: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_71, [4, 512, 3072]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_93: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_263, 0.5)
    pow_12: "f32[4, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_263, 3.0)
    mul_94: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
    add_107: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(view_263, mul_94);  view_263 = mul_94 = None
    mul_95: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_107, 0.7978845608028654);  add_107 = None
    tanh_11: "f32[4, 512, 3072]" = torch.ops.aten.tanh.default(mul_95);  mul_95 = None
    add_108: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
    mul_96: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_93, add_108);  mul_93 = add_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:411, code: ffn_output = self.ffn_output(ffn_output)
    view_264: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_96, [2048, 3072]);  mul_96 = None
    permute_132: "f32[3072, 768]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
    addmm_72: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg20_1, view_264, permute_132);  arg20_1 = view_264 = permute_132 = None
    view_265: "f32[4, 512, 768]" = torch.ops.aten.view.default(addmm_72, [4, 512, 768]);  addmm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
    add_109: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_265, add_106);  view_265 = add_106 = None
    var_mean_24 = torch.ops.aten.var_mean.correction(add_109, [2], correction = 0, keepdim = True)
    getitem_48: "f32[4, 512, 1]" = var_mean_24[0]
    getitem_49: "f32[4, 512, 1]" = var_mean_24[1];  var_mean_24 = None
    add_110: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
    rsqrt_24: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
    sub_37: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_109, getitem_49);  add_109 = getitem_49 = None
    mul_97: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_24);  sub_37 = rsqrt_24 = None
    mul_98: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_97, arg21_1);  mul_97 = arg21_1 = None
    add_111: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_98, arg22_1);  mul_98 = arg22_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:880, code: hidden_states = self.dense(hidden_states)
    view_266: "f32[2048, 768]" = torch.ops.aten.view.default(add_111, [2048, 768]);  add_111 = None
    permute_133: "f32[768, 128]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
    addmm_73: "f32[2048, 128]" = torch.ops.aten.addmm.default(arg24_1, view_266, permute_133);  arg24_1 = view_266 = permute_133 = None
    view_267: "f32[4, 512, 128]" = torch.ops.aten.view.default(addmm_73, [4, 512, 128]);  addmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_99: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(view_267, 0.5)
    pow_13: "f32[4, 512, 128]" = torch.ops.aten.pow.Tensor_Scalar(view_267, 3.0)
    mul_100: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(pow_13, 0.044715);  pow_13 = None
    add_112: "f32[4, 512, 128]" = torch.ops.aten.add.Tensor(view_267, mul_100);  view_267 = mul_100 = None
    mul_101: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(add_112, 0.7978845608028654);  add_112 = None
    tanh_12: "f32[4, 512, 128]" = torch.ops.aten.tanh.default(mul_101);  mul_101 = None
    add_113: "f32[4, 512, 128]" = torch.ops.aten.add.Tensor(tanh_12, 1.0);  tanh_12 = None
    mul_102: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(mul_99, add_113);  mul_99 = add_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:882, code: hidden_states = self.LayerNorm(hidden_states)
    var_mean_25 = torch.ops.aten.var_mean.correction(mul_102, [2], correction = 0, keepdim = True)
    getitem_50: "f32[4, 512, 1]" = var_mean_25[0]
    getitem_51: "f32[4, 512, 1]" = var_mean_25[1];  var_mean_25 = None
    add_114: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
    rsqrt_25: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
    sub_38: "f32[4, 512, 128]" = torch.ops.aten.sub.Tensor(mul_102, getitem_51);  mul_102 = getitem_51 = None
    mul_103: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_25);  sub_38 = rsqrt_25 = None
    mul_104: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(mul_103, arg25_1);  mul_103 = arg25_1 = None
    add_115: "f32[4, 512, 128]" = torch.ops.aten.add.Tensor(mul_104, arg26_1);  mul_104 = arg26_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:883, code: hidden_states = self.decoder(hidden_states)
    view_268: "f32[2048, 128]" = torch.ops.aten.view.default(add_115, [2048, 128]);  add_115 = None
    permute_134: "f32[128, 30000]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
    addmm_74: "f32[2048, 30000]" = torch.ops.aten.addmm.default(arg28_1, view_268, permute_134);  arg28_1 = view_268 = permute_134 = None
    view_269: "f32[4, 512, 30000]" = torch.ops.aten.view.default(addmm_74, [4, 512, 30000]);  addmm_74 = None
    return (view_269,)
    