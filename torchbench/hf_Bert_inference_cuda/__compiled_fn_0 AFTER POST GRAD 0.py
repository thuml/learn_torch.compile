from __future__ import annotations



def forward(self, arg0_1: "f32[30522, 768]", arg1_1: "f32[2, 768]", arg2_1: "f32[512, 768]", arg3_1: "f32[768]", arg4_1: "f32[768]", arg5_1: "f32[768, 768]", arg6_1: "f32[768]", arg7_1: "f32[768, 768]", arg8_1: "f32[768]", arg9_1: "f32[768, 768]", arg10_1: "f32[768]", arg11_1: "f32[768, 768]", arg12_1: "f32[768]", arg13_1: "f32[768]", arg14_1: "f32[768]", arg15_1: "f32[3072, 768]", arg16_1: "f32[3072]", arg17_1: "f32[768, 3072]", arg18_1: "f32[768]", arg19_1: "f32[768]", arg20_1: "f32[768]", arg21_1: "f32[768, 768]", arg22_1: "f32[768]", arg23_1: "f32[768, 768]", arg24_1: "f32[768]", arg25_1: "f32[768, 768]", arg26_1: "f32[768]", arg27_1: "f32[768, 768]", arg28_1: "f32[768]", arg29_1: "f32[768]", arg30_1: "f32[768]", arg31_1: "f32[3072, 768]", arg32_1: "f32[3072]", arg33_1: "f32[768, 3072]", arg34_1: "f32[768]", arg35_1: "f32[768]", arg36_1: "f32[768]", arg37_1: "f32[768, 768]", arg38_1: "f32[768]", arg39_1: "f32[768, 768]", arg40_1: "f32[768]", arg41_1: "f32[768, 768]", arg42_1: "f32[768]", arg43_1: "f32[768, 768]", arg44_1: "f32[768]", arg45_1: "f32[768]", arg46_1: "f32[768]", arg47_1: "f32[3072, 768]", arg48_1: "f32[3072]", arg49_1: "f32[768, 3072]", arg50_1: "f32[768]", arg51_1: "f32[768]", arg52_1: "f32[768]", arg53_1: "f32[768, 768]", arg54_1: "f32[768]", arg55_1: "f32[768, 768]", arg56_1: "f32[768]", arg57_1: "f32[768, 768]", arg58_1: "f32[768]", arg59_1: "f32[768, 768]", arg60_1: "f32[768]", arg61_1: "f32[768]", arg62_1: "f32[768]", arg63_1: "f32[3072, 768]", arg64_1: "f32[3072]", arg65_1: "f32[768, 3072]", arg66_1: "f32[768]", arg67_1: "f32[768]", arg68_1: "f32[768]", arg69_1: "f32[768, 768]", arg70_1: "f32[768]", arg71_1: "f32[768, 768]", arg72_1: "f32[768]", arg73_1: "f32[768, 768]", arg74_1: "f32[768]", arg75_1: "f32[768, 768]", arg76_1: "f32[768]", arg77_1: "f32[768]", arg78_1: "f32[768]", arg79_1: "f32[3072, 768]", arg80_1: "f32[3072]", arg81_1: "f32[768, 3072]", arg82_1: "f32[768]", arg83_1: "f32[768]", arg84_1: "f32[768]", arg85_1: "f32[768, 768]", arg86_1: "f32[768]", arg87_1: "f32[768, 768]", arg88_1: "f32[768]", arg89_1: "f32[768, 768]", arg90_1: "f32[768]", arg91_1: "f32[768, 768]", arg92_1: "f32[768]", arg93_1: "f32[768]", arg94_1: "f32[768]", arg95_1: "f32[3072, 768]", arg96_1: "f32[3072]", arg97_1: "f32[768, 3072]", arg98_1: "f32[768]", arg99_1: "f32[768]", arg100_1: "f32[768]", arg101_1: "f32[768, 768]", arg102_1: "f32[768]", arg103_1: "f32[768, 768]", arg104_1: "f32[768]", arg105_1: "f32[768, 768]", arg106_1: "f32[768]", arg107_1: "f32[768, 768]", arg108_1: "f32[768]", arg109_1: "f32[768]", arg110_1: "f32[768]", arg111_1: "f32[3072, 768]", arg112_1: "f32[3072]", arg113_1: "f32[768, 3072]", arg114_1: "f32[768]", arg115_1: "f32[768]", arg116_1: "f32[768]", arg117_1: "f32[768, 768]", arg118_1: "f32[768]", arg119_1: "f32[768, 768]", arg120_1: "f32[768]", arg121_1: "f32[768, 768]", arg122_1: "f32[768]", arg123_1: "f32[768, 768]", arg124_1: "f32[768]", arg125_1: "f32[768]", arg126_1: "f32[768]", arg127_1: "f32[3072, 768]", arg128_1: "f32[3072]", arg129_1: "f32[768, 3072]", arg130_1: "f32[768]", arg131_1: "f32[768]", arg132_1: "f32[768]", arg133_1: "f32[768, 768]", arg134_1: "f32[768]", arg135_1: "f32[768, 768]", arg136_1: "f32[768]", arg137_1: "f32[768, 768]", arg138_1: "f32[768]", arg139_1: "f32[768, 768]", arg140_1: "f32[768]", arg141_1: "f32[768]", arg142_1: "f32[768]", arg143_1: "f32[3072, 768]", arg144_1: "f32[3072]", arg145_1: "f32[768, 3072]", arg146_1: "f32[768]", arg147_1: "f32[768]", arg148_1: "f32[768]", arg149_1: "f32[768, 768]", arg150_1: "f32[768]", arg151_1: "f32[768, 768]", arg152_1: "f32[768]", arg153_1: "f32[768, 768]", arg154_1: "f32[768]", arg155_1: "f32[768, 768]", arg156_1: "f32[768]", arg157_1: "f32[768]", arg158_1: "f32[768]", arg159_1: "f32[3072, 768]", arg160_1: "f32[3072]", arg161_1: "f32[768, 3072]", arg162_1: "f32[768]", arg163_1: "f32[768]", arg164_1: "f32[768]", arg165_1: "f32[768, 768]", arg166_1: "f32[768]", arg167_1: "f32[768, 768]", arg168_1: "f32[768]", arg169_1: "f32[768, 768]", arg170_1: "f32[768]", arg171_1: "f32[768, 768]", arg172_1: "f32[768]", arg173_1: "f32[768]", arg174_1: "f32[768]", arg175_1: "f32[3072, 768]", arg176_1: "f32[3072]", arg177_1: "f32[768, 3072]", arg178_1: "f32[768]", arg179_1: "f32[768]", arg180_1: "f32[768]", arg181_1: "f32[768, 768]", arg182_1: "f32[768]", arg183_1: "f32[768, 768]", arg184_1: "f32[768]", arg185_1: "f32[768, 768]", arg186_1: "f32[768]", arg187_1: "f32[768, 768]", arg188_1: "f32[768]", arg189_1: "f32[768]", arg190_1: "f32[768]", arg191_1: "f32[3072, 768]", arg192_1: "f32[3072]", arg193_1: "f32[768, 3072]", arg194_1: "f32[768]", arg195_1: "f32[768]", arg196_1: "f32[768]", arg197_1: "f32[768, 768]", arg198_1: "f32[768]", arg199_1: "f32[768]", arg200_1: "f32[768]", arg201_1: "f32[30522, 768]", arg202_1: "f32[30522]", arg203_1: "i64[1, 512]", arg204_1: "i64[1, 512]", arg205_1: "i64[4, 512]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:983, code: attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
    full: "f32[4, 512]" = torch.ops.aten.full.default([4, 512], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:916, code: extended_attention_mask = attention_mask[:, None, None, :]
    unsqueeze: "f32[4, 1, 512]" = torch.ops.aten.unsqueeze.default(full, 1);  full = None
    unsqueeze_1: "f32[4, 1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:928, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    sub: "f32[4, 1, 1, 512]" = torch.ops.aten.sub.Tensor(1.0, unsqueeze_1);  unsqueeze_1 = None
    full_default: "f32[4, 1, 1, 512]" = torch.ops.aten.full.default([4, 1, 1, 512], -0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:232, code: inputs_embeds = self.word_embeddings(input_ids)
    embedding: "f32[4, 512, 768]" = torch.ops.aten.embedding.default(arg0_1, arg205_1, 0);  arg0_1 = arg205_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:988, code: buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
    expand: "i64[4, 512]" = torch.ops.aten.expand.default(arg203_1, [4, 512]);  arg203_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:233, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    embedding_1: "f32[4, 512, 768]" = torch.ops.aten.embedding.default(arg1_1, expand);  arg1_1 = expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:235, code: embeddings = inputs_embeds + token_type_embeddings
    add: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:237, code: position_embeddings = self.position_embeddings(position_ids)
    embedding_2: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg2_1, arg204_1);  arg2_1 = arg204_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:238, code: embeddings += position_embeddings
    add_1: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:239, code: embeddings = self.LayerNorm(embeddings)
    var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
    getitem: "f32[4, 512, 1]" = var_mean[0]
    getitem_1: "f32[4, 512, 1]" = var_mean[1];  var_mean = None
    sub_1: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
    add_2: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
    rsqrt: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
    mul_1: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = rsqrt = None
    mul_2: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
    add_3: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_2, arg4_1);  mul_2 = arg4_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_3, [2048, 768])
    permute: "f32[768, 768]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
    addmm: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg6_1, view, permute);  arg6_1 = view = permute = None
    view_1: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm, [4, 512, 768]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_8: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_1, [4, 512, 12, 64]);  view_1 = None
    
    # No stacktrace found for following nodes
    permute_default_33: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_2: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_3, [2048, 768])
    permute_1: "f32[768, 768]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
    addmm_1: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg8_1, view_2, permute_1);  arg8_1 = view_2 = permute_1 = None
    view_3: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_1, [4, 512, 768]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_4: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_3, [4, 512, 12, 64]);  view_3 = None
    
    # No stacktrace found for following nodes
    permute_default_34: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_5: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_3, [2048, 768])
    permute_3: "f32[768, 768]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
    addmm_2: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg10_1, view_5, permute_3);  arg10_1 = view_5 = permute_3 = None
    view_6: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_2, [4, 512, 768]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_7: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_6, [4, 512, 12, 64]);  view_6 = None
    
    # No stacktrace found for following nodes
    permute_default_35: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
    _scaled_dot_product_efficient_attention_default_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_33, permute_default_34, permute_default_35, None, True, scale = 0.125);  permute_default_33 = permute_default_34 = permute_default_35 = None
    getitem_63: "f32[4, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default_11[0];  _scaled_dot_product_efficient_attention_default_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_7: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_63, [0, 2, 1, 3]);  getitem_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_15: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(permute_7, [4, 512, 768]);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_16: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_15, [2048, 768]);  view_15 = None
    permute_8: "f32[768, 768]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
    
    # No stacktrace found for following nodes
    mm_default_36: "f32[2048, 768]" = torch.ops.aten.mm.default(view_16, permute_8);  view_16 = permute_8 = None
    add_tensor_36: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_36, arg12_1);  mm_default_36 = arg12_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_17: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_36, [4, 512, 768]);  add_tensor_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_5: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_17, add_3);  view_17 = add_3 = None
    var_mean_1 = torch.ops.aten.var_mean.correction(add_5, [2], correction = 0, keepdim = True)
    getitem_2: "f32[4, 512, 1]" = var_mean_1[0]
    getitem_3: "f32[4, 512, 1]" = var_mean_1[1];  var_mean_1 = None
    sub_3: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_5, getitem_3);  add_5 = getitem_3 = None
    add_6: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
    rsqrt_1: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    mul_3: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = rsqrt_1 = None
    mul_4: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_3, arg13_1);  mul_3 = arg13_1 = None
    add_7: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_4, arg14_1);  mul_4 = arg14_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_18: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_7, [2048, 768])
    permute_9: "f32[768, 3072]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
    
    # No stacktrace found for following nodes
    mm_default_35: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_18, permute_9);  view_18 = permute_9 = None
    add_tensor_35: "f32[2048, 3072]" = torch.ops.aten.add.Tensor(mm_default_35, arg16_1);  mm_default_35 = arg16_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_19: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(add_tensor_35, [4, 512, 3072]);  add_tensor_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_5: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.5)
    mul_6: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476);  view_19 = None
    erf: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
    add_8: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_7: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_5, add_8);  mul_5 = add_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_20: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_7, [2048, 3072]);  mul_7 = None
    permute_10: "f32[3072, 768]" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
    
    # No stacktrace found for following nodes
    mm_default_34: "f32[2048, 768]" = torch.ops.aten.mm.default(view_20, permute_10);  view_20 = permute_10 = None
    add_tensor_34: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_34, arg18_1);  mm_default_34 = arg18_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_21: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_34, [4, 512, 768]);  add_tensor_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_9: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_21, add_7);  view_21 = add_7 = None
    var_mean_2 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
    getitem_4: "f32[4, 512, 1]" = var_mean_2[0]
    getitem_5: "f32[4, 512, 1]" = var_mean_2[1];  var_mean_2 = None
    sub_4: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_9, getitem_5);  add_9 = getitem_5 = None
    add_10: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
    rsqrt_2: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
    mul_8: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = rsqrt_2 = None
    mul_9: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_8, arg19_1);  mul_8 = arg19_1 = None
    add_11: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_9, arg20_1);  mul_9 = arg20_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_22: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_11, [2048, 768])
    permute_11: "f32[768, 768]" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
    addmm_6: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg22_1, view_22, permute_11);  arg22_1 = view_22 = permute_11 = None
    view_23: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_6, [4, 512, 768]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_30: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_23, [4, 512, 12, 64]);  view_23 = None
    
    # No stacktrace found for following nodes
    permute_default_30: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_24: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_11, [2048, 768])
    permute_12: "f32[768, 768]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
    addmm_7: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg24_1, view_24, permute_12);  arg24_1 = view_24 = permute_12 = None
    view_25: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_7, [4, 512, 768]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_26: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_25, [4, 512, 12, 64]);  view_25 = None
    
    # No stacktrace found for following nodes
    permute_default_31: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_27: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_11, [2048, 768])
    permute_14: "f32[768, 768]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
    addmm_8: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg26_1, view_27, permute_14);  arg26_1 = view_27 = permute_14 = None
    view_28: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_8, [4, 512, 768]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_29: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_28, [4, 512, 12, 64]);  view_28 = None
    
    # No stacktrace found for following nodes
    permute_default_32: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
    _scaled_dot_product_efficient_attention_default_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_30, permute_default_31, permute_default_32, None, True, scale = 0.125);  permute_default_30 = permute_default_31 = permute_default_32 = None
    getitem_62: "f32[4, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default_10[0];  _scaled_dot_product_efficient_attention_default_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_18: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_62, [0, 2, 1, 3]);  getitem_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_37: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(permute_18, [4, 512, 768]);  permute_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_38: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_37, [2048, 768]);  view_37 = None
    permute_19: "f32[768, 768]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
    
    # No stacktrace found for following nodes
    mm_default_33: "f32[2048, 768]" = torch.ops.aten.mm.default(view_38, permute_19);  view_38 = permute_19 = None
    add_tensor_33: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_33, arg28_1);  mm_default_33 = arg28_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_39: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_33, [4, 512, 768]);  add_tensor_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_13: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_39, add_11);  view_39 = add_11 = None
    var_mean_3 = torch.ops.aten.var_mean.correction(add_13, [2], correction = 0, keepdim = True)
    getitem_6: "f32[4, 512, 1]" = var_mean_3[0]
    getitem_7: "f32[4, 512, 1]" = var_mean_3[1];  var_mean_3 = None
    sub_6: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_13, getitem_7);  add_13 = getitem_7 = None
    add_14: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
    rsqrt_3: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
    mul_10: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_3);  sub_6 = rsqrt_3 = None
    mul_11: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_10, arg29_1);  mul_10 = arg29_1 = None
    add_15: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_11, arg30_1);  mul_11 = arg30_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_40: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_15, [2048, 768])
    permute_20: "f32[768, 3072]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
    
    # No stacktrace found for following nodes
    mm_default_32: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_40, permute_20);  view_40 = permute_20 = None
    add_tensor_32: "f32[2048, 3072]" = torch.ops.aten.add.Tensor(mm_default_32, arg32_1);  mm_default_32 = arg32_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_41: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(add_tensor_32, [4, 512, 3072]);  add_tensor_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_12: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, 0.5)
    mul_13: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476);  view_41 = None
    erf_1: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
    add_16: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_14: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_12, add_16);  mul_12 = add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_42: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_14, [2048, 3072]);  mul_14 = None
    permute_21: "f32[3072, 768]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
    
    # No stacktrace found for following nodes
    mm_default_31: "f32[2048, 768]" = torch.ops.aten.mm.default(view_42, permute_21);  view_42 = permute_21 = None
    add_tensor_31: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_31, arg34_1);  mm_default_31 = arg34_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_43: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_31, [4, 512, 768]);  add_tensor_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_17: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_43, add_15);  view_43 = add_15 = None
    var_mean_4 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
    getitem_8: "f32[4, 512, 1]" = var_mean_4[0]
    getitem_9: "f32[4, 512, 1]" = var_mean_4[1];  var_mean_4 = None
    sub_7: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_17, getitem_9);  add_17 = getitem_9 = None
    add_18: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
    rsqrt_4: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    mul_15: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_4);  sub_7 = rsqrt_4 = None
    mul_16: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_15, arg35_1);  mul_15 = arg35_1 = None
    add_19: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_16, arg36_1);  mul_16 = arg36_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_44: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_19, [2048, 768])
    permute_22: "f32[768, 768]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
    addmm_12: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg38_1, view_44, permute_22);  arg38_1 = view_44 = permute_22 = None
    view_45: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_12, [4, 512, 768]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_52: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_45, [4, 512, 12, 64]);  view_45 = None
    
    # No stacktrace found for following nodes
    permute_default_27: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_46: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_19, [2048, 768])
    permute_23: "f32[768, 768]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
    addmm_13: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg40_1, view_46, permute_23);  arg40_1 = view_46 = permute_23 = None
    view_47: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_13, [4, 512, 768]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_48: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_47, [4, 512, 12, 64]);  view_47 = None
    
    # No stacktrace found for following nodes
    permute_default_28: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_49: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_19, [2048, 768])
    permute_25: "f32[768, 768]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
    addmm_14: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg42_1, view_49, permute_25);  arg42_1 = view_49 = permute_25 = None
    view_50: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_14, [4, 512, 768]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_51: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_50, [4, 512, 12, 64]);  view_50 = None
    
    # No stacktrace found for following nodes
    permute_default_29: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None
    _scaled_dot_product_efficient_attention_default_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_27, permute_default_28, permute_default_29, None, True, scale = 0.125);  permute_default_27 = permute_default_28 = permute_default_29 = None
    getitem_61: "f32[4, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default_9[0];  _scaled_dot_product_efficient_attention_default_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_29: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_61, [0, 2, 1, 3]);  getitem_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_59: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(permute_29, [4, 512, 768]);  permute_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_60: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_59, [2048, 768]);  view_59 = None
    permute_30: "f32[768, 768]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
    
    # No stacktrace found for following nodes
    mm_default_30: "f32[2048, 768]" = torch.ops.aten.mm.default(view_60, permute_30);  view_60 = permute_30 = None
    add_tensor_30: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_30, arg44_1);  mm_default_30 = arg44_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_61: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_30, [4, 512, 768]);  add_tensor_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_21: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_61, add_19);  view_61 = add_19 = None
    var_mean_5 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
    getitem_10: "f32[4, 512, 1]" = var_mean_5[0]
    getitem_11: "f32[4, 512, 1]" = var_mean_5[1];  var_mean_5 = None
    sub_9: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_21, getitem_11);  add_21 = getitem_11 = None
    add_22: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
    rsqrt_5: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
    mul_17: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_5);  sub_9 = rsqrt_5 = None
    mul_18: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_17, arg45_1);  mul_17 = arg45_1 = None
    add_23: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_18, arg46_1);  mul_18 = arg46_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_62: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_23, [2048, 768])
    permute_31: "f32[768, 3072]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
    
    # No stacktrace found for following nodes
    mm_default_29: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_62, permute_31);  view_62 = permute_31 = None
    add_tensor_29: "f32[2048, 3072]" = torch.ops.aten.add.Tensor(mm_default_29, arg48_1);  mm_default_29 = arg48_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_63: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(add_tensor_29, [4, 512, 3072]);  add_tensor_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_19: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, 0.5)
    mul_20: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476);  view_63 = None
    erf_2: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_20);  mul_20 = None
    add_24: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_21: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_19, add_24);  mul_19 = add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_64: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_21, [2048, 3072]);  mul_21 = None
    permute_32: "f32[3072, 768]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
    
    # No stacktrace found for following nodes
    mm_default_28: "f32[2048, 768]" = torch.ops.aten.mm.default(view_64, permute_32);  view_64 = permute_32 = None
    add_tensor_28: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_28, arg50_1);  mm_default_28 = arg50_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_65: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_28, [4, 512, 768]);  add_tensor_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_25: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_65, add_23);  view_65 = add_23 = None
    var_mean_6 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
    getitem_12: "f32[4, 512, 1]" = var_mean_6[0]
    getitem_13: "f32[4, 512, 1]" = var_mean_6[1];  var_mean_6 = None
    sub_10: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_25, getitem_13);  add_25 = getitem_13 = None
    add_26: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
    rsqrt_6: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
    mul_22: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_6);  sub_10 = rsqrt_6 = None
    mul_23: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_22, arg51_1);  mul_22 = arg51_1 = None
    add_27: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_23, arg52_1);  mul_23 = arg52_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_66: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_27, [2048, 768])
    permute_33: "f32[768, 768]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
    addmm_18: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg54_1, view_66, permute_33);  arg54_1 = view_66 = permute_33 = None
    view_67: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_18, [4, 512, 768]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_74: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_67, [4, 512, 12, 64]);  view_67 = None
    
    # No stacktrace found for following nodes
    permute_default_24: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_68: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_27, [2048, 768])
    permute_34: "f32[768, 768]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
    addmm_19: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg56_1, view_68, permute_34);  arg56_1 = view_68 = permute_34 = None
    view_69: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_19, [4, 512, 768]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_70: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_69, [4, 512, 12, 64]);  view_69 = None
    
    # No stacktrace found for following nodes
    permute_default_25: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_71: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_27, [2048, 768])
    permute_36: "f32[768, 768]" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
    addmm_20: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg58_1, view_71, permute_36);  arg58_1 = view_71 = permute_36 = None
    view_72: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_20, [4, 512, 768]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_73: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_72, [4, 512, 12, 64]);  view_72 = None
    
    # No stacktrace found for following nodes
    permute_default_26: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None
    _scaled_dot_product_efficient_attention_default_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_24, permute_default_25, permute_default_26, None, True, scale = 0.125);  permute_default_24 = permute_default_25 = permute_default_26 = None
    getitem_60: "f32[4, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default_8[0];  _scaled_dot_product_efficient_attention_default_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_40: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3]);  getitem_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_81: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(permute_40, [4, 512, 768]);  permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_82: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_81, [2048, 768]);  view_81 = None
    permute_41: "f32[768, 768]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
    
    # No stacktrace found for following nodes
    mm_default_27: "f32[2048, 768]" = torch.ops.aten.mm.default(view_82, permute_41);  view_82 = permute_41 = None
    add_tensor_27: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_27, arg60_1);  mm_default_27 = arg60_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_83: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_27, [4, 512, 768]);  add_tensor_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_29: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_83, add_27);  view_83 = add_27 = None
    var_mean_7 = torch.ops.aten.var_mean.correction(add_29, [2], correction = 0, keepdim = True)
    getitem_14: "f32[4, 512, 1]" = var_mean_7[0]
    getitem_15: "f32[4, 512, 1]" = var_mean_7[1];  var_mean_7 = None
    sub_12: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_29, getitem_15);  add_29 = getitem_15 = None
    add_30: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
    rsqrt_7: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
    mul_24: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_7);  sub_12 = rsqrt_7 = None
    mul_25: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_24, arg61_1);  mul_24 = arg61_1 = None
    add_31: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_25, arg62_1);  mul_25 = arg62_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_84: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_31, [2048, 768])
    permute_42: "f32[768, 3072]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
    
    # No stacktrace found for following nodes
    mm_default_26: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_84, permute_42);  view_84 = permute_42 = None
    add_tensor_26: "f32[2048, 3072]" = torch.ops.aten.add.Tensor(mm_default_26, arg64_1);  mm_default_26 = arg64_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_85: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(add_tensor_26, [4, 512, 3072]);  add_tensor_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_26: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, 0.5)
    mul_27: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476);  view_85 = None
    erf_3: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_27);  mul_27 = None
    add_32: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_28: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_26, add_32);  mul_26 = add_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_86: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_28, [2048, 3072]);  mul_28 = None
    permute_43: "f32[3072, 768]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
    
    # No stacktrace found for following nodes
    mm_default_25: "f32[2048, 768]" = torch.ops.aten.mm.default(view_86, permute_43);  view_86 = permute_43 = None
    add_tensor_25: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_25, arg66_1);  mm_default_25 = arg66_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_87: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_25, [4, 512, 768]);  add_tensor_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_33: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_87, add_31);  view_87 = add_31 = None
    var_mean_8 = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
    getitem_16: "f32[4, 512, 1]" = var_mean_8[0]
    getitem_17: "f32[4, 512, 1]" = var_mean_8[1];  var_mean_8 = None
    sub_13: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_33, getitem_17);  add_33 = getitem_17 = None
    add_34: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
    rsqrt_8: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
    mul_29: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_8);  sub_13 = rsqrt_8 = None
    mul_30: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_29, arg67_1);  mul_29 = arg67_1 = None
    add_35: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_30, arg68_1);  mul_30 = arg68_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_88: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_35, [2048, 768])
    permute_44: "f32[768, 768]" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
    addmm_24: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg70_1, view_88, permute_44);  arg70_1 = view_88 = permute_44 = None
    view_89: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_24, [4, 512, 768]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_96: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_89, [4, 512, 12, 64]);  view_89 = None
    
    # No stacktrace found for following nodes
    permute_default_21: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_90: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_35, [2048, 768])
    permute_45: "f32[768, 768]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
    addmm_25: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg72_1, view_90, permute_45);  arg72_1 = view_90 = permute_45 = None
    view_91: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_25, [4, 512, 768]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_92: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_91, [4, 512, 12, 64]);  view_91 = None
    
    # No stacktrace found for following nodes
    permute_default_22: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_93: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_35, [2048, 768])
    permute_47: "f32[768, 768]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
    addmm_26: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg74_1, view_93, permute_47);  arg74_1 = view_93 = permute_47 = None
    view_94: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_26, [4, 512, 768]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_95: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_94, [4, 512, 12, 64]);  view_94 = None
    
    # No stacktrace found for following nodes
    permute_default_23: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None
    _scaled_dot_product_efficient_attention_default_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_21, permute_default_22, permute_default_23, None, True, scale = 0.125);  permute_default_21 = permute_default_22 = permute_default_23 = None
    getitem_59: "f32[4, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default_7[0];  _scaled_dot_product_efficient_attention_default_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_51: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_59, [0, 2, 1, 3]);  getitem_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_103: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(permute_51, [4, 512, 768]);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_104: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_103, [2048, 768]);  view_103 = None
    permute_52: "f32[768, 768]" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
    
    # No stacktrace found for following nodes
    mm_default_24: "f32[2048, 768]" = torch.ops.aten.mm.default(view_104, permute_52);  view_104 = permute_52 = None
    add_tensor_24: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_24, arg76_1);  mm_default_24 = arg76_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_105: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_24, [4, 512, 768]);  add_tensor_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_37: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_105, add_35);  view_105 = add_35 = None
    var_mean_9 = torch.ops.aten.var_mean.correction(add_37, [2], correction = 0, keepdim = True)
    getitem_18: "f32[4, 512, 1]" = var_mean_9[0]
    getitem_19: "f32[4, 512, 1]" = var_mean_9[1];  var_mean_9 = None
    sub_15: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_37, getitem_19);  add_37 = getitem_19 = None
    add_38: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
    rsqrt_9: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
    mul_31: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_9);  sub_15 = rsqrt_9 = None
    mul_32: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_31, arg77_1);  mul_31 = arg77_1 = None
    add_39: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_32, arg78_1);  mul_32 = arg78_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_106: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_39, [2048, 768])
    permute_53: "f32[768, 3072]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
    
    # No stacktrace found for following nodes
    mm_default_23: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_106, permute_53);  view_106 = permute_53 = None
    add_tensor_23: "f32[2048, 3072]" = torch.ops.aten.add.Tensor(mm_default_23, arg80_1);  mm_default_23 = arg80_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_107: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(add_tensor_23, [4, 512, 3072]);  add_tensor_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_33: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.5)
    mul_34: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476);  view_107 = None
    erf_4: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_34);  mul_34 = None
    add_40: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_35: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_33, add_40);  mul_33 = add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_108: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_35, [2048, 3072]);  mul_35 = None
    permute_54: "f32[3072, 768]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
    
    # No stacktrace found for following nodes
    mm_default_22: "f32[2048, 768]" = torch.ops.aten.mm.default(view_108, permute_54);  view_108 = permute_54 = None
    add_tensor_22: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_22, arg82_1);  mm_default_22 = arg82_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_109: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_22, [4, 512, 768]);  add_tensor_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_41: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_109, add_39);  view_109 = add_39 = None
    var_mean_10 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
    getitem_20: "f32[4, 512, 1]" = var_mean_10[0]
    getitem_21: "f32[4, 512, 1]" = var_mean_10[1];  var_mean_10 = None
    sub_16: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_41, getitem_21);  add_41 = getitem_21 = None
    add_42: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
    rsqrt_10: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    mul_36: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_10);  sub_16 = rsqrt_10 = None
    mul_37: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_36, arg83_1);  mul_36 = arg83_1 = None
    add_43: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_37, arg84_1);  mul_37 = arg84_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_110: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_43, [2048, 768])
    permute_55: "f32[768, 768]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
    addmm_30: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg86_1, view_110, permute_55);  arg86_1 = view_110 = permute_55 = None
    view_111: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_30, [4, 512, 768]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_118: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_111, [4, 512, 12, 64]);  view_111 = None
    
    # No stacktrace found for following nodes
    permute_default_18: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_112: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_43, [2048, 768])
    permute_56: "f32[768, 768]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
    addmm_31: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg88_1, view_112, permute_56);  arg88_1 = view_112 = permute_56 = None
    view_113: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_31, [4, 512, 768]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_114: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_113, [4, 512, 12, 64]);  view_113 = None
    
    # No stacktrace found for following nodes
    permute_default_19: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_115: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_43, [2048, 768])
    permute_58: "f32[768, 768]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
    addmm_32: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg90_1, view_115, permute_58);  arg90_1 = view_115 = permute_58 = None
    view_116: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_32, [4, 512, 768]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_117: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_116, [4, 512, 12, 64]);  view_116 = None
    
    # No stacktrace found for following nodes
    permute_default_20: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None
    _scaled_dot_product_efficient_attention_default_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_18, permute_default_19, permute_default_20, None, True, scale = 0.125);  permute_default_18 = permute_default_19 = permute_default_20 = None
    getitem_58: "f32[4, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default_6[0];  _scaled_dot_product_efficient_attention_default_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_62: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_58, [0, 2, 1, 3]);  getitem_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_125: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(permute_62, [4, 512, 768]);  permute_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_126: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_125, [2048, 768]);  view_125 = None
    permute_63: "f32[768, 768]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
    
    # No stacktrace found for following nodes
    mm_default_21: "f32[2048, 768]" = torch.ops.aten.mm.default(view_126, permute_63);  view_126 = permute_63 = None
    add_tensor_21: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_21, arg92_1);  mm_default_21 = arg92_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_127: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_21, [4, 512, 768]);  add_tensor_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_45: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_127, add_43);  view_127 = add_43 = None
    var_mean_11 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
    getitem_22: "f32[4, 512, 1]" = var_mean_11[0]
    getitem_23: "f32[4, 512, 1]" = var_mean_11[1];  var_mean_11 = None
    sub_18: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_45, getitem_23);  add_45 = getitem_23 = None
    add_46: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
    rsqrt_11: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
    mul_38: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_11);  sub_18 = rsqrt_11 = None
    mul_39: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_38, arg93_1);  mul_38 = arg93_1 = None
    add_47: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_39, arg94_1);  mul_39 = arg94_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_128: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_47, [2048, 768])
    permute_64: "f32[768, 3072]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
    
    # No stacktrace found for following nodes
    mm_default_20: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_128, permute_64);  view_128 = permute_64 = None
    add_tensor_20: "f32[2048, 3072]" = torch.ops.aten.add.Tensor(mm_default_20, arg96_1);  mm_default_20 = arg96_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_129: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(add_tensor_20, [4, 512, 3072]);  add_tensor_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_40: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, 0.5)
    mul_41: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476);  view_129 = None
    erf_5: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_41);  mul_41 = None
    add_48: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_42: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_40, add_48);  mul_40 = add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_130: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_42, [2048, 3072]);  mul_42 = None
    permute_65: "f32[3072, 768]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
    
    # No stacktrace found for following nodes
    mm_default_19: "f32[2048, 768]" = torch.ops.aten.mm.default(view_130, permute_65);  view_130 = permute_65 = None
    add_tensor_19: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_19, arg98_1);  mm_default_19 = arg98_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_131: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_19, [4, 512, 768]);  add_tensor_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_49: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_131, add_47);  view_131 = add_47 = None
    var_mean_12 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
    getitem_24: "f32[4, 512, 1]" = var_mean_12[0]
    getitem_25: "f32[4, 512, 1]" = var_mean_12[1];  var_mean_12 = None
    sub_19: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_49, getitem_25);  add_49 = getitem_25 = None
    add_50: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
    rsqrt_12: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
    mul_43: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_12);  sub_19 = rsqrt_12 = None
    mul_44: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_43, arg99_1);  mul_43 = arg99_1 = None
    add_51: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_44, arg100_1);  mul_44 = arg100_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_132: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_51, [2048, 768])
    permute_66: "f32[768, 768]" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
    addmm_36: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg102_1, view_132, permute_66);  arg102_1 = view_132 = permute_66 = None
    view_133: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_36, [4, 512, 768]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_140: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_133, [4, 512, 12, 64]);  view_133 = None
    
    # No stacktrace found for following nodes
    permute_default_15: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_134: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_51, [2048, 768])
    permute_67: "f32[768, 768]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
    addmm_37: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg104_1, view_134, permute_67);  arg104_1 = view_134 = permute_67 = None
    view_135: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_37, [4, 512, 768]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_136: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_135, [4, 512, 12, 64]);  view_135 = None
    
    # No stacktrace found for following nodes
    permute_default_16: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_137: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_51, [2048, 768])
    permute_69: "f32[768, 768]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
    addmm_38: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg106_1, view_137, permute_69);  arg106_1 = view_137 = permute_69 = None
    view_138: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_38, [4, 512, 768]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_139: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_138, [4, 512, 12, 64]);  view_138 = None
    
    # No stacktrace found for following nodes
    permute_default_17: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_139, [0, 2, 1, 3]);  view_139 = None
    _scaled_dot_product_efficient_attention_default_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_15, permute_default_16, permute_default_17, None, True, scale = 0.125);  permute_default_15 = permute_default_16 = permute_default_17 = None
    getitem_57: "f32[4, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default_5[0];  _scaled_dot_product_efficient_attention_default_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_73: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_57, [0, 2, 1, 3]);  getitem_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_147: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(permute_73, [4, 512, 768]);  permute_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_148: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_147, [2048, 768]);  view_147 = None
    permute_74: "f32[768, 768]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
    
    # No stacktrace found for following nodes
    mm_default_18: "f32[2048, 768]" = torch.ops.aten.mm.default(view_148, permute_74);  view_148 = permute_74 = None
    add_tensor_18: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_18, arg108_1);  mm_default_18 = arg108_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_149: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_18, [4, 512, 768]);  add_tensor_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_53: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_149, add_51);  view_149 = add_51 = None
    var_mean_13 = torch.ops.aten.var_mean.correction(add_53, [2], correction = 0, keepdim = True)
    getitem_26: "f32[4, 512, 1]" = var_mean_13[0]
    getitem_27: "f32[4, 512, 1]" = var_mean_13[1];  var_mean_13 = None
    sub_21: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_53, getitem_27);  add_53 = getitem_27 = None
    add_54: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
    rsqrt_13: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
    mul_45: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_13);  sub_21 = rsqrt_13 = None
    mul_46: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_45, arg109_1);  mul_45 = arg109_1 = None
    add_55: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_46, arg110_1);  mul_46 = arg110_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_150: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_55, [2048, 768])
    permute_75: "f32[768, 3072]" = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
    
    # No stacktrace found for following nodes
    mm_default_17: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_150, permute_75);  view_150 = permute_75 = None
    add_tensor_17: "f32[2048, 3072]" = torch.ops.aten.add.Tensor(mm_default_17, arg112_1);  mm_default_17 = arg112_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_151: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(add_tensor_17, [4, 512, 3072]);  add_tensor_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_47: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, 0.5)
    mul_48: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476);  view_151 = None
    erf_6: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_48);  mul_48 = None
    add_56: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_49: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_47, add_56);  mul_47 = add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_152: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_49, [2048, 3072]);  mul_49 = None
    permute_76: "f32[3072, 768]" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
    
    # No stacktrace found for following nodes
    mm_default_16: "f32[2048, 768]" = torch.ops.aten.mm.default(view_152, permute_76);  view_152 = permute_76 = None
    add_tensor_16: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_16, arg114_1);  mm_default_16 = arg114_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_153: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_16, [4, 512, 768]);  add_tensor_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_57: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_153, add_55);  view_153 = add_55 = None
    var_mean_14 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
    getitem_28: "f32[4, 512, 1]" = var_mean_14[0]
    getitem_29: "f32[4, 512, 1]" = var_mean_14[1];  var_mean_14 = None
    sub_22: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_57, getitem_29);  add_57 = getitem_29 = None
    add_58: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
    rsqrt_14: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
    mul_50: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_14);  sub_22 = rsqrt_14 = None
    mul_51: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_50, arg115_1);  mul_50 = arg115_1 = None
    add_59: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_51, arg116_1);  mul_51 = arg116_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_154: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_59, [2048, 768])
    permute_77: "f32[768, 768]" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
    addmm_42: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg118_1, view_154, permute_77);  arg118_1 = view_154 = permute_77 = None
    view_155: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_42, [4, 512, 768]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_162: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_155, [4, 512, 12, 64]);  view_155 = None
    
    # No stacktrace found for following nodes
    permute_default_12: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_156: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_59, [2048, 768])
    permute_78: "f32[768, 768]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
    addmm_43: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg120_1, view_156, permute_78);  arg120_1 = view_156 = permute_78 = None
    view_157: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_43, [4, 512, 768]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_158: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_157, [4, 512, 12, 64]);  view_157 = None
    
    # No stacktrace found for following nodes
    permute_default_13: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_158, [0, 2, 1, 3]);  view_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_159: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_59, [2048, 768])
    permute_80: "f32[768, 768]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
    addmm_44: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg122_1, view_159, permute_80);  arg122_1 = view_159 = permute_80 = None
    view_160: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_44, [4, 512, 768]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_161: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_160, [4, 512, 12, 64]);  view_160 = None
    
    # No stacktrace found for following nodes
    permute_default_14: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_161, [0, 2, 1, 3]);  view_161 = None
    _scaled_dot_product_efficient_attention_default_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_12, permute_default_13, permute_default_14, None, True, scale = 0.125);  permute_default_12 = permute_default_13 = permute_default_14 = None
    getitem_56: "f32[4, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default_4[0];  _scaled_dot_product_efficient_attention_default_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_84: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_56, [0, 2, 1, 3]);  getitem_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_169: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(permute_84, [4, 512, 768]);  permute_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_170: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_169, [2048, 768]);  view_169 = None
    permute_85: "f32[768, 768]" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
    
    # No stacktrace found for following nodes
    mm_default_15: "f32[2048, 768]" = torch.ops.aten.mm.default(view_170, permute_85);  view_170 = permute_85 = None
    add_tensor_15: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_15, arg124_1);  mm_default_15 = arg124_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_171: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_15, [4, 512, 768]);  add_tensor_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_61: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_171, add_59);  view_171 = add_59 = None
    var_mean_15 = torch.ops.aten.var_mean.correction(add_61, [2], correction = 0, keepdim = True)
    getitem_30: "f32[4, 512, 1]" = var_mean_15[0]
    getitem_31: "f32[4, 512, 1]" = var_mean_15[1];  var_mean_15 = None
    sub_24: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_61, getitem_31);  add_61 = getitem_31 = None
    add_62: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
    rsqrt_15: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
    mul_52: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_15);  sub_24 = rsqrt_15 = None
    mul_53: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_52, arg125_1);  mul_52 = arg125_1 = None
    add_63: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_53, arg126_1);  mul_53 = arg126_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_172: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_63, [2048, 768])
    permute_86: "f32[768, 3072]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
    
    # No stacktrace found for following nodes
    mm_default_14: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_172, permute_86);  view_172 = permute_86 = None
    add_tensor_14: "f32[2048, 3072]" = torch.ops.aten.add.Tensor(mm_default_14, arg128_1);  mm_default_14 = arg128_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_173: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(add_tensor_14, [4, 512, 3072]);  add_tensor_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_54: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.5)
    mul_55: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476);  view_173 = None
    erf_7: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
    add_64: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_56: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_54, add_64);  mul_54 = add_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_174: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_56, [2048, 3072]);  mul_56 = None
    permute_87: "f32[3072, 768]" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
    
    # No stacktrace found for following nodes
    mm_default_13: "f32[2048, 768]" = torch.ops.aten.mm.default(view_174, permute_87);  view_174 = permute_87 = None
    add_tensor_13: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_13, arg130_1);  mm_default_13 = arg130_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_175: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_13, [4, 512, 768]);  add_tensor_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_65: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_175, add_63);  view_175 = add_63 = None
    var_mean_16 = torch.ops.aten.var_mean.correction(add_65, [2], correction = 0, keepdim = True)
    getitem_32: "f32[4, 512, 1]" = var_mean_16[0]
    getitem_33: "f32[4, 512, 1]" = var_mean_16[1];  var_mean_16 = None
    sub_25: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_65, getitem_33);  add_65 = getitem_33 = None
    add_66: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
    rsqrt_16: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    mul_57: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_16);  sub_25 = rsqrt_16 = None
    mul_58: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_57, arg131_1);  mul_57 = arg131_1 = None
    add_67: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_58, arg132_1);  mul_58 = arg132_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_176: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_67, [2048, 768])
    permute_88: "f32[768, 768]" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
    addmm_48: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg134_1, view_176, permute_88);  arg134_1 = view_176 = permute_88 = None
    view_177: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_48, [4, 512, 768]);  addmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_184: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_177, [4, 512, 12, 64]);  view_177 = None
    
    # No stacktrace found for following nodes
    permute_default_9: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_178: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_67, [2048, 768])
    permute_89: "f32[768, 768]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
    addmm_49: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg136_1, view_178, permute_89);  arg136_1 = view_178 = permute_89 = None
    view_179: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_49, [4, 512, 768]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_180: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_179, [4, 512, 12, 64]);  view_179 = None
    
    # No stacktrace found for following nodes
    permute_default_10: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_180, [0, 2, 1, 3]);  view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_181: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_67, [2048, 768])
    permute_91: "f32[768, 768]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
    addmm_50: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg138_1, view_181, permute_91);  arg138_1 = view_181 = permute_91 = None
    view_182: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_50, [4, 512, 768]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_183: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_182, [4, 512, 12, 64]);  view_182 = None
    
    # No stacktrace found for following nodes
    permute_default_11: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None
    _scaled_dot_product_efficient_attention_default_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_9, permute_default_10, permute_default_11, None, True, scale = 0.125);  permute_default_9 = permute_default_10 = permute_default_11 = None
    getitem_55: "f32[4, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default_3[0];  _scaled_dot_product_efficient_attention_default_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_95: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_55, [0, 2, 1, 3]);  getitem_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_191: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(permute_95, [4, 512, 768]);  permute_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_192: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_191, [2048, 768]);  view_191 = None
    permute_96: "f32[768, 768]" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
    
    # No stacktrace found for following nodes
    mm_default_12: "f32[2048, 768]" = torch.ops.aten.mm.default(view_192, permute_96);  view_192 = permute_96 = None
    add_tensor_12: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_12, arg140_1);  mm_default_12 = arg140_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_193: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_12, [4, 512, 768]);  add_tensor_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_69: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_193, add_67);  view_193 = add_67 = None
    var_mean_17 = torch.ops.aten.var_mean.correction(add_69, [2], correction = 0, keepdim = True)
    getitem_34: "f32[4, 512, 1]" = var_mean_17[0]
    getitem_35: "f32[4, 512, 1]" = var_mean_17[1];  var_mean_17 = None
    sub_27: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_69, getitem_35);  add_69 = getitem_35 = None
    add_70: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
    rsqrt_17: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
    mul_59: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_17);  sub_27 = rsqrt_17 = None
    mul_60: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_59, arg141_1);  mul_59 = arg141_1 = None
    add_71: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_60, arg142_1);  mul_60 = arg142_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_194: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_71, [2048, 768])
    permute_97: "f32[768, 3072]" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
    
    # No stacktrace found for following nodes
    mm_default_11: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_194, permute_97);  view_194 = permute_97 = None
    add_tensor_11: "f32[2048, 3072]" = torch.ops.aten.add.Tensor(mm_default_11, arg144_1);  mm_default_11 = arg144_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_195: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(add_tensor_11, [4, 512, 3072]);  add_tensor_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_61: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, 0.5)
    mul_62: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476);  view_195 = None
    erf_8: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_62);  mul_62 = None
    add_72: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_63: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_61, add_72);  mul_61 = add_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_196: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_63, [2048, 3072]);  mul_63 = None
    permute_98: "f32[3072, 768]" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
    
    # No stacktrace found for following nodes
    mm_default_10: "f32[2048, 768]" = torch.ops.aten.mm.default(view_196, permute_98);  view_196 = permute_98 = None
    add_tensor_10: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_10, arg146_1);  mm_default_10 = arg146_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_197: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_10, [4, 512, 768]);  add_tensor_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_73: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_197, add_71);  view_197 = add_71 = None
    var_mean_18 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
    getitem_36: "f32[4, 512, 1]" = var_mean_18[0]
    getitem_37: "f32[4, 512, 1]" = var_mean_18[1];  var_mean_18 = None
    sub_28: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_73, getitem_37);  add_73 = getitem_37 = None
    add_74: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
    rsqrt_18: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    mul_64: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_18);  sub_28 = rsqrt_18 = None
    mul_65: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_64, arg147_1);  mul_64 = arg147_1 = None
    add_75: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_65, arg148_1);  mul_65 = arg148_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_198: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_75, [2048, 768])
    permute_99: "f32[768, 768]" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
    addmm_54: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg150_1, view_198, permute_99);  arg150_1 = view_198 = permute_99 = None
    view_199: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_54, [4, 512, 768]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_206: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_199, [4, 512, 12, 64]);  view_199 = None
    
    # No stacktrace found for following nodes
    permute_default_6: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_200: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_75, [2048, 768])
    permute_100: "f32[768, 768]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
    addmm_55: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg152_1, view_200, permute_100);  arg152_1 = view_200 = permute_100 = None
    view_201: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_55, [4, 512, 768]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_202: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_201, [4, 512, 12, 64]);  view_201 = None
    
    # No stacktrace found for following nodes
    permute_default_7: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_202, [0, 2, 1, 3]);  view_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_203: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_75, [2048, 768])
    permute_102: "f32[768, 768]" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
    addmm_56: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg154_1, view_203, permute_102);  arg154_1 = view_203 = permute_102 = None
    view_204: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_56, [4, 512, 768]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_205: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_204, [4, 512, 12, 64]);  view_204 = None
    
    # No stacktrace found for following nodes
    permute_default_8: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_205, [0, 2, 1, 3]);  view_205 = None
    _scaled_dot_product_efficient_attention_default_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_6, permute_default_7, permute_default_8, None, True, scale = 0.125);  permute_default_6 = permute_default_7 = permute_default_8 = None
    getitem_54: "f32[4, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default_2[0];  _scaled_dot_product_efficient_attention_default_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_106: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_54, [0, 2, 1, 3]);  getitem_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_213: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(permute_106, [4, 512, 768]);  permute_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_214: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_213, [2048, 768]);  view_213 = None
    permute_107: "f32[768, 768]" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
    
    # No stacktrace found for following nodes
    mm_default_9: "f32[2048, 768]" = torch.ops.aten.mm.default(view_214, permute_107);  view_214 = permute_107 = None
    add_tensor_9: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_9, arg156_1);  mm_default_9 = arg156_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_215: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_9, [4, 512, 768]);  add_tensor_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_77: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_215, add_75);  view_215 = add_75 = None
    var_mean_19 = torch.ops.aten.var_mean.correction(add_77, [2], correction = 0, keepdim = True)
    getitem_38: "f32[4, 512, 1]" = var_mean_19[0]
    getitem_39: "f32[4, 512, 1]" = var_mean_19[1];  var_mean_19 = None
    sub_30: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_77, getitem_39);  add_77 = getitem_39 = None
    add_78: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
    rsqrt_19: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    mul_66: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_19);  sub_30 = rsqrt_19 = None
    mul_67: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_66, arg157_1);  mul_66 = arg157_1 = None
    add_79: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_67, arg158_1);  mul_67 = arg158_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_216: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_79, [2048, 768])
    permute_108: "f32[768, 3072]" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
    
    # No stacktrace found for following nodes
    mm_default_8: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_216, permute_108);  view_216 = permute_108 = None
    add_tensor_8: "f32[2048, 3072]" = torch.ops.aten.add.Tensor(mm_default_8, arg160_1);  mm_default_8 = arg160_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_217: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(add_tensor_8, [4, 512, 3072]);  add_tensor_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_68: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, 0.5)
    mul_69: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476);  view_217 = None
    erf_9: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
    add_80: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_70: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_68, add_80);  mul_68 = add_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_218: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_70, [2048, 3072]);  mul_70 = None
    permute_109: "f32[3072, 768]" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
    
    # No stacktrace found for following nodes
    mm_default_7: "f32[2048, 768]" = torch.ops.aten.mm.default(view_218, permute_109);  view_218 = permute_109 = None
    add_tensor_7: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_7, arg162_1);  mm_default_7 = arg162_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_219: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_7, [4, 512, 768]);  add_tensor_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_81: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_219, add_79);  view_219 = add_79 = None
    var_mean_20 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
    getitem_40: "f32[4, 512, 1]" = var_mean_20[0]
    getitem_41: "f32[4, 512, 1]" = var_mean_20[1];  var_mean_20 = None
    sub_31: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_81, getitem_41);  add_81 = getitem_41 = None
    add_82: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
    rsqrt_20: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
    mul_71: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_20);  sub_31 = rsqrt_20 = None
    mul_72: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_71, arg163_1);  mul_71 = arg163_1 = None
    add_83: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_72, arg164_1);  mul_72 = arg164_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_220: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_83, [2048, 768])
    permute_110: "f32[768, 768]" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
    addmm_60: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg166_1, view_220, permute_110);  arg166_1 = view_220 = permute_110 = None
    view_221: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_60, [4, 512, 768]);  addmm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_228: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_221, [4, 512, 12, 64]);  view_221 = None
    
    # No stacktrace found for following nodes
    permute_default_3: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_222: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_83, [2048, 768])
    permute_111: "f32[768, 768]" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
    addmm_61: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg168_1, view_222, permute_111);  arg168_1 = view_222 = permute_111 = None
    view_223: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_61, [4, 512, 768]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_224: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_223, [4, 512, 12, 64]);  view_223 = None
    
    # No stacktrace found for following nodes
    permute_default_4: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_225: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_83, [2048, 768])
    permute_113: "f32[768, 768]" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
    addmm_62: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg170_1, view_225, permute_113);  arg170_1 = view_225 = permute_113 = None
    view_226: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_62, [4, 512, 768]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_227: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_226, [4, 512, 12, 64]);  view_226 = None
    
    # No stacktrace found for following nodes
    permute_default_5: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_227, [0, 2, 1, 3]);  view_227 = None
    _scaled_dot_product_efficient_attention_default_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default_3, permute_default_4, permute_default_5, None, True, scale = 0.125);  permute_default_3 = permute_default_4 = permute_default_5 = None
    getitem_53: "f32[4, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default_1[0];  _scaled_dot_product_efficient_attention_default_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_117: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_53, [0, 2, 1, 3]);  getitem_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_235: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(permute_117, [4, 512, 768]);  permute_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_236: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_235, [2048, 768]);  view_235 = None
    permute_118: "f32[768, 768]" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
    
    # No stacktrace found for following nodes
    mm_default_6: "f32[2048, 768]" = torch.ops.aten.mm.default(view_236, permute_118);  view_236 = permute_118 = None
    add_tensor_6: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_6, arg172_1);  mm_default_6 = arg172_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_237: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_6, [4, 512, 768]);  add_tensor_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_85: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_237, add_83);  view_237 = add_83 = None
    var_mean_21 = torch.ops.aten.var_mean.correction(add_85, [2], correction = 0, keepdim = True)
    getitem_42: "f32[4, 512, 1]" = var_mean_21[0]
    getitem_43: "f32[4, 512, 1]" = var_mean_21[1];  var_mean_21 = None
    sub_33: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_85, getitem_43);  add_85 = getitem_43 = None
    add_86: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
    rsqrt_21: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
    mul_73: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_21);  sub_33 = rsqrt_21 = None
    mul_74: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_73, arg173_1);  mul_73 = arg173_1 = None
    add_87: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_74, arg174_1);  mul_74 = arg174_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_238: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_87, [2048, 768])
    permute_119: "f32[768, 3072]" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
    
    # No stacktrace found for following nodes
    mm_default_5: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_238, permute_119);  view_238 = permute_119 = None
    add_tensor_5: "f32[2048, 3072]" = torch.ops.aten.add.Tensor(mm_default_5, arg176_1);  mm_default_5 = arg176_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_239: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(add_tensor_5, [4, 512, 3072]);  add_tensor_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_75: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, 0.5)
    mul_76: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, 0.7071067811865476);  view_239 = None
    erf_10: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_76);  mul_76 = None
    add_88: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_77: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_75, add_88);  mul_75 = add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_240: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_77, [2048, 3072]);  mul_77 = None
    permute_120: "f32[3072, 768]" = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
    
    # No stacktrace found for following nodes
    mm_default_4: "f32[2048, 768]" = torch.ops.aten.mm.default(view_240, permute_120);  view_240 = permute_120 = None
    add_tensor_4: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_4, arg178_1);  mm_default_4 = arg178_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_241: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_4, [4, 512, 768]);  add_tensor_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_89: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_241, add_87);  view_241 = add_87 = None
    var_mean_22 = torch.ops.aten.var_mean.correction(add_89, [2], correction = 0, keepdim = True)
    getitem_44: "f32[4, 512, 1]" = var_mean_22[0]
    getitem_45: "f32[4, 512, 1]" = var_mean_22[1];  var_mean_22 = None
    sub_34: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_89, getitem_45);  add_89 = getitem_45 = None
    add_90: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
    rsqrt_22: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
    mul_78: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_22);  sub_34 = rsqrt_22 = None
    mul_79: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_78, arg179_1);  mul_78 = arg179_1 = None
    add_91: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_79, arg180_1);  mul_79 = arg180_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_242: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_91, [2048, 768])
    permute_121: "f32[768, 768]" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
    addmm_66: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg182_1, view_242, permute_121);  arg182_1 = view_242 = permute_121 = None
    view_243: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_66, [4, 512, 768]);  addmm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_250: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_243, [4, 512, 12, 64]);  view_243 = None
    
    # No stacktrace found for following nodes
    permute_default: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_244: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_91, [2048, 768])
    permute_122: "f32[768, 768]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
    addmm_67: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg184_1, view_244, permute_122);  arg184_1 = view_244 = permute_122 = None
    view_245: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_67, [4, 512, 768]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_246: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_245, [4, 512, 12, 64]);  view_245 = None
    
    # No stacktrace found for following nodes
    permute_default_1: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_247: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_91, [2048, 768])
    permute_124: "f32[768, 768]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
    addmm_68: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg186_1, view_247, permute_124);  arg186_1 = view_247 = permute_124 = None
    view_248: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_68, [4, 512, 768]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_249: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_248, [4, 512, 12, 64]);  view_248 = None
    
    # No stacktrace found for following nodes
    permute_default_2: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_249, [0, 2, 1, 3]);  view_249 = None
    _scaled_dot_product_efficient_attention_default = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_default, permute_default_1, permute_default_2, None, True, scale = 0.125);  permute_default = permute_default_1 = permute_default_2 = None
    getitem_52: "f32[4, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default[0];  _scaled_dot_product_efficient_attention_default = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_128: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_52, [0, 2, 1, 3]);  getitem_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_257: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(permute_128, [4, 512, 768]);  permute_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_258: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_257, [2048, 768]);  view_257 = None
    permute_129: "f32[768, 768]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
    
    # No stacktrace found for following nodes
    mm_default_3: "f32[2048, 768]" = torch.ops.aten.mm.default(view_258, permute_129);  view_258 = permute_129 = None
    add_tensor_3: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_3, arg188_1);  mm_default_3 = arg188_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_259: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_3, [4, 512, 768]);  add_tensor_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_93: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_259, add_91);  view_259 = add_91 = None
    var_mean_23 = torch.ops.aten.var_mean.correction(add_93, [2], correction = 0, keepdim = True)
    getitem_46: "f32[4, 512, 1]" = var_mean_23[0]
    getitem_47: "f32[4, 512, 1]" = var_mean_23[1];  var_mean_23 = None
    sub_36: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_93, getitem_47);  add_93 = getitem_47 = None
    add_94: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
    rsqrt_23: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
    mul_80: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_23);  sub_36 = rsqrt_23 = None
    mul_81: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_80, arg189_1);  mul_80 = arg189_1 = None
    add_95: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_81, arg190_1);  mul_81 = arg190_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_260: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_95, [2048, 768])
    permute_130: "f32[768, 3072]" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
    
    # No stacktrace found for following nodes
    mm_default_2: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_260, permute_130);  view_260 = permute_130 = None
    add_tensor_2: "f32[2048, 3072]" = torch.ops.aten.add.Tensor(mm_default_2, arg192_1);  mm_default_2 = arg192_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_261: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(add_tensor_2, [4, 512, 3072]);  add_tensor_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_82: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, 0.5)
    mul_83: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476);  view_261 = None
    erf_11: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
    add_96: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_84: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_82, add_96);  mul_82 = add_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_262: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_84, [2048, 3072]);  mul_84 = None
    permute_131: "f32[3072, 768]" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
    
    # No stacktrace found for following nodes
    mm_default_1: "f32[2048, 768]" = torch.ops.aten.mm.default(view_262, permute_131);  view_262 = permute_131 = None
    add_tensor_1: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_1, arg194_1);  mm_default_1 = arg194_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_263: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_1, [4, 512, 768]);  add_tensor_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_97: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_263, add_95);  view_263 = add_95 = None
    var_mean_24 = torch.ops.aten.var_mean.correction(add_97, [2], correction = 0, keepdim = True)
    getitem_48: "f32[4, 512, 1]" = var_mean_24[0]
    getitem_49: "f32[4, 512, 1]" = var_mean_24[1];  var_mean_24 = None
    sub_37: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_97, getitem_49);  add_97 = getitem_49 = None
    add_98: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
    rsqrt_24: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
    mul_85: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_24);  sub_37 = rsqrt_24 = None
    mul_86: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_85, arg195_1);  mul_85 = arg195_1 = None
    add_99: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_86, arg196_1);  mul_86 = arg196_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:680, code: hidden_states = self.dense(hidden_states)
    view_264: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_99, [2048, 768]);  add_99 = None
    permute_132: "f32[768, 768]" = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
    
    # No stacktrace found for following nodes
    mm_default: "f32[2048, 768]" = torch.ops.aten.mm.default(view_264, permute_132);  view_264 = permute_132 = None
    add_tensor: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default, arg198_1);  mm_default = arg198_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:680, code: hidden_states = self.dense(hidden_states)
    view_265: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor, [4, 512, 768]);  add_tensor = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_87: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_265, 0.5)
    mul_88: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_265, 0.7071067811865476);  view_265 = None
    erf_12: "f32[4, 512, 768]" = torch.ops.aten.erf.default(mul_88);  mul_88 = None
    add_100: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_89: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_87, add_100);  mul_87 = add_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:682, code: hidden_states = self.LayerNorm(hidden_states)
    var_mean_25 = torch.ops.aten.var_mean.correction(mul_89, [2], correction = 0, keepdim = True)
    getitem_50: "f32[4, 512, 1]" = var_mean_25[0]
    getitem_51: "f32[4, 512, 1]" = var_mean_25[1];  var_mean_25 = None
    sub_38: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_89, getitem_51);  mul_89 = getitem_51 = None
    add_101: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
    rsqrt_25: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
    mul_90: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_25);  sub_38 = rsqrt_25 = None
    mul_91: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_90, arg199_1);  mul_90 = arg199_1 = None
    add_102: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_91, arg200_1);  mul_91 = arg200_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:702, code: hidden_states = self.decoder(hidden_states)
    view_266: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_102, [2048, 768]);  add_102 = None
    permute_133: "f32[768, 30522]" = torch.ops.aten.permute.default(arg201_1, [1, 0]);  arg201_1 = None
    addmm_73: "f32[2048, 30522]" = torch.ops.aten.addmm.default(arg202_1, view_266, permute_133);  arg202_1 = view_266 = permute_133 = None
    view_267: "f32[4, 512, 30522]" = torch.ops.aten.reshape.default(addmm_73, [4, 512, 30522]);  addmm_73 = None
    return (view_267,)
    