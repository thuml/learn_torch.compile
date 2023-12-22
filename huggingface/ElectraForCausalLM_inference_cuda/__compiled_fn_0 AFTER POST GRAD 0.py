from __future__ import annotations



def forward(self, arg0_1: "f32[30522, 128]", arg1_1: "f32[2, 128]", arg2_1: "f32[512, 128]", arg3_1: "f32[128]", arg4_1: "f32[128]", arg5_1: "f32[256, 128]", arg6_1: "f32[256]", arg7_1: "f32[256, 256]", arg8_1: "f32[256]", arg9_1: "f32[256, 256]", arg10_1: "f32[256]", arg11_1: "f32[256, 256]", arg12_1: "f32[256]", arg13_1: "f32[256, 256]", arg14_1: "f32[256]", arg15_1: "f32[256]", arg16_1: "f32[256]", arg17_1: "f32[1024, 256]", arg18_1: "f32[1024]", arg19_1: "f32[256, 1024]", arg20_1: "f32[256]", arg21_1: "f32[256]", arg22_1: "f32[256]", arg23_1: "f32[256, 256]", arg24_1: "f32[256]", arg25_1: "f32[256, 256]", arg26_1: "f32[256]", arg27_1: "f32[256, 256]", arg28_1: "f32[256]", arg29_1: "f32[256, 256]", arg30_1: "f32[256]", arg31_1: "f32[256]", arg32_1: "f32[256]", arg33_1: "f32[1024, 256]", arg34_1: "f32[1024]", arg35_1: "f32[256, 1024]", arg36_1: "f32[256]", arg37_1: "f32[256]", arg38_1: "f32[256]", arg39_1: "f32[256, 256]", arg40_1: "f32[256]", arg41_1: "f32[256, 256]", arg42_1: "f32[256]", arg43_1: "f32[256, 256]", arg44_1: "f32[256]", arg45_1: "f32[256, 256]", arg46_1: "f32[256]", arg47_1: "f32[256]", arg48_1: "f32[256]", arg49_1: "f32[1024, 256]", arg50_1: "f32[1024]", arg51_1: "f32[256, 1024]", arg52_1: "f32[256]", arg53_1: "f32[256]", arg54_1: "f32[256]", arg55_1: "f32[256, 256]", arg56_1: "f32[256]", arg57_1: "f32[256, 256]", arg58_1: "f32[256]", arg59_1: "f32[256, 256]", arg60_1: "f32[256]", arg61_1: "f32[256, 256]", arg62_1: "f32[256]", arg63_1: "f32[256]", arg64_1: "f32[256]", arg65_1: "f32[1024, 256]", arg66_1: "f32[1024]", arg67_1: "f32[256, 1024]", arg68_1: "f32[256]", arg69_1: "f32[256]", arg70_1: "f32[256]", arg71_1: "f32[256, 256]", arg72_1: "f32[256]", arg73_1: "f32[256, 256]", arg74_1: "f32[256]", arg75_1: "f32[256, 256]", arg76_1: "f32[256]", arg77_1: "f32[256, 256]", arg78_1: "f32[256]", arg79_1: "f32[256]", arg80_1: "f32[256]", arg81_1: "f32[1024, 256]", arg82_1: "f32[1024]", arg83_1: "f32[256, 1024]", arg84_1: "f32[256]", arg85_1: "f32[256]", arg86_1: "f32[256]", arg87_1: "f32[256, 256]", arg88_1: "f32[256]", arg89_1: "f32[256, 256]", arg90_1: "f32[256]", arg91_1: "f32[256, 256]", arg92_1: "f32[256]", arg93_1: "f32[256, 256]", arg94_1: "f32[256]", arg95_1: "f32[256]", arg96_1: "f32[256]", arg97_1: "f32[1024, 256]", arg98_1: "f32[1024]", arg99_1: "f32[256, 1024]", arg100_1: "f32[256]", arg101_1: "f32[256]", arg102_1: "f32[256]", arg103_1: "f32[256, 256]", arg104_1: "f32[256]", arg105_1: "f32[256, 256]", arg106_1: "f32[256]", arg107_1: "f32[256, 256]", arg108_1: "f32[256]", arg109_1: "f32[256, 256]", arg110_1: "f32[256]", arg111_1: "f32[256]", arg112_1: "f32[256]", arg113_1: "f32[1024, 256]", arg114_1: "f32[1024]", arg115_1: "f32[256, 1024]", arg116_1: "f32[256]", arg117_1: "f32[256]", arg118_1: "f32[256]", arg119_1: "f32[256, 256]", arg120_1: "f32[256]", arg121_1: "f32[256, 256]", arg122_1: "f32[256]", arg123_1: "f32[256, 256]", arg124_1: "f32[256]", arg125_1: "f32[256, 256]", arg126_1: "f32[256]", arg127_1: "f32[256]", arg128_1: "f32[256]", arg129_1: "f32[1024, 256]", arg130_1: "f32[1024]", arg131_1: "f32[256, 1024]", arg132_1: "f32[256]", arg133_1: "f32[256]", arg134_1: "f32[256]", arg135_1: "f32[256, 256]", arg136_1: "f32[256]", arg137_1: "f32[256, 256]", arg138_1: "f32[256]", arg139_1: "f32[256, 256]", arg140_1: "f32[256]", arg141_1: "f32[256, 256]", arg142_1: "f32[256]", arg143_1: "f32[256]", arg144_1: "f32[256]", arg145_1: "f32[1024, 256]", arg146_1: "f32[1024]", arg147_1: "f32[256, 1024]", arg148_1: "f32[256]", arg149_1: "f32[256]", arg150_1: "f32[256]", arg151_1: "f32[256, 256]", arg152_1: "f32[256]", arg153_1: "f32[256, 256]", arg154_1: "f32[256]", arg155_1: "f32[256, 256]", arg156_1: "f32[256]", arg157_1: "f32[256, 256]", arg158_1: "f32[256]", arg159_1: "f32[256]", arg160_1: "f32[256]", arg161_1: "f32[1024, 256]", arg162_1: "f32[1024]", arg163_1: "f32[256, 1024]", arg164_1: "f32[256]", arg165_1: "f32[256]", arg166_1: "f32[256]", arg167_1: "f32[256, 256]", arg168_1: "f32[256]", arg169_1: "f32[256, 256]", arg170_1: "f32[256]", arg171_1: "f32[256, 256]", arg172_1: "f32[256]", arg173_1: "f32[256, 256]", arg174_1: "f32[256]", arg175_1: "f32[256]", arg176_1: "f32[256]", arg177_1: "f32[1024, 256]", arg178_1: "f32[1024]", arg179_1: "f32[256, 1024]", arg180_1: "f32[256]", arg181_1: "f32[256]", arg182_1: "f32[256]", arg183_1: "f32[256, 256]", arg184_1: "f32[256]", arg185_1: "f32[256, 256]", arg186_1: "f32[256]", arg187_1: "f32[256, 256]", arg188_1: "f32[256]", arg189_1: "f32[256, 256]", arg190_1: "f32[256]", arg191_1: "f32[256]", arg192_1: "f32[256]", arg193_1: "f32[1024, 256]", arg194_1: "f32[1024]", arg195_1: "f32[256, 1024]", arg196_1: "f32[256]", arg197_1: "f32[256]", arg198_1: "f32[256]", arg199_1: "f32[128, 256]", arg200_1: "f32[128]", arg201_1: "f32[128]", arg202_1: "f32[128]", arg203_1: "f32[30522, 128]", arg204_1: "f32[30522]", arg205_1: "i64[1, 512]", arg206_1: "i64[1, 512]", arg207_1: "i64[1, 512]", arg208_1: "i64[1, 512]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:885, code: attention_mask = torch.ones(input_shape, device=device)
    full: "f32[1, 512]" = torch.ops.aten.full.default([1, 512], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:916, code: extended_attention_mask = attention_mask[:, None, None, :]
    unsqueeze: "f32[1, 1, 512]" = torch.ops.aten.unsqueeze.default(full, 1);  full = None
    unsqueeze_1: "f32[1, 1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:928, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    sub: "f32[1, 1, 1, 512]" = torch.ops.aten.sub.Tensor(1.0, unsqueeze_1);  unsqueeze_1 = None
    full_default: "f32[1, 1, 1, 512]" = torch.ops.aten.full.default([1, 1, 1, 512], -0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:203, code: inputs_embeds = self.word_embeddings(input_ids)
    embedding: "f32[1, 512, 128]" = torch.ops.aten.embedding.default(arg0_1, arg208_1, 0);  arg0_1 = arg208_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:889, code: buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
    expand: "i64[1, 512]" = torch.ops.aten.expand.default(arg205_1, [1, 512]);  arg205_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:204, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    embedding_1: "f32[1, 512, 128]" = torch.ops.aten.embedding.default(arg1_1, expand);  arg1_1 = expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:206, code: embeddings = inputs_embeds + token_type_embeddings
    add: "f32[1, 512, 128]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:208, code: position_embeddings = self.position_embeddings(position_ids)
    embedding_2: "f32[1, 512, 128]" = torch.ops.aten.embedding.default(arg2_1, arg206_1);  arg2_1 = arg206_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:209, code: embeddings += position_embeddings
    add_1: "f32[1, 512, 128]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:210, code: embeddings = self.LayerNorm(embeddings)
    var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 512, 1]" = var_mean[0]
    getitem_1: "f32[1, 512, 1]" = var_mean[1];  var_mean = None
    sub_1: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
    add_2: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
    rsqrt: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
    mul_1: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = rsqrt = None
    mul_2: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
    add_3: "f32[1, 512, 128]" = torch.ops.aten.add.Tensor(mul_2, arg4_1);  mul_2 = arg4_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:918, code: hidden_states = self.embeddings_project(hidden_states)
    view: "f32[512, 128]" = torch.ops.aten.reshape.default(add_3, [512, 128]);  add_3 = None
    permute: "f32[128, 256]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
    addmm: "f32[512, 256]" = torch.ops.aten.addmm.default(arg6_1, view, permute);  arg6_1 = view = permute = None
    view_1: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(addmm, [1, 512, 256]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_2: "f32[512, 256]" = torch.ops.aten.reshape.default(view_1, [512, 256])
    permute_1: "f32[256, 256]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
    
    # No stacktrace found for following nodes
    mm_default_72: "f32[512, 256]" = torch.ops.aten.mm.default(view_2, permute_1);  view_2 = permute_1 = None
    add_tensor_72: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_72, arg8_1);  mm_default_72 = arg8_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_3: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_72, [1, 512, 256]);  add_tensor_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_10: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_3, [1, 512, 4, 64]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_6: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
    
    # No stacktrace found for following nodes
    clone_default_33: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format);  permute_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_4: "f32[512, 256]" = torch.ops.aten.reshape.default(view_1, [512, 256])
    permute_2: "f32[256, 256]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
    
    # No stacktrace found for following nodes
    mm_default_71: "f32[512, 256]" = torch.ops.aten.mm.default(view_4, permute_2);  view_4 = permute_2 = None
    add_tensor_71: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_71, arg10_1);  mm_default_71 = arg10_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_5: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_71, [1, 512, 256]);  add_tensor_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_6: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_5, [1, 512, 4, 64]);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_3: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
    
    # No stacktrace found for following nodes
    clone_default_34: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_7: "f32[512, 256]" = torch.ops.aten.reshape.default(view_1, [512, 256])
    permute_4: "f32[256, 256]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
    
    # No stacktrace found for following nodes
    mm_default_70: "f32[512, 256]" = torch.ops.aten.mm.default(view_7, permute_4);  view_7 = permute_4 = None
    add_tensor_70: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_70, arg12_1);  mm_default_70 = arg12_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_8: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_70, [1, 512, 256]);  add_tensor_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_9: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_8, [1, 512, 4, 64]);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_5: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
    
    # No stacktrace found for following nodes
    clone_default_35: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    _scaled_dot_product_efficient_attention_default_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_33, clone_default_34, clone_default_35, None, False, scale = 0.125);  clone_default_33 = clone_default_34 = clone_default_35 = None
    getitem_63: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_default_11[0];  _scaled_dot_product_efficient_attention_default_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_8: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_63, [0, 2, 1, 3]);  getitem_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_17: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(permute_8, [1, 512, 256]);  permute_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_18: "f32[512, 256]" = torch.ops.aten.reshape.default(view_17, [512, 256]);  view_17 = None
    permute_9: "f32[256, 256]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
    
    # No stacktrace found for following nodes
    mm_default_69: "f32[512, 256]" = torch.ops.aten.mm.default(view_18, permute_9);  view_18 = permute_9 = None
    add_tensor_69: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_69, arg14_1);  mm_default_69 = arg14_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_19: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_69, [1, 512, 256]);  add_tensor_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_5: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(view_19, view_1);  view_19 = view_1 = None
    var_mean_1 = torch.ops.aten.var_mean.correction(add_5, [2], correction = 0, keepdim = True)
    getitem_2: "f32[1, 512, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 512, 1]" = var_mean_1[1];  var_mean_1 = None
    sub_3: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_5, getitem_3);  add_5 = getitem_3 = None
    add_6: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
    rsqrt_1: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    mul_3: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = rsqrt_1 = None
    mul_4: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_3, arg15_1);  mul_3 = arg15_1 = None
    add_7: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_4, arg16_1);  mul_4 = arg16_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_20: "f32[512, 256]" = torch.ops.aten.reshape.default(add_7, [512, 256])
    permute_10: "f32[256, 1024]" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
    
    # No stacktrace found for following nodes
    mm_default_68: "f32[512, 1024]" = torch.ops.aten.mm.default(view_20, permute_10);  view_20 = permute_10 = None
    add_tensor_68: "f32[512, 1024]" = torch.ops.aten.add.Tensor(mm_default_68, arg18_1);  mm_default_68 = arg18_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_21: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_68, [1, 512, 1024]);  add_tensor_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_5: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_21, 0.5)
    mul_6: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_21, 0.7071067811865476);  view_21 = None
    erf: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
    add_8: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_7: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_5, add_8);  mul_5 = add_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_22: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_7, [512, 1024]);  mul_7 = None
    permute_11: "f32[1024, 256]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
    
    # No stacktrace found for following nodes
    mm_default_67: "f32[512, 256]" = torch.ops.aten.mm.default(view_22, permute_11);  view_22 = permute_11 = None
    add_tensor_67: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_67, arg20_1);  mm_default_67 = arg20_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_23: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_67, [1, 512, 256]);  add_tensor_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_9: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(view_23, add_7);  view_23 = add_7 = None
    var_mean_2 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
    getitem_4: "f32[1, 512, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 512, 1]" = var_mean_2[1];  var_mean_2 = None
    sub_4: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_9, getitem_5);  add_9 = getitem_5 = None
    add_10: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
    rsqrt_2: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
    mul_8: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = rsqrt_2 = None
    mul_9: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_8, arg21_1);  mul_8 = arg21_1 = None
    add_11: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_9, arg22_1);  mul_9 = arg22_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_24: "f32[512, 256]" = torch.ops.aten.reshape.default(add_11, [512, 256])
    permute_12: "f32[256, 256]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
    
    # No stacktrace found for following nodes
    mm_default_66: "f32[512, 256]" = torch.ops.aten.mm.default(view_24, permute_12);  view_24 = permute_12 = None
    add_tensor_66: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_66, arg24_1);  mm_default_66 = arg24_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_25: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_66, [1, 512, 256]);  add_tensor_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_32: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_25, [1, 512, 4, 64]);  view_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_17: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None
    
    # No stacktrace found for following nodes
    clone_default_30: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_17, memory_format = torch.contiguous_format);  permute_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_26: "f32[512, 256]" = torch.ops.aten.reshape.default(add_11, [512, 256])
    permute_13: "f32[256, 256]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
    
    # No stacktrace found for following nodes
    mm_default_65: "f32[512, 256]" = torch.ops.aten.mm.default(view_26, permute_13);  view_26 = permute_13 = None
    add_tensor_65: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_65, arg26_1);  mm_default_65 = arg26_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_27: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_65, [1, 512, 256]);  add_tensor_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_28: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_27, [1, 512, 4, 64]);  view_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_14: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
    
    # No stacktrace found for following nodes
    clone_default_31: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_29: "f32[512, 256]" = torch.ops.aten.reshape.default(add_11, [512, 256])
    permute_15: "f32[256, 256]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
    
    # No stacktrace found for following nodes
    mm_default_64: "f32[512, 256]" = torch.ops.aten.mm.default(view_29, permute_15);  view_29 = permute_15 = None
    add_tensor_64: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_64, arg28_1);  mm_default_64 = arg28_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_30: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_64, [1, 512, 256]);  add_tensor_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_31: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_30, [1, 512, 4, 64]);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_16: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None
    
    # No stacktrace found for following nodes
    clone_default_32: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
    _scaled_dot_product_efficient_attention_default_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_30, clone_default_31, clone_default_32, None, False, scale = 0.125);  clone_default_30 = clone_default_31 = clone_default_32 = None
    getitem_62: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_default_10[0];  _scaled_dot_product_efficient_attention_default_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_19: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_62, [0, 2, 1, 3]);  getitem_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_39: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(permute_19, [1, 512, 256]);  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_40: "f32[512, 256]" = torch.ops.aten.reshape.default(view_39, [512, 256]);  view_39 = None
    permute_20: "f32[256, 256]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
    
    # No stacktrace found for following nodes
    mm_default_63: "f32[512, 256]" = torch.ops.aten.mm.default(view_40, permute_20);  view_40 = permute_20 = None
    add_tensor_63: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_63, arg30_1);  mm_default_63 = arg30_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_41: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_63, [1, 512, 256]);  add_tensor_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_13: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(view_41, add_11);  view_41 = add_11 = None
    var_mean_3 = torch.ops.aten.var_mean.correction(add_13, [2], correction = 0, keepdim = True)
    getitem_6: "f32[1, 512, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 512, 1]" = var_mean_3[1];  var_mean_3 = None
    sub_6: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_13, getitem_7);  add_13 = getitem_7 = None
    add_14: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
    rsqrt_3: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
    mul_10: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_3);  sub_6 = rsqrt_3 = None
    mul_11: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_10, arg31_1);  mul_10 = arg31_1 = None
    add_15: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_11, arg32_1);  mul_11 = arg32_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_42: "f32[512, 256]" = torch.ops.aten.reshape.default(add_15, [512, 256])
    permute_21: "f32[256, 1024]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
    
    # No stacktrace found for following nodes
    mm_default_62: "f32[512, 1024]" = torch.ops.aten.mm.default(view_42, permute_21);  view_42 = permute_21 = None
    add_tensor_62: "f32[512, 1024]" = torch.ops.aten.add.Tensor(mm_default_62, arg34_1);  mm_default_62 = arg34_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_43: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_62, [1, 512, 1024]);  add_tensor_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_12: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_43, 0.5)
    mul_13: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_43, 0.7071067811865476);  view_43 = None
    erf_1: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
    add_16: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_14: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_12, add_16);  mul_12 = add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_44: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_14, [512, 1024]);  mul_14 = None
    permute_22: "f32[1024, 256]" = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
    
    # No stacktrace found for following nodes
    mm_default_61: "f32[512, 256]" = torch.ops.aten.mm.default(view_44, permute_22);  view_44 = permute_22 = None
    add_tensor_61: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_61, arg36_1);  mm_default_61 = arg36_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_45: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_61, [1, 512, 256]);  add_tensor_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_17: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(view_45, add_15);  view_45 = add_15 = None
    var_mean_4 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
    getitem_8: "f32[1, 512, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 512, 1]" = var_mean_4[1];  var_mean_4 = None
    sub_7: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_17, getitem_9);  add_17 = getitem_9 = None
    add_18: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
    rsqrt_4: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    mul_15: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_4);  sub_7 = rsqrt_4 = None
    mul_16: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_15, arg37_1);  mul_15 = arg37_1 = None
    add_19: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_16, arg38_1);  mul_16 = arg38_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_46: "f32[512, 256]" = torch.ops.aten.reshape.default(add_19, [512, 256])
    permute_23: "f32[256, 256]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
    
    # No stacktrace found for following nodes
    mm_default_60: "f32[512, 256]" = torch.ops.aten.mm.default(view_46, permute_23);  view_46 = permute_23 = None
    add_tensor_60: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_60, arg40_1);  mm_default_60 = arg40_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_47: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_60, [1, 512, 256]);  add_tensor_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_54: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_47, [1, 512, 4, 64]);  view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_28: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
    
    # No stacktrace found for following nodes
    clone_default_27: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_28, memory_format = torch.contiguous_format);  permute_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_48: "f32[512, 256]" = torch.ops.aten.reshape.default(add_19, [512, 256])
    permute_24: "f32[256, 256]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
    
    # No stacktrace found for following nodes
    mm_default_59: "f32[512, 256]" = torch.ops.aten.mm.default(view_48, permute_24);  view_48 = permute_24 = None
    add_tensor_59: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_59, arg42_1);  mm_default_59 = arg42_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_49: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_59, [1, 512, 256]);  add_tensor_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_50: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_49, [1, 512, 4, 64]);  view_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_25: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
    
    # No stacktrace found for following nodes
    clone_default_28: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_25, memory_format = torch.contiguous_format);  permute_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_51: "f32[512, 256]" = torch.ops.aten.reshape.default(add_19, [512, 256])
    permute_26: "f32[256, 256]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
    
    # No stacktrace found for following nodes
    mm_default_58: "f32[512, 256]" = torch.ops.aten.mm.default(view_51, permute_26);  view_51 = permute_26 = None
    add_tensor_58: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_58, arg44_1);  mm_default_58 = arg44_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_52: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_58, [1, 512, 256]);  add_tensor_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_53: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_52, [1, 512, 4, 64]);  view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_27: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
    
    # No stacktrace found for following nodes
    clone_default_29: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
    _scaled_dot_product_efficient_attention_default_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_27, clone_default_28, clone_default_29, None, False, scale = 0.125);  clone_default_27 = clone_default_28 = clone_default_29 = None
    getitem_61: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_default_9[0];  _scaled_dot_product_efficient_attention_default_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_30: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_61, [0, 2, 1, 3]);  getitem_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_61: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(permute_30, [1, 512, 256]);  permute_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_62: "f32[512, 256]" = torch.ops.aten.reshape.default(view_61, [512, 256]);  view_61 = None
    permute_31: "f32[256, 256]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
    
    # No stacktrace found for following nodes
    mm_default_57: "f32[512, 256]" = torch.ops.aten.mm.default(view_62, permute_31);  view_62 = permute_31 = None
    add_tensor_57: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_57, arg46_1);  mm_default_57 = arg46_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_63: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_57, [1, 512, 256]);  add_tensor_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_21: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(view_63, add_19);  view_63 = add_19 = None
    var_mean_5 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
    getitem_10: "f32[1, 512, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 512, 1]" = var_mean_5[1];  var_mean_5 = None
    sub_9: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_21, getitem_11);  add_21 = getitem_11 = None
    add_22: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
    rsqrt_5: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
    mul_17: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_5);  sub_9 = rsqrt_5 = None
    mul_18: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_17, arg47_1);  mul_17 = arg47_1 = None
    add_23: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_18, arg48_1);  mul_18 = arg48_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_64: "f32[512, 256]" = torch.ops.aten.reshape.default(add_23, [512, 256])
    permute_32: "f32[256, 1024]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
    
    # No stacktrace found for following nodes
    mm_default_56: "f32[512, 1024]" = torch.ops.aten.mm.default(view_64, permute_32);  view_64 = permute_32 = None
    add_tensor_56: "f32[512, 1024]" = torch.ops.aten.add.Tensor(mm_default_56, arg50_1);  mm_default_56 = arg50_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_65: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_56, [1, 512, 1024]);  add_tensor_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_19: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_65, 0.5)
    mul_20: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_65, 0.7071067811865476);  view_65 = None
    erf_2: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_20);  mul_20 = None
    add_24: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_21: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_19, add_24);  mul_19 = add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_66: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_21, [512, 1024]);  mul_21 = None
    permute_33: "f32[1024, 256]" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
    
    # No stacktrace found for following nodes
    mm_default_55: "f32[512, 256]" = torch.ops.aten.mm.default(view_66, permute_33);  view_66 = permute_33 = None
    add_tensor_55: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_55, arg52_1);  mm_default_55 = arg52_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_67: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_55, [1, 512, 256]);  add_tensor_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_25: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(view_67, add_23);  view_67 = add_23 = None
    var_mean_6 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
    getitem_12: "f32[1, 512, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 512, 1]" = var_mean_6[1];  var_mean_6 = None
    sub_10: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_25, getitem_13);  add_25 = getitem_13 = None
    add_26: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
    rsqrt_6: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
    mul_22: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_6);  sub_10 = rsqrt_6 = None
    mul_23: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_22, arg53_1);  mul_22 = arg53_1 = None
    add_27: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_23, arg54_1);  mul_23 = arg54_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_68: "f32[512, 256]" = torch.ops.aten.reshape.default(add_27, [512, 256])
    permute_34: "f32[256, 256]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
    
    # No stacktrace found for following nodes
    mm_default_54: "f32[512, 256]" = torch.ops.aten.mm.default(view_68, permute_34);  view_68 = permute_34 = None
    add_tensor_54: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_54, arg56_1);  mm_default_54 = arg56_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_69: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_54, [1, 512, 256]);  add_tensor_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_76: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_69, [1, 512, 4, 64]);  view_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_39: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
    
    # No stacktrace found for following nodes
    clone_default_24: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_39, memory_format = torch.contiguous_format);  permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_70: "f32[512, 256]" = torch.ops.aten.reshape.default(add_27, [512, 256])
    permute_35: "f32[256, 256]" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
    
    # No stacktrace found for following nodes
    mm_default_53: "f32[512, 256]" = torch.ops.aten.mm.default(view_70, permute_35);  view_70 = permute_35 = None
    add_tensor_53: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_53, arg58_1);  mm_default_53 = arg58_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_71: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_53, [1, 512, 256]);  add_tensor_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_72: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_71, [1, 512, 4, 64]);  view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_36: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
    
    # No stacktrace found for following nodes
    clone_default_25: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_36, memory_format = torch.contiguous_format);  permute_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_73: "f32[512, 256]" = torch.ops.aten.reshape.default(add_27, [512, 256])
    permute_37: "f32[256, 256]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
    
    # No stacktrace found for following nodes
    mm_default_52: "f32[512, 256]" = torch.ops.aten.mm.default(view_73, permute_37);  view_73 = permute_37 = None
    add_tensor_52: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_52, arg60_1);  mm_default_52 = arg60_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_74: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_52, [1, 512, 256]);  add_tensor_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_75: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_74, [1, 512, 4, 64]);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_38: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
    
    # No stacktrace found for following nodes
    clone_default_26: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
    _scaled_dot_product_efficient_attention_default_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_24, clone_default_25, clone_default_26, None, False, scale = 0.125);  clone_default_24 = clone_default_25 = clone_default_26 = None
    getitem_60: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_default_8[0];  _scaled_dot_product_efficient_attention_default_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_41: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3]);  getitem_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_83: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(permute_41, [1, 512, 256]);  permute_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_84: "f32[512, 256]" = torch.ops.aten.reshape.default(view_83, [512, 256]);  view_83 = None
    permute_42: "f32[256, 256]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
    
    # No stacktrace found for following nodes
    mm_default_51: "f32[512, 256]" = torch.ops.aten.mm.default(view_84, permute_42);  view_84 = permute_42 = None
    add_tensor_51: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_51, arg62_1);  mm_default_51 = arg62_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_85: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_51, [1, 512, 256]);  add_tensor_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_29: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(view_85, add_27);  view_85 = add_27 = None
    var_mean_7 = torch.ops.aten.var_mean.correction(add_29, [2], correction = 0, keepdim = True)
    getitem_14: "f32[1, 512, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 512, 1]" = var_mean_7[1];  var_mean_7 = None
    sub_12: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_29, getitem_15);  add_29 = getitem_15 = None
    add_30: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
    rsqrt_7: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
    mul_24: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_7);  sub_12 = rsqrt_7 = None
    mul_25: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_24, arg63_1);  mul_24 = arg63_1 = None
    add_31: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_25, arg64_1);  mul_25 = arg64_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_86: "f32[512, 256]" = torch.ops.aten.reshape.default(add_31, [512, 256])
    permute_43: "f32[256, 1024]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
    
    # No stacktrace found for following nodes
    mm_default_50: "f32[512, 1024]" = torch.ops.aten.mm.default(view_86, permute_43);  view_86 = permute_43 = None
    add_tensor_50: "f32[512, 1024]" = torch.ops.aten.add.Tensor(mm_default_50, arg66_1);  mm_default_50 = arg66_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_87: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_50, [1, 512, 1024]);  add_tensor_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_26: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_87, 0.5)
    mul_27: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_87, 0.7071067811865476);  view_87 = None
    erf_3: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_27);  mul_27 = None
    add_32: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_28: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_26, add_32);  mul_26 = add_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_88: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_28, [512, 1024]);  mul_28 = None
    permute_44: "f32[1024, 256]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
    
    # No stacktrace found for following nodes
    mm_default_49: "f32[512, 256]" = torch.ops.aten.mm.default(view_88, permute_44);  view_88 = permute_44 = None
    add_tensor_49: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_49, arg68_1);  mm_default_49 = arg68_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_89: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_49, [1, 512, 256]);  add_tensor_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_33: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(view_89, add_31);  view_89 = add_31 = None
    var_mean_8 = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
    getitem_16: "f32[1, 512, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 512, 1]" = var_mean_8[1];  var_mean_8 = None
    sub_13: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_33, getitem_17);  add_33 = getitem_17 = None
    add_34: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
    rsqrt_8: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
    mul_29: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_8);  sub_13 = rsqrt_8 = None
    mul_30: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_29, arg69_1);  mul_29 = arg69_1 = None
    add_35: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_30, arg70_1);  mul_30 = arg70_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_90: "f32[512, 256]" = torch.ops.aten.reshape.default(add_35, [512, 256])
    permute_45: "f32[256, 256]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
    
    # No stacktrace found for following nodes
    mm_default_48: "f32[512, 256]" = torch.ops.aten.mm.default(view_90, permute_45);  view_90 = permute_45 = None
    add_tensor_48: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_48, arg72_1);  mm_default_48 = arg72_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_91: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_48, [1, 512, 256]);  add_tensor_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_98: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_91, [1, 512, 4, 64]);  view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_50: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None
    
    # No stacktrace found for following nodes
    clone_default_21: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_50, memory_format = torch.contiguous_format);  permute_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_92: "f32[512, 256]" = torch.ops.aten.reshape.default(add_35, [512, 256])
    permute_46: "f32[256, 256]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
    
    # No stacktrace found for following nodes
    mm_default_47: "f32[512, 256]" = torch.ops.aten.mm.default(view_92, permute_46);  view_92 = permute_46 = None
    add_tensor_47: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_47, arg74_1);  mm_default_47 = arg74_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_93: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_47, [1, 512, 256]);  add_tensor_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_94: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_93, [1, 512, 4, 64]);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_47: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
    
    # No stacktrace found for following nodes
    clone_default_22: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_47, memory_format = torch.contiguous_format);  permute_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_95: "f32[512, 256]" = torch.ops.aten.reshape.default(add_35, [512, 256])
    permute_48: "f32[256, 256]" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
    
    # No stacktrace found for following nodes
    mm_default_46: "f32[512, 256]" = torch.ops.aten.mm.default(view_95, permute_48);  view_95 = permute_48 = None
    add_tensor_46: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_46, arg76_1);  mm_default_46 = arg76_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_96: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_46, [1, 512, 256]);  add_tensor_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_97: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_96, [1, 512, 4, 64]);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_49: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_97, [0, 2, 1, 3]);  view_97 = None
    
    # No stacktrace found for following nodes
    clone_default_23: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
    _scaled_dot_product_efficient_attention_default_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_21, clone_default_22, clone_default_23, None, False, scale = 0.125);  clone_default_21 = clone_default_22 = clone_default_23 = None
    getitem_59: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_default_7[0];  _scaled_dot_product_efficient_attention_default_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_52: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_59, [0, 2, 1, 3]);  getitem_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_105: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(permute_52, [1, 512, 256]);  permute_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_106: "f32[512, 256]" = torch.ops.aten.reshape.default(view_105, [512, 256]);  view_105 = None
    permute_53: "f32[256, 256]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
    
    # No stacktrace found for following nodes
    mm_default_45: "f32[512, 256]" = torch.ops.aten.mm.default(view_106, permute_53);  view_106 = permute_53 = None
    add_tensor_45: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_45, arg78_1);  mm_default_45 = arg78_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_107: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_45, [1, 512, 256]);  add_tensor_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_37: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(view_107, add_35);  view_107 = add_35 = None
    var_mean_9 = torch.ops.aten.var_mean.correction(add_37, [2], correction = 0, keepdim = True)
    getitem_18: "f32[1, 512, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 512, 1]" = var_mean_9[1];  var_mean_9 = None
    sub_15: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_37, getitem_19);  add_37 = getitem_19 = None
    add_38: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
    rsqrt_9: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
    mul_31: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_9);  sub_15 = rsqrt_9 = None
    mul_32: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_31, arg79_1);  mul_31 = arg79_1 = None
    add_39: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_32, arg80_1);  mul_32 = arg80_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_108: "f32[512, 256]" = torch.ops.aten.reshape.default(add_39, [512, 256])
    permute_54: "f32[256, 1024]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
    
    # No stacktrace found for following nodes
    mm_default_44: "f32[512, 1024]" = torch.ops.aten.mm.default(view_108, permute_54);  view_108 = permute_54 = None
    add_tensor_44: "f32[512, 1024]" = torch.ops.aten.add.Tensor(mm_default_44, arg82_1);  mm_default_44 = arg82_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_109: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_44, [1, 512, 1024]);  add_tensor_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_33: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_109, 0.5)
    mul_34: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_109, 0.7071067811865476);  view_109 = None
    erf_4: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_34);  mul_34 = None
    add_40: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_35: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_33, add_40);  mul_33 = add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_110: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_35, [512, 1024]);  mul_35 = None
    permute_55: "f32[1024, 256]" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
    
    # No stacktrace found for following nodes
    mm_default_43: "f32[512, 256]" = torch.ops.aten.mm.default(view_110, permute_55);  view_110 = permute_55 = None
    add_tensor_43: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_43, arg84_1);  mm_default_43 = arg84_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_111: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_43, [1, 512, 256]);  add_tensor_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_41: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(view_111, add_39);  view_111 = add_39 = None
    var_mean_10 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
    getitem_20: "f32[1, 512, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 512, 1]" = var_mean_10[1];  var_mean_10 = None
    sub_16: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_41, getitem_21);  add_41 = getitem_21 = None
    add_42: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
    rsqrt_10: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    mul_36: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_10);  sub_16 = rsqrt_10 = None
    mul_37: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_36, arg85_1);  mul_36 = arg85_1 = None
    add_43: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_37, arg86_1);  mul_37 = arg86_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_112: "f32[512, 256]" = torch.ops.aten.reshape.default(add_43, [512, 256])
    permute_56: "f32[256, 256]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
    
    # No stacktrace found for following nodes
    mm_default_42: "f32[512, 256]" = torch.ops.aten.mm.default(view_112, permute_56);  view_112 = permute_56 = None
    add_tensor_42: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_42, arg88_1);  mm_default_42 = arg88_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_113: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_42, [1, 512, 256]);  add_tensor_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_120: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_113, [1, 512, 4, 64]);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_61: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
    
    # No stacktrace found for following nodes
    clone_default_18: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_61, memory_format = torch.contiguous_format);  permute_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_114: "f32[512, 256]" = torch.ops.aten.reshape.default(add_43, [512, 256])
    permute_57: "f32[256, 256]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
    
    # No stacktrace found for following nodes
    mm_default_41: "f32[512, 256]" = torch.ops.aten.mm.default(view_114, permute_57);  view_114 = permute_57 = None
    add_tensor_41: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_41, arg90_1);  mm_default_41 = arg90_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_115: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_41, [1, 512, 256]);  add_tensor_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_116: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_115, [1, 512, 4, 64]);  view_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_58: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_116, [0, 2, 1, 3]);  view_116 = None
    
    # No stacktrace found for following nodes
    clone_default_19: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_58, memory_format = torch.contiguous_format);  permute_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_117: "f32[512, 256]" = torch.ops.aten.reshape.default(add_43, [512, 256])
    permute_59: "f32[256, 256]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
    
    # No stacktrace found for following nodes
    mm_default_40: "f32[512, 256]" = torch.ops.aten.mm.default(view_117, permute_59);  view_117 = permute_59 = None
    add_tensor_40: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_40, arg92_1);  mm_default_40 = arg92_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_118: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_40, [1, 512, 256]);  add_tensor_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_119: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_118, [1, 512, 4, 64]);  view_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_60: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_119, [0, 2, 1, 3]);  view_119 = None
    
    # No stacktrace found for following nodes
    clone_default_20: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
    _scaled_dot_product_efficient_attention_default_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_18, clone_default_19, clone_default_20, None, False, scale = 0.125);  clone_default_18 = clone_default_19 = clone_default_20 = None
    getitem_58: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_default_6[0];  _scaled_dot_product_efficient_attention_default_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_63: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_58, [0, 2, 1, 3]);  getitem_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_127: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(permute_63, [1, 512, 256]);  permute_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_128: "f32[512, 256]" = torch.ops.aten.reshape.default(view_127, [512, 256]);  view_127 = None
    permute_64: "f32[256, 256]" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
    
    # No stacktrace found for following nodes
    mm_default_39: "f32[512, 256]" = torch.ops.aten.mm.default(view_128, permute_64);  view_128 = permute_64 = None
    add_tensor_39: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_39, arg94_1);  mm_default_39 = arg94_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_129: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_39, [1, 512, 256]);  add_tensor_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_45: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(view_129, add_43);  view_129 = add_43 = None
    var_mean_11 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
    getitem_22: "f32[1, 512, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 512, 1]" = var_mean_11[1];  var_mean_11 = None
    sub_18: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_45, getitem_23);  add_45 = getitem_23 = None
    add_46: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
    rsqrt_11: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
    mul_38: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_11);  sub_18 = rsqrt_11 = None
    mul_39: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_38, arg95_1);  mul_38 = arg95_1 = None
    add_47: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_39, arg96_1);  mul_39 = arg96_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_130: "f32[512, 256]" = torch.ops.aten.reshape.default(add_47, [512, 256])
    permute_65: "f32[256, 1024]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
    
    # No stacktrace found for following nodes
    mm_default_38: "f32[512, 1024]" = torch.ops.aten.mm.default(view_130, permute_65);  view_130 = permute_65 = None
    add_tensor_38: "f32[512, 1024]" = torch.ops.aten.add.Tensor(mm_default_38, arg98_1);  mm_default_38 = arg98_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_131: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_38, [1, 512, 1024]);  add_tensor_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_40: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_131, 0.5)
    mul_41: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_131, 0.7071067811865476);  view_131 = None
    erf_5: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_41);  mul_41 = None
    add_48: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_42: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_40, add_48);  mul_40 = add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_132: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_42, [512, 1024]);  mul_42 = None
    permute_66: "f32[1024, 256]" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
    
    # No stacktrace found for following nodes
    mm_default_37: "f32[512, 256]" = torch.ops.aten.mm.default(view_132, permute_66);  view_132 = permute_66 = None
    add_tensor_37: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_37, arg100_1);  mm_default_37 = arg100_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_133: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_37, [1, 512, 256]);  add_tensor_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_49: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(view_133, add_47);  view_133 = add_47 = None
    var_mean_12 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
    getitem_24: "f32[1, 512, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 512, 1]" = var_mean_12[1];  var_mean_12 = None
    sub_19: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_49, getitem_25);  add_49 = getitem_25 = None
    add_50: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
    rsqrt_12: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
    mul_43: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_12);  sub_19 = rsqrt_12 = None
    mul_44: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_43, arg101_1);  mul_43 = arg101_1 = None
    add_51: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_44, arg102_1);  mul_44 = arg102_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_134: "f32[512, 256]" = torch.ops.aten.reshape.default(add_51, [512, 256])
    permute_67: "f32[256, 256]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
    
    # No stacktrace found for following nodes
    mm_default_36: "f32[512, 256]" = torch.ops.aten.mm.default(view_134, permute_67);  view_134 = permute_67 = None
    add_tensor_36: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_36, arg104_1);  mm_default_36 = arg104_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_135: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_36, [1, 512, 256]);  add_tensor_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_142: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_135, [1, 512, 4, 64]);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_72: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_142, [0, 2, 1, 3]);  view_142 = None
    
    # No stacktrace found for following nodes
    clone_default_15: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_72, memory_format = torch.contiguous_format);  permute_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_136: "f32[512, 256]" = torch.ops.aten.reshape.default(add_51, [512, 256])
    permute_68: "f32[256, 256]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
    
    # No stacktrace found for following nodes
    mm_default_35: "f32[512, 256]" = torch.ops.aten.mm.default(view_136, permute_68);  view_136 = permute_68 = None
    add_tensor_35: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_35, arg106_1);  mm_default_35 = arg106_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_137: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_35, [1, 512, 256]);  add_tensor_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_138: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_137, [1, 512, 4, 64]);  view_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_69: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None
    
    # No stacktrace found for following nodes
    clone_default_16: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_69, memory_format = torch.contiguous_format);  permute_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_139: "f32[512, 256]" = torch.ops.aten.reshape.default(add_51, [512, 256])
    permute_70: "f32[256, 256]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
    
    # No stacktrace found for following nodes
    mm_default_34: "f32[512, 256]" = torch.ops.aten.mm.default(view_139, permute_70);  view_139 = permute_70 = None
    add_tensor_34: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_34, arg108_1);  mm_default_34 = arg108_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_140: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_34, [1, 512, 256]);  add_tensor_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_141: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_140, [1, 512, 4, 64]);  view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_71: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_141, [0, 2, 1, 3]);  view_141 = None
    
    # No stacktrace found for following nodes
    clone_default_17: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
    _scaled_dot_product_efficient_attention_default_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_15, clone_default_16, clone_default_17, None, False, scale = 0.125);  clone_default_15 = clone_default_16 = clone_default_17 = None
    getitem_57: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_default_5[0];  _scaled_dot_product_efficient_attention_default_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_74: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_57, [0, 2, 1, 3]);  getitem_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_149: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(permute_74, [1, 512, 256]);  permute_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_150: "f32[512, 256]" = torch.ops.aten.reshape.default(view_149, [512, 256]);  view_149 = None
    permute_75: "f32[256, 256]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
    
    # No stacktrace found for following nodes
    mm_default_33: "f32[512, 256]" = torch.ops.aten.mm.default(view_150, permute_75);  view_150 = permute_75 = None
    add_tensor_33: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_33, arg110_1);  mm_default_33 = arg110_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_151: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_33, [1, 512, 256]);  add_tensor_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_53: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(view_151, add_51);  view_151 = add_51 = None
    var_mean_13 = torch.ops.aten.var_mean.correction(add_53, [2], correction = 0, keepdim = True)
    getitem_26: "f32[1, 512, 1]" = var_mean_13[0]
    getitem_27: "f32[1, 512, 1]" = var_mean_13[1];  var_mean_13 = None
    sub_21: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_53, getitem_27);  add_53 = getitem_27 = None
    add_54: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
    rsqrt_13: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
    mul_45: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_13);  sub_21 = rsqrt_13 = None
    mul_46: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_45, arg111_1);  mul_45 = arg111_1 = None
    add_55: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_46, arg112_1);  mul_46 = arg112_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_152: "f32[512, 256]" = torch.ops.aten.reshape.default(add_55, [512, 256])
    permute_76: "f32[256, 1024]" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
    
    # No stacktrace found for following nodes
    mm_default_32: "f32[512, 1024]" = torch.ops.aten.mm.default(view_152, permute_76);  view_152 = permute_76 = None
    add_tensor_32: "f32[512, 1024]" = torch.ops.aten.add.Tensor(mm_default_32, arg114_1);  mm_default_32 = arg114_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_153: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_32, [1, 512, 1024]);  add_tensor_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_47: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_153, 0.5)
    mul_48: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_153, 0.7071067811865476);  view_153 = None
    erf_6: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_48);  mul_48 = None
    add_56: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_49: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_47, add_56);  mul_47 = add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_154: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_49, [512, 1024]);  mul_49 = None
    permute_77: "f32[1024, 256]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
    
    # No stacktrace found for following nodes
    mm_default_31: "f32[512, 256]" = torch.ops.aten.mm.default(view_154, permute_77);  view_154 = permute_77 = None
    add_tensor_31: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_31, arg116_1);  mm_default_31 = arg116_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_155: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_31, [1, 512, 256]);  add_tensor_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_57: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(view_155, add_55);  view_155 = add_55 = None
    var_mean_14 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 512, 1]" = var_mean_14[0]
    getitem_29: "f32[1, 512, 1]" = var_mean_14[1];  var_mean_14 = None
    sub_22: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_57, getitem_29);  add_57 = getitem_29 = None
    add_58: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
    rsqrt_14: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
    mul_50: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_14);  sub_22 = rsqrt_14 = None
    mul_51: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_50, arg117_1);  mul_50 = arg117_1 = None
    add_59: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_51, arg118_1);  mul_51 = arg118_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_156: "f32[512, 256]" = torch.ops.aten.reshape.default(add_59, [512, 256])
    permute_78: "f32[256, 256]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
    
    # No stacktrace found for following nodes
    mm_default_30: "f32[512, 256]" = torch.ops.aten.mm.default(view_156, permute_78);  view_156 = permute_78 = None
    add_tensor_30: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_30, arg120_1);  mm_default_30 = arg120_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_157: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_30, [1, 512, 256]);  add_tensor_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_164: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_157, [1, 512, 4, 64]);  view_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_83: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_164, [0, 2, 1, 3]);  view_164 = None
    
    # No stacktrace found for following nodes
    clone_default_12: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_83, memory_format = torch.contiguous_format);  permute_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_158: "f32[512, 256]" = torch.ops.aten.reshape.default(add_59, [512, 256])
    permute_79: "f32[256, 256]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
    
    # No stacktrace found for following nodes
    mm_default_29: "f32[512, 256]" = torch.ops.aten.mm.default(view_158, permute_79);  view_158 = permute_79 = None
    add_tensor_29: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_29, arg122_1);  mm_default_29 = arg122_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_159: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_29, [1, 512, 256]);  add_tensor_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_160: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_159, [1, 512, 4, 64]);  view_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_80: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_160, [0, 2, 1, 3]);  view_160 = None
    
    # No stacktrace found for following nodes
    clone_default_13: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format);  permute_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_161: "f32[512, 256]" = torch.ops.aten.reshape.default(add_59, [512, 256])
    permute_81: "f32[256, 256]" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
    
    # No stacktrace found for following nodes
    mm_default_28: "f32[512, 256]" = torch.ops.aten.mm.default(view_161, permute_81);  view_161 = permute_81 = None
    add_tensor_28: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_28, arg124_1);  mm_default_28 = arg124_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_162: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_28, [1, 512, 256]);  add_tensor_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_163: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_162, [1, 512, 4, 64]);  view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_82: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_163, [0, 2, 1, 3]);  view_163 = None
    
    # No stacktrace found for following nodes
    clone_default_14: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
    _scaled_dot_product_efficient_attention_default_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_12, clone_default_13, clone_default_14, None, False, scale = 0.125);  clone_default_12 = clone_default_13 = clone_default_14 = None
    getitem_56: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_default_4[0];  _scaled_dot_product_efficient_attention_default_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_85: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_56, [0, 2, 1, 3]);  getitem_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_171: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(permute_85, [1, 512, 256]);  permute_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_172: "f32[512, 256]" = torch.ops.aten.reshape.default(view_171, [512, 256]);  view_171 = None
    permute_86: "f32[256, 256]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
    
    # No stacktrace found for following nodes
    mm_default_27: "f32[512, 256]" = torch.ops.aten.mm.default(view_172, permute_86);  view_172 = permute_86 = None
    add_tensor_27: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_27, arg126_1);  mm_default_27 = arg126_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_173: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_27, [1, 512, 256]);  add_tensor_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_61: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(view_173, add_59);  view_173 = add_59 = None
    var_mean_15 = torch.ops.aten.var_mean.correction(add_61, [2], correction = 0, keepdim = True)
    getitem_30: "f32[1, 512, 1]" = var_mean_15[0]
    getitem_31: "f32[1, 512, 1]" = var_mean_15[1];  var_mean_15 = None
    sub_24: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_61, getitem_31);  add_61 = getitem_31 = None
    add_62: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
    rsqrt_15: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
    mul_52: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_15);  sub_24 = rsqrt_15 = None
    mul_53: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_52, arg127_1);  mul_52 = arg127_1 = None
    add_63: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_53, arg128_1);  mul_53 = arg128_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_174: "f32[512, 256]" = torch.ops.aten.reshape.default(add_63, [512, 256])
    permute_87: "f32[256, 1024]" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
    
    # No stacktrace found for following nodes
    mm_default_26: "f32[512, 1024]" = torch.ops.aten.mm.default(view_174, permute_87);  view_174 = permute_87 = None
    add_tensor_26: "f32[512, 1024]" = torch.ops.aten.add.Tensor(mm_default_26, arg130_1);  mm_default_26 = arg130_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_175: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_26, [1, 512, 1024]);  add_tensor_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_54: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_175, 0.5)
    mul_55: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_175, 0.7071067811865476);  view_175 = None
    erf_7: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
    add_64: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_56: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_54, add_64);  mul_54 = add_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_176: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_56, [512, 1024]);  mul_56 = None
    permute_88: "f32[1024, 256]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
    
    # No stacktrace found for following nodes
    mm_default_25: "f32[512, 256]" = torch.ops.aten.mm.default(view_176, permute_88);  view_176 = permute_88 = None
    add_tensor_25: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_25, arg132_1);  mm_default_25 = arg132_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_177: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_25, [1, 512, 256]);  add_tensor_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_65: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(view_177, add_63);  view_177 = add_63 = None
    var_mean_16 = torch.ops.aten.var_mean.correction(add_65, [2], correction = 0, keepdim = True)
    getitem_32: "f32[1, 512, 1]" = var_mean_16[0]
    getitem_33: "f32[1, 512, 1]" = var_mean_16[1];  var_mean_16 = None
    sub_25: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_65, getitem_33);  add_65 = getitem_33 = None
    add_66: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
    rsqrt_16: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    mul_57: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_16);  sub_25 = rsqrt_16 = None
    mul_58: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_57, arg133_1);  mul_57 = arg133_1 = None
    add_67: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_58, arg134_1);  mul_58 = arg134_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_178: "f32[512, 256]" = torch.ops.aten.reshape.default(add_67, [512, 256])
    permute_89: "f32[256, 256]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
    
    # No stacktrace found for following nodes
    mm_default_24: "f32[512, 256]" = torch.ops.aten.mm.default(view_178, permute_89);  view_178 = permute_89 = None
    add_tensor_24: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_24, arg136_1);  mm_default_24 = arg136_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_179: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_24, [1, 512, 256]);  add_tensor_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_186: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_179, [1, 512, 4, 64]);  view_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_94: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
    
    # No stacktrace found for following nodes
    clone_default_9: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_94, memory_format = torch.contiguous_format);  permute_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_180: "f32[512, 256]" = torch.ops.aten.reshape.default(add_67, [512, 256])
    permute_90: "f32[256, 256]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
    
    # No stacktrace found for following nodes
    mm_default_23: "f32[512, 256]" = torch.ops.aten.mm.default(view_180, permute_90);  view_180 = permute_90 = None
    add_tensor_23: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_23, arg138_1);  mm_default_23 = arg138_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_181: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_23, [1, 512, 256]);  add_tensor_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_182: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_181, [1, 512, 4, 64]);  view_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_91: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None
    
    # No stacktrace found for following nodes
    clone_default_10: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_91, memory_format = torch.contiguous_format);  permute_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_183: "f32[512, 256]" = torch.ops.aten.reshape.default(add_67, [512, 256])
    permute_92: "f32[256, 256]" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
    
    # No stacktrace found for following nodes
    mm_default_22: "f32[512, 256]" = torch.ops.aten.mm.default(view_183, permute_92);  view_183 = permute_92 = None
    add_tensor_22: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_22, arg140_1);  mm_default_22 = arg140_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_184: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_22, [1, 512, 256]);  add_tensor_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_185: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_184, [1, 512, 4, 64]);  view_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_93: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None
    
    # No stacktrace found for following nodes
    clone_default_11: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
    _scaled_dot_product_efficient_attention_default_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_9, clone_default_10, clone_default_11, None, False, scale = 0.125);  clone_default_9 = clone_default_10 = clone_default_11 = None
    getitem_55: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_default_3[0];  _scaled_dot_product_efficient_attention_default_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_96: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_55, [0, 2, 1, 3]);  getitem_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_193: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(permute_96, [1, 512, 256]);  permute_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_194: "f32[512, 256]" = torch.ops.aten.reshape.default(view_193, [512, 256]);  view_193 = None
    permute_97: "f32[256, 256]" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
    
    # No stacktrace found for following nodes
    mm_default_21: "f32[512, 256]" = torch.ops.aten.mm.default(view_194, permute_97);  view_194 = permute_97 = None
    add_tensor_21: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_21, arg142_1);  mm_default_21 = arg142_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_195: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_21, [1, 512, 256]);  add_tensor_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_69: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(view_195, add_67);  view_195 = add_67 = None
    var_mean_17 = torch.ops.aten.var_mean.correction(add_69, [2], correction = 0, keepdim = True)
    getitem_34: "f32[1, 512, 1]" = var_mean_17[0]
    getitem_35: "f32[1, 512, 1]" = var_mean_17[1];  var_mean_17 = None
    sub_27: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_69, getitem_35);  add_69 = getitem_35 = None
    add_70: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
    rsqrt_17: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
    mul_59: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_17);  sub_27 = rsqrt_17 = None
    mul_60: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_59, arg143_1);  mul_59 = arg143_1 = None
    add_71: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_60, arg144_1);  mul_60 = arg144_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_196: "f32[512, 256]" = torch.ops.aten.reshape.default(add_71, [512, 256])
    permute_98: "f32[256, 1024]" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
    
    # No stacktrace found for following nodes
    mm_default_20: "f32[512, 1024]" = torch.ops.aten.mm.default(view_196, permute_98);  view_196 = permute_98 = None
    add_tensor_20: "f32[512, 1024]" = torch.ops.aten.add.Tensor(mm_default_20, arg146_1);  mm_default_20 = arg146_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_197: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_20, [1, 512, 1024]);  add_tensor_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_61: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_197, 0.5)
    mul_62: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_197, 0.7071067811865476);  view_197 = None
    erf_8: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_62);  mul_62 = None
    add_72: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_63: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_61, add_72);  mul_61 = add_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_198: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_63, [512, 1024]);  mul_63 = None
    permute_99: "f32[1024, 256]" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
    
    # No stacktrace found for following nodes
    mm_default_19: "f32[512, 256]" = torch.ops.aten.mm.default(view_198, permute_99);  view_198 = permute_99 = None
    add_tensor_19: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_19, arg148_1);  mm_default_19 = arg148_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_199: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_19, [1, 512, 256]);  add_tensor_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_73: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(view_199, add_71);  view_199 = add_71 = None
    var_mean_18 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
    getitem_36: "f32[1, 512, 1]" = var_mean_18[0]
    getitem_37: "f32[1, 512, 1]" = var_mean_18[1];  var_mean_18 = None
    sub_28: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_73, getitem_37);  add_73 = getitem_37 = None
    add_74: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
    rsqrt_18: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    mul_64: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_18);  sub_28 = rsqrt_18 = None
    mul_65: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_64, arg149_1);  mul_64 = arg149_1 = None
    add_75: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_65, arg150_1);  mul_65 = arg150_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_200: "f32[512, 256]" = torch.ops.aten.reshape.default(add_75, [512, 256])
    permute_100: "f32[256, 256]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
    
    # No stacktrace found for following nodes
    mm_default_18: "f32[512, 256]" = torch.ops.aten.mm.default(view_200, permute_100);  view_200 = permute_100 = None
    add_tensor_18: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_18, arg152_1);  mm_default_18 = arg152_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_201: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_18, [1, 512, 256]);  add_tensor_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_208: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_201, [1, 512, 4, 64]);  view_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_105: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None
    
    # No stacktrace found for following nodes
    clone_default_6: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_105, memory_format = torch.contiguous_format);  permute_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_202: "f32[512, 256]" = torch.ops.aten.reshape.default(add_75, [512, 256])
    permute_101: "f32[256, 256]" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
    
    # No stacktrace found for following nodes
    mm_default_17: "f32[512, 256]" = torch.ops.aten.mm.default(view_202, permute_101);  view_202 = permute_101 = None
    add_tensor_17: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_17, arg154_1);  mm_default_17 = arg154_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_203: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_17, [1, 512, 256]);  add_tensor_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_204: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_203, [1, 512, 4, 64]);  view_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_102: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_204, [0, 2, 1, 3]);  view_204 = None
    
    # No stacktrace found for following nodes
    clone_default_7: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_102, memory_format = torch.contiguous_format);  permute_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_205: "f32[512, 256]" = torch.ops.aten.reshape.default(add_75, [512, 256])
    permute_103: "f32[256, 256]" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
    
    # No stacktrace found for following nodes
    mm_default_16: "f32[512, 256]" = torch.ops.aten.mm.default(view_205, permute_103);  view_205 = permute_103 = None
    add_tensor_16: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_16, arg156_1);  mm_default_16 = arg156_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_206: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_16, [1, 512, 256]);  add_tensor_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_207: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_206, [1, 512, 4, 64]);  view_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_104: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_207, [0, 2, 1, 3]);  view_207 = None
    
    # No stacktrace found for following nodes
    clone_default_8: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
    _scaled_dot_product_efficient_attention_default_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_6, clone_default_7, clone_default_8, None, False, scale = 0.125);  clone_default_6 = clone_default_7 = clone_default_8 = None
    getitem_54: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_default_2[0];  _scaled_dot_product_efficient_attention_default_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_107: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_54, [0, 2, 1, 3]);  getitem_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_215: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(permute_107, [1, 512, 256]);  permute_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_216: "f32[512, 256]" = torch.ops.aten.reshape.default(view_215, [512, 256]);  view_215 = None
    permute_108: "f32[256, 256]" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
    
    # No stacktrace found for following nodes
    mm_default_15: "f32[512, 256]" = torch.ops.aten.mm.default(view_216, permute_108);  view_216 = permute_108 = None
    add_tensor_15: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_15, arg158_1);  mm_default_15 = arg158_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_217: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_15, [1, 512, 256]);  add_tensor_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_77: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(view_217, add_75);  view_217 = add_75 = None
    var_mean_19 = torch.ops.aten.var_mean.correction(add_77, [2], correction = 0, keepdim = True)
    getitem_38: "f32[1, 512, 1]" = var_mean_19[0]
    getitem_39: "f32[1, 512, 1]" = var_mean_19[1];  var_mean_19 = None
    sub_30: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_77, getitem_39);  add_77 = getitem_39 = None
    add_78: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
    rsqrt_19: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    mul_66: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_19);  sub_30 = rsqrt_19 = None
    mul_67: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_66, arg159_1);  mul_66 = arg159_1 = None
    add_79: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_67, arg160_1);  mul_67 = arg160_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_218: "f32[512, 256]" = torch.ops.aten.reshape.default(add_79, [512, 256])
    permute_109: "f32[256, 1024]" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
    
    # No stacktrace found for following nodes
    mm_default_14: "f32[512, 1024]" = torch.ops.aten.mm.default(view_218, permute_109);  view_218 = permute_109 = None
    add_tensor_14: "f32[512, 1024]" = torch.ops.aten.add.Tensor(mm_default_14, arg162_1);  mm_default_14 = arg162_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_219: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_14, [1, 512, 1024]);  add_tensor_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_68: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_219, 0.5)
    mul_69: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_219, 0.7071067811865476);  view_219 = None
    erf_9: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
    add_80: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_70: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_68, add_80);  mul_68 = add_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_220: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_70, [512, 1024]);  mul_70 = None
    permute_110: "f32[1024, 256]" = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
    
    # No stacktrace found for following nodes
    mm_default_13: "f32[512, 256]" = torch.ops.aten.mm.default(view_220, permute_110);  view_220 = permute_110 = None
    add_tensor_13: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_13, arg164_1);  mm_default_13 = arg164_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_221: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_13, [1, 512, 256]);  add_tensor_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_81: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(view_221, add_79);  view_221 = add_79 = None
    var_mean_20 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
    getitem_40: "f32[1, 512, 1]" = var_mean_20[0]
    getitem_41: "f32[1, 512, 1]" = var_mean_20[1];  var_mean_20 = None
    sub_31: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_81, getitem_41);  add_81 = getitem_41 = None
    add_82: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
    rsqrt_20: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
    mul_71: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_20);  sub_31 = rsqrt_20 = None
    mul_72: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_71, arg165_1);  mul_71 = arg165_1 = None
    add_83: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_72, arg166_1);  mul_72 = arg166_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_222: "f32[512, 256]" = torch.ops.aten.reshape.default(add_83, [512, 256])
    permute_111: "f32[256, 256]" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
    
    # No stacktrace found for following nodes
    mm_default_12: "f32[512, 256]" = torch.ops.aten.mm.default(view_222, permute_111);  view_222 = permute_111 = None
    add_tensor_12: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_12, arg168_1);  mm_default_12 = arg168_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_223: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_12, [1, 512, 256]);  add_tensor_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_230: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_223, [1, 512, 4, 64]);  view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_116: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_230, [0, 2, 1, 3]);  view_230 = None
    
    # No stacktrace found for following nodes
    clone_default_3: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_116, memory_format = torch.contiguous_format);  permute_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_224: "f32[512, 256]" = torch.ops.aten.reshape.default(add_83, [512, 256])
    permute_112: "f32[256, 256]" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
    
    # No stacktrace found for following nodes
    mm_default_11: "f32[512, 256]" = torch.ops.aten.mm.default(view_224, permute_112);  view_224 = permute_112 = None
    add_tensor_11: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_11, arg170_1);  mm_default_11 = arg170_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_225: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_11, [1, 512, 256]);  add_tensor_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_226: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_225, [1, 512, 4, 64]);  view_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_113: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
    
    # No stacktrace found for following nodes
    clone_default_4: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_113, memory_format = torch.contiguous_format);  permute_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_227: "f32[512, 256]" = torch.ops.aten.reshape.default(add_83, [512, 256])
    permute_114: "f32[256, 256]" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
    
    # No stacktrace found for following nodes
    mm_default_10: "f32[512, 256]" = torch.ops.aten.mm.default(view_227, permute_114);  view_227 = permute_114 = None
    add_tensor_10: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_10, arg172_1);  mm_default_10 = arg172_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_228: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_10, [1, 512, 256]);  add_tensor_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_229: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_228, [1, 512, 4, 64]);  view_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_115: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_229, [0, 2, 1, 3]);  view_229 = None
    
    # No stacktrace found for following nodes
    clone_default_5: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
    _scaled_dot_product_efficient_attention_default_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_3, clone_default_4, clone_default_5, None, False, scale = 0.125);  clone_default_3 = clone_default_4 = clone_default_5 = None
    getitem_53: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_default_1[0];  _scaled_dot_product_efficient_attention_default_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_118: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_53, [0, 2, 1, 3]);  getitem_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_237: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(permute_118, [1, 512, 256]);  permute_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_238: "f32[512, 256]" = torch.ops.aten.reshape.default(view_237, [512, 256]);  view_237 = None
    permute_119: "f32[256, 256]" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
    
    # No stacktrace found for following nodes
    mm_default_9: "f32[512, 256]" = torch.ops.aten.mm.default(view_238, permute_119);  view_238 = permute_119 = None
    add_tensor_9: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_9, arg174_1);  mm_default_9 = arg174_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_239: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_9, [1, 512, 256]);  add_tensor_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_85: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(view_239, add_83);  view_239 = add_83 = None
    var_mean_21 = torch.ops.aten.var_mean.correction(add_85, [2], correction = 0, keepdim = True)
    getitem_42: "f32[1, 512, 1]" = var_mean_21[0]
    getitem_43: "f32[1, 512, 1]" = var_mean_21[1];  var_mean_21 = None
    sub_33: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_85, getitem_43);  add_85 = getitem_43 = None
    add_86: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
    rsqrt_21: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
    mul_73: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_21);  sub_33 = rsqrt_21 = None
    mul_74: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_73, arg175_1);  mul_73 = arg175_1 = None
    add_87: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_74, arg176_1);  mul_74 = arg176_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_240: "f32[512, 256]" = torch.ops.aten.reshape.default(add_87, [512, 256])
    permute_120: "f32[256, 1024]" = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
    
    # No stacktrace found for following nodes
    mm_default_8: "f32[512, 1024]" = torch.ops.aten.mm.default(view_240, permute_120);  view_240 = permute_120 = None
    add_tensor_8: "f32[512, 1024]" = torch.ops.aten.add.Tensor(mm_default_8, arg178_1);  mm_default_8 = arg178_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_241: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_8, [1, 512, 1024]);  add_tensor_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_75: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_241, 0.5)
    mul_76: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_241, 0.7071067811865476);  view_241 = None
    erf_10: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_76);  mul_76 = None
    add_88: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_77: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_75, add_88);  mul_75 = add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_242: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_77, [512, 1024]);  mul_77 = None
    permute_121: "f32[1024, 256]" = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
    
    # No stacktrace found for following nodes
    mm_default_7: "f32[512, 256]" = torch.ops.aten.mm.default(view_242, permute_121);  view_242 = permute_121 = None
    add_tensor_7: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_7, arg180_1);  mm_default_7 = arg180_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_243: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_7, [1, 512, 256]);  add_tensor_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_89: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(view_243, add_87);  view_243 = add_87 = None
    var_mean_22 = torch.ops.aten.var_mean.correction(add_89, [2], correction = 0, keepdim = True)
    getitem_44: "f32[1, 512, 1]" = var_mean_22[0]
    getitem_45: "f32[1, 512, 1]" = var_mean_22[1];  var_mean_22 = None
    sub_34: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_89, getitem_45);  add_89 = getitem_45 = None
    add_90: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
    rsqrt_22: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
    mul_78: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_22);  sub_34 = rsqrt_22 = None
    mul_79: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_78, arg181_1);  mul_78 = arg181_1 = None
    add_91: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_79, arg182_1);  mul_79 = arg182_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_244: "f32[512, 256]" = torch.ops.aten.reshape.default(add_91, [512, 256])
    permute_122: "f32[256, 256]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
    
    # No stacktrace found for following nodes
    mm_default_6: "f32[512, 256]" = torch.ops.aten.mm.default(view_244, permute_122);  view_244 = permute_122 = None
    add_tensor_6: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_6, arg184_1);  mm_default_6 = arg184_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_245: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_6, [1, 512, 256]);  add_tensor_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_252: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_245, [1, 512, 4, 64]);  view_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_127: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None
    
    # No stacktrace found for following nodes
    clone_default: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_127, memory_format = torch.contiguous_format);  permute_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_246: "f32[512, 256]" = torch.ops.aten.reshape.default(add_91, [512, 256])
    permute_123: "f32[256, 256]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
    
    # No stacktrace found for following nodes
    mm_default_5: "f32[512, 256]" = torch.ops.aten.mm.default(view_246, permute_123);  view_246 = permute_123 = None
    add_tensor_5: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_5, arg186_1);  mm_default_5 = arg186_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_247: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_5, [1, 512, 256]);  add_tensor_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_248: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_247, [1, 512, 4, 64]);  view_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_124: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_248, [0, 2, 1, 3]);  view_248 = None
    
    # No stacktrace found for following nodes
    clone_default_1: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_124, memory_format = torch.contiguous_format);  permute_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_249: "f32[512, 256]" = torch.ops.aten.reshape.default(add_91, [512, 256])
    permute_125: "f32[256, 256]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
    
    # No stacktrace found for following nodes
    mm_default_4: "f32[512, 256]" = torch.ops.aten.mm.default(view_249, permute_125);  view_249 = permute_125 = None
    add_tensor_4: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_4, arg188_1);  mm_default_4 = arg188_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_250: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_4, [1, 512, 256]);  add_tensor_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_251: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_250, [1, 512, 4, 64]);  view_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_126: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_251, [0, 2, 1, 3]);  view_251 = None
    
    # No stacktrace found for following nodes
    clone_default_2: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
    _scaled_dot_product_efficient_attention_default = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default, clone_default_1, clone_default_2, None, False, scale = 0.125);  clone_default = clone_default_1 = clone_default_2 = None
    getitem_52: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_default[0];  _scaled_dot_product_efficient_attention_default = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_129: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_52, [0, 2, 1, 3]);  getitem_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_259: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(permute_129, [1, 512, 256]);  permute_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_260: "f32[512, 256]" = torch.ops.aten.reshape.default(view_259, [512, 256]);  view_259 = None
    permute_130: "f32[256, 256]" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
    
    # No stacktrace found for following nodes
    mm_default_3: "f32[512, 256]" = torch.ops.aten.mm.default(view_260, permute_130);  view_260 = permute_130 = None
    add_tensor_3: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_3, arg190_1);  mm_default_3 = arg190_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_261: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_3, [1, 512, 256]);  add_tensor_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_93: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(view_261, add_91);  view_261 = add_91 = None
    var_mean_23 = torch.ops.aten.var_mean.correction(add_93, [2], correction = 0, keepdim = True)
    getitem_46: "f32[1, 512, 1]" = var_mean_23[0]
    getitem_47: "f32[1, 512, 1]" = var_mean_23[1];  var_mean_23 = None
    sub_36: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_93, getitem_47);  add_93 = getitem_47 = None
    add_94: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
    rsqrt_23: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
    mul_80: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_23);  sub_36 = rsqrt_23 = None
    mul_81: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_80, arg191_1);  mul_80 = arg191_1 = None
    add_95: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_81, arg192_1);  mul_81 = arg192_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_262: "f32[512, 256]" = torch.ops.aten.reshape.default(add_95, [512, 256])
    permute_131: "f32[256, 1024]" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
    
    # No stacktrace found for following nodes
    mm_default_2: "f32[512, 1024]" = torch.ops.aten.mm.default(view_262, permute_131);  view_262 = permute_131 = None
    add_tensor_2: "f32[512, 1024]" = torch.ops.aten.add.Tensor(mm_default_2, arg194_1);  mm_default_2 = arg194_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_263: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(add_tensor_2, [1, 512, 1024]);  add_tensor_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_82: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_263, 0.5)
    mul_83: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_263, 0.7071067811865476);  view_263 = None
    erf_11: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
    add_96: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_84: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_82, add_96);  mul_82 = add_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_264: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_84, [512, 1024]);  mul_84 = None
    permute_132: "f32[1024, 256]" = torch.ops.aten.permute.default(arg195_1, [1, 0]);  arg195_1 = None
    
    # No stacktrace found for following nodes
    mm_default_1: "f32[512, 256]" = torch.ops.aten.mm.default(view_264, permute_132);  view_264 = permute_132 = None
    add_tensor_1: "f32[512, 256]" = torch.ops.aten.add.Tensor(mm_default_1, arg196_1);  mm_default_1 = arg196_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_265: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(add_tensor_1, [1, 512, 256]);  add_tensor_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_97: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(view_265, add_95);  view_265 = add_95 = None
    var_mean_24 = torch.ops.aten.var_mean.correction(add_97, [2], correction = 0, keepdim = True)
    getitem_48: "f32[1, 512, 1]" = var_mean_24[0]
    getitem_49: "f32[1, 512, 1]" = var_mean_24[1];  var_mean_24 = None
    sub_37: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_97, getitem_49);  add_97 = getitem_49 = None
    add_98: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
    rsqrt_24: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
    mul_85: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_24);  sub_37 = rsqrt_24 = None
    mul_86: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_85, arg197_1);  mul_85 = arg197_1 = None
    add_99: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_86, arg198_1);  mul_86 = arg198_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:660, code: hidden_states = self.dense(generator_hidden_states)
    view_266: "f32[512, 256]" = torch.ops.aten.reshape.default(add_99, [512, 256]);  add_99 = None
    permute_133: "f32[256, 128]" = torch.ops.aten.permute.default(arg199_1, [1, 0]);  arg199_1 = None
    
    # No stacktrace found for following nodes
    mm_default: "f32[512, 128]" = torch.ops.aten.mm.default(view_266, permute_133);  view_266 = permute_133 = None
    add_tensor: "f32[512, 128]" = torch.ops.aten.add.Tensor(mm_default, arg200_1);  mm_default = arg200_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:660, code: hidden_states = self.dense(generator_hidden_states)
    view_267: "f32[1, 512, 128]" = torch.ops.aten.reshape.default(add_tensor, [1, 512, 128]);  add_tensor = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_87: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(view_267, 0.5)
    mul_88: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(view_267, 0.7071067811865476);  view_267 = None
    erf_12: "f32[1, 512, 128]" = torch.ops.aten.erf.default(mul_88);  mul_88 = None
    add_100: "f32[1, 512, 128]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_89: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_87, add_100);  mul_87 = add_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:662, code: hidden_states = self.LayerNorm(hidden_states)
    var_mean_25 = torch.ops.aten.var_mean.correction(mul_89, [2], correction = 0, keepdim = True)
    getitem_50: "f32[1, 512, 1]" = var_mean_25[0]
    getitem_51: "f32[1, 512, 1]" = var_mean_25[1];  var_mean_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1646, code: labels = labels[:, 1:].contiguous()
    slice_9: "i64[1, 511]" = torch.ops.aten.slice.Tensor(arg207_1, 1, 1, 9223372036854775807);  arg207_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1648, code: lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    view_271: "i64[511]" = torch.ops.aten.reshape.default(slice_9, [-1]);  slice_9 = None
    ne_1: "b8[511]" = torch.ops.aten.ne.Scalar(view_271, -100)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:662, code: hidden_states = self.LayerNorm(hidden_states)
    sub_38: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(mul_89, getitem_51);  mul_89 = getitem_51 = None
    add_101: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
    rsqrt_25: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
    mul_90: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_25);  sub_38 = rsqrt_25 = None
    mul_91: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_90, arg201_1);  mul_90 = arg201_1 = None
    add_102: "f32[1, 512, 128]" = torch.ops.aten.add.Tensor(mul_91, arg202_1);  mul_91 = arg202_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1640, code: prediction_scores = self.generator_lm_head(self.generator_predictions(sequence_output))
    view_268: "f32[512, 128]" = torch.ops.aten.reshape.default(add_102, [512, 128]);  add_102 = None
    permute_134: "f32[128, 30522]" = torch.ops.aten.permute.default(arg203_1, [1, 0]);  arg203_1 = None
    addmm_74: "f32[512, 30522]" = torch.ops.aten.addmm.default(arg204_1, view_268, permute_134);  arg204_1 = view_268 = permute_134 = None
    view_269: "f32[1, 512, 30522]" = torch.ops.aten.reshape.default(addmm_74, [1, 512, 30522]);  addmm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1645, code: shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
    slice_5: "f32[1, 512, 30522]" = torch.ops.aten.slice.Tensor(view_269, 0, 0, 9223372036854775807)
    slice_6: "f32[1, 511, 30522]" = torch.ops.aten.slice.Tensor(slice_5, 1, 0, -1);  slice_5 = None
    slice_7: "f32[1, 511, 30522]" = torch.ops.aten.slice.Tensor(slice_6, 2, 0, 9223372036854775807);  slice_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1648, code: lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    view_270: "f32[511, 30522]" = torch.ops.aten.reshape.default(slice_7, [-1, 30522]);  slice_7 = None
    amax_12: "f32[511, 1]" = torch.ops.aten.amax.default(view_270, [1], True)
    sub_39: "f32[511, 30522]" = torch.ops.aten.sub.Tensor(view_270, amax_12);  view_270 = amax_12 = None
    exp_12: "f32[511, 30522]" = torch.ops.aten.exp.default(sub_39)
    sum_13: "f32[511, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [1], True);  exp_12 = None
    log: "f32[511, 1]" = torch.ops.aten.log.default(sum_13);  sum_13 = None
    sub_40: "f32[511, 30522]" = torch.ops.aten.sub.Tensor(sub_39, log);  sub_39 = log = None
    ne: "b8[511]" = torch.ops.aten.ne.Scalar(view_271, -100)
    full_default_1: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where: "i64[511]" = torch.ops.aten.where.self(ne, view_271, full_default_1);  ne = full_default_1 = None
    unsqueeze_2: "i64[511, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
    gather: "f32[511, 1]" = torch.ops.aten.gather.default(sub_40, 1, unsqueeze_2);  sub_40 = unsqueeze_2 = None
    squeeze: "f32[511]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[511]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
    full_default_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where_1: "f32[511]" = torch.ops.aten.where.self(ne_1, neg, full_default_2);  ne_1 = neg = full_default_2 = None
    sum_15: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
    ne_2: "b8[511]" = torch.ops.aten.ne.Scalar(view_271, -100);  view_271 = None
    sum_14: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None
    div_24: "f32[]" = torch.ops.aten.div.Tensor(sum_15, convert_element_type);  sum_15 = convert_element_type = None
    return (div_24, view_269)
    