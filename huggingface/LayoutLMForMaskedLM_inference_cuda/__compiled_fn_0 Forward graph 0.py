from __future__ import annotations



def forward(self, arg0_1: "f32[30522, 768]", arg1_1: "f32[512, 768]", arg2_1: "f32[1024, 768]", arg3_1: "f32[1024, 768]", arg4_1: "f32[1024, 768]", arg5_1: "f32[1024, 768]", arg6_1: "f32[2, 768]", arg7_1: "f32[768]", arg8_1: "f32[768]", arg9_1: "f32[768, 768]", arg10_1: "f32[768]", arg11_1: "f32[768, 768]", arg12_1: "f32[768]", arg13_1: "f32[768, 768]", arg14_1: "f32[768]", arg15_1: "f32[768, 768]", arg16_1: "f32[768]", arg17_1: "f32[768]", arg18_1: "f32[768]", arg19_1: "f32[3072, 768]", arg20_1: "f32[3072]", arg21_1: "f32[768, 3072]", arg22_1: "f32[768]", arg23_1: "f32[768]", arg24_1: "f32[768]", arg25_1: "f32[768, 768]", arg26_1: "f32[768]", arg27_1: "f32[768, 768]", arg28_1: "f32[768]", arg29_1: "f32[768, 768]", arg30_1: "f32[768]", arg31_1: "f32[768, 768]", arg32_1: "f32[768]", arg33_1: "f32[768]", arg34_1: "f32[768]", arg35_1: "f32[3072, 768]", arg36_1: "f32[3072]", arg37_1: "f32[768, 3072]", arg38_1: "f32[768]", arg39_1: "f32[768]", arg40_1: "f32[768]", arg41_1: "f32[768, 768]", arg42_1: "f32[768]", arg43_1: "f32[768, 768]", arg44_1: "f32[768]", arg45_1: "f32[768, 768]", arg46_1: "f32[768]", arg47_1: "f32[768, 768]", arg48_1: "f32[768]", arg49_1: "f32[768]", arg50_1: "f32[768]", arg51_1: "f32[3072, 768]", arg52_1: "f32[3072]", arg53_1: "f32[768, 3072]", arg54_1: "f32[768]", arg55_1: "f32[768]", arg56_1: "f32[768]", arg57_1: "f32[768, 768]", arg58_1: "f32[768]", arg59_1: "f32[768, 768]", arg60_1: "f32[768]", arg61_1: "f32[768, 768]", arg62_1: "f32[768]", arg63_1: "f32[768, 768]", arg64_1: "f32[768]", arg65_1: "f32[768]", arg66_1: "f32[768]", arg67_1: "f32[3072, 768]", arg68_1: "f32[3072]", arg69_1: "f32[768, 3072]", arg70_1: "f32[768]", arg71_1: "f32[768]", arg72_1: "f32[768]", arg73_1: "f32[768, 768]", arg74_1: "f32[768]", arg75_1: "f32[768, 768]", arg76_1: "f32[768]", arg77_1: "f32[768, 768]", arg78_1: "f32[768]", arg79_1: "f32[768, 768]", arg80_1: "f32[768]", arg81_1: "f32[768]", arg82_1: "f32[768]", arg83_1: "f32[3072, 768]", arg84_1: "f32[3072]", arg85_1: "f32[768, 3072]", arg86_1: "f32[768]", arg87_1: "f32[768]", arg88_1: "f32[768]", arg89_1: "f32[768, 768]", arg90_1: "f32[768]", arg91_1: "f32[768, 768]", arg92_1: "f32[768]", arg93_1: "f32[768, 768]", arg94_1: "f32[768]", arg95_1: "f32[768, 768]", arg96_1: "f32[768]", arg97_1: "f32[768]", arg98_1: "f32[768]", arg99_1: "f32[3072, 768]", arg100_1: "f32[3072]", arg101_1: "f32[768, 3072]", arg102_1: "f32[768]", arg103_1: "f32[768]", arg104_1: "f32[768]", arg105_1: "f32[768, 768]", arg106_1: "f32[768]", arg107_1: "f32[768, 768]", arg108_1: "f32[768]", arg109_1: "f32[768, 768]", arg110_1: "f32[768]", arg111_1: "f32[768, 768]", arg112_1: "f32[768]", arg113_1: "f32[768]", arg114_1: "f32[768]", arg115_1: "f32[3072, 768]", arg116_1: "f32[3072]", arg117_1: "f32[768, 3072]", arg118_1: "f32[768]", arg119_1: "f32[768]", arg120_1: "f32[768]", arg121_1: "f32[768, 768]", arg122_1: "f32[768]", arg123_1: "f32[768, 768]", arg124_1: "f32[768]", arg125_1: "f32[768, 768]", arg126_1: "f32[768]", arg127_1: "f32[768, 768]", arg128_1: "f32[768]", arg129_1: "f32[768]", arg130_1: "f32[768]", arg131_1: "f32[3072, 768]", arg132_1: "f32[3072]", arg133_1: "f32[768, 3072]", arg134_1: "f32[768]", arg135_1: "f32[768]", arg136_1: "f32[768]", arg137_1: "f32[768, 768]", arg138_1: "f32[768]", arg139_1: "f32[768, 768]", arg140_1: "f32[768]", arg141_1: "f32[768, 768]", arg142_1: "f32[768]", arg143_1: "f32[768, 768]", arg144_1: "f32[768]", arg145_1: "f32[768]", arg146_1: "f32[768]", arg147_1: "f32[3072, 768]", arg148_1: "f32[3072]", arg149_1: "f32[768, 3072]", arg150_1: "f32[768]", arg151_1: "f32[768]", arg152_1: "f32[768]", arg153_1: "f32[768, 768]", arg154_1: "f32[768]", arg155_1: "f32[768, 768]", arg156_1: "f32[768]", arg157_1: "f32[768, 768]", arg158_1: "f32[768]", arg159_1: "f32[768, 768]", arg160_1: "f32[768]", arg161_1: "f32[768]", arg162_1: "f32[768]", arg163_1: "f32[3072, 768]", arg164_1: "f32[3072]", arg165_1: "f32[768, 3072]", arg166_1: "f32[768]", arg167_1: "f32[768]", arg168_1: "f32[768]", arg169_1: "f32[768, 768]", arg170_1: "f32[768]", arg171_1: "f32[768, 768]", arg172_1: "f32[768]", arg173_1: "f32[768, 768]", arg174_1: "f32[768]", arg175_1: "f32[768, 768]", arg176_1: "f32[768]", arg177_1: "f32[768]", arg178_1: "f32[768]", arg179_1: "f32[3072, 768]", arg180_1: "f32[3072]", arg181_1: "f32[768, 3072]", arg182_1: "f32[768]", arg183_1: "f32[768]", arg184_1: "f32[768]", arg185_1: "f32[768, 768]", arg186_1: "f32[768]", arg187_1: "f32[768, 768]", arg188_1: "f32[768]", arg189_1: "f32[768, 768]", arg190_1: "f32[768]", arg191_1: "f32[768, 768]", arg192_1: "f32[768]", arg193_1: "f32[768]", arg194_1: "f32[768]", arg195_1: "f32[3072, 768]", arg196_1: "f32[3072]", arg197_1: "f32[768, 3072]", arg198_1: "f32[768]", arg199_1: "f32[768]", arg200_1: "f32[768]", arg201_1: "f32[768, 768]", arg202_1: "f32[768]", arg203_1: "f32[768, 768]", arg204_1: "f32[768]", arg205_1: "f32[768]", arg206_1: "f32[768]", arg207_1: "f32[30522, 768]", arg208_1: "f32[30522]", arg209_1: "i64[1, 512]", arg210_1: "i64[1, 512]", arg211_1: "i64[1, 512]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:808, code: attention_mask = torch.ones(input_shape, device=device)
    full: "f32[1, 512]" = torch.ops.aten.full.default([1, 512], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:810, code: token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
    full_1: "i64[1, 512]" = torch.ops.aten.full.default([1, 512], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:813, code: bbox = torch.zeros(input_shape + (4,), dtype=torch.long, device=device)
    full_2: "i64[1, 512, 4]" = torch.ops.aten.full.default([1, 512, 4], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:815, code: extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    unsqueeze: "f32[1, 1, 512]" = torch.ops.aten.unsqueeze.default(full, 1);  full = None
    unsqueeze_1: "f32[1, 1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:818, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min
    sub: "f32[1, 1, 1, 512]" = torch.ops.aten.sub.Tensor(1.0, unsqueeze_1);  unsqueeze_1 = None
    mul: "f32[1, 1, 1, 512]" = torch.ops.aten.mul.Tensor(sub, -3.4028234663852886e+38);  sub = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:93, code: position_ids = self.position_ids[:, :seq_length]
    slice_1: "i64[1, 512]" = torch.ops.aten.slice.Tensor(arg209_1, 0, 0, 9223372036854775807);  arg209_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:99, code: inputs_embeds = self.word_embeddings(input_ids)
    embedding: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg0_1, arg210_1, 0);  arg0_1 = arg210_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:102, code: position_embeddings = self.position_embeddings(position_ids)
    embedding_1: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg1_1, slice_1);  arg1_1 = slice_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:104, code: left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
    slice_2: "i64[1, 512, 4]" = torch.ops.aten.slice.Tensor(full_2, 0, 0, 9223372036854775807)
    slice_3: "i64[1, 512, 4]" = torch.ops.aten.slice.Tensor(slice_2, 1, 0, 9223372036854775807);  slice_2 = None
    select: "i64[1, 512]" = torch.ops.aten.select.int(slice_3, 2, 0);  slice_3 = None
    embedding_2: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg2_1, select);  select = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:105, code: upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
    slice_4: "i64[1, 512, 4]" = torch.ops.aten.slice.Tensor(full_2, 0, 0, 9223372036854775807)
    slice_5: "i64[1, 512, 4]" = torch.ops.aten.slice.Tensor(slice_4, 1, 0, 9223372036854775807);  slice_4 = None
    select_1: "i64[1, 512]" = torch.ops.aten.select.int(slice_5, 2, 1);  slice_5 = None
    embedding_3: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg3_1, select_1);  select_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:106, code: right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
    slice_6: "i64[1, 512, 4]" = torch.ops.aten.slice.Tensor(full_2, 0, 0, 9223372036854775807)
    slice_7: "i64[1, 512, 4]" = torch.ops.aten.slice.Tensor(slice_6, 1, 0, 9223372036854775807);  slice_6 = None
    select_2: "i64[1, 512]" = torch.ops.aten.select.int(slice_7, 2, 2);  slice_7 = None
    embedding_4: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg2_1, select_2);  arg2_1 = select_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:107, code: lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
    slice_8: "i64[1, 512, 4]" = torch.ops.aten.slice.Tensor(full_2, 0, 0, 9223372036854775807)
    slice_9: "i64[1, 512, 4]" = torch.ops.aten.slice.Tensor(slice_8, 1, 0, 9223372036854775807);  slice_8 = None
    select_3: "i64[1, 512]" = torch.ops.aten.select.int(slice_9, 2, 3);  slice_9 = None
    embedding_5: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg3_1, select_3);  arg3_1 = select_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:111, code: h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
    slice_10: "i64[1, 512, 4]" = torch.ops.aten.slice.Tensor(full_2, 0, 0, 9223372036854775807)
    slice_11: "i64[1, 512, 4]" = torch.ops.aten.slice.Tensor(slice_10, 1, 0, 9223372036854775807);  slice_10 = None
    select_4: "i64[1, 512]" = torch.ops.aten.select.int(slice_11, 2, 3);  slice_11 = None
    slice_12: "i64[1, 512, 4]" = torch.ops.aten.slice.Tensor(full_2, 0, 0, 9223372036854775807)
    slice_13: "i64[1, 512, 4]" = torch.ops.aten.slice.Tensor(slice_12, 1, 0, 9223372036854775807);  slice_12 = None
    select_5: "i64[1, 512]" = torch.ops.aten.select.int(slice_13, 2, 1);  slice_13 = None
    sub_1: "i64[1, 512]" = torch.ops.aten.sub.Tensor(select_4, select_5);  select_4 = select_5 = None
    embedding_6: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg4_1, sub_1);  arg4_1 = sub_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:112, code: w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])
    slice_14: "i64[1, 512, 4]" = torch.ops.aten.slice.Tensor(full_2, 0, 0, 9223372036854775807)
    slice_15: "i64[1, 512, 4]" = torch.ops.aten.slice.Tensor(slice_14, 1, 0, 9223372036854775807);  slice_14 = None
    select_6: "i64[1, 512]" = torch.ops.aten.select.int(slice_15, 2, 2);  slice_15 = None
    slice_16: "i64[1, 512, 4]" = torch.ops.aten.slice.Tensor(full_2, 0, 0, 9223372036854775807);  full_2 = None
    slice_17: "i64[1, 512, 4]" = torch.ops.aten.slice.Tensor(slice_16, 1, 0, 9223372036854775807);  slice_16 = None
    select_7: "i64[1, 512]" = torch.ops.aten.select.int(slice_17, 2, 0);  slice_17 = None
    sub_2: "i64[1, 512]" = torch.ops.aten.sub.Tensor(select_6, select_7);  select_6 = select_7 = None
    embedding_7: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg5_1, sub_2);  arg5_1 = sub_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:113, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    embedding_8: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg6_1, full_1);  arg6_1 = full_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:116, code: words_embeddings
    add: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
    add_1: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
    add_2: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_1, embedding_3);  add_1 = embedding_3 = None
    add_3: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_2, embedding_4);  add_2 = embedding_4 = None
    add_4: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_3, embedding_5);  add_3 = embedding_5 = None
    add_5: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_4, embedding_6);  add_4 = embedding_6 = None
    add_6: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_5, embedding_7);  add_5 = embedding_7 = None
    add_7: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_6, embedding_8);  add_6 = embedding_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:126, code: embeddings = self.LayerNorm(embeddings)
    var_mean = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 512, 1]" = var_mean[0]
    getitem_1: "f32[1, 512, 1]" = var_mean[1];  var_mean = None
    add_8: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
    rsqrt: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
    sub_3: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_7, getitem_1);  add_7 = getitem_1 = None
    mul_1: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt);  sub_3 = rsqrt = None
    mul_2: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_1, arg7_1);  mul_1 = arg7_1 = None
    add_9: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_2, arg8_1);  mul_2 = arg8_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:127, code: embeddings = self.dropout(embeddings)
    clone: "f32[1, 512, 768]" = torch.ops.aten.clone.default(add_9);  add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view: "f32[512, 768]" = torch.ops.aten.view.default(clone, [512, 768])
    permute: "f32[768, 768]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
    addmm: "f32[512, 768]" = torch.ops.aten.addmm.default(arg10_1, view, permute);  arg10_1 = view = permute = None
    view_1: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm, [1, 512, 768]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_2: "f32[512, 768]" = torch.ops.aten.view.default(clone, [512, 768])
    permute_1: "f32[768, 768]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
    addmm_1: "f32[512, 768]" = torch.ops.aten.addmm.default(arg12_1, view_2, permute_1);  arg12_1 = view_2 = permute_1 = None
    view_3: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_1, [1, 512, 768]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_4: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_3, [1, 512, 12, 64]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_2: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_5: "f32[512, 768]" = torch.ops.aten.view.default(clone, [512, 768])
    permute_3: "f32[768, 768]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
    addmm_2: "f32[512, 768]" = torch.ops.aten.addmm.default(arg14_1, view_5, permute_3);  arg14_1 = view_5 = permute_3 = None
    view_6: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_2, [1, 512, 768]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_7: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_6, [1, 512, 12, 64]);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_4: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_8: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_1, [1, 512, 12, 64]);  view_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_5: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_6: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_2, [0, 1, 3, 2]);  permute_2 = None
    expand: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_5, [1, 12, 512, 64]);  permute_5 = None
    view_9: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand, [12, 512, 64]);  expand = None
    expand_1: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_6, [1, 12, 64, 512]);  permute_6 = None
    view_10: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_1, [12, 64, 512]);  expand_1 = None
    bmm: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_9, view_10);  view_9 = view_10 = None
    view_11: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm, [1, 12, 512, 512]);  bmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:237, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_11, 8.0);  view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:240, code: attention_scores = attention_scores + attention_mask
    add_10: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(div, mul);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:243, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_10, [-1], True)
    sub_4: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_10, amax);  add_10 = amax = None
    exp: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
    sum_1: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div_1: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:247, code: attention_probs = self.dropout(attention_probs)
    clone_1: "f32[1, 12, 512, 512]" = torch.ops.aten.clone.default(div_1);  div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:253, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_2: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(clone_1, [1, 12, 512, 512]);  clone_1 = None
    view_12: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_2, [12, 512, 512]);  expand_2 = None
    expand_3: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_4, [1, 12, 512, 64]);  permute_4 = None
    view_13: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_3, [12, 512, 64]);  expand_3 = None
    bmm_1: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_12, view_13);  view_12 = view_13 = None
    view_14: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_1, [1, 12, 512, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_7: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
    clone_2: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_15: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_2, [1, 512, 768]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_16: "f32[512, 768]" = torch.ops.aten.view.default(view_15, [512, 768]);  view_15 = None
    permute_8: "f32[768, 768]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
    addmm_3: "f32[512, 768]" = torch.ops.aten.addmm.default(arg16_1, view_16, permute_8);  arg16_1 = view_16 = permute_8 = None
    view_17: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_3, [1, 512, 768]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    clone_3: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_17);  view_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_11: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_3, clone);  clone_3 = clone = None
    var_mean_1 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
    getitem_2: "f32[1, 512, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 512, 1]" = var_mean_1[1];  var_mean_1 = None
    add_12: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
    rsqrt_1: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
    sub_5: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_11, getitem_3);  add_11 = getitem_3 = None
    mul_3: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_1);  sub_5 = rsqrt_1 = None
    mul_4: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_3, arg17_1);  mul_3 = arg17_1 = None
    add_13: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_4, arg18_1);  mul_4 = arg18_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_18: "f32[512, 768]" = torch.ops.aten.view.default(add_13, [512, 768])
    permute_9: "f32[768, 3072]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
    addmm_4: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg20_1, view_18, permute_9);  arg20_1 = view_18 = permute_9 = None
    view_19: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_4, [1, 512, 3072]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_5: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.5)
    mul_6: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476);  view_19 = None
    erf: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
    add_14: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_7: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_5, add_14);  mul_5 = add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_20: "f32[512, 3072]" = torch.ops.aten.view.default(mul_7, [512, 3072]);  mul_7 = None
    permute_10: "f32[3072, 768]" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
    addmm_5: "f32[512, 768]" = torch.ops.aten.addmm.default(arg22_1, view_20, permute_10);  arg22_1 = view_20 = permute_10 = None
    view_21: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_5, [1, 512, 768]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    clone_4: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_21);  view_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_15: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_4, add_13);  clone_4 = add_13 = None
    var_mean_2 = torch.ops.aten.var_mean.correction(add_15, [2], correction = 0, keepdim = True)
    getitem_4: "f32[1, 512, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 512, 1]" = var_mean_2[1];  var_mean_2 = None
    add_16: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
    rsqrt_2: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    sub_6: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_15, getitem_5);  add_15 = getitem_5 = None
    mul_8: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_2);  sub_6 = rsqrt_2 = None
    mul_9: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_8, arg23_1);  mul_8 = arg23_1 = None
    add_17: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_9, arg24_1);  mul_9 = arg24_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_22: "f32[512, 768]" = torch.ops.aten.view.default(add_17, [512, 768])
    permute_11: "f32[768, 768]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
    addmm_6: "f32[512, 768]" = torch.ops.aten.addmm.default(arg26_1, view_22, permute_11);  arg26_1 = view_22 = permute_11 = None
    view_23: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_6, [1, 512, 768]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_24: "f32[512, 768]" = torch.ops.aten.view.default(add_17, [512, 768])
    permute_12: "f32[768, 768]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
    addmm_7: "f32[512, 768]" = torch.ops.aten.addmm.default(arg28_1, view_24, permute_12);  arg28_1 = view_24 = permute_12 = None
    view_25: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_7, [1, 512, 768]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_26: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_25, [1, 512, 12, 64]);  view_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_13: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_27: "f32[512, 768]" = torch.ops.aten.view.default(add_17, [512, 768])
    permute_14: "f32[768, 768]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
    addmm_8: "f32[512, 768]" = torch.ops.aten.addmm.default(arg30_1, view_27, permute_14);  arg30_1 = view_27 = permute_14 = None
    view_28: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_8, [1, 512, 768]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_29: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_28, [1, 512, 12, 64]);  view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_15: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_30: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_23, [1, 512, 12, 64]);  view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_16: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_17: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_13, [0, 1, 3, 2]);  permute_13 = None
    expand_4: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_16, [1, 12, 512, 64]);  permute_16 = None
    view_31: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_4, [12, 512, 64]);  expand_4 = None
    expand_5: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_17, [1, 12, 64, 512]);  permute_17 = None
    view_32: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_5, [12, 64, 512]);  expand_5 = None
    bmm_2: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_31, view_32);  view_31 = view_32 = None
    view_33: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_2, [1, 12, 512, 512]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:237, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_2: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_33, 8.0);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:240, code: attention_scores = attention_scores + attention_mask
    add_18: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(div_2, mul);  div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:243, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_1: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_18, [-1], True)
    sub_7: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_18, amax_1);  add_18 = amax_1 = None
    exp_1: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
    sum_2: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_3: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:247, code: attention_probs = self.dropout(attention_probs)
    clone_5: "f32[1, 12, 512, 512]" = torch.ops.aten.clone.default(div_3);  div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:253, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_6: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(clone_5, [1, 12, 512, 512]);  clone_5 = None
    view_34: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_6, [12, 512, 512]);  expand_6 = None
    expand_7: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_15, [1, 12, 512, 64]);  permute_15 = None
    view_35: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_7, [12, 512, 64]);  expand_7 = None
    bmm_3: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_34, view_35);  view_34 = view_35 = None
    view_36: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_3, [1, 12, 512, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_18: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None
    clone_6: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_37: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_6, [1, 512, 768]);  clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_38: "f32[512, 768]" = torch.ops.aten.view.default(view_37, [512, 768]);  view_37 = None
    permute_19: "f32[768, 768]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
    addmm_9: "f32[512, 768]" = torch.ops.aten.addmm.default(arg32_1, view_38, permute_19);  arg32_1 = view_38 = permute_19 = None
    view_39: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_9, [1, 512, 768]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    clone_7: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_39);  view_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_19: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_7, add_17);  clone_7 = add_17 = None
    var_mean_3 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
    getitem_6: "f32[1, 512, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 512, 1]" = var_mean_3[1];  var_mean_3 = None
    add_20: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
    rsqrt_3: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
    sub_8: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_19, getitem_7);  add_19 = getitem_7 = None
    mul_10: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_3);  sub_8 = rsqrt_3 = None
    mul_11: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_10, arg33_1);  mul_10 = arg33_1 = None
    add_21: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_11, arg34_1);  mul_11 = arg34_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_40: "f32[512, 768]" = torch.ops.aten.view.default(add_21, [512, 768])
    permute_20: "f32[768, 3072]" = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
    addmm_10: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg36_1, view_40, permute_20);  arg36_1 = view_40 = permute_20 = None
    view_41: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_10, [1, 512, 3072]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_12: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, 0.5)
    mul_13: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476);  view_41 = None
    erf_1: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
    add_22: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_14: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_12, add_22);  mul_12 = add_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_42: "f32[512, 3072]" = torch.ops.aten.view.default(mul_14, [512, 3072]);  mul_14 = None
    permute_21: "f32[3072, 768]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
    addmm_11: "f32[512, 768]" = torch.ops.aten.addmm.default(arg38_1, view_42, permute_21);  arg38_1 = view_42 = permute_21 = None
    view_43: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_11, [1, 512, 768]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    clone_8: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_43);  view_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_23: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_8, add_21);  clone_8 = add_21 = None
    var_mean_4 = torch.ops.aten.var_mean.correction(add_23, [2], correction = 0, keepdim = True)
    getitem_8: "f32[1, 512, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 512, 1]" = var_mean_4[1];  var_mean_4 = None
    add_24: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
    rsqrt_4: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
    sub_9: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_23, getitem_9);  add_23 = getitem_9 = None
    mul_15: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_4);  sub_9 = rsqrt_4 = None
    mul_16: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_15, arg39_1);  mul_15 = arg39_1 = None
    add_25: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_16, arg40_1);  mul_16 = arg40_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_44: "f32[512, 768]" = torch.ops.aten.view.default(add_25, [512, 768])
    permute_22: "f32[768, 768]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
    addmm_12: "f32[512, 768]" = torch.ops.aten.addmm.default(arg42_1, view_44, permute_22);  arg42_1 = view_44 = permute_22 = None
    view_45: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_12, [1, 512, 768]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_46: "f32[512, 768]" = torch.ops.aten.view.default(add_25, [512, 768])
    permute_23: "f32[768, 768]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
    addmm_13: "f32[512, 768]" = torch.ops.aten.addmm.default(arg44_1, view_46, permute_23);  arg44_1 = view_46 = permute_23 = None
    view_47: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_13, [1, 512, 768]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_48: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_47, [1, 512, 12, 64]);  view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_24: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_49: "f32[512, 768]" = torch.ops.aten.view.default(add_25, [512, 768])
    permute_25: "f32[768, 768]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
    addmm_14: "f32[512, 768]" = torch.ops.aten.addmm.default(arg46_1, view_49, permute_25);  arg46_1 = view_49 = permute_25 = None
    view_50: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_14, [1, 512, 768]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_51: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_50, [1, 512, 12, 64]);  view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_26: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_52: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_45, [1, 512, 12, 64]);  view_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_27: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_28: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_24, [0, 1, 3, 2]);  permute_24 = None
    expand_8: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_27, [1, 12, 512, 64]);  permute_27 = None
    view_53: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_8, [12, 512, 64]);  expand_8 = None
    expand_9: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_28, [1, 12, 64, 512]);  permute_28 = None
    view_54: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_9, [12, 64, 512]);  expand_9 = None
    bmm_4: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_53, view_54);  view_53 = view_54 = None
    view_55: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_4, [1, 12, 512, 512]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:237, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_4: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_55, 8.0);  view_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:240, code: attention_scores = attention_scores + attention_mask
    add_26: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(div_4, mul);  div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:243, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_2: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_26, [-1], True)
    sub_10: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_26, amax_2);  add_26 = amax_2 = None
    exp_2: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
    sum_3: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_5: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:247, code: attention_probs = self.dropout(attention_probs)
    clone_9: "f32[1, 12, 512, 512]" = torch.ops.aten.clone.default(div_5);  div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:253, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_10: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(clone_9, [1, 12, 512, 512]);  clone_9 = None
    view_56: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_10, [12, 512, 512]);  expand_10 = None
    expand_11: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_26, [1, 12, 512, 64]);  permute_26 = None
    view_57: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_11, [12, 512, 64]);  expand_11 = None
    bmm_5: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_56, view_57);  view_56 = view_57 = None
    view_58: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_5, [1, 12, 512, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_29: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
    clone_10: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_59: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_10, [1, 512, 768]);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_60: "f32[512, 768]" = torch.ops.aten.view.default(view_59, [512, 768]);  view_59 = None
    permute_30: "f32[768, 768]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
    addmm_15: "f32[512, 768]" = torch.ops.aten.addmm.default(arg48_1, view_60, permute_30);  arg48_1 = view_60 = permute_30 = None
    view_61: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_15, [1, 512, 768]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    clone_11: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_61);  view_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_27: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_11, add_25);  clone_11 = add_25 = None
    var_mean_5 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
    getitem_10: "f32[1, 512, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 512, 1]" = var_mean_5[1];  var_mean_5 = None
    add_28: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
    rsqrt_5: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    sub_11: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_27, getitem_11);  add_27 = getitem_11 = None
    mul_17: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_5);  sub_11 = rsqrt_5 = None
    mul_18: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_17, arg49_1);  mul_17 = arg49_1 = None
    add_29: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_18, arg50_1);  mul_18 = arg50_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_62: "f32[512, 768]" = torch.ops.aten.view.default(add_29, [512, 768])
    permute_31: "f32[768, 3072]" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
    addmm_16: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg52_1, view_62, permute_31);  arg52_1 = view_62 = permute_31 = None
    view_63: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_16, [1, 512, 3072]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_19: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, 0.5)
    mul_20: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476);  view_63 = None
    erf_2: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_20);  mul_20 = None
    add_30: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_21: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_19, add_30);  mul_19 = add_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_64: "f32[512, 3072]" = torch.ops.aten.view.default(mul_21, [512, 3072]);  mul_21 = None
    permute_32: "f32[3072, 768]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
    addmm_17: "f32[512, 768]" = torch.ops.aten.addmm.default(arg54_1, view_64, permute_32);  arg54_1 = view_64 = permute_32 = None
    view_65: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_17, [1, 512, 768]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    clone_12: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_65);  view_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_31: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_12, add_29);  clone_12 = add_29 = None
    var_mean_6 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
    getitem_12: "f32[1, 512, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 512, 1]" = var_mean_6[1];  var_mean_6 = None
    add_32: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
    rsqrt_6: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    sub_12: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_31, getitem_13);  add_31 = getitem_13 = None
    mul_22: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_6);  sub_12 = rsqrt_6 = None
    mul_23: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_22, arg55_1);  mul_22 = arg55_1 = None
    add_33: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_23, arg56_1);  mul_23 = arg56_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_66: "f32[512, 768]" = torch.ops.aten.view.default(add_33, [512, 768])
    permute_33: "f32[768, 768]" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
    addmm_18: "f32[512, 768]" = torch.ops.aten.addmm.default(arg58_1, view_66, permute_33);  arg58_1 = view_66 = permute_33 = None
    view_67: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_18, [1, 512, 768]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_68: "f32[512, 768]" = torch.ops.aten.view.default(add_33, [512, 768])
    permute_34: "f32[768, 768]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
    addmm_19: "f32[512, 768]" = torch.ops.aten.addmm.default(arg60_1, view_68, permute_34);  arg60_1 = view_68 = permute_34 = None
    view_69: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_19, [1, 512, 768]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_70: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_69, [1, 512, 12, 64]);  view_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_35: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_71: "f32[512, 768]" = torch.ops.aten.view.default(add_33, [512, 768])
    permute_36: "f32[768, 768]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
    addmm_20: "f32[512, 768]" = torch.ops.aten.addmm.default(arg62_1, view_71, permute_36);  arg62_1 = view_71 = permute_36 = None
    view_72: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_20, [1, 512, 768]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_73: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_72, [1, 512, 12, 64]);  view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_37: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_74: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_67, [1, 512, 12, 64]);  view_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_38: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_39: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_35, [0, 1, 3, 2]);  permute_35 = None
    expand_12: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_38, [1, 12, 512, 64]);  permute_38 = None
    view_75: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_12, [12, 512, 64]);  expand_12 = None
    expand_13: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_39, [1, 12, 64, 512]);  permute_39 = None
    view_76: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_13, [12, 64, 512]);  expand_13 = None
    bmm_6: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_75, view_76);  view_75 = view_76 = None
    view_77: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_6, [1, 12, 512, 512]);  bmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:237, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_6: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_77, 8.0);  view_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:240, code: attention_scores = attention_scores + attention_mask
    add_34: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(div_6, mul);  div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:243, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_3: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_34, [-1], True)
    sub_13: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_34, amax_3);  add_34 = amax_3 = None
    exp_3: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
    sum_4: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_7: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:247, code: attention_probs = self.dropout(attention_probs)
    clone_13: "f32[1, 12, 512, 512]" = torch.ops.aten.clone.default(div_7);  div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:253, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_14: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(clone_13, [1, 12, 512, 512]);  clone_13 = None
    view_78: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_14, [12, 512, 512]);  expand_14 = None
    expand_15: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_37, [1, 12, 512, 64]);  permute_37 = None
    view_79: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_15, [12, 512, 64]);  expand_15 = None
    bmm_7: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_78, view_79);  view_78 = view_79 = None
    view_80: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_7, [1, 12, 512, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_40: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_80, [0, 2, 1, 3]);  view_80 = None
    clone_14: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_81: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_14, [1, 512, 768]);  clone_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_82: "f32[512, 768]" = torch.ops.aten.view.default(view_81, [512, 768]);  view_81 = None
    permute_41: "f32[768, 768]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
    addmm_21: "f32[512, 768]" = torch.ops.aten.addmm.default(arg64_1, view_82, permute_41);  arg64_1 = view_82 = permute_41 = None
    view_83: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_21, [1, 512, 768]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    clone_15: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_83);  view_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_35: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_15, add_33);  clone_15 = add_33 = None
    var_mean_7 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
    getitem_14: "f32[1, 512, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 512, 1]" = var_mean_7[1];  var_mean_7 = None
    add_36: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
    rsqrt_7: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    sub_14: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_15);  add_35 = getitem_15 = None
    mul_24: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_7);  sub_14 = rsqrt_7 = None
    mul_25: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_24, arg65_1);  mul_24 = arg65_1 = None
    add_37: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_25, arg66_1);  mul_25 = arg66_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_84: "f32[512, 768]" = torch.ops.aten.view.default(add_37, [512, 768])
    permute_42: "f32[768, 3072]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
    addmm_22: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg68_1, view_84, permute_42);  arg68_1 = view_84 = permute_42 = None
    view_85: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_22, [1, 512, 3072]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_26: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, 0.5)
    mul_27: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476);  view_85 = None
    erf_3: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_27);  mul_27 = None
    add_38: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_28: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_26, add_38);  mul_26 = add_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_86: "f32[512, 3072]" = torch.ops.aten.view.default(mul_28, [512, 3072]);  mul_28 = None
    permute_43: "f32[3072, 768]" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
    addmm_23: "f32[512, 768]" = torch.ops.aten.addmm.default(arg70_1, view_86, permute_43);  arg70_1 = view_86 = permute_43 = None
    view_87: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_23, [1, 512, 768]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    clone_16: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_87);  view_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_39: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_16, add_37);  clone_16 = add_37 = None
    var_mean_8 = torch.ops.aten.var_mean.correction(add_39, [2], correction = 0, keepdim = True)
    getitem_16: "f32[1, 512, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 512, 1]" = var_mean_8[1];  var_mean_8 = None
    add_40: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
    rsqrt_8: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
    sub_15: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_39, getitem_17);  add_39 = getitem_17 = None
    mul_29: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_8);  sub_15 = rsqrt_8 = None
    mul_30: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_29, arg71_1);  mul_29 = arg71_1 = None
    add_41: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_30, arg72_1);  mul_30 = arg72_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_88: "f32[512, 768]" = torch.ops.aten.view.default(add_41, [512, 768])
    permute_44: "f32[768, 768]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
    addmm_24: "f32[512, 768]" = torch.ops.aten.addmm.default(arg74_1, view_88, permute_44);  arg74_1 = view_88 = permute_44 = None
    view_89: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_24, [1, 512, 768]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_90: "f32[512, 768]" = torch.ops.aten.view.default(add_41, [512, 768])
    permute_45: "f32[768, 768]" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
    addmm_25: "f32[512, 768]" = torch.ops.aten.addmm.default(arg76_1, view_90, permute_45);  arg76_1 = view_90 = permute_45 = None
    view_91: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_25, [1, 512, 768]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_92: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_91, [1, 512, 12, 64]);  view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_46: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_93: "f32[512, 768]" = torch.ops.aten.view.default(add_41, [512, 768])
    permute_47: "f32[768, 768]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
    addmm_26: "f32[512, 768]" = torch.ops.aten.addmm.default(arg78_1, view_93, permute_47);  arg78_1 = view_93 = permute_47 = None
    view_94: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_26, [1, 512, 768]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_95: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_94, [1, 512, 12, 64]);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_48: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_96: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_89, [1, 512, 12, 64]);  view_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_49: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_50: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_46, [0, 1, 3, 2]);  permute_46 = None
    expand_16: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_49, [1, 12, 512, 64]);  permute_49 = None
    view_97: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_16, [12, 512, 64]);  expand_16 = None
    expand_17: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_50, [1, 12, 64, 512]);  permute_50 = None
    view_98: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_17, [12, 64, 512]);  expand_17 = None
    bmm_8: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_97, view_98);  view_97 = view_98 = None
    view_99: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_8, [1, 12, 512, 512]);  bmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:237, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_8: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_99, 8.0);  view_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:240, code: attention_scores = attention_scores + attention_mask
    add_42: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(div_8, mul);  div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:243, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_4: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_42, [-1], True)
    sub_16: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_42, amax_4);  add_42 = amax_4 = None
    exp_4: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
    sum_5: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_9: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:247, code: attention_probs = self.dropout(attention_probs)
    clone_17: "f32[1, 12, 512, 512]" = torch.ops.aten.clone.default(div_9);  div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:253, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_18: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(clone_17, [1, 12, 512, 512]);  clone_17 = None
    view_100: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_18, [12, 512, 512]);  expand_18 = None
    expand_19: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_48, [1, 12, 512, 64]);  permute_48 = None
    view_101: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_19, [12, 512, 64]);  expand_19 = None
    bmm_9: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_100, view_101);  view_100 = view_101 = None
    view_102: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_9, [1, 12, 512, 64]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_51: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
    clone_18: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_103: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_18, [1, 512, 768]);  clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_104: "f32[512, 768]" = torch.ops.aten.view.default(view_103, [512, 768]);  view_103 = None
    permute_52: "f32[768, 768]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
    addmm_27: "f32[512, 768]" = torch.ops.aten.addmm.default(arg80_1, view_104, permute_52);  arg80_1 = view_104 = permute_52 = None
    view_105: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_27, [1, 512, 768]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    clone_19: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_105);  view_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_43: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_19, add_41);  clone_19 = add_41 = None
    var_mean_9 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
    getitem_18: "f32[1, 512, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 512, 1]" = var_mean_9[1];  var_mean_9 = None
    add_44: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
    rsqrt_9: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
    sub_17: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_43, getitem_19);  add_43 = getitem_19 = None
    mul_31: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_9);  sub_17 = rsqrt_9 = None
    mul_32: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_31, arg81_1);  mul_31 = arg81_1 = None
    add_45: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_32, arg82_1);  mul_32 = arg82_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_106: "f32[512, 768]" = torch.ops.aten.view.default(add_45, [512, 768])
    permute_53: "f32[768, 3072]" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
    addmm_28: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg84_1, view_106, permute_53);  arg84_1 = view_106 = permute_53 = None
    view_107: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_28, [1, 512, 3072]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_33: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.5)
    mul_34: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476);  view_107 = None
    erf_4: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_34);  mul_34 = None
    add_46: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_35: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_33, add_46);  mul_33 = add_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_108: "f32[512, 3072]" = torch.ops.aten.view.default(mul_35, [512, 3072]);  mul_35 = None
    permute_54: "f32[3072, 768]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
    addmm_29: "f32[512, 768]" = torch.ops.aten.addmm.default(arg86_1, view_108, permute_54);  arg86_1 = view_108 = permute_54 = None
    view_109: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_29, [1, 512, 768]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    clone_20: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_109);  view_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_47: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_20, add_45);  clone_20 = add_45 = None
    var_mean_10 = torch.ops.aten.var_mean.correction(add_47, [2], correction = 0, keepdim = True)
    getitem_20: "f32[1, 512, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 512, 1]" = var_mean_10[1];  var_mean_10 = None
    add_48: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
    rsqrt_10: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
    sub_18: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_47, getitem_21);  add_47 = getitem_21 = None
    mul_36: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_10);  sub_18 = rsqrt_10 = None
    mul_37: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_36, arg87_1);  mul_36 = arg87_1 = None
    add_49: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_37, arg88_1);  mul_37 = arg88_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_110: "f32[512, 768]" = torch.ops.aten.view.default(add_49, [512, 768])
    permute_55: "f32[768, 768]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
    addmm_30: "f32[512, 768]" = torch.ops.aten.addmm.default(arg90_1, view_110, permute_55);  arg90_1 = view_110 = permute_55 = None
    view_111: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_30, [1, 512, 768]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_112: "f32[512, 768]" = torch.ops.aten.view.default(add_49, [512, 768])
    permute_56: "f32[768, 768]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
    addmm_31: "f32[512, 768]" = torch.ops.aten.addmm.default(arg92_1, view_112, permute_56);  arg92_1 = view_112 = permute_56 = None
    view_113: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_31, [1, 512, 768]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_114: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_113, [1, 512, 12, 64]);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_57: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_115: "f32[512, 768]" = torch.ops.aten.view.default(add_49, [512, 768])
    permute_58: "f32[768, 768]" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
    addmm_32: "f32[512, 768]" = torch.ops.aten.addmm.default(arg94_1, view_115, permute_58);  arg94_1 = view_115 = permute_58 = None
    view_116: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_32, [1, 512, 768]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_117: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_116, [1, 512, 12, 64]);  view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_59: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_118: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_111, [1, 512, 12, 64]);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_60: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_61: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_57, [0, 1, 3, 2]);  permute_57 = None
    expand_20: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_60, [1, 12, 512, 64]);  permute_60 = None
    view_119: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_20, [12, 512, 64]);  expand_20 = None
    expand_21: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_61, [1, 12, 64, 512]);  permute_61 = None
    view_120: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_21, [12, 64, 512]);  expand_21 = None
    bmm_10: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_119, view_120);  view_119 = view_120 = None
    view_121: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_10, [1, 12, 512, 512]);  bmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:237, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_10: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_121, 8.0);  view_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:240, code: attention_scores = attention_scores + attention_mask
    add_50: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(div_10, mul);  div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:243, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_5: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_50, [-1], True)
    sub_19: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_50, amax_5);  add_50 = amax_5 = None
    exp_5: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
    sum_6: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_11: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:247, code: attention_probs = self.dropout(attention_probs)
    clone_21: "f32[1, 12, 512, 512]" = torch.ops.aten.clone.default(div_11);  div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:253, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_22: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(clone_21, [1, 12, 512, 512]);  clone_21 = None
    view_122: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_22, [12, 512, 512]);  expand_22 = None
    expand_23: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_59, [1, 12, 512, 64]);  permute_59 = None
    view_123: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_23, [12, 512, 64]);  expand_23 = None
    bmm_11: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_122, view_123);  view_122 = view_123 = None
    view_124: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_11, [1, 12, 512, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_62: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None
    clone_22: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_125: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_22, [1, 512, 768]);  clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_126: "f32[512, 768]" = torch.ops.aten.view.default(view_125, [512, 768]);  view_125 = None
    permute_63: "f32[768, 768]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
    addmm_33: "f32[512, 768]" = torch.ops.aten.addmm.default(arg96_1, view_126, permute_63);  arg96_1 = view_126 = permute_63 = None
    view_127: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_33, [1, 512, 768]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    clone_23: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_127);  view_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_51: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_23, add_49);  clone_23 = add_49 = None
    var_mean_11 = torch.ops.aten.var_mean.correction(add_51, [2], correction = 0, keepdim = True)
    getitem_22: "f32[1, 512, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 512, 1]" = var_mean_11[1];  var_mean_11 = None
    add_52: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
    rsqrt_11: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    sub_20: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_51, getitem_23);  add_51 = getitem_23 = None
    mul_38: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_11);  sub_20 = rsqrt_11 = None
    mul_39: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_38, arg97_1);  mul_38 = arg97_1 = None
    add_53: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_39, arg98_1);  mul_39 = arg98_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_128: "f32[512, 768]" = torch.ops.aten.view.default(add_53, [512, 768])
    permute_64: "f32[768, 3072]" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
    addmm_34: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg100_1, view_128, permute_64);  arg100_1 = view_128 = permute_64 = None
    view_129: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_34, [1, 512, 3072]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_40: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, 0.5)
    mul_41: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476);  view_129 = None
    erf_5: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_41);  mul_41 = None
    add_54: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_42: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_40, add_54);  mul_40 = add_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_130: "f32[512, 3072]" = torch.ops.aten.view.default(mul_42, [512, 3072]);  mul_42 = None
    permute_65: "f32[3072, 768]" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
    addmm_35: "f32[512, 768]" = torch.ops.aten.addmm.default(arg102_1, view_130, permute_65);  arg102_1 = view_130 = permute_65 = None
    view_131: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_35, [1, 512, 768]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    clone_24: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_131);  view_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_55: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_24, add_53);  clone_24 = add_53 = None
    var_mean_12 = torch.ops.aten.var_mean.correction(add_55, [2], correction = 0, keepdim = True)
    getitem_24: "f32[1, 512, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 512, 1]" = var_mean_12[1];  var_mean_12 = None
    add_56: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
    rsqrt_12: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    sub_21: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_55, getitem_25);  add_55 = getitem_25 = None
    mul_43: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_12);  sub_21 = rsqrt_12 = None
    mul_44: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_43, arg103_1);  mul_43 = arg103_1 = None
    add_57: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_44, arg104_1);  mul_44 = arg104_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_132: "f32[512, 768]" = torch.ops.aten.view.default(add_57, [512, 768])
    permute_66: "f32[768, 768]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
    addmm_36: "f32[512, 768]" = torch.ops.aten.addmm.default(arg106_1, view_132, permute_66);  arg106_1 = view_132 = permute_66 = None
    view_133: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_36, [1, 512, 768]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_134: "f32[512, 768]" = torch.ops.aten.view.default(add_57, [512, 768])
    permute_67: "f32[768, 768]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
    addmm_37: "f32[512, 768]" = torch.ops.aten.addmm.default(arg108_1, view_134, permute_67);  arg108_1 = view_134 = permute_67 = None
    view_135: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_37, [1, 512, 768]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_136: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_135, [1, 512, 12, 64]);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_68: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_137: "f32[512, 768]" = torch.ops.aten.view.default(add_57, [512, 768])
    permute_69: "f32[768, 768]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
    addmm_38: "f32[512, 768]" = torch.ops.aten.addmm.default(arg110_1, view_137, permute_69);  arg110_1 = view_137 = permute_69 = None
    view_138: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_38, [1, 512, 768]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_139: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_138, [1, 512, 12, 64]);  view_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_70: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_139, [0, 2, 1, 3]);  view_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_140: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_133, [1, 512, 12, 64]);  view_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_71: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_72: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_68, [0, 1, 3, 2]);  permute_68 = None
    expand_24: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_71, [1, 12, 512, 64]);  permute_71 = None
    view_141: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_24, [12, 512, 64]);  expand_24 = None
    expand_25: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_72, [1, 12, 64, 512]);  permute_72 = None
    view_142: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_25, [12, 64, 512]);  expand_25 = None
    bmm_12: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_141, view_142);  view_141 = view_142 = None
    view_143: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_12, [1, 12, 512, 512]);  bmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:237, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_12: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_143, 8.0);  view_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:240, code: attention_scores = attention_scores + attention_mask
    add_58: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(div_12, mul);  div_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:243, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_6: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_58, [-1], True)
    sub_22: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_58, amax_6);  add_58 = amax_6 = None
    exp_6: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
    sum_7: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_13: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:247, code: attention_probs = self.dropout(attention_probs)
    clone_25: "f32[1, 12, 512, 512]" = torch.ops.aten.clone.default(div_13);  div_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:253, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_26: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(clone_25, [1, 12, 512, 512]);  clone_25 = None
    view_144: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_26, [12, 512, 512]);  expand_26 = None
    expand_27: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_70, [1, 12, 512, 64]);  permute_70 = None
    view_145: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_27, [12, 512, 64]);  expand_27 = None
    bmm_13: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_144, view_145);  view_144 = view_145 = None
    view_146: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_13, [1, 12, 512, 64]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_73: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
    clone_26: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_147: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_26, [1, 512, 768]);  clone_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_148: "f32[512, 768]" = torch.ops.aten.view.default(view_147, [512, 768]);  view_147 = None
    permute_74: "f32[768, 768]" = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
    addmm_39: "f32[512, 768]" = torch.ops.aten.addmm.default(arg112_1, view_148, permute_74);  arg112_1 = view_148 = permute_74 = None
    view_149: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_39, [1, 512, 768]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    clone_27: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_149);  view_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_59: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_27, add_57);  clone_27 = add_57 = None
    var_mean_13 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
    getitem_26: "f32[1, 512, 1]" = var_mean_13[0]
    getitem_27: "f32[1, 512, 1]" = var_mean_13[1];  var_mean_13 = None
    add_60: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
    rsqrt_13: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    sub_23: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_59, getitem_27);  add_59 = getitem_27 = None
    mul_45: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_13);  sub_23 = rsqrt_13 = None
    mul_46: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_45, arg113_1);  mul_45 = arg113_1 = None
    add_61: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_46, arg114_1);  mul_46 = arg114_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_150: "f32[512, 768]" = torch.ops.aten.view.default(add_61, [512, 768])
    permute_75: "f32[768, 3072]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
    addmm_40: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg116_1, view_150, permute_75);  arg116_1 = view_150 = permute_75 = None
    view_151: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_40, [1, 512, 3072]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_47: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, 0.5)
    mul_48: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476);  view_151 = None
    erf_6: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_48);  mul_48 = None
    add_62: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_49: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_47, add_62);  mul_47 = add_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_152: "f32[512, 3072]" = torch.ops.aten.view.default(mul_49, [512, 3072]);  mul_49 = None
    permute_76: "f32[3072, 768]" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
    addmm_41: "f32[512, 768]" = torch.ops.aten.addmm.default(arg118_1, view_152, permute_76);  arg118_1 = view_152 = permute_76 = None
    view_153: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_41, [1, 512, 768]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    clone_28: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_153);  view_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_63: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_28, add_61);  clone_28 = add_61 = None
    var_mean_14 = torch.ops.aten.var_mean.correction(add_63, [2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 512, 1]" = var_mean_14[0]
    getitem_29: "f32[1, 512, 1]" = var_mean_14[1];  var_mean_14 = None
    add_64: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
    rsqrt_14: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
    sub_24: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_63, getitem_29);  add_63 = getitem_29 = None
    mul_50: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_14);  sub_24 = rsqrt_14 = None
    mul_51: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_50, arg119_1);  mul_50 = arg119_1 = None
    add_65: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_51, arg120_1);  mul_51 = arg120_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_154: "f32[512, 768]" = torch.ops.aten.view.default(add_65, [512, 768])
    permute_77: "f32[768, 768]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
    addmm_42: "f32[512, 768]" = torch.ops.aten.addmm.default(arg122_1, view_154, permute_77);  arg122_1 = view_154 = permute_77 = None
    view_155: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_42, [1, 512, 768]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_156: "f32[512, 768]" = torch.ops.aten.view.default(add_65, [512, 768])
    permute_78: "f32[768, 768]" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
    addmm_43: "f32[512, 768]" = torch.ops.aten.addmm.default(arg124_1, view_156, permute_78);  arg124_1 = view_156 = permute_78 = None
    view_157: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_43, [1, 512, 768]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_158: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_157, [1, 512, 12, 64]);  view_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_79: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_158, [0, 2, 1, 3]);  view_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_159: "f32[512, 768]" = torch.ops.aten.view.default(add_65, [512, 768])
    permute_80: "f32[768, 768]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
    addmm_44: "f32[512, 768]" = torch.ops.aten.addmm.default(arg126_1, view_159, permute_80);  arg126_1 = view_159 = permute_80 = None
    view_160: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_44, [1, 512, 768]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_161: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_160, [1, 512, 12, 64]);  view_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_81: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_161, [0, 2, 1, 3]);  view_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_162: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_155, [1, 512, 12, 64]);  view_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_82: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_83: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_79, [0, 1, 3, 2]);  permute_79 = None
    expand_28: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_82, [1, 12, 512, 64]);  permute_82 = None
    view_163: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_28, [12, 512, 64]);  expand_28 = None
    expand_29: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_83, [1, 12, 64, 512]);  permute_83 = None
    view_164: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_29, [12, 64, 512]);  expand_29 = None
    bmm_14: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_163, view_164);  view_163 = view_164 = None
    view_165: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_14, [1, 12, 512, 512]);  bmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:237, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_14: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_165, 8.0);  view_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:240, code: attention_scores = attention_scores + attention_mask
    add_66: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(div_14, mul);  div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:243, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_7: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_66, [-1], True)
    sub_25: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_66, amax_7);  add_66 = amax_7 = None
    exp_7: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
    sum_8: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_15: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:247, code: attention_probs = self.dropout(attention_probs)
    clone_29: "f32[1, 12, 512, 512]" = torch.ops.aten.clone.default(div_15);  div_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:253, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_30: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(clone_29, [1, 12, 512, 512]);  clone_29 = None
    view_166: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_30, [12, 512, 512]);  expand_30 = None
    expand_31: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_81, [1, 12, 512, 64]);  permute_81 = None
    view_167: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_31, [12, 512, 64]);  expand_31 = None
    bmm_15: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_166, view_167);  view_166 = view_167 = None
    view_168: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_15, [1, 12, 512, 64]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_84: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
    clone_30: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_169: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_30, [1, 512, 768]);  clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_170: "f32[512, 768]" = torch.ops.aten.view.default(view_169, [512, 768]);  view_169 = None
    permute_85: "f32[768, 768]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
    addmm_45: "f32[512, 768]" = torch.ops.aten.addmm.default(arg128_1, view_170, permute_85);  arg128_1 = view_170 = permute_85 = None
    view_171: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_45, [1, 512, 768]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    clone_31: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_171);  view_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_67: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_31, add_65);  clone_31 = add_65 = None
    var_mean_15 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
    getitem_30: "f32[1, 512, 1]" = var_mean_15[0]
    getitem_31: "f32[1, 512, 1]" = var_mean_15[1];  var_mean_15 = None
    add_68: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
    rsqrt_15: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
    sub_26: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_67, getitem_31);  add_67 = getitem_31 = None
    mul_52: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_15);  sub_26 = rsqrt_15 = None
    mul_53: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_52, arg129_1);  mul_52 = arg129_1 = None
    add_69: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_53, arg130_1);  mul_53 = arg130_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_172: "f32[512, 768]" = torch.ops.aten.view.default(add_69, [512, 768])
    permute_86: "f32[768, 3072]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
    addmm_46: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg132_1, view_172, permute_86);  arg132_1 = view_172 = permute_86 = None
    view_173: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_46, [1, 512, 3072]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_54: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.5)
    mul_55: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476);  view_173 = None
    erf_7: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
    add_70: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_56: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_54, add_70);  mul_54 = add_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_174: "f32[512, 3072]" = torch.ops.aten.view.default(mul_56, [512, 3072]);  mul_56 = None
    permute_87: "f32[3072, 768]" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
    addmm_47: "f32[512, 768]" = torch.ops.aten.addmm.default(arg134_1, view_174, permute_87);  arg134_1 = view_174 = permute_87 = None
    view_175: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_47, [1, 512, 768]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    clone_32: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_175);  view_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_71: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_32, add_69);  clone_32 = add_69 = None
    var_mean_16 = torch.ops.aten.var_mean.correction(add_71, [2], correction = 0, keepdim = True)
    getitem_32: "f32[1, 512, 1]" = var_mean_16[0]
    getitem_33: "f32[1, 512, 1]" = var_mean_16[1];  var_mean_16 = None
    add_72: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
    rsqrt_16: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
    sub_27: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_71, getitem_33);  add_71 = getitem_33 = None
    mul_57: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_16);  sub_27 = rsqrt_16 = None
    mul_58: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_57, arg135_1);  mul_57 = arg135_1 = None
    add_73: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_58, arg136_1);  mul_58 = arg136_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_176: "f32[512, 768]" = torch.ops.aten.view.default(add_73, [512, 768])
    permute_88: "f32[768, 768]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
    addmm_48: "f32[512, 768]" = torch.ops.aten.addmm.default(arg138_1, view_176, permute_88);  arg138_1 = view_176 = permute_88 = None
    view_177: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_48, [1, 512, 768]);  addmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_178: "f32[512, 768]" = torch.ops.aten.view.default(add_73, [512, 768])
    permute_89: "f32[768, 768]" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
    addmm_49: "f32[512, 768]" = torch.ops.aten.addmm.default(arg140_1, view_178, permute_89);  arg140_1 = view_178 = permute_89 = None
    view_179: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_49, [1, 512, 768]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_180: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_179, [1, 512, 12, 64]);  view_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_90: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_180, [0, 2, 1, 3]);  view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_181: "f32[512, 768]" = torch.ops.aten.view.default(add_73, [512, 768])
    permute_91: "f32[768, 768]" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
    addmm_50: "f32[512, 768]" = torch.ops.aten.addmm.default(arg142_1, view_181, permute_91);  arg142_1 = view_181 = permute_91 = None
    view_182: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_50, [1, 512, 768]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_183: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_182, [1, 512, 12, 64]);  view_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_92: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_184: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_177, [1, 512, 12, 64]);  view_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_93: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_94: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_90, [0, 1, 3, 2]);  permute_90 = None
    expand_32: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_93, [1, 12, 512, 64]);  permute_93 = None
    view_185: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_32, [12, 512, 64]);  expand_32 = None
    expand_33: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_94, [1, 12, 64, 512]);  permute_94 = None
    view_186: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_33, [12, 64, 512]);  expand_33 = None
    bmm_16: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_185, view_186);  view_185 = view_186 = None
    view_187: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_16, [1, 12, 512, 512]);  bmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:237, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_16: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_187, 8.0);  view_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:240, code: attention_scores = attention_scores + attention_mask
    add_74: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(div_16, mul);  div_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:243, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_8: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_74, [-1], True)
    sub_28: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_74, amax_8);  add_74 = amax_8 = None
    exp_8: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
    sum_9: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_17: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:247, code: attention_probs = self.dropout(attention_probs)
    clone_33: "f32[1, 12, 512, 512]" = torch.ops.aten.clone.default(div_17);  div_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:253, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_34: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(clone_33, [1, 12, 512, 512]);  clone_33 = None
    view_188: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_34, [12, 512, 512]);  expand_34 = None
    expand_35: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_92, [1, 12, 512, 64]);  permute_92 = None
    view_189: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_35, [12, 512, 64]);  expand_35 = None
    bmm_17: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_188, view_189);  view_188 = view_189 = None
    view_190: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_17, [1, 12, 512, 64]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_95: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_190, [0, 2, 1, 3]);  view_190 = None
    clone_34: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_191: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_34, [1, 512, 768]);  clone_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_192: "f32[512, 768]" = torch.ops.aten.view.default(view_191, [512, 768]);  view_191 = None
    permute_96: "f32[768, 768]" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
    addmm_51: "f32[512, 768]" = torch.ops.aten.addmm.default(arg144_1, view_192, permute_96);  arg144_1 = view_192 = permute_96 = None
    view_193: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_51, [1, 512, 768]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    clone_35: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_193);  view_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_75: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_35, add_73);  clone_35 = add_73 = None
    var_mean_17 = torch.ops.aten.var_mean.correction(add_75, [2], correction = 0, keepdim = True)
    getitem_34: "f32[1, 512, 1]" = var_mean_17[0]
    getitem_35: "f32[1, 512, 1]" = var_mean_17[1];  var_mean_17 = None
    add_76: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
    rsqrt_17: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
    sub_29: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_75, getitem_35);  add_75 = getitem_35 = None
    mul_59: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_17);  sub_29 = rsqrt_17 = None
    mul_60: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_59, arg145_1);  mul_59 = arg145_1 = None
    add_77: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_60, arg146_1);  mul_60 = arg146_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_194: "f32[512, 768]" = torch.ops.aten.view.default(add_77, [512, 768])
    permute_97: "f32[768, 3072]" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
    addmm_52: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg148_1, view_194, permute_97);  arg148_1 = view_194 = permute_97 = None
    view_195: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_52, [1, 512, 3072]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_61: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, 0.5)
    mul_62: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476);  view_195 = None
    erf_8: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_62);  mul_62 = None
    add_78: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_63: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_61, add_78);  mul_61 = add_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_196: "f32[512, 3072]" = torch.ops.aten.view.default(mul_63, [512, 3072]);  mul_63 = None
    permute_98: "f32[3072, 768]" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
    addmm_53: "f32[512, 768]" = torch.ops.aten.addmm.default(arg150_1, view_196, permute_98);  arg150_1 = view_196 = permute_98 = None
    view_197: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_53, [1, 512, 768]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    clone_36: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_197);  view_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_79: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_36, add_77);  clone_36 = add_77 = None
    var_mean_18 = torch.ops.aten.var_mean.correction(add_79, [2], correction = 0, keepdim = True)
    getitem_36: "f32[1, 512, 1]" = var_mean_18[0]
    getitem_37: "f32[1, 512, 1]" = var_mean_18[1];  var_mean_18 = None
    add_80: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
    rsqrt_18: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
    sub_30: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_79, getitem_37);  add_79 = getitem_37 = None
    mul_64: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_18);  sub_30 = rsqrt_18 = None
    mul_65: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_64, arg151_1);  mul_64 = arg151_1 = None
    add_81: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_65, arg152_1);  mul_65 = arg152_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_198: "f32[512, 768]" = torch.ops.aten.view.default(add_81, [512, 768])
    permute_99: "f32[768, 768]" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
    addmm_54: "f32[512, 768]" = torch.ops.aten.addmm.default(arg154_1, view_198, permute_99);  arg154_1 = view_198 = permute_99 = None
    view_199: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_54, [1, 512, 768]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_200: "f32[512, 768]" = torch.ops.aten.view.default(add_81, [512, 768])
    permute_100: "f32[768, 768]" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
    addmm_55: "f32[512, 768]" = torch.ops.aten.addmm.default(arg156_1, view_200, permute_100);  arg156_1 = view_200 = permute_100 = None
    view_201: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_55, [1, 512, 768]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_202: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_201, [1, 512, 12, 64]);  view_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_101: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_202, [0, 2, 1, 3]);  view_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_203: "f32[512, 768]" = torch.ops.aten.view.default(add_81, [512, 768])
    permute_102: "f32[768, 768]" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
    addmm_56: "f32[512, 768]" = torch.ops.aten.addmm.default(arg158_1, view_203, permute_102);  arg158_1 = view_203 = permute_102 = None
    view_204: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_56, [1, 512, 768]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_205: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_204, [1, 512, 12, 64]);  view_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_103: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_205, [0, 2, 1, 3]);  view_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_206: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_199, [1, 512, 12, 64]);  view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_104: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_105: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_101, [0, 1, 3, 2]);  permute_101 = None
    expand_36: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_104, [1, 12, 512, 64]);  permute_104 = None
    view_207: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_36, [12, 512, 64]);  expand_36 = None
    expand_37: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_105, [1, 12, 64, 512]);  permute_105 = None
    view_208: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_37, [12, 64, 512]);  expand_37 = None
    bmm_18: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_207, view_208);  view_207 = view_208 = None
    view_209: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_18, [1, 12, 512, 512]);  bmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:237, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_18: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_209, 8.0);  view_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:240, code: attention_scores = attention_scores + attention_mask
    add_82: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(div_18, mul);  div_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:243, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_9: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_82, [-1], True)
    sub_31: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_82, amax_9);  add_82 = amax_9 = None
    exp_9: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
    sum_10: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_19: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:247, code: attention_probs = self.dropout(attention_probs)
    clone_37: "f32[1, 12, 512, 512]" = torch.ops.aten.clone.default(div_19);  div_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:253, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_38: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(clone_37, [1, 12, 512, 512]);  clone_37 = None
    view_210: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_38, [12, 512, 512]);  expand_38 = None
    expand_39: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_103, [1, 12, 512, 64]);  permute_103 = None
    view_211: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_39, [12, 512, 64]);  expand_39 = None
    bmm_19: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_210, view_211);  view_210 = view_211 = None
    view_212: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_19, [1, 12, 512, 64]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_106: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None
    clone_38: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_213: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_38, [1, 512, 768]);  clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_214: "f32[512, 768]" = torch.ops.aten.view.default(view_213, [512, 768]);  view_213 = None
    permute_107: "f32[768, 768]" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
    addmm_57: "f32[512, 768]" = torch.ops.aten.addmm.default(arg160_1, view_214, permute_107);  arg160_1 = view_214 = permute_107 = None
    view_215: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_57, [1, 512, 768]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    clone_39: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_215);  view_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_83: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_39, add_81);  clone_39 = add_81 = None
    var_mean_19 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
    getitem_38: "f32[1, 512, 1]" = var_mean_19[0]
    getitem_39: "f32[1, 512, 1]" = var_mean_19[1];  var_mean_19 = None
    add_84: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
    rsqrt_19: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
    sub_32: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_83, getitem_39);  add_83 = getitem_39 = None
    mul_66: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_19);  sub_32 = rsqrt_19 = None
    mul_67: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_66, arg161_1);  mul_66 = arg161_1 = None
    add_85: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_67, arg162_1);  mul_67 = arg162_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_216: "f32[512, 768]" = torch.ops.aten.view.default(add_85, [512, 768])
    permute_108: "f32[768, 3072]" = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
    addmm_58: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg164_1, view_216, permute_108);  arg164_1 = view_216 = permute_108 = None
    view_217: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_58, [1, 512, 3072]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_68: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, 0.5)
    mul_69: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476);  view_217 = None
    erf_9: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
    add_86: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_70: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_68, add_86);  mul_68 = add_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_218: "f32[512, 3072]" = torch.ops.aten.view.default(mul_70, [512, 3072]);  mul_70 = None
    permute_109: "f32[3072, 768]" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
    addmm_59: "f32[512, 768]" = torch.ops.aten.addmm.default(arg166_1, view_218, permute_109);  arg166_1 = view_218 = permute_109 = None
    view_219: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_59, [1, 512, 768]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    clone_40: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_219);  view_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_87: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_40, add_85);  clone_40 = add_85 = None
    var_mean_20 = torch.ops.aten.var_mean.correction(add_87, [2], correction = 0, keepdim = True)
    getitem_40: "f32[1, 512, 1]" = var_mean_20[0]
    getitem_41: "f32[1, 512, 1]" = var_mean_20[1];  var_mean_20 = None
    add_88: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
    rsqrt_20: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
    sub_33: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_87, getitem_41);  add_87 = getitem_41 = None
    mul_71: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_20);  sub_33 = rsqrt_20 = None
    mul_72: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_71, arg167_1);  mul_71 = arg167_1 = None
    add_89: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_72, arg168_1);  mul_72 = arg168_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_220: "f32[512, 768]" = torch.ops.aten.view.default(add_89, [512, 768])
    permute_110: "f32[768, 768]" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
    addmm_60: "f32[512, 768]" = torch.ops.aten.addmm.default(arg170_1, view_220, permute_110);  arg170_1 = view_220 = permute_110 = None
    view_221: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_60, [1, 512, 768]);  addmm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_222: "f32[512, 768]" = torch.ops.aten.view.default(add_89, [512, 768])
    permute_111: "f32[768, 768]" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
    addmm_61: "f32[512, 768]" = torch.ops.aten.addmm.default(arg172_1, view_222, permute_111);  arg172_1 = view_222 = permute_111 = None
    view_223: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_61, [1, 512, 768]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_224: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_223, [1, 512, 12, 64]);  view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_112: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_225: "f32[512, 768]" = torch.ops.aten.view.default(add_89, [512, 768])
    permute_113: "f32[768, 768]" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
    addmm_62: "f32[512, 768]" = torch.ops.aten.addmm.default(arg174_1, view_225, permute_113);  arg174_1 = view_225 = permute_113 = None
    view_226: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_62, [1, 512, 768]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_227: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_226, [1, 512, 12, 64]);  view_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_114: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_227, [0, 2, 1, 3]);  view_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_228: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_221, [1, 512, 12, 64]);  view_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_115: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_116: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_112, [0, 1, 3, 2]);  permute_112 = None
    expand_40: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_115, [1, 12, 512, 64]);  permute_115 = None
    view_229: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_40, [12, 512, 64]);  expand_40 = None
    expand_41: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_116, [1, 12, 64, 512]);  permute_116 = None
    view_230: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_41, [12, 64, 512]);  expand_41 = None
    bmm_20: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_229, view_230);  view_229 = view_230 = None
    view_231: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_20, [1, 12, 512, 512]);  bmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:237, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_20: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_231, 8.0);  view_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:240, code: attention_scores = attention_scores + attention_mask
    add_90: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(div_20, mul);  div_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:243, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_10: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_90, [-1], True)
    sub_34: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_90, amax_10);  add_90 = amax_10 = None
    exp_10: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
    sum_11: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_21: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:247, code: attention_probs = self.dropout(attention_probs)
    clone_41: "f32[1, 12, 512, 512]" = torch.ops.aten.clone.default(div_21);  div_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:253, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_42: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(clone_41, [1, 12, 512, 512]);  clone_41 = None
    view_232: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_42, [12, 512, 512]);  expand_42 = None
    expand_43: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_114, [1, 12, 512, 64]);  permute_114 = None
    view_233: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_43, [12, 512, 64]);  expand_43 = None
    bmm_21: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_232, view_233);  view_232 = view_233 = None
    view_234: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_21, [1, 12, 512, 64]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_117: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_234, [0, 2, 1, 3]);  view_234 = None
    clone_42: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_235: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_42, [1, 512, 768]);  clone_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_236: "f32[512, 768]" = torch.ops.aten.view.default(view_235, [512, 768]);  view_235 = None
    permute_118: "f32[768, 768]" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
    addmm_63: "f32[512, 768]" = torch.ops.aten.addmm.default(arg176_1, view_236, permute_118);  arg176_1 = view_236 = permute_118 = None
    view_237: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_63, [1, 512, 768]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    clone_43: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_237);  view_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_91: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_43, add_89);  clone_43 = add_89 = None
    var_mean_21 = torch.ops.aten.var_mean.correction(add_91, [2], correction = 0, keepdim = True)
    getitem_42: "f32[1, 512, 1]" = var_mean_21[0]
    getitem_43: "f32[1, 512, 1]" = var_mean_21[1];  var_mean_21 = None
    add_92: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
    rsqrt_21: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
    sub_35: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_91, getitem_43);  add_91 = getitem_43 = None
    mul_73: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_21);  sub_35 = rsqrt_21 = None
    mul_74: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_73, arg177_1);  mul_73 = arg177_1 = None
    add_93: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_74, arg178_1);  mul_74 = arg178_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_238: "f32[512, 768]" = torch.ops.aten.view.default(add_93, [512, 768])
    permute_119: "f32[768, 3072]" = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
    addmm_64: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg180_1, view_238, permute_119);  arg180_1 = view_238 = permute_119 = None
    view_239: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_64, [1, 512, 3072]);  addmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_75: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, 0.5)
    mul_76: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, 0.7071067811865476);  view_239 = None
    erf_10: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_76);  mul_76 = None
    add_94: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_77: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_75, add_94);  mul_75 = add_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_240: "f32[512, 3072]" = torch.ops.aten.view.default(mul_77, [512, 3072]);  mul_77 = None
    permute_120: "f32[3072, 768]" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
    addmm_65: "f32[512, 768]" = torch.ops.aten.addmm.default(arg182_1, view_240, permute_120);  arg182_1 = view_240 = permute_120 = None
    view_241: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_65, [1, 512, 768]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    clone_44: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_241);  view_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_95: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_44, add_93);  clone_44 = add_93 = None
    var_mean_22 = torch.ops.aten.var_mean.correction(add_95, [2], correction = 0, keepdim = True)
    getitem_44: "f32[1, 512, 1]" = var_mean_22[0]
    getitem_45: "f32[1, 512, 1]" = var_mean_22[1];  var_mean_22 = None
    add_96: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
    rsqrt_22: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
    sub_36: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_95, getitem_45);  add_95 = getitem_45 = None
    mul_78: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_22);  sub_36 = rsqrt_22 = None
    mul_79: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_78, arg183_1);  mul_78 = arg183_1 = None
    add_97: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_79, arg184_1);  mul_79 = arg184_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_242: "f32[512, 768]" = torch.ops.aten.view.default(add_97, [512, 768])
    permute_121: "f32[768, 768]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
    addmm_66: "f32[512, 768]" = torch.ops.aten.addmm.default(arg186_1, view_242, permute_121);  arg186_1 = view_242 = permute_121 = None
    view_243: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_66, [1, 512, 768]);  addmm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_244: "f32[512, 768]" = torch.ops.aten.view.default(add_97, [512, 768])
    permute_122: "f32[768, 768]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
    addmm_67: "f32[512, 768]" = torch.ops.aten.addmm.default(arg188_1, view_244, permute_122);  arg188_1 = view_244 = permute_122 = None
    view_245: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_67, [1, 512, 768]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_246: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_245, [1, 512, 12, 64]);  view_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_123: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_247: "f32[512, 768]" = torch.ops.aten.view.default(add_97, [512, 768])
    permute_124: "f32[768, 768]" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
    addmm_68: "f32[512, 768]" = torch.ops.aten.addmm.default(arg190_1, view_247, permute_124);  arg190_1 = view_247 = permute_124 = None
    view_248: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_68, [1, 512, 768]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_249: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_248, [1, 512, 12, 64]);  view_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_125: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_249, [0, 2, 1, 3]);  view_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_250: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_243, [1, 512, 12, 64]);  view_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_126: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_127: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_123, [0, 1, 3, 2]);  permute_123 = None
    expand_44: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_126, [1, 12, 512, 64]);  permute_126 = None
    view_251: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_44, [12, 512, 64]);  expand_44 = None
    expand_45: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_127, [1, 12, 64, 512]);  permute_127 = None
    view_252: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_45, [12, 64, 512]);  expand_45 = None
    bmm_22: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_251, view_252);  view_251 = view_252 = None
    view_253: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_22, [1, 12, 512, 512]);  bmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:237, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_22: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_253, 8.0);  view_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:240, code: attention_scores = attention_scores + attention_mask
    add_98: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(div_22, mul);  div_22 = mul = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:243, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_11: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_98, [-1], True)
    sub_37: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_98, amax_11);  add_98 = amax_11 = None
    exp_11: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
    sum_12: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_23: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:247, code: attention_probs = self.dropout(attention_probs)
    clone_45: "f32[1, 12, 512, 512]" = torch.ops.aten.clone.default(div_23);  div_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:253, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_46: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(clone_45, [1, 12, 512, 512]);  clone_45 = None
    view_254: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_46, [12, 512, 512]);  expand_46 = None
    expand_47: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_125, [1, 12, 512, 64]);  permute_125 = None
    view_255: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_47, [12, 512, 64]);  expand_47 = None
    bmm_23: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_254, view_255);  view_254 = view_255 = None
    view_256: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_23, [1, 12, 512, 64]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_128: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_256, [0, 2, 1, 3]);  view_256 = None
    clone_46: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_257: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_46, [1, 512, 768]);  clone_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_258: "f32[512, 768]" = torch.ops.aten.view.default(view_257, [512, 768]);  view_257 = None
    permute_129: "f32[768, 768]" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
    addmm_69: "f32[512, 768]" = torch.ops.aten.addmm.default(arg192_1, view_258, permute_129);  arg192_1 = view_258 = permute_129 = None
    view_259: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_69, [1, 512, 768]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    clone_47: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_259);  view_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_99: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_47, add_97);  clone_47 = add_97 = None
    var_mean_23 = torch.ops.aten.var_mean.correction(add_99, [2], correction = 0, keepdim = True)
    getitem_46: "f32[1, 512, 1]" = var_mean_23[0]
    getitem_47: "f32[1, 512, 1]" = var_mean_23[1];  var_mean_23 = None
    add_100: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
    rsqrt_23: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
    sub_38: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_99, getitem_47);  add_99 = getitem_47 = None
    mul_80: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_23);  sub_38 = rsqrt_23 = None
    mul_81: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_80, arg193_1);  mul_80 = arg193_1 = None
    add_101: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_81, arg194_1);  mul_81 = arg194_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_260: "f32[512, 768]" = torch.ops.aten.view.default(add_101, [512, 768])
    permute_130: "f32[768, 3072]" = torch.ops.aten.permute.default(arg195_1, [1, 0]);  arg195_1 = None
    addmm_70: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg196_1, view_260, permute_130);  arg196_1 = view_260 = permute_130 = None
    view_261: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_70, [1, 512, 3072]);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_82: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, 0.5)
    mul_83: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476);  view_261 = None
    erf_11: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
    add_102: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_84: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_82, add_102);  mul_82 = add_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_262: "f32[512, 3072]" = torch.ops.aten.view.default(mul_84, [512, 3072]);  mul_84 = None
    permute_131: "f32[3072, 768]" = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
    addmm_71: "f32[512, 768]" = torch.ops.aten.addmm.default(arg198_1, view_262, permute_131);  arg198_1 = view_262 = permute_131 = None
    view_263: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_71, [1, 512, 768]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    clone_48: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_263);  view_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_103: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_48, add_101);  clone_48 = add_101 = None
    var_mean_24 = torch.ops.aten.var_mean.correction(add_103, [2], correction = 0, keepdim = True)
    getitem_48: "f32[1, 512, 1]" = var_mean_24[0]
    getitem_49: "f32[1, 512, 1]" = var_mean_24[1];  var_mean_24 = None
    add_104: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
    rsqrt_24: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
    sub_39: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_103, getitem_49);  add_103 = getitem_49 = None
    mul_85: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_24);  sub_39 = rsqrt_24 = None
    mul_86: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_85, arg199_1);  mul_85 = arg199_1 = None
    add_105: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_86, arg200_1);  mul_86 = arg200_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:576, code: hidden_states = self.dense(hidden_states)
    view_264: "f32[512, 768]" = torch.ops.aten.view.default(add_105, [512, 768]);  add_105 = None
    permute_133: "f32[768, 768]" = torch.ops.aten.permute.default(arg203_1, [1, 0]);  arg203_1 = None
    addmm_73: "f32[512, 768]" = torch.ops.aten.addmm.default(arg204_1, view_264, permute_133);  arg204_1 = view_264 = permute_133 = None
    view_265: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_73, [1, 512, 768]);  addmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_87: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_265, 0.5)
    mul_88: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_265, 0.7071067811865476);  view_265 = None
    erf_12: "f32[1, 512, 768]" = torch.ops.aten.erf.default(mul_88);  mul_88 = None
    add_106: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_89: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_87, add_106);  mul_87 = add_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:578, code: hidden_states = self.LayerNorm(hidden_states)
    var_mean_25 = torch.ops.aten.var_mean.correction(mul_89, [2], correction = 0, keepdim = True)
    getitem_50: "f32[1, 512, 1]" = var_mean_25[0]
    getitem_51: "f32[1, 512, 1]" = var_mean_25[1];  var_mean_25 = None
    add_107: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
    rsqrt_25: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_107);  add_107 = None
    sub_40: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_89, getitem_51);  mul_89 = getitem_51 = None
    mul_90: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_25);  sub_40 = rsqrt_25 = None
    mul_91: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_90, arg205_1);  mul_90 = arg205_1 = None
    add_108: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_91, arg206_1);  mul_91 = arg206_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:599, code: hidden_states = self.decoder(hidden_states)
    view_266: "f32[512, 768]" = torch.ops.aten.view.default(add_108, [512, 768]);  add_108 = None
    permute_134: "f32[768, 30522]" = torch.ops.aten.permute.default(arg207_1, [1, 0]);  arg207_1 = None
    addmm_74: "f32[512, 30522]" = torch.ops.aten.addmm.default(arg208_1, view_266, permute_134);  arg208_1 = view_266 = permute_134 = None
    view_267: "f32[1, 512, 30522]" = torch.ops.aten.view.default(addmm_74, [1, 512, 30522]);  addmm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:969, code: prediction_scores.view(-1, self.config.vocab_size),
    view_268: "f32[512, 30522]" = torch.ops.aten.view.default(view_267, [-1, 30522])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:970, code: labels.view(-1),
    view_269: "i64[512]" = torch.ops.aten.view.default(arg211_1, [-1]);  arg211_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:968, code: masked_lm_loss = loss_fct(
    amax_12: "f32[512, 1]" = torch.ops.aten.amax.default(view_268, [1], True)
    sub_41: "f32[512, 30522]" = torch.ops.aten.sub.Tensor(view_268, amax_12);  view_268 = amax_12 = None
    exp_12: "f32[512, 30522]" = torch.ops.aten.exp.default(sub_41)
    sum_13: "f32[512, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [1], True);  exp_12 = None
    log: "f32[512, 1]" = torch.ops.aten.log.default(sum_13);  sum_13 = None
    sub_42: "f32[512, 30522]" = torch.ops.aten.sub.Tensor(sub_41, log);  sub_41 = log = None
    ne: "b8[512]" = torch.ops.aten.ne.Scalar(view_269, -100)
    scalar_tensor: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0))
    where: "i64[512]" = torch.ops.aten.where.self(ne, view_269, scalar_tensor);  ne = scalar_tensor = None
    unsqueeze_2: "i64[512, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
    gather: "f32[512, 1]" = torch.ops.aten.gather.default(sub_42, 1, unsqueeze_2);  sub_42 = unsqueeze_2 = None
    squeeze: "f32[512]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[512]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
    ne_1: "b8[512]" = torch.ops.aten.ne.Scalar(view_269, -100)
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_1: "f32[512]" = torch.ops.aten.where.self(ne_1, neg, scalar_tensor_1);  ne_1 = neg = scalar_tensor_1 = None
    ne_2: "b8[512]" = torch.ops.aten.ne.Scalar(view_269, -100);  view_269 = None
    sum_14: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None
    sum_15: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
    div_24: "f32[]" = torch.ops.aten.div.Tensor(sum_15, convert_element_type);  sum_15 = convert_element_type = None
    return (div_24, view_267)
    