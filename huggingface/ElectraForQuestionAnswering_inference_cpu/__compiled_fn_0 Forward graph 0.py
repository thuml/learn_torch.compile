from __future__ import annotations



def forward(self, arg0_1: "f32[30522, 128]", arg1_1: "f32[2, 128]", arg2_1: "f32[512, 128]", arg3_1: "f32[128]", arg4_1: "f32[128]", arg5_1: "f32[256, 128]", arg6_1: "f32[256]", arg7_1: "f32[256, 256]", arg8_1: "f32[256]", arg9_1: "f32[256, 256]", arg10_1: "f32[256]", arg11_1: "f32[256, 256]", arg12_1: "f32[256]", arg13_1: "f32[256, 256]", arg14_1: "f32[256]", arg15_1: "f32[256]", arg16_1: "f32[256]", arg17_1: "f32[1024, 256]", arg18_1: "f32[1024]", arg19_1: "f32[256, 1024]", arg20_1: "f32[256]", arg21_1: "f32[256]", arg22_1: "f32[256]", arg23_1: "f32[256, 256]", arg24_1: "f32[256]", arg25_1: "f32[256, 256]", arg26_1: "f32[256]", arg27_1: "f32[256, 256]", arg28_1: "f32[256]", arg29_1: "f32[256, 256]", arg30_1: "f32[256]", arg31_1: "f32[256]", arg32_1: "f32[256]", arg33_1: "f32[1024, 256]", arg34_1: "f32[1024]", arg35_1: "f32[256, 1024]", arg36_1: "f32[256]", arg37_1: "f32[256]", arg38_1: "f32[256]", arg39_1: "f32[256, 256]", arg40_1: "f32[256]", arg41_1: "f32[256, 256]", arg42_1: "f32[256]", arg43_1: "f32[256, 256]", arg44_1: "f32[256]", arg45_1: "f32[256, 256]", arg46_1: "f32[256]", arg47_1: "f32[256]", arg48_1: "f32[256]", arg49_1: "f32[1024, 256]", arg50_1: "f32[1024]", arg51_1: "f32[256, 1024]", arg52_1: "f32[256]", arg53_1: "f32[256]", arg54_1: "f32[256]", arg55_1: "f32[256, 256]", arg56_1: "f32[256]", arg57_1: "f32[256, 256]", arg58_1: "f32[256]", arg59_1: "f32[256, 256]", arg60_1: "f32[256]", arg61_1: "f32[256, 256]", arg62_1: "f32[256]", arg63_1: "f32[256]", arg64_1: "f32[256]", arg65_1: "f32[1024, 256]", arg66_1: "f32[1024]", arg67_1: "f32[256, 1024]", arg68_1: "f32[256]", arg69_1: "f32[256]", arg70_1: "f32[256]", arg71_1: "f32[256, 256]", arg72_1: "f32[256]", arg73_1: "f32[256, 256]", arg74_1: "f32[256]", arg75_1: "f32[256, 256]", arg76_1: "f32[256]", arg77_1: "f32[256, 256]", arg78_1: "f32[256]", arg79_1: "f32[256]", arg80_1: "f32[256]", arg81_1: "f32[1024, 256]", arg82_1: "f32[1024]", arg83_1: "f32[256, 1024]", arg84_1: "f32[256]", arg85_1: "f32[256]", arg86_1: "f32[256]", arg87_1: "f32[256, 256]", arg88_1: "f32[256]", arg89_1: "f32[256, 256]", arg90_1: "f32[256]", arg91_1: "f32[256, 256]", arg92_1: "f32[256]", arg93_1: "f32[256, 256]", arg94_1: "f32[256]", arg95_1: "f32[256]", arg96_1: "f32[256]", arg97_1: "f32[1024, 256]", arg98_1: "f32[1024]", arg99_1: "f32[256, 1024]", arg100_1: "f32[256]", arg101_1: "f32[256]", arg102_1: "f32[256]", arg103_1: "f32[256, 256]", arg104_1: "f32[256]", arg105_1: "f32[256, 256]", arg106_1: "f32[256]", arg107_1: "f32[256, 256]", arg108_1: "f32[256]", arg109_1: "f32[256, 256]", arg110_1: "f32[256]", arg111_1: "f32[256]", arg112_1: "f32[256]", arg113_1: "f32[1024, 256]", arg114_1: "f32[1024]", arg115_1: "f32[256, 1024]", arg116_1: "f32[256]", arg117_1: "f32[256]", arg118_1: "f32[256]", arg119_1: "f32[256, 256]", arg120_1: "f32[256]", arg121_1: "f32[256, 256]", arg122_1: "f32[256]", arg123_1: "f32[256, 256]", arg124_1: "f32[256]", arg125_1: "f32[256, 256]", arg126_1: "f32[256]", arg127_1: "f32[256]", arg128_1: "f32[256]", arg129_1: "f32[1024, 256]", arg130_1: "f32[1024]", arg131_1: "f32[256, 1024]", arg132_1: "f32[256]", arg133_1: "f32[256]", arg134_1: "f32[256]", arg135_1: "f32[256, 256]", arg136_1: "f32[256]", arg137_1: "f32[256, 256]", arg138_1: "f32[256]", arg139_1: "f32[256, 256]", arg140_1: "f32[256]", arg141_1: "f32[256, 256]", arg142_1: "f32[256]", arg143_1: "f32[256]", arg144_1: "f32[256]", arg145_1: "f32[1024, 256]", arg146_1: "f32[1024]", arg147_1: "f32[256, 1024]", arg148_1: "f32[256]", arg149_1: "f32[256]", arg150_1: "f32[256]", arg151_1: "f32[256, 256]", arg152_1: "f32[256]", arg153_1: "f32[256, 256]", arg154_1: "f32[256]", arg155_1: "f32[256, 256]", arg156_1: "f32[256]", arg157_1: "f32[256, 256]", arg158_1: "f32[256]", arg159_1: "f32[256]", arg160_1: "f32[256]", arg161_1: "f32[1024, 256]", arg162_1: "f32[1024]", arg163_1: "f32[256, 1024]", arg164_1: "f32[256]", arg165_1: "f32[256]", arg166_1: "f32[256]", arg167_1: "f32[256, 256]", arg168_1: "f32[256]", arg169_1: "f32[256, 256]", arg170_1: "f32[256]", arg171_1: "f32[256, 256]", arg172_1: "f32[256]", arg173_1: "f32[256, 256]", arg174_1: "f32[256]", arg175_1: "f32[256]", arg176_1: "f32[256]", arg177_1: "f32[1024, 256]", arg178_1: "f32[1024]", arg179_1: "f32[256, 1024]", arg180_1: "f32[256]", arg181_1: "f32[256]", arg182_1: "f32[256]", arg183_1: "f32[256, 256]", arg184_1: "f32[256]", arg185_1: "f32[256, 256]", arg186_1: "f32[256]", arg187_1: "f32[256, 256]", arg188_1: "f32[256]", arg189_1: "f32[256, 256]", arg190_1: "f32[256]", arg191_1: "f32[256]", arg192_1: "f32[256]", arg193_1: "f32[1024, 256]", arg194_1: "f32[1024]", arg195_1: "f32[256, 1024]", arg196_1: "f32[256]", arg197_1: "f32[256]", arg198_1: "f32[256]", arg199_1: "f32[2, 256]", arg200_1: "f32[2]", arg201_1: "i64[1, 512]", arg202_1: "i64[1, 512]", arg203_1: "i64[1, 512]", arg204_1: "i64[1]", arg205_1: "i64[1]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:885, code: attention_mask = torch.ones(input_shape, device=device)
    full: "f32[1, 512]" = torch.ops.aten.full.default([1, 512], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:888, code: buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
    slice_1: "i64[1, 512]" = torch.ops.aten.slice.Tensor(arg201_1, 0, 0, 9223372036854775807);  arg201_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:889, code: buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
    expand: "i64[1, 512]" = torch.ops.aten.expand.default(slice_1, [1, 512]);  slice_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:916, code: extended_attention_mask = attention_mask[:, None, None, :]
    slice_2: "f32[1, 512]" = torch.ops.aten.slice.Tensor(full, 0, 0, 9223372036854775807);  full = None
    unsqueeze: "f32[1, 1, 512]" = torch.ops.aten.unsqueeze.default(slice_2, 1);  slice_2 = None
    unsqueeze_1: "f32[1, 1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
    slice_3: "f32[1, 1, 1, 512]" = torch.ops.aten.slice.Tensor(unsqueeze_1, 3, 0, 9223372036854775807);  unsqueeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:928, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    sub: "f32[1, 1, 1, 512]" = torch.ops.aten.sub.Tensor(1.0, slice_3);  slice_3 = None
    mul: "f32[1, 1, 1, 512]" = torch.ops.aten.mul.Tensor(sub, -3.4028234663852886e+38);  sub = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:189, code: position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
    slice_4: "i64[1, 512]" = torch.ops.aten.slice.Tensor(arg202_1, 0, 0, 9223372036854775807);  arg202_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:203, code: inputs_embeds = self.word_embeddings(input_ids)
    embedding: "f32[1, 512, 128]" = torch.ops.aten.embedding.default(arg0_1, arg203_1, 0);  arg0_1 = arg203_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:204, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    embedding_1: "f32[1, 512, 128]" = torch.ops.aten.embedding.default(arg1_1, expand);  arg1_1 = expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:206, code: embeddings = inputs_embeds + token_type_embeddings
    add: "f32[1, 512, 128]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:208, code: position_embeddings = self.position_embeddings(position_ids)
    embedding_2: "f32[1, 512, 128]" = torch.ops.aten.embedding.default(arg2_1, slice_4);  arg2_1 = slice_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:209, code: embeddings += position_embeddings
    add_1: "f32[1, 512, 128]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:210, code: embeddings = self.LayerNorm(embeddings)
    var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 512, 1]" = var_mean[0]
    getitem_1: "f32[1, 512, 1]" = var_mean[1];  var_mean = None
    add_2: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
    rsqrt: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
    sub_1: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
    mul_1: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = rsqrt = None
    mul_2: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
    add_3: "f32[1, 512, 128]" = torch.ops.aten.add.Tensor(mul_2, arg4_1);  mul_2 = arg4_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:211, code: embeddings = self.dropout(embeddings)
    clone: "f32[1, 512, 128]" = torch.ops.aten.clone.default(add_3);  add_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:918, code: hidden_states = self.embeddings_project(hidden_states)
    view: "f32[512, 128]" = torch.ops.aten.view.default(clone, [512, 128]);  clone = None
    permute: "f32[128, 256]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
    addmm: "f32[512, 256]" = torch.ops.aten.addmm.default(arg6_1, view, permute);  arg6_1 = view = permute = None
    view_1: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm, [1, 512, 256]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_2: "f32[512, 256]" = torch.ops.aten.view.default(view_1, [512, 256])
    permute_1: "f32[256, 256]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
    addmm_1: "f32[512, 256]" = torch.ops.aten.addmm.default(arg8_1, view_2, permute_1);  arg8_1 = view_2 = permute_1 = None
    view_3: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_1, [1, 512, 256]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_4: "f32[512, 256]" = torch.ops.aten.view.default(view_1, [512, 256])
    permute_2: "f32[256, 256]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
    addmm_2: "f32[512, 256]" = torch.ops.aten.addmm.default(arg10_1, view_4, permute_2);  arg10_1 = view_4 = permute_2 = None
    view_5: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_2, [1, 512, 256]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_6: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_5, [1, 512, 4, 64]);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_3: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_7: "f32[512, 256]" = torch.ops.aten.view.default(view_1, [512, 256])
    permute_4: "f32[256, 256]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
    addmm_3: "f32[512, 256]" = torch.ops.aten.addmm.default(arg12_1, view_7, permute_4);  arg12_1 = view_7 = permute_4 = None
    view_8: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_3, [1, 512, 256]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_9: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_8, [1, 512, 4, 64]);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_5: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_10: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_3, [1, 512, 4, 64]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_6: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_7: "f32[1, 4, 64, 512]" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
    expand_1: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_6, [1, 4, 512, 64]);  permute_6 = None
    view_11: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_1, [4, 512, 64]);  expand_1 = None
    expand_2: "f32[1, 4, 64, 512]" = torch.ops.aten.expand.default(permute_7, [1, 4, 64, 512]);  permute_7 = None
    view_12: "f32[4, 64, 512]" = torch.ops.aten.view.default(expand_2, [4, 64, 512]);  expand_2 = None
    bmm: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_11, view_12);  view_11 = view_12 = None
    view_13: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm, [1, 4, 512, 512]);  bmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(view_13, 8.0);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:324, code: attention_scores = attention_scores + attention_mask
    add_4: "f32[1, 4, 512, 512]" = torch.ops.aten.add.Tensor(div, mul);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax: "f32[1, 4, 512, 1]" = torch.ops.aten.amax.default(add_4, [-1], True)
    sub_2: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_4, amax);  add_4 = amax = None
    exp: "f32[1, 4, 512, 512]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
    sum_1: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div_1: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    clone_1: "f32[1, 4, 512, 512]" = torch.ops.aten.clone.default(div_1);  div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_3: "f32[1, 4, 512, 512]" = torch.ops.aten.expand.default(clone_1, [1, 4, 512, 512]);  clone_1 = None
    view_14: "f32[4, 512, 512]" = torch.ops.aten.view.default(expand_3, [4, 512, 512]);  expand_3 = None
    expand_4: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_5, [1, 4, 512, 64]);  permute_5 = None
    view_15: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_4, [4, 512, 64]);  expand_4 = None
    bmm_1: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_14, view_15);  view_14 = view_15 = None
    view_16: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_1, [1, 4, 512, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_8: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
    clone_2: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_17: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_2, [1, 512, 256]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_18: "f32[512, 256]" = torch.ops.aten.view.default(view_17, [512, 256]);  view_17 = None
    permute_9: "f32[256, 256]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
    addmm_4: "f32[512, 256]" = torch.ops.aten.addmm.default(arg14_1, view_18, permute_9);  arg14_1 = view_18 = permute_9 = None
    view_19: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_4, [1, 512, 256]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    clone_3: "f32[1, 512, 256]" = torch.ops.aten.clone.default(view_19);  view_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_5: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(clone_3, view_1);  clone_3 = view_1 = None
    var_mean_1 = torch.ops.aten.var_mean.correction(add_5, [2], correction = 0, keepdim = True)
    getitem_2: "f32[1, 512, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 512, 1]" = var_mean_1[1];  var_mean_1 = None
    add_6: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
    rsqrt_1: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_3: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_5, getitem_3);  add_5 = getitem_3 = None
    mul_3: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = rsqrt_1 = None
    mul_4: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_3, arg15_1);  mul_3 = arg15_1 = None
    add_7: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_4, arg16_1);  mul_4 = arg16_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_20: "f32[512, 256]" = torch.ops.aten.view.default(add_7, [512, 256])
    permute_10: "f32[256, 1024]" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
    addmm_5: "f32[512, 1024]" = torch.ops.aten.addmm.default(arg18_1, view_20, permute_10);  arg18_1 = view_20 = permute_10 = None
    view_21: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_5, [1, 512, 1024]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_5: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_21, 0.5)
    mul_6: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_21, 0.7071067811865476);  view_21 = None
    erf: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
    add_8: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_7: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_5, add_8);  mul_5 = add_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_22: "f32[512, 1024]" = torch.ops.aten.view.default(mul_7, [512, 1024]);  mul_7 = None
    permute_11: "f32[1024, 256]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
    addmm_6: "f32[512, 256]" = torch.ops.aten.addmm.default(arg20_1, view_22, permute_11);  arg20_1 = view_22 = permute_11 = None
    view_23: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_6, [1, 512, 256]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    clone_4: "f32[1, 512, 256]" = torch.ops.aten.clone.default(view_23);  view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_9: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(clone_4, add_7);  clone_4 = add_7 = None
    var_mean_2 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
    getitem_4: "f32[1, 512, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 512, 1]" = var_mean_2[1];  var_mean_2 = None
    add_10: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
    rsqrt_2: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
    sub_4: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_9, getitem_5);  add_9 = getitem_5 = None
    mul_8: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = rsqrt_2 = None
    mul_9: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_8, arg21_1);  mul_8 = arg21_1 = None
    add_11: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_9, arg22_1);  mul_9 = arg22_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_24: "f32[512, 256]" = torch.ops.aten.view.default(add_11, [512, 256])
    permute_12: "f32[256, 256]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
    addmm_7: "f32[512, 256]" = torch.ops.aten.addmm.default(arg24_1, view_24, permute_12);  arg24_1 = view_24 = permute_12 = None
    view_25: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_7, [1, 512, 256]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_26: "f32[512, 256]" = torch.ops.aten.view.default(add_11, [512, 256])
    permute_13: "f32[256, 256]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
    addmm_8: "f32[512, 256]" = torch.ops.aten.addmm.default(arg26_1, view_26, permute_13);  arg26_1 = view_26 = permute_13 = None
    view_27: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_8, [1, 512, 256]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_28: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_27, [1, 512, 4, 64]);  view_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_14: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_29: "f32[512, 256]" = torch.ops.aten.view.default(add_11, [512, 256])
    permute_15: "f32[256, 256]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
    addmm_9: "f32[512, 256]" = torch.ops.aten.addmm.default(arg28_1, view_29, permute_15);  arg28_1 = view_29 = permute_15 = None
    view_30: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_9, [1, 512, 256]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_31: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_30, [1, 512, 4, 64]);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_16: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_32: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_25, [1, 512, 4, 64]);  view_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_17: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_18: "f32[1, 4, 64, 512]" = torch.ops.aten.permute.default(permute_14, [0, 1, 3, 2]);  permute_14 = None
    expand_5: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_17, [1, 4, 512, 64]);  permute_17 = None
    view_33: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_5, [4, 512, 64]);  expand_5 = None
    expand_6: "f32[1, 4, 64, 512]" = torch.ops.aten.expand.default(permute_18, [1, 4, 64, 512]);  permute_18 = None
    view_34: "f32[4, 64, 512]" = torch.ops.aten.view.default(expand_6, [4, 64, 512]);  expand_6 = None
    bmm_2: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_33, view_34);  view_33 = view_34 = None
    view_35: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_2, [1, 4, 512, 512]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_2: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(view_35, 8.0);  view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:324, code: attention_scores = attention_scores + attention_mask
    add_12: "f32[1, 4, 512, 512]" = torch.ops.aten.add.Tensor(div_2, mul);  div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_1: "f32[1, 4, 512, 1]" = torch.ops.aten.amax.default(add_12, [-1], True)
    sub_5: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_12, amax_1);  add_12 = amax_1 = None
    exp_1: "f32[1, 4, 512, 512]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
    sum_2: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_3: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    clone_5: "f32[1, 4, 512, 512]" = torch.ops.aten.clone.default(div_3);  div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_7: "f32[1, 4, 512, 512]" = torch.ops.aten.expand.default(clone_5, [1, 4, 512, 512]);  clone_5 = None
    view_36: "f32[4, 512, 512]" = torch.ops.aten.view.default(expand_7, [4, 512, 512]);  expand_7 = None
    expand_8: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_16, [1, 4, 512, 64]);  permute_16 = None
    view_37: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_8, [4, 512, 64]);  expand_8 = None
    bmm_3: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_36, view_37);  view_36 = view_37 = None
    view_38: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_3, [1, 4, 512, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_19: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
    clone_6: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_39: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_6, [1, 512, 256]);  clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_40: "f32[512, 256]" = torch.ops.aten.view.default(view_39, [512, 256]);  view_39 = None
    permute_20: "f32[256, 256]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
    addmm_10: "f32[512, 256]" = torch.ops.aten.addmm.default(arg30_1, view_40, permute_20);  arg30_1 = view_40 = permute_20 = None
    view_41: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_10, [1, 512, 256]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    clone_7: "f32[1, 512, 256]" = torch.ops.aten.clone.default(view_41);  view_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_13: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(clone_7, add_11);  clone_7 = add_11 = None
    var_mean_3 = torch.ops.aten.var_mean.correction(add_13, [2], correction = 0, keepdim = True)
    getitem_6: "f32[1, 512, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 512, 1]" = var_mean_3[1];  var_mean_3 = None
    add_14: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
    rsqrt_3: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
    sub_6: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_13, getitem_7);  add_13 = getitem_7 = None
    mul_10: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_3);  sub_6 = rsqrt_3 = None
    mul_11: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_10, arg31_1);  mul_10 = arg31_1 = None
    add_15: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_11, arg32_1);  mul_11 = arg32_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_42: "f32[512, 256]" = torch.ops.aten.view.default(add_15, [512, 256])
    permute_21: "f32[256, 1024]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
    addmm_11: "f32[512, 1024]" = torch.ops.aten.addmm.default(arg34_1, view_42, permute_21);  arg34_1 = view_42 = permute_21 = None
    view_43: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_11, [1, 512, 1024]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_12: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_43, 0.5)
    mul_13: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_43, 0.7071067811865476);  view_43 = None
    erf_1: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
    add_16: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_14: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_12, add_16);  mul_12 = add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_44: "f32[512, 1024]" = torch.ops.aten.view.default(mul_14, [512, 1024]);  mul_14 = None
    permute_22: "f32[1024, 256]" = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
    addmm_12: "f32[512, 256]" = torch.ops.aten.addmm.default(arg36_1, view_44, permute_22);  arg36_1 = view_44 = permute_22 = None
    view_45: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_12, [1, 512, 256]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    clone_8: "f32[1, 512, 256]" = torch.ops.aten.clone.default(view_45);  view_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_17: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(clone_8, add_15);  clone_8 = add_15 = None
    var_mean_4 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
    getitem_8: "f32[1, 512, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 512, 1]" = var_mean_4[1];  var_mean_4 = None
    add_18: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
    rsqrt_4: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    sub_7: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_17, getitem_9);  add_17 = getitem_9 = None
    mul_15: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_4);  sub_7 = rsqrt_4 = None
    mul_16: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_15, arg37_1);  mul_15 = arg37_1 = None
    add_19: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_16, arg38_1);  mul_16 = arg38_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_46: "f32[512, 256]" = torch.ops.aten.view.default(add_19, [512, 256])
    permute_23: "f32[256, 256]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
    addmm_13: "f32[512, 256]" = torch.ops.aten.addmm.default(arg40_1, view_46, permute_23);  arg40_1 = view_46 = permute_23 = None
    view_47: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_13, [1, 512, 256]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_48: "f32[512, 256]" = torch.ops.aten.view.default(add_19, [512, 256])
    permute_24: "f32[256, 256]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
    addmm_14: "f32[512, 256]" = torch.ops.aten.addmm.default(arg42_1, view_48, permute_24);  arg42_1 = view_48 = permute_24 = None
    view_49: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_14, [1, 512, 256]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_50: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_49, [1, 512, 4, 64]);  view_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_25: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_51: "f32[512, 256]" = torch.ops.aten.view.default(add_19, [512, 256])
    permute_26: "f32[256, 256]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
    addmm_15: "f32[512, 256]" = torch.ops.aten.addmm.default(arg44_1, view_51, permute_26);  arg44_1 = view_51 = permute_26 = None
    view_52: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_15, [1, 512, 256]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_53: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_52, [1, 512, 4, 64]);  view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_27: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_54: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_47, [1, 512, 4, 64]);  view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_28: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_29: "f32[1, 4, 64, 512]" = torch.ops.aten.permute.default(permute_25, [0, 1, 3, 2]);  permute_25 = None
    expand_9: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_28, [1, 4, 512, 64]);  permute_28 = None
    view_55: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_9, [4, 512, 64]);  expand_9 = None
    expand_10: "f32[1, 4, 64, 512]" = torch.ops.aten.expand.default(permute_29, [1, 4, 64, 512]);  permute_29 = None
    view_56: "f32[4, 64, 512]" = torch.ops.aten.view.default(expand_10, [4, 64, 512]);  expand_10 = None
    bmm_4: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_55, view_56);  view_55 = view_56 = None
    view_57: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_4, [1, 4, 512, 512]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_4: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(view_57, 8.0);  view_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:324, code: attention_scores = attention_scores + attention_mask
    add_20: "f32[1, 4, 512, 512]" = torch.ops.aten.add.Tensor(div_4, mul);  div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_2: "f32[1, 4, 512, 1]" = torch.ops.aten.amax.default(add_20, [-1], True)
    sub_8: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_20, amax_2);  add_20 = amax_2 = None
    exp_2: "f32[1, 4, 512, 512]" = torch.ops.aten.exp.default(sub_8);  sub_8 = None
    sum_3: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_5: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    clone_9: "f32[1, 4, 512, 512]" = torch.ops.aten.clone.default(div_5);  div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_11: "f32[1, 4, 512, 512]" = torch.ops.aten.expand.default(clone_9, [1, 4, 512, 512]);  clone_9 = None
    view_58: "f32[4, 512, 512]" = torch.ops.aten.view.default(expand_11, [4, 512, 512]);  expand_11 = None
    expand_12: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_27, [1, 4, 512, 64]);  permute_27 = None
    view_59: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_12, [4, 512, 64]);  expand_12 = None
    bmm_5: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_58, view_59);  view_58 = view_59 = None
    view_60: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_5, [1, 4, 512, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_30: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_60, [0, 2, 1, 3]);  view_60 = None
    clone_10: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_61: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_10, [1, 512, 256]);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_62: "f32[512, 256]" = torch.ops.aten.view.default(view_61, [512, 256]);  view_61 = None
    permute_31: "f32[256, 256]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
    addmm_16: "f32[512, 256]" = torch.ops.aten.addmm.default(arg46_1, view_62, permute_31);  arg46_1 = view_62 = permute_31 = None
    view_63: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_16, [1, 512, 256]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    clone_11: "f32[1, 512, 256]" = torch.ops.aten.clone.default(view_63);  view_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_21: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(clone_11, add_19);  clone_11 = add_19 = None
    var_mean_5 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
    getitem_10: "f32[1, 512, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 512, 1]" = var_mean_5[1];  var_mean_5 = None
    add_22: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
    rsqrt_5: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
    sub_9: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_21, getitem_11);  add_21 = getitem_11 = None
    mul_17: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_5);  sub_9 = rsqrt_5 = None
    mul_18: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_17, arg47_1);  mul_17 = arg47_1 = None
    add_23: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_18, arg48_1);  mul_18 = arg48_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_64: "f32[512, 256]" = torch.ops.aten.view.default(add_23, [512, 256])
    permute_32: "f32[256, 1024]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
    addmm_17: "f32[512, 1024]" = torch.ops.aten.addmm.default(arg50_1, view_64, permute_32);  arg50_1 = view_64 = permute_32 = None
    view_65: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_17, [1, 512, 1024]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_19: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_65, 0.5)
    mul_20: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_65, 0.7071067811865476);  view_65 = None
    erf_2: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_20);  mul_20 = None
    add_24: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_21: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_19, add_24);  mul_19 = add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_66: "f32[512, 1024]" = torch.ops.aten.view.default(mul_21, [512, 1024]);  mul_21 = None
    permute_33: "f32[1024, 256]" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
    addmm_18: "f32[512, 256]" = torch.ops.aten.addmm.default(arg52_1, view_66, permute_33);  arg52_1 = view_66 = permute_33 = None
    view_67: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_18, [1, 512, 256]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    clone_12: "f32[1, 512, 256]" = torch.ops.aten.clone.default(view_67);  view_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_25: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(clone_12, add_23);  clone_12 = add_23 = None
    var_mean_6 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
    getitem_12: "f32[1, 512, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 512, 1]" = var_mean_6[1];  var_mean_6 = None
    add_26: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
    rsqrt_6: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
    sub_10: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_25, getitem_13);  add_25 = getitem_13 = None
    mul_22: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_6);  sub_10 = rsqrt_6 = None
    mul_23: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_22, arg53_1);  mul_22 = arg53_1 = None
    add_27: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_23, arg54_1);  mul_23 = arg54_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_68: "f32[512, 256]" = torch.ops.aten.view.default(add_27, [512, 256])
    permute_34: "f32[256, 256]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
    addmm_19: "f32[512, 256]" = torch.ops.aten.addmm.default(arg56_1, view_68, permute_34);  arg56_1 = view_68 = permute_34 = None
    view_69: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_19, [1, 512, 256]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_70: "f32[512, 256]" = torch.ops.aten.view.default(add_27, [512, 256])
    permute_35: "f32[256, 256]" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
    addmm_20: "f32[512, 256]" = torch.ops.aten.addmm.default(arg58_1, view_70, permute_35);  arg58_1 = view_70 = permute_35 = None
    view_71: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_20, [1, 512, 256]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_72: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_71, [1, 512, 4, 64]);  view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_36: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_73: "f32[512, 256]" = torch.ops.aten.view.default(add_27, [512, 256])
    permute_37: "f32[256, 256]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
    addmm_21: "f32[512, 256]" = torch.ops.aten.addmm.default(arg60_1, view_73, permute_37);  arg60_1 = view_73 = permute_37 = None
    view_74: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_21, [1, 512, 256]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_75: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_74, [1, 512, 4, 64]);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_38: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_76: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_69, [1, 512, 4, 64]);  view_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_39: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_40: "f32[1, 4, 64, 512]" = torch.ops.aten.permute.default(permute_36, [0, 1, 3, 2]);  permute_36 = None
    expand_13: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_39, [1, 4, 512, 64]);  permute_39 = None
    view_77: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_13, [4, 512, 64]);  expand_13 = None
    expand_14: "f32[1, 4, 64, 512]" = torch.ops.aten.expand.default(permute_40, [1, 4, 64, 512]);  permute_40 = None
    view_78: "f32[4, 64, 512]" = torch.ops.aten.view.default(expand_14, [4, 64, 512]);  expand_14 = None
    bmm_6: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_77, view_78);  view_77 = view_78 = None
    view_79: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_6, [1, 4, 512, 512]);  bmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_6: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(view_79, 8.0);  view_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:324, code: attention_scores = attention_scores + attention_mask
    add_28: "f32[1, 4, 512, 512]" = torch.ops.aten.add.Tensor(div_6, mul);  div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_3: "f32[1, 4, 512, 1]" = torch.ops.aten.amax.default(add_28, [-1], True)
    sub_11: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_28, amax_3);  add_28 = amax_3 = None
    exp_3: "f32[1, 4, 512, 512]" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
    sum_4: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_7: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    clone_13: "f32[1, 4, 512, 512]" = torch.ops.aten.clone.default(div_7);  div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_15: "f32[1, 4, 512, 512]" = torch.ops.aten.expand.default(clone_13, [1, 4, 512, 512]);  clone_13 = None
    view_80: "f32[4, 512, 512]" = torch.ops.aten.view.default(expand_15, [4, 512, 512]);  expand_15 = None
    expand_16: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_38, [1, 4, 512, 64]);  permute_38 = None
    view_81: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_16, [4, 512, 64]);  expand_16 = None
    bmm_7: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_80, view_81);  view_80 = view_81 = None
    view_82: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_7, [1, 4, 512, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_41: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
    clone_14: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_41, memory_format = torch.contiguous_format);  permute_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_83: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_14, [1, 512, 256]);  clone_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_84: "f32[512, 256]" = torch.ops.aten.view.default(view_83, [512, 256]);  view_83 = None
    permute_42: "f32[256, 256]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
    addmm_22: "f32[512, 256]" = torch.ops.aten.addmm.default(arg62_1, view_84, permute_42);  arg62_1 = view_84 = permute_42 = None
    view_85: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_22, [1, 512, 256]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    clone_15: "f32[1, 512, 256]" = torch.ops.aten.clone.default(view_85);  view_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_29: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(clone_15, add_27);  clone_15 = add_27 = None
    var_mean_7 = torch.ops.aten.var_mean.correction(add_29, [2], correction = 0, keepdim = True)
    getitem_14: "f32[1, 512, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 512, 1]" = var_mean_7[1];  var_mean_7 = None
    add_30: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
    rsqrt_7: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
    sub_12: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_29, getitem_15);  add_29 = getitem_15 = None
    mul_24: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_7);  sub_12 = rsqrt_7 = None
    mul_25: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_24, arg63_1);  mul_24 = arg63_1 = None
    add_31: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_25, arg64_1);  mul_25 = arg64_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_86: "f32[512, 256]" = torch.ops.aten.view.default(add_31, [512, 256])
    permute_43: "f32[256, 1024]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
    addmm_23: "f32[512, 1024]" = torch.ops.aten.addmm.default(arg66_1, view_86, permute_43);  arg66_1 = view_86 = permute_43 = None
    view_87: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_23, [1, 512, 1024]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_26: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_87, 0.5)
    mul_27: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_87, 0.7071067811865476);  view_87 = None
    erf_3: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_27);  mul_27 = None
    add_32: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_28: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_26, add_32);  mul_26 = add_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_88: "f32[512, 1024]" = torch.ops.aten.view.default(mul_28, [512, 1024]);  mul_28 = None
    permute_44: "f32[1024, 256]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
    addmm_24: "f32[512, 256]" = torch.ops.aten.addmm.default(arg68_1, view_88, permute_44);  arg68_1 = view_88 = permute_44 = None
    view_89: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_24, [1, 512, 256]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    clone_16: "f32[1, 512, 256]" = torch.ops.aten.clone.default(view_89);  view_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_33: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(clone_16, add_31);  clone_16 = add_31 = None
    var_mean_8 = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
    getitem_16: "f32[1, 512, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 512, 1]" = var_mean_8[1];  var_mean_8 = None
    add_34: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
    rsqrt_8: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
    sub_13: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_33, getitem_17);  add_33 = getitem_17 = None
    mul_29: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_8);  sub_13 = rsqrt_8 = None
    mul_30: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_29, arg69_1);  mul_29 = arg69_1 = None
    add_35: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_30, arg70_1);  mul_30 = arg70_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_90: "f32[512, 256]" = torch.ops.aten.view.default(add_35, [512, 256])
    permute_45: "f32[256, 256]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
    addmm_25: "f32[512, 256]" = torch.ops.aten.addmm.default(arg72_1, view_90, permute_45);  arg72_1 = view_90 = permute_45 = None
    view_91: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_25, [1, 512, 256]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_92: "f32[512, 256]" = torch.ops.aten.view.default(add_35, [512, 256])
    permute_46: "f32[256, 256]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
    addmm_26: "f32[512, 256]" = torch.ops.aten.addmm.default(arg74_1, view_92, permute_46);  arg74_1 = view_92 = permute_46 = None
    view_93: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_26, [1, 512, 256]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_94: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_93, [1, 512, 4, 64]);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_47: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_95: "f32[512, 256]" = torch.ops.aten.view.default(add_35, [512, 256])
    permute_48: "f32[256, 256]" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
    addmm_27: "f32[512, 256]" = torch.ops.aten.addmm.default(arg76_1, view_95, permute_48);  arg76_1 = view_95 = permute_48 = None
    view_96: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_27, [1, 512, 256]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_97: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_96, [1, 512, 4, 64]);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_49: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_97, [0, 2, 1, 3]);  view_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_98: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_91, [1, 512, 4, 64]);  view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_50: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_51: "f32[1, 4, 64, 512]" = torch.ops.aten.permute.default(permute_47, [0, 1, 3, 2]);  permute_47 = None
    expand_17: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_50, [1, 4, 512, 64]);  permute_50 = None
    view_99: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_17, [4, 512, 64]);  expand_17 = None
    expand_18: "f32[1, 4, 64, 512]" = torch.ops.aten.expand.default(permute_51, [1, 4, 64, 512]);  permute_51 = None
    view_100: "f32[4, 64, 512]" = torch.ops.aten.view.default(expand_18, [4, 64, 512]);  expand_18 = None
    bmm_8: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_99, view_100);  view_99 = view_100 = None
    view_101: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_8, [1, 4, 512, 512]);  bmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_8: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(view_101, 8.0);  view_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:324, code: attention_scores = attention_scores + attention_mask
    add_36: "f32[1, 4, 512, 512]" = torch.ops.aten.add.Tensor(div_8, mul);  div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_4: "f32[1, 4, 512, 1]" = torch.ops.aten.amax.default(add_36, [-1], True)
    sub_14: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_36, amax_4);  add_36 = amax_4 = None
    exp_4: "f32[1, 4, 512, 512]" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
    sum_5: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_9: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    clone_17: "f32[1, 4, 512, 512]" = torch.ops.aten.clone.default(div_9);  div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_19: "f32[1, 4, 512, 512]" = torch.ops.aten.expand.default(clone_17, [1, 4, 512, 512]);  clone_17 = None
    view_102: "f32[4, 512, 512]" = torch.ops.aten.view.default(expand_19, [4, 512, 512]);  expand_19 = None
    expand_20: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_49, [1, 4, 512, 64]);  permute_49 = None
    view_103: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_20, [4, 512, 64]);  expand_20 = None
    bmm_9: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_102, view_103);  view_102 = view_103 = None
    view_104: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_9, [1, 4, 512, 64]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_52: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None
    clone_18: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_105: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_18, [1, 512, 256]);  clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_106: "f32[512, 256]" = torch.ops.aten.view.default(view_105, [512, 256]);  view_105 = None
    permute_53: "f32[256, 256]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
    addmm_28: "f32[512, 256]" = torch.ops.aten.addmm.default(arg78_1, view_106, permute_53);  arg78_1 = view_106 = permute_53 = None
    view_107: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_28, [1, 512, 256]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    clone_19: "f32[1, 512, 256]" = torch.ops.aten.clone.default(view_107);  view_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_37: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(clone_19, add_35);  clone_19 = add_35 = None
    var_mean_9 = torch.ops.aten.var_mean.correction(add_37, [2], correction = 0, keepdim = True)
    getitem_18: "f32[1, 512, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 512, 1]" = var_mean_9[1];  var_mean_9 = None
    add_38: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
    rsqrt_9: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
    sub_15: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_37, getitem_19);  add_37 = getitem_19 = None
    mul_31: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_9);  sub_15 = rsqrt_9 = None
    mul_32: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_31, arg79_1);  mul_31 = arg79_1 = None
    add_39: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_32, arg80_1);  mul_32 = arg80_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_108: "f32[512, 256]" = torch.ops.aten.view.default(add_39, [512, 256])
    permute_54: "f32[256, 1024]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
    addmm_29: "f32[512, 1024]" = torch.ops.aten.addmm.default(arg82_1, view_108, permute_54);  arg82_1 = view_108 = permute_54 = None
    view_109: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_29, [1, 512, 1024]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_33: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_109, 0.5)
    mul_34: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_109, 0.7071067811865476);  view_109 = None
    erf_4: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_34);  mul_34 = None
    add_40: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_35: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_33, add_40);  mul_33 = add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_110: "f32[512, 1024]" = torch.ops.aten.view.default(mul_35, [512, 1024]);  mul_35 = None
    permute_55: "f32[1024, 256]" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
    addmm_30: "f32[512, 256]" = torch.ops.aten.addmm.default(arg84_1, view_110, permute_55);  arg84_1 = view_110 = permute_55 = None
    view_111: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_30, [1, 512, 256]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    clone_20: "f32[1, 512, 256]" = torch.ops.aten.clone.default(view_111);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_41: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(clone_20, add_39);  clone_20 = add_39 = None
    var_mean_10 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
    getitem_20: "f32[1, 512, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 512, 1]" = var_mean_10[1];  var_mean_10 = None
    add_42: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
    rsqrt_10: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_16: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_41, getitem_21);  add_41 = getitem_21 = None
    mul_36: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_10);  sub_16 = rsqrt_10 = None
    mul_37: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_36, arg85_1);  mul_36 = arg85_1 = None
    add_43: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_37, arg86_1);  mul_37 = arg86_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_112: "f32[512, 256]" = torch.ops.aten.view.default(add_43, [512, 256])
    permute_56: "f32[256, 256]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
    addmm_31: "f32[512, 256]" = torch.ops.aten.addmm.default(arg88_1, view_112, permute_56);  arg88_1 = view_112 = permute_56 = None
    view_113: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_31, [1, 512, 256]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_114: "f32[512, 256]" = torch.ops.aten.view.default(add_43, [512, 256])
    permute_57: "f32[256, 256]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
    addmm_32: "f32[512, 256]" = torch.ops.aten.addmm.default(arg90_1, view_114, permute_57);  arg90_1 = view_114 = permute_57 = None
    view_115: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_32, [1, 512, 256]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_116: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_115, [1, 512, 4, 64]);  view_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_58: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_116, [0, 2, 1, 3]);  view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_117: "f32[512, 256]" = torch.ops.aten.view.default(add_43, [512, 256])
    permute_59: "f32[256, 256]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
    addmm_33: "f32[512, 256]" = torch.ops.aten.addmm.default(arg92_1, view_117, permute_59);  arg92_1 = view_117 = permute_59 = None
    view_118: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_33, [1, 512, 256]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_119: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_118, [1, 512, 4, 64]);  view_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_60: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_119, [0, 2, 1, 3]);  view_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_120: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_113, [1, 512, 4, 64]);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_61: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_62: "f32[1, 4, 64, 512]" = torch.ops.aten.permute.default(permute_58, [0, 1, 3, 2]);  permute_58 = None
    expand_21: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_61, [1, 4, 512, 64]);  permute_61 = None
    view_121: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_21, [4, 512, 64]);  expand_21 = None
    expand_22: "f32[1, 4, 64, 512]" = torch.ops.aten.expand.default(permute_62, [1, 4, 64, 512]);  permute_62 = None
    view_122: "f32[4, 64, 512]" = torch.ops.aten.view.default(expand_22, [4, 64, 512]);  expand_22 = None
    bmm_10: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_121, view_122);  view_121 = view_122 = None
    view_123: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_10, [1, 4, 512, 512]);  bmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_10: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(view_123, 8.0);  view_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:324, code: attention_scores = attention_scores + attention_mask
    add_44: "f32[1, 4, 512, 512]" = torch.ops.aten.add.Tensor(div_10, mul);  div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_5: "f32[1, 4, 512, 1]" = torch.ops.aten.amax.default(add_44, [-1], True)
    sub_17: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_44, amax_5);  add_44 = amax_5 = None
    exp_5: "f32[1, 4, 512, 512]" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
    sum_6: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_11: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    clone_21: "f32[1, 4, 512, 512]" = torch.ops.aten.clone.default(div_11);  div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_23: "f32[1, 4, 512, 512]" = torch.ops.aten.expand.default(clone_21, [1, 4, 512, 512]);  clone_21 = None
    view_124: "f32[4, 512, 512]" = torch.ops.aten.view.default(expand_23, [4, 512, 512]);  expand_23 = None
    expand_24: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_60, [1, 4, 512, 64]);  permute_60 = None
    view_125: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_24, [4, 512, 64]);  expand_24 = None
    bmm_11: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_124, view_125);  view_124 = view_125 = None
    view_126: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_11, [1, 4, 512, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_63: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
    clone_22: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format);  permute_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_127: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_22, [1, 512, 256]);  clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_128: "f32[512, 256]" = torch.ops.aten.view.default(view_127, [512, 256]);  view_127 = None
    permute_64: "f32[256, 256]" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
    addmm_34: "f32[512, 256]" = torch.ops.aten.addmm.default(arg94_1, view_128, permute_64);  arg94_1 = view_128 = permute_64 = None
    view_129: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_34, [1, 512, 256]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    clone_23: "f32[1, 512, 256]" = torch.ops.aten.clone.default(view_129);  view_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_45: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(clone_23, add_43);  clone_23 = add_43 = None
    var_mean_11 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
    getitem_22: "f32[1, 512, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 512, 1]" = var_mean_11[1];  var_mean_11 = None
    add_46: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
    rsqrt_11: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
    sub_18: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_45, getitem_23);  add_45 = getitem_23 = None
    mul_38: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_11);  sub_18 = rsqrt_11 = None
    mul_39: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_38, arg95_1);  mul_38 = arg95_1 = None
    add_47: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_39, arg96_1);  mul_39 = arg96_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_130: "f32[512, 256]" = torch.ops.aten.view.default(add_47, [512, 256])
    permute_65: "f32[256, 1024]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
    addmm_35: "f32[512, 1024]" = torch.ops.aten.addmm.default(arg98_1, view_130, permute_65);  arg98_1 = view_130 = permute_65 = None
    view_131: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_35, [1, 512, 1024]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_40: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_131, 0.5)
    mul_41: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_131, 0.7071067811865476);  view_131 = None
    erf_5: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_41);  mul_41 = None
    add_48: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_42: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_40, add_48);  mul_40 = add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_132: "f32[512, 1024]" = torch.ops.aten.view.default(mul_42, [512, 1024]);  mul_42 = None
    permute_66: "f32[1024, 256]" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
    addmm_36: "f32[512, 256]" = torch.ops.aten.addmm.default(arg100_1, view_132, permute_66);  arg100_1 = view_132 = permute_66 = None
    view_133: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_36, [1, 512, 256]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    clone_24: "f32[1, 512, 256]" = torch.ops.aten.clone.default(view_133);  view_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_49: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(clone_24, add_47);  clone_24 = add_47 = None
    var_mean_12 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
    getitem_24: "f32[1, 512, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 512, 1]" = var_mean_12[1];  var_mean_12 = None
    add_50: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
    rsqrt_12: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
    sub_19: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_49, getitem_25);  add_49 = getitem_25 = None
    mul_43: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_12);  sub_19 = rsqrt_12 = None
    mul_44: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_43, arg101_1);  mul_43 = arg101_1 = None
    add_51: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_44, arg102_1);  mul_44 = arg102_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_134: "f32[512, 256]" = torch.ops.aten.view.default(add_51, [512, 256])
    permute_67: "f32[256, 256]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
    addmm_37: "f32[512, 256]" = torch.ops.aten.addmm.default(arg104_1, view_134, permute_67);  arg104_1 = view_134 = permute_67 = None
    view_135: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_37, [1, 512, 256]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_136: "f32[512, 256]" = torch.ops.aten.view.default(add_51, [512, 256])
    permute_68: "f32[256, 256]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
    addmm_38: "f32[512, 256]" = torch.ops.aten.addmm.default(arg106_1, view_136, permute_68);  arg106_1 = view_136 = permute_68 = None
    view_137: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_38, [1, 512, 256]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_138: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_137, [1, 512, 4, 64]);  view_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_69: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_139: "f32[512, 256]" = torch.ops.aten.view.default(add_51, [512, 256])
    permute_70: "f32[256, 256]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
    addmm_39: "f32[512, 256]" = torch.ops.aten.addmm.default(arg108_1, view_139, permute_70);  arg108_1 = view_139 = permute_70 = None
    view_140: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_39, [1, 512, 256]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_141: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_140, [1, 512, 4, 64]);  view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_71: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_141, [0, 2, 1, 3]);  view_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_142: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_135, [1, 512, 4, 64]);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_72: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_142, [0, 2, 1, 3]);  view_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_73: "f32[1, 4, 64, 512]" = torch.ops.aten.permute.default(permute_69, [0, 1, 3, 2]);  permute_69 = None
    expand_25: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_72, [1, 4, 512, 64]);  permute_72 = None
    view_143: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_25, [4, 512, 64]);  expand_25 = None
    expand_26: "f32[1, 4, 64, 512]" = torch.ops.aten.expand.default(permute_73, [1, 4, 64, 512]);  permute_73 = None
    view_144: "f32[4, 64, 512]" = torch.ops.aten.view.default(expand_26, [4, 64, 512]);  expand_26 = None
    bmm_12: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_143, view_144);  view_143 = view_144 = None
    view_145: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_12, [1, 4, 512, 512]);  bmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_12: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(view_145, 8.0);  view_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:324, code: attention_scores = attention_scores + attention_mask
    add_52: "f32[1, 4, 512, 512]" = torch.ops.aten.add.Tensor(div_12, mul);  div_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_6: "f32[1, 4, 512, 1]" = torch.ops.aten.amax.default(add_52, [-1], True)
    sub_20: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_52, amax_6);  add_52 = amax_6 = None
    exp_6: "f32[1, 4, 512, 512]" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
    sum_7: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_13: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    clone_25: "f32[1, 4, 512, 512]" = torch.ops.aten.clone.default(div_13);  div_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_27: "f32[1, 4, 512, 512]" = torch.ops.aten.expand.default(clone_25, [1, 4, 512, 512]);  clone_25 = None
    view_146: "f32[4, 512, 512]" = torch.ops.aten.view.default(expand_27, [4, 512, 512]);  expand_27 = None
    expand_28: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_71, [1, 4, 512, 64]);  permute_71 = None
    view_147: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_28, [4, 512, 64]);  expand_28 = None
    bmm_13: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_146, view_147);  view_146 = view_147 = None
    view_148: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_13, [1, 4, 512, 64]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_74: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
    clone_26: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_74, memory_format = torch.contiguous_format);  permute_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_149: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_26, [1, 512, 256]);  clone_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_150: "f32[512, 256]" = torch.ops.aten.view.default(view_149, [512, 256]);  view_149 = None
    permute_75: "f32[256, 256]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
    addmm_40: "f32[512, 256]" = torch.ops.aten.addmm.default(arg110_1, view_150, permute_75);  arg110_1 = view_150 = permute_75 = None
    view_151: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_40, [1, 512, 256]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    clone_27: "f32[1, 512, 256]" = torch.ops.aten.clone.default(view_151);  view_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_53: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(clone_27, add_51);  clone_27 = add_51 = None
    var_mean_13 = torch.ops.aten.var_mean.correction(add_53, [2], correction = 0, keepdim = True)
    getitem_26: "f32[1, 512, 1]" = var_mean_13[0]
    getitem_27: "f32[1, 512, 1]" = var_mean_13[1];  var_mean_13 = None
    add_54: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
    rsqrt_13: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
    sub_21: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_53, getitem_27);  add_53 = getitem_27 = None
    mul_45: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_13);  sub_21 = rsqrt_13 = None
    mul_46: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_45, arg111_1);  mul_45 = arg111_1 = None
    add_55: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_46, arg112_1);  mul_46 = arg112_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_152: "f32[512, 256]" = torch.ops.aten.view.default(add_55, [512, 256])
    permute_76: "f32[256, 1024]" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
    addmm_41: "f32[512, 1024]" = torch.ops.aten.addmm.default(arg114_1, view_152, permute_76);  arg114_1 = view_152 = permute_76 = None
    view_153: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_41, [1, 512, 1024]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_47: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_153, 0.5)
    mul_48: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_153, 0.7071067811865476);  view_153 = None
    erf_6: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_48);  mul_48 = None
    add_56: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_49: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_47, add_56);  mul_47 = add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_154: "f32[512, 1024]" = torch.ops.aten.view.default(mul_49, [512, 1024]);  mul_49 = None
    permute_77: "f32[1024, 256]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
    addmm_42: "f32[512, 256]" = torch.ops.aten.addmm.default(arg116_1, view_154, permute_77);  arg116_1 = view_154 = permute_77 = None
    view_155: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_42, [1, 512, 256]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    clone_28: "f32[1, 512, 256]" = torch.ops.aten.clone.default(view_155);  view_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_57: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(clone_28, add_55);  clone_28 = add_55 = None
    var_mean_14 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 512, 1]" = var_mean_14[0]
    getitem_29: "f32[1, 512, 1]" = var_mean_14[1];  var_mean_14 = None
    add_58: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
    rsqrt_14: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
    sub_22: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_57, getitem_29);  add_57 = getitem_29 = None
    mul_50: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_14);  sub_22 = rsqrt_14 = None
    mul_51: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_50, arg117_1);  mul_50 = arg117_1 = None
    add_59: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_51, arg118_1);  mul_51 = arg118_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_156: "f32[512, 256]" = torch.ops.aten.view.default(add_59, [512, 256])
    permute_78: "f32[256, 256]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
    addmm_43: "f32[512, 256]" = torch.ops.aten.addmm.default(arg120_1, view_156, permute_78);  arg120_1 = view_156 = permute_78 = None
    view_157: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_43, [1, 512, 256]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_158: "f32[512, 256]" = torch.ops.aten.view.default(add_59, [512, 256])
    permute_79: "f32[256, 256]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
    addmm_44: "f32[512, 256]" = torch.ops.aten.addmm.default(arg122_1, view_158, permute_79);  arg122_1 = view_158 = permute_79 = None
    view_159: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_44, [1, 512, 256]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_160: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_159, [1, 512, 4, 64]);  view_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_80: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_160, [0, 2, 1, 3]);  view_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_161: "f32[512, 256]" = torch.ops.aten.view.default(add_59, [512, 256])
    permute_81: "f32[256, 256]" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
    addmm_45: "f32[512, 256]" = torch.ops.aten.addmm.default(arg124_1, view_161, permute_81);  arg124_1 = view_161 = permute_81 = None
    view_162: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_45, [1, 512, 256]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_163: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_162, [1, 512, 4, 64]);  view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_82: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_163, [0, 2, 1, 3]);  view_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_164: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_157, [1, 512, 4, 64]);  view_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_83: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_164, [0, 2, 1, 3]);  view_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_84: "f32[1, 4, 64, 512]" = torch.ops.aten.permute.default(permute_80, [0, 1, 3, 2]);  permute_80 = None
    expand_29: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_83, [1, 4, 512, 64]);  permute_83 = None
    view_165: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_29, [4, 512, 64]);  expand_29 = None
    expand_30: "f32[1, 4, 64, 512]" = torch.ops.aten.expand.default(permute_84, [1, 4, 64, 512]);  permute_84 = None
    view_166: "f32[4, 64, 512]" = torch.ops.aten.view.default(expand_30, [4, 64, 512]);  expand_30 = None
    bmm_14: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_165, view_166);  view_165 = view_166 = None
    view_167: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_14, [1, 4, 512, 512]);  bmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_14: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(view_167, 8.0);  view_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:324, code: attention_scores = attention_scores + attention_mask
    add_60: "f32[1, 4, 512, 512]" = torch.ops.aten.add.Tensor(div_14, mul);  div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_7: "f32[1, 4, 512, 1]" = torch.ops.aten.amax.default(add_60, [-1], True)
    sub_23: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_60, amax_7);  add_60 = amax_7 = None
    exp_7: "f32[1, 4, 512, 512]" = torch.ops.aten.exp.default(sub_23);  sub_23 = None
    sum_8: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_15: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    clone_29: "f32[1, 4, 512, 512]" = torch.ops.aten.clone.default(div_15);  div_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_31: "f32[1, 4, 512, 512]" = torch.ops.aten.expand.default(clone_29, [1, 4, 512, 512]);  clone_29 = None
    view_168: "f32[4, 512, 512]" = torch.ops.aten.view.default(expand_31, [4, 512, 512]);  expand_31 = None
    expand_32: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_82, [1, 4, 512, 64]);  permute_82 = None
    view_169: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_32, [4, 512, 64]);  expand_32 = None
    bmm_15: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_168, view_169);  view_168 = view_169 = None
    view_170: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_15, [1, 4, 512, 64]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_85: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_170, [0, 2, 1, 3]);  view_170 = None
    clone_30: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_85, memory_format = torch.contiguous_format);  permute_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_171: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_30, [1, 512, 256]);  clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_172: "f32[512, 256]" = torch.ops.aten.view.default(view_171, [512, 256]);  view_171 = None
    permute_86: "f32[256, 256]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
    addmm_46: "f32[512, 256]" = torch.ops.aten.addmm.default(arg126_1, view_172, permute_86);  arg126_1 = view_172 = permute_86 = None
    view_173: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_46, [1, 512, 256]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    clone_31: "f32[1, 512, 256]" = torch.ops.aten.clone.default(view_173);  view_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_61: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(clone_31, add_59);  clone_31 = add_59 = None
    var_mean_15 = torch.ops.aten.var_mean.correction(add_61, [2], correction = 0, keepdim = True)
    getitem_30: "f32[1, 512, 1]" = var_mean_15[0]
    getitem_31: "f32[1, 512, 1]" = var_mean_15[1];  var_mean_15 = None
    add_62: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
    rsqrt_15: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
    sub_24: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_61, getitem_31);  add_61 = getitem_31 = None
    mul_52: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_15);  sub_24 = rsqrt_15 = None
    mul_53: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_52, arg127_1);  mul_52 = arg127_1 = None
    add_63: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_53, arg128_1);  mul_53 = arg128_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_174: "f32[512, 256]" = torch.ops.aten.view.default(add_63, [512, 256])
    permute_87: "f32[256, 1024]" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
    addmm_47: "f32[512, 1024]" = torch.ops.aten.addmm.default(arg130_1, view_174, permute_87);  arg130_1 = view_174 = permute_87 = None
    view_175: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_47, [1, 512, 1024]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_54: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_175, 0.5)
    mul_55: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_175, 0.7071067811865476);  view_175 = None
    erf_7: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
    add_64: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_56: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_54, add_64);  mul_54 = add_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_176: "f32[512, 1024]" = torch.ops.aten.view.default(mul_56, [512, 1024]);  mul_56 = None
    permute_88: "f32[1024, 256]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
    addmm_48: "f32[512, 256]" = torch.ops.aten.addmm.default(arg132_1, view_176, permute_88);  arg132_1 = view_176 = permute_88 = None
    view_177: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_48, [1, 512, 256]);  addmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    clone_32: "f32[1, 512, 256]" = torch.ops.aten.clone.default(view_177);  view_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_65: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(clone_32, add_63);  clone_32 = add_63 = None
    var_mean_16 = torch.ops.aten.var_mean.correction(add_65, [2], correction = 0, keepdim = True)
    getitem_32: "f32[1, 512, 1]" = var_mean_16[0]
    getitem_33: "f32[1, 512, 1]" = var_mean_16[1];  var_mean_16 = None
    add_66: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
    rsqrt_16: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    sub_25: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_65, getitem_33);  add_65 = getitem_33 = None
    mul_57: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_16);  sub_25 = rsqrt_16 = None
    mul_58: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_57, arg133_1);  mul_57 = arg133_1 = None
    add_67: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_58, arg134_1);  mul_58 = arg134_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_178: "f32[512, 256]" = torch.ops.aten.view.default(add_67, [512, 256])
    permute_89: "f32[256, 256]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
    addmm_49: "f32[512, 256]" = torch.ops.aten.addmm.default(arg136_1, view_178, permute_89);  arg136_1 = view_178 = permute_89 = None
    view_179: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_49, [1, 512, 256]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_180: "f32[512, 256]" = torch.ops.aten.view.default(add_67, [512, 256])
    permute_90: "f32[256, 256]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
    addmm_50: "f32[512, 256]" = torch.ops.aten.addmm.default(arg138_1, view_180, permute_90);  arg138_1 = view_180 = permute_90 = None
    view_181: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_50, [1, 512, 256]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_182: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_181, [1, 512, 4, 64]);  view_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_91: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_183: "f32[512, 256]" = torch.ops.aten.view.default(add_67, [512, 256])
    permute_92: "f32[256, 256]" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
    addmm_51: "f32[512, 256]" = torch.ops.aten.addmm.default(arg140_1, view_183, permute_92);  arg140_1 = view_183 = permute_92 = None
    view_184: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_51, [1, 512, 256]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_185: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_184, [1, 512, 4, 64]);  view_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_93: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_186: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_179, [1, 512, 4, 64]);  view_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_94: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_95: "f32[1, 4, 64, 512]" = torch.ops.aten.permute.default(permute_91, [0, 1, 3, 2]);  permute_91 = None
    expand_33: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_94, [1, 4, 512, 64]);  permute_94 = None
    view_187: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_33, [4, 512, 64]);  expand_33 = None
    expand_34: "f32[1, 4, 64, 512]" = torch.ops.aten.expand.default(permute_95, [1, 4, 64, 512]);  permute_95 = None
    view_188: "f32[4, 64, 512]" = torch.ops.aten.view.default(expand_34, [4, 64, 512]);  expand_34 = None
    bmm_16: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_187, view_188);  view_187 = view_188 = None
    view_189: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_16, [1, 4, 512, 512]);  bmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_16: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(view_189, 8.0);  view_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:324, code: attention_scores = attention_scores + attention_mask
    add_68: "f32[1, 4, 512, 512]" = torch.ops.aten.add.Tensor(div_16, mul);  div_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_8: "f32[1, 4, 512, 1]" = torch.ops.aten.amax.default(add_68, [-1], True)
    sub_26: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_68, amax_8);  add_68 = amax_8 = None
    exp_8: "f32[1, 4, 512, 512]" = torch.ops.aten.exp.default(sub_26);  sub_26 = None
    sum_9: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_17: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    clone_33: "f32[1, 4, 512, 512]" = torch.ops.aten.clone.default(div_17);  div_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_35: "f32[1, 4, 512, 512]" = torch.ops.aten.expand.default(clone_33, [1, 4, 512, 512]);  clone_33 = None
    view_190: "f32[4, 512, 512]" = torch.ops.aten.view.default(expand_35, [4, 512, 512]);  expand_35 = None
    expand_36: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_93, [1, 4, 512, 64]);  permute_93 = None
    view_191: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_36, [4, 512, 64]);  expand_36 = None
    bmm_17: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_190, view_191);  view_190 = view_191 = None
    view_192: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_17, [1, 4, 512, 64]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_96: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None
    clone_34: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_96, memory_format = torch.contiguous_format);  permute_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_193: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_34, [1, 512, 256]);  clone_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_194: "f32[512, 256]" = torch.ops.aten.view.default(view_193, [512, 256]);  view_193 = None
    permute_97: "f32[256, 256]" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
    addmm_52: "f32[512, 256]" = torch.ops.aten.addmm.default(arg142_1, view_194, permute_97);  arg142_1 = view_194 = permute_97 = None
    view_195: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_52, [1, 512, 256]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    clone_35: "f32[1, 512, 256]" = torch.ops.aten.clone.default(view_195);  view_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_69: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(clone_35, add_67);  clone_35 = add_67 = None
    var_mean_17 = torch.ops.aten.var_mean.correction(add_69, [2], correction = 0, keepdim = True)
    getitem_34: "f32[1, 512, 1]" = var_mean_17[0]
    getitem_35: "f32[1, 512, 1]" = var_mean_17[1];  var_mean_17 = None
    add_70: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
    rsqrt_17: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
    sub_27: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_69, getitem_35);  add_69 = getitem_35 = None
    mul_59: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_17);  sub_27 = rsqrt_17 = None
    mul_60: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_59, arg143_1);  mul_59 = arg143_1 = None
    add_71: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_60, arg144_1);  mul_60 = arg144_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_196: "f32[512, 256]" = torch.ops.aten.view.default(add_71, [512, 256])
    permute_98: "f32[256, 1024]" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
    addmm_53: "f32[512, 1024]" = torch.ops.aten.addmm.default(arg146_1, view_196, permute_98);  arg146_1 = view_196 = permute_98 = None
    view_197: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_53, [1, 512, 1024]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_61: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_197, 0.5)
    mul_62: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_197, 0.7071067811865476);  view_197 = None
    erf_8: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_62);  mul_62 = None
    add_72: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_63: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_61, add_72);  mul_61 = add_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_198: "f32[512, 1024]" = torch.ops.aten.view.default(mul_63, [512, 1024]);  mul_63 = None
    permute_99: "f32[1024, 256]" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
    addmm_54: "f32[512, 256]" = torch.ops.aten.addmm.default(arg148_1, view_198, permute_99);  arg148_1 = view_198 = permute_99 = None
    view_199: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_54, [1, 512, 256]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    clone_36: "f32[1, 512, 256]" = torch.ops.aten.clone.default(view_199);  view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_73: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(clone_36, add_71);  clone_36 = add_71 = None
    var_mean_18 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
    getitem_36: "f32[1, 512, 1]" = var_mean_18[0]
    getitem_37: "f32[1, 512, 1]" = var_mean_18[1];  var_mean_18 = None
    add_74: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
    rsqrt_18: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    sub_28: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_73, getitem_37);  add_73 = getitem_37 = None
    mul_64: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_18);  sub_28 = rsqrt_18 = None
    mul_65: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_64, arg149_1);  mul_64 = arg149_1 = None
    add_75: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_65, arg150_1);  mul_65 = arg150_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_200: "f32[512, 256]" = torch.ops.aten.view.default(add_75, [512, 256])
    permute_100: "f32[256, 256]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
    addmm_55: "f32[512, 256]" = torch.ops.aten.addmm.default(arg152_1, view_200, permute_100);  arg152_1 = view_200 = permute_100 = None
    view_201: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_55, [1, 512, 256]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_202: "f32[512, 256]" = torch.ops.aten.view.default(add_75, [512, 256])
    permute_101: "f32[256, 256]" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
    addmm_56: "f32[512, 256]" = torch.ops.aten.addmm.default(arg154_1, view_202, permute_101);  arg154_1 = view_202 = permute_101 = None
    view_203: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_56, [1, 512, 256]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_204: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_203, [1, 512, 4, 64]);  view_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_102: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_204, [0, 2, 1, 3]);  view_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_205: "f32[512, 256]" = torch.ops.aten.view.default(add_75, [512, 256])
    permute_103: "f32[256, 256]" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
    addmm_57: "f32[512, 256]" = torch.ops.aten.addmm.default(arg156_1, view_205, permute_103);  arg156_1 = view_205 = permute_103 = None
    view_206: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_57, [1, 512, 256]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_207: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_206, [1, 512, 4, 64]);  view_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_104: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_207, [0, 2, 1, 3]);  view_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_208: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_201, [1, 512, 4, 64]);  view_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_105: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_106: "f32[1, 4, 64, 512]" = torch.ops.aten.permute.default(permute_102, [0, 1, 3, 2]);  permute_102 = None
    expand_37: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_105, [1, 4, 512, 64]);  permute_105 = None
    view_209: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_37, [4, 512, 64]);  expand_37 = None
    expand_38: "f32[1, 4, 64, 512]" = torch.ops.aten.expand.default(permute_106, [1, 4, 64, 512]);  permute_106 = None
    view_210: "f32[4, 64, 512]" = torch.ops.aten.view.default(expand_38, [4, 64, 512]);  expand_38 = None
    bmm_18: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_209, view_210);  view_209 = view_210 = None
    view_211: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_18, [1, 4, 512, 512]);  bmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_18: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(view_211, 8.0);  view_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:324, code: attention_scores = attention_scores + attention_mask
    add_76: "f32[1, 4, 512, 512]" = torch.ops.aten.add.Tensor(div_18, mul);  div_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_9: "f32[1, 4, 512, 1]" = torch.ops.aten.amax.default(add_76, [-1], True)
    sub_29: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_76, amax_9);  add_76 = amax_9 = None
    exp_9: "f32[1, 4, 512, 512]" = torch.ops.aten.exp.default(sub_29);  sub_29 = None
    sum_10: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_19: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    clone_37: "f32[1, 4, 512, 512]" = torch.ops.aten.clone.default(div_19);  div_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_39: "f32[1, 4, 512, 512]" = torch.ops.aten.expand.default(clone_37, [1, 4, 512, 512]);  clone_37 = None
    view_212: "f32[4, 512, 512]" = torch.ops.aten.view.default(expand_39, [4, 512, 512]);  expand_39 = None
    expand_40: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_104, [1, 4, 512, 64]);  permute_104 = None
    view_213: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_40, [4, 512, 64]);  expand_40 = None
    bmm_19: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_212, view_213);  view_212 = view_213 = None
    view_214: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_19, [1, 4, 512, 64]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_107: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_214, [0, 2, 1, 3]);  view_214 = None
    clone_38: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_107, memory_format = torch.contiguous_format);  permute_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_215: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_38, [1, 512, 256]);  clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_216: "f32[512, 256]" = torch.ops.aten.view.default(view_215, [512, 256]);  view_215 = None
    permute_108: "f32[256, 256]" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
    addmm_58: "f32[512, 256]" = torch.ops.aten.addmm.default(arg158_1, view_216, permute_108);  arg158_1 = view_216 = permute_108 = None
    view_217: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_58, [1, 512, 256]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    clone_39: "f32[1, 512, 256]" = torch.ops.aten.clone.default(view_217);  view_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_77: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(clone_39, add_75);  clone_39 = add_75 = None
    var_mean_19 = torch.ops.aten.var_mean.correction(add_77, [2], correction = 0, keepdim = True)
    getitem_38: "f32[1, 512, 1]" = var_mean_19[0]
    getitem_39: "f32[1, 512, 1]" = var_mean_19[1];  var_mean_19 = None
    add_78: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
    rsqrt_19: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    sub_30: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_77, getitem_39);  add_77 = getitem_39 = None
    mul_66: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_19);  sub_30 = rsqrt_19 = None
    mul_67: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_66, arg159_1);  mul_66 = arg159_1 = None
    add_79: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_67, arg160_1);  mul_67 = arg160_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_218: "f32[512, 256]" = torch.ops.aten.view.default(add_79, [512, 256])
    permute_109: "f32[256, 1024]" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
    addmm_59: "f32[512, 1024]" = torch.ops.aten.addmm.default(arg162_1, view_218, permute_109);  arg162_1 = view_218 = permute_109 = None
    view_219: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_59, [1, 512, 1024]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_68: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_219, 0.5)
    mul_69: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_219, 0.7071067811865476);  view_219 = None
    erf_9: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
    add_80: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_70: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_68, add_80);  mul_68 = add_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_220: "f32[512, 1024]" = torch.ops.aten.view.default(mul_70, [512, 1024]);  mul_70 = None
    permute_110: "f32[1024, 256]" = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
    addmm_60: "f32[512, 256]" = torch.ops.aten.addmm.default(arg164_1, view_220, permute_110);  arg164_1 = view_220 = permute_110 = None
    view_221: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_60, [1, 512, 256]);  addmm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    clone_40: "f32[1, 512, 256]" = torch.ops.aten.clone.default(view_221);  view_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_81: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(clone_40, add_79);  clone_40 = add_79 = None
    var_mean_20 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
    getitem_40: "f32[1, 512, 1]" = var_mean_20[0]
    getitem_41: "f32[1, 512, 1]" = var_mean_20[1];  var_mean_20 = None
    add_82: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
    rsqrt_20: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
    sub_31: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_81, getitem_41);  add_81 = getitem_41 = None
    mul_71: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_20);  sub_31 = rsqrt_20 = None
    mul_72: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_71, arg165_1);  mul_71 = arg165_1 = None
    add_83: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_72, arg166_1);  mul_72 = arg166_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_222: "f32[512, 256]" = torch.ops.aten.view.default(add_83, [512, 256])
    permute_111: "f32[256, 256]" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
    addmm_61: "f32[512, 256]" = torch.ops.aten.addmm.default(arg168_1, view_222, permute_111);  arg168_1 = view_222 = permute_111 = None
    view_223: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_61, [1, 512, 256]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_224: "f32[512, 256]" = torch.ops.aten.view.default(add_83, [512, 256])
    permute_112: "f32[256, 256]" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
    addmm_62: "f32[512, 256]" = torch.ops.aten.addmm.default(arg170_1, view_224, permute_112);  arg170_1 = view_224 = permute_112 = None
    view_225: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_62, [1, 512, 256]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_226: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_225, [1, 512, 4, 64]);  view_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_113: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_227: "f32[512, 256]" = torch.ops.aten.view.default(add_83, [512, 256])
    permute_114: "f32[256, 256]" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
    addmm_63: "f32[512, 256]" = torch.ops.aten.addmm.default(arg172_1, view_227, permute_114);  arg172_1 = view_227 = permute_114 = None
    view_228: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_63, [1, 512, 256]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_229: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_228, [1, 512, 4, 64]);  view_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_115: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_229, [0, 2, 1, 3]);  view_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_230: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_223, [1, 512, 4, 64]);  view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_116: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_230, [0, 2, 1, 3]);  view_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_117: "f32[1, 4, 64, 512]" = torch.ops.aten.permute.default(permute_113, [0, 1, 3, 2]);  permute_113 = None
    expand_41: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_116, [1, 4, 512, 64]);  permute_116 = None
    view_231: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_41, [4, 512, 64]);  expand_41 = None
    expand_42: "f32[1, 4, 64, 512]" = torch.ops.aten.expand.default(permute_117, [1, 4, 64, 512]);  permute_117 = None
    view_232: "f32[4, 64, 512]" = torch.ops.aten.view.default(expand_42, [4, 64, 512]);  expand_42 = None
    bmm_20: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_231, view_232);  view_231 = view_232 = None
    view_233: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_20, [1, 4, 512, 512]);  bmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_20: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(view_233, 8.0);  view_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:324, code: attention_scores = attention_scores + attention_mask
    add_84: "f32[1, 4, 512, 512]" = torch.ops.aten.add.Tensor(div_20, mul);  div_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_10: "f32[1, 4, 512, 1]" = torch.ops.aten.amax.default(add_84, [-1], True)
    sub_32: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_84, amax_10);  add_84 = amax_10 = None
    exp_10: "f32[1, 4, 512, 512]" = torch.ops.aten.exp.default(sub_32);  sub_32 = None
    sum_11: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_21: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    clone_41: "f32[1, 4, 512, 512]" = torch.ops.aten.clone.default(div_21);  div_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_43: "f32[1, 4, 512, 512]" = torch.ops.aten.expand.default(clone_41, [1, 4, 512, 512]);  clone_41 = None
    view_234: "f32[4, 512, 512]" = torch.ops.aten.view.default(expand_43, [4, 512, 512]);  expand_43 = None
    expand_44: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_115, [1, 4, 512, 64]);  permute_115 = None
    view_235: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_44, [4, 512, 64]);  expand_44 = None
    bmm_21: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_234, view_235);  view_234 = view_235 = None
    view_236: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_21, [1, 4, 512, 64]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_118: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_236, [0, 2, 1, 3]);  view_236 = None
    clone_42: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_118, memory_format = torch.contiguous_format);  permute_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_237: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_42, [1, 512, 256]);  clone_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_238: "f32[512, 256]" = torch.ops.aten.view.default(view_237, [512, 256]);  view_237 = None
    permute_119: "f32[256, 256]" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
    addmm_64: "f32[512, 256]" = torch.ops.aten.addmm.default(arg174_1, view_238, permute_119);  arg174_1 = view_238 = permute_119 = None
    view_239: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_64, [1, 512, 256]);  addmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    clone_43: "f32[1, 512, 256]" = torch.ops.aten.clone.default(view_239);  view_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_85: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(clone_43, add_83);  clone_43 = add_83 = None
    var_mean_21 = torch.ops.aten.var_mean.correction(add_85, [2], correction = 0, keepdim = True)
    getitem_42: "f32[1, 512, 1]" = var_mean_21[0]
    getitem_43: "f32[1, 512, 1]" = var_mean_21[1];  var_mean_21 = None
    add_86: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
    rsqrt_21: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
    sub_33: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_85, getitem_43);  add_85 = getitem_43 = None
    mul_73: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_21);  sub_33 = rsqrt_21 = None
    mul_74: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_73, arg175_1);  mul_73 = arg175_1 = None
    add_87: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_74, arg176_1);  mul_74 = arg176_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_240: "f32[512, 256]" = torch.ops.aten.view.default(add_87, [512, 256])
    permute_120: "f32[256, 1024]" = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
    addmm_65: "f32[512, 1024]" = torch.ops.aten.addmm.default(arg178_1, view_240, permute_120);  arg178_1 = view_240 = permute_120 = None
    view_241: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_65, [1, 512, 1024]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_75: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_241, 0.5)
    mul_76: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_241, 0.7071067811865476);  view_241 = None
    erf_10: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_76);  mul_76 = None
    add_88: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_77: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_75, add_88);  mul_75 = add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_242: "f32[512, 1024]" = torch.ops.aten.view.default(mul_77, [512, 1024]);  mul_77 = None
    permute_121: "f32[1024, 256]" = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
    addmm_66: "f32[512, 256]" = torch.ops.aten.addmm.default(arg180_1, view_242, permute_121);  arg180_1 = view_242 = permute_121 = None
    view_243: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_66, [1, 512, 256]);  addmm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    clone_44: "f32[1, 512, 256]" = torch.ops.aten.clone.default(view_243);  view_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_89: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(clone_44, add_87);  clone_44 = add_87 = None
    var_mean_22 = torch.ops.aten.var_mean.correction(add_89, [2], correction = 0, keepdim = True)
    getitem_44: "f32[1, 512, 1]" = var_mean_22[0]
    getitem_45: "f32[1, 512, 1]" = var_mean_22[1];  var_mean_22 = None
    add_90: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
    rsqrt_22: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
    sub_34: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_89, getitem_45);  add_89 = getitem_45 = None
    mul_78: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_22);  sub_34 = rsqrt_22 = None
    mul_79: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_78, arg181_1);  mul_78 = arg181_1 = None
    add_91: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_79, arg182_1);  mul_79 = arg182_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_244: "f32[512, 256]" = torch.ops.aten.view.default(add_91, [512, 256])
    permute_122: "f32[256, 256]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
    addmm_67: "f32[512, 256]" = torch.ops.aten.addmm.default(arg184_1, view_244, permute_122);  arg184_1 = view_244 = permute_122 = None
    view_245: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_67, [1, 512, 256]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_246: "f32[512, 256]" = torch.ops.aten.view.default(add_91, [512, 256])
    permute_123: "f32[256, 256]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
    addmm_68: "f32[512, 256]" = torch.ops.aten.addmm.default(arg186_1, view_246, permute_123);  arg186_1 = view_246 = permute_123 = None
    view_247: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_68, [1, 512, 256]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_248: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_247, [1, 512, 4, 64]);  view_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_124: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_248, [0, 2, 1, 3]);  view_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_249: "f32[512, 256]" = torch.ops.aten.view.default(add_91, [512, 256])
    permute_125: "f32[256, 256]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
    addmm_69: "f32[512, 256]" = torch.ops.aten.addmm.default(arg188_1, view_249, permute_125);  arg188_1 = view_249 = permute_125 = None
    view_250: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_69, [1, 512, 256]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_251: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_250, [1, 512, 4, 64]);  view_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_126: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_251, [0, 2, 1, 3]);  view_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_252: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_245, [1, 512, 4, 64]);  view_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_127: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_128: "f32[1, 4, 64, 512]" = torch.ops.aten.permute.default(permute_124, [0, 1, 3, 2]);  permute_124 = None
    expand_45: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_127, [1, 4, 512, 64]);  permute_127 = None
    view_253: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_45, [4, 512, 64]);  expand_45 = None
    expand_46: "f32[1, 4, 64, 512]" = torch.ops.aten.expand.default(permute_128, [1, 4, 64, 512]);  permute_128 = None
    view_254: "f32[4, 64, 512]" = torch.ops.aten.view.default(expand_46, [4, 64, 512]);  expand_46 = None
    bmm_22: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_253, view_254);  view_253 = view_254 = None
    view_255: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_22, [1, 4, 512, 512]);  bmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_22: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(view_255, 8.0);  view_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:324, code: attention_scores = attention_scores + attention_mask
    add_92: "f32[1, 4, 512, 512]" = torch.ops.aten.add.Tensor(div_22, mul);  div_22 = mul = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_11: "f32[1, 4, 512, 1]" = torch.ops.aten.amax.default(add_92, [-1], True)
    sub_35: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_92, amax_11);  add_92 = amax_11 = None
    exp_11: "f32[1, 4, 512, 512]" = torch.ops.aten.exp.default(sub_35);  sub_35 = None
    sum_12: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_23: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    clone_45: "f32[1, 4, 512, 512]" = torch.ops.aten.clone.default(div_23);  div_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_47: "f32[1, 4, 512, 512]" = torch.ops.aten.expand.default(clone_45, [1, 4, 512, 512]);  clone_45 = None
    view_256: "f32[4, 512, 512]" = torch.ops.aten.view.default(expand_47, [4, 512, 512]);  expand_47 = None
    expand_48: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_126, [1, 4, 512, 64]);  permute_126 = None
    view_257: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_48, [4, 512, 64]);  expand_48 = None
    bmm_23: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_256, view_257);  view_256 = view_257 = None
    view_258: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_23, [1, 4, 512, 64]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_129: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_258, [0, 2, 1, 3]);  view_258 = None
    clone_46: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_259: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_46, [1, 512, 256]);  clone_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_260: "f32[512, 256]" = torch.ops.aten.view.default(view_259, [512, 256]);  view_259 = None
    permute_130: "f32[256, 256]" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
    addmm_70: "f32[512, 256]" = torch.ops.aten.addmm.default(arg190_1, view_260, permute_130);  arg190_1 = view_260 = permute_130 = None
    view_261: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_70, [1, 512, 256]);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    clone_47: "f32[1, 512, 256]" = torch.ops.aten.clone.default(view_261);  view_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_93: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(clone_47, add_91);  clone_47 = add_91 = None
    var_mean_23 = torch.ops.aten.var_mean.correction(add_93, [2], correction = 0, keepdim = True)
    getitem_46: "f32[1, 512, 1]" = var_mean_23[0]
    getitem_47: "f32[1, 512, 1]" = var_mean_23[1];  var_mean_23 = None
    add_94: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
    rsqrt_23: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
    sub_36: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_93, getitem_47);  add_93 = getitem_47 = None
    mul_80: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_23);  sub_36 = rsqrt_23 = None
    mul_81: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_80, arg191_1);  mul_80 = arg191_1 = None
    add_95: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_81, arg192_1);  mul_81 = arg192_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_262: "f32[512, 256]" = torch.ops.aten.view.default(add_95, [512, 256])
    permute_131: "f32[256, 1024]" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
    addmm_71: "f32[512, 1024]" = torch.ops.aten.addmm.default(arg194_1, view_262, permute_131);  arg194_1 = view_262 = permute_131 = None
    view_263: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_71, [1, 512, 1024]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_82: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_263, 0.5)
    mul_83: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_263, 0.7071067811865476);  view_263 = None
    erf_11: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
    add_96: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_84: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_82, add_96);  mul_82 = add_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_264: "f32[512, 1024]" = torch.ops.aten.view.default(mul_84, [512, 1024]);  mul_84 = None
    permute_132: "f32[1024, 256]" = torch.ops.aten.permute.default(arg195_1, [1, 0]);  arg195_1 = None
    addmm_72: "f32[512, 256]" = torch.ops.aten.addmm.default(arg196_1, view_264, permute_132);  arg196_1 = view_264 = permute_132 = None
    view_265: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_72, [1, 512, 256]);  addmm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    clone_48: "f32[1, 512, 256]" = torch.ops.aten.clone.default(view_265);  view_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_97: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(clone_48, add_95);  clone_48 = add_95 = None
    var_mean_24 = torch.ops.aten.var_mean.correction(add_97, [2], correction = 0, keepdim = True)
    getitem_48: "f32[1, 512, 1]" = var_mean_24[0]
    getitem_49: "f32[1, 512, 1]" = var_mean_24[1];  var_mean_24 = None
    add_98: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
    rsqrt_24: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
    sub_37: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_97, getitem_49);  add_97 = getitem_49 = None
    mul_85: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_24);  sub_37 = rsqrt_24 = None
    mul_86: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_85, arg197_1);  mul_85 = arg197_1 = None
    add_99: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_86, arg198_1);  mul_86 = arg198_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1404, code: logits = self.qa_outputs(sequence_output)
    view_266: "f32[512, 256]" = torch.ops.aten.view.default(add_99, [512, 256]);  add_99 = None
    permute_133: "f32[256, 2]" = torch.ops.aten.permute.default(arg199_1, [1, 0]);  arg199_1 = None
    addmm_73: "f32[512, 2]" = torch.ops.aten.addmm.default(arg200_1, view_266, permute_133);  arg200_1 = view_266 = permute_133 = None
    view_267: "f32[1, 512, 2]" = torch.ops.aten.view.default(addmm_73, [1, 512, 2]);  addmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1405, code: start_logits, end_logits = logits.split(1, dim=-1)
    split_with_sizes = torch.ops.aten.split_with_sizes.default(view_267, [1, 1], 2);  view_267 = None
    getitem_50: "f32[1, 512, 1]" = split_with_sizes[0]
    getitem_51: "f32[1, 512, 1]" = split_with_sizes[1];  split_with_sizes = None
    
    # No stacktrace found for following nodes
    squeeze: "f32[1, 512]" = torch.ops.aten.squeeze.dim(getitem_50, -1);  getitem_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1406, code: start_logits = start_logits.squeeze(-1).contiguous()
    clone_49: "f32[1, 512]" = torch.ops.aten.clone.default(squeeze, memory_format = torch.contiguous_format);  squeeze = None
    
    # No stacktrace found for following nodes
    squeeze_1: "f32[1, 512]" = torch.ops.aten.squeeze.dim(getitem_51, -1);  getitem_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1407, code: end_logits = end_logits.squeeze(-1).contiguous()
    clone_50: "f32[1, 512]" = torch.ops.aten.clone.default(squeeze_1, memory_format = torch.contiguous_format);  squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1418, code: start_positions = start_positions.clamp(0, ignored_index)
    clamp_min: "i64[1]" = torch.ops.aten.clamp_min.default(arg204_1, 0);  arg204_1 = None
    clamp_max: "i64[1]" = torch.ops.aten.clamp_max.default(clamp_min, 512);  clamp_min = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1419, code: end_positions = end_positions.clamp(0, ignored_index)
    clamp_min_1: "i64[1]" = torch.ops.aten.clamp_min.default(arg205_1, 0);  arg205_1 = None
    clamp_max_1: "i64[1]" = torch.ops.aten.clamp_max.default(clamp_min_1, 512);  clamp_min_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1422, code: start_loss = loss_fct(start_logits, start_positions)
    amax_12: "f32[1, 1]" = torch.ops.aten.amax.default(clone_49, [1], True)
    sub_38: "f32[1, 512]" = torch.ops.aten.sub.Tensor(clone_49, amax_12);  amax_12 = None
    exp_12: "f32[1, 512]" = torch.ops.aten.exp.default(sub_38)
    sum_13: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [1], True);  exp_12 = None
    log: "f32[1, 1]" = torch.ops.aten.log.default(sum_13);  sum_13 = None
    sub_39: "f32[1, 512]" = torch.ops.aten.sub.Tensor(sub_38, log);  sub_38 = log = None
    ne: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max, 512)
    scalar_tensor: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'))
    where: "i64[1]" = torch.ops.aten.where.self(ne, clamp_max, scalar_tensor);  ne = scalar_tensor = None
    unsqueeze_2: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
    gather: "f32[1, 1]" = torch.ops.aten.gather.default(sub_39, 1, unsqueeze_2);  sub_39 = unsqueeze_2 = None
    squeeze_2: "f32[1]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[1]" = torch.ops.aten.neg.default(squeeze_2);  squeeze_2 = None
    ne_1: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max, 512)
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_1: "f32[1]" = torch.ops.aten.where.self(ne_1, neg, scalar_tensor_1);  ne_1 = neg = scalar_tensor_1 = None
    ne_2: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max, 512);  clamp_max = None
    sum_14: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None
    sum_15: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
    div_24: "f32[]" = torch.ops.aten.div.Tensor(sum_15, convert_element_type);  sum_15 = convert_element_type = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1423, code: end_loss = loss_fct(end_logits, end_positions)
    amax_13: "f32[1, 1]" = torch.ops.aten.amax.default(clone_50, [1], True)
    sub_40: "f32[1, 512]" = torch.ops.aten.sub.Tensor(clone_50, amax_13);  amax_13 = None
    exp_13: "f32[1, 512]" = torch.ops.aten.exp.default(sub_40)
    sum_16: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [1], True);  exp_13 = None
    log_1: "f32[1, 1]" = torch.ops.aten.log.default(sum_16);  sum_16 = None
    sub_41: "f32[1, 512]" = torch.ops.aten.sub.Tensor(sub_40, log_1);  sub_40 = log_1 = None
    ne_3: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max_1, 512)
    scalar_tensor_2: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'))
    where_2: "i64[1]" = torch.ops.aten.where.self(ne_3, clamp_max_1, scalar_tensor_2);  ne_3 = scalar_tensor_2 = None
    unsqueeze_3: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(where_2, 1);  where_2 = None
    gather_1: "f32[1, 1]" = torch.ops.aten.gather.default(sub_41, 1, unsqueeze_3);  sub_41 = unsqueeze_3 = None
    squeeze_3: "f32[1]" = torch.ops.aten.squeeze.dim(gather_1, 1);  gather_1 = None
    neg_1: "f32[1]" = torch.ops.aten.neg.default(squeeze_3);  squeeze_3 = None
    ne_4: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max_1, 512)
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_3: "f32[1]" = torch.ops.aten.where.self(ne_4, neg_1, scalar_tensor_3);  ne_4 = neg_1 = scalar_tensor_3 = None
    ne_5: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max_1, 512);  clamp_max_1 = None
    sum_17: "i64[]" = torch.ops.aten.sum.default(ne_5);  ne_5 = None
    convert_element_type_1: "f32[]" = torch.ops.prims.convert_element_type.default(sum_17, torch.float32);  sum_17 = None
    sum_18: "f32[]" = torch.ops.aten.sum.default(where_3);  where_3 = None
    div_25: "f32[]" = torch.ops.aten.div.Tensor(sum_18, convert_element_type_1);  sum_18 = convert_element_type_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1424, code: total_loss = (start_loss + end_loss) / 2
    add_100: "f32[]" = torch.ops.aten.add.Tensor(div_24, div_25);  div_24 = div_25 = None
    div_26: "f32[]" = torch.ops.aten.div.Tensor(add_100, 2);  add_100 = None
    return (div_26, clone_49, clone_50)
    