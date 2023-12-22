from __future__ import annotations



def forward(self, arg0_1: "f32[384, 1]", arg1_1: "f32[384, 1]", arg2_1: "f32[384, 1]", arg3_1: "f32[384, 1]", arg4_1: "f32[384, 1]", arg5_1: "f32[384, 1]", arg6_1: "f32[384, 1]", arg7_1: "f32[384, 1]", arg8_1: "f32[384, 1]", arg9_1: "f32[384, 1]", arg10_1: "f32[384, 1]", arg11_1: "f32[384, 1]", arg12_1: "f32[30522, 768]", arg13_1: "f32[512, 768]", arg14_1: "f32[2, 768]", arg15_1: "f32[768]", arg16_1: "f32[768]", arg17_1: "f32[384, 768]", arg18_1: "f32[384]", arg19_1: "f32[384, 768]", arg20_1: "f32[384]", arg21_1: "f32[384, 768]", arg22_1: "f32[384]", arg23_1: "f32[768, 1, 9]", arg24_1: "f32[384, 768, 1]", arg25_1: "f32[54, 384]", arg26_1: "f32[54]", arg27_1: "f32[384, 768]", arg28_1: "f32[384]", arg29_1: "f32[768, 768]", arg30_1: "f32[768]", arg31_1: "f32[768]", arg32_1: "f32[768]", arg33_1: "f32[3072, 768]", arg34_1: "f32[3072]", arg35_1: "f32[768, 3072]", arg36_1: "f32[768]", arg37_1: "f32[768]", arg38_1: "f32[768]", arg39_1: "f32[384, 768]", arg40_1: "f32[384]", arg41_1: "f32[384, 768]", arg42_1: "f32[384]", arg43_1: "f32[384, 768]", arg44_1: "f32[384]", arg45_1: "f32[768, 1, 9]", arg46_1: "f32[384, 768, 1]", arg47_1: "f32[54, 384]", arg48_1: "f32[54]", arg49_1: "f32[384, 768]", arg50_1: "f32[384]", arg51_1: "f32[768, 768]", arg52_1: "f32[768]", arg53_1: "f32[768]", arg54_1: "f32[768]", arg55_1: "f32[3072, 768]", arg56_1: "f32[3072]", arg57_1: "f32[768, 3072]", arg58_1: "f32[768]", arg59_1: "f32[768]", arg60_1: "f32[768]", arg61_1: "f32[384, 768]", arg62_1: "f32[384]", arg63_1: "f32[384, 768]", arg64_1: "f32[384]", arg65_1: "f32[384, 768]", arg66_1: "f32[384]", arg67_1: "f32[768, 1, 9]", arg68_1: "f32[384, 768, 1]", arg69_1: "f32[54, 384]", arg70_1: "f32[54]", arg71_1: "f32[384, 768]", arg72_1: "f32[384]", arg73_1: "f32[768, 768]", arg74_1: "f32[768]", arg75_1: "f32[768]", arg76_1: "f32[768]", arg77_1: "f32[3072, 768]", arg78_1: "f32[3072]", arg79_1: "f32[768, 3072]", arg80_1: "f32[768]", arg81_1: "f32[768]", arg82_1: "f32[768]", arg83_1: "f32[384, 768]", arg84_1: "f32[384]", arg85_1: "f32[384, 768]", arg86_1: "f32[384]", arg87_1: "f32[384, 768]", arg88_1: "f32[384]", arg89_1: "f32[768, 1, 9]", arg90_1: "f32[384, 768, 1]", arg91_1: "f32[54, 384]", arg92_1: "f32[54]", arg93_1: "f32[384, 768]", arg94_1: "f32[384]", arg95_1: "f32[768, 768]", arg96_1: "f32[768]", arg97_1: "f32[768]", arg98_1: "f32[768]", arg99_1: "f32[3072, 768]", arg100_1: "f32[3072]", arg101_1: "f32[768, 3072]", arg102_1: "f32[768]", arg103_1: "f32[768]", arg104_1: "f32[768]", arg105_1: "f32[384, 768]", arg106_1: "f32[384]", arg107_1: "f32[384, 768]", arg108_1: "f32[384]", arg109_1: "f32[384, 768]", arg110_1: "f32[384]", arg111_1: "f32[768, 1, 9]", arg112_1: "f32[384, 768, 1]", arg113_1: "f32[54, 384]", arg114_1: "f32[54]", arg115_1: "f32[384, 768]", arg116_1: "f32[384]", arg117_1: "f32[768, 768]", arg118_1: "f32[768]", arg119_1: "f32[768]", arg120_1: "f32[768]", arg121_1: "f32[3072, 768]", arg122_1: "f32[3072]", arg123_1: "f32[768, 3072]", arg124_1: "f32[768]", arg125_1: "f32[768]", arg126_1: "f32[768]", arg127_1: "f32[384, 768]", arg128_1: "f32[384]", arg129_1: "f32[384, 768]", arg130_1: "f32[384]", arg131_1: "f32[384, 768]", arg132_1: "f32[384]", arg133_1: "f32[768, 1, 9]", arg134_1: "f32[384, 768, 1]", arg135_1: "f32[54, 384]", arg136_1: "f32[54]", arg137_1: "f32[384, 768]", arg138_1: "f32[384]", arg139_1: "f32[768, 768]", arg140_1: "f32[768]", arg141_1: "f32[768]", arg142_1: "f32[768]", arg143_1: "f32[3072, 768]", arg144_1: "f32[3072]", arg145_1: "f32[768, 3072]", arg146_1: "f32[768]", arg147_1: "f32[768]", arg148_1: "f32[768]", arg149_1: "f32[384, 768]", arg150_1: "f32[384]", arg151_1: "f32[384, 768]", arg152_1: "f32[384]", arg153_1: "f32[384, 768]", arg154_1: "f32[384]", arg155_1: "f32[768, 1, 9]", arg156_1: "f32[384, 768, 1]", arg157_1: "f32[54, 384]", arg158_1: "f32[54]", arg159_1: "f32[384, 768]", arg160_1: "f32[384]", arg161_1: "f32[768, 768]", arg162_1: "f32[768]", arg163_1: "f32[768]", arg164_1: "f32[768]", arg165_1: "f32[3072, 768]", arg166_1: "f32[3072]", arg167_1: "f32[768, 3072]", arg168_1: "f32[768]", arg169_1: "f32[768]", arg170_1: "f32[768]", arg171_1: "f32[384, 768]", arg172_1: "f32[384]", arg173_1: "f32[384, 768]", arg174_1: "f32[384]", arg175_1: "f32[384, 768]", arg176_1: "f32[384]", arg177_1: "f32[768, 1, 9]", arg178_1: "f32[384, 768, 1]", arg179_1: "f32[54, 384]", arg180_1: "f32[54]", arg181_1: "f32[384, 768]", arg182_1: "f32[384]", arg183_1: "f32[768, 768]", arg184_1: "f32[768]", arg185_1: "f32[768]", arg186_1: "f32[768]", arg187_1: "f32[3072, 768]", arg188_1: "f32[3072]", arg189_1: "f32[768, 3072]", arg190_1: "f32[768]", arg191_1: "f32[768]", arg192_1: "f32[768]", arg193_1: "f32[384, 768]", arg194_1: "f32[384]", arg195_1: "f32[384, 768]", arg196_1: "f32[384]", arg197_1: "f32[384, 768]", arg198_1: "f32[384]", arg199_1: "f32[768, 1, 9]", arg200_1: "f32[384, 768, 1]", arg201_1: "f32[54, 384]", arg202_1: "f32[54]", arg203_1: "f32[384, 768]", arg204_1: "f32[384]", arg205_1: "f32[768, 768]", arg206_1: "f32[768]", arg207_1: "f32[768]", arg208_1: "f32[768]", arg209_1: "f32[3072, 768]", arg210_1: "f32[3072]", arg211_1: "f32[768, 3072]", arg212_1: "f32[768]", arg213_1: "f32[768]", arg214_1: "f32[768]", arg215_1: "f32[384, 768]", arg216_1: "f32[384]", arg217_1: "f32[384, 768]", arg218_1: "f32[384]", arg219_1: "f32[384, 768]", arg220_1: "f32[384]", arg221_1: "f32[768, 1, 9]", arg222_1: "f32[384, 768, 1]", arg223_1: "f32[54, 384]", arg224_1: "f32[54]", arg225_1: "f32[384, 768]", arg226_1: "f32[384]", arg227_1: "f32[768, 768]", arg228_1: "f32[768]", arg229_1: "f32[768]", arg230_1: "f32[768]", arg231_1: "f32[3072, 768]", arg232_1: "f32[3072]", arg233_1: "f32[768, 3072]", arg234_1: "f32[768]", arg235_1: "f32[768]", arg236_1: "f32[768]", arg237_1: "f32[384, 768]", arg238_1: "f32[384]", arg239_1: "f32[384, 768]", arg240_1: "f32[384]", arg241_1: "f32[384, 768]", arg242_1: "f32[384]", arg243_1: "f32[768, 1, 9]", arg244_1: "f32[384, 768, 1]", arg245_1: "f32[54, 384]", arg246_1: "f32[54]", arg247_1: "f32[384, 768]", arg248_1: "f32[384]", arg249_1: "f32[768, 768]", arg250_1: "f32[768]", arg251_1: "f32[768]", arg252_1: "f32[768]", arg253_1: "f32[3072, 768]", arg254_1: "f32[3072]", arg255_1: "f32[768, 3072]", arg256_1: "f32[768]", arg257_1: "f32[768]", arg258_1: "f32[768]", arg259_1: "f32[384, 768]", arg260_1: "f32[384]", arg261_1: "f32[384, 768]", arg262_1: "f32[384]", arg263_1: "f32[384, 768]", arg264_1: "f32[384]", arg265_1: "f32[768, 1, 9]", arg266_1: "f32[384, 768, 1]", arg267_1: "f32[54, 384]", arg268_1: "f32[54]", arg269_1: "f32[384, 768]", arg270_1: "f32[384]", arg271_1: "f32[768, 768]", arg272_1: "f32[768]", arg273_1: "f32[768]", arg274_1: "f32[768]", arg275_1: "f32[3072, 768]", arg276_1: "f32[3072]", arg277_1: "f32[768, 3072]", arg278_1: "f32[768]", arg279_1: "f32[768]", arg280_1: "f32[768]", arg281_1: "f32[768, 768]", arg282_1: "f32[768]", arg283_1: "f32[768]", arg284_1: "f32[768]", arg285_1: "f32[30522, 768]", arg286_1: "f32[30522]", arg287_1: "i64[1, 512]", arg288_1: "i64[1, 512]", arg289_1: "i64[1, 512]", arg290_1: "i64[1, 512]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:832, code: attention_mask = torch.ones(input_shape, device=device)
    full: "f32[1, 512]" = torch.ops.aten.full.default([1, 512], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:835, code: buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
    slice_1: "i64[1, 512]" = torch.ops.aten.slice.Tensor(arg287_1, 0, 0, 9223372036854775807);  arg287_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:836, code: buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
    expand: "i64[1, 512]" = torch.ops.aten.expand.default(slice_1, [1, 512]);  slice_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:916, code: extended_attention_mask = attention_mask[:, None, None, :]
    slice_2: "f32[1, 512]" = torch.ops.aten.slice.Tensor(full, 0, 0, 9223372036854775807);  full = None
    unsqueeze: "f32[1, 1, 512]" = torch.ops.aten.unsqueeze.default(slice_2, 1);  slice_2 = None
    unsqueeze_1: "f32[1, 1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
    slice_3: "f32[1, 1, 1, 512]" = torch.ops.aten.slice.Tensor(unsqueeze_1, 3, 0, 9223372036854775807);  unsqueeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:928, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    sub: "f32[1, 1, 1, 512]" = torch.ops.aten.sub.Tensor(1.0, slice_3);  slice_3 = None
    mul: "f32[1, 1, 1, 512]" = torch.ops.aten.mul.Tensor(sub, -3.4028234663852886e+38);  sub = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:216, code: position_ids = self.position_ids[:, :seq_length]
    slice_4: "i64[1, 512]" = torch.ops.aten.slice.Tensor(arg288_1, 0, 0, 9223372036854775807);  arg288_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:230, code: inputs_embeds = self.word_embeddings(input_ids)
    embedding: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg12_1, arg289_1, 0);  arg12_1 = arg289_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:231, code: position_embeddings = self.position_embeddings(position_ids)
    embedding_1: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg13_1, slice_4);  arg13_1 = slice_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:232, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    embedding_2: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg14_1, expand);  arg14_1 = expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:234, code: embeddings = inputs_embeds + position_embeddings + token_type_embeddings
    add: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
    add_1: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:235, code: embeddings = self.LayerNorm(embeddings)
    var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 512, 1]" = var_mean[0]
    getitem_1: "f32[1, 512, 1]" = var_mean[1];  var_mean = None
    add_2: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
    rsqrt: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
    sub_1: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
    mul_1: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = rsqrt = None
    mul_2: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_1, arg15_1);  mul_1 = arg15_1 = None
    add_3: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_2, arg16_1);  mul_2 = arg16_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:236, code: embeddings = self.dropout(embeddings)
    clone: "f32[1, 512, 768]" = torch.ops.aten.clone.default(add_3);  add_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view: "f32[512, 768]" = torch.ops.aten.view.default(clone, [512, 768])
    permute: "f32[768, 384]" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
    addmm: "f32[512, 384]" = torch.ops.aten.addmm.default(arg18_1, view, permute);  arg18_1 = view = permute = None
    view_1: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm, [1, 512, 384]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_2: "f32[512, 768]" = torch.ops.aten.view.default(clone, [512, 768])
    permute_1: "f32[768, 384]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
    addmm_1: "f32[512, 384]" = torch.ops.aten.addmm.default(arg20_1, view_2, permute_1);  arg20_1 = view_2 = permute_1 = None
    view_3: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_1, [1, 512, 384]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_4: "f32[512, 768]" = torch.ops.aten.view.default(clone, [512, 768])
    permute_2: "f32[768, 384]" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
    addmm_2: "f32[512, 384]" = torch.ops.aten.addmm.default(arg22_1, view_4, permute_2);  arg22_1 = view_4 = permute_2 = None
    view_5: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_2, [1, 512, 384]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_3: "f32[1, 768, 512]" = torch.ops.aten.permute.default(clone, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution: "f32[1, 768, 512]" = torch.ops.aten.convolution.default(permute_3, arg23_1, None, [1], [4], [1], False, [0], 768);  permute_3 = arg23_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_1: "f32[1, 384, 512]" = torch.ops.aten.convolution.default(convolution, arg24_1, None, [1], [0], [1], False, [0], 1);  convolution = arg24_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_4: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_1, arg0_1);  convolution_1 = arg0_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_6: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_1, [1, 512, 6, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_5: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_7: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_3, [1, 512, 6, 64]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_6: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_8: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_5, [1, 512, 6, 64]);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_7: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_8: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_4, [0, 2, 1]);  add_4 = None
    mul_3: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(permute_8, view_1);  permute_8 = view_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    permute_9: "f32[384, 54]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
    view_9: "f32[512, 384]" = torch.ops.aten.view.default(mul_3, [512, 384]);  mul_3 = None
    mm: "f32[512, 54]" = torch.ops.aten.mm.default(view_9, permute_9);  view_9 = permute_9 = None
    view_10: "f32[1, 512, 54]" = torch.ops.aten.view.default(mm, [1, 512, 54]);  mm = None
    add_5: "f32[1, 512, 54]" = torch.ops.aten.add.Tensor(view_10, arg26_1);  view_10 = arg26_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_11: "f32[3072, 9, 1]" = torch.ops.aten.view.default(add_5, [-1, 9, 1]);  add_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    amax: "f32[3072, 1, 1]" = torch.ops.aten.amax.default(view_11, [1], True)
    sub_2: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(view_11, amax);  view_11 = amax = None
    exp: "f32[3072, 9, 1]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
    sum_1: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True)
    div: "f32[3072, 9, 1]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_12: "f32[512, 768]" = torch.ops.aten.view.default(clone, [512, 768])
    permute_10: "f32[768, 384]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
    addmm_3: "f32[512, 384]" = torch.ops.aten.addmm.default(arg28_1, view_12, permute_10);  arg28_1 = view_12 = permute_10 = None
    view_13: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_3, [1, 512, 384]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_14: "f32[1, 512, 384]" = torch.ops.aten.view.default(view_13, [1, -1, 384]);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    permute_11: "f32[1, 384, 512]" = torch.ops.aten.permute.default(view_14, [0, 2, 1]);  view_14 = None
    clone_1: "f32[1, 384, 512]" = torch.ops.aten.clone.default(permute_11, memory_format = torch.contiguous_format);  permute_11 = None
    unsqueeze_2: "f32[1, 384, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_1, -1);  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    iota: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_3: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
    iota_1: "i64[9]" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_4: "i64[9, 1]" = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
    add_6: "i64[9, 512]" = torch.ops.aten.add.Tensor(unsqueeze_3, unsqueeze_4);  unsqueeze_3 = unsqueeze_4 = None
    iota_2: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_5: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_2, 0);  iota_2 = None
    iota_3: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_6: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_3, -1);  iota_3 = None
    add_7: "i64[1, 1]" = torch.ops.aten.add.Tensor(unsqueeze_5, unsqueeze_6);  unsqueeze_5 = unsqueeze_6 = None
    constant_pad_nd: "f32[1, 384, 520, 1]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_2, [0, 0, 4, 4], 0.0);  unsqueeze_2 = None
    unsqueeze_7: "i64[9, 512, 1]" = torch.ops.aten.unsqueeze.default(add_6, -1);  add_6 = None
    unsqueeze_8: "i64[9, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, -1);  unsqueeze_7 = None
    slice_5: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(constant_pad_nd, 0, 0, 9223372036854775807);  constant_pad_nd = None
    slice_6: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(slice_5, 1, 0, 9223372036854775807);  slice_5 = None
    index: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.index.Tensor(slice_6, [None, None, unsqueeze_8, add_7]);  slice_6 = unsqueeze_8 = add_7 = None
    permute_12: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.permute.default(index, [0, 1, 2, 4, 3, 5]);  index = None
    view_15: "f32[1, 3456, 512]" = torch.ops.aten.view.default(permute_12, [1, 3456, 512]);  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    permute_13: "f32[1, 512, 3456]" = torch.ops.aten.permute.default(view_15, [0, 2, 1]);  view_15 = None
    view_16: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(permute_13, [1, 512, 384, 9]);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    clone_2: "f32[1, 512, 384, 9]" = torch.ops.aten.clone.default(view_16, memory_format = torch.contiguous_format);  view_16 = None
    view_17: "f32[3072, 64, 9]" = torch.ops.aten.view.default(clone_2, [3072, 64, 9]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    expand_1: "f32[3072, 64, 9]" = torch.ops.aten.expand.default(view_17, [3072, 64, 9]);  view_17 = None
    view_18: "f32[3072, 64, 9]" = torch.ops.aten.view.default(expand_1, [3072, 64, 9]);  expand_1 = None
    expand_2: "f32[3072, 9, 1]" = torch.ops.aten.expand.default(div, [3072, 9, 1]);  div = None
    view_19: "f32[3072, 9, 1]" = torch.ops.aten.view.default(expand_2, [3072, 9, 1]);  expand_2 = None
    bmm: "f32[3072, 64, 1]" = torch.ops.aten.bmm.default(view_18, view_19);  view_18 = view_19 = None
    view_20: "f32[3072, 64, 1]" = torch.ops.aten.view.default(bmm, [3072, 64, 1]);  bmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    view_21: "f32[512, 384]" = torch.ops.aten.view.default(view_20, [-1, 384]);  view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_14: "f32[1, 6, 64, 512]" = torch.ops.aten.permute.default(permute_6, [0, 1, 3, 2]);  permute_6 = None
    expand_3: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_5, [1, 6, 512, 64]);  permute_5 = None
    view_22: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_3, [6, 512, 64]);  expand_3 = None
    expand_4: "f32[1, 6, 64, 512]" = torch.ops.aten.expand.default(permute_14, [1, 6, 64, 512]);  permute_14 = None
    view_23: "f32[6, 64, 512]" = torch.ops.aten.view.default(expand_4, [6, 64, 512]);  expand_4 = None
    bmm_1: "f32[6, 512, 512]" = torch.ops.aten.bmm.default(view_22, view_23);  view_22 = view_23 = None
    view_24: "f32[1, 6, 512, 512]" = torch.ops.aten.view.default(bmm_1, [1, 6, 512, 512]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_1: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(view_24, 8.0);  view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:397, code: attention_scores = attention_scores + attention_mask
    add_8: "f32[1, 6, 512, 512]" = torch.ops.aten.add.Tensor(div_1, mul);  div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_1: "f32[1, 6, 512, 1]" = torch.ops.aten.amax.default(add_8, [-1], True)
    sub_3: "f32[1, 6, 512, 512]" = torch.ops.aten.sub.Tensor(add_8, amax_1);  add_8 = amax_1 = None
    exp_1: "f32[1, 6, 512, 512]" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
    sum_2: "f32[1, 6, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_2: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    clone_3: "f32[1, 6, 512, 512]" = torch.ops.aten.clone.default(div_2);  div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_5: "f32[1, 6, 512, 512]" = torch.ops.aten.expand.default(clone_3, [1, 6, 512, 512]);  clone_3 = None
    view_25: "f32[6, 512, 512]" = torch.ops.aten.view.default(expand_5, [6, 512, 512]);  expand_5 = None
    expand_6: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_7, [1, 6, 512, 64]);  permute_7 = None
    view_26: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_6, [6, 512, 64]);  expand_6 = None
    bmm_2: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(view_25, view_26);  view_25 = view_26 = None
    view_27: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_2, [1, 6, 512, 64]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_15: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None
    clone_4: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_28: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_21, [1, -1, 6, 64]);  view_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    cat: "f32[1, 512, 12, 64]" = torch.ops.aten.cat.default([clone_4, view_28], 2);  clone_4 = view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_29: "f32[1, 512, 768]" = torch.ops.aten.view.default(cat, [1, 512, 768]);  cat = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_30: "f32[512, 768]" = torch.ops.aten.view.default(view_29, [512, 768]);  view_29 = None
    permute_16: "f32[768, 768]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
    addmm_4: "f32[512, 768]" = torch.ops.aten.addmm.default(arg30_1, view_30, permute_16);  arg30_1 = view_30 = permute_16 = None
    view_31: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_4, [1, 512, 768]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    clone_5: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_31);  view_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_9: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_5, clone);  clone_5 = clone = None
    var_mean_1 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
    getitem_2: "f32[1, 512, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 512, 1]" = var_mean_1[1];  var_mean_1 = None
    add_10: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
    rsqrt_1: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
    sub_4: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_9, getitem_3);  add_9 = getitem_3 = None
    mul_4: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_1);  sub_4 = rsqrt_1 = None
    mul_5: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_4, arg31_1);  mul_4 = arg31_1 = None
    add_11: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_5, arg32_1);  mul_5 = arg32_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_32: "f32[512, 768]" = torch.ops.aten.view.default(add_11, [512, 768])
    permute_17: "f32[768, 3072]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
    addmm_5: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg34_1, view_32, permute_17);  arg34_1 = view_32 = permute_17 = None
    view_33: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_5, [1, 512, 3072]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_6: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_33, 0.5)
    mul_7: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_33, 0.7071067811865476);  view_33 = None
    erf: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_7);  mul_7 = None
    add_12: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_8: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_6, add_12);  mul_6 = add_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_34: "f32[512, 3072]" = torch.ops.aten.view.default(mul_8, [512, 3072]);  mul_8 = None
    permute_18: "f32[3072, 768]" = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
    addmm_6: "f32[512, 768]" = torch.ops.aten.addmm.default(arg36_1, view_34, permute_18);  arg36_1 = view_34 = permute_18 = None
    view_35: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_6, [1, 512, 768]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    clone_6: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_35);  view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_13: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_6, add_11);  clone_6 = add_11 = None
    var_mean_2 = torch.ops.aten.var_mean.correction(add_13, [2], correction = 0, keepdim = True)
    getitem_4: "f32[1, 512, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 512, 1]" = var_mean_2[1];  var_mean_2 = None
    add_14: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
    rsqrt_2: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
    sub_5: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_13, getitem_5);  add_13 = getitem_5 = None
    mul_9: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_2);  sub_5 = rsqrt_2 = None
    mul_10: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_9, arg37_1);  mul_9 = arg37_1 = None
    add_15: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_10, arg38_1);  mul_10 = arg38_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_36: "f32[512, 768]" = torch.ops.aten.view.default(add_15, [512, 768])
    permute_19: "f32[768, 384]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
    addmm_7: "f32[512, 384]" = torch.ops.aten.addmm.default(arg40_1, view_36, permute_19);  arg40_1 = view_36 = permute_19 = None
    view_37: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_7, [1, 512, 384]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_38: "f32[512, 768]" = torch.ops.aten.view.default(add_15, [512, 768])
    permute_20: "f32[768, 384]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
    addmm_8: "f32[512, 384]" = torch.ops.aten.addmm.default(arg42_1, view_38, permute_20);  arg42_1 = view_38 = permute_20 = None
    view_39: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_8, [1, 512, 384]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_40: "f32[512, 768]" = torch.ops.aten.view.default(add_15, [512, 768])
    permute_21: "f32[768, 384]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
    addmm_9: "f32[512, 384]" = torch.ops.aten.addmm.default(arg44_1, view_40, permute_21);  arg44_1 = view_40 = permute_21 = None
    view_41: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_9, [1, 512, 384]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_22: "f32[1, 768, 512]" = torch.ops.aten.permute.default(add_15, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_2: "f32[1, 768, 512]" = torch.ops.aten.convolution.default(permute_22, arg45_1, None, [1], [4], [1], False, [0], 768);  permute_22 = arg45_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_3: "f32[1, 384, 512]" = torch.ops.aten.convolution.default(convolution_2, arg46_1, None, [1], [0], [1], False, [0], 1);  convolution_2 = arg46_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_16: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_3, arg1_1);  convolution_3 = arg1_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_42: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_37, [1, 512, 6, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_24: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_43: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_39, [1, 512, 6, 64]);  view_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_25: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_43, [0, 2, 1, 3]);  view_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_44: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_41, [1, 512, 6, 64]);  view_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_26: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_44, [0, 2, 1, 3]);  view_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_27: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_16, [0, 2, 1]);  add_16 = None
    mul_11: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(permute_27, view_37);  permute_27 = view_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    permute_28: "f32[384, 54]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
    view_45: "f32[512, 384]" = torch.ops.aten.view.default(mul_11, [512, 384]);  mul_11 = None
    mm_1: "f32[512, 54]" = torch.ops.aten.mm.default(view_45, permute_28);  view_45 = permute_28 = None
    view_46: "f32[1, 512, 54]" = torch.ops.aten.view.default(mm_1, [1, 512, 54]);  mm_1 = None
    add_17: "f32[1, 512, 54]" = torch.ops.aten.add.Tensor(view_46, arg48_1);  view_46 = arg48_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_47: "f32[3072, 9, 1]" = torch.ops.aten.view.default(add_17, [-1, 9, 1]);  add_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    amax_2: "f32[3072, 1, 1]" = torch.ops.aten.amax.default(view_47, [1], True)
    sub_6: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(view_47, amax_2);  view_47 = amax_2 = None
    exp_2: "f32[3072, 9, 1]" = torch.ops.aten.exp.default(sub_6);  sub_6 = None
    sum_3: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [1], True)
    div_3: "f32[3072, 9, 1]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_48: "f32[512, 768]" = torch.ops.aten.view.default(add_15, [512, 768])
    permute_29: "f32[768, 384]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
    addmm_10: "f32[512, 384]" = torch.ops.aten.addmm.default(arg50_1, view_48, permute_29);  arg50_1 = view_48 = permute_29 = None
    view_49: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_10, [1, 512, 384]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_50: "f32[1, 512, 384]" = torch.ops.aten.view.default(view_49, [1, -1, 384]);  view_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    permute_30: "f32[1, 384, 512]" = torch.ops.aten.permute.default(view_50, [0, 2, 1]);  view_50 = None
    clone_7: "f32[1, 384, 512]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
    unsqueeze_9: "f32[1, 384, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_7, -1);  clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    iota_4: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_10: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota_4, 0);  iota_4 = None
    iota_5: "i64[9]" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_11: "i64[9, 1]" = torch.ops.aten.unsqueeze.default(iota_5, -1);  iota_5 = None
    add_18: "i64[9, 512]" = torch.ops.aten.add.Tensor(unsqueeze_10, unsqueeze_11);  unsqueeze_10 = unsqueeze_11 = None
    iota_6: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_12: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_6, 0);  iota_6 = None
    iota_7: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_13: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_7, -1);  iota_7 = None
    add_19: "i64[1, 1]" = torch.ops.aten.add.Tensor(unsqueeze_12, unsqueeze_13);  unsqueeze_12 = unsqueeze_13 = None
    constant_pad_nd_1: "f32[1, 384, 520, 1]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_9, [0, 0, 4, 4], 0.0);  unsqueeze_9 = None
    unsqueeze_14: "i64[9, 512, 1]" = torch.ops.aten.unsqueeze.default(add_18, -1);  add_18 = None
    unsqueeze_15: "i64[9, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    slice_7: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(constant_pad_nd_1, 0, 0, 9223372036854775807);  constant_pad_nd_1 = None
    slice_8: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(slice_7, 1, 0, 9223372036854775807);  slice_7 = None
    index_1: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.index.Tensor(slice_8, [None, None, unsqueeze_15, add_19]);  slice_8 = unsqueeze_15 = add_19 = None
    permute_31: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.permute.default(index_1, [0, 1, 2, 4, 3, 5]);  index_1 = None
    view_51: "f32[1, 3456, 512]" = torch.ops.aten.view.default(permute_31, [1, 3456, 512]);  permute_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    permute_32: "f32[1, 512, 3456]" = torch.ops.aten.permute.default(view_51, [0, 2, 1]);  view_51 = None
    view_52: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(permute_32, [1, 512, 384, 9]);  permute_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    clone_8: "f32[1, 512, 384, 9]" = torch.ops.aten.clone.default(view_52, memory_format = torch.contiguous_format);  view_52 = None
    view_53: "f32[3072, 64, 9]" = torch.ops.aten.view.default(clone_8, [3072, 64, 9]);  clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    expand_7: "f32[3072, 64, 9]" = torch.ops.aten.expand.default(view_53, [3072, 64, 9]);  view_53 = None
    view_54: "f32[3072, 64, 9]" = torch.ops.aten.view.default(expand_7, [3072, 64, 9]);  expand_7 = None
    expand_8: "f32[3072, 9, 1]" = torch.ops.aten.expand.default(div_3, [3072, 9, 1]);  div_3 = None
    view_55: "f32[3072, 9, 1]" = torch.ops.aten.view.default(expand_8, [3072, 9, 1]);  expand_8 = None
    bmm_3: "f32[3072, 64, 1]" = torch.ops.aten.bmm.default(view_54, view_55);  view_54 = view_55 = None
    view_56: "f32[3072, 64, 1]" = torch.ops.aten.view.default(bmm_3, [3072, 64, 1]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    view_57: "f32[512, 384]" = torch.ops.aten.view.default(view_56, [-1, 384]);  view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_33: "f32[1, 6, 64, 512]" = torch.ops.aten.permute.default(permute_25, [0, 1, 3, 2]);  permute_25 = None
    expand_9: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_24, [1, 6, 512, 64]);  permute_24 = None
    view_58: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_9, [6, 512, 64]);  expand_9 = None
    expand_10: "f32[1, 6, 64, 512]" = torch.ops.aten.expand.default(permute_33, [1, 6, 64, 512]);  permute_33 = None
    view_59: "f32[6, 64, 512]" = torch.ops.aten.view.default(expand_10, [6, 64, 512]);  expand_10 = None
    bmm_4: "f32[6, 512, 512]" = torch.ops.aten.bmm.default(view_58, view_59);  view_58 = view_59 = None
    view_60: "f32[1, 6, 512, 512]" = torch.ops.aten.view.default(bmm_4, [1, 6, 512, 512]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_4: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(view_60, 8.0);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:397, code: attention_scores = attention_scores + attention_mask
    add_20: "f32[1, 6, 512, 512]" = torch.ops.aten.add.Tensor(div_4, mul);  div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_3: "f32[1, 6, 512, 1]" = torch.ops.aten.amax.default(add_20, [-1], True)
    sub_7: "f32[1, 6, 512, 512]" = torch.ops.aten.sub.Tensor(add_20, amax_3);  add_20 = amax_3 = None
    exp_3: "f32[1, 6, 512, 512]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
    sum_4: "f32[1, 6, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_5: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    clone_9: "f32[1, 6, 512, 512]" = torch.ops.aten.clone.default(div_5);  div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_11: "f32[1, 6, 512, 512]" = torch.ops.aten.expand.default(clone_9, [1, 6, 512, 512]);  clone_9 = None
    view_61: "f32[6, 512, 512]" = torch.ops.aten.view.default(expand_11, [6, 512, 512]);  expand_11 = None
    expand_12: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_26, [1, 6, 512, 64]);  permute_26 = None
    view_62: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_12, [6, 512, 64]);  expand_12 = None
    bmm_5: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(view_61, view_62);  view_61 = view_62 = None
    view_63: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_5, [1, 6, 512, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_34: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_63, [0, 2, 1, 3]);  view_63 = None
    clone_10: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format);  permute_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_64: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_57, [1, -1, 6, 64]);  view_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    cat_1: "f32[1, 512, 12, 64]" = torch.ops.aten.cat.default([clone_10, view_64], 2);  clone_10 = view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_65: "f32[1, 512, 768]" = torch.ops.aten.view.default(cat_1, [1, 512, 768]);  cat_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_66: "f32[512, 768]" = torch.ops.aten.view.default(view_65, [512, 768]);  view_65 = None
    permute_35: "f32[768, 768]" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
    addmm_11: "f32[512, 768]" = torch.ops.aten.addmm.default(arg52_1, view_66, permute_35);  arg52_1 = view_66 = permute_35 = None
    view_67: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_11, [1, 512, 768]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    clone_11: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_67);  view_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_21: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_11, add_15);  clone_11 = add_15 = None
    var_mean_3 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
    getitem_6: "f32[1, 512, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 512, 1]" = var_mean_3[1];  var_mean_3 = None
    add_22: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
    rsqrt_3: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
    sub_8: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_21, getitem_7);  add_21 = getitem_7 = None
    mul_12: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_3);  sub_8 = rsqrt_3 = None
    mul_13: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_12, arg53_1);  mul_12 = arg53_1 = None
    add_23: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_13, arg54_1);  mul_13 = arg54_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_68: "f32[512, 768]" = torch.ops.aten.view.default(add_23, [512, 768])
    permute_36: "f32[768, 3072]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
    addmm_12: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg56_1, view_68, permute_36);  arg56_1 = view_68 = permute_36 = None
    view_69: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_12, [1, 512, 3072]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_14: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_69, 0.5)
    mul_15: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_69, 0.7071067811865476);  view_69 = None
    erf_1: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_15);  mul_15 = None
    add_24: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_16: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_14, add_24);  mul_14 = add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_70: "f32[512, 3072]" = torch.ops.aten.view.default(mul_16, [512, 3072]);  mul_16 = None
    permute_37: "f32[3072, 768]" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
    addmm_13: "f32[512, 768]" = torch.ops.aten.addmm.default(arg58_1, view_70, permute_37);  arg58_1 = view_70 = permute_37 = None
    view_71: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_13, [1, 512, 768]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    clone_12: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_71);  view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_25: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_12, add_23);  clone_12 = add_23 = None
    var_mean_4 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
    getitem_8: "f32[1, 512, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 512, 1]" = var_mean_4[1];  var_mean_4 = None
    add_26: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
    rsqrt_4: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
    sub_9: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_25, getitem_9);  add_25 = getitem_9 = None
    mul_17: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_4);  sub_9 = rsqrt_4 = None
    mul_18: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_17, arg59_1);  mul_17 = arg59_1 = None
    add_27: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_18, arg60_1);  mul_18 = arg60_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_72: "f32[512, 768]" = torch.ops.aten.view.default(add_27, [512, 768])
    permute_38: "f32[768, 384]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
    addmm_14: "f32[512, 384]" = torch.ops.aten.addmm.default(arg62_1, view_72, permute_38);  arg62_1 = view_72 = permute_38 = None
    view_73: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_14, [1, 512, 384]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_74: "f32[512, 768]" = torch.ops.aten.view.default(add_27, [512, 768])
    permute_39: "f32[768, 384]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
    addmm_15: "f32[512, 384]" = torch.ops.aten.addmm.default(arg64_1, view_74, permute_39);  arg64_1 = view_74 = permute_39 = None
    view_75: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_15, [1, 512, 384]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_76: "f32[512, 768]" = torch.ops.aten.view.default(add_27, [512, 768])
    permute_40: "f32[768, 384]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
    addmm_16: "f32[512, 384]" = torch.ops.aten.addmm.default(arg66_1, view_76, permute_40);  arg66_1 = view_76 = permute_40 = None
    view_77: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_16, [1, 512, 384]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_41: "f32[1, 768, 512]" = torch.ops.aten.permute.default(add_27, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_4: "f32[1, 768, 512]" = torch.ops.aten.convolution.default(permute_41, arg67_1, None, [1], [4], [1], False, [0], 768);  permute_41 = arg67_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_5: "f32[1, 384, 512]" = torch.ops.aten.convolution.default(convolution_4, arg68_1, None, [1], [0], [1], False, [0], 1);  convolution_4 = arg68_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_28: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_5, arg2_1);  convolution_5 = arg2_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_78: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_73, [1, 512, 6, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_43: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_79: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_75, [1, 512, 6, 64]);  view_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_44: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_79, [0, 2, 1, 3]);  view_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_80: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_77, [1, 512, 6, 64]);  view_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_45: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_80, [0, 2, 1, 3]);  view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_46: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_28, [0, 2, 1]);  add_28 = None
    mul_19: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(permute_46, view_73);  permute_46 = view_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    permute_47: "f32[384, 54]" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
    view_81: "f32[512, 384]" = torch.ops.aten.view.default(mul_19, [512, 384]);  mul_19 = None
    mm_2: "f32[512, 54]" = torch.ops.aten.mm.default(view_81, permute_47);  view_81 = permute_47 = None
    view_82: "f32[1, 512, 54]" = torch.ops.aten.view.default(mm_2, [1, 512, 54]);  mm_2 = None
    add_29: "f32[1, 512, 54]" = torch.ops.aten.add.Tensor(view_82, arg70_1);  view_82 = arg70_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_83: "f32[3072, 9, 1]" = torch.ops.aten.view.default(add_29, [-1, 9, 1]);  add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    amax_4: "f32[3072, 1, 1]" = torch.ops.aten.amax.default(view_83, [1], True)
    sub_10: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(view_83, amax_4);  view_83 = amax_4 = None
    exp_4: "f32[3072, 9, 1]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
    sum_5: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [1], True)
    div_6: "f32[3072, 9, 1]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_84: "f32[512, 768]" = torch.ops.aten.view.default(add_27, [512, 768])
    permute_48: "f32[768, 384]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
    addmm_17: "f32[512, 384]" = torch.ops.aten.addmm.default(arg72_1, view_84, permute_48);  arg72_1 = view_84 = permute_48 = None
    view_85: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_17, [1, 512, 384]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_86: "f32[1, 512, 384]" = torch.ops.aten.view.default(view_85, [1, -1, 384]);  view_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    permute_49: "f32[1, 384, 512]" = torch.ops.aten.permute.default(view_86, [0, 2, 1]);  view_86 = None
    clone_13: "f32[1, 384, 512]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
    unsqueeze_16: "f32[1, 384, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_13, -1);  clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    iota_8: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_17: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota_8, 0);  iota_8 = None
    iota_9: "i64[9]" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_18: "i64[9, 1]" = torch.ops.aten.unsqueeze.default(iota_9, -1);  iota_9 = None
    add_30: "i64[9, 512]" = torch.ops.aten.add.Tensor(unsqueeze_17, unsqueeze_18);  unsqueeze_17 = unsqueeze_18 = None
    iota_10: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_19: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_10, 0);  iota_10 = None
    iota_11: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_20: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_11, -1);  iota_11 = None
    add_31: "i64[1, 1]" = torch.ops.aten.add.Tensor(unsqueeze_19, unsqueeze_20);  unsqueeze_19 = unsqueeze_20 = None
    constant_pad_nd_2: "f32[1, 384, 520, 1]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_16, [0, 0, 4, 4], 0.0);  unsqueeze_16 = None
    unsqueeze_21: "i64[9, 512, 1]" = torch.ops.aten.unsqueeze.default(add_30, -1);  add_30 = None
    unsqueeze_22: "i64[9, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_21, -1);  unsqueeze_21 = None
    slice_9: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(constant_pad_nd_2, 0, 0, 9223372036854775807);  constant_pad_nd_2 = None
    slice_10: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(slice_9, 1, 0, 9223372036854775807);  slice_9 = None
    index_2: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.index.Tensor(slice_10, [None, None, unsqueeze_22, add_31]);  slice_10 = unsqueeze_22 = add_31 = None
    permute_50: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.permute.default(index_2, [0, 1, 2, 4, 3, 5]);  index_2 = None
    view_87: "f32[1, 3456, 512]" = torch.ops.aten.view.default(permute_50, [1, 3456, 512]);  permute_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    permute_51: "f32[1, 512, 3456]" = torch.ops.aten.permute.default(view_87, [0, 2, 1]);  view_87 = None
    view_88: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(permute_51, [1, 512, 384, 9]);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    clone_14: "f32[1, 512, 384, 9]" = torch.ops.aten.clone.default(view_88, memory_format = torch.contiguous_format);  view_88 = None
    view_89: "f32[3072, 64, 9]" = torch.ops.aten.view.default(clone_14, [3072, 64, 9]);  clone_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    expand_13: "f32[3072, 64, 9]" = torch.ops.aten.expand.default(view_89, [3072, 64, 9]);  view_89 = None
    view_90: "f32[3072, 64, 9]" = torch.ops.aten.view.default(expand_13, [3072, 64, 9]);  expand_13 = None
    expand_14: "f32[3072, 9, 1]" = torch.ops.aten.expand.default(div_6, [3072, 9, 1]);  div_6 = None
    view_91: "f32[3072, 9, 1]" = torch.ops.aten.view.default(expand_14, [3072, 9, 1]);  expand_14 = None
    bmm_6: "f32[3072, 64, 1]" = torch.ops.aten.bmm.default(view_90, view_91);  view_90 = view_91 = None
    view_92: "f32[3072, 64, 1]" = torch.ops.aten.view.default(bmm_6, [3072, 64, 1]);  bmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    view_93: "f32[512, 384]" = torch.ops.aten.view.default(view_92, [-1, 384]);  view_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_52: "f32[1, 6, 64, 512]" = torch.ops.aten.permute.default(permute_44, [0, 1, 3, 2]);  permute_44 = None
    expand_15: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_43, [1, 6, 512, 64]);  permute_43 = None
    view_94: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_15, [6, 512, 64]);  expand_15 = None
    expand_16: "f32[1, 6, 64, 512]" = torch.ops.aten.expand.default(permute_52, [1, 6, 64, 512]);  permute_52 = None
    view_95: "f32[6, 64, 512]" = torch.ops.aten.view.default(expand_16, [6, 64, 512]);  expand_16 = None
    bmm_7: "f32[6, 512, 512]" = torch.ops.aten.bmm.default(view_94, view_95);  view_94 = view_95 = None
    view_96: "f32[1, 6, 512, 512]" = torch.ops.aten.view.default(bmm_7, [1, 6, 512, 512]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_7: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(view_96, 8.0);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:397, code: attention_scores = attention_scores + attention_mask
    add_32: "f32[1, 6, 512, 512]" = torch.ops.aten.add.Tensor(div_7, mul);  div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_5: "f32[1, 6, 512, 1]" = torch.ops.aten.amax.default(add_32, [-1], True)
    sub_11: "f32[1, 6, 512, 512]" = torch.ops.aten.sub.Tensor(add_32, amax_5);  add_32 = amax_5 = None
    exp_5: "f32[1, 6, 512, 512]" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
    sum_6: "f32[1, 6, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_8: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    clone_15: "f32[1, 6, 512, 512]" = torch.ops.aten.clone.default(div_8);  div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_17: "f32[1, 6, 512, 512]" = torch.ops.aten.expand.default(clone_15, [1, 6, 512, 512]);  clone_15 = None
    view_97: "f32[6, 512, 512]" = torch.ops.aten.view.default(expand_17, [6, 512, 512]);  expand_17 = None
    expand_18: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_45, [1, 6, 512, 64]);  permute_45 = None
    view_98: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_18, [6, 512, 64]);  expand_18 = None
    bmm_8: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(view_97, view_98);  view_97 = view_98 = None
    view_99: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_8, [1, 6, 512, 64]);  bmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_53: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_99, [0, 2, 1, 3]);  view_99 = None
    clone_16: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_53, memory_format = torch.contiguous_format);  permute_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_100: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_93, [1, -1, 6, 64]);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    cat_2: "f32[1, 512, 12, 64]" = torch.ops.aten.cat.default([clone_16, view_100], 2);  clone_16 = view_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_101: "f32[1, 512, 768]" = torch.ops.aten.view.default(cat_2, [1, 512, 768]);  cat_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_102: "f32[512, 768]" = torch.ops.aten.view.default(view_101, [512, 768]);  view_101 = None
    permute_54: "f32[768, 768]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
    addmm_18: "f32[512, 768]" = torch.ops.aten.addmm.default(arg74_1, view_102, permute_54);  arg74_1 = view_102 = permute_54 = None
    view_103: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_18, [1, 512, 768]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    clone_17: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_103);  view_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_33: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_17, add_27);  clone_17 = add_27 = None
    var_mean_5 = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
    getitem_10: "f32[1, 512, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 512, 1]" = var_mean_5[1];  var_mean_5 = None
    add_34: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
    rsqrt_5: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
    sub_12: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_33, getitem_11);  add_33 = getitem_11 = None
    mul_20: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_5);  sub_12 = rsqrt_5 = None
    mul_21: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_20, arg75_1);  mul_20 = arg75_1 = None
    add_35: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_21, arg76_1);  mul_21 = arg76_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_104: "f32[512, 768]" = torch.ops.aten.view.default(add_35, [512, 768])
    permute_55: "f32[768, 3072]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
    addmm_19: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg78_1, view_104, permute_55);  arg78_1 = view_104 = permute_55 = None
    view_105: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_19, [1, 512, 3072]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_22: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_105, 0.5)
    mul_23: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_105, 0.7071067811865476);  view_105 = None
    erf_2: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_23);  mul_23 = None
    add_36: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_24: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_22, add_36);  mul_22 = add_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_106: "f32[512, 3072]" = torch.ops.aten.view.default(mul_24, [512, 3072]);  mul_24 = None
    permute_56: "f32[3072, 768]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
    addmm_20: "f32[512, 768]" = torch.ops.aten.addmm.default(arg80_1, view_106, permute_56);  arg80_1 = view_106 = permute_56 = None
    view_107: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_20, [1, 512, 768]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    clone_18: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_107);  view_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_37: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_18, add_35);  clone_18 = add_35 = None
    var_mean_6 = torch.ops.aten.var_mean.correction(add_37, [2], correction = 0, keepdim = True)
    getitem_12: "f32[1, 512, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 512, 1]" = var_mean_6[1];  var_mean_6 = None
    add_38: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
    rsqrt_6: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
    sub_13: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_37, getitem_13);  add_37 = getitem_13 = None
    mul_25: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_6);  sub_13 = rsqrt_6 = None
    mul_26: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_25, arg81_1);  mul_25 = arg81_1 = None
    add_39: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_26, arg82_1);  mul_26 = arg82_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_108: "f32[512, 768]" = torch.ops.aten.view.default(add_39, [512, 768])
    permute_57: "f32[768, 384]" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
    addmm_21: "f32[512, 384]" = torch.ops.aten.addmm.default(arg84_1, view_108, permute_57);  arg84_1 = view_108 = permute_57 = None
    view_109: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_21, [1, 512, 384]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_110: "f32[512, 768]" = torch.ops.aten.view.default(add_39, [512, 768])
    permute_58: "f32[768, 384]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
    addmm_22: "f32[512, 384]" = torch.ops.aten.addmm.default(arg86_1, view_110, permute_58);  arg86_1 = view_110 = permute_58 = None
    view_111: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_22, [1, 512, 384]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_112: "f32[512, 768]" = torch.ops.aten.view.default(add_39, [512, 768])
    permute_59: "f32[768, 384]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
    addmm_23: "f32[512, 384]" = torch.ops.aten.addmm.default(arg88_1, view_112, permute_59);  arg88_1 = view_112 = permute_59 = None
    view_113: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_23, [1, 512, 384]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_60: "f32[1, 768, 512]" = torch.ops.aten.permute.default(add_39, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_6: "f32[1, 768, 512]" = torch.ops.aten.convolution.default(permute_60, arg89_1, None, [1], [4], [1], False, [0], 768);  permute_60 = arg89_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_7: "f32[1, 384, 512]" = torch.ops.aten.convolution.default(convolution_6, arg90_1, None, [1], [0], [1], False, [0], 1);  convolution_6 = arg90_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_40: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_7, arg3_1);  convolution_7 = arg3_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_114: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_109, [1, 512, 6, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_62: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_115: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_111, [1, 512, 6, 64]);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_63: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_115, [0, 2, 1, 3]);  view_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_116: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_113, [1, 512, 6, 64]);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_64: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_116, [0, 2, 1, 3]);  view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_65: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_40, [0, 2, 1]);  add_40 = None
    mul_27: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(permute_65, view_109);  permute_65 = view_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    permute_66: "f32[384, 54]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
    view_117: "f32[512, 384]" = torch.ops.aten.view.default(mul_27, [512, 384]);  mul_27 = None
    mm_3: "f32[512, 54]" = torch.ops.aten.mm.default(view_117, permute_66);  view_117 = permute_66 = None
    view_118: "f32[1, 512, 54]" = torch.ops.aten.view.default(mm_3, [1, 512, 54]);  mm_3 = None
    add_41: "f32[1, 512, 54]" = torch.ops.aten.add.Tensor(view_118, arg92_1);  view_118 = arg92_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_119: "f32[3072, 9, 1]" = torch.ops.aten.view.default(add_41, [-1, 9, 1]);  add_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    amax_6: "f32[3072, 1, 1]" = torch.ops.aten.amax.default(view_119, [1], True)
    sub_14: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(view_119, amax_6);  view_119 = amax_6 = None
    exp_6: "f32[3072, 9, 1]" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
    sum_7: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [1], True)
    div_9: "f32[3072, 9, 1]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_120: "f32[512, 768]" = torch.ops.aten.view.default(add_39, [512, 768])
    permute_67: "f32[768, 384]" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
    addmm_24: "f32[512, 384]" = torch.ops.aten.addmm.default(arg94_1, view_120, permute_67);  arg94_1 = view_120 = permute_67 = None
    view_121: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_24, [1, 512, 384]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_122: "f32[1, 512, 384]" = torch.ops.aten.view.default(view_121, [1, -1, 384]);  view_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    permute_68: "f32[1, 384, 512]" = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None
    clone_19: "f32[1, 384, 512]" = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
    unsqueeze_23: "f32[1, 384, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_19, -1);  clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    iota_12: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_24: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota_12, 0);  iota_12 = None
    iota_13: "i64[9]" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_25: "i64[9, 1]" = torch.ops.aten.unsqueeze.default(iota_13, -1);  iota_13 = None
    add_42: "i64[9, 512]" = torch.ops.aten.add.Tensor(unsqueeze_24, unsqueeze_25);  unsqueeze_24 = unsqueeze_25 = None
    iota_14: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_26: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_14, 0);  iota_14 = None
    iota_15: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_27: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_15, -1);  iota_15 = None
    add_43: "i64[1, 1]" = torch.ops.aten.add.Tensor(unsqueeze_26, unsqueeze_27);  unsqueeze_26 = unsqueeze_27 = None
    constant_pad_nd_3: "f32[1, 384, 520, 1]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_23, [0, 0, 4, 4], 0.0);  unsqueeze_23 = None
    unsqueeze_28: "i64[9, 512, 1]" = torch.ops.aten.unsqueeze.default(add_42, -1);  add_42 = None
    unsqueeze_29: "i64[9, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    slice_11: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(constant_pad_nd_3, 0, 0, 9223372036854775807);  constant_pad_nd_3 = None
    slice_12: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(slice_11, 1, 0, 9223372036854775807);  slice_11 = None
    index_3: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.index.Tensor(slice_12, [None, None, unsqueeze_29, add_43]);  slice_12 = unsqueeze_29 = add_43 = None
    permute_69: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.permute.default(index_3, [0, 1, 2, 4, 3, 5]);  index_3 = None
    view_123: "f32[1, 3456, 512]" = torch.ops.aten.view.default(permute_69, [1, 3456, 512]);  permute_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    permute_70: "f32[1, 512, 3456]" = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None
    view_124: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(permute_70, [1, 512, 384, 9]);  permute_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    clone_20: "f32[1, 512, 384, 9]" = torch.ops.aten.clone.default(view_124, memory_format = torch.contiguous_format);  view_124 = None
    view_125: "f32[3072, 64, 9]" = torch.ops.aten.view.default(clone_20, [3072, 64, 9]);  clone_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    expand_19: "f32[3072, 64, 9]" = torch.ops.aten.expand.default(view_125, [3072, 64, 9]);  view_125 = None
    view_126: "f32[3072, 64, 9]" = torch.ops.aten.view.default(expand_19, [3072, 64, 9]);  expand_19 = None
    expand_20: "f32[3072, 9, 1]" = torch.ops.aten.expand.default(div_9, [3072, 9, 1]);  div_9 = None
    view_127: "f32[3072, 9, 1]" = torch.ops.aten.view.default(expand_20, [3072, 9, 1]);  expand_20 = None
    bmm_9: "f32[3072, 64, 1]" = torch.ops.aten.bmm.default(view_126, view_127);  view_126 = view_127 = None
    view_128: "f32[3072, 64, 1]" = torch.ops.aten.view.default(bmm_9, [3072, 64, 1]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    view_129: "f32[512, 384]" = torch.ops.aten.view.default(view_128, [-1, 384]);  view_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_71: "f32[1, 6, 64, 512]" = torch.ops.aten.permute.default(permute_63, [0, 1, 3, 2]);  permute_63 = None
    expand_21: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_62, [1, 6, 512, 64]);  permute_62 = None
    view_130: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_21, [6, 512, 64]);  expand_21 = None
    expand_22: "f32[1, 6, 64, 512]" = torch.ops.aten.expand.default(permute_71, [1, 6, 64, 512]);  permute_71 = None
    view_131: "f32[6, 64, 512]" = torch.ops.aten.view.default(expand_22, [6, 64, 512]);  expand_22 = None
    bmm_10: "f32[6, 512, 512]" = torch.ops.aten.bmm.default(view_130, view_131);  view_130 = view_131 = None
    view_132: "f32[1, 6, 512, 512]" = torch.ops.aten.view.default(bmm_10, [1, 6, 512, 512]);  bmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_10: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(view_132, 8.0);  view_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:397, code: attention_scores = attention_scores + attention_mask
    add_44: "f32[1, 6, 512, 512]" = torch.ops.aten.add.Tensor(div_10, mul);  div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_7: "f32[1, 6, 512, 1]" = torch.ops.aten.amax.default(add_44, [-1], True)
    sub_15: "f32[1, 6, 512, 512]" = torch.ops.aten.sub.Tensor(add_44, amax_7);  add_44 = amax_7 = None
    exp_7: "f32[1, 6, 512, 512]" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
    sum_8: "f32[1, 6, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_11: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    clone_21: "f32[1, 6, 512, 512]" = torch.ops.aten.clone.default(div_11);  div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_23: "f32[1, 6, 512, 512]" = torch.ops.aten.expand.default(clone_21, [1, 6, 512, 512]);  clone_21 = None
    view_133: "f32[6, 512, 512]" = torch.ops.aten.view.default(expand_23, [6, 512, 512]);  expand_23 = None
    expand_24: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_64, [1, 6, 512, 64]);  permute_64 = None
    view_134: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_24, [6, 512, 64]);  expand_24 = None
    bmm_11: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(view_133, view_134);  view_133 = view_134 = None
    view_135: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_11, [1, 6, 512, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_72: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_135, [0, 2, 1, 3]);  view_135 = None
    clone_22: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_72, memory_format = torch.contiguous_format);  permute_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_136: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_129, [1, -1, 6, 64]);  view_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    cat_3: "f32[1, 512, 12, 64]" = torch.ops.aten.cat.default([clone_22, view_136], 2);  clone_22 = view_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_137: "f32[1, 512, 768]" = torch.ops.aten.view.default(cat_3, [1, 512, 768]);  cat_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_138: "f32[512, 768]" = torch.ops.aten.view.default(view_137, [512, 768]);  view_137 = None
    permute_73: "f32[768, 768]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
    addmm_25: "f32[512, 768]" = torch.ops.aten.addmm.default(arg96_1, view_138, permute_73);  arg96_1 = view_138 = permute_73 = None
    view_139: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_25, [1, 512, 768]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    clone_23: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_139);  view_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_45: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_23, add_39);  clone_23 = add_39 = None
    var_mean_7 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
    getitem_14: "f32[1, 512, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 512, 1]" = var_mean_7[1];  var_mean_7 = None
    add_46: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
    rsqrt_7: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
    sub_16: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_45, getitem_15);  add_45 = getitem_15 = None
    mul_28: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_7);  sub_16 = rsqrt_7 = None
    mul_29: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_28, arg97_1);  mul_28 = arg97_1 = None
    add_47: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_29, arg98_1);  mul_29 = arg98_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_140: "f32[512, 768]" = torch.ops.aten.view.default(add_47, [512, 768])
    permute_74: "f32[768, 3072]" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
    addmm_26: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg100_1, view_140, permute_74);  arg100_1 = view_140 = permute_74 = None
    view_141: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_26, [1, 512, 3072]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_30: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_141, 0.5)
    mul_31: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_141, 0.7071067811865476);  view_141 = None
    erf_3: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_31);  mul_31 = None
    add_48: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_32: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_30, add_48);  mul_30 = add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_142: "f32[512, 3072]" = torch.ops.aten.view.default(mul_32, [512, 3072]);  mul_32 = None
    permute_75: "f32[3072, 768]" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
    addmm_27: "f32[512, 768]" = torch.ops.aten.addmm.default(arg102_1, view_142, permute_75);  arg102_1 = view_142 = permute_75 = None
    view_143: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_27, [1, 512, 768]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    clone_24: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_143);  view_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_49: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_24, add_47);  clone_24 = add_47 = None
    var_mean_8 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
    getitem_16: "f32[1, 512, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 512, 1]" = var_mean_8[1];  var_mean_8 = None
    add_50: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
    rsqrt_8: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
    sub_17: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_49, getitem_17);  add_49 = getitem_17 = None
    mul_33: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_8);  sub_17 = rsqrt_8 = None
    mul_34: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_33, arg103_1);  mul_33 = arg103_1 = None
    add_51: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_34, arg104_1);  mul_34 = arg104_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_144: "f32[512, 768]" = torch.ops.aten.view.default(add_51, [512, 768])
    permute_76: "f32[768, 384]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
    addmm_28: "f32[512, 384]" = torch.ops.aten.addmm.default(arg106_1, view_144, permute_76);  arg106_1 = view_144 = permute_76 = None
    view_145: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_28, [1, 512, 384]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_146: "f32[512, 768]" = torch.ops.aten.view.default(add_51, [512, 768])
    permute_77: "f32[768, 384]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
    addmm_29: "f32[512, 384]" = torch.ops.aten.addmm.default(arg108_1, view_146, permute_77);  arg108_1 = view_146 = permute_77 = None
    view_147: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_29, [1, 512, 384]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_148: "f32[512, 768]" = torch.ops.aten.view.default(add_51, [512, 768])
    permute_78: "f32[768, 384]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
    addmm_30: "f32[512, 384]" = torch.ops.aten.addmm.default(arg110_1, view_148, permute_78);  arg110_1 = view_148 = permute_78 = None
    view_149: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_30, [1, 512, 384]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_79: "f32[1, 768, 512]" = torch.ops.aten.permute.default(add_51, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_8: "f32[1, 768, 512]" = torch.ops.aten.convolution.default(permute_79, arg111_1, None, [1], [4], [1], False, [0], 768);  permute_79 = arg111_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_9: "f32[1, 384, 512]" = torch.ops.aten.convolution.default(convolution_8, arg112_1, None, [1], [0], [1], False, [0], 1);  convolution_8 = arg112_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_52: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_9, arg4_1);  convolution_9 = arg4_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_150: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_145, [1, 512, 6, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_81: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_150, [0, 2, 1, 3]);  view_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_151: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_147, [1, 512, 6, 64]);  view_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_82: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_151, [0, 2, 1, 3]);  view_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_152: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_149, [1, 512, 6, 64]);  view_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_83: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_152, [0, 2, 1, 3]);  view_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_84: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_52, [0, 2, 1]);  add_52 = None
    mul_35: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(permute_84, view_145);  permute_84 = view_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    permute_85: "f32[384, 54]" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
    view_153: "f32[512, 384]" = torch.ops.aten.view.default(mul_35, [512, 384]);  mul_35 = None
    mm_4: "f32[512, 54]" = torch.ops.aten.mm.default(view_153, permute_85);  view_153 = permute_85 = None
    view_154: "f32[1, 512, 54]" = torch.ops.aten.view.default(mm_4, [1, 512, 54]);  mm_4 = None
    add_53: "f32[1, 512, 54]" = torch.ops.aten.add.Tensor(view_154, arg114_1);  view_154 = arg114_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_155: "f32[3072, 9, 1]" = torch.ops.aten.view.default(add_53, [-1, 9, 1]);  add_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    amax_8: "f32[3072, 1, 1]" = torch.ops.aten.amax.default(view_155, [1], True)
    sub_18: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(view_155, amax_8);  view_155 = amax_8 = None
    exp_8: "f32[3072, 9, 1]" = torch.ops.aten.exp.default(sub_18);  sub_18 = None
    sum_9: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [1], True)
    div_12: "f32[3072, 9, 1]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_156: "f32[512, 768]" = torch.ops.aten.view.default(add_51, [512, 768])
    permute_86: "f32[768, 384]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
    addmm_31: "f32[512, 384]" = torch.ops.aten.addmm.default(arg116_1, view_156, permute_86);  arg116_1 = view_156 = permute_86 = None
    view_157: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_31, [1, 512, 384]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_158: "f32[1, 512, 384]" = torch.ops.aten.view.default(view_157, [1, -1, 384]);  view_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    permute_87: "f32[1, 384, 512]" = torch.ops.aten.permute.default(view_158, [0, 2, 1]);  view_158 = None
    clone_25: "f32[1, 384, 512]" = torch.ops.aten.clone.default(permute_87, memory_format = torch.contiguous_format);  permute_87 = None
    unsqueeze_30: "f32[1, 384, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_25, -1);  clone_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    iota_16: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_31: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota_16, 0);  iota_16 = None
    iota_17: "i64[9]" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_32: "i64[9, 1]" = torch.ops.aten.unsqueeze.default(iota_17, -1);  iota_17 = None
    add_54: "i64[9, 512]" = torch.ops.aten.add.Tensor(unsqueeze_31, unsqueeze_32);  unsqueeze_31 = unsqueeze_32 = None
    iota_18: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_33: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_18, 0);  iota_18 = None
    iota_19: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_34: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_19, -1);  iota_19 = None
    add_55: "i64[1, 1]" = torch.ops.aten.add.Tensor(unsqueeze_33, unsqueeze_34);  unsqueeze_33 = unsqueeze_34 = None
    constant_pad_nd_4: "f32[1, 384, 520, 1]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_30, [0, 0, 4, 4], 0.0);  unsqueeze_30 = None
    unsqueeze_35: "i64[9, 512, 1]" = torch.ops.aten.unsqueeze.default(add_54, -1);  add_54 = None
    unsqueeze_36: "i64[9, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_35, -1);  unsqueeze_35 = None
    slice_13: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(constant_pad_nd_4, 0, 0, 9223372036854775807);  constant_pad_nd_4 = None
    slice_14: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(slice_13, 1, 0, 9223372036854775807);  slice_13 = None
    index_4: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.index.Tensor(slice_14, [None, None, unsqueeze_36, add_55]);  slice_14 = unsqueeze_36 = add_55 = None
    permute_88: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.permute.default(index_4, [0, 1, 2, 4, 3, 5]);  index_4 = None
    view_159: "f32[1, 3456, 512]" = torch.ops.aten.view.default(permute_88, [1, 3456, 512]);  permute_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    permute_89: "f32[1, 512, 3456]" = torch.ops.aten.permute.default(view_159, [0, 2, 1]);  view_159 = None
    view_160: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(permute_89, [1, 512, 384, 9]);  permute_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    clone_26: "f32[1, 512, 384, 9]" = torch.ops.aten.clone.default(view_160, memory_format = torch.contiguous_format);  view_160 = None
    view_161: "f32[3072, 64, 9]" = torch.ops.aten.view.default(clone_26, [3072, 64, 9]);  clone_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    expand_25: "f32[3072, 64, 9]" = torch.ops.aten.expand.default(view_161, [3072, 64, 9]);  view_161 = None
    view_162: "f32[3072, 64, 9]" = torch.ops.aten.view.default(expand_25, [3072, 64, 9]);  expand_25 = None
    expand_26: "f32[3072, 9, 1]" = torch.ops.aten.expand.default(div_12, [3072, 9, 1]);  div_12 = None
    view_163: "f32[3072, 9, 1]" = torch.ops.aten.view.default(expand_26, [3072, 9, 1]);  expand_26 = None
    bmm_12: "f32[3072, 64, 1]" = torch.ops.aten.bmm.default(view_162, view_163);  view_162 = view_163 = None
    view_164: "f32[3072, 64, 1]" = torch.ops.aten.view.default(bmm_12, [3072, 64, 1]);  bmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    view_165: "f32[512, 384]" = torch.ops.aten.view.default(view_164, [-1, 384]);  view_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_90: "f32[1, 6, 64, 512]" = torch.ops.aten.permute.default(permute_82, [0, 1, 3, 2]);  permute_82 = None
    expand_27: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_81, [1, 6, 512, 64]);  permute_81 = None
    view_166: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_27, [6, 512, 64]);  expand_27 = None
    expand_28: "f32[1, 6, 64, 512]" = torch.ops.aten.expand.default(permute_90, [1, 6, 64, 512]);  permute_90 = None
    view_167: "f32[6, 64, 512]" = torch.ops.aten.view.default(expand_28, [6, 64, 512]);  expand_28 = None
    bmm_13: "f32[6, 512, 512]" = torch.ops.aten.bmm.default(view_166, view_167);  view_166 = view_167 = None
    view_168: "f32[1, 6, 512, 512]" = torch.ops.aten.view.default(bmm_13, [1, 6, 512, 512]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_13: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(view_168, 8.0);  view_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:397, code: attention_scores = attention_scores + attention_mask
    add_56: "f32[1, 6, 512, 512]" = torch.ops.aten.add.Tensor(div_13, mul);  div_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_9: "f32[1, 6, 512, 1]" = torch.ops.aten.amax.default(add_56, [-1], True)
    sub_19: "f32[1, 6, 512, 512]" = torch.ops.aten.sub.Tensor(add_56, amax_9);  add_56 = amax_9 = None
    exp_9: "f32[1, 6, 512, 512]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
    sum_10: "f32[1, 6, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_14: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    clone_27: "f32[1, 6, 512, 512]" = torch.ops.aten.clone.default(div_14);  div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_29: "f32[1, 6, 512, 512]" = torch.ops.aten.expand.default(clone_27, [1, 6, 512, 512]);  clone_27 = None
    view_169: "f32[6, 512, 512]" = torch.ops.aten.view.default(expand_29, [6, 512, 512]);  expand_29 = None
    expand_30: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_83, [1, 6, 512, 64]);  permute_83 = None
    view_170: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_30, [6, 512, 64]);  expand_30 = None
    bmm_14: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(view_169, view_170);  view_169 = view_170 = None
    view_171: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_14, [1, 6, 512, 64]);  bmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_91: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_171, [0, 2, 1, 3]);  view_171 = None
    clone_28: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_91, memory_format = torch.contiguous_format);  permute_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_172: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_165, [1, -1, 6, 64]);  view_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    cat_4: "f32[1, 512, 12, 64]" = torch.ops.aten.cat.default([clone_28, view_172], 2);  clone_28 = view_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_173: "f32[1, 512, 768]" = torch.ops.aten.view.default(cat_4, [1, 512, 768]);  cat_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_174: "f32[512, 768]" = torch.ops.aten.view.default(view_173, [512, 768]);  view_173 = None
    permute_92: "f32[768, 768]" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
    addmm_32: "f32[512, 768]" = torch.ops.aten.addmm.default(arg118_1, view_174, permute_92);  arg118_1 = view_174 = permute_92 = None
    view_175: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_32, [1, 512, 768]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    clone_29: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_175);  view_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_57: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_29, add_51);  clone_29 = add_51 = None
    var_mean_9 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
    getitem_18: "f32[1, 512, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 512, 1]" = var_mean_9[1];  var_mean_9 = None
    add_58: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
    rsqrt_9: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
    sub_20: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_57, getitem_19);  add_57 = getitem_19 = None
    mul_36: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_9);  sub_20 = rsqrt_9 = None
    mul_37: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_36, arg119_1);  mul_36 = arg119_1 = None
    add_59: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_37, arg120_1);  mul_37 = arg120_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_176: "f32[512, 768]" = torch.ops.aten.view.default(add_59, [512, 768])
    permute_93: "f32[768, 3072]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
    addmm_33: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg122_1, view_176, permute_93);  arg122_1 = view_176 = permute_93 = None
    view_177: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_33, [1, 512, 3072]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_38: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_177, 0.5)
    mul_39: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_177, 0.7071067811865476);  view_177 = None
    erf_4: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_39);  mul_39 = None
    add_60: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_40: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_38, add_60);  mul_38 = add_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_178: "f32[512, 3072]" = torch.ops.aten.view.default(mul_40, [512, 3072]);  mul_40 = None
    permute_94: "f32[3072, 768]" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
    addmm_34: "f32[512, 768]" = torch.ops.aten.addmm.default(arg124_1, view_178, permute_94);  arg124_1 = view_178 = permute_94 = None
    view_179: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_34, [1, 512, 768]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    clone_30: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_179);  view_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_61: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_30, add_59);  clone_30 = add_59 = None
    var_mean_10 = torch.ops.aten.var_mean.correction(add_61, [2], correction = 0, keepdim = True)
    getitem_20: "f32[1, 512, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 512, 1]" = var_mean_10[1];  var_mean_10 = None
    add_62: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
    rsqrt_10: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
    sub_21: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_61, getitem_21);  add_61 = getitem_21 = None
    mul_41: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_10);  sub_21 = rsqrt_10 = None
    mul_42: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_41, arg125_1);  mul_41 = arg125_1 = None
    add_63: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_42, arg126_1);  mul_42 = arg126_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_180: "f32[512, 768]" = torch.ops.aten.view.default(add_63, [512, 768])
    permute_95: "f32[768, 384]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
    addmm_35: "f32[512, 384]" = torch.ops.aten.addmm.default(arg128_1, view_180, permute_95);  arg128_1 = view_180 = permute_95 = None
    view_181: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_35, [1, 512, 384]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_182: "f32[512, 768]" = torch.ops.aten.view.default(add_63, [512, 768])
    permute_96: "f32[768, 384]" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
    addmm_36: "f32[512, 384]" = torch.ops.aten.addmm.default(arg130_1, view_182, permute_96);  arg130_1 = view_182 = permute_96 = None
    view_183: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_36, [1, 512, 384]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_184: "f32[512, 768]" = torch.ops.aten.view.default(add_63, [512, 768])
    permute_97: "f32[768, 384]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
    addmm_37: "f32[512, 384]" = torch.ops.aten.addmm.default(arg132_1, view_184, permute_97);  arg132_1 = view_184 = permute_97 = None
    view_185: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_37, [1, 512, 384]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_98: "f32[1, 768, 512]" = torch.ops.aten.permute.default(add_63, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_10: "f32[1, 768, 512]" = torch.ops.aten.convolution.default(permute_98, arg133_1, None, [1], [4], [1], False, [0], 768);  permute_98 = arg133_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_11: "f32[1, 384, 512]" = torch.ops.aten.convolution.default(convolution_10, arg134_1, None, [1], [0], [1], False, [0], 1);  convolution_10 = arg134_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_64: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_11, arg5_1);  convolution_11 = arg5_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_186: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_181, [1, 512, 6, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_100: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_187: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_183, [1, 512, 6, 64]);  view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_101: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_187, [0, 2, 1, 3]);  view_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_188: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_185, [1, 512, 6, 64]);  view_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_102: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_188, [0, 2, 1, 3]);  view_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_103: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_64, [0, 2, 1]);  add_64 = None
    mul_43: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(permute_103, view_181);  permute_103 = view_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    permute_104: "f32[384, 54]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
    view_189: "f32[512, 384]" = torch.ops.aten.view.default(mul_43, [512, 384]);  mul_43 = None
    mm_5: "f32[512, 54]" = torch.ops.aten.mm.default(view_189, permute_104);  view_189 = permute_104 = None
    view_190: "f32[1, 512, 54]" = torch.ops.aten.view.default(mm_5, [1, 512, 54]);  mm_5 = None
    add_65: "f32[1, 512, 54]" = torch.ops.aten.add.Tensor(view_190, arg136_1);  view_190 = arg136_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_191: "f32[3072, 9, 1]" = torch.ops.aten.view.default(add_65, [-1, 9, 1]);  add_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    amax_10: "f32[3072, 1, 1]" = torch.ops.aten.amax.default(view_191, [1], True)
    sub_22: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(view_191, amax_10);  view_191 = amax_10 = None
    exp_10: "f32[3072, 9, 1]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
    sum_11: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [1], True)
    div_15: "f32[3072, 9, 1]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_192: "f32[512, 768]" = torch.ops.aten.view.default(add_63, [512, 768])
    permute_105: "f32[768, 384]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
    addmm_38: "f32[512, 384]" = torch.ops.aten.addmm.default(arg138_1, view_192, permute_105);  arg138_1 = view_192 = permute_105 = None
    view_193: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_38, [1, 512, 384]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_194: "f32[1, 512, 384]" = torch.ops.aten.view.default(view_193, [1, -1, 384]);  view_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    permute_106: "f32[1, 384, 512]" = torch.ops.aten.permute.default(view_194, [0, 2, 1]);  view_194 = None
    clone_31: "f32[1, 384, 512]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
    unsqueeze_37: "f32[1, 384, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_31, -1);  clone_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    iota_20: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_38: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota_20, 0);  iota_20 = None
    iota_21: "i64[9]" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_39: "i64[9, 1]" = torch.ops.aten.unsqueeze.default(iota_21, -1);  iota_21 = None
    add_66: "i64[9, 512]" = torch.ops.aten.add.Tensor(unsqueeze_38, unsqueeze_39);  unsqueeze_38 = unsqueeze_39 = None
    iota_22: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_40: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_22, 0);  iota_22 = None
    iota_23: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_41: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_23, -1);  iota_23 = None
    add_67: "i64[1, 1]" = torch.ops.aten.add.Tensor(unsqueeze_40, unsqueeze_41);  unsqueeze_40 = unsqueeze_41 = None
    constant_pad_nd_5: "f32[1, 384, 520, 1]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_37, [0, 0, 4, 4], 0.0);  unsqueeze_37 = None
    unsqueeze_42: "i64[9, 512, 1]" = torch.ops.aten.unsqueeze.default(add_66, -1);  add_66 = None
    unsqueeze_43: "i64[9, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    slice_15: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(constant_pad_nd_5, 0, 0, 9223372036854775807);  constant_pad_nd_5 = None
    slice_16: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(slice_15, 1, 0, 9223372036854775807);  slice_15 = None
    index_5: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.index.Tensor(slice_16, [None, None, unsqueeze_43, add_67]);  slice_16 = unsqueeze_43 = add_67 = None
    permute_107: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.permute.default(index_5, [0, 1, 2, 4, 3, 5]);  index_5 = None
    view_195: "f32[1, 3456, 512]" = torch.ops.aten.view.default(permute_107, [1, 3456, 512]);  permute_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    permute_108: "f32[1, 512, 3456]" = torch.ops.aten.permute.default(view_195, [0, 2, 1]);  view_195 = None
    view_196: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(permute_108, [1, 512, 384, 9]);  permute_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    clone_32: "f32[1, 512, 384, 9]" = torch.ops.aten.clone.default(view_196, memory_format = torch.contiguous_format);  view_196 = None
    view_197: "f32[3072, 64, 9]" = torch.ops.aten.view.default(clone_32, [3072, 64, 9]);  clone_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    expand_31: "f32[3072, 64, 9]" = torch.ops.aten.expand.default(view_197, [3072, 64, 9]);  view_197 = None
    view_198: "f32[3072, 64, 9]" = torch.ops.aten.view.default(expand_31, [3072, 64, 9]);  expand_31 = None
    expand_32: "f32[3072, 9, 1]" = torch.ops.aten.expand.default(div_15, [3072, 9, 1]);  div_15 = None
    view_199: "f32[3072, 9, 1]" = torch.ops.aten.view.default(expand_32, [3072, 9, 1]);  expand_32 = None
    bmm_15: "f32[3072, 64, 1]" = torch.ops.aten.bmm.default(view_198, view_199);  view_198 = view_199 = None
    view_200: "f32[3072, 64, 1]" = torch.ops.aten.view.default(bmm_15, [3072, 64, 1]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    view_201: "f32[512, 384]" = torch.ops.aten.view.default(view_200, [-1, 384]);  view_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_109: "f32[1, 6, 64, 512]" = torch.ops.aten.permute.default(permute_101, [0, 1, 3, 2]);  permute_101 = None
    expand_33: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_100, [1, 6, 512, 64]);  permute_100 = None
    view_202: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_33, [6, 512, 64]);  expand_33 = None
    expand_34: "f32[1, 6, 64, 512]" = torch.ops.aten.expand.default(permute_109, [1, 6, 64, 512]);  permute_109 = None
    view_203: "f32[6, 64, 512]" = torch.ops.aten.view.default(expand_34, [6, 64, 512]);  expand_34 = None
    bmm_16: "f32[6, 512, 512]" = torch.ops.aten.bmm.default(view_202, view_203);  view_202 = view_203 = None
    view_204: "f32[1, 6, 512, 512]" = torch.ops.aten.view.default(bmm_16, [1, 6, 512, 512]);  bmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_16: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(view_204, 8.0);  view_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:397, code: attention_scores = attention_scores + attention_mask
    add_68: "f32[1, 6, 512, 512]" = torch.ops.aten.add.Tensor(div_16, mul);  div_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_11: "f32[1, 6, 512, 1]" = torch.ops.aten.amax.default(add_68, [-1], True)
    sub_23: "f32[1, 6, 512, 512]" = torch.ops.aten.sub.Tensor(add_68, amax_11);  add_68 = amax_11 = None
    exp_11: "f32[1, 6, 512, 512]" = torch.ops.aten.exp.default(sub_23);  sub_23 = None
    sum_12: "f32[1, 6, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_17: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    clone_33: "f32[1, 6, 512, 512]" = torch.ops.aten.clone.default(div_17);  div_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_35: "f32[1, 6, 512, 512]" = torch.ops.aten.expand.default(clone_33, [1, 6, 512, 512]);  clone_33 = None
    view_205: "f32[6, 512, 512]" = torch.ops.aten.view.default(expand_35, [6, 512, 512]);  expand_35 = None
    expand_36: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_102, [1, 6, 512, 64]);  permute_102 = None
    view_206: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_36, [6, 512, 64]);  expand_36 = None
    bmm_17: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(view_205, view_206);  view_205 = view_206 = None
    view_207: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_17, [1, 6, 512, 64]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_110: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_207, [0, 2, 1, 3]);  view_207 = None
    clone_34: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_110, memory_format = torch.contiguous_format);  permute_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_208: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_201, [1, -1, 6, 64]);  view_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    cat_5: "f32[1, 512, 12, 64]" = torch.ops.aten.cat.default([clone_34, view_208], 2);  clone_34 = view_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_209: "f32[1, 512, 768]" = torch.ops.aten.view.default(cat_5, [1, 512, 768]);  cat_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_210: "f32[512, 768]" = torch.ops.aten.view.default(view_209, [512, 768]);  view_209 = None
    permute_111: "f32[768, 768]" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
    addmm_39: "f32[512, 768]" = torch.ops.aten.addmm.default(arg140_1, view_210, permute_111);  arg140_1 = view_210 = permute_111 = None
    view_211: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_39, [1, 512, 768]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    clone_35: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_211);  view_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_69: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_35, add_63);  clone_35 = add_63 = None
    var_mean_11 = torch.ops.aten.var_mean.correction(add_69, [2], correction = 0, keepdim = True)
    getitem_22: "f32[1, 512, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 512, 1]" = var_mean_11[1];  var_mean_11 = None
    add_70: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
    rsqrt_11: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
    sub_24: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_69, getitem_23);  add_69 = getitem_23 = None
    mul_44: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_11);  sub_24 = rsqrt_11 = None
    mul_45: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_44, arg141_1);  mul_44 = arg141_1 = None
    add_71: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_45, arg142_1);  mul_45 = arg142_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_212: "f32[512, 768]" = torch.ops.aten.view.default(add_71, [512, 768])
    permute_112: "f32[768, 3072]" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
    addmm_40: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg144_1, view_212, permute_112);  arg144_1 = view_212 = permute_112 = None
    view_213: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_40, [1, 512, 3072]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_46: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_213, 0.5)
    mul_47: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_213, 0.7071067811865476);  view_213 = None
    erf_5: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
    add_72: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_48: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_46, add_72);  mul_46 = add_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_214: "f32[512, 3072]" = torch.ops.aten.view.default(mul_48, [512, 3072]);  mul_48 = None
    permute_113: "f32[3072, 768]" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
    addmm_41: "f32[512, 768]" = torch.ops.aten.addmm.default(arg146_1, view_214, permute_113);  arg146_1 = view_214 = permute_113 = None
    view_215: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_41, [1, 512, 768]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    clone_36: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_215);  view_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_73: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_36, add_71);  clone_36 = add_71 = None
    var_mean_12 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
    getitem_24: "f32[1, 512, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 512, 1]" = var_mean_12[1];  var_mean_12 = None
    add_74: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
    rsqrt_12: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    sub_25: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_73, getitem_25);  add_73 = getitem_25 = None
    mul_49: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_12);  sub_25 = rsqrt_12 = None
    mul_50: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_49, arg147_1);  mul_49 = arg147_1 = None
    add_75: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_50, arg148_1);  mul_50 = arg148_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_216: "f32[512, 768]" = torch.ops.aten.view.default(add_75, [512, 768])
    permute_114: "f32[768, 384]" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
    addmm_42: "f32[512, 384]" = torch.ops.aten.addmm.default(arg150_1, view_216, permute_114);  arg150_1 = view_216 = permute_114 = None
    view_217: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_42, [1, 512, 384]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_218: "f32[512, 768]" = torch.ops.aten.view.default(add_75, [512, 768])
    permute_115: "f32[768, 384]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
    addmm_43: "f32[512, 384]" = torch.ops.aten.addmm.default(arg152_1, view_218, permute_115);  arg152_1 = view_218 = permute_115 = None
    view_219: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_43, [1, 512, 384]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_220: "f32[512, 768]" = torch.ops.aten.view.default(add_75, [512, 768])
    permute_116: "f32[768, 384]" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
    addmm_44: "f32[512, 384]" = torch.ops.aten.addmm.default(arg154_1, view_220, permute_116);  arg154_1 = view_220 = permute_116 = None
    view_221: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_44, [1, 512, 384]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_117: "f32[1, 768, 512]" = torch.ops.aten.permute.default(add_75, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_12: "f32[1, 768, 512]" = torch.ops.aten.convolution.default(permute_117, arg155_1, None, [1], [4], [1], False, [0], 768);  permute_117 = arg155_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_13: "f32[1, 384, 512]" = torch.ops.aten.convolution.default(convolution_12, arg156_1, None, [1], [0], [1], False, [0], 1);  convolution_12 = arg156_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_76: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_13, arg6_1);  convolution_13 = arg6_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_222: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_217, [1, 512, 6, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_119: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_223: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_219, [1, 512, 6, 64]);  view_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_120: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_223, [0, 2, 1, 3]);  view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_224: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_221, [1, 512, 6, 64]);  view_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_121: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_122: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_76, [0, 2, 1]);  add_76 = None
    mul_51: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(permute_122, view_217);  permute_122 = view_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    permute_123: "f32[384, 54]" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
    view_225: "f32[512, 384]" = torch.ops.aten.view.default(mul_51, [512, 384]);  mul_51 = None
    mm_6: "f32[512, 54]" = torch.ops.aten.mm.default(view_225, permute_123);  view_225 = permute_123 = None
    view_226: "f32[1, 512, 54]" = torch.ops.aten.view.default(mm_6, [1, 512, 54]);  mm_6 = None
    add_77: "f32[1, 512, 54]" = torch.ops.aten.add.Tensor(view_226, arg158_1);  view_226 = arg158_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_227: "f32[3072, 9, 1]" = torch.ops.aten.view.default(add_77, [-1, 9, 1]);  add_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    amax_12: "f32[3072, 1, 1]" = torch.ops.aten.amax.default(view_227, [1], True)
    sub_26: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(view_227, amax_12);  view_227 = amax_12 = None
    exp_12: "f32[3072, 9, 1]" = torch.ops.aten.exp.default(sub_26);  sub_26 = None
    sum_13: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [1], True)
    div_18: "f32[3072, 9, 1]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_228: "f32[512, 768]" = torch.ops.aten.view.default(add_75, [512, 768])
    permute_124: "f32[768, 384]" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
    addmm_45: "f32[512, 384]" = torch.ops.aten.addmm.default(arg160_1, view_228, permute_124);  arg160_1 = view_228 = permute_124 = None
    view_229: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_45, [1, 512, 384]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_230: "f32[1, 512, 384]" = torch.ops.aten.view.default(view_229, [1, -1, 384]);  view_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    permute_125: "f32[1, 384, 512]" = torch.ops.aten.permute.default(view_230, [0, 2, 1]);  view_230 = None
    clone_37: "f32[1, 384, 512]" = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
    unsqueeze_44: "f32[1, 384, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_37, -1);  clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    iota_24: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_45: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota_24, 0);  iota_24 = None
    iota_25: "i64[9]" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_46: "i64[9, 1]" = torch.ops.aten.unsqueeze.default(iota_25, -1);  iota_25 = None
    add_78: "i64[9, 512]" = torch.ops.aten.add.Tensor(unsqueeze_45, unsqueeze_46);  unsqueeze_45 = unsqueeze_46 = None
    iota_26: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_47: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_26, 0);  iota_26 = None
    iota_27: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_48: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_27, -1);  iota_27 = None
    add_79: "i64[1, 1]" = torch.ops.aten.add.Tensor(unsqueeze_47, unsqueeze_48);  unsqueeze_47 = unsqueeze_48 = None
    constant_pad_nd_6: "f32[1, 384, 520, 1]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_44, [0, 0, 4, 4], 0.0);  unsqueeze_44 = None
    unsqueeze_49: "i64[9, 512, 1]" = torch.ops.aten.unsqueeze.default(add_78, -1);  add_78 = None
    unsqueeze_50: "i64[9, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_49, -1);  unsqueeze_49 = None
    slice_17: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(constant_pad_nd_6, 0, 0, 9223372036854775807);  constant_pad_nd_6 = None
    slice_18: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(slice_17, 1, 0, 9223372036854775807);  slice_17 = None
    index_6: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.index.Tensor(slice_18, [None, None, unsqueeze_50, add_79]);  slice_18 = unsqueeze_50 = add_79 = None
    permute_126: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.permute.default(index_6, [0, 1, 2, 4, 3, 5]);  index_6 = None
    view_231: "f32[1, 3456, 512]" = torch.ops.aten.view.default(permute_126, [1, 3456, 512]);  permute_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    permute_127: "f32[1, 512, 3456]" = torch.ops.aten.permute.default(view_231, [0, 2, 1]);  view_231 = None
    view_232: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(permute_127, [1, 512, 384, 9]);  permute_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    clone_38: "f32[1, 512, 384, 9]" = torch.ops.aten.clone.default(view_232, memory_format = torch.contiguous_format);  view_232 = None
    view_233: "f32[3072, 64, 9]" = torch.ops.aten.view.default(clone_38, [3072, 64, 9]);  clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    expand_37: "f32[3072, 64, 9]" = torch.ops.aten.expand.default(view_233, [3072, 64, 9]);  view_233 = None
    view_234: "f32[3072, 64, 9]" = torch.ops.aten.view.default(expand_37, [3072, 64, 9]);  expand_37 = None
    expand_38: "f32[3072, 9, 1]" = torch.ops.aten.expand.default(div_18, [3072, 9, 1]);  div_18 = None
    view_235: "f32[3072, 9, 1]" = torch.ops.aten.view.default(expand_38, [3072, 9, 1]);  expand_38 = None
    bmm_18: "f32[3072, 64, 1]" = torch.ops.aten.bmm.default(view_234, view_235);  view_234 = view_235 = None
    view_236: "f32[3072, 64, 1]" = torch.ops.aten.view.default(bmm_18, [3072, 64, 1]);  bmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    view_237: "f32[512, 384]" = torch.ops.aten.view.default(view_236, [-1, 384]);  view_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_128: "f32[1, 6, 64, 512]" = torch.ops.aten.permute.default(permute_120, [0, 1, 3, 2]);  permute_120 = None
    expand_39: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_119, [1, 6, 512, 64]);  permute_119 = None
    view_238: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_39, [6, 512, 64]);  expand_39 = None
    expand_40: "f32[1, 6, 64, 512]" = torch.ops.aten.expand.default(permute_128, [1, 6, 64, 512]);  permute_128 = None
    view_239: "f32[6, 64, 512]" = torch.ops.aten.view.default(expand_40, [6, 64, 512]);  expand_40 = None
    bmm_19: "f32[6, 512, 512]" = torch.ops.aten.bmm.default(view_238, view_239);  view_238 = view_239 = None
    view_240: "f32[1, 6, 512, 512]" = torch.ops.aten.view.default(bmm_19, [1, 6, 512, 512]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_19: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(view_240, 8.0);  view_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:397, code: attention_scores = attention_scores + attention_mask
    add_80: "f32[1, 6, 512, 512]" = torch.ops.aten.add.Tensor(div_19, mul);  div_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_13: "f32[1, 6, 512, 1]" = torch.ops.aten.amax.default(add_80, [-1], True)
    sub_27: "f32[1, 6, 512, 512]" = torch.ops.aten.sub.Tensor(add_80, amax_13);  add_80 = amax_13 = None
    exp_13: "f32[1, 6, 512, 512]" = torch.ops.aten.exp.default(sub_27);  sub_27 = None
    sum_14: "f32[1, 6, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
    div_20: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    clone_39: "f32[1, 6, 512, 512]" = torch.ops.aten.clone.default(div_20);  div_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_41: "f32[1, 6, 512, 512]" = torch.ops.aten.expand.default(clone_39, [1, 6, 512, 512]);  clone_39 = None
    view_241: "f32[6, 512, 512]" = torch.ops.aten.view.default(expand_41, [6, 512, 512]);  expand_41 = None
    expand_42: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_121, [1, 6, 512, 64]);  permute_121 = None
    view_242: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_42, [6, 512, 64]);  expand_42 = None
    bmm_20: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(view_241, view_242);  view_241 = view_242 = None
    view_243: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_20, [1, 6, 512, 64]);  bmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_129: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_243, [0, 2, 1, 3]);  view_243 = None
    clone_40: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_244: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_237, [1, -1, 6, 64]);  view_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    cat_6: "f32[1, 512, 12, 64]" = torch.ops.aten.cat.default([clone_40, view_244], 2);  clone_40 = view_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_245: "f32[1, 512, 768]" = torch.ops.aten.view.default(cat_6, [1, 512, 768]);  cat_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_246: "f32[512, 768]" = torch.ops.aten.view.default(view_245, [512, 768]);  view_245 = None
    permute_130: "f32[768, 768]" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
    addmm_46: "f32[512, 768]" = torch.ops.aten.addmm.default(arg162_1, view_246, permute_130);  arg162_1 = view_246 = permute_130 = None
    view_247: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_46, [1, 512, 768]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    clone_41: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_247);  view_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_81: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_41, add_75);  clone_41 = add_75 = None
    var_mean_13 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
    getitem_26: "f32[1, 512, 1]" = var_mean_13[0]
    getitem_27: "f32[1, 512, 1]" = var_mean_13[1];  var_mean_13 = None
    add_82: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
    rsqrt_13: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
    sub_28: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_81, getitem_27);  add_81 = getitem_27 = None
    mul_52: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_13);  sub_28 = rsqrt_13 = None
    mul_53: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_52, arg163_1);  mul_52 = arg163_1 = None
    add_83: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_53, arg164_1);  mul_53 = arg164_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_248: "f32[512, 768]" = torch.ops.aten.view.default(add_83, [512, 768])
    permute_131: "f32[768, 3072]" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
    addmm_47: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg166_1, view_248, permute_131);  arg166_1 = view_248 = permute_131 = None
    view_249: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_47, [1, 512, 3072]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_54: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_249, 0.5)
    mul_55: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_249, 0.7071067811865476);  view_249 = None
    erf_6: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
    add_84: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_56: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_54, add_84);  mul_54 = add_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_250: "f32[512, 3072]" = torch.ops.aten.view.default(mul_56, [512, 3072]);  mul_56 = None
    permute_132: "f32[3072, 768]" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
    addmm_48: "f32[512, 768]" = torch.ops.aten.addmm.default(arg168_1, view_250, permute_132);  arg168_1 = view_250 = permute_132 = None
    view_251: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_48, [1, 512, 768]);  addmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    clone_42: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_251);  view_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_85: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_42, add_83);  clone_42 = add_83 = None
    var_mean_14 = torch.ops.aten.var_mean.correction(add_85, [2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 512, 1]" = var_mean_14[0]
    getitem_29: "f32[1, 512, 1]" = var_mean_14[1];  var_mean_14 = None
    add_86: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
    rsqrt_14: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
    sub_29: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_85, getitem_29);  add_85 = getitem_29 = None
    mul_57: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_14);  sub_29 = rsqrt_14 = None
    mul_58: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_57, arg169_1);  mul_57 = arg169_1 = None
    add_87: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_58, arg170_1);  mul_58 = arg170_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_252: "f32[512, 768]" = torch.ops.aten.view.default(add_87, [512, 768])
    permute_133: "f32[768, 384]" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
    addmm_49: "f32[512, 384]" = torch.ops.aten.addmm.default(arg172_1, view_252, permute_133);  arg172_1 = view_252 = permute_133 = None
    view_253: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_49, [1, 512, 384]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_254: "f32[512, 768]" = torch.ops.aten.view.default(add_87, [512, 768])
    permute_134: "f32[768, 384]" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
    addmm_50: "f32[512, 384]" = torch.ops.aten.addmm.default(arg174_1, view_254, permute_134);  arg174_1 = view_254 = permute_134 = None
    view_255: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_50, [1, 512, 384]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_256: "f32[512, 768]" = torch.ops.aten.view.default(add_87, [512, 768])
    permute_135: "f32[768, 384]" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
    addmm_51: "f32[512, 384]" = torch.ops.aten.addmm.default(arg176_1, view_256, permute_135);  arg176_1 = view_256 = permute_135 = None
    view_257: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_51, [1, 512, 384]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_136: "f32[1, 768, 512]" = torch.ops.aten.permute.default(add_87, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_14: "f32[1, 768, 512]" = torch.ops.aten.convolution.default(permute_136, arg177_1, None, [1], [4], [1], False, [0], 768);  permute_136 = arg177_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_15: "f32[1, 384, 512]" = torch.ops.aten.convolution.default(convolution_14, arg178_1, None, [1], [0], [1], False, [0], 1);  convolution_14 = arg178_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_88: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_15, arg7_1);  convolution_15 = arg7_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_258: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_253, [1, 512, 6, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_138: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_258, [0, 2, 1, 3]);  view_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_259: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_255, [1, 512, 6, 64]);  view_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_139: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_259, [0, 2, 1, 3]);  view_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_260: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_257, [1, 512, 6, 64]);  view_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_140: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_260, [0, 2, 1, 3]);  view_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_141: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_88, [0, 2, 1]);  add_88 = None
    mul_59: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(permute_141, view_253);  permute_141 = view_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    permute_142: "f32[384, 54]" = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
    view_261: "f32[512, 384]" = torch.ops.aten.view.default(mul_59, [512, 384]);  mul_59 = None
    mm_7: "f32[512, 54]" = torch.ops.aten.mm.default(view_261, permute_142);  view_261 = permute_142 = None
    view_262: "f32[1, 512, 54]" = torch.ops.aten.view.default(mm_7, [1, 512, 54]);  mm_7 = None
    add_89: "f32[1, 512, 54]" = torch.ops.aten.add.Tensor(view_262, arg180_1);  view_262 = arg180_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_263: "f32[3072, 9, 1]" = torch.ops.aten.view.default(add_89, [-1, 9, 1]);  add_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    amax_14: "f32[3072, 1, 1]" = torch.ops.aten.amax.default(view_263, [1], True)
    sub_30: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(view_263, amax_14);  view_263 = amax_14 = None
    exp_14: "f32[3072, 9, 1]" = torch.ops.aten.exp.default(sub_30);  sub_30 = None
    sum_15: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [1], True)
    div_21: "f32[3072, 9, 1]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_264: "f32[512, 768]" = torch.ops.aten.view.default(add_87, [512, 768])
    permute_143: "f32[768, 384]" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
    addmm_52: "f32[512, 384]" = torch.ops.aten.addmm.default(arg182_1, view_264, permute_143);  arg182_1 = view_264 = permute_143 = None
    view_265: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_52, [1, 512, 384]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_266: "f32[1, 512, 384]" = torch.ops.aten.view.default(view_265, [1, -1, 384]);  view_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    permute_144: "f32[1, 384, 512]" = torch.ops.aten.permute.default(view_266, [0, 2, 1]);  view_266 = None
    clone_43: "f32[1, 384, 512]" = torch.ops.aten.clone.default(permute_144, memory_format = torch.contiguous_format);  permute_144 = None
    unsqueeze_51: "f32[1, 384, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_43, -1);  clone_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    iota_28: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_52: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota_28, 0);  iota_28 = None
    iota_29: "i64[9]" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_53: "i64[9, 1]" = torch.ops.aten.unsqueeze.default(iota_29, -1);  iota_29 = None
    add_90: "i64[9, 512]" = torch.ops.aten.add.Tensor(unsqueeze_52, unsqueeze_53);  unsqueeze_52 = unsqueeze_53 = None
    iota_30: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_54: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_30, 0);  iota_30 = None
    iota_31: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_55: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_31, -1);  iota_31 = None
    add_91: "i64[1, 1]" = torch.ops.aten.add.Tensor(unsqueeze_54, unsqueeze_55);  unsqueeze_54 = unsqueeze_55 = None
    constant_pad_nd_7: "f32[1, 384, 520, 1]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_51, [0, 0, 4, 4], 0.0);  unsqueeze_51 = None
    unsqueeze_56: "i64[9, 512, 1]" = torch.ops.aten.unsqueeze.default(add_90, -1);  add_90 = None
    unsqueeze_57: "i64[9, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    slice_19: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(constant_pad_nd_7, 0, 0, 9223372036854775807);  constant_pad_nd_7 = None
    slice_20: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(slice_19, 1, 0, 9223372036854775807);  slice_19 = None
    index_7: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.index.Tensor(slice_20, [None, None, unsqueeze_57, add_91]);  slice_20 = unsqueeze_57 = add_91 = None
    permute_145: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.permute.default(index_7, [0, 1, 2, 4, 3, 5]);  index_7 = None
    view_267: "f32[1, 3456, 512]" = torch.ops.aten.view.default(permute_145, [1, 3456, 512]);  permute_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    permute_146: "f32[1, 512, 3456]" = torch.ops.aten.permute.default(view_267, [0, 2, 1]);  view_267 = None
    view_268: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(permute_146, [1, 512, 384, 9]);  permute_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    clone_44: "f32[1, 512, 384, 9]" = torch.ops.aten.clone.default(view_268, memory_format = torch.contiguous_format);  view_268 = None
    view_269: "f32[3072, 64, 9]" = torch.ops.aten.view.default(clone_44, [3072, 64, 9]);  clone_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    expand_43: "f32[3072, 64, 9]" = torch.ops.aten.expand.default(view_269, [3072, 64, 9]);  view_269 = None
    view_270: "f32[3072, 64, 9]" = torch.ops.aten.view.default(expand_43, [3072, 64, 9]);  expand_43 = None
    expand_44: "f32[3072, 9, 1]" = torch.ops.aten.expand.default(div_21, [3072, 9, 1]);  div_21 = None
    view_271: "f32[3072, 9, 1]" = torch.ops.aten.view.default(expand_44, [3072, 9, 1]);  expand_44 = None
    bmm_21: "f32[3072, 64, 1]" = torch.ops.aten.bmm.default(view_270, view_271);  view_270 = view_271 = None
    view_272: "f32[3072, 64, 1]" = torch.ops.aten.view.default(bmm_21, [3072, 64, 1]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    view_273: "f32[512, 384]" = torch.ops.aten.view.default(view_272, [-1, 384]);  view_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_147: "f32[1, 6, 64, 512]" = torch.ops.aten.permute.default(permute_139, [0, 1, 3, 2]);  permute_139 = None
    expand_45: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_138, [1, 6, 512, 64]);  permute_138 = None
    view_274: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_45, [6, 512, 64]);  expand_45 = None
    expand_46: "f32[1, 6, 64, 512]" = torch.ops.aten.expand.default(permute_147, [1, 6, 64, 512]);  permute_147 = None
    view_275: "f32[6, 64, 512]" = torch.ops.aten.view.default(expand_46, [6, 64, 512]);  expand_46 = None
    bmm_22: "f32[6, 512, 512]" = torch.ops.aten.bmm.default(view_274, view_275);  view_274 = view_275 = None
    view_276: "f32[1, 6, 512, 512]" = torch.ops.aten.view.default(bmm_22, [1, 6, 512, 512]);  bmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_22: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(view_276, 8.0);  view_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:397, code: attention_scores = attention_scores + attention_mask
    add_92: "f32[1, 6, 512, 512]" = torch.ops.aten.add.Tensor(div_22, mul);  div_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_15: "f32[1, 6, 512, 1]" = torch.ops.aten.amax.default(add_92, [-1], True)
    sub_31: "f32[1, 6, 512, 512]" = torch.ops.aten.sub.Tensor(add_92, amax_15);  add_92 = amax_15 = None
    exp_15: "f32[1, 6, 512, 512]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
    sum_16: "f32[1, 6, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
    div_23: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    clone_45: "f32[1, 6, 512, 512]" = torch.ops.aten.clone.default(div_23);  div_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_47: "f32[1, 6, 512, 512]" = torch.ops.aten.expand.default(clone_45, [1, 6, 512, 512]);  clone_45 = None
    view_277: "f32[6, 512, 512]" = torch.ops.aten.view.default(expand_47, [6, 512, 512]);  expand_47 = None
    expand_48: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_140, [1, 6, 512, 64]);  permute_140 = None
    view_278: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_48, [6, 512, 64]);  expand_48 = None
    bmm_23: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(view_277, view_278);  view_277 = view_278 = None
    view_279: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_23, [1, 6, 512, 64]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_148: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_279, [0, 2, 1, 3]);  view_279 = None
    clone_46: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_148, memory_format = torch.contiguous_format);  permute_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_280: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_273, [1, -1, 6, 64]);  view_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    cat_7: "f32[1, 512, 12, 64]" = torch.ops.aten.cat.default([clone_46, view_280], 2);  clone_46 = view_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_281: "f32[1, 512, 768]" = torch.ops.aten.view.default(cat_7, [1, 512, 768]);  cat_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_282: "f32[512, 768]" = torch.ops.aten.view.default(view_281, [512, 768]);  view_281 = None
    permute_149: "f32[768, 768]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
    addmm_53: "f32[512, 768]" = torch.ops.aten.addmm.default(arg184_1, view_282, permute_149);  arg184_1 = view_282 = permute_149 = None
    view_283: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_53, [1, 512, 768]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    clone_47: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_283);  view_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_93: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_47, add_87);  clone_47 = add_87 = None
    var_mean_15 = torch.ops.aten.var_mean.correction(add_93, [2], correction = 0, keepdim = True)
    getitem_30: "f32[1, 512, 1]" = var_mean_15[0]
    getitem_31: "f32[1, 512, 1]" = var_mean_15[1];  var_mean_15 = None
    add_94: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
    rsqrt_15: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
    sub_32: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_93, getitem_31);  add_93 = getitem_31 = None
    mul_60: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_15);  sub_32 = rsqrt_15 = None
    mul_61: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_60, arg185_1);  mul_60 = arg185_1 = None
    add_95: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_61, arg186_1);  mul_61 = arg186_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_284: "f32[512, 768]" = torch.ops.aten.view.default(add_95, [512, 768])
    permute_150: "f32[768, 3072]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
    addmm_54: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg188_1, view_284, permute_150);  arg188_1 = view_284 = permute_150 = None
    view_285: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_54, [1, 512, 3072]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_62: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_285, 0.5)
    mul_63: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_285, 0.7071067811865476);  view_285 = None
    erf_7: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_63);  mul_63 = None
    add_96: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_64: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_62, add_96);  mul_62 = add_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_286: "f32[512, 3072]" = torch.ops.aten.view.default(mul_64, [512, 3072]);  mul_64 = None
    permute_151: "f32[3072, 768]" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
    addmm_55: "f32[512, 768]" = torch.ops.aten.addmm.default(arg190_1, view_286, permute_151);  arg190_1 = view_286 = permute_151 = None
    view_287: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_55, [1, 512, 768]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    clone_48: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_287);  view_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_97: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_48, add_95);  clone_48 = add_95 = None
    var_mean_16 = torch.ops.aten.var_mean.correction(add_97, [2], correction = 0, keepdim = True)
    getitem_32: "f32[1, 512, 1]" = var_mean_16[0]
    getitem_33: "f32[1, 512, 1]" = var_mean_16[1];  var_mean_16 = None
    add_98: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
    rsqrt_16: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
    sub_33: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_97, getitem_33);  add_97 = getitem_33 = None
    mul_65: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_16);  sub_33 = rsqrt_16 = None
    mul_66: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_65, arg191_1);  mul_65 = arg191_1 = None
    add_99: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_66, arg192_1);  mul_66 = arg192_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_288: "f32[512, 768]" = torch.ops.aten.view.default(add_99, [512, 768])
    permute_152: "f32[768, 384]" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
    addmm_56: "f32[512, 384]" = torch.ops.aten.addmm.default(arg194_1, view_288, permute_152);  arg194_1 = view_288 = permute_152 = None
    view_289: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_56, [1, 512, 384]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_290: "f32[512, 768]" = torch.ops.aten.view.default(add_99, [512, 768])
    permute_153: "f32[768, 384]" = torch.ops.aten.permute.default(arg195_1, [1, 0]);  arg195_1 = None
    addmm_57: "f32[512, 384]" = torch.ops.aten.addmm.default(arg196_1, view_290, permute_153);  arg196_1 = view_290 = permute_153 = None
    view_291: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_57, [1, 512, 384]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_292: "f32[512, 768]" = torch.ops.aten.view.default(add_99, [512, 768])
    permute_154: "f32[768, 384]" = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
    addmm_58: "f32[512, 384]" = torch.ops.aten.addmm.default(arg198_1, view_292, permute_154);  arg198_1 = view_292 = permute_154 = None
    view_293: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_58, [1, 512, 384]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_155: "f32[1, 768, 512]" = torch.ops.aten.permute.default(add_99, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_16: "f32[1, 768, 512]" = torch.ops.aten.convolution.default(permute_155, arg199_1, None, [1], [4], [1], False, [0], 768);  permute_155 = arg199_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_17: "f32[1, 384, 512]" = torch.ops.aten.convolution.default(convolution_16, arg200_1, None, [1], [0], [1], False, [0], 1);  convolution_16 = arg200_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_100: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_17, arg8_1);  convolution_17 = arg8_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_294: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_289, [1, 512, 6, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_157: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_294, [0, 2, 1, 3]);  view_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_295: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_291, [1, 512, 6, 64]);  view_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_158: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_295, [0, 2, 1, 3]);  view_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_296: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_293, [1, 512, 6, 64]);  view_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_159: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_296, [0, 2, 1, 3]);  view_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_160: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_100, [0, 2, 1]);  add_100 = None
    mul_67: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(permute_160, view_289);  permute_160 = view_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    permute_161: "f32[384, 54]" = torch.ops.aten.permute.default(arg201_1, [1, 0]);  arg201_1 = None
    view_297: "f32[512, 384]" = torch.ops.aten.view.default(mul_67, [512, 384]);  mul_67 = None
    mm_8: "f32[512, 54]" = torch.ops.aten.mm.default(view_297, permute_161);  view_297 = permute_161 = None
    view_298: "f32[1, 512, 54]" = torch.ops.aten.view.default(mm_8, [1, 512, 54]);  mm_8 = None
    add_101: "f32[1, 512, 54]" = torch.ops.aten.add.Tensor(view_298, arg202_1);  view_298 = arg202_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_299: "f32[3072, 9, 1]" = torch.ops.aten.view.default(add_101, [-1, 9, 1]);  add_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    amax_16: "f32[3072, 1, 1]" = torch.ops.aten.amax.default(view_299, [1], True)
    sub_34: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(view_299, amax_16);  view_299 = amax_16 = None
    exp_16: "f32[3072, 9, 1]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
    sum_17: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [1], True)
    div_24: "f32[3072, 9, 1]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_300: "f32[512, 768]" = torch.ops.aten.view.default(add_99, [512, 768])
    permute_162: "f32[768, 384]" = torch.ops.aten.permute.default(arg203_1, [1, 0]);  arg203_1 = None
    addmm_59: "f32[512, 384]" = torch.ops.aten.addmm.default(arg204_1, view_300, permute_162);  arg204_1 = view_300 = permute_162 = None
    view_301: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_59, [1, 512, 384]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_302: "f32[1, 512, 384]" = torch.ops.aten.view.default(view_301, [1, -1, 384]);  view_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    permute_163: "f32[1, 384, 512]" = torch.ops.aten.permute.default(view_302, [0, 2, 1]);  view_302 = None
    clone_49: "f32[1, 384, 512]" = torch.ops.aten.clone.default(permute_163, memory_format = torch.contiguous_format);  permute_163 = None
    unsqueeze_58: "f32[1, 384, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_49, -1);  clone_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    iota_32: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_59: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota_32, 0);  iota_32 = None
    iota_33: "i64[9]" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_60: "i64[9, 1]" = torch.ops.aten.unsqueeze.default(iota_33, -1);  iota_33 = None
    add_102: "i64[9, 512]" = torch.ops.aten.add.Tensor(unsqueeze_59, unsqueeze_60);  unsqueeze_59 = unsqueeze_60 = None
    iota_34: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_61: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_34, 0);  iota_34 = None
    iota_35: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_62: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_35, -1);  iota_35 = None
    add_103: "i64[1, 1]" = torch.ops.aten.add.Tensor(unsqueeze_61, unsqueeze_62);  unsqueeze_61 = unsqueeze_62 = None
    constant_pad_nd_8: "f32[1, 384, 520, 1]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_58, [0, 0, 4, 4], 0.0);  unsqueeze_58 = None
    unsqueeze_63: "i64[9, 512, 1]" = torch.ops.aten.unsqueeze.default(add_102, -1);  add_102 = None
    unsqueeze_64: "i64[9, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_63, -1);  unsqueeze_63 = None
    slice_21: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(constant_pad_nd_8, 0, 0, 9223372036854775807);  constant_pad_nd_8 = None
    slice_22: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(slice_21, 1, 0, 9223372036854775807);  slice_21 = None
    index_8: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.index.Tensor(slice_22, [None, None, unsqueeze_64, add_103]);  slice_22 = unsqueeze_64 = add_103 = None
    permute_164: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.permute.default(index_8, [0, 1, 2, 4, 3, 5]);  index_8 = None
    view_303: "f32[1, 3456, 512]" = torch.ops.aten.view.default(permute_164, [1, 3456, 512]);  permute_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    permute_165: "f32[1, 512, 3456]" = torch.ops.aten.permute.default(view_303, [0, 2, 1]);  view_303 = None
    view_304: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(permute_165, [1, 512, 384, 9]);  permute_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    clone_50: "f32[1, 512, 384, 9]" = torch.ops.aten.clone.default(view_304, memory_format = torch.contiguous_format);  view_304 = None
    view_305: "f32[3072, 64, 9]" = torch.ops.aten.view.default(clone_50, [3072, 64, 9]);  clone_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    expand_49: "f32[3072, 64, 9]" = torch.ops.aten.expand.default(view_305, [3072, 64, 9]);  view_305 = None
    view_306: "f32[3072, 64, 9]" = torch.ops.aten.view.default(expand_49, [3072, 64, 9]);  expand_49 = None
    expand_50: "f32[3072, 9, 1]" = torch.ops.aten.expand.default(div_24, [3072, 9, 1]);  div_24 = None
    view_307: "f32[3072, 9, 1]" = torch.ops.aten.view.default(expand_50, [3072, 9, 1]);  expand_50 = None
    bmm_24: "f32[3072, 64, 1]" = torch.ops.aten.bmm.default(view_306, view_307);  view_306 = view_307 = None
    view_308: "f32[3072, 64, 1]" = torch.ops.aten.view.default(bmm_24, [3072, 64, 1]);  bmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    view_309: "f32[512, 384]" = torch.ops.aten.view.default(view_308, [-1, 384]);  view_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_166: "f32[1, 6, 64, 512]" = torch.ops.aten.permute.default(permute_158, [0, 1, 3, 2]);  permute_158 = None
    expand_51: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_157, [1, 6, 512, 64]);  permute_157 = None
    view_310: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_51, [6, 512, 64]);  expand_51 = None
    expand_52: "f32[1, 6, 64, 512]" = torch.ops.aten.expand.default(permute_166, [1, 6, 64, 512]);  permute_166 = None
    view_311: "f32[6, 64, 512]" = torch.ops.aten.view.default(expand_52, [6, 64, 512]);  expand_52 = None
    bmm_25: "f32[6, 512, 512]" = torch.ops.aten.bmm.default(view_310, view_311);  view_310 = view_311 = None
    view_312: "f32[1, 6, 512, 512]" = torch.ops.aten.view.default(bmm_25, [1, 6, 512, 512]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_25: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(view_312, 8.0);  view_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:397, code: attention_scores = attention_scores + attention_mask
    add_104: "f32[1, 6, 512, 512]" = torch.ops.aten.add.Tensor(div_25, mul);  div_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_17: "f32[1, 6, 512, 1]" = torch.ops.aten.amax.default(add_104, [-1], True)
    sub_35: "f32[1, 6, 512, 512]" = torch.ops.aten.sub.Tensor(add_104, amax_17);  add_104 = amax_17 = None
    exp_17: "f32[1, 6, 512, 512]" = torch.ops.aten.exp.default(sub_35);  sub_35 = None
    sum_18: "f32[1, 6, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
    div_26: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    clone_51: "f32[1, 6, 512, 512]" = torch.ops.aten.clone.default(div_26);  div_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_53: "f32[1, 6, 512, 512]" = torch.ops.aten.expand.default(clone_51, [1, 6, 512, 512]);  clone_51 = None
    view_313: "f32[6, 512, 512]" = torch.ops.aten.view.default(expand_53, [6, 512, 512]);  expand_53 = None
    expand_54: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_159, [1, 6, 512, 64]);  permute_159 = None
    view_314: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_54, [6, 512, 64]);  expand_54 = None
    bmm_26: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(view_313, view_314);  view_313 = view_314 = None
    view_315: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_26, [1, 6, 512, 64]);  bmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_167: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_315, [0, 2, 1, 3]);  view_315 = None
    clone_52: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_167, memory_format = torch.contiguous_format);  permute_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_316: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_309, [1, -1, 6, 64]);  view_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    cat_8: "f32[1, 512, 12, 64]" = torch.ops.aten.cat.default([clone_52, view_316], 2);  clone_52 = view_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_317: "f32[1, 512, 768]" = torch.ops.aten.view.default(cat_8, [1, 512, 768]);  cat_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_318: "f32[512, 768]" = torch.ops.aten.view.default(view_317, [512, 768]);  view_317 = None
    permute_168: "f32[768, 768]" = torch.ops.aten.permute.default(arg205_1, [1, 0]);  arg205_1 = None
    addmm_60: "f32[512, 768]" = torch.ops.aten.addmm.default(arg206_1, view_318, permute_168);  arg206_1 = view_318 = permute_168 = None
    view_319: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_60, [1, 512, 768]);  addmm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    clone_53: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_319);  view_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_105: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_53, add_99);  clone_53 = add_99 = None
    var_mean_17 = torch.ops.aten.var_mean.correction(add_105, [2], correction = 0, keepdim = True)
    getitem_34: "f32[1, 512, 1]" = var_mean_17[0]
    getitem_35: "f32[1, 512, 1]" = var_mean_17[1];  var_mean_17 = None
    add_106: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
    rsqrt_17: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
    sub_36: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_105, getitem_35);  add_105 = getitem_35 = None
    mul_68: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_17);  sub_36 = rsqrt_17 = None
    mul_69: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_68, arg207_1);  mul_68 = arg207_1 = None
    add_107: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_69, arg208_1);  mul_69 = arg208_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_320: "f32[512, 768]" = torch.ops.aten.view.default(add_107, [512, 768])
    permute_169: "f32[768, 3072]" = torch.ops.aten.permute.default(arg209_1, [1, 0]);  arg209_1 = None
    addmm_61: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg210_1, view_320, permute_169);  arg210_1 = view_320 = permute_169 = None
    view_321: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_61, [1, 512, 3072]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_70: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_321, 0.5)
    mul_71: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_321, 0.7071067811865476);  view_321 = None
    erf_8: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_71);  mul_71 = None
    add_108: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_72: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_70, add_108);  mul_70 = add_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_322: "f32[512, 3072]" = torch.ops.aten.view.default(mul_72, [512, 3072]);  mul_72 = None
    permute_170: "f32[3072, 768]" = torch.ops.aten.permute.default(arg211_1, [1, 0]);  arg211_1 = None
    addmm_62: "f32[512, 768]" = torch.ops.aten.addmm.default(arg212_1, view_322, permute_170);  arg212_1 = view_322 = permute_170 = None
    view_323: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_62, [1, 512, 768]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    clone_54: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_323);  view_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_109: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_54, add_107);  clone_54 = add_107 = None
    var_mean_18 = torch.ops.aten.var_mean.correction(add_109, [2], correction = 0, keepdim = True)
    getitem_36: "f32[1, 512, 1]" = var_mean_18[0]
    getitem_37: "f32[1, 512, 1]" = var_mean_18[1];  var_mean_18 = None
    add_110: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
    rsqrt_18: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
    sub_37: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_109, getitem_37);  add_109 = getitem_37 = None
    mul_73: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_18);  sub_37 = rsqrt_18 = None
    mul_74: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_73, arg213_1);  mul_73 = arg213_1 = None
    add_111: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_74, arg214_1);  mul_74 = arg214_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_324: "f32[512, 768]" = torch.ops.aten.view.default(add_111, [512, 768])
    permute_171: "f32[768, 384]" = torch.ops.aten.permute.default(arg215_1, [1, 0]);  arg215_1 = None
    addmm_63: "f32[512, 384]" = torch.ops.aten.addmm.default(arg216_1, view_324, permute_171);  arg216_1 = view_324 = permute_171 = None
    view_325: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_63, [1, 512, 384]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_326: "f32[512, 768]" = torch.ops.aten.view.default(add_111, [512, 768])
    permute_172: "f32[768, 384]" = torch.ops.aten.permute.default(arg217_1, [1, 0]);  arg217_1 = None
    addmm_64: "f32[512, 384]" = torch.ops.aten.addmm.default(arg218_1, view_326, permute_172);  arg218_1 = view_326 = permute_172 = None
    view_327: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_64, [1, 512, 384]);  addmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_328: "f32[512, 768]" = torch.ops.aten.view.default(add_111, [512, 768])
    permute_173: "f32[768, 384]" = torch.ops.aten.permute.default(arg219_1, [1, 0]);  arg219_1 = None
    addmm_65: "f32[512, 384]" = torch.ops.aten.addmm.default(arg220_1, view_328, permute_173);  arg220_1 = view_328 = permute_173 = None
    view_329: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_65, [1, 512, 384]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_174: "f32[1, 768, 512]" = torch.ops.aten.permute.default(add_111, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_18: "f32[1, 768, 512]" = torch.ops.aten.convolution.default(permute_174, arg221_1, None, [1], [4], [1], False, [0], 768);  permute_174 = arg221_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_19: "f32[1, 384, 512]" = torch.ops.aten.convolution.default(convolution_18, arg222_1, None, [1], [0], [1], False, [0], 1);  convolution_18 = arg222_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_112: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_19, arg9_1);  convolution_19 = arg9_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_330: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_325, [1, 512, 6, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_176: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_330, [0, 2, 1, 3]);  view_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_331: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_327, [1, 512, 6, 64]);  view_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_177: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_331, [0, 2, 1, 3]);  view_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_332: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_329, [1, 512, 6, 64]);  view_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_178: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_332, [0, 2, 1, 3]);  view_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_179: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_112, [0, 2, 1]);  add_112 = None
    mul_75: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(permute_179, view_325);  permute_179 = view_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    permute_180: "f32[384, 54]" = torch.ops.aten.permute.default(arg223_1, [1, 0]);  arg223_1 = None
    view_333: "f32[512, 384]" = torch.ops.aten.view.default(mul_75, [512, 384]);  mul_75 = None
    mm_9: "f32[512, 54]" = torch.ops.aten.mm.default(view_333, permute_180);  view_333 = permute_180 = None
    view_334: "f32[1, 512, 54]" = torch.ops.aten.view.default(mm_9, [1, 512, 54]);  mm_9 = None
    add_113: "f32[1, 512, 54]" = torch.ops.aten.add.Tensor(view_334, arg224_1);  view_334 = arg224_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_335: "f32[3072, 9, 1]" = torch.ops.aten.view.default(add_113, [-1, 9, 1]);  add_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    amax_18: "f32[3072, 1, 1]" = torch.ops.aten.amax.default(view_335, [1], True)
    sub_38: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(view_335, amax_18);  view_335 = amax_18 = None
    exp_18: "f32[3072, 9, 1]" = torch.ops.aten.exp.default(sub_38);  sub_38 = None
    sum_19: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [1], True)
    div_27: "f32[3072, 9, 1]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_336: "f32[512, 768]" = torch.ops.aten.view.default(add_111, [512, 768])
    permute_181: "f32[768, 384]" = torch.ops.aten.permute.default(arg225_1, [1, 0]);  arg225_1 = None
    addmm_66: "f32[512, 384]" = torch.ops.aten.addmm.default(arg226_1, view_336, permute_181);  arg226_1 = view_336 = permute_181 = None
    view_337: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_66, [1, 512, 384]);  addmm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_338: "f32[1, 512, 384]" = torch.ops.aten.view.default(view_337, [1, -1, 384]);  view_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    permute_182: "f32[1, 384, 512]" = torch.ops.aten.permute.default(view_338, [0, 2, 1]);  view_338 = None
    clone_55: "f32[1, 384, 512]" = torch.ops.aten.clone.default(permute_182, memory_format = torch.contiguous_format);  permute_182 = None
    unsqueeze_65: "f32[1, 384, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_55, -1);  clone_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    iota_36: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_66: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota_36, 0);  iota_36 = None
    iota_37: "i64[9]" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_67: "i64[9, 1]" = torch.ops.aten.unsqueeze.default(iota_37, -1);  iota_37 = None
    add_114: "i64[9, 512]" = torch.ops.aten.add.Tensor(unsqueeze_66, unsqueeze_67);  unsqueeze_66 = unsqueeze_67 = None
    iota_38: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_68: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_38, 0);  iota_38 = None
    iota_39: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_69: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_39, -1);  iota_39 = None
    add_115: "i64[1, 1]" = torch.ops.aten.add.Tensor(unsqueeze_68, unsqueeze_69);  unsqueeze_68 = unsqueeze_69 = None
    constant_pad_nd_9: "f32[1, 384, 520, 1]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_65, [0, 0, 4, 4], 0.0);  unsqueeze_65 = None
    unsqueeze_70: "i64[9, 512, 1]" = torch.ops.aten.unsqueeze.default(add_114, -1);  add_114 = None
    unsqueeze_71: "i64[9, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    slice_23: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(constant_pad_nd_9, 0, 0, 9223372036854775807);  constant_pad_nd_9 = None
    slice_24: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(slice_23, 1, 0, 9223372036854775807);  slice_23 = None
    index_9: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.index.Tensor(slice_24, [None, None, unsqueeze_71, add_115]);  slice_24 = unsqueeze_71 = add_115 = None
    permute_183: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.permute.default(index_9, [0, 1, 2, 4, 3, 5]);  index_9 = None
    view_339: "f32[1, 3456, 512]" = torch.ops.aten.view.default(permute_183, [1, 3456, 512]);  permute_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    permute_184: "f32[1, 512, 3456]" = torch.ops.aten.permute.default(view_339, [0, 2, 1]);  view_339 = None
    view_340: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(permute_184, [1, 512, 384, 9]);  permute_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    clone_56: "f32[1, 512, 384, 9]" = torch.ops.aten.clone.default(view_340, memory_format = torch.contiguous_format);  view_340 = None
    view_341: "f32[3072, 64, 9]" = torch.ops.aten.view.default(clone_56, [3072, 64, 9]);  clone_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    expand_55: "f32[3072, 64, 9]" = torch.ops.aten.expand.default(view_341, [3072, 64, 9]);  view_341 = None
    view_342: "f32[3072, 64, 9]" = torch.ops.aten.view.default(expand_55, [3072, 64, 9]);  expand_55 = None
    expand_56: "f32[3072, 9, 1]" = torch.ops.aten.expand.default(div_27, [3072, 9, 1]);  div_27 = None
    view_343: "f32[3072, 9, 1]" = torch.ops.aten.view.default(expand_56, [3072, 9, 1]);  expand_56 = None
    bmm_27: "f32[3072, 64, 1]" = torch.ops.aten.bmm.default(view_342, view_343);  view_342 = view_343 = None
    view_344: "f32[3072, 64, 1]" = torch.ops.aten.view.default(bmm_27, [3072, 64, 1]);  bmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    view_345: "f32[512, 384]" = torch.ops.aten.view.default(view_344, [-1, 384]);  view_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_185: "f32[1, 6, 64, 512]" = torch.ops.aten.permute.default(permute_177, [0, 1, 3, 2]);  permute_177 = None
    expand_57: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_176, [1, 6, 512, 64]);  permute_176 = None
    view_346: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_57, [6, 512, 64]);  expand_57 = None
    expand_58: "f32[1, 6, 64, 512]" = torch.ops.aten.expand.default(permute_185, [1, 6, 64, 512]);  permute_185 = None
    view_347: "f32[6, 64, 512]" = torch.ops.aten.view.default(expand_58, [6, 64, 512]);  expand_58 = None
    bmm_28: "f32[6, 512, 512]" = torch.ops.aten.bmm.default(view_346, view_347);  view_346 = view_347 = None
    view_348: "f32[1, 6, 512, 512]" = torch.ops.aten.view.default(bmm_28, [1, 6, 512, 512]);  bmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_28: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(view_348, 8.0);  view_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:397, code: attention_scores = attention_scores + attention_mask
    add_116: "f32[1, 6, 512, 512]" = torch.ops.aten.add.Tensor(div_28, mul);  div_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_19: "f32[1, 6, 512, 1]" = torch.ops.aten.amax.default(add_116, [-1], True)
    sub_39: "f32[1, 6, 512, 512]" = torch.ops.aten.sub.Tensor(add_116, amax_19);  add_116 = amax_19 = None
    exp_19: "f32[1, 6, 512, 512]" = torch.ops.aten.exp.default(sub_39);  sub_39 = None
    sum_20: "f32[1, 6, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
    div_29: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    clone_57: "f32[1, 6, 512, 512]" = torch.ops.aten.clone.default(div_29);  div_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_59: "f32[1, 6, 512, 512]" = torch.ops.aten.expand.default(clone_57, [1, 6, 512, 512]);  clone_57 = None
    view_349: "f32[6, 512, 512]" = torch.ops.aten.view.default(expand_59, [6, 512, 512]);  expand_59 = None
    expand_60: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_178, [1, 6, 512, 64]);  permute_178 = None
    view_350: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_60, [6, 512, 64]);  expand_60 = None
    bmm_29: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(view_349, view_350);  view_349 = view_350 = None
    view_351: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_29, [1, 6, 512, 64]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_186: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_351, [0, 2, 1, 3]);  view_351 = None
    clone_58: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_186, memory_format = torch.contiguous_format);  permute_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_352: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_345, [1, -1, 6, 64]);  view_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    cat_9: "f32[1, 512, 12, 64]" = torch.ops.aten.cat.default([clone_58, view_352], 2);  clone_58 = view_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_353: "f32[1, 512, 768]" = torch.ops.aten.view.default(cat_9, [1, 512, 768]);  cat_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_354: "f32[512, 768]" = torch.ops.aten.view.default(view_353, [512, 768]);  view_353 = None
    permute_187: "f32[768, 768]" = torch.ops.aten.permute.default(arg227_1, [1, 0]);  arg227_1 = None
    addmm_67: "f32[512, 768]" = torch.ops.aten.addmm.default(arg228_1, view_354, permute_187);  arg228_1 = view_354 = permute_187 = None
    view_355: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_67, [1, 512, 768]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    clone_59: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_355);  view_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_117: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_59, add_111);  clone_59 = add_111 = None
    var_mean_19 = torch.ops.aten.var_mean.correction(add_117, [2], correction = 0, keepdim = True)
    getitem_38: "f32[1, 512, 1]" = var_mean_19[0]
    getitem_39: "f32[1, 512, 1]" = var_mean_19[1];  var_mean_19 = None
    add_118: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
    rsqrt_19: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
    sub_40: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_117, getitem_39);  add_117 = getitem_39 = None
    mul_76: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_19);  sub_40 = rsqrt_19 = None
    mul_77: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_76, arg229_1);  mul_76 = arg229_1 = None
    add_119: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_77, arg230_1);  mul_77 = arg230_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_356: "f32[512, 768]" = torch.ops.aten.view.default(add_119, [512, 768])
    permute_188: "f32[768, 3072]" = torch.ops.aten.permute.default(arg231_1, [1, 0]);  arg231_1 = None
    addmm_68: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg232_1, view_356, permute_188);  arg232_1 = view_356 = permute_188 = None
    view_357: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_68, [1, 512, 3072]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_78: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_357, 0.5)
    mul_79: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_357, 0.7071067811865476);  view_357 = None
    erf_9: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_79);  mul_79 = None
    add_120: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_80: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_78, add_120);  mul_78 = add_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_358: "f32[512, 3072]" = torch.ops.aten.view.default(mul_80, [512, 3072]);  mul_80 = None
    permute_189: "f32[3072, 768]" = torch.ops.aten.permute.default(arg233_1, [1, 0]);  arg233_1 = None
    addmm_69: "f32[512, 768]" = torch.ops.aten.addmm.default(arg234_1, view_358, permute_189);  arg234_1 = view_358 = permute_189 = None
    view_359: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_69, [1, 512, 768]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    clone_60: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_359);  view_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_121: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_60, add_119);  clone_60 = add_119 = None
    var_mean_20 = torch.ops.aten.var_mean.correction(add_121, [2], correction = 0, keepdim = True)
    getitem_40: "f32[1, 512, 1]" = var_mean_20[0]
    getitem_41: "f32[1, 512, 1]" = var_mean_20[1];  var_mean_20 = None
    add_122: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
    rsqrt_20: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
    sub_41: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_121, getitem_41);  add_121 = getitem_41 = None
    mul_81: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_20);  sub_41 = rsqrt_20 = None
    mul_82: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_81, arg235_1);  mul_81 = arg235_1 = None
    add_123: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_82, arg236_1);  mul_82 = arg236_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_360: "f32[512, 768]" = torch.ops.aten.view.default(add_123, [512, 768])
    permute_190: "f32[768, 384]" = torch.ops.aten.permute.default(arg237_1, [1, 0]);  arg237_1 = None
    addmm_70: "f32[512, 384]" = torch.ops.aten.addmm.default(arg238_1, view_360, permute_190);  arg238_1 = view_360 = permute_190 = None
    view_361: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_70, [1, 512, 384]);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_362: "f32[512, 768]" = torch.ops.aten.view.default(add_123, [512, 768])
    permute_191: "f32[768, 384]" = torch.ops.aten.permute.default(arg239_1, [1, 0]);  arg239_1 = None
    addmm_71: "f32[512, 384]" = torch.ops.aten.addmm.default(arg240_1, view_362, permute_191);  arg240_1 = view_362 = permute_191 = None
    view_363: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_71, [1, 512, 384]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_364: "f32[512, 768]" = torch.ops.aten.view.default(add_123, [512, 768])
    permute_192: "f32[768, 384]" = torch.ops.aten.permute.default(arg241_1, [1, 0]);  arg241_1 = None
    addmm_72: "f32[512, 384]" = torch.ops.aten.addmm.default(arg242_1, view_364, permute_192);  arg242_1 = view_364 = permute_192 = None
    view_365: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_72, [1, 512, 384]);  addmm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_193: "f32[1, 768, 512]" = torch.ops.aten.permute.default(add_123, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_20: "f32[1, 768, 512]" = torch.ops.aten.convolution.default(permute_193, arg243_1, None, [1], [4], [1], False, [0], 768);  permute_193 = arg243_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_21: "f32[1, 384, 512]" = torch.ops.aten.convolution.default(convolution_20, arg244_1, None, [1], [0], [1], False, [0], 1);  convolution_20 = arg244_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_124: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_21, arg10_1);  convolution_21 = arg10_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_366: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_361, [1, 512, 6, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_195: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_366, [0, 2, 1, 3]);  view_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_367: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_363, [1, 512, 6, 64]);  view_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_196: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_367, [0, 2, 1, 3]);  view_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_368: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_365, [1, 512, 6, 64]);  view_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_197: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_368, [0, 2, 1, 3]);  view_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_198: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_124, [0, 2, 1]);  add_124 = None
    mul_83: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(permute_198, view_361);  permute_198 = view_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    permute_199: "f32[384, 54]" = torch.ops.aten.permute.default(arg245_1, [1, 0]);  arg245_1 = None
    view_369: "f32[512, 384]" = torch.ops.aten.view.default(mul_83, [512, 384]);  mul_83 = None
    mm_10: "f32[512, 54]" = torch.ops.aten.mm.default(view_369, permute_199);  view_369 = permute_199 = None
    view_370: "f32[1, 512, 54]" = torch.ops.aten.view.default(mm_10, [1, 512, 54]);  mm_10 = None
    add_125: "f32[1, 512, 54]" = torch.ops.aten.add.Tensor(view_370, arg246_1);  view_370 = arg246_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_371: "f32[3072, 9, 1]" = torch.ops.aten.view.default(add_125, [-1, 9, 1]);  add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    amax_20: "f32[3072, 1, 1]" = torch.ops.aten.amax.default(view_371, [1], True)
    sub_42: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(view_371, amax_20);  view_371 = amax_20 = None
    exp_20: "f32[3072, 9, 1]" = torch.ops.aten.exp.default(sub_42);  sub_42 = None
    sum_21: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [1], True)
    div_30: "f32[3072, 9, 1]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_372: "f32[512, 768]" = torch.ops.aten.view.default(add_123, [512, 768])
    permute_200: "f32[768, 384]" = torch.ops.aten.permute.default(arg247_1, [1, 0]);  arg247_1 = None
    addmm_73: "f32[512, 384]" = torch.ops.aten.addmm.default(arg248_1, view_372, permute_200);  arg248_1 = view_372 = permute_200 = None
    view_373: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_73, [1, 512, 384]);  addmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_374: "f32[1, 512, 384]" = torch.ops.aten.view.default(view_373, [1, -1, 384]);  view_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    permute_201: "f32[1, 384, 512]" = torch.ops.aten.permute.default(view_374, [0, 2, 1]);  view_374 = None
    clone_61: "f32[1, 384, 512]" = torch.ops.aten.clone.default(permute_201, memory_format = torch.contiguous_format);  permute_201 = None
    unsqueeze_72: "f32[1, 384, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_61, -1);  clone_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    iota_40: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_73: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota_40, 0);  iota_40 = None
    iota_41: "i64[9]" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_74: "i64[9, 1]" = torch.ops.aten.unsqueeze.default(iota_41, -1);  iota_41 = None
    add_126: "i64[9, 512]" = torch.ops.aten.add.Tensor(unsqueeze_73, unsqueeze_74);  unsqueeze_73 = unsqueeze_74 = None
    iota_42: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_75: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_42, 0);  iota_42 = None
    iota_43: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_76: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_43, -1);  iota_43 = None
    add_127: "i64[1, 1]" = torch.ops.aten.add.Tensor(unsqueeze_75, unsqueeze_76);  unsqueeze_75 = unsqueeze_76 = None
    constant_pad_nd_10: "f32[1, 384, 520, 1]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_72, [0, 0, 4, 4], 0.0);  unsqueeze_72 = None
    unsqueeze_77: "i64[9, 512, 1]" = torch.ops.aten.unsqueeze.default(add_126, -1);  add_126 = None
    unsqueeze_78: "i64[9, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_77, -1);  unsqueeze_77 = None
    slice_25: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(constant_pad_nd_10, 0, 0, 9223372036854775807);  constant_pad_nd_10 = None
    slice_26: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(slice_25, 1, 0, 9223372036854775807);  slice_25 = None
    index_10: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.index.Tensor(slice_26, [None, None, unsqueeze_78, add_127]);  slice_26 = unsqueeze_78 = add_127 = None
    permute_202: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.permute.default(index_10, [0, 1, 2, 4, 3, 5]);  index_10 = None
    view_375: "f32[1, 3456, 512]" = torch.ops.aten.view.default(permute_202, [1, 3456, 512]);  permute_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    permute_203: "f32[1, 512, 3456]" = torch.ops.aten.permute.default(view_375, [0, 2, 1]);  view_375 = None
    view_376: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(permute_203, [1, 512, 384, 9]);  permute_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    clone_62: "f32[1, 512, 384, 9]" = torch.ops.aten.clone.default(view_376, memory_format = torch.contiguous_format);  view_376 = None
    view_377: "f32[3072, 64, 9]" = torch.ops.aten.view.default(clone_62, [3072, 64, 9]);  clone_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    expand_61: "f32[3072, 64, 9]" = torch.ops.aten.expand.default(view_377, [3072, 64, 9]);  view_377 = None
    view_378: "f32[3072, 64, 9]" = torch.ops.aten.view.default(expand_61, [3072, 64, 9]);  expand_61 = None
    expand_62: "f32[3072, 9, 1]" = torch.ops.aten.expand.default(div_30, [3072, 9, 1]);  div_30 = None
    view_379: "f32[3072, 9, 1]" = torch.ops.aten.view.default(expand_62, [3072, 9, 1]);  expand_62 = None
    bmm_30: "f32[3072, 64, 1]" = torch.ops.aten.bmm.default(view_378, view_379);  view_378 = view_379 = None
    view_380: "f32[3072, 64, 1]" = torch.ops.aten.view.default(bmm_30, [3072, 64, 1]);  bmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    view_381: "f32[512, 384]" = torch.ops.aten.view.default(view_380, [-1, 384]);  view_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_204: "f32[1, 6, 64, 512]" = torch.ops.aten.permute.default(permute_196, [0, 1, 3, 2]);  permute_196 = None
    expand_63: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_195, [1, 6, 512, 64]);  permute_195 = None
    view_382: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_63, [6, 512, 64]);  expand_63 = None
    expand_64: "f32[1, 6, 64, 512]" = torch.ops.aten.expand.default(permute_204, [1, 6, 64, 512]);  permute_204 = None
    view_383: "f32[6, 64, 512]" = torch.ops.aten.view.default(expand_64, [6, 64, 512]);  expand_64 = None
    bmm_31: "f32[6, 512, 512]" = torch.ops.aten.bmm.default(view_382, view_383);  view_382 = view_383 = None
    view_384: "f32[1, 6, 512, 512]" = torch.ops.aten.view.default(bmm_31, [1, 6, 512, 512]);  bmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_31: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(view_384, 8.0);  view_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:397, code: attention_scores = attention_scores + attention_mask
    add_128: "f32[1, 6, 512, 512]" = torch.ops.aten.add.Tensor(div_31, mul);  div_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_21: "f32[1, 6, 512, 1]" = torch.ops.aten.amax.default(add_128, [-1], True)
    sub_43: "f32[1, 6, 512, 512]" = torch.ops.aten.sub.Tensor(add_128, amax_21);  add_128 = amax_21 = None
    exp_21: "f32[1, 6, 512, 512]" = torch.ops.aten.exp.default(sub_43);  sub_43 = None
    sum_22: "f32[1, 6, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
    div_32: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    clone_63: "f32[1, 6, 512, 512]" = torch.ops.aten.clone.default(div_32);  div_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_65: "f32[1, 6, 512, 512]" = torch.ops.aten.expand.default(clone_63, [1, 6, 512, 512]);  clone_63 = None
    view_385: "f32[6, 512, 512]" = torch.ops.aten.view.default(expand_65, [6, 512, 512]);  expand_65 = None
    expand_66: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_197, [1, 6, 512, 64]);  permute_197 = None
    view_386: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_66, [6, 512, 64]);  expand_66 = None
    bmm_32: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(view_385, view_386);  view_385 = view_386 = None
    view_387: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_32, [1, 6, 512, 64]);  bmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_205: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_387, [0, 2, 1, 3]);  view_387 = None
    clone_64: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_388: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_381, [1, -1, 6, 64]);  view_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    cat_10: "f32[1, 512, 12, 64]" = torch.ops.aten.cat.default([clone_64, view_388], 2);  clone_64 = view_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_389: "f32[1, 512, 768]" = torch.ops.aten.view.default(cat_10, [1, 512, 768]);  cat_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_390: "f32[512, 768]" = torch.ops.aten.view.default(view_389, [512, 768]);  view_389 = None
    permute_206: "f32[768, 768]" = torch.ops.aten.permute.default(arg249_1, [1, 0]);  arg249_1 = None
    addmm_74: "f32[512, 768]" = torch.ops.aten.addmm.default(arg250_1, view_390, permute_206);  arg250_1 = view_390 = permute_206 = None
    view_391: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_74, [1, 512, 768]);  addmm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    clone_65: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_391);  view_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_129: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_65, add_123);  clone_65 = add_123 = None
    var_mean_21 = torch.ops.aten.var_mean.correction(add_129, [2], correction = 0, keepdim = True)
    getitem_42: "f32[1, 512, 1]" = var_mean_21[0]
    getitem_43: "f32[1, 512, 1]" = var_mean_21[1];  var_mean_21 = None
    add_130: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
    rsqrt_21: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
    sub_44: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_129, getitem_43);  add_129 = getitem_43 = None
    mul_84: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_21);  sub_44 = rsqrt_21 = None
    mul_85: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_84, arg251_1);  mul_84 = arg251_1 = None
    add_131: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_85, arg252_1);  mul_85 = arg252_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_392: "f32[512, 768]" = torch.ops.aten.view.default(add_131, [512, 768])
    permute_207: "f32[768, 3072]" = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
    addmm_75: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg254_1, view_392, permute_207);  arg254_1 = view_392 = permute_207 = None
    view_393: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_75, [1, 512, 3072]);  addmm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_86: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_393, 0.5)
    mul_87: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_393, 0.7071067811865476);  view_393 = None
    erf_10: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_87);  mul_87 = None
    add_132: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_88: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_86, add_132);  mul_86 = add_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_394: "f32[512, 3072]" = torch.ops.aten.view.default(mul_88, [512, 3072]);  mul_88 = None
    permute_208: "f32[3072, 768]" = torch.ops.aten.permute.default(arg255_1, [1, 0]);  arg255_1 = None
    addmm_76: "f32[512, 768]" = torch.ops.aten.addmm.default(arg256_1, view_394, permute_208);  arg256_1 = view_394 = permute_208 = None
    view_395: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_76, [1, 512, 768]);  addmm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    clone_66: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_395);  view_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_133: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_66, add_131);  clone_66 = add_131 = None
    var_mean_22 = torch.ops.aten.var_mean.correction(add_133, [2], correction = 0, keepdim = True)
    getitem_44: "f32[1, 512, 1]" = var_mean_22[0]
    getitem_45: "f32[1, 512, 1]" = var_mean_22[1];  var_mean_22 = None
    add_134: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
    rsqrt_22: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
    sub_45: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_133, getitem_45);  add_133 = getitem_45 = None
    mul_89: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_22);  sub_45 = rsqrt_22 = None
    mul_90: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_89, arg257_1);  mul_89 = arg257_1 = None
    add_135: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_90, arg258_1);  mul_90 = arg258_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:351, code: mixed_query_layer = self.query(hidden_states)
    view_396: "f32[512, 768]" = torch.ops.aten.view.default(add_135, [512, 768])
    permute_209: "f32[768, 384]" = torch.ops.aten.permute.default(arg259_1, [1, 0]);  arg259_1 = None
    addmm_77: "f32[512, 384]" = torch.ops.aten.addmm.default(arg260_1, view_396, permute_209);  arg260_1 = view_396 = permute_209 = None
    view_397: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_77, [1, 512, 384]);  addmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:360, code: mixed_key_layer = self.key(hidden_states)
    view_398: "f32[512, 768]" = torch.ops.aten.view.default(add_135, [512, 768])
    permute_210: "f32[768, 384]" = torch.ops.aten.permute.default(arg261_1, [1, 0]);  arg261_1 = None
    addmm_78: "f32[512, 384]" = torch.ops.aten.addmm.default(arg262_1, view_398, permute_210);  arg262_1 = view_398 = permute_210 = None
    view_399: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_78, [1, 512, 384]);  addmm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:361, code: mixed_value_layer = self.value(hidden_states)
    view_400: "f32[512, 768]" = torch.ops.aten.view.default(add_135, [512, 768])
    permute_211: "f32[768, 384]" = torch.ops.aten.permute.default(arg263_1, [1, 0]);  arg263_1 = None
    addmm_79: "f32[512, 384]" = torch.ops.aten.addmm.default(arg264_1, view_400, permute_211);  arg264_1 = view_400 = permute_211 = None
    view_401: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_79, [1, 512, 384]);  addmm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:363, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
    permute_212: "f32[1, 768, 512]" = torch.ops.aten.permute.default(add_135, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:292, code: x = self.depthwise(hidden_states)
    convolution_22: "f32[1, 768, 512]" = torch.ops.aten.convolution.default(permute_212, arg265_1, None, [1], [4], [1], False, [0], 768);  permute_212 = arg265_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:293, code: x = self.pointwise(x)
    convolution_23: "f32[1, 384, 512]" = torch.ops.aten.convolution.default(convolution_22, arg266_1, None, [1], [0], [1], False, [0], 1);  convolution_22 = arg266_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:294, code: x += self.bias
    add_136: "f32[1, 384, 512]" = torch.ops.aten.add.Tensor(convolution_23, arg11_1);  convolution_23 = arg11_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_402: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_397, [1, 512, 6, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_214: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_402, [0, 2, 1, 3]);  view_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_403: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_399, [1, 512, 6, 64]);  view_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_215: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_403, [0, 2, 1, 3]);  view_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:340, code: x = x.view(*new_x_shape)
    view_404: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_401, [1, 512, 6, 64]);  view_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:341, code: return x.permute(0, 2, 1, 3)
    permute_216: "f32[1, 6, 512, 64]" = torch.ops.aten.permute.default(view_404, [0, 2, 1, 3]);  view_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:369, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
    permute_217: "f32[1, 512, 384]" = torch.ops.aten.permute.default(add_136, [0, 2, 1]);  add_136 = None
    mul_91: "f32[1, 512, 384]" = torch.ops.aten.mul.Tensor(permute_217, view_397);  permute_217 = view_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:371, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
    permute_218: "f32[384, 54]" = torch.ops.aten.permute.default(arg267_1, [1, 0]);  arg267_1 = None
    view_405: "f32[512, 384]" = torch.ops.aten.view.default(mul_91, [512, 384]);  mul_91 = None
    mm_11: "f32[512, 54]" = torch.ops.aten.mm.default(view_405, permute_218);  view_405 = permute_218 = None
    view_406: "f32[1, 512, 54]" = torch.ops.aten.view.default(mm_11, [1, 512, 54]);  mm_11 = None
    add_137: "f32[1, 512, 54]" = torch.ops.aten.add.Tensor(view_406, arg268_1);  view_406 = arg268_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:372, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
    view_407: "f32[3072, 9, 1]" = torch.ops.aten.view.default(add_137, [-1, 9, 1]);  add_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:373, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
    amax_22: "f32[3072, 1, 1]" = torch.ops.aten.amax.default(view_407, [1], True)
    sub_46: "f32[3072, 9, 1]" = torch.ops.aten.sub.Tensor(view_407, amax_22);  view_407 = amax_22 = None
    exp_22: "f32[3072, 9, 1]" = torch.ops.aten.exp.default(sub_46);  sub_46 = None
    sum_23: "f32[3072, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [1], True)
    div_33: "f32[3072, 9, 1]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:375, code: conv_out_layer = self.conv_out_layer(hidden_states)
    view_408: "f32[512, 768]" = torch.ops.aten.view.default(add_135, [512, 768])
    permute_219: "f32[768, 384]" = torch.ops.aten.permute.default(arg269_1, [1, 0]);  arg269_1 = None
    addmm_80: "f32[512, 384]" = torch.ops.aten.addmm.default(arg270_1, view_408, permute_219);  arg270_1 = view_408 = permute_219 = None
    view_409: "f32[1, 512, 384]" = torch.ops.aten.view.default(addmm_80, [1, 512, 384]);  addmm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:376, code: conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
    view_410: "f32[1, 512, 384]" = torch.ops.aten.view.default(view_409, [1, -1, 384]);  view_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:377, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
    permute_220: "f32[1, 384, 512]" = torch.ops.aten.permute.default(view_410, [0, 2, 1]);  view_410 = None
    clone_67: "f32[1, 384, 512]" = torch.ops.aten.clone.default(permute_220, memory_format = torch.contiguous_format);  permute_220 = None
    unsqueeze_79: "f32[1, 384, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_67, -1);  clone_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:378, code: conv_out_layer = nn.functional.unfold(
    iota_44: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_80: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota_44, 0);  iota_44 = None
    iota_45: "i64[9]" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_81: "i64[9, 1]" = torch.ops.aten.unsqueeze.default(iota_45, -1);  iota_45 = None
    add_138: "i64[9, 512]" = torch.ops.aten.add.Tensor(unsqueeze_80, unsqueeze_81);  unsqueeze_80 = unsqueeze_81 = None
    iota_46: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_82: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_46, 0);  iota_46 = None
    iota_47: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_83: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_47, -1);  iota_47 = None
    add_139: "i64[1, 1]" = torch.ops.aten.add.Tensor(unsqueeze_82, unsqueeze_83);  unsqueeze_82 = unsqueeze_83 = None
    constant_pad_nd_11: "f32[1, 384, 520, 1]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_79, [0, 0, 4, 4], 0.0);  unsqueeze_79 = None
    unsqueeze_84: "i64[9, 512, 1]" = torch.ops.aten.unsqueeze.default(add_138, -1);  add_138 = None
    unsqueeze_85: "i64[9, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    slice_27: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(constant_pad_nd_11, 0, 0, 9223372036854775807);  constant_pad_nd_11 = None
    slice_28: "f32[1, 384, 520, 1]" = torch.ops.aten.slice.Tensor(slice_27, 1, 0, 9223372036854775807);  slice_27 = None
    index_11: "f32[1, 384, 9, 512, 1, 1]" = torch.ops.aten.index.Tensor(slice_28, [None, None, unsqueeze_85, add_139]);  slice_28 = unsqueeze_85 = add_139 = None
    permute_221: "f32[1, 384, 9, 1, 512, 1]" = torch.ops.aten.permute.default(index_11, [0, 1, 2, 4, 3, 5]);  index_11 = None
    view_411: "f32[1, 3456, 512]" = torch.ops.aten.view.default(permute_221, [1, 3456, 512]);  permute_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:385, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
    permute_222: "f32[1, 512, 3456]" = torch.ops.aten.permute.default(view_411, [0, 2, 1]);  view_411 = None
    view_412: "f32[1, 512, 384, 9]" = torch.ops.aten.view.default(permute_222, [1, 512, 384, 9]);  permute_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:388, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
    clone_68: "f32[1, 512, 384, 9]" = torch.ops.aten.clone.default(view_412, memory_format = torch.contiguous_format);  view_412 = None
    view_413: "f32[3072, 64, 9]" = torch.ops.aten.view.default(clone_68, [3072, 64, 9]);  clone_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:389, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
    expand_67: "f32[3072, 64, 9]" = torch.ops.aten.expand.default(view_413, [3072, 64, 9]);  view_413 = None
    view_414: "f32[3072, 64, 9]" = torch.ops.aten.view.default(expand_67, [3072, 64, 9]);  expand_67 = None
    expand_68: "f32[3072, 9, 1]" = torch.ops.aten.expand.default(div_33, [3072, 9, 1]);  div_33 = None
    view_415: "f32[3072, 9, 1]" = torch.ops.aten.view.default(expand_68, [3072, 9, 1]);  expand_68 = None
    bmm_33: "f32[3072, 64, 1]" = torch.ops.aten.bmm.default(view_414, view_415);  view_414 = view_415 = None
    view_416: "f32[3072, 64, 1]" = torch.ops.aten.view.default(bmm_33, [3072, 64, 1]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:390, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
    view_417: "f32[512, 384]" = torch.ops.aten.view.default(view_416, [-1, 384]);  view_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:393, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_223: "f32[1, 6, 64, 512]" = torch.ops.aten.permute.default(permute_215, [0, 1, 3, 2]);  permute_215 = None
    expand_69: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_214, [1, 6, 512, 64]);  permute_214 = None
    view_418: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_69, [6, 512, 64]);  expand_69 = None
    expand_70: "f32[1, 6, 64, 512]" = torch.ops.aten.expand.default(permute_223, [1, 6, 64, 512]);  permute_223 = None
    view_419: "f32[6, 64, 512]" = torch.ops.aten.view.default(expand_70, [6, 64, 512]);  expand_70 = None
    bmm_34: "f32[6, 512, 512]" = torch.ops.aten.bmm.default(view_418, view_419);  view_418 = view_419 = None
    view_420: "f32[1, 6, 512, 512]" = torch.ops.aten.view.default(bmm_34, [1, 6, 512, 512]);  bmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:394, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_34: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(view_420, 8.0);  view_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:397, code: attention_scores = attention_scores + attention_mask
    add_140: "f32[1, 6, 512, 512]" = torch.ops.aten.add.Tensor(div_34, mul);  div_34 = mul = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:400, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_23: "f32[1, 6, 512, 1]" = torch.ops.aten.amax.default(add_140, [-1], True)
    sub_47: "f32[1, 6, 512, 512]" = torch.ops.aten.sub.Tensor(add_140, amax_23);  add_140 = amax_23 = None
    exp_23: "f32[1, 6, 512, 512]" = torch.ops.aten.exp.default(sub_47);  sub_47 = None
    sum_24: "f32[1, 6, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
    div_35: "f32[1, 6, 512, 512]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:404, code: attention_probs = self.dropout(attention_probs)
    clone_69: "f32[1, 6, 512, 512]" = torch.ops.aten.clone.default(div_35);  div_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:410, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_71: "f32[1, 6, 512, 512]" = torch.ops.aten.expand.default(clone_69, [1, 6, 512, 512]);  clone_69 = None
    view_421: "f32[6, 512, 512]" = torch.ops.aten.view.default(expand_71, [6, 512, 512]);  expand_71 = None
    expand_72: "f32[1, 6, 512, 64]" = torch.ops.aten.expand.default(permute_216, [1, 6, 512, 64]);  permute_216 = None
    view_422: "f32[6, 512, 64]" = torch.ops.aten.view.default(expand_72, [6, 512, 64]);  expand_72 = None
    bmm_35: "f32[6, 512, 64]" = torch.ops.aten.bmm.default(view_421, view_422);  view_421 = view_422 = None
    view_423: "f32[1, 6, 512, 64]" = torch.ops.aten.view.default(bmm_35, [1, 6, 512, 64]);  bmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:411, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_224: "f32[1, 512, 6, 64]" = torch.ops.aten.permute.default(view_423, [0, 2, 1, 3]);  view_423 = None
    clone_70: "f32[1, 512, 6, 64]" = torch.ops.aten.clone.default(permute_224, memory_format = torch.contiguous_format);  permute_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:413, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
    view_424: "f32[1, 512, 6, 64]" = torch.ops.aten.view.default(view_417, [1, -1, 6, 64]);  view_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:414, code: context_layer = torch.cat([context_layer, conv_out], 2)
    cat_11: "f32[1, 512, 12, 64]" = torch.ops.aten.cat.default([clone_70, view_424], 2);  clone_70 = view_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:420, code: context_layer = context_layer.view(*new_context_layer_shape)
    view_425: "f32[1, 512, 768]" = torch.ops.aten.view.default(cat_11, [1, 512, 768]);  cat_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:434, code: hidden_states = self.dense(hidden_states)
    view_426: "f32[512, 768]" = torch.ops.aten.view.default(view_425, [512, 768]);  view_425 = None
    permute_225: "f32[768, 768]" = torch.ops.aten.permute.default(arg271_1, [1, 0]);  arg271_1 = None
    addmm_81: "f32[512, 768]" = torch.ops.aten.addmm.default(arg272_1, view_426, permute_225);  arg272_1 = view_426 = permute_225 = None
    view_427: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_81, [1, 512, 768]);  addmm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:435, code: hidden_states = self.dropout(hidden_states)
    clone_71: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_427);  view_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:436, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_141: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_71, add_135);  clone_71 = add_135 = None
    var_mean_23 = torch.ops.aten.var_mean.correction(add_141, [2], correction = 0, keepdim = True)
    getitem_46: "f32[1, 512, 1]" = var_mean_23[0]
    getitem_47: "f32[1, 512, 1]" = var_mean_23[1];  var_mean_23 = None
    add_142: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
    rsqrt_23: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_142);  add_142 = None
    sub_48: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_141, getitem_47);  add_141 = getitem_47 = None
    mul_92: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_23);  sub_48 = rsqrt_23 = None
    mul_93: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_92, arg273_1);  mul_92 = arg273_1 = None
    add_143: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_93, arg274_1);  mul_93 = arg274_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:522, code: hidden_states = self.dense(hidden_states)
    view_428: "f32[512, 768]" = torch.ops.aten.view.default(add_143, [512, 768])
    permute_226: "f32[768, 3072]" = torch.ops.aten.permute.default(arg275_1, [1, 0]);  arg275_1 = None
    addmm_82: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg276_1, view_428, permute_226);  arg276_1 = view_428 = permute_226 = None
    view_429: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_82, [1, 512, 3072]);  addmm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_94: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_429, 0.5)
    mul_95: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_429, 0.7071067811865476);  view_429 = None
    erf_11: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_95);  mul_95 = None
    add_144: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_96: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_94, add_144);  mul_94 = add_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:540, code: hidden_states = self.dense(hidden_states)
    view_430: "f32[512, 3072]" = torch.ops.aten.view.default(mul_96, [512, 3072]);  mul_96 = None
    permute_227: "f32[3072, 768]" = torch.ops.aten.permute.default(arg277_1, [1, 0]);  arg277_1 = None
    addmm_83: "f32[512, 768]" = torch.ops.aten.addmm.default(arg278_1, view_430, permute_227);  arg278_1 = view_430 = permute_227 = None
    view_431: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_83, [1, 512, 768]);  addmm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:541, code: hidden_states = self.dropout(hidden_states)
    clone_72: "f32[1, 512, 768]" = torch.ops.aten.clone.default(view_431);  view_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:542, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_145: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(clone_72, add_143);  clone_72 = add_143 = None
    var_mean_24 = torch.ops.aten.var_mean.correction(add_145, [2], correction = 0, keepdim = True)
    getitem_48: "f32[1, 512, 1]" = var_mean_24[0]
    getitem_49: "f32[1, 512, 1]" = var_mean_24[1];  var_mean_24 = None
    add_146: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
    rsqrt_24: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
    sub_49: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_145, getitem_49);  add_145 = getitem_49 = None
    mul_97: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_24);  sub_49 = rsqrt_24 = None
    mul_98: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_97, arg279_1);  mul_97 = arg279_1 = None
    add_147: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_98, arg280_1);  mul_98 = arg280_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:873, code: hidden_states = self.dense(generator_hidden_states)
    view_432: "f32[512, 768]" = torch.ops.aten.view.default(add_147, [512, 768]);  add_147 = None
    permute_228: "f32[768, 768]" = torch.ops.aten.permute.default(arg281_1, [1, 0]);  arg281_1 = None
    addmm_84: "f32[512, 768]" = torch.ops.aten.addmm.default(arg282_1, view_432, permute_228);  arg282_1 = view_432 = permute_228 = None
    view_433: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_84, [1, 512, 768]);  addmm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_99: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_433, 0.5)
    mul_100: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_433, 0.7071067811865476);  view_433 = None
    erf_12: "f32[1, 512, 768]" = torch.ops.aten.erf.default(mul_100);  mul_100 = None
    add_148: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_101: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_99, add_148);  mul_99 = add_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:875, code: hidden_states = self.LayerNorm(hidden_states)
    var_mean_25 = torch.ops.aten.var_mean.correction(mul_101, [2], correction = 0, keepdim = True)
    getitem_50: "f32[1, 512, 1]" = var_mean_25[0]
    getitem_51: "f32[1, 512, 1]" = var_mean_25[1];  var_mean_25 = None
    add_149: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
    rsqrt_25: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
    sub_50: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_101, getitem_51);  mul_101 = getitem_51 = None
    mul_102: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_25);  sub_50 = rsqrt_25 = None
    mul_103: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_102, arg283_1);  mul_102 = arg283_1 = None
    add_150: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_103, arg284_1);  mul_103 = arg284_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:941, code: prediction_scores = self.generator_lm_head(prediction_scores)
    view_434: "f32[512, 768]" = torch.ops.aten.view.default(add_150, [512, 768]);  add_150 = None
    permute_229: "f32[768, 30522]" = torch.ops.aten.permute.default(arg285_1, [1, 0]);  arg285_1 = None
    addmm_85: "f32[512, 30522]" = torch.ops.aten.addmm.default(arg286_1, view_434, permute_229);  arg286_1 = view_434 = permute_229 = None
    view_435: "f32[1, 512, 30522]" = torch.ops.aten.view.default(addmm_85, [1, 512, 30522]);  addmm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/convbert/modeling_convbert.py:947, code: loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    view_436: "f32[512, 30522]" = torch.ops.aten.view.default(view_435, [-1, 30522])
    view_437: "i64[512]" = torch.ops.aten.view.default(arg290_1, [-1]);  arg290_1 = None
    amax_24: "f32[512, 1]" = torch.ops.aten.amax.default(view_436, [1], True)
    sub_51: "f32[512, 30522]" = torch.ops.aten.sub.Tensor(view_436, amax_24);  view_436 = amax_24 = None
    exp_24: "f32[512, 30522]" = torch.ops.aten.exp.default(sub_51)
    sum_25: "f32[512, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
    log: "f32[512, 1]" = torch.ops.aten.log.default(sum_25);  sum_25 = None
    sub_52: "f32[512, 30522]" = torch.ops.aten.sub.Tensor(sub_51, log);  sub_51 = log = None
    ne: "b8[512]" = torch.ops.aten.ne.Scalar(view_437, -100)
    scalar_tensor: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'))
    where: "i64[512]" = torch.ops.aten.where.self(ne, view_437, scalar_tensor);  ne = scalar_tensor = None
    unsqueeze_86: "i64[512, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
    gather: "f32[512, 1]" = torch.ops.aten.gather.default(sub_52, 1, unsqueeze_86);  sub_52 = unsqueeze_86 = None
    squeeze: "f32[512]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[512]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
    ne_1: "b8[512]" = torch.ops.aten.ne.Scalar(view_437, -100)
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_1: "f32[512]" = torch.ops.aten.where.self(ne_1, neg, scalar_tensor_1);  ne_1 = neg = scalar_tensor_1 = None
    ne_2: "b8[512]" = torch.ops.aten.ne.Scalar(view_437, -100);  view_437 = None
    sum_26: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
    sum_27: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
    div_36: "f32[]" = torch.ops.aten.div.Tensor(sum_27, convert_element_type);  sum_27 = convert_element_type = None
    return (div_36, view_435)
    