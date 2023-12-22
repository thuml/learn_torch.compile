from __future__ import annotations



def forward(self, arg0_1: "f32[1026, 768]", arg1_1: "f32[1026, 768]", arg2_1: "f32[50005, 768]", arg3_1: "f32[768]", arg4_1: "f32[768]", arg5_1: "f32[768, 768]", arg6_1: "f32[768]", arg7_1: "f32[768, 768]", arg8_1: "f32[768]", arg9_1: "f32[768, 768]", arg10_1: "f32[768]", arg11_1: "f32[768, 768]", arg12_1: "f32[768]", arg13_1: "f32[768]", arg14_1: "f32[768]", arg15_1: "f32[3072, 768]", arg16_1: "f32[3072]", arg17_1: "f32[768, 3072]", arg18_1: "f32[768]", arg19_1: "f32[768]", arg20_1: "f32[768]", arg21_1: "f32[768, 768]", arg22_1: "f32[768]", arg23_1: "f32[768, 768]", arg24_1: "f32[768]", arg25_1: "f32[768, 768]", arg26_1: "f32[768]", arg27_1: "f32[768, 768]", arg28_1: "f32[768]", arg29_1: "f32[768]", arg30_1: "f32[768]", arg31_1: "f32[3072, 768]", arg32_1: "f32[3072]", arg33_1: "f32[768, 3072]", arg34_1: "f32[768]", arg35_1: "f32[768]", arg36_1: "f32[768]", arg37_1: "f32[768, 768]", arg38_1: "f32[768]", arg39_1: "f32[768, 768]", arg40_1: "f32[768]", arg41_1: "f32[768, 768]", arg42_1: "f32[768]", arg43_1: "f32[768, 768]", arg44_1: "f32[768]", arg45_1: "f32[768]", arg46_1: "f32[768]", arg47_1: "f32[3072, 768]", arg48_1: "f32[3072]", arg49_1: "f32[768, 3072]", arg50_1: "f32[768]", arg51_1: "f32[768]", arg52_1: "f32[768]", arg53_1: "f32[768, 768]", arg54_1: "f32[768]", arg55_1: "f32[768, 768]", arg56_1: "f32[768]", arg57_1: "f32[768, 768]", arg58_1: "f32[768]", arg59_1: "f32[768, 768]", arg60_1: "f32[768]", arg61_1: "f32[768]", arg62_1: "f32[768]", arg63_1: "f32[3072, 768]", arg64_1: "f32[3072]", arg65_1: "f32[768, 3072]", arg66_1: "f32[768]", arg67_1: "f32[768]", arg68_1: "f32[768]", arg69_1: "f32[768, 768]", arg70_1: "f32[768]", arg71_1: "f32[768, 768]", arg72_1: "f32[768]", arg73_1: "f32[768, 768]", arg74_1: "f32[768]", arg75_1: "f32[768, 768]", arg76_1: "f32[768]", arg77_1: "f32[768]", arg78_1: "f32[768]", arg79_1: "f32[3072, 768]", arg80_1: "f32[3072]", arg81_1: "f32[768, 3072]", arg82_1: "f32[768]", arg83_1: "f32[768]", arg84_1: "f32[768]", arg85_1: "f32[768, 768]", arg86_1: "f32[768]", arg87_1: "f32[768, 768]", arg88_1: "f32[768]", arg89_1: "f32[768, 768]", arg90_1: "f32[768]", arg91_1: "f32[768, 768]", arg92_1: "f32[768]", arg93_1: "f32[768]", arg94_1: "f32[768]", arg95_1: "f32[3072, 768]", arg96_1: "f32[3072]", arg97_1: "f32[768, 3072]", arg98_1: "f32[768]", arg99_1: "f32[768]", arg100_1: "f32[768]", arg101_1: "f32[50005, 768]", arg102_1: "f32[768]", arg103_1: "f32[768]", arg104_1: "f32[768, 768]", arg105_1: "f32[768]", arg106_1: "f32[768, 768]", arg107_1: "f32[768]", arg108_1: "f32[768, 768]", arg109_1: "f32[768]", arg110_1: "f32[768, 768]", arg111_1: "f32[768]", arg112_1: "f32[768]", arg113_1: "f32[768]", arg114_1: "f32[768, 768]", arg115_1: "f32[768]", arg116_1: "f32[768, 768]", arg117_1: "f32[768]", arg118_1: "f32[768, 768]", arg119_1: "f32[768]", arg120_1: "f32[768, 768]", arg121_1: "f32[768]", arg122_1: "f32[768]", arg123_1: "f32[768]", arg124_1: "f32[3072, 768]", arg125_1: "f32[3072]", arg126_1: "f32[768, 3072]", arg127_1: "f32[768]", arg128_1: "f32[768]", arg129_1: "f32[768]", arg130_1: "f32[768, 768]", arg131_1: "f32[768]", arg132_1: "f32[768, 768]", arg133_1: "f32[768]", arg134_1: "f32[768, 768]", arg135_1: "f32[768]", arg136_1: "f32[768, 768]", arg137_1: "f32[768]", arg138_1: "f32[768]", arg139_1: "f32[768]", arg140_1: "f32[768, 768]", arg141_1: "f32[768]", arg142_1: "f32[768, 768]", arg143_1: "f32[768]", arg144_1: "f32[768, 768]", arg145_1: "f32[768]", arg146_1: "f32[768, 768]", arg147_1: "f32[768]", arg148_1: "f32[768]", arg149_1: "f32[768]", arg150_1: "f32[3072, 768]", arg151_1: "f32[3072]", arg152_1: "f32[768, 3072]", arg153_1: "f32[768]", arg154_1: "f32[768]", arg155_1: "f32[768]", arg156_1: "f32[768, 768]", arg157_1: "f32[768]", arg158_1: "f32[768, 768]", arg159_1: "f32[768]", arg160_1: "f32[768, 768]", arg161_1: "f32[768]", arg162_1: "f32[768, 768]", arg163_1: "f32[768]", arg164_1: "f32[768]", arg165_1: "f32[768]", arg166_1: "f32[768, 768]", arg167_1: "f32[768]", arg168_1: "f32[768, 768]", arg169_1: "f32[768]", arg170_1: "f32[768, 768]", arg171_1: "f32[768]", arg172_1: "f32[768, 768]", arg173_1: "f32[768]", arg174_1: "f32[768]", arg175_1: "f32[768]", arg176_1: "f32[3072, 768]", arg177_1: "f32[3072]", arg178_1: "f32[768, 3072]", arg179_1: "f32[768]", arg180_1: "f32[768]", arg181_1: "f32[768]", arg182_1: "f32[768, 768]", arg183_1: "f32[768]", arg184_1: "f32[768, 768]", arg185_1: "f32[768]", arg186_1: "f32[768, 768]", arg187_1: "f32[768]", arg188_1: "f32[768, 768]", arg189_1: "f32[768]", arg190_1: "f32[768]", arg191_1: "f32[768]", arg192_1: "f32[768, 768]", arg193_1: "f32[768]", arg194_1: "f32[768, 768]", arg195_1: "f32[768]", arg196_1: "f32[768, 768]", arg197_1: "f32[768]", arg198_1: "f32[768, 768]", arg199_1: "f32[768]", arg200_1: "f32[768]", arg201_1: "f32[768]", arg202_1: "f32[3072, 768]", arg203_1: "f32[3072]", arg204_1: "f32[768, 3072]", arg205_1: "f32[768]", arg206_1: "f32[768]", arg207_1: "f32[768]", arg208_1: "f32[768, 768]", arg209_1: "f32[768]", arg210_1: "f32[768, 768]", arg211_1: "f32[768]", arg212_1: "f32[768, 768]", arg213_1: "f32[768]", arg214_1: "f32[768, 768]", arg215_1: "f32[768]", arg216_1: "f32[768]", arg217_1: "f32[768]", arg218_1: "f32[768, 768]", arg219_1: "f32[768]", arg220_1: "f32[768, 768]", arg221_1: "f32[768]", arg222_1: "f32[768, 768]", arg223_1: "f32[768]", arg224_1: "f32[768, 768]", arg225_1: "f32[768]", arg226_1: "f32[768]", arg227_1: "f32[768]", arg228_1: "f32[3072, 768]", arg229_1: "f32[3072]", arg230_1: "f32[768, 3072]", arg231_1: "f32[768]", arg232_1: "f32[768]", arg233_1: "f32[768]", arg234_1: "f32[768, 768]", arg235_1: "f32[768]", arg236_1: "f32[768, 768]", arg237_1: "f32[768]", arg238_1: "f32[768, 768]", arg239_1: "f32[768]", arg240_1: "f32[768, 768]", arg241_1: "f32[768]", arg242_1: "f32[768]", arg243_1: "f32[768]", arg244_1: "f32[768, 768]", arg245_1: "f32[768]", arg246_1: "f32[768, 768]", arg247_1: "f32[768]", arg248_1: "f32[768, 768]", arg249_1: "f32[768]", arg250_1: "f32[768, 768]", arg251_1: "f32[768]", arg252_1: "f32[768]", arg253_1: "f32[768]", arg254_1: "f32[3072, 768]", arg255_1: "f32[3072]", arg256_1: "f32[768, 3072]", arg257_1: "f32[768]", arg258_1: "f32[768]", arg259_1: "f32[768]", arg260_1: "f32[50005, 768]", arg261_1: "f32[1, 50005]", arg262_1: "i64[1, 1024]", arg263_1: "i64[1, 1024]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:764, code: input_ids = input_ids.view(-1, input_ids.shape[-1])
    view_1: "i64[1, 1024]" = torch.ops.aten.reshape.default(arg263_1, [-1, 1024]);  arg263_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:771, code: inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
    embedding: "f32[1, 1024, 768]" = torch.ops.aten.embedding.default(arg2_1, view_1, 1);  arg2_1 = view_1 = None
    mul: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(embedding, 27.712812921102035);  embedding = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:129, code: positions = torch.arange(
    iota: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:131, code: ).expand(bsz, -1)
    expand_1: "i64[1, 1024]" = torch.ops.aten.expand.default(iota, [1, -1]);  iota = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:133, code: return super().forward(positions + self.offset)
    add: "i64[1, 1024]" = torch.ops.aten.add.Tensor(expand_1, 2);  expand_1 = None
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/sparse.py:163, code: return F.embedding(
    embedding_1: "f32[1, 1024, 768]" = torch.ops.aten.embedding.default(arg0_1, add);  arg0_1 = add = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:776, code: hidden_states = inputs_embeds + embed_pos
    add_1: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul, embedding_1);  mul = embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:777, code: hidden_states = self.layernorm_embedding(hidden_states)
    var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 1024, 1]" = var_mean[0]
    getitem_1: "f32[1, 1024, 1]" = var_mean[1];  var_mean = None
    sub_1: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
    add_2: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
    mul_1: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = rsqrt = None
    mul_2: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
    add_3: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_2, arg4_1);  mul_2 = arg4_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_2: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_3, [1024, 768])
    permute: "f32[768, 768]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
    
    # No stacktrace found for following nodes
    mm_default_95: "f32[1024, 768]" = torch.ops.aten.mm.default(view_2, permute);  view_2 = permute = None
    add_tensor_95: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_95, arg6_1);  mm_default_95 = arg6_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_3: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_95, [1, 1024, 768]);  add_tensor_95 = None
    mul_3: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_3, 0.125);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_10: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(mul_3, [1, 1024, 12, 64]);  mul_3 = None
    permute_5: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
    clone_5: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:227, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_11: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_5, [12, -1, 64]);  clone_5 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_33: "f32[1, 12, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_11, 0);  view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:213, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_4: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_3, [1024, 768])
    permute_1: "f32[768, 768]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
    
    # No stacktrace found for following nodes
    mm_default_94: "f32[1024, 768]" = torch.ops.aten.mm.default(view_4, permute_1);  view_4 = permute_1 = None
    add_tensor_94: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_94, arg8_1);  mm_default_94 = arg8_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:213, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_5: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_94, [1, 1024, 768]);  add_tensor_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_6: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_5, [1, -1, 12, 64]);  view_5 = None
    permute_2: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
    clone_3: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:228, code: key_states = key_states.reshape(*proj_shape)
    view_12: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_3, [12, -1, 64]);  clone_3 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_34: "f32[1, 12, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_12, 0);  view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:214, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_7: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_3, [1024, 768])
    permute_3: "f32[768, 768]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
    
    # No stacktrace found for following nodes
    mm_default_93: "f32[1024, 768]" = torch.ops.aten.mm.default(view_7, permute_3);  view_7 = permute_3 = None
    add_tensor_93: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_93, arg10_1);  mm_default_93 = arg10_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:214, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_8: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_93, [1, 1024, 768]);  add_tensor_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_9: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_8, [1, -1, 12, 64]);  view_8 = None
    permute_4: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
    clone_4: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:229, code: value_states = value_states.reshape(*proj_shape)
    view_13: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_4, [12, -1, 64]);  clone_4 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_35: "f32[1, 12, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_13, 0);  view_13 = None
    _scaled_dot_product_efficient_attention_default_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_33, unsqueeze_default_34, unsqueeze_default_35, None, True, scale = 1.0);  unsqueeze_default_33 = unsqueeze_default_34 = unsqueeze_default_35 = None
    getitem_75: "f32[1, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_default_11[0];  _scaled_dot_product_efficient_attention_default_11 = None
    squeeze_dim_11: "f32[12, 1024, 64]" = torch.ops.aten.squeeze.dim(getitem_75, 0);  getitem_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:279, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_14: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(squeeze_dim_11, [1, 12, 1024, 64]);  squeeze_dim_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:280, code: attn_output = attn_output.transpose(1, 2)
    permute_7: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:284, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_7: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_15: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_7, [1, 1024, 768]);  clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_16: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_15, [1024, 768]);  view_15 = None
    permute_8: "f32[768, 768]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
    
    # No stacktrace found for following nodes
    mm_default_92: "f32[1024, 768]" = torch.ops.aten.mm.default(view_16, permute_8);  view_16 = permute_8 = None
    add_tensor_92: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_92, arg12_1);  mm_default_92 = arg12_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_17: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_92, [1, 1024, 768]);  add_tensor_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:335, code: hidden_states = residual + hidden_states
    add_4: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_3, view_17);  add_3 = view_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:336, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_1 = torch.ops.aten.var_mean.correction(add_4, [2], correction = 0, keepdim = True)
    getitem_2: "f32[1, 1024, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 1024, 1]" = var_mean_1[1];  var_mean_1 = None
    sub_3: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_4, getitem_3);  add_4 = getitem_3 = None
    add_5: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
    rsqrt_1: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
    mul_4: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = rsqrt_1 = None
    mul_5: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_4, arg13_1);  mul_4 = arg13_1 = None
    add_6: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_5, arg14_1);  mul_5 = arg14_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:339, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_18: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_6, [1024, 768])
    permute_9: "f32[768, 3072]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
    
    # No stacktrace found for following nodes
    mm_default_91: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_18, permute_9);  view_18 = permute_9 = None
    add_tensor_91: "f32[1024, 3072]" = torch.ops.aten.add.Tensor(mm_default_91, arg16_1);  mm_default_91 = arg16_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:339, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_19: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(add_tensor_91, [1, 1024, 3072]);  add_tensor_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_6: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.5)
    mul_7: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476);  view_19 = None
    erf: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_7);  mul_7 = None
    add_7: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_8: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_6, add_7);  mul_6 = add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:341, code: hidden_states = self.fc2(hidden_states)
    view_20: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_8, [1024, 3072]);  mul_8 = None
    permute_10: "f32[3072, 768]" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
    
    # No stacktrace found for following nodes
    mm_default_90: "f32[1024, 768]" = torch.ops.aten.mm.default(view_20, permute_10);  view_20 = permute_10 = None
    add_tensor_90: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_90, arg18_1);  mm_default_90 = arg18_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:341, code: hidden_states = self.fc2(hidden_states)
    view_21: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_90, [1, 1024, 768]);  add_tensor_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:343, code: hidden_states = residual + hidden_states
    add_8: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_6, view_21);  add_6 = view_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:344, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
    getitem_4: "f32[1, 1024, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 1024, 1]" = var_mean_2[1];  var_mean_2 = None
    sub_4: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_8, getitem_5);  add_8 = getitem_5 = None
    add_9: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
    rsqrt_2: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
    mul_9: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = rsqrt_2 = None
    mul_10: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_9, arg19_1);  mul_9 = arg19_1 = None
    add_10: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_10, arg20_1);  mul_10 = arg20_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_22: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_10, [1024, 768])
    permute_11: "f32[768, 768]" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
    
    # No stacktrace found for following nodes
    mm_default_89: "f32[1024, 768]" = torch.ops.aten.mm.default(view_22, permute_11);  view_22 = permute_11 = None
    add_tensor_89: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_89, arg22_1);  mm_default_89 = arg22_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_23: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_89, [1, 1024, 768]);  add_tensor_89 = None
    mul_11: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_23, 0.125);  view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_30: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(mul_11, [1, 1024, 12, 64]);  mul_11 = None
    permute_16: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
    clone_13: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:227, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_31: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_13, [12, -1, 64]);  clone_13 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_30: "f32[1, 12, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_31, 0);  view_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:213, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_24: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_10, [1024, 768])
    permute_12: "f32[768, 768]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
    
    # No stacktrace found for following nodes
    mm_default_88: "f32[1024, 768]" = torch.ops.aten.mm.default(view_24, permute_12);  view_24 = permute_12 = None
    add_tensor_88: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_88, arg24_1);  mm_default_88 = arg24_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:213, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_25: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_88, [1, 1024, 768]);  add_tensor_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_26: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_25, [1, -1, 12, 64]);  view_25 = None
    permute_13: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
    clone_11: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:228, code: key_states = key_states.reshape(*proj_shape)
    view_32: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_11, [12, -1, 64]);  clone_11 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_31: "f32[1, 12, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_32, 0);  view_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:214, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_27: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_10, [1024, 768])
    permute_14: "f32[768, 768]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
    
    # No stacktrace found for following nodes
    mm_default_87: "f32[1024, 768]" = torch.ops.aten.mm.default(view_27, permute_14);  view_27 = permute_14 = None
    add_tensor_87: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_87, arg26_1);  mm_default_87 = arg26_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:214, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_28: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_87, [1, 1024, 768]);  add_tensor_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_29: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_28, [1, -1, 12, 64]);  view_28 = None
    permute_15: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
    clone_12: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:229, code: value_states = value_states.reshape(*proj_shape)
    view_33: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_12, [12, -1, 64]);  clone_12 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_32: "f32[1, 12, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_33, 0);  view_33 = None
    _scaled_dot_product_efficient_attention_default_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_30, unsqueeze_default_31, unsqueeze_default_32, None, True, scale = 1.0);  unsqueeze_default_30 = unsqueeze_default_31 = unsqueeze_default_32 = None
    getitem_74: "f32[1, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_default_10[0];  _scaled_dot_product_efficient_attention_default_10 = None
    squeeze_dim_10: "f32[12, 1024, 64]" = torch.ops.aten.squeeze.dim(getitem_74, 0);  getitem_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:279, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_34: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(squeeze_dim_10, [1, 12, 1024, 64]);  squeeze_dim_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:280, code: attn_output = attn_output.transpose(1, 2)
    permute_18: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:284, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_15: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
    view_35: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_15, [1, 1024, 768]);  clone_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_36: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_35, [1024, 768]);  view_35 = None
    permute_19: "f32[768, 768]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
    
    # No stacktrace found for following nodes
    mm_default_86: "f32[1024, 768]" = torch.ops.aten.mm.default(view_36, permute_19);  view_36 = permute_19 = None
    add_tensor_86: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_86, arg28_1);  mm_default_86 = arg28_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_37: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_86, [1, 1024, 768]);  add_tensor_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:335, code: hidden_states = residual + hidden_states
    add_11: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_10, view_37);  add_10 = view_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:336, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_3 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
    getitem_6: "f32[1, 1024, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 1024, 1]" = var_mean_3[1];  var_mean_3 = None
    sub_6: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_11, getitem_7);  add_11 = getitem_7 = None
    add_12: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
    rsqrt_3: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
    mul_12: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_3);  sub_6 = rsqrt_3 = None
    mul_13: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_12, arg29_1);  mul_12 = arg29_1 = None
    add_13: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_13, arg30_1);  mul_13 = arg30_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:339, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_38: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_13, [1024, 768])
    permute_20: "f32[768, 3072]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
    
    # No stacktrace found for following nodes
    mm_default_85: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_38, permute_20);  view_38 = permute_20 = None
    add_tensor_85: "f32[1024, 3072]" = torch.ops.aten.add.Tensor(mm_default_85, arg32_1);  mm_default_85 = arg32_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:339, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_39: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(add_tensor_85, [1, 1024, 3072]);  add_tensor_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_14: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_39, 0.5)
    mul_15: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_39, 0.7071067811865476);  view_39 = None
    erf_1: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_15);  mul_15 = None
    add_14: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_16: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_14, add_14);  mul_14 = add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:341, code: hidden_states = self.fc2(hidden_states)
    view_40: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_16, [1024, 3072]);  mul_16 = None
    permute_21: "f32[3072, 768]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
    
    # No stacktrace found for following nodes
    mm_default_84: "f32[1024, 768]" = torch.ops.aten.mm.default(view_40, permute_21);  view_40 = permute_21 = None
    add_tensor_84: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_84, arg34_1);  mm_default_84 = arg34_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:341, code: hidden_states = self.fc2(hidden_states)
    view_41: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_84, [1, 1024, 768]);  add_tensor_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:343, code: hidden_states = residual + hidden_states
    add_15: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_13, view_41);  add_13 = view_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:344, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_4 = torch.ops.aten.var_mean.correction(add_15, [2], correction = 0, keepdim = True)
    getitem_8: "f32[1, 1024, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 1024, 1]" = var_mean_4[1];  var_mean_4 = None
    sub_7: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_15, getitem_9);  add_15 = getitem_9 = None
    add_16: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
    rsqrt_4: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    mul_17: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_4);  sub_7 = rsqrt_4 = None
    mul_18: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_17, arg35_1);  mul_17 = arg35_1 = None
    add_17: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_18, arg36_1);  mul_18 = arg36_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_42: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_17, [1024, 768])
    permute_22: "f32[768, 768]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
    
    # No stacktrace found for following nodes
    mm_default_83: "f32[1024, 768]" = torch.ops.aten.mm.default(view_42, permute_22);  view_42 = permute_22 = None
    add_tensor_83: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_83, arg38_1);  mm_default_83 = arg38_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_43: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_83, [1, 1024, 768]);  add_tensor_83 = None
    mul_19: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_43, 0.125);  view_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_50: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(mul_19, [1, 1024, 12, 64]);  mul_19 = None
    permute_27: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
    clone_21: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:227, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_51: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_21, [12, -1, 64]);  clone_21 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_27: "f32[1, 12, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_51, 0);  view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:213, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_44: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_17, [1024, 768])
    permute_23: "f32[768, 768]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
    
    # No stacktrace found for following nodes
    mm_default_82: "f32[1024, 768]" = torch.ops.aten.mm.default(view_44, permute_23);  view_44 = permute_23 = None
    add_tensor_82: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_82, arg40_1);  mm_default_82 = arg40_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:213, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_45: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_82, [1, 1024, 768]);  add_tensor_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_46: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_45, [1, -1, 12, 64]);  view_45 = None
    permute_24: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None
    clone_19: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:228, code: key_states = key_states.reshape(*proj_shape)
    view_52: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_19, [12, -1, 64]);  clone_19 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_28: "f32[1, 12, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_52, 0);  view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:214, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_47: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_17, [1024, 768])
    permute_25: "f32[768, 768]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
    
    # No stacktrace found for following nodes
    mm_default_81: "f32[1024, 768]" = torch.ops.aten.mm.default(view_47, permute_25);  view_47 = permute_25 = None
    add_tensor_81: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_81, arg42_1);  mm_default_81 = arg42_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:214, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_48: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_81, [1, 1024, 768]);  add_tensor_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_49: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_48, [1, -1, 12, 64]);  view_48 = None
    permute_26: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_49, [0, 2, 1, 3]);  view_49 = None
    clone_20: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:229, code: value_states = value_states.reshape(*proj_shape)
    view_53: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_20, [12, -1, 64]);  clone_20 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_29: "f32[1, 12, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_53, 0);  view_53 = None
    _scaled_dot_product_efficient_attention_default_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_27, unsqueeze_default_28, unsqueeze_default_29, None, True, scale = 1.0);  unsqueeze_default_27 = unsqueeze_default_28 = unsqueeze_default_29 = None
    getitem_73: "f32[1, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_default_9[0];  _scaled_dot_product_efficient_attention_default_9 = None
    squeeze_dim_9: "f32[12, 1024, 64]" = torch.ops.aten.squeeze.dim(getitem_73, 0);  getitem_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:279, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_54: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(squeeze_dim_9, [1, 12, 1024, 64]);  squeeze_dim_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:280, code: attn_output = attn_output.transpose(1, 2)
    permute_29: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:284, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_23: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    view_55: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_23, [1, 1024, 768]);  clone_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_56: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_55, [1024, 768]);  view_55 = None
    permute_30: "f32[768, 768]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
    
    # No stacktrace found for following nodes
    mm_default_80: "f32[1024, 768]" = torch.ops.aten.mm.default(view_56, permute_30);  view_56 = permute_30 = None
    add_tensor_80: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_80, arg44_1);  mm_default_80 = arg44_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_57: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_80, [1, 1024, 768]);  add_tensor_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:335, code: hidden_states = residual + hidden_states
    add_18: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_17, view_57);  add_17 = view_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:336, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_5 = torch.ops.aten.var_mean.correction(add_18, [2], correction = 0, keepdim = True)
    getitem_10: "f32[1, 1024, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 1024, 1]" = var_mean_5[1];  var_mean_5 = None
    sub_9: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_18, getitem_11);  add_18 = getitem_11 = None
    add_19: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
    rsqrt_5: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
    mul_20: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_5);  sub_9 = rsqrt_5 = None
    mul_21: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_20, arg45_1);  mul_20 = arg45_1 = None
    add_20: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_21, arg46_1);  mul_21 = arg46_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:339, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_58: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_20, [1024, 768])
    permute_31: "f32[768, 3072]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
    
    # No stacktrace found for following nodes
    mm_default_79: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_58, permute_31);  view_58 = permute_31 = None
    add_tensor_79: "f32[1024, 3072]" = torch.ops.aten.add.Tensor(mm_default_79, arg48_1);  mm_default_79 = arg48_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:339, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_59: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(add_tensor_79, [1, 1024, 3072]);  add_tensor_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_22: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_59, 0.5)
    mul_23: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_59, 0.7071067811865476);  view_59 = None
    erf_2: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_23);  mul_23 = None
    add_21: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_24: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_22, add_21);  mul_22 = add_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:341, code: hidden_states = self.fc2(hidden_states)
    view_60: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_24, [1024, 3072]);  mul_24 = None
    permute_32: "f32[3072, 768]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
    
    # No stacktrace found for following nodes
    mm_default_78: "f32[1024, 768]" = torch.ops.aten.mm.default(view_60, permute_32);  view_60 = permute_32 = None
    add_tensor_78: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_78, arg50_1);  mm_default_78 = arg50_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:341, code: hidden_states = self.fc2(hidden_states)
    view_61: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_78, [1, 1024, 768]);  add_tensor_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:343, code: hidden_states = residual + hidden_states
    add_22: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_20, view_61);  add_20 = view_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:344, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_6 = torch.ops.aten.var_mean.correction(add_22, [2], correction = 0, keepdim = True)
    getitem_12: "f32[1, 1024, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 1024, 1]" = var_mean_6[1];  var_mean_6 = None
    sub_10: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_22, getitem_13);  add_22 = getitem_13 = None
    add_23: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
    rsqrt_6: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
    mul_25: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_6);  sub_10 = rsqrt_6 = None
    mul_26: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_25, arg51_1);  mul_25 = arg51_1 = None
    add_24: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_26, arg52_1);  mul_26 = arg52_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_62: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_24, [1024, 768])
    permute_33: "f32[768, 768]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
    
    # No stacktrace found for following nodes
    mm_default_77: "f32[1024, 768]" = torch.ops.aten.mm.default(view_62, permute_33);  view_62 = permute_33 = None
    add_tensor_77: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_77, arg54_1);  mm_default_77 = arg54_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_63: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_77, [1, 1024, 768]);  add_tensor_77 = None
    mul_27: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_63, 0.125);  view_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_70: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(mul_27, [1, 1024, 12, 64]);  mul_27 = None
    permute_38: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None
    clone_29: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:227, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_71: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_29, [12, -1, 64]);  clone_29 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_24: "f32[1, 12, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_71, 0);  view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:213, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_64: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_24, [1024, 768])
    permute_34: "f32[768, 768]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
    
    # No stacktrace found for following nodes
    mm_default_76: "f32[1024, 768]" = torch.ops.aten.mm.default(view_64, permute_34);  view_64 = permute_34 = None
    add_tensor_76: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_76, arg56_1);  mm_default_76 = arg56_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:213, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_65: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_76, [1, 1024, 768]);  add_tensor_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_66: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_65, [1, -1, 12, 64]);  view_65 = None
    permute_35: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None
    clone_27: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:228, code: key_states = key_states.reshape(*proj_shape)
    view_72: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_27, [12, -1, 64]);  clone_27 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_25: "f32[1, 12, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_72, 0);  view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:214, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_67: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_24, [1024, 768])
    permute_36: "f32[768, 768]" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
    
    # No stacktrace found for following nodes
    mm_default_75: "f32[1024, 768]" = torch.ops.aten.mm.default(view_67, permute_36);  view_67 = permute_36 = None
    add_tensor_75: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_75, arg58_1);  mm_default_75 = arg58_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:214, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_68: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_75, [1, 1024, 768]);  add_tensor_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_69: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_68, [1, -1, 12, 64]);  view_68 = None
    permute_37: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_69, [0, 2, 1, 3]);  view_69 = None
    clone_28: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:229, code: value_states = value_states.reshape(*proj_shape)
    view_73: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_28, [12, -1, 64]);  clone_28 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_26: "f32[1, 12, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_73, 0);  view_73 = None
    _scaled_dot_product_efficient_attention_default_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_24, unsqueeze_default_25, unsqueeze_default_26, None, True, scale = 1.0);  unsqueeze_default_24 = unsqueeze_default_25 = unsqueeze_default_26 = None
    getitem_72: "f32[1, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_default_8[0];  _scaled_dot_product_efficient_attention_default_8 = None
    squeeze_dim_8: "f32[12, 1024, 64]" = torch.ops.aten.squeeze.dim(getitem_72, 0);  getitem_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:279, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_74: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(squeeze_dim_8, [1, 12, 1024, 64]);  squeeze_dim_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:280, code: attn_output = attn_output.transpose(1, 2)
    permute_40: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:284, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_31: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
    view_75: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_31, [1, 1024, 768]);  clone_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_76: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_75, [1024, 768]);  view_75 = None
    permute_41: "f32[768, 768]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
    
    # No stacktrace found for following nodes
    mm_default_74: "f32[1024, 768]" = torch.ops.aten.mm.default(view_76, permute_41);  view_76 = permute_41 = None
    add_tensor_74: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_74, arg60_1);  mm_default_74 = arg60_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_77: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_74, [1, 1024, 768]);  add_tensor_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:335, code: hidden_states = residual + hidden_states
    add_25: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_24, view_77);  add_24 = view_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:336, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_7 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
    getitem_14: "f32[1, 1024, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 1024, 1]" = var_mean_7[1];  var_mean_7 = None
    sub_12: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_25, getitem_15);  add_25 = getitem_15 = None
    add_26: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
    rsqrt_7: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
    mul_28: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_7);  sub_12 = rsqrt_7 = None
    mul_29: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_28, arg61_1);  mul_28 = arg61_1 = None
    add_27: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_29, arg62_1);  mul_29 = arg62_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:339, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_78: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_27, [1024, 768])
    permute_42: "f32[768, 3072]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
    
    # No stacktrace found for following nodes
    mm_default_73: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_78, permute_42);  view_78 = permute_42 = None
    add_tensor_73: "f32[1024, 3072]" = torch.ops.aten.add.Tensor(mm_default_73, arg64_1);  mm_default_73 = arg64_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:339, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_79: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(add_tensor_73, [1, 1024, 3072]);  add_tensor_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_30: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_79, 0.5)
    mul_31: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_79, 0.7071067811865476);  view_79 = None
    erf_3: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_31);  mul_31 = None
    add_28: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_32: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_30, add_28);  mul_30 = add_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:341, code: hidden_states = self.fc2(hidden_states)
    view_80: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_32, [1024, 3072]);  mul_32 = None
    permute_43: "f32[3072, 768]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
    
    # No stacktrace found for following nodes
    mm_default_72: "f32[1024, 768]" = torch.ops.aten.mm.default(view_80, permute_43);  view_80 = permute_43 = None
    add_tensor_72: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_72, arg66_1);  mm_default_72 = arg66_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:341, code: hidden_states = self.fc2(hidden_states)
    view_81: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_72, [1, 1024, 768]);  add_tensor_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:343, code: hidden_states = residual + hidden_states
    add_29: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_27, view_81);  add_27 = view_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:344, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_8 = torch.ops.aten.var_mean.correction(add_29, [2], correction = 0, keepdim = True)
    getitem_16: "f32[1, 1024, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 1024, 1]" = var_mean_8[1];  var_mean_8 = None
    sub_13: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_29, getitem_17);  add_29 = getitem_17 = None
    add_30: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
    rsqrt_8: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
    mul_33: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_8);  sub_13 = rsqrt_8 = None
    mul_34: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_33, arg67_1);  mul_33 = arg67_1 = None
    add_31: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_34, arg68_1);  mul_34 = arg68_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_82: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_31, [1024, 768])
    permute_44: "f32[768, 768]" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
    
    # No stacktrace found for following nodes
    mm_default_71: "f32[1024, 768]" = torch.ops.aten.mm.default(view_82, permute_44);  view_82 = permute_44 = None
    add_tensor_71: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_71, arg70_1);  mm_default_71 = arg70_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_83: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_71, [1, 1024, 768]);  add_tensor_71 = None
    mul_35: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_83, 0.125);  view_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_90: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(mul_35, [1, 1024, 12, 64]);  mul_35 = None
    permute_49: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None
    clone_37: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:227, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_91: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_37, [12, -1, 64]);  clone_37 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_21: "f32[1, 12, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_91, 0);  view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:213, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_84: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_31, [1024, 768])
    permute_45: "f32[768, 768]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
    
    # No stacktrace found for following nodes
    mm_default_70: "f32[1024, 768]" = torch.ops.aten.mm.default(view_84, permute_45);  view_84 = permute_45 = None
    add_tensor_70: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_70, arg72_1);  mm_default_70 = arg72_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:213, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_85: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_70, [1, 1024, 768]);  add_tensor_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_86: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_85, [1, -1, 12, 64]);  view_85 = None
    permute_46: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_86, [0, 2, 1, 3]);  view_86 = None
    clone_35: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:228, code: key_states = key_states.reshape(*proj_shape)
    view_92: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_35, [12, -1, 64]);  clone_35 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_22: "f32[1, 12, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_92, 0);  view_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:214, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_87: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_31, [1024, 768])
    permute_47: "f32[768, 768]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
    
    # No stacktrace found for following nodes
    mm_default_69: "f32[1024, 768]" = torch.ops.aten.mm.default(view_87, permute_47);  view_87 = permute_47 = None
    add_tensor_69: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_69, arg74_1);  mm_default_69 = arg74_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:214, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_88: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_69, [1, 1024, 768]);  add_tensor_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_89: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_88, [1, -1, 12, 64]);  view_88 = None
    permute_48: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_89, [0, 2, 1, 3]);  view_89 = None
    clone_36: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:229, code: value_states = value_states.reshape(*proj_shape)
    view_93: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_36, [12, -1, 64]);  clone_36 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_23: "f32[1, 12, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_93, 0);  view_93 = None
    _scaled_dot_product_efficient_attention_default_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_21, unsqueeze_default_22, unsqueeze_default_23, None, True, scale = 1.0);  unsqueeze_default_21 = unsqueeze_default_22 = unsqueeze_default_23 = None
    getitem_71: "f32[1, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_default_7[0];  _scaled_dot_product_efficient_attention_default_7 = None
    squeeze_dim_7: "f32[12, 1024, 64]" = torch.ops.aten.squeeze.dim(getitem_71, 0);  getitem_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:279, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_94: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(squeeze_dim_7, [1, 12, 1024, 64]);  squeeze_dim_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:280, code: attn_output = attn_output.transpose(1, 2)
    permute_51: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:284, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_39: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
    view_95: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_39, [1, 1024, 768]);  clone_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_96: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_95, [1024, 768]);  view_95 = None
    permute_52: "f32[768, 768]" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
    
    # No stacktrace found for following nodes
    mm_default_68: "f32[1024, 768]" = torch.ops.aten.mm.default(view_96, permute_52);  view_96 = permute_52 = None
    add_tensor_68: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_68, arg76_1);  mm_default_68 = arg76_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_97: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_68, [1, 1024, 768]);  add_tensor_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:335, code: hidden_states = residual + hidden_states
    add_32: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_31, view_97);  add_31 = view_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:336, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_9 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
    getitem_18: "f32[1, 1024, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 1024, 1]" = var_mean_9[1];  var_mean_9 = None
    sub_15: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_32, getitem_19);  add_32 = getitem_19 = None
    add_33: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
    rsqrt_9: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
    mul_36: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_9);  sub_15 = rsqrt_9 = None
    mul_37: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_36, arg77_1);  mul_36 = arg77_1 = None
    add_34: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_37, arg78_1);  mul_37 = arg78_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:339, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_98: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_34, [1024, 768])
    permute_53: "f32[768, 3072]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
    
    # No stacktrace found for following nodes
    mm_default_67: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_98, permute_53);  view_98 = permute_53 = None
    add_tensor_67: "f32[1024, 3072]" = torch.ops.aten.add.Tensor(mm_default_67, arg80_1);  mm_default_67 = arg80_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:339, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_99: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(add_tensor_67, [1, 1024, 3072]);  add_tensor_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_38: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_99, 0.5)
    mul_39: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_99, 0.7071067811865476);  view_99 = None
    erf_4: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_39);  mul_39 = None
    add_35: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_40: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_38, add_35);  mul_38 = add_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:341, code: hidden_states = self.fc2(hidden_states)
    view_100: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_40, [1024, 3072]);  mul_40 = None
    permute_54: "f32[3072, 768]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
    
    # No stacktrace found for following nodes
    mm_default_66: "f32[1024, 768]" = torch.ops.aten.mm.default(view_100, permute_54);  view_100 = permute_54 = None
    add_tensor_66: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_66, arg82_1);  mm_default_66 = arg82_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:341, code: hidden_states = self.fc2(hidden_states)
    view_101: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_66, [1, 1024, 768]);  add_tensor_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:343, code: hidden_states = residual + hidden_states
    add_36: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_34, view_101);  add_34 = view_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:344, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_10 = torch.ops.aten.var_mean.correction(add_36, [2], correction = 0, keepdim = True)
    getitem_20: "f32[1, 1024, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 1024, 1]" = var_mean_10[1];  var_mean_10 = None
    sub_16: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_36, getitem_21);  add_36 = getitem_21 = None
    add_37: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
    rsqrt_10: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
    mul_41: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_10);  sub_16 = rsqrt_10 = None
    mul_42: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_41, arg83_1);  mul_41 = arg83_1 = None
    add_38: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_42, arg84_1);  mul_42 = arg84_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_102: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_38, [1024, 768])
    permute_55: "f32[768, 768]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
    
    # No stacktrace found for following nodes
    mm_default_65: "f32[1024, 768]" = torch.ops.aten.mm.default(view_102, permute_55);  view_102 = permute_55 = None
    add_tensor_65: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_65, arg86_1);  mm_default_65 = arg86_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_103: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_65, [1, 1024, 768]);  add_tensor_65 = None
    mul_43: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_103, 0.125);  view_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_110: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(mul_43, [1, 1024, 12, 64]);  mul_43 = None
    permute_60: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_110, [0, 2, 1, 3]);  view_110 = None
    clone_45: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:227, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_111: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_45, [12, -1, 64]);  clone_45 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_18: "f32[1, 12, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_111, 0);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:213, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_104: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_38, [1024, 768])
    permute_56: "f32[768, 768]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
    
    # No stacktrace found for following nodes
    mm_default_64: "f32[1024, 768]" = torch.ops.aten.mm.default(view_104, permute_56);  view_104 = permute_56 = None
    add_tensor_64: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_64, arg88_1);  mm_default_64 = arg88_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:213, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_105: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_64, [1, 1024, 768]);  add_tensor_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_106: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_105, [1, -1, 12, 64]);  view_105 = None
    permute_57: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_106, [0, 2, 1, 3]);  view_106 = None
    clone_43: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_57, memory_format = torch.contiguous_format);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:228, code: key_states = key_states.reshape(*proj_shape)
    view_112: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_43, [12, -1, 64]);  clone_43 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_19: "f32[1, 12, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_112, 0);  view_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:214, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_107: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_38, [1024, 768])
    permute_58: "f32[768, 768]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
    
    # No stacktrace found for following nodes
    mm_default_63: "f32[1024, 768]" = torch.ops.aten.mm.default(view_107, permute_58);  view_107 = permute_58 = None
    add_tensor_63: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_63, arg90_1);  mm_default_63 = arg90_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:214, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_108: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_63, [1, 1024, 768]);  add_tensor_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_109: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_108, [1, -1, 12, 64]);  view_108 = None
    permute_59: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_109, [0, 2, 1, 3]);  view_109 = None
    clone_44: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:229, code: value_states = value_states.reshape(*proj_shape)
    view_113: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_44, [12, -1, 64]);  clone_44 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_20: "f32[1, 12, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_113, 0);  view_113 = None
    _scaled_dot_product_efficient_attention_default_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_18, unsqueeze_default_19, unsqueeze_default_20, None, True, scale = 1.0);  unsqueeze_default_18 = unsqueeze_default_19 = unsqueeze_default_20 = None
    getitem_70: "f32[1, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_default_6[0];  _scaled_dot_product_efficient_attention_default_6 = None
    squeeze_dim_6: "f32[12, 1024, 64]" = torch.ops.aten.squeeze.dim(getitem_70, 0);  getitem_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:279, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_114: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(squeeze_dim_6, [1, 12, 1024, 64]);  squeeze_dim_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:280, code: attn_output = attn_output.transpose(1, 2)
    permute_62: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:284, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_47: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    view_115: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_47, [1, 1024, 768]);  clone_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_116: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_115, [1024, 768]);  view_115 = None
    permute_63: "f32[768, 768]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
    
    # No stacktrace found for following nodes
    mm_default_62: "f32[1024, 768]" = torch.ops.aten.mm.default(view_116, permute_63);  view_116 = permute_63 = None
    add_tensor_62: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_62, arg92_1);  mm_default_62 = arg92_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_117: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_62, [1, 1024, 768]);  add_tensor_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:335, code: hidden_states = residual + hidden_states
    add_39: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_38, view_117);  add_38 = view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:336, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_11 = torch.ops.aten.var_mean.correction(add_39, [2], correction = 0, keepdim = True)
    getitem_22: "f32[1, 1024, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 1024, 1]" = var_mean_11[1];  var_mean_11 = None
    sub_18: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_39, getitem_23);  add_39 = getitem_23 = None
    add_40: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
    rsqrt_11: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
    mul_44: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_11);  sub_18 = rsqrt_11 = None
    mul_45: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_44, arg93_1);  mul_44 = arg93_1 = None
    add_41: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_45, arg94_1);  mul_45 = arg94_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:339, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_118: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_41, [1024, 768])
    permute_64: "f32[768, 3072]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
    
    # No stacktrace found for following nodes
    mm_default_61: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_118, permute_64);  view_118 = permute_64 = None
    add_tensor_61: "f32[1024, 3072]" = torch.ops.aten.add.Tensor(mm_default_61, arg96_1);  mm_default_61 = arg96_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:339, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_119: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(add_tensor_61, [1, 1024, 3072]);  add_tensor_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_46: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_119, 0.5)
    mul_47: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_119, 0.7071067811865476);  view_119 = None
    erf_5: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
    add_42: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_48: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_46, add_42);  mul_46 = add_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:341, code: hidden_states = self.fc2(hidden_states)
    view_120: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_48, [1024, 3072]);  mul_48 = None
    permute_65: "f32[3072, 768]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
    
    # No stacktrace found for following nodes
    mm_default_60: "f32[1024, 768]" = torch.ops.aten.mm.default(view_120, permute_65);  view_120 = permute_65 = None
    add_tensor_60: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_60, arg98_1);  mm_default_60 = arg98_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:341, code: hidden_states = self.fc2(hidden_states)
    view_121: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_60, [1, 1024, 768]);  add_tensor_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:343, code: hidden_states = residual + hidden_states
    add_43: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_41, view_121);  add_41 = view_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:344, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_12 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
    getitem_24: "f32[1, 1024, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 1024, 1]" = var_mean_12[1];  var_mean_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:70, code: prev_output_tokens.masked_fill_(prev_output_tokens == -100, pad_token_id)
    eq: "b8[1, 1024]" = torch.ops.aten.eq.Scalar(arg262_1, -100)
    full_default: "i64[]" = torch.ops.aten.full.default([], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where: "i64[1, 1024]" = torch.ops.aten.where.self(eq, full_default, arg262_1);  eq = full_default = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:74, code: prev_output_tokens[:, 1:] = prev_output_tokens[:, :-1].clone()
    slice_8: "i64[1, 1023]" = torch.ops.aten.slice.Tensor(where, 1, 1, 9223372036854775807)
    slice_4: "i64[1, 1023]" = torch.ops.aten.slice.Tensor(where, 1, 0, -1)
    clone_1: "i64[1, 1023]" = torch.ops.aten.clone.default(slice_4);  slice_4 = None
    copy: "i64[1, 1023]" = torch.ops.aten.copy.default(slice_8, clone_1);  slice_8 = clone_1 = None
    slice_scatter: "i64[1, 1024]" = torch.ops.aten.slice_scatter.default(where, copy, 1, 1, 9223372036854775807);  copy = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:75, code: prev_output_tokens[:, 0] = decoder_start_tokens
    select_1: "i64[1]" = torch.ops.aten.select.int(slice_scatter, 1, 0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:72, code: index_of_eos = (prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    ne: "b8[1, 1024]" = torch.ops.aten.ne.Scalar(where, 1)
    sum_1: "i64[1]" = torch.ops.aten.sum.dim_IntList(ne, [1]);  ne = None
    sub: "i64[1]" = torch.ops.aten.sub.Tensor(sum_1, 1);  sum_1 = None
    unsqueeze: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(sub, -1);  sub = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:73, code: decoder_start_tokens = prev_output_tokens.gather(1, index_of_eos).squeeze()
    gather: "i64[1, 1]" = torch.ops.aten.gather.default(where, 1, unsqueeze);  where = unsqueeze = None
    
    # No stacktrace found for following nodes
    squeeze: "i64[]" = torch.ops.aten.squeeze.default(gather);  gather = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:75, code: prev_output_tokens[:, 0] = decoder_start_tokens
    expand: "i64[1]" = torch.ops.aten.expand.default(squeeze, [1]);  squeeze = None
    copy_1: "i64[1]" = torch.ops.aten.copy.default(select_1, expand);  select_1 = expand = None
    select_scatter: "i64[1, 1024]" = torch.ops.aten.select_scatter.default(slice_scatter, copy_1, 1, 0);  slice_scatter = copy_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:1013, code: inputs_embeds = self.embed_tokens(input) * self.embed_scale
    embedding_2: "f32[1, 1024, 768]" = torch.ops.aten.embedding.default(arg101_1, select_scatter, 1);  arg101_1 = select_scatter = None
    mul_51: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(embedding_2, 27.712812921102035);  embedding_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:129, code: positions = torch.arange(
    iota_2: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:131, code: ).expand(bsz, -1)
    expand_3: "i64[1, 1024]" = torch.ops.aten.expand.default(iota_2, [1, -1]);  iota_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:133, code: return super().forward(positions + self.offset)
    add_47: "i64[1, 1024]" = torch.ops.aten.add.Tensor(expand_3, 2);  expand_3 = None
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/sparse.py:163, code: return F.embedding(
    embedding_3: "f32[1, 1024, 768]" = torch.ops.aten.embedding.default(arg1_1, add_47);  arg1_1 = add_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:1028, code: hidden_states = inputs_embeds + positions
    add_48: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_51, embedding_3);  mul_51 = embedding_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:1029, code: hidden_states = self.layernorm_embedding(hidden_states)
    var_mean_13 = torch.ops.aten.var_mean.correction(add_48, [2], correction = 0, keepdim = True)
    getitem_26: "f32[1, 1024, 1]" = var_mean_13[0]
    getitem_27: "f32[1, 1024, 1]" = var_mean_13[1];  var_mean_13 = None
    sub_20: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_48, getitem_27);  add_48 = getitem_27 = None
    add_49: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
    rsqrt_13: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    mul_52: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = rsqrt_13 = None
    mul_53: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_52, arg102_1);  mul_52 = arg102_1 = None
    add_50: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_53, arg103_1);  mul_53 = arg103_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_124: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_50, [1024, 768])
    permute_66: "f32[768, 768]" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
    
    # No stacktrace found for following nodes
    mm_default_59: "f32[1024, 768]" = torch.ops.aten.mm.default(view_124, permute_66);  view_124 = permute_66 = None
    add_tensor_59: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_59, arg105_1);  mm_default_59 = arg105_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_125: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_59, [1, 1024, 768]);  add_tensor_59 = None
    mul_54: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_125, 0.125);  view_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_132: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(mul_54, [1, 1024, 12, 64]);  mul_54 = None
    permute_71: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_132, [0, 2, 1, 3]);  view_132 = None
    clone_54: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:227, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_133: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_54, [12, -1, 64]);  clone_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:213, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_126: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_50, [1024, 768])
    permute_67: "f32[768, 768]" = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
    
    # No stacktrace found for following nodes
    mm_default_58: "f32[1024, 768]" = torch.ops.aten.mm.default(view_126, permute_67);  view_126 = permute_67 = None
    add_tensor_58: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_58, arg107_1);  mm_default_58 = arg107_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:213, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_127: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_58, [1, 1024, 768]);  add_tensor_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_128: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_127, [1, -1, 12, 64]);  view_127 = None
    permute_68: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_128, [0, 2, 1, 3]);  view_128 = None
    clone_52: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:228, code: key_states = key_states.reshape(*proj_shape)
    view_134: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_52, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:232, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_72: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_134, [0, 2, 1]);  view_134 = None
    bmm_12: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_133, permute_72);  view_133 = permute_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:245, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_136: "f32[1, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_12, [1, 12, 1024, 1024]);  bmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:89, code: mask_cond = torch.arange(mask.size(-1), device=device)
    iota_1: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:90, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    add_46: "i64[1024]" = torch.ops.aten.add.Tensor(iota_1, 1)
    view_123: "i64[1024, 1]" = torch.ops.aten.reshape.default(add_46, [1024, 1]);  add_46 = None
    lt: "b8[1024, 1024]" = torch.ops.aten.lt.Tensor(iota_1, view_123);  iota_1 = view_123 = None
    full_default_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:88, code: mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    full_default_1: "f32[1024, 1024]" = torch.ops.aten.full.default([1024, 1024], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:90, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    where_1: "f32[1024, 1024]" = torch.ops.aten.where.self(lt, full_default_2, full_default_1);  lt = full_default_2 = full_default_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:245, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    unsqueeze_3: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(where_1, 0);  where_1 = None
    unsqueeze_4: "f32[1, 1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 1);  unsqueeze_3 = None
    expand_4: "f32[1, 1, 1024, 1024]" = torch.ops.aten.expand.default(unsqueeze_4, [1, 1, 1024, 1024]);  unsqueeze_4 = None
    add_51: "f32[1, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_136, expand_4);  view_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:246, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_137: "f32[12, 1024, 1024]" = torch.ops.aten.reshape.default(add_51, [12, 1024, 1024]);  add_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:248, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_6: "f32[12, 1024, 1]" = torch.ops.aten.amax.default(view_137, [-1], True)
    sub_21: "f32[12, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_137, amax_6);  view_137 = amax_6 = None
    exp_6: "f32[12, 1024, 1024]" = torch.ops.aten.exp.default(sub_21);  sub_21 = None
    sum_8: "f32[12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_6: "f32[12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_6, sum_8);  exp_6 = sum_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:214, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_129: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_50, [1024, 768])
    permute_69: "f32[768, 768]" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
    
    # No stacktrace found for following nodes
    mm_default_57: "f32[1024, 768]" = torch.ops.aten.mm.default(view_129, permute_69);  view_129 = permute_69 = None
    add_tensor_57: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_57, arg109_1);  mm_default_57 = arg109_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:214, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_130: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_57, [1, 1024, 768]);  add_tensor_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_131: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_130, [1, -1, 12, 64]);  view_130 = None
    permute_70: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_131, [0, 2, 1, 3]);  view_131 = None
    clone_53: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:229, code: value_states = value_states.reshape(*proj_shape)
    view_135: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_53, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:271, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_13: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(div_6, view_135);  div_6 = view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:279, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_138: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_13, [1, 12, 1024, 64]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:280, code: attn_output = attn_output.transpose(1, 2)
    permute_73: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:284, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_56: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
    view_139: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_56, [1, 1024, 768]);  clone_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_140: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_139, [1024, 768]);  view_139 = None
    permute_74: "f32[768, 768]" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
    
    # No stacktrace found for following nodes
    mm_default_56: "f32[1024, 768]" = torch.ops.aten.mm.default(view_140, permute_74);  view_140 = permute_74 = None
    add_tensor_56: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_56, arg111_1);  mm_default_56 = arg111_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_141: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_56, [1, 1024, 768]);  add_tensor_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:432, code: hidden_states = residual + hidden_states
    add_52: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_50, view_141);  add_50 = view_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:433, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_14 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 1024, 1]" = var_mean_14[0]
    getitem_29: "f32[1, 1024, 1]" = var_mean_14[1];  var_mean_14 = None
    sub_22: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_52, getitem_29);  add_52 = getitem_29 = None
    add_53: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
    rsqrt_14: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    mul_55: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_14);  sub_22 = rsqrt_14 = None
    mul_56: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_55, arg112_1);  mul_55 = arg112_1 = None
    add_54: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_56, arg113_1);  mul_56 = arg113_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_142: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_54, [1024, 768])
    permute_75: "f32[768, 768]" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
    
    # No stacktrace found for following nodes
    mm_default_55: "f32[1024, 768]" = torch.ops.aten.mm.default(view_142, permute_75);  view_142 = permute_75 = None
    add_tensor_55: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_55, arg115_1);  mm_default_55 = arg115_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_143: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_55, [1, 1024, 768]);  add_tensor_55 = None
    mul_57: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_143, 0.125);  view_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_150: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(mul_57, [1, 1024, 12, 64]);  mul_57 = None
    permute_80: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_150, [0, 2, 1, 3]);  view_150 = None
    clone_60: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format);  permute_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:227, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_151: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_60, [12, -1, 64]);  clone_60 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_15: "f32[1, 12, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_151, 0);  view_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:344, code: hidden_states = self.final_layer_norm(hidden_states)
    sub_19: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_43, getitem_25);  add_43 = getitem_25 = None
    add_44: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
    rsqrt_12: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
    mul_49: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_12);  sub_19 = rsqrt_12 = None
    mul_50: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_49, arg99_1);  mul_49 = arg99_1 = None
    add_45: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_50, arg100_1);  mul_50 = arg100_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:203, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_144: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_45, [1024, 768])
    permute_76: "f32[768, 768]" = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
    
    # No stacktrace found for following nodes
    mm_default_54: "f32[1024, 768]" = torch.ops.aten.mm.default(view_144, permute_76);  view_144 = permute_76 = None
    add_tensor_54: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_54, arg117_1);  mm_default_54 = arg117_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:203, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_145: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_54, [1, 1024, 768]);  add_tensor_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_146: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_145, [1, -1, 12, 64]);  view_145 = None
    permute_77: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
    clone_58: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_77, memory_format = torch.contiguous_format);  permute_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:228, code: key_states = key_states.reshape(*proj_shape)
    view_152: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_58, [12, -1, 64])
    
    # No stacktrace found for following nodes
    unsqueeze_default_16: "f32[1, 12, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_152, 0);  view_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:204, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_147: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_45, [1024, 768])
    permute_78: "f32[768, 768]" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
    
    # No stacktrace found for following nodes
    mm_default_53: "f32[1024, 768]" = torch.ops.aten.mm.default(view_147, permute_78);  view_147 = permute_78 = None
    add_tensor_53: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_53, arg119_1);  mm_default_53 = arg119_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:204, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_148: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_53, [1, 1024, 768]);  add_tensor_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_149: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_148, [1, -1, 12, 64]);  view_148 = None
    permute_79: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_149, [0, 2, 1, 3]);  view_149 = None
    clone_59: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:229, code: value_states = value_states.reshape(*proj_shape)
    view_153: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_59, [12, -1, 64])
    
    # No stacktrace found for following nodes
    unsqueeze_default_17: "f32[1, 12, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_153, 0);  view_153 = None
    _scaled_dot_product_efficient_attention_default_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_15, unsqueeze_default_16, unsqueeze_default_17, None, True, scale = 1.0);  unsqueeze_default_15 = unsqueeze_default_16 = unsqueeze_default_17 = None
    getitem_69: "f32[1, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_default_5[0];  _scaled_dot_product_efficient_attention_default_5 = None
    squeeze_dim_5: "f32[12, 1024, 64]" = torch.ops.aten.squeeze.dim(getitem_69, 0);  getitem_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:279, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_154: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(squeeze_dim_5, [1, 12, 1024, 64]);  squeeze_dim_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:280, code: attn_output = attn_output.transpose(1, 2)
    permute_82: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_154, [0, 2, 1, 3]);  view_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:284, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_62: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
    view_155: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_62, [1, 1024, 768]);  clone_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_156: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_155, [1024, 768]);  view_155 = None
    permute_83: "f32[768, 768]" = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
    
    # No stacktrace found for following nodes
    mm_default_52: "f32[1024, 768]" = torch.ops.aten.mm.default(view_156, permute_83);  view_156 = permute_83 = None
    add_tensor_52: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_52, arg121_1);  mm_default_52 = arg121_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_157: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_52, [1, 1024, 768]);  add_tensor_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:452, code: hidden_states = residual + hidden_states
    add_55: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_54, view_157);  add_54 = view_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:453, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_15 = torch.ops.aten.var_mean.correction(add_55, [2], correction = 0, keepdim = True)
    getitem_30: "f32[1, 1024, 1]" = var_mean_15[0]
    getitem_31: "f32[1, 1024, 1]" = var_mean_15[1];  var_mean_15 = None
    sub_24: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_55, getitem_31);  add_55 = getitem_31 = None
    add_56: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
    rsqrt_15: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    mul_58: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_15);  sub_24 = rsqrt_15 = None
    mul_59: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_58, arg122_1);  mul_58 = arg122_1 = None
    add_57: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_59, arg123_1);  mul_59 = arg123_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_158: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_57, [1024, 768])
    permute_84: "f32[768, 3072]" = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
    
    # No stacktrace found for following nodes
    mm_default_51: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_158, permute_84);  view_158 = permute_84 = None
    add_tensor_51: "f32[1024, 3072]" = torch.ops.aten.add.Tensor(mm_default_51, arg125_1);  mm_default_51 = arg125_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_159: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(add_tensor_51, [1, 1024, 3072]);  add_tensor_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_60: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_159, 0.5)
    mul_61: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_159, 0.7071067811865476);  view_159 = None
    erf_6: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_61);  mul_61 = None
    add_58: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_62: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_60, add_58);  mul_60 = add_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_160: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_62, [1024, 3072]);  mul_62 = None
    permute_85: "f32[3072, 768]" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
    
    # No stacktrace found for following nodes
    mm_default_50: "f32[1024, 768]" = torch.ops.aten.mm.default(view_160, permute_85);  view_160 = permute_85 = None
    add_tensor_50: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_50, arg127_1);  mm_default_50 = arg127_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_161: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_50, [1, 1024, 768]);  add_tensor_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:464, code: hidden_states = residual + hidden_states
    add_59: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_57, view_161);  add_57 = view_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:465, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_16 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
    getitem_32: "f32[1, 1024, 1]" = var_mean_16[0]
    getitem_33: "f32[1, 1024, 1]" = var_mean_16[1];  var_mean_16 = None
    sub_25: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_59, getitem_33);  add_59 = getitem_33 = None
    add_60: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
    rsqrt_16: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    mul_63: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_16);  sub_25 = rsqrt_16 = None
    mul_64: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_63, arg128_1);  mul_63 = arg128_1 = None
    add_61: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_64, arg129_1);  mul_64 = arg129_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_162: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_61, [1024, 768])
    permute_86: "f32[768, 768]" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
    
    # No stacktrace found for following nodes
    mm_default_49: "f32[1024, 768]" = torch.ops.aten.mm.default(view_162, permute_86);  view_162 = permute_86 = None
    add_tensor_49: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_49, arg131_1);  mm_default_49 = arg131_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_163: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_49, [1, 1024, 768]);  add_tensor_49 = None
    mul_65: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_163, 0.125);  view_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_170: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(mul_65, [1, 1024, 12, 64]);  mul_65 = None
    permute_91: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_170, [0, 2, 1, 3]);  view_170 = None
    clone_68: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_91, memory_format = torch.contiguous_format);  permute_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:227, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_171: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_68, [12, -1, 64]);  clone_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:213, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_164: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_61, [1024, 768])
    permute_87: "f32[768, 768]" = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
    
    # No stacktrace found for following nodes
    mm_default_48: "f32[1024, 768]" = torch.ops.aten.mm.default(view_164, permute_87);  view_164 = permute_87 = None
    add_tensor_48: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_48, arg133_1);  mm_default_48 = arg133_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:213, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_165: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_48, [1, 1024, 768]);  add_tensor_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_166: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_165, [1, -1, 12, 64]);  view_165 = None
    permute_88: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_166, [0, 2, 1, 3]);  view_166 = None
    clone_66: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_88, memory_format = torch.contiguous_format);  permute_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:228, code: key_states = key_states.reshape(*proj_shape)
    view_172: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_66, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:232, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_92: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_172, [0, 2, 1]);  view_172 = None
    bmm_16: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_171, permute_92);  view_171 = permute_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:245, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_174: "f32[1, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_16, [1, 12, 1024, 1024]);  bmm_16 = None
    add_62: "f32[1, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_174, expand_4);  view_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:246, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_175: "f32[12, 1024, 1024]" = torch.ops.aten.reshape.default(add_62, [12, 1024, 1024]);  add_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:248, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_8: "f32[12, 1024, 1]" = torch.ops.aten.amax.default(view_175, [-1], True)
    sub_26: "f32[12, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_175, amax_8);  view_175 = amax_8 = None
    exp_8: "f32[12, 1024, 1024]" = torch.ops.aten.exp.default(sub_26);  sub_26 = None
    sum_10: "f32[12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_8: "f32[12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_8, sum_10);  exp_8 = sum_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:214, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_167: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_61, [1024, 768])
    permute_89: "f32[768, 768]" = torch.ops.aten.permute.default(arg134_1, [1, 0]);  arg134_1 = None
    
    # No stacktrace found for following nodes
    mm_default_47: "f32[1024, 768]" = torch.ops.aten.mm.default(view_167, permute_89);  view_167 = permute_89 = None
    add_tensor_47: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_47, arg135_1);  mm_default_47 = arg135_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:214, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_168: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_47, [1, 1024, 768]);  add_tensor_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_169: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_168, [1, -1, 12, 64]);  view_168 = None
    permute_90: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_169, [0, 2, 1, 3]);  view_169 = None
    clone_67: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:229, code: value_states = value_states.reshape(*proj_shape)
    view_173: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_67, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:271, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_17: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(div_8, view_173);  div_8 = view_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:279, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_176: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_17, [1, 12, 1024, 64]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:280, code: attn_output = attn_output.transpose(1, 2)
    permute_93: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_176, [0, 2, 1, 3]);  view_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:284, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_70: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
    view_177: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_70, [1, 1024, 768]);  clone_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_178: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_177, [1024, 768]);  view_177 = None
    permute_94: "f32[768, 768]" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
    
    # No stacktrace found for following nodes
    mm_default_46: "f32[1024, 768]" = torch.ops.aten.mm.default(view_178, permute_94);  view_178 = permute_94 = None
    add_tensor_46: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_46, arg137_1);  mm_default_46 = arg137_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_179: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_46, [1, 1024, 768]);  add_tensor_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:432, code: hidden_states = residual + hidden_states
    add_63: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_61, view_179);  add_61 = view_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:433, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_17 = torch.ops.aten.var_mean.correction(add_63, [2], correction = 0, keepdim = True)
    getitem_34: "f32[1, 1024, 1]" = var_mean_17[0]
    getitem_35: "f32[1, 1024, 1]" = var_mean_17[1];  var_mean_17 = None
    sub_27: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_63, getitem_35);  add_63 = getitem_35 = None
    add_64: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
    rsqrt_17: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
    mul_66: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_17);  sub_27 = rsqrt_17 = None
    mul_67: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_66, arg138_1);  mul_66 = arg138_1 = None
    add_65: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_67, arg139_1);  mul_67 = arg139_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_180: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_65, [1024, 768])
    permute_95: "f32[768, 768]" = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
    
    # No stacktrace found for following nodes
    mm_default_45: "f32[1024, 768]" = torch.ops.aten.mm.default(view_180, permute_95);  view_180 = permute_95 = None
    add_tensor_45: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_45, arg141_1);  mm_default_45 = arg141_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_181: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_45, [1, 1024, 768]);  add_tensor_45 = None
    mul_68: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_181, 0.125);  view_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_188: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(mul_68, [1, 1024, 12, 64]);  mul_68 = None
    permute_100: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_188, [0, 2, 1, 3]);  view_188 = None
    clone_74: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_100, memory_format = torch.contiguous_format);  permute_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:227, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_189: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_74, [12, -1, 64]);  clone_74 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_12: "f32[1, 12, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_189, 0);  view_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:203, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_182: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_45, [1024, 768])
    permute_96: "f32[768, 768]" = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
    
    # No stacktrace found for following nodes
    mm_default_44: "f32[1024, 768]" = torch.ops.aten.mm.default(view_182, permute_96);  view_182 = permute_96 = None
    add_tensor_44: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_44, arg143_1);  mm_default_44 = arg143_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:203, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_183: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_44, [1, 1024, 768]);  add_tensor_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_184: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_183, [1, -1, 12, 64]);  view_183 = None
    permute_97: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
    clone_72: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_97, memory_format = torch.contiguous_format);  permute_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:228, code: key_states = key_states.reshape(*proj_shape)
    view_190: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_72, [12, -1, 64])
    
    # No stacktrace found for following nodes
    unsqueeze_default_13: "f32[1, 12, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_190, 0);  view_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:204, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_185: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_45, [1024, 768])
    permute_98: "f32[768, 768]" = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
    
    # No stacktrace found for following nodes
    mm_default_43: "f32[1024, 768]" = torch.ops.aten.mm.default(view_185, permute_98);  view_185 = permute_98 = None
    add_tensor_43: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_43, arg145_1);  mm_default_43 = arg145_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:204, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_186: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_43, [1, 1024, 768]);  add_tensor_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_187: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_186, [1, -1, 12, 64]);  view_186 = None
    permute_99: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_187, [0, 2, 1, 3]);  view_187 = None
    clone_73: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_99, memory_format = torch.contiguous_format);  permute_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:229, code: value_states = value_states.reshape(*proj_shape)
    view_191: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_73, [12, -1, 64])
    
    # No stacktrace found for following nodes
    unsqueeze_default_14: "f32[1, 12, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_191, 0);  view_191 = None
    _scaled_dot_product_efficient_attention_default_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_12, unsqueeze_default_13, unsqueeze_default_14, None, True, scale = 1.0);  unsqueeze_default_12 = unsqueeze_default_13 = unsqueeze_default_14 = None
    getitem_68: "f32[1, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_default_4[0];  _scaled_dot_product_efficient_attention_default_4 = None
    squeeze_dim_4: "f32[12, 1024, 64]" = torch.ops.aten.squeeze.dim(getitem_68, 0);  getitem_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:279, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_192: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(squeeze_dim_4, [1, 12, 1024, 64]);  squeeze_dim_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:280, code: attn_output = attn_output.transpose(1, 2)
    permute_102: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:284, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_76: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_102, memory_format = torch.contiguous_format);  permute_102 = None
    view_193: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_76, [1, 1024, 768]);  clone_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_194: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_193, [1024, 768]);  view_193 = None
    permute_103: "f32[768, 768]" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
    
    # No stacktrace found for following nodes
    mm_default_42: "f32[1024, 768]" = torch.ops.aten.mm.default(view_194, permute_103);  view_194 = permute_103 = None
    add_tensor_42: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_42, arg147_1);  mm_default_42 = arg147_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_195: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_42, [1, 1024, 768]);  add_tensor_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:452, code: hidden_states = residual + hidden_states
    add_66: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_65, view_195);  add_65 = view_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:453, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_18 = torch.ops.aten.var_mean.correction(add_66, [2], correction = 0, keepdim = True)
    getitem_36: "f32[1, 1024, 1]" = var_mean_18[0]
    getitem_37: "f32[1, 1024, 1]" = var_mean_18[1];  var_mean_18 = None
    sub_29: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_66, getitem_37);  add_66 = getitem_37 = None
    add_67: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
    rsqrt_18: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
    mul_69: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_18);  sub_29 = rsqrt_18 = None
    mul_70: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_69, arg148_1);  mul_69 = arg148_1 = None
    add_68: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_70, arg149_1);  mul_70 = arg149_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_196: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_68, [1024, 768])
    permute_104: "f32[768, 3072]" = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
    
    # No stacktrace found for following nodes
    mm_default_41: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_196, permute_104);  view_196 = permute_104 = None
    add_tensor_41: "f32[1024, 3072]" = torch.ops.aten.add.Tensor(mm_default_41, arg151_1);  mm_default_41 = arg151_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_197: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(add_tensor_41, [1, 1024, 3072]);  add_tensor_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_71: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_197, 0.5)
    mul_72: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_197, 0.7071067811865476);  view_197 = None
    erf_7: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_72);  mul_72 = None
    add_69: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_73: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_71, add_69);  mul_71 = add_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_198: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_73, [1024, 3072]);  mul_73 = None
    permute_105: "f32[3072, 768]" = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
    
    # No stacktrace found for following nodes
    mm_default_40: "f32[1024, 768]" = torch.ops.aten.mm.default(view_198, permute_105);  view_198 = permute_105 = None
    add_tensor_40: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_40, arg153_1);  mm_default_40 = arg153_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_199: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_40, [1, 1024, 768]);  add_tensor_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:464, code: hidden_states = residual + hidden_states
    add_70: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_68, view_199);  add_68 = view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:465, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_19 = torch.ops.aten.var_mean.correction(add_70, [2], correction = 0, keepdim = True)
    getitem_38: "f32[1, 1024, 1]" = var_mean_19[0]
    getitem_39: "f32[1, 1024, 1]" = var_mean_19[1];  var_mean_19 = None
    sub_30: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_70, getitem_39);  add_70 = getitem_39 = None
    add_71: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
    rsqrt_19: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
    mul_74: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_19);  sub_30 = rsqrt_19 = None
    mul_75: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_74, arg154_1);  mul_74 = arg154_1 = None
    add_72: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_75, arg155_1);  mul_75 = arg155_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_200: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_72, [1024, 768])
    permute_106: "f32[768, 768]" = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
    
    # No stacktrace found for following nodes
    mm_default_39: "f32[1024, 768]" = torch.ops.aten.mm.default(view_200, permute_106);  view_200 = permute_106 = None
    add_tensor_39: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_39, arg157_1);  mm_default_39 = arg157_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_201: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_39, [1, 1024, 768]);  add_tensor_39 = None
    mul_76: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_201, 0.125);  view_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_208: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(mul_76, [1, 1024, 12, 64]);  mul_76 = None
    permute_111: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None
    clone_82: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_111, memory_format = torch.contiguous_format);  permute_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:227, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_209: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_82, [12, -1, 64]);  clone_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:213, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_202: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_72, [1024, 768])
    permute_107: "f32[768, 768]" = torch.ops.aten.permute.default(arg158_1, [1, 0]);  arg158_1 = None
    
    # No stacktrace found for following nodes
    mm_default_38: "f32[1024, 768]" = torch.ops.aten.mm.default(view_202, permute_107);  view_202 = permute_107 = None
    add_tensor_38: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_38, arg159_1);  mm_default_38 = arg159_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:213, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_203: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_38, [1, 1024, 768]);  add_tensor_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_204: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_203, [1, -1, 12, 64]);  view_203 = None
    permute_108: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_204, [0, 2, 1, 3]);  view_204 = None
    clone_80: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_108, memory_format = torch.contiguous_format);  permute_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:228, code: key_states = key_states.reshape(*proj_shape)
    view_210: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_80, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:232, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_112: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None
    bmm_20: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_209, permute_112);  view_209 = permute_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:245, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_212: "f32[1, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_20, [1, 12, 1024, 1024]);  bmm_20 = None
    add_73: "f32[1, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_212, expand_4);  view_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:246, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_213: "f32[12, 1024, 1024]" = torch.ops.aten.reshape.default(add_73, [12, 1024, 1024]);  add_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:248, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_10: "f32[12, 1024, 1]" = torch.ops.aten.amax.default(view_213, [-1], True)
    sub_31: "f32[12, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_213, amax_10);  view_213 = amax_10 = None
    exp_10: "f32[12, 1024, 1024]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
    sum_12: "f32[12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_10: "f32[12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_10, sum_12);  exp_10 = sum_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:214, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_205: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_72, [1024, 768])
    permute_109: "f32[768, 768]" = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
    
    # No stacktrace found for following nodes
    mm_default_37: "f32[1024, 768]" = torch.ops.aten.mm.default(view_205, permute_109);  view_205 = permute_109 = None
    add_tensor_37: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_37, arg161_1);  mm_default_37 = arg161_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:214, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_206: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_37, [1, 1024, 768]);  add_tensor_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_207: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_206, [1, -1, 12, 64]);  view_206 = None
    permute_110: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_207, [0, 2, 1, 3]);  view_207 = None
    clone_81: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_110, memory_format = torch.contiguous_format);  permute_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:229, code: value_states = value_states.reshape(*proj_shape)
    view_211: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_81, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:271, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_21: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(div_10, view_211);  div_10 = view_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:279, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_214: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_21, [1, 12, 1024, 64]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:280, code: attn_output = attn_output.transpose(1, 2)
    permute_113: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_214, [0, 2, 1, 3]);  view_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:284, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_84: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_113, memory_format = torch.contiguous_format);  permute_113 = None
    view_215: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_84, [1, 1024, 768]);  clone_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_216: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_215, [1024, 768]);  view_215 = None
    permute_114: "f32[768, 768]" = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
    
    # No stacktrace found for following nodes
    mm_default_36: "f32[1024, 768]" = torch.ops.aten.mm.default(view_216, permute_114);  view_216 = permute_114 = None
    add_tensor_36: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_36, arg163_1);  mm_default_36 = arg163_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_217: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_36, [1, 1024, 768]);  add_tensor_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:432, code: hidden_states = residual + hidden_states
    add_74: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_72, view_217);  add_72 = view_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:433, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_20 = torch.ops.aten.var_mean.correction(add_74, [2], correction = 0, keepdim = True)
    getitem_40: "f32[1, 1024, 1]" = var_mean_20[0]
    getitem_41: "f32[1, 1024, 1]" = var_mean_20[1];  var_mean_20 = None
    sub_32: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_74, getitem_41);  add_74 = getitem_41 = None
    add_75: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
    rsqrt_20: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
    mul_77: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_20);  sub_32 = rsqrt_20 = None
    mul_78: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_77, arg164_1);  mul_77 = arg164_1 = None
    add_76: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_78, arg165_1);  mul_78 = arg165_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_218: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_76, [1024, 768])
    permute_115: "f32[768, 768]" = torch.ops.aten.permute.default(arg166_1, [1, 0]);  arg166_1 = None
    
    # No stacktrace found for following nodes
    mm_default_35: "f32[1024, 768]" = torch.ops.aten.mm.default(view_218, permute_115);  view_218 = permute_115 = None
    add_tensor_35: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_35, arg167_1);  mm_default_35 = arg167_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_219: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_35, [1, 1024, 768]);  add_tensor_35 = None
    mul_79: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_219, 0.125);  view_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_226: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(mul_79, [1, 1024, 12, 64]);  mul_79 = None
    permute_120: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
    clone_88: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_120, memory_format = torch.contiguous_format);  permute_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:227, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_227: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_88, [12, -1, 64]);  clone_88 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_9: "f32[1, 12, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_227, 0);  view_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:203, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_220: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_45, [1024, 768])
    permute_116: "f32[768, 768]" = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
    
    # No stacktrace found for following nodes
    mm_default_34: "f32[1024, 768]" = torch.ops.aten.mm.default(view_220, permute_116);  view_220 = permute_116 = None
    add_tensor_34: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_34, arg169_1);  mm_default_34 = arg169_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:203, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_221: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_34, [1, 1024, 768]);  add_tensor_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_222: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_221, [1, -1, 12, 64]);  view_221 = None
    permute_117: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None
    clone_86: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:228, code: key_states = key_states.reshape(*proj_shape)
    view_228: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_86, [12, -1, 64])
    
    # No stacktrace found for following nodes
    unsqueeze_default_10: "f32[1, 12, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_228, 0);  view_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:204, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_223: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_45, [1024, 768])
    permute_118: "f32[768, 768]" = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
    
    # No stacktrace found for following nodes
    mm_default_33: "f32[1024, 768]" = torch.ops.aten.mm.default(view_223, permute_118);  view_223 = permute_118 = None
    add_tensor_33: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_33, arg171_1);  mm_default_33 = arg171_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:204, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_224: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_33, [1, 1024, 768]);  add_tensor_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_225: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_224, [1, -1, 12, 64]);  view_224 = None
    permute_119: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_225, [0, 2, 1, 3]);  view_225 = None
    clone_87: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_119, memory_format = torch.contiguous_format);  permute_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:229, code: value_states = value_states.reshape(*proj_shape)
    view_229: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_87, [12, -1, 64])
    
    # No stacktrace found for following nodes
    unsqueeze_default_11: "f32[1, 12, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_229, 0);  view_229 = None
    _scaled_dot_product_efficient_attention_default_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_9, unsqueeze_default_10, unsqueeze_default_11, None, True, scale = 1.0);  unsqueeze_default_9 = unsqueeze_default_10 = unsqueeze_default_11 = None
    getitem_67: "f32[1, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_default_3[0];  _scaled_dot_product_efficient_attention_default_3 = None
    squeeze_dim_3: "f32[12, 1024, 64]" = torch.ops.aten.squeeze.dim(getitem_67, 0);  getitem_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:279, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_230: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(squeeze_dim_3, [1, 12, 1024, 64]);  squeeze_dim_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:280, code: attn_output = attn_output.transpose(1, 2)
    permute_122: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_230, [0, 2, 1, 3]);  view_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:284, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_90: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_122, memory_format = torch.contiguous_format);  permute_122 = None
    view_231: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_90, [1, 1024, 768]);  clone_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_232: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_231, [1024, 768]);  view_231 = None
    permute_123: "f32[768, 768]" = torch.ops.aten.permute.default(arg172_1, [1, 0]);  arg172_1 = None
    
    # No stacktrace found for following nodes
    mm_default_32: "f32[1024, 768]" = torch.ops.aten.mm.default(view_232, permute_123);  view_232 = permute_123 = None
    add_tensor_32: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_32, arg173_1);  mm_default_32 = arg173_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_233: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_32, [1, 1024, 768]);  add_tensor_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:452, code: hidden_states = residual + hidden_states
    add_77: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_76, view_233);  add_76 = view_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:453, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_21 = torch.ops.aten.var_mean.correction(add_77, [2], correction = 0, keepdim = True)
    getitem_42: "f32[1, 1024, 1]" = var_mean_21[0]
    getitem_43: "f32[1, 1024, 1]" = var_mean_21[1];  var_mean_21 = None
    sub_34: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_77, getitem_43);  add_77 = getitem_43 = None
    add_78: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
    rsqrt_21: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    mul_80: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_21);  sub_34 = rsqrt_21 = None
    mul_81: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_80, arg174_1);  mul_80 = arg174_1 = None
    add_79: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_81, arg175_1);  mul_81 = arg175_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_234: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_79, [1024, 768])
    permute_124: "f32[768, 3072]" = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
    
    # No stacktrace found for following nodes
    mm_default_31: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_234, permute_124);  view_234 = permute_124 = None
    add_tensor_31: "f32[1024, 3072]" = torch.ops.aten.add.Tensor(mm_default_31, arg177_1);  mm_default_31 = arg177_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_235: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(add_tensor_31, [1, 1024, 3072]);  add_tensor_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_82: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_235, 0.5)
    mul_83: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_235, 0.7071067811865476);  view_235 = None
    erf_8: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
    add_80: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_84: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_82, add_80);  mul_82 = add_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_236: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_84, [1024, 3072]);  mul_84 = None
    permute_125: "f32[3072, 768]" = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
    
    # No stacktrace found for following nodes
    mm_default_30: "f32[1024, 768]" = torch.ops.aten.mm.default(view_236, permute_125);  view_236 = permute_125 = None
    add_tensor_30: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_30, arg179_1);  mm_default_30 = arg179_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_237: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_30, [1, 1024, 768]);  add_tensor_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:464, code: hidden_states = residual + hidden_states
    add_81: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_79, view_237);  add_79 = view_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:465, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_22 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
    getitem_44: "f32[1, 1024, 1]" = var_mean_22[0]
    getitem_45: "f32[1, 1024, 1]" = var_mean_22[1];  var_mean_22 = None
    sub_35: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_81, getitem_45);  add_81 = getitem_45 = None
    add_82: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
    rsqrt_22: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
    mul_85: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_22);  sub_35 = rsqrt_22 = None
    mul_86: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_85, arg180_1);  mul_85 = arg180_1 = None
    add_83: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_86, arg181_1);  mul_86 = arg181_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_238: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_83, [1024, 768])
    permute_126: "f32[768, 768]" = torch.ops.aten.permute.default(arg182_1, [1, 0]);  arg182_1 = None
    
    # No stacktrace found for following nodes
    mm_default_29: "f32[1024, 768]" = torch.ops.aten.mm.default(view_238, permute_126);  view_238 = permute_126 = None
    add_tensor_29: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_29, arg183_1);  mm_default_29 = arg183_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_239: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_29, [1, 1024, 768]);  add_tensor_29 = None
    mul_87: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_239, 0.125);  view_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_246: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(mul_87, [1, 1024, 12, 64]);  mul_87 = None
    permute_131: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None
    clone_96: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_131, memory_format = torch.contiguous_format);  permute_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:227, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_247: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_96, [12, -1, 64]);  clone_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:213, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_240: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_83, [1024, 768])
    permute_127: "f32[768, 768]" = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
    
    # No stacktrace found for following nodes
    mm_default_28: "f32[1024, 768]" = torch.ops.aten.mm.default(view_240, permute_127);  view_240 = permute_127 = None
    add_tensor_28: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_28, arg185_1);  mm_default_28 = arg185_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:213, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_241: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_28, [1, 1024, 768]);  add_tensor_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_242: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_241, [1, -1, 12, 64]);  view_241 = None
    permute_128: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_242, [0, 2, 1, 3]);  view_242 = None
    clone_94: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:228, code: key_states = key_states.reshape(*proj_shape)
    view_248: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_94, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:232, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_132: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_248, [0, 2, 1]);  view_248 = None
    bmm_24: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_247, permute_132);  view_247 = permute_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:245, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_250: "f32[1, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_24, [1, 12, 1024, 1024]);  bmm_24 = None
    add_84: "f32[1, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_250, expand_4);  view_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:246, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_251: "f32[12, 1024, 1024]" = torch.ops.aten.reshape.default(add_84, [12, 1024, 1024]);  add_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:248, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_12: "f32[12, 1024, 1]" = torch.ops.aten.amax.default(view_251, [-1], True)
    sub_36: "f32[12, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_251, amax_12);  view_251 = amax_12 = None
    exp_12: "f32[12, 1024, 1024]" = torch.ops.aten.exp.default(sub_36);  sub_36 = None
    sum_14: "f32[12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
    div_12: "f32[12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_12, sum_14);  exp_12 = sum_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:214, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_243: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_83, [1024, 768])
    permute_129: "f32[768, 768]" = torch.ops.aten.permute.default(arg186_1, [1, 0]);  arg186_1 = None
    
    # No stacktrace found for following nodes
    mm_default_27: "f32[1024, 768]" = torch.ops.aten.mm.default(view_243, permute_129);  view_243 = permute_129 = None
    add_tensor_27: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_27, arg187_1);  mm_default_27 = arg187_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:214, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_244: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_27, [1, 1024, 768]);  add_tensor_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_245: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_244, [1, -1, 12, 64]);  view_244 = None
    permute_130: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_245, [0, 2, 1, 3]);  view_245 = None
    clone_95: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_130, memory_format = torch.contiguous_format);  permute_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:229, code: value_states = value_states.reshape(*proj_shape)
    view_249: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_95, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:271, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_25: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(div_12, view_249);  div_12 = view_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:279, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_252: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_25, [1, 12, 1024, 64]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:280, code: attn_output = attn_output.transpose(1, 2)
    permute_133: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:284, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_98: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_133, memory_format = torch.contiguous_format);  permute_133 = None
    view_253: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_98, [1, 1024, 768]);  clone_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_254: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_253, [1024, 768]);  view_253 = None
    permute_134: "f32[768, 768]" = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
    
    # No stacktrace found for following nodes
    mm_default_26: "f32[1024, 768]" = torch.ops.aten.mm.default(view_254, permute_134);  view_254 = permute_134 = None
    add_tensor_26: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_26, arg189_1);  mm_default_26 = arg189_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_255: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_26, [1, 1024, 768]);  add_tensor_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:432, code: hidden_states = residual + hidden_states
    add_85: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_83, view_255);  add_83 = view_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:433, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_23 = torch.ops.aten.var_mean.correction(add_85, [2], correction = 0, keepdim = True)
    getitem_46: "f32[1, 1024, 1]" = var_mean_23[0]
    getitem_47: "f32[1, 1024, 1]" = var_mean_23[1];  var_mean_23 = None
    sub_37: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_85, getitem_47);  add_85 = getitem_47 = None
    add_86: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
    rsqrt_23: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
    mul_88: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_23);  sub_37 = rsqrt_23 = None
    mul_89: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_88, arg190_1);  mul_88 = arg190_1 = None
    add_87: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_89, arg191_1);  mul_89 = arg191_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_256: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_87, [1024, 768])
    permute_135: "f32[768, 768]" = torch.ops.aten.permute.default(arg192_1, [1, 0]);  arg192_1 = None
    
    # No stacktrace found for following nodes
    mm_default_25: "f32[1024, 768]" = torch.ops.aten.mm.default(view_256, permute_135);  view_256 = permute_135 = None
    add_tensor_25: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_25, arg193_1);  mm_default_25 = arg193_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_257: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_25, [1, 1024, 768]);  add_tensor_25 = None
    mul_90: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_257, 0.125);  view_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_264: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(mul_90, [1, 1024, 12, 64]);  mul_90 = None
    permute_140: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_264, [0, 2, 1, 3]);  view_264 = None
    clone_102: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_140, memory_format = torch.contiguous_format);  permute_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:227, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_265: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_102, [12, -1, 64]);  clone_102 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_6: "f32[1, 12, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_265, 0);  view_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:203, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_258: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_45, [1024, 768])
    permute_136: "f32[768, 768]" = torch.ops.aten.permute.default(arg194_1, [1, 0]);  arg194_1 = None
    
    # No stacktrace found for following nodes
    mm_default_24: "f32[1024, 768]" = torch.ops.aten.mm.default(view_258, permute_136);  view_258 = permute_136 = None
    add_tensor_24: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_24, arg195_1);  mm_default_24 = arg195_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:203, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_259: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_24, [1, 1024, 768]);  add_tensor_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_260: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_259, [1, -1, 12, 64]);  view_259 = None
    permute_137: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_260, [0, 2, 1, 3]);  view_260 = None
    clone_100: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_137, memory_format = torch.contiguous_format);  permute_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:228, code: key_states = key_states.reshape(*proj_shape)
    view_266: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_100, [12, -1, 64])
    
    # No stacktrace found for following nodes
    unsqueeze_default_7: "f32[1, 12, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_266, 0);  view_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:204, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_261: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_45, [1024, 768])
    permute_138: "f32[768, 768]" = torch.ops.aten.permute.default(arg196_1, [1, 0]);  arg196_1 = None
    
    # No stacktrace found for following nodes
    mm_default_23: "f32[1024, 768]" = torch.ops.aten.mm.default(view_261, permute_138);  view_261 = permute_138 = None
    add_tensor_23: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_23, arg197_1);  mm_default_23 = arg197_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:204, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_262: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_23, [1, 1024, 768]);  add_tensor_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_263: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_262, [1, -1, 12, 64]);  view_262 = None
    permute_139: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_263, [0, 2, 1, 3]);  view_263 = None
    clone_101: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:229, code: value_states = value_states.reshape(*proj_shape)
    view_267: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_101, [12, -1, 64])
    
    # No stacktrace found for following nodes
    unsqueeze_default_8: "f32[1, 12, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_267, 0);  view_267 = None
    _scaled_dot_product_efficient_attention_default_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_6, unsqueeze_default_7, unsqueeze_default_8, None, True, scale = 1.0);  unsqueeze_default_6 = unsqueeze_default_7 = unsqueeze_default_8 = None
    getitem_66: "f32[1, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_default_2[0];  _scaled_dot_product_efficient_attention_default_2 = None
    squeeze_dim_2: "f32[12, 1024, 64]" = torch.ops.aten.squeeze.dim(getitem_66, 0);  getitem_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:279, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_268: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(squeeze_dim_2, [1, 12, 1024, 64]);  squeeze_dim_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:280, code: attn_output = attn_output.transpose(1, 2)
    permute_142: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_268, [0, 2, 1, 3]);  view_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:284, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_104: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_142, memory_format = torch.contiguous_format);  permute_142 = None
    view_269: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_104, [1, 1024, 768]);  clone_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_270: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_269, [1024, 768]);  view_269 = None
    permute_143: "f32[768, 768]" = torch.ops.aten.permute.default(arg198_1, [1, 0]);  arg198_1 = None
    
    # No stacktrace found for following nodes
    mm_default_22: "f32[1024, 768]" = torch.ops.aten.mm.default(view_270, permute_143);  view_270 = permute_143 = None
    add_tensor_22: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_22, arg199_1);  mm_default_22 = arg199_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_271: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_22, [1, 1024, 768]);  add_tensor_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:452, code: hidden_states = residual + hidden_states
    add_88: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_87, view_271);  add_87 = view_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:453, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_24 = torch.ops.aten.var_mean.correction(add_88, [2], correction = 0, keepdim = True)
    getitem_48: "f32[1, 1024, 1]" = var_mean_24[0]
    getitem_49: "f32[1, 1024, 1]" = var_mean_24[1];  var_mean_24 = None
    sub_39: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_88, getitem_49);  add_88 = getitem_49 = None
    add_89: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
    rsqrt_24: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
    mul_91: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_24);  sub_39 = rsqrt_24 = None
    mul_92: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_91, arg200_1);  mul_91 = arg200_1 = None
    add_90: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_92, arg201_1);  mul_92 = arg201_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_272: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_90, [1024, 768])
    permute_144: "f32[768, 3072]" = torch.ops.aten.permute.default(arg202_1, [1, 0]);  arg202_1 = None
    
    # No stacktrace found for following nodes
    mm_default_21: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_272, permute_144);  view_272 = permute_144 = None
    add_tensor_21: "f32[1024, 3072]" = torch.ops.aten.add.Tensor(mm_default_21, arg203_1);  mm_default_21 = arg203_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_273: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(add_tensor_21, [1, 1024, 3072]);  add_tensor_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_93: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_273, 0.5)
    mul_94: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_273, 0.7071067811865476);  view_273 = None
    erf_9: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_94);  mul_94 = None
    add_91: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_95: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_93, add_91);  mul_93 = add_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_274: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_95, [1024, 3072]);  mul_95 = None
    permute_145: "f32[3072, 768]" = torch.ops.aten.permute.default(arg204_1, [1, 0]);  arg204_1 = None
    
    # No stacktrace found for following nodes
    mm_default_20: "f32[1024, 768]" = torch.ops.aten.mm.default(view_274, permute_145);  view_274 = permute_145 = None
    add_tensor_20: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_20, arg205_1);  mm_default_20 = arg205_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_275: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_20, [1, 1024, 768]);  add_tensor_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:464, code: hidden_states = residual + hidden_states
    add_92: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_90, view_275);  add_90 = view_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:465, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_25 = torch.ops.aten.var_mean.correction(add_92, [2], correction = 0, keepdim = True)
    getitem_50: "f32[1, 1024, 1]" = var_mean_25[0]
    getitem_51: "f32[1, 1024, 1]" = var_mean_25[1];  var_mean_25 = None
    sub_40: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_92, getitem_51);  add_92 = getitem_51 = None
    add_93: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
    rsqrt_25: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
    mul_96: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_25);  sub_40 = rsqrt_25 = None
    mul_97: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_96, arg206_1);  mul_96 = arg206_1 = None
    add_94: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_97, arg207_1);  mul_97 = arg207_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_276: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_94, [1024, 768])
    permute_146: "f32[768, 768]" = torch.ops.aten.permute.default(arg208_1, [1, 0]);  arg208_1 = None
    
    # No stacktrace found for following nodes
    mm_default_19: "f32[1024, 768]" = torch.ops.aten.mm.default(view_276, permute_146);  view_276 = permute_146 = None
    add_tensor_19: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_19, arg209_1);  mm_default_19 = arg209_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_277: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_19, [1, 1024, 768]);  add_tensor_19 = None
    mul_98: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_277, 0.125);  view_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_284: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(mul_98, [1, 1024, 12, 64]);  mul_98 = None
    permute_151: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_284, [0, 2, 1, 3]);  view_284 = None
    clone_110: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_151, memory_format = torch.contiguous_format);  permute_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:227, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_285: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_110, [12, -1, 64]);  clone_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:213, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_278: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_94, [1024, 768])
    permute_147: "f32[768, 768]" = torch.ops.aten.permute.default(arg210_1, [1, 0]);  arg210_1 = None
    
    # No stacktrace found for following nodes
    mm_default_18: "f32[1024, 768]" = torch.ops.aten.mm.default(view_278, permute_147);  view_278 = permute_147 = None
    add_tensor_18: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_18, arg211_1);  mm_default_18 = arg211_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:213, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_279: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_18, [1, 1024, 768]);  add_tensor_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_280: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_279, [1, -1, 12, 64]);  view_279 = None
    permute_148: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_280, [0, 2, 1, 3]);  view_280 = None
    clone_108: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_148, memory_format = torch.contiguous_format);  permute_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:228, code: key_states = key_states.reshape(*proj_shape)
    view_286: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_108, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:232, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_152: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_286, [0, 2, 1]);  view_286 = None
    bmm_28: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_285, permute_152);  view_285 = permute_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:245, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_288: "f32[1, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_28, [1, 12, 1024, 1024]);  bmm_28 = None
    add_95: "f32[1, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_288, expand_4);  view_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:246, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_289: "f32[12, 1024, 1024]" = torch.ops.aten.reshape.default(add_95, [12, 1024, 1024]);  add_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:248, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_14: "f32[12, 1024, 1]" = torch.ops.aten.amax.default(view_289, [-1], True)
    sub_41: "f32[12, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_289, amax_14);  view_289 = amax_14 = None
    exp_14: "f32[12, 1024, 1024]" = torch.ops.aten.exp.default(sub_41);  sub_41 = None
    sum_16: "f32[12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
    div_14: "f32[12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_14, sum_16);  exp_14 = sum_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:214, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_281: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_94, [1024, 768])
    permute_149: "f32[768, 768]" = torch.ops.aten.permute.default(arg212_1, [1, 0]);  arg212_1 = None
    
    # No stacktrace found for following nodes
    mm_default_17: "f32[1024, 768]" = torch.ops.aten.mm.default(view_281, permute_149);  view_281 = permute_149 = None
    add_tensor_17: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_17, arg213_1);  mm_default_17 = arg213_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:214, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_282: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_17, [1, 1024, 768]);  add_tensor_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_283: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_282, [1, -1, 12, 64]);  view_282 = None
    permute_150: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_283, [0, 2, 1, 3]);  view_283 = None
    clone_109: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:229, code: value_states = value_states.reshape(*proj_shape)
    view_287: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_109, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:271, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_29: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(div_14, view_287);  div_14 = view_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:279, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_290: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_29, [1, 12, 1024, 64]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:280, code: attn_output = attn_output.transpose(1, 2)
    permute_153: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_290, [0, 2, 1, 3]);  view_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:284, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_112: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_153, memory_format = torch.contiguous_format);  permute_153 = None
    view_291: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_112, [1, 1024, 768]);  clone_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_292: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_291, [1024, 768]);  view_291 = None
    permute_154: "f32[768, 768]" = torch.ops.aten.permute.default(arg214_1, [1, 0]);  arg214_1 = None
    
    # No stacktrace found for following nodes
    mm_default_16: "f32[1024, 768]" = torch.ops.aten.mm.default(view_292, permute_154);  view_292 = permute_154 = None
    add_tensor_16: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_16, arg215_1);  mm_default_16 = arg215_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_293: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_16, [1, 1024, 768]);  add_tensor_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:432, code: hidden_states = residual + hidden_states
    add_96: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_94, view_293);  add_94 = view_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:433, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_26 = torch.ops.aten.var_mean.correction(add_96, [2], correction = 0, keepdim = True)
    getitem_52: "f32[1, 1024, 1]" = var_mean_26[0]
    getitem_53: "f32[1, 1024, 1]" = var_mean_26[1];  var_mean_26 = None
    sub_42: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_96, getitem_53);  add_96 = getitem_53 = None
    add_97: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
    rsqrt_26: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
    mul_99: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_26);  sub_42 = rsqrt_26 = None
    mul_100: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_99, arg216_1);  mul_99 = arg216_1 = None
    add_98: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_100, arg217_1);  mul_100 = arg217_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_294: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_98, [1024, 768])
    permute_155: "f32[768, 768]" = torch.ops.aten.permute.default(arg218_1, [1, 0]);  arg218_1 = None
    
    # No stacktrace found for following nodes
    mm_default_15: "f32[1024, 768]" = torch.ops.aten.mm.default(view_294, permute_155);  view_294 = permute_155 = None
    add_tensor_15: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_15, arg219_1);  mm_default_15 = arg219_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_295: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_15, [1, 1024, 768]);  add_tensor_15 = None
    mul_101: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_295, 0.125);  view_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_302: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(mul_101, [1, 1024, 12, 64]);  mul_101 = None
    permute_160: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_302, [0, 2, 1, 3]);  view_302 = None
    clone_116: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_160, memory_format = torch.contiguous_format);  permute_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:227, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_303: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_116, [12, -1, 64]);  clone_116 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_3: "f32[1, 12, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_303, 0);  view_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:203, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_296: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_45, [1024, 768])
    permute_156: "f32[768, 768]" = torch.ops.aten.permute.default(arg220_1, [1, 0]);  arg220_1 = None
    
    # No stacktrace found for following nodes
    mm_default_14: "f32[1024, 768]" = torch.ops.aten.mm.default(view_296, permute_156);  view_296 = permute_156 = None
    add_tensor_14: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_14, arg221_1);  mm_default_14 = arg221_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:203, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_297: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_14, [1, 1024, 768]);  add_tensor_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_298: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_297, [1, -1, 12, 64]);  view_297 = None
    permute_157: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_298, [0, 2, 1, 3]);  view_298 = None
    clone_114: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_157, memory_format = torch.contiguous_format);  permute_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:228, code: key_states = key_states.reshape(*proj_shape)
    view_304: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_114, [12, -1, 64])
    
    # No stacktrace found for following nodes
    unsqueeze_default_4: "f32[1, 12, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_304, 0);  view_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:204, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_299: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_45, [1024, 768])
    permute_158: "f32[768, 768]" = torch.ops.aten.permute.default(arg222_1, [1, 0]);  arg222_1 = None
    
    # No stacktrace found for following nodes
    mm_default_13: "f32[1024, 768]" = torch.ops.aten.mm.default(view_299, permute_158);  view_299 = permute_158 = None
    add_tensor_13: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_13, arg223_1);  mm_default_13 = arg223_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:204, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_300: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_13, [1, 1024, 768]);  add_tensor_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_301: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_300, [1, -1, 12, 64]);  view_300 = None
    permute_159: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_301, [0, 2, 1, 3]);  view_301 = None
    clone_115: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_159, memory_format = torch.contiguous_format);  permute_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:229, code: value_states = value_states.reshape(*proj_shape)
    view_305: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_115, [12, -1, 64])
    
    # No stacktrace found for following nodes
    unsqueeze_default_5: "f32[1, 12, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_305, 0);  view_305 = None
    _scaled_dot_product_efficient_attention_default_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_3, unsqueeze_default_4, unsqueeze_default_5, None, True, scale = 1.0);  unsqueeze_default_3 = unsqueeze_default_4 = unsqueeze_default_5 = None
    getitem_65: "f32[1, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_default_1[0];  _scaled_dot_product_efficient_attention_default_1 = None
    squeeze_dim_1: "f32[12, 1024, 64]" = torch.ops.aten.squeeze.dim(getitem_65, 0);  getitem_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:279, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_306: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(squeeze_dim_1, [1, 12, 1024, 64]);  squeeze_dim_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:280, code: attn_output = attn_output.transpose(1, 2)
    permute_162: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_306, [0, 2, 1, 3]);  view_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:284, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_118: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_162, memory_format = torch.contiguous_format);  permute_162 = None
    view_307: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_118, [1, 1024, 768]);  clone_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_308: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_307, [1024, 768]);  view_307 = None
    permute_163: "f32[768, 768]" = torch.ops.aten.permute.default(arg224_1, [1, 0]);  arg224_1 = None
    
    # No stacktrace found for following nodes
    mm_default_12: "f32[1024, 768]" = torch.ops.aten.mm.default(view_308, permute_163);  view_308 = permute_163 = None
    add_tensor_12: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_12, arg225_1);  mm_default_12 = arg225_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_309: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_12, [1, 1024, 768]);  add_tensor_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:452, code: hidden_states = residual + hidden_states
    add_99: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_98, view_309);  add_98 = view_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:453, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_27 = torch.ops.aten.var_mean.correction(add_99, [2], correction = 0, keepdim = True)
    getitem_54: "f32[1, 1024, 1]" = var_mean_27[0]
    getitem_55: "f32[1, 1024, 1]" = var_mean_27[1];  var_mean_27 = None
    sub_44: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_99, getitem_55);  add_99 = getitem_55 = None
    add_100: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
    rsqrt_27: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
    mul_102: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_27);  sub_44 = rsqrt_27 = None
    mul_103: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_102, arg226_1);  mul_102 = arg226_1 = None
    add_101: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_103, arg227_1);  mul_103 = arg227_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_310: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_101, [1024, 768])
    permute_164: "f32[768, 3072]" = torch.ops.aten.permute.default(arg228_1, [1, 0]);  arg228_1 = None
    
    # No stacktrace found for following nodes
    mm_default_11: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_310, permute_164);  view_310 = permute_164 = None
    add_tensor_11: "f32[1024, 3072]" = torch.ops.aten.add.Tensor(mm_default_11, arg229_1);  mm_default_11 = arg229_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_311: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(add_tensor_11, [1, 1024, 3072]);  add_tensor_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_104: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_311, 0.5)
    mul_105: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_311, 0.7071067811865476);  view_311 = None
    erf_10: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_105);  mul_105 = None
    add_102: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_106: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_104, add_102);  mul_104 = add_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_312: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_106, [1024, 3072]);  mul_106 = None
    permute_165: "f32[3072, 768]" = torch.ops.aten.permute.default(arg230_1, [1, 0]);  arg230_1 = None
    
    # No stacktrace found for following nodes
    mm_default_10: "f32[1024, 768]" = torch.ops.aten.mm.default(view_312, permute_165);  view_312 = permute_165 = None
    add_tensor_10: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_10, arg231_1);  mm_default_10 = arg231_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_313: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_10, [1, 1024, 768]);  add_tensor_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:464, code: hidden_states = residual + hidden_states
    add_103: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_101, view_313);  add_101 = view_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:465, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_28 = torch.ops.aten.var_mean.correction(add_103, [2], correction = 0, keepdim = True)
    getitem_56: "f32[1, 1024, 1]" = var_mean_28[0]
    getitem_57: "f32[1, 1024, 1]" = var_mean_28[1];  var_mean_28 = None
    sub_45: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_103, getitem_57);  add_103 = getitem_57 = None
    add_104: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
    rsqrt_28: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
    mul_107: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_28);  sub_45 = rsqrt_28 = None
    mul_108: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_107, arg232_1);  mul_107 = arg232_1 = None
    add_105: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_108, arg233_1);  mul_108 = arg233_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_314: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_105, [1024, 768])
    permute_166: "f32[768, 768]" = torch.ops.aten.permute.default(arg234_1, [1, 0]);  arg234_1 = None
    
    # No stacktrace found for following nodes
    mm_default_9: "f32[1024, 768]" = torch.ops.aten.mm.default(view_314, permute_166);  view_314 = permute_166 = None
    add_tensor_9: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_9, arg235_1);  mm_default_9 = arg235_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_315: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_9, [1, 1024, 768]);  add_tensor_9 = None
    mul_109: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_315, 0.125);  view_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_322: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(mul_109, [1, 1024, 12, 64]);  mul_109 = None
    permute_171: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_322, [0, 2, 1, 3]);  view_322 = None
    clone_124: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_171, memory_format = torch.contiguous_format);  permute_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:227, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_323: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_124, [12, -1, 64]);  clone_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:213, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_316: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_105, [1024, 768])
    permute_167: "f32[768, 768]" = torch.ops.aten.permute.default(arg236_1, [1, 0]);  arg236_1 = None
    
    # No stacktrace found for following nodes
    mm_default_8: "f32[1024, 768]" = torch.ops.aten.mm.default(view_316, permute_167);  view_316 = permute_167 = None
    add_tensor_8: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_8, arg237_1);  mm_default_8 = arg237_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:213, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_317: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_8, [1, 1024, 768]);  add_tensor_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_318: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_317, [1, -1, 12, 64]);  view_317 = None
    permute_168: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_318, [0, 2, 1, 3]);  view_318 = None
    clone_122: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_168, memory_format = torch.contiguous_format);  permute_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:228, code: key_states = key_states.reshape(*proj_shape)
    view_324: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_122, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:232, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_172: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_324, [0, 2, 1]);  view_324 = None
    bmm_32: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_323, permute_172);  view_323 = permute_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:245, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_326: "f32[1, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_32, [1, 12, 1024, 1024]);  bmm_32 = None
    add_106: "f32[1, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_326, expand_4);  view_326 = expand_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:246, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_327: "f32[12, 1024, 1024]" = torch.ops.aten.reshape.default(add_106, [12, 1024, 1024]);  add_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:248, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_16: "f32[12, 1024, 1]" = torch.ops.aten.amax.default(view_327, [-1], True)
    sub_46: "f32[12, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_327, amax_16);  view_327 = amax_16 = None
    exp_16: "f32[12, 1024, 1024]" = torch.ops.aten.exp.default(sub_46);  sub_46 = None
    sum_18: "f32[12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
    div_16: "f32[12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_16, sum_18);  exp_16 = sum_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:214, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_319: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_105, [1024, 768])
    permute_169: "f32[768, 768]" = torch.ops.aten.permute.default(arg238_1, [1, 0]);  arg238_1 = None
    
    # No stacktrace found for following nodes
    mm_default_7: "f32[1024, 768]" = torch.ops.aten.mm.default(view_319, permute_169);  view_319 = permute_169 = None
    add_tensor_7: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_7, arg239_1);  mm_default_7 = arg239_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:214, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_320: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_7, [1, 1024, 768]);  add_tensor_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_321: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_320, [1, -1, 12, 64]);  view_320 = None
    permute_170: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_321, [0, 2, 1, 3]);  view_321 = None
    clone_123: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_170, memory_format = torch.contiguous_format);  permute_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:229, code: value_states = value_states.reshape(*proj_shape)
    view_325: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_123, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:271, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_33: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(div_16, view_325);  div_16 = view_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:279, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_328: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_33, [1, 12, 1024, 64]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:280, code: attn_output = attn_output.transpose(1, 2)
    permute_173: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_328, [0, 2, 1, 3]);  view_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:284, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_126: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_173, memory_format = torch.contiguous_format);  permute_173 = None
    view_329: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_126, [1, 1024, 768]);  clone_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_330: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_329, [1024, 768]);  view_329 = None
    permute_174: "f32[768, 768]" = torch.ops.aten.permute.default(arg240_1, [1, 0]);  arg240_1 = None
    
    # No stacktrace found for following nodes
    mm_default_6: "f32[1024, 768]" = torch.ops.aten.mm.default(view_330, permute_174);  view_330 = permute_174 = None
    add_tensor_6: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_6, arg241_1);  mm_default_6 = arg241_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_331: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_6, [1, 1024, 768]);  add_tensor_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:432, code: hidden_states = residual + hidden_states
    add_107: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_105, view_331);  add_105 = view_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:433, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_29 = torch.ops.aten.var_mean.correction(add_107, [2], correction = 0, keepdim = True)
    getitem_58: "f32[1, 1024, 1]" = var_mean_29[0]
    getitem_59: "f32[1, 1024, 1]" = var_mean_29[1];  var_mean_29 = None
    sub_47: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_107, getitem_59);  add_107 = getitem_59 = None
    add_108: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
    rsqrt_29: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
    mul_110: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_29);  sub_47 = rsqrt_29 = None
    mul_111: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_110, arg242_1);  mul_110 = arg242_1 = None
    add_109: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_111, arg243_1);  mul_111 = arg243_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_332: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_109, [1024, 768])
    permute_175: "f32[768, 768]" = torch.ops.aten.permute.default(arg244_1, [1, 0]);  arg244_1 = None
    
    # No stacktrace found for following nodes
    mm_default_5: "f32[1024, 768]" = torch.ops.aten.mm.default(view_332, permute_175);  view_332 = permute_175 = None
    add_tensor_5: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_5, arg245_1);  mm_default_5 = arg245_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_333: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_5, [1, 1024, 768]);  add_tensor_5 = None
    mul_112: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_333, 0.125);  view_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_340: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(mul_112, [1, 1024, 12, 64]);  mul_112 = None
    permute_180: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_340, [0, 2, 1, 3]);  view_340 = None
    clone_130: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_180, memory_format = torch.contiguous_format);  permute_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:227, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_341: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_130, [12, -1, 64]);  clone_130 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default: "f32[1, 12, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_341, 0);  view_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:203, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_334: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_45, [1024, 768])
    permute_176: "f32[768, 768]" = torch.ops.aten.permute.default(arg246_1, [1, 0]);  arg246_1 = None
    
    # No stacktrace found for following nodes
    mm_default_4: "f32[1024, 768]" = torch.ops.aten.mm.default(view_334, permute_176);  view_334 = permute_176 = None
    add_tensor_4: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_4, arg247_1);  mm_default_4 = arg247_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:203, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_335: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_4, [1, 1024, 768]);  add_tensor_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_336: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_335, [1, -1, 12, 64]);  view_335 = None
    permute_177: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_336, [0, 2, 1, 3]);  view_336 = None
    clone_128: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:228, code: key_states = key_states.reshape(*proj_shape)
    view_342: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_128, [12, -1, 64])
    
    # No stacktrace found for following nodes
    unsqueeze_default_1: "f32[1, 12, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_342, 0);  view_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:204, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_337: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_45, [1024, 768])
    permute_178: "f32[768, 768]" = torch.ops.aten.permute.default(arg248_1, [1, 0]);  arg248_1 = None
    
    # No stacktrace found for following nodes
    mm_default_3: "f32[1024, 768]" = torch.ops.aten.mm.default(view_337, permute_178);  view_337 = permute_178 = None
    add_tensor_3: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_3, arg249_1);  mm_default_3 = arg249_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:204, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_338: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_3, [1, 1024, 768]);  add_tensor_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_339: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_338, [1, -1, 12, 64]);  view_338 = None
    permute_179: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_339, [0, 2, 1, 3]);  view_339 = None
    clone_129: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_179, memory_format = torch.contiguous_format);  permute_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:229, code: value_states = value_states.reshape(*proj_shape)
    view_343: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_129, [12, -1, 64])
    
    # No stacktrace found for following nodes
    unsqueeze_default_2: "f32[1, 12, 1024, 64]" = torch.ops.aten.unsqueeze.default(view_343, 0);  view_343 = None
    _scaled_dot_product_efficient_attention_default = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default, unsqueeze_default_1, unsqueeze_default_2, None, True, scale = 1.0);  unsqueeze_default = unsqueeze_default_1 = unsqueeze_default_2 = None
    getitem_64: "f32[1, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_default[0];  _scaled_dot_product_efficient_attention_default = None
    squeeze_dim: "f32[12, 1024, 64]" = torch.ops.aten.squeeze.dim(getitem_64, 0);  getitem_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:279, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_344: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(squeeze_dim, [1, 12, 1024, 64]);  squeeze_dim = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:280, code: attn_output = attn_output.transpose(1, 2)
    permute_182: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_344, [0, 2, 1, 3]);  view_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:284, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_132: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_182, memory_format = torch.contiguous_format);  permute_182 = None
    view_345: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_132, [1, 1024, 768]);  clone_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_346: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_345, [1024, 768]);  view_345 = None
    permute_183: "f32[768, 768]" = torch.ops.aten.permute.default(arg250_1, [1, 0]);  arg250_1 = None
    
    # No stacktrace found for following nodes
    mm_default_2: "f32[1024, 768]" = torch.ops.aten.mm.default(view_346, permute_183);  view_346 = permute_183 = None
    add_tensor_2: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_2, arg251_1);  mm_default_2 = arg251_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_347: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_2, [1, 1024, 768]);  add_tensor_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:452, code: hidden_states = residual + hidden_states
    add_110: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_109, view_347);  add_109 = view_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:453, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_30 = torch.ops.aten.var_mean.correction(add_110, [2], correction = 0, keepdim = True)
    getitem_60: "f32[1, 1024, 1]" = var_mean_30[0]
    getitem_61: "f32[1, 1024, 1]" = var_mean_30[1];  var_mean_30 = None
    sub_49: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_110, getitem_61);  add_110 = getitem_61 = None
    add_111: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
    rsqrt_30: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
    mul_113: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_30);  sub_49 = rsqrt_30 = None
    mul_114: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_113, arg252_1);  mul_113 = arg252_1 = None
    add_112: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_114, arg253_1);  mul_114 = arg253_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_348: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_112, [1024, 768])
    permute_184: "f32[768, 3072]" = torch.ops.aten.permute.default(arg254_1, [1, 0]);  arg254_1 = None
    
    # No stacktrace found for following nodes
    mm_default_1: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_348, permute_184);  view_348 = permute_184 = None
    add_tensor_1: "f32[1024, 3072]" = torch.ops.aten.add.Tensor(mm_default_1, arg255_1);  mm_default_1 = arg255_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_349: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(add_tensor_1, [1, 1024, 3072]);  add_tensor_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_115: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_349, 0.5)
    mul_116: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_349, 0.7071067811865476);  view_349 = None
    erf_11: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_116);  mul_116 = None
    add_113: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_117: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_115, add_113);  mul_115 = add_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_350: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_117, [1024, 3072]);  mul_117 = None
    permute_185: "f32[3072, 768]" = torch.ops.aten.permute.default(arg256_1, [1, 0]);  arg256_1 = None
    
    # No stacktrace found for following nodes
    mm_default: "f32[1024, 768]" = torch.ops.aten.mm.default(view_350, permute_185);  view_350 = permute_185 = None
    add_tensor: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default, arg257_1);  mm_default = arg257_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_351: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor, [1, 1024, 768]);  add_tensor = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:464, code: hidden_states = residual + hidden_states
    add_114: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_112, view_351);  add_112 = view_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:465, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_31 = torch.ops.aten.var_mean.correction(add_114, [2], correction = 0, keepdim = True)
    getitem_62: "f32[1, 1024, 1]" = var_mean_31[0]
    getitem_63: "f32[1, 1024, 1]" = var_mean_31[1];  var_mean_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:1350, code: masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
    view_355: "i64[1024]" = torch.ops.aten.reshape.default(arg262_1, [-1]);  arg262_1 = None
    ne_2: "b8[1024]" = torch.ops.aten.ne.Scalar(view_355, -100)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:465, code: hidden_states = self.final_layer_norm(hidden_states)
    sub_50: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_114, getitem_63);  add_114 = getitem_63 = None
    add_115: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
    rsqrt_31: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
    mul_118: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_31);  sub_50 = rsqrt_31 = None
    mul_119: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_118, arg258_1);  mul_118 = arg258_1 = None
    add_116: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_119, arg259_1);  mul_119 = arg259_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:1344, code: lm_logits = self.lm_head(outputs[0])
    view_352: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_116, [1024, 768]);  add_116 = None
    permute_186: "f32[768, 50005]" = torch.ops.aten.permute.default(arg260_1, [1, 0]);  arg260_1 = None
    mm: "f32[1024, 50005]" = torch.ops.aten.mm.default(view_352, permute_186);  view_352 = permute_186 = None
    view_353: "f32[1, 1024, 50005]" = torch.ops.aten.reshape.default(mm, [1, 1024, 50005]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:1345, code: lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)
    add_117: "f32[1, 1024, 50005]" = torch.ops.aten.add.Tensor(view_353, arg261_1);  view_353 = arg261_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:1350, code: masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
    view_354: "f32[1024, 50005]" = torch.ops.aten.reshape.default(add_117, [-1, 50005])
    amax_18: "f32[1024, 1]" = torch.ops.aten.amax.default(view_354, [1], True)
    sub_51: "f32[1024, 50005]" = torch.ops.aten.sub.Tensor(view_354, amax_18);  view_354 = amax_18 = None
    exp_18: "f32[1024, 50005]" = torch.ops.aten.exp.default(sub_51)
    sum_20: "f32[1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [1], True);  exp_18 = None
    log: "f32[1024, 1]" = torch.ops.aten.log.default(sum_20);  sum_20 = None
    sub_52: "f32[1024, 50005]" = torch.ops.aten.sub.Tensor(sub_51, log);  sub_51 = log = None
    ne_1: "b8[1024]" = torch.ops.aten.ne.Scalar(view_355, -100)
    full_default_3: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where_2: "i64[1024]" = torch.ops.aten.where.self(ne_1, view_355, full_default_3);  ne_1 = full_default_3 = None
    unsqueeze_5: "i64[1024, 1]" = torch.ops.aten.unsqueeze.default(where_2, 1);  where_2 = None
    gather_1: "f32[1024, 1]" = torch.ops.aten.gather.default(sub_52, 1, unsqueeze_5);  sub_52 = unsqueeze_5 = None
    squeeze_1: "f32[1024]" = torch.ops.aten.squeeze.dim(gather_1, 1);  gather_1 = None
    neg: "f32[1024]" = torch.ops.aten.neg.default(squeeze_1);  squeeze_1 = None
    full_default_4: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where_3: "f32[1024]" = torch.ops.aten.where.self(ne_2, neg, full_default_4);  ne_2 = neg = full_default_4 = None
    sum_22: "f32[]" = torch.ops.aten.sum.default(where_3);  where_3 = None
    ne_3: "b8[1024]" = torch.ops.aten.ne.Scalar(view_355, -100);  view_355 = None
    sum_21: "i64[]" = torch.ops.aten.sum.default(ne_3);  ne_3 = None
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_21, torch.float32);  sum_21 = None
    div_18: "f32[]" = torch.ops.aten.div.Tensor(sum_22, convert_element_type);  sum_22 = convert_element_type = None
    return (div_18, add_117, clone_52, clone_53, clone_58, clone_59, clone_66, clone_67, clone_72, clone_73, clone_80, clone_81, clone_86, clone_87, clone_94, clone_95, clone_100, clone_101, clone_108, clone_109, clone_114, clone_115, clone_122, clone_123, clone_128, clone_129, add_45)
    