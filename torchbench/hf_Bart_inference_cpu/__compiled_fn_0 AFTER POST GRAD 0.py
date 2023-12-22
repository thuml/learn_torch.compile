from __future__ import annotations



def forward(self, arg0_1: "f32[1026, 768]", arg1_1: "f32[1026, 768]", arg2_1: "f32[50265, 768]", arg3_1: "f32[768]", arg4_1: "f32[768]", arg5_1: "f32[768, 768]", arg6_1: "f32[768]", arg7_1: "f32[768, 768]", arg8_1: "f32[768]", arg9_1: "f32[768, 768]", arg10_1: "f32[768]", arg11_1: "f32[768, 768]", arg12_1: "f32[768]", arg13_1: "f32[768]", arg14_1: "f32[768]", arg15_1: "f32[3072, 768]", arg16_1: "f32[3072]", arg17_1: "f32[768, 3072]", arg18_1: "f32[768]", arg19_1: "f32[768]", arg20_1: "f32[768]", arg21_1: "f32[768, 768]", arg22_1: "f32[768]", arg23_1: "f32[768, 768]", arg24_1: "f32[768]", arg25_1: "f32[768, 768]", arg26_1: "f32[768]", arg27_1: "f32[768, 768]", arg28_1: "f32[768]", arg29_1: "f32[768]", arg30_1: "f32[768]", arg31_1: "f32[3072, 768]", arg32_1: "f32[3072]", arg33_1: "f32[768, 3072]", arg34_1: "f32[768]", arg35_1: "f32[768]", arg36_1: "f32[768]", arg37_1: "f32[768, 768]", arg38_1: "f32[768]", arg39_1: "f32[768, 768]", arg40_1: "f32[768]", arg41_1: "f32[768, 768]", arg42_1: "f32[768]", arg43_1: "f32[768, 768]", arg44_1: "f32[768]", arg45_1: "f32[768]", arg46_1: "f32[768]", arg47_1: "f32[3072, 768]", arg48_1: "f32[3072]", arg49_1: "f32[768, 3072]", arg50_1: "f32[768]", arg51_1: "f32[768]", arg52_1: "f32[768]", arg53_1: "f32[768, 768]", arg54_1: "f32[768]", arg55_1: "f32[768, 768]", arg56_1: "f32[768]", arg57_1: "f32[768, 768]", arg58_1: "f32[768]", arg59_1: "f32[768, 768]", arg60_1: "f32[768]", arg61_1: "f32[768]", arg62_1: "f32[768]", arg63_1: "f32[3072, 768]", arg64_1: "f32[3072]", arg65_1: "f32[768, 3072]", arg66_1: "f32[768]", arg67_1: "f32[768]", arg68_1: "f32[768]", arg69_1: "f32[768, 768]", arg70_1: "f32[768]", arg71_1: "f32[768, 768]", arg72_1: "f32[768]", arg73_1: "f32[768, 768]", arg74_1: "f32[768]", arg75_1: "f32[768, 768]", arg76_1: "f32[768]", arg77_1: "f32[768]", arg78_1: "f32[768]", arg79_1: "f32[3072, 768]", arg80_1: "f32[3072]", arg81_1: "f32[768, 3072]", arg82_1: "f32[768]", arg83_1: "f32[768]", arg84_1: "f32[768]", arg85_1: "f32[768, 768]", arg86_1: "f32[768]", arg87_1: "f32[768, 768]", arg88_1: "f32[768]", arg89_1: "f32[768, 768]", arg90_1: "f32[768]", arg91_1: "f32[768, 768]", arg92_1: "f32[768]", arg93_1: "f32[768]", arg94_1: "f32[768]", arg95_1: "f32[3072, 768]", arg96_1: "f32[3072]", arg97_1: "f32[768, 3072]", arg98_1: "f32[768]", arg99_1: "f32[768]", arg100_1: "f32[768]", arg101_1: "f32[50265, 768]", arg102_1: "f32[768]", arg103_1: "f32[768]", arg104_1: "f32[768, 768]", arg105_1: "f32[768]", arg106_1: "f32[768, 768]", arg107_1: "f32[768]", arg108_1: "f32[768, 768]", arg109_1: "f32[768]", arg110_1: "f32[768, 768]", arg111_1: "f32[768]", arg112_1: "f32[768]", arg113_1: "f32[768]", arg114_1: "f32[768, 768]", arg115_1: "f32[768]", arg116_1: "f32[768, 768]", arg117_1: "f32[768]", arg118_1: "f32[768, 768]", arg119_1: "f32[768]", arg120_1: "f32[768, 768]", arg121_1: "f32[768]", arg122_1: "f32[768]", arg123_1: "f32[768]", arg124_1: "f32[3072, 768]", arg125_1: "f32[3072]", arg126_1: "f32[768, 3072]", arg127_1: "f32[768]", arg128_1: "f32[768]", arg129_1: "f32[768]", arg130_1: "f32[768, 768]", arg131_1: "f32[768]", arg132_1: "f32[768, 768]", arg133_1: "f32[768]", arg134_1: "f32[768, 768]", arg135_1: "f32[768]", arg136_1: "f32[768, 768]", arg137_1: "f32[768]", arg138_1: "f32[768]", arg139_1: "f32[768]", arg140_1: "f32[768, 768]", arg141_1: "f32[768]", arg142_1: "f32[768, 768]", arg143_1: "f32[768]", arg144_1: "f32[768, 768]", arg145_1: "f32[768]", arg146_1: "f32[768, 768]", arg147_1: "f32[768]", arg148_1: "f32[768]", arg149_1: "f32[768]", arg150_1: "f32[3072, 768]", arg151_1: "f32[3072]", arg152_1: "f32[768, 3072]", arg153_1: "f32[768]", arg154_1: "f32[768]", arg155_1: "f32[768]", arg156_1: "f32[768, 768]", arg157_1: "f32[768]", arg158_1: "f32[768, 768]", arg159_1: "f32[768]", arg160_1: "f32[768, 768]", arg161_1: "f32[768]", arg162_1: "f32[768, 768]", arg163_1: "f32[768]", arg164_1: "f32[768]", arg165_1: "f32[768]", arg166_1: "f32[768, 768]", arg167_1: "f32[768]", arg168_1: "f32[768, 768]", arg169_1: "f32[768]", arg170_1: "f32[768, 768]", arg171_1: "f32[768]", arg172_1: "f32[768, 768]", arg173_1: "f32[768]", arg174_1: "f32[768]", arg175_1: "f32[768]", arg176_1: "f32[3072, 768]", arg177_1: "f32[3072]", arg178_1: "f32[768, 3072]", arg179_1: "f32[768]", arg180_1: "f32[768]", arg181_1: "f32[768]", arg182_1: "f32[768, 768]", arg183_1: "f32[768]", arg184_1: "f32[768, 768]", arg185_1: "f32[768]", arg186_1: "f32[768, 768]", arg187_1: "f32[768]", arg188_1: "f32[768, 768]", arg189_1: "f32[768]", arg190_1: "f32[768]", arg191_1: "f32[768]", arg192_1: "f32[768, 768]", arg193_1: "f32[768]", arg194_1: "f32[768, 768]", arg195_1: "f32[768]", arg196_1: "f32[768, 768]", arg197_1: "f32[768]", arg198_1: "f32[768, 768]", arg199_1: "f32[768]", arg200_1: "f32[768]", arg201_1: "f32[768]", arg202_1: "f32[3072, 768]", arg203_1: "f32[3072]", arg204_1: "f32[768, 3072]", arg205_1: "f32[768]", arg206_1: "f32[768]", arg207_1: "f32[768]", arg208_1: "f32[768, 768]", arg209_1: "f32[768]", arg210_1: "f32[768, 768]", arg211_1: "f32[768]", arg212_1: "f32[768, 768]", arg213_1: "f32[768]", arg214_1: "f32[768, 768]", arg215_1: "f32[768]", arg216_1: "f32[768]", arg217_1: "f32[768]", arg218_1: "f32[768, 768]", arg219_1: "f32[768]", arg220_1: "f32[768, 768]", arg221_1: "f32[768]", arg222_1: "f32[768, 768]", arg223_1: "f32[768]", arg224_1: "f32[768, 768]", arg225_1: "f32[768]", arg226_1: "f32[768]", arg227_1: "f32[768]", arg228_1: "f32[3072, 768]", arg229_1: "f32[3072]", arg230_1: "f32[768, 3072]", arg231_1: "f32[768]", arg232_1: "f32[768]", arg233_1: "f32[768]", arg234_1: "f32[768, 768]", arg235_1: "f32[768]", arg236_1: "f32[768, 768]", arg237_1: "f32[768]", arg238_1: "f32[768, 768]", arg239_1: "f32[768]", arg240_1: "f32[768, 768]", arg241_1: "f32[768]", arg242_1: "f32[768]", arg243_1: "f32[768]", arg244_1: "f32[768, 768]", arg245_1: "f32[768]", arg246_1: "f32[768, 768]", arg247_1: "f32[768]", arg248_1: "f32[768, 768]", arg249_1: "f32[768]", arg250_1: "f32[768, 768]", arg251_1: "f32[768]", arg252_1: "f32[768]", arg253_1: "f32[768]", arg254_1: "f32[3072, 768]", arg255_1: "f32[3072]", arg256_1: "f32[768, 3072]", arg257_1: "f32[768]", arg258_1: "f32[768]", arg259_1: "f32[768]", arg260_1: "f32[50265, 768]", arg261_1: "f32[1, 50265]", arg262_1: "i64[4, 512]", arg263_1: "i64[4, 512]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:811, code: input_ids = input_ids.view(-1, input_ids.shape[-1])
    view: "i64[4, 512]" = torch.ops.aten.reshape.default(arg262_1, [-1, 512]);  arg262_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:818, code: inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
    embedding: "f32[4, 512, 768]" = torch.ops.aten.embedding.default(arg2_1, view, 1);  arg2_1 = view = None
    mul: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(embedding, 1.0);  embedding = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:135, code: positions = torch.arange(
    iota: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:137, code: ).expand(bsz, -1)
    expand: "i64[4, 512]" = torch.ops.aten.expand.default(iota, [4, -1]);  iota = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:139, code: return super().forward(positions + self.offset)
    add: "i64[4, 512]" = torch.ops.aten.add.Tensor(expand, 2);  expand = None
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/sparse.py:163, code: return F.embedding(
    embedding_1: "f32[4, 512, 768]" = torch.ops.aten.embedding.default(arg0_1, add);  arg0_1 = add = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:823, code: hidden_states = inputs_embeds + embed_pos
    add_1: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul, embedding_1);  mul = embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:824, code: hidden_states = self.layernorm_embedding(hidden_states)
    var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
    getitem: "f32[4, 512, 1]" = var_mean[0]
    getitem_1: "f32[4, 512, 1]" = var_mean[1];  var_mean = None
    sub: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
    add_2: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
    mul_1: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
    mul_2: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
    add_3: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_2, arg4_1);  mul_2 = arg4_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_1: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_3, [2048, 768])
    permute: "f32[768, 768]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
    addmm: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg6_1, view_1, permute);  arg6_1 = view_1 = permute = None
    view_2: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm, [4, 512, 768]);  addmm = None
    mul_3: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_2, 0.125);  view_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_9: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(mul_3, [4, 512, 12, 64]);  mul_3 = None
    permute_5: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
    clone_3: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_10: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_3, [48, -1, 64]);  clone_3 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_33: "f32[1, 48, 512, 64]" = torch.ops.aten.unsqueeze.default(view_10, 0);  view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_3: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_3, [2048, 768])
    permute_1: "f32[768, 768]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
    addmm_1: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg8_1, view_3, permute_1);  arg8_1 = view_3 = permute_1 = None
    view_4: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_1, [4, 512, 768]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_5: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_4, [4, -1, 12, 64]);  view_4 = None
    permute_2: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
    clone_1: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_11: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_1, [48, -1, 64]);  clone_1 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_34: "f32[1, 48, 512, 64]" = torch.ops.aten.unsqueeze.default(view_11, 0);  view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_6: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_3, [2048, 768])
    permute_3: "f32[768, 768]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
    addmm_2: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg10_1, view_6, permute_3);  arg10_1 = view_6 = permute_3 = None
    view_7: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_2, [4, 512, 768]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_8: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_7, [4, -1, 12, 64]);  view_7 = None
    permute_4: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
    clone_2: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_12: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_2, [48, -1, 64]);  clone_2 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_35: "f32[1, 48, 512, 64]" = torch.ops.aten.unsqueeze.default(view_12, 0);  view_12 = None
    _scaled_dot_product_flash_attention_default_11 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_33, unsqueeze_default_34, unsqueeze_default_35, scale = 1.0);  unsqueeze_default_33 = unsqueeze_default_34 = unsqueeze_default_35 = None
    getitem_75: "f32[1, 48, 512, 64]" = _scaled_dot_product_flash_attention_default_11[0];  _scaled_dot_product_flash_attention_default_11 = None
    squeeze_dim_11: "f32[48, 512, 64]" = torch.ops.aten.squeeze.dim(getitem_75, 0);  getitem_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_13: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(squeeze_dim_11, [4, 12, 512, 64]);  squeeze_dim_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_7: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_13, [0, 2, 1, 3]);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_5: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_14: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_5, [4, 512, 768]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_15: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_14, [2048, 768]);  view_14 = None
    permute_8: "f32[768, 768]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
    addmm_3: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg12_1, view_15, permute_8);  arg12_1 = view_15 = permute_8 = None
    view_16: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_3, [4, 512, 768]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:339, code: hidden_states = residual + hidden_states
    add_4: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_3, view_16);  add_3 = view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:340, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_1 = torch.ops.aten.var_mean.correction(add_4, [2], correction = 0, keepdim = True)
    getitem_2: "f32[4, 512, 1]" = var_mean_1[0]
    getitem_3: "f32[4, 512, 1]" = var_mean_1[1];  var_mean_1 = None
    sub_2: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_4, getitem_3);  add_4 = getitem_3 = None
    add_5: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
    rsqrt_1: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
    mul_4: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
    mul_5: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_4, arg13_1);  mul_4 = arg13_1 = None
    add_6: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_5, arg14_1);  mul_5 = arg14_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_17: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_6, [2048, 768])
    permute_9: "f32[768, 3072]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
    addmm_4: "f32[2048, 3072]" = torch.ops.aten.addmm.default(arg16_1, view_17, permute_9);  arg16_1 = view_17 = permute_9 = None
    view_18: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_4, [4, 512, 3072]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_6: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_18, 0.5)
    mul_7: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_18, 0.7071067811865476);  view_18 = None
    erf: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_7);  mul_7 = None
    add_7: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_8: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_6, add_7);  mul_6 = add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:345, code: hidden_states = self.fc2(hidden_states)
    view_19: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_8, [2048, 3072]);  mul_8 = None
    permute_10: "f32[3072, 768]" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
    addmm_5: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg18_1, view_19, permute_10);  arg18_1 = view_19 = permute_10 = None
    view_20: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_5, [4, 512, 768]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:347, code: hidden_states = residual + hidden_states
    add_8: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_6, view_20);  add_6 = view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:348, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
    getitem_4: "f32[4, 512, 1]" = var_mean_2[0]
    getitem_5: "f32[4, 512, 1]" = var_mean_2[1];  var_mean_2 = None
    sub_3: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_8, getitem_5);  add_8 = getitem_5 = None
    add_9: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
    rsqrt_2: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
    mul_9: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
    mul_10: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_9, arg19_1);  mul_9 = arg19_1 = None
    add_10: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_10, arg20_1);  mul_10 = arg20_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_21: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_10, [2048, 768])
    permute_11: "f32[768, 768]" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
    addmm_6: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg22_1, view_21, permute_11);  arg22_1 = view_21 = permute_11 = None
    view_22: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_6, [4, 512, 768]);  addmm_6 = None
    mul_11: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_22, 0.125);  view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_29: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(mul_11, [4, 512, 12, 64]);  mul_11 = None
    permute_16: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
    clone_11: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_30: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_11, [48, -1, 64]);  clone_11 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_30: "f32[1, 48, 512, 64]" = torch.ops.aten.unsqueeze.default(view_30, 0);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_23: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_10, [2048, 768])
    permute_12: "f32[768, 768]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
    addmm_7: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg24_1, view_23, permute_12);  arg24_1 = view_23 = permute_12 = None
    view_24: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_7, [4, 512, 768]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_25: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_24, [4, -1, 12, 64]);  view_24 = None
    permute_13: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_25, [0, 2, 1, 3]);  view_25 = None
    clone_9: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_31: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_9, [48, -1, 64]);  clone_9 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_31: "f32[1, 48, 512, 64]" = torch.ops.aten.unsqueeze.default(view_31, 0);  view_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_26: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_10, [2048, 768])
    permute_14: "f32[768, 768]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
    addmm_8: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg26_1, view_26, permute_14);  arg26_1 = view_26 = permute_14 = None
    view_27: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_8, [4, 512, 768]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_28: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_27, [4, -1, 12, 64]);  view_27 = None
    permute_15: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
    clone_10: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_32: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_10, [48, -1, 64]);  clone_10 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_32: "f32[1, 48, 512, 64]" = torch.ops.aten.unsqueeze.default(view_32, 0);  view_32 = None
    _scaled_dot_product_flash_attention_default_10 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_30, unsqueeze_default_31, unsqueeze_default_32, scale = 1.0);  unsqueeze_default_30 = unsqueeze_default_31 = unsqueeze_default_32 = None
    getitem_74: "f32[1, 48, 512, 64]" = _scaled_dot_product_flash_attention_default_10[0];  _scaled_dot_product_flash_attention_default_10 = None
    squeeze_dim_10: "f32[48, 512, 64]" = torch.ops.aten.squeeze.dim(getitem_74, 0);  getitem_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_33: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(squeeze_dim_10, [4, 12, 512, 64]);  squeeze_dim_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_18: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3]);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_13: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
    view_34: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_13, [4, 512, 768]);  clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_35: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_34, [2048, 768]);  view_34 = None
    permute_19: "f32[768, 768]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
    addmm_9: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg28_1, view_35, permute_19);  arg28_1 = view_35 = permute_19 = None
    view_36: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_9, [4, 512, 768]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:339, code: hidden_states = residual + hidden_states
    add_11: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_10, view_36);  add_10 = view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:340, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_3 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
    getitem_6: "f32[4, 512, 1]" = var_mean_3[0]
    getitem_7: "f32[4, 512, 1]" = var_mean_3[1];  var_mean_3 = None
    sub_5: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_11, getitem_7);  add_11 = getitem_7 = None
    add_12: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
    rsqrt_3: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
    mul_12: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
    mul_13: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_12, arg29_1);  mul_12 = arg29_1 = None
    add_13: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_13, arg30_1);  mul_13 = arg30_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_37: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_13, [2048, 768])
    permute_20: "f32[768, 3072]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
    addmm_10: "f32[2048, 3072]" = torch.ops.aten.addmm.default(arg32_1, view_37, permute_20);  arg32_1 = view_37 = permute_20 = None
    view_38: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_10, [4, 512, 3072]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_14: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_38, 0.5)
    mul_15: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_38, 0.7071067811865476);  view_38 = None
    erf_1: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_15);  mul_15 = None
    add_14: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_16: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_14, add_14);  mul_14 = add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:345, code: hidden_states = self.fc2(hidden_states)
    view_39: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_16, [2048, 3072]);  mul_16 = None
    permute_21: "f32[3072, 768]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
    addmm_11: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg34_1, view_39, permute_21);  arg34_1 = view_39 = permute_21 = None
    view_40: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_11, [4, 512, 768]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:347, code: hidden_states = residual + hidden_states
    add_15: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_13, view_40);  add_13 = view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:348, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_4 = torch.ops.aten.var_mean.correction(add_15, [2], correction = 0, keepdim = True)
    getitem_8: "f32[4, 512, 1]" = var_mean_4[0]
    getitem_9: "f32[4, 512, 1]" = var_mean_4[1];  var_mean_4 = None
    sub_6: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_15, getitem_9);  add_15 = getitem_9 = None
    add_16: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
    rsqrt_4: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    mul_17: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
    mul_18: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_17, arg35_1);  mul_17 = arg35_1 = None
    add_17: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_18, arg36_1);  mul_18 = arg36_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_41: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_17, [2048, 768])
    permute_22: "f32[768, 768]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
    addmm_12: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg38_1, view_41, permute_22);  arg38_1 = view_41 = permute_22 = None
    view_42: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_12, [4, 512, 768]);  addmm_12 = None
    mul_19: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_42, 0.125);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_49: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(mul_19, [4, 512, 12, 64]);  mul_19 = None
    permute_27: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_49, [0, 2, 1, 3]);  view_49 = None
    clone_19: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_50: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_19, [48, -1, 64]);  clone_19 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_27: "f32[1, 48, 512, 64]" = torch.ops.aten.unsqueeze.default(view_50, 0);  view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_43: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_17, [2048, 768])
    permute_23: "f32[768, 768]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
    addmm_13: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg40_1, view_43, permute_23);  arg40_1 = view_43 = permute_23 = None
    view_44: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_13, [4, 512, 768]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_45: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_44, [4, -1, 12, 64]);  view_44 = None
    permute_24: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_45, [0, 2, 1, 3]);  view_45 = None
    clone_17: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_51: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_17, [48, -1, 64]);  clone_17 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_28: "f32[1, 48, 512, 64]" = torch.ops.aten.unsqueeze.default(view_51, 0);  view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_46: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_17, [2048, 768])
    permute_25: "f32[768, 768]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
    addmm_14: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg42_1, view_46, permute_25);  arg42_1 = view_46 = permute_25 = None
    view_47: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_14, [4, 512, 768]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_48: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_47, [4, -1, 12, 64]);  view_47 = None
    permute_26: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
    clone_18: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_52: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_18, [48, -1, 64]);  clone_18 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_29: "f32[1, 48, 512, 64]" = torch.ops.aten.unsqueeze.default(view_52, 0);  view_52 = None
    _scaled_dot_product_flash_attention_default_9 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_27, unsqueeze_default_28, unsqueeze_default_29, scale = 1.0);  unsqueeze_default_27 = unsqueeze_default_28 = unsqueeze_default_29 = None
    getitem_73: "f32[1, 48, 512, 64]" = _scaled_dot_product_flash_attention_default_9[0];  _scaled_dot_product_flash_attention_default_9 = None
    squeeze_dim_9: "f32[48, 512, 64]" = torch.ops.aten.squeeze.dim(getitem_73, 0);  getitem_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_53: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(squeeze_dim_9, [4, 12, 512, 64]);  squeeze_dim_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_29: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_21: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    view_54: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_21, [4, 512, 768]);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_55: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_54, [2048, 768]);  view_54 = None
    permute_30: "f32[768, 768]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
    addmm_15: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg44_1, view_55, permute_30);  arg44_1 = view_55 = permute_30 = None
    view_56: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_15, [4, 512, 768]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:339, code: hidden_states = residual + hidden_states
    add_18: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_17, view_56);  add_17 = view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:340, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_5 = torch.ops.aten.var_mean.correction(add_18, [2], correction = 0, keepdim = True)
    getitem_10: "f32[4, 512, 1]" = var_mean_5[0]
    getitem_11: "f32[4, 512, 1]" = var_mean_5[1];  var_mean_5 = None
    sub_8: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_18, getitem_11);  add_18 = getitem_11 = None
    add_19: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
    rsqrt_5: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
    mul_20: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
    mul_21: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_20, arg45_1);  mul_20 = arg45_1 = None
    add_20: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_21, arg46_1);  mul_21 = arg46_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_57: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_20, [2048, 768])
    permute_31: "f32[768, 3072]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
    addmm_16: "f32[2048, 3072]" = torch.ops.aten.addmm.default(arg48_1, view_57, permute_31);  arg48_1 = view_57 = permute_31 = None
    view_58: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_16, [4, 512, 3072]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_22: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_58, 0.5)
    mul_23: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_58, 0.7071067811865476);  view_58 = None
    erf_2: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_23);  mul_23 = None
    add_21: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_24: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_22, add_21);  mul_22 = add_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:345, code: hidden_states = self.fc2(hidden_states)
    view_59: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_24, [2048, 3072]);  mul_24 = None
    permute_32: "f32[3072, 768]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
    addmm_17: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg50_1, view_59, permute_32);  arg50_1 = view_59 = permute_32 = None
    view_60: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_17, [4, 512, 768]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:347, code: hidden_states = residual + hidden_states
    add_22: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_20, view_60);  add_20 = view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:348, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_6 = torch.ops.aten.var_mean.correction(add_22, [2], correction = 0, keepdim = True)
    getitem_12: "f32[4, 512, 1]" = var_mean_6[0]
    getitem_13: "f32[4, 512, 1]" = var_mean_6[1];  var_mean_6 = None
    sub_9: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_22, getitem_13);  add_22 = getitem_13 = None
    add_23: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
    rsqrt_6: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
    mul_25: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = rsqrt_6 = None
    mul_26: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_25, arg51_1);  mul_25 = arg51_1 = None
    add_24: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_26, arg52_1);  mul_26 = arg52_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_61: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_24, [2048, 768])
    permute_33: "f32[768, 768]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
    addmm_18: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg54_1, view_61, permute_33);  arg54_1 = view_61 = permute_33 = None
    view_62: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_18, [4, 512, 768]);  addmm_18 = None
    mul_27: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_62, 0.125);  view_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_69: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(mul_27, [4, 512, 12, 64]);  mul_27 = None
    permute_38: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_69, [0, 2, 1, 3]);  view_69 = None
    clone_27: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_70: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_27, [48, -1, 64]);  clone_27 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_24: "f32[1, 48, 512, 64]" = torch.ops.aten.unsqueeze.default(view_70, 0);  view_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_63: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_24, [2048, 768])
    permute_34: "f32[768, 768]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
    addmm_19: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg56_1, view_63, permute_34);  arg56_1 = view_63 = permute_34 = None
    view_64: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_19, [4, 512, 768]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_65: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_64, [4, -1, 12, 64]);  view_64 = None
    permute_35: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None
    clone_25: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_71: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_25, [48, -1, 64]);  clone_25 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_25: "f32[1, 48, 512, 64]" = torch.ops.aten.unsqueeze.default(view_71, 0);  view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_66: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_24, [2048, 768])
    permute_36: "f32[768, 768]" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
    addmm_20: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg58_1, view_66, permute_36);  arg58_1 = view_66 = permute_36 = None
    view_67: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_20, [4, 512, 768]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_68: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_67, [4, -1, 12, 64]);  view_67 = None
    permute_37: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None
    clone_26: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_72: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_26, [48, -1, 64]);  clone_26 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_26: "f32[1, 48, 512, 64]" = torch.ops.aten.unsqueeze.default(view_72, 0);  view_72 = None
    _scaled_dot_product_flash_attention_default_8 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_24, unsqueeze_default_25, unsqueeze_default_26, scale = 1.0);  unsqueeze_default_24 = unsqueeze_default_25 = unsqueeze_default_26 = None
    getitem_72: "f32[1, 48, 512, 64]" = _scaled_dot_product_flash_attention_default_8[0];  _scaled_dot_product_flash_attention_default_8 = None
    squeeze_dim_8: "f32[48, 512, 64]" = torch.ops.aten.squeeze.dim(getitem_72, 0);  getitem_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_73: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(squeeze_dim_8, [4, 12, 512, 64]);  squeeze_dim_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_40: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_29: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
    view_74: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_29, [4, 512, 768]);  clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_75: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_74, [2048, 768]);  view_74 = None
    permute_41: "f32[768, 768]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
    addmm_21: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg60_1, view_75, permute_41);  arg60_1 = view_75 = permute_41 = None
    view_76: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_21, [4, 512, 768]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:339, code: hidden_states = residual + hidden_states
    add_25: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_24, view_76);  add_24 = view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:340, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_7 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
    getitem_14: "f32[4, 512, 1]" = var_mean_7[0]
    getitem_15: "f32[4, 512, 1]" = var_mean_7[1];  var_mean_7 = None
    sub_11: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_25, getitem_15);  add_25 = getitem_15 = None
    add_26: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
    rsqrt_7: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
    mul_28: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
    mul_29: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_28, arg61_1);  mul_28 = arg61_1 = None
    add_27: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_29, arg62_1);  mul_29 = arg62_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_77: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_27, [2048, 768])
    permute_42: "f32[768, 3072]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
    addmm_22: "f32[2048, 3072]" = torch.ops.aten.addmm.default(arg64_1, view_77, permute_42);  arg64_1 = view_77 = permute_42 = None
    view_78: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_22, [4, 512, 3072]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_30: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_78, 0.5)
    mul_31: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_78, 0.7071067811865476);  view_78 = None
    erf_3: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_31);  mul_31 = None
    add_28: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_32: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_30, add_28);  mul_30 = add_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:345, code: hidden_states = self.fc2(hidden_states)
    view_79: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_32, [2048, 3072]);  mul_32 = None
    permute_43: "f32[3072, 768]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
    addmm_23: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg66_1, view_79, permute_43);  arg66_1 = view_79 = permute_43 = None
    view_80: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_23, [4, 512, 768]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:347, code: hidden_states = residual + hidden_states
    add_29: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_27, view_80);  add_27 = view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:348, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_8 = torch.ops.aten.var_mean.correction(add_29, [2], correction = 0, keepdim = True)
    getitem_16: "f32[4, 512, 1]" = var_mean_8[0]
    getitem_17: "f32[4, 512, 1]" = var_mean_8[1];  var_mean_8 = None
    sub_12: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_29, getitem_17);  add_29 = getitem_17 = None
    add_30: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
    rsqrt_8: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
    mul_33: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = rsqrt_8 = None
    mul_34: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_33, arg67_1);  mul_33 = arg67_1 = None
    add_31: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_34, arg68_1);  mul_34 = arg68_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_81: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_31, [2048, 768])
    permute_44: "f32[768, 768]" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
    addmm_24: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg70_1, view_81, permute_44);  arg70_1 = view_81 = permute_44 = None
    view_82: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_24, [4, 512, 768]);  addmm_24 = None
    mul_35: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_82, 0.125);  view_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_89: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(mul_35, [4, 512, 12, 64]);  mul_35 = None
    permute_49: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_89, [0, 2, 1, 3]);  view_89 = None
    clone_35: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_90: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_35, [48, -1, 64]);  clone_35 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_21: "f32[1, 48, 512, 64]" = torch.ops.aten.unsqueeze.default(view_90, 0);  view_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_83: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_31, [2048, 768])
    permute_45: "f32[768, 768]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
    addmm_25: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg72_1, view_83, permute_45);  arg72_1 = view_83 = permute_45 = None
    view_84: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_25, [4, 512, 768]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_85: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_84, [4, -1, 12, 64]);  view_84 = None
    permute_46: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_85, [0, 2, 1, 3]);  view_85 = None
    clone_33: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_91: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_33, [48, -1, 64]);  clone_33 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_22: "f32[1, 48, 512, 64]" = torch.ops.aten.unsqueeze.default(view_91, 0);  view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_86: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_31, [2048, 768])
    permute_47: "f32[768, 768]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
    addmm_26: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg74_1, view_86, permute_47);  arg74_1 = view_86 = permute_47 = None
    view_87: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_26, [4, 512, 768]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_88: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_87, [4, -1, 12, 64]);  view_87 = None
    permute_48: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None
    clone_34: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_92: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_34, [48, -1, 64]);  clone_34 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_23: "f32[1, 48, 512, 64]" = torch.ops.aten.unsqueeze.default(view_92, 0);  view_92 = None
    _scaled_dot_product_flash_attention_default_7 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_21, unsqueeze_default_22, unsqueeze_default_23, scale = 1.0);  unsqueeze_default_21 = unsqueeze_default_22 = unsqueeze_default_23 = None
    getitem_71: "f32[1, 48, 512, 64]" = _scaled_dot_product_flash_attention_default_7[0];  _scaled_dot_product_flash_attention_default_7 = None
    squeeze_dim_7: "f32[48, 512, 64]" = torch.ops.aten.squeeze.dim(getitem_71, 0);  getitem_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_93: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(squeeze_dim_7, [4, 12, 512, 64]);  squeeze_dim_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_51: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_37: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
    view_94: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_37, [4, 512, 768]);  clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_95: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_94, [2048, 768]);  view_94 = None
    permute_52: "f32[768, 768]" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
    addmm_27: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg76_1, view_95, permute_52);  arg76_1 = view_95 = permute_52 = None
    view_96: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_27, [4, 512, 768]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:339, code: hidden_states = residual + hidden_states
    add_32: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_31, view_96);  add_31 = view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:340, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_9 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
    getitem_18: "f32[4, 512, 1]" = var_mean_9[0]
    getitem_19: "f32[4, 512, 1]" = var_mean_9[1];  var_mean_9 = None
    sub_14: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_32, getitem_19);  add_32 = getitem_19 = None
    add_33: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
    rsqrt_9: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
    mul_36: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
    mul_37: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_36, arg77_1);  mul_36 = arg77_1 = None
    add_34: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_37, arg78_1);  mul_37 = arg78_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_97: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_34, [2048, 768])
    permute_53: "f32[768, 3072]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
    addmm_28: "f32[2048, 3072]" = torch.ops.aten.addmm.default(arg80_1, view_97, permute_53);  arg80_1 = view_97 = permute_53 = None
    view_98: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_28, [4, 512, 3072]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_38: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_98, 0.5)
    mul_39: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_98, 0.7071067811865476);  view_98 = None
    erf_4: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_39);  mul_39 = None
    add_35: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_40: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_38, add_35);  mul_38 = add_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:345, code: hidden_states = self.fc2(hidden_states)
    view_99: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_40, [2048, 3072]);  mul_40 = None
    permute_54: "f32[3072, 768]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
    addmm_29: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg82_1, view_99, permute_54);  arg82_1 = view_99 = permute_54 = None
    view_100: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_29, [4, 512, 768]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:347, code: hidden_states = residual + hidden_states
    add_36: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_34, view_100);  add_34 = view_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:348, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_10 = torch.ops.aten.var_mean.correction(add_36, [2], correction = 0, keepdim = True)
    getitem_20: "f32[4, 512, 1]" = var_mean_10[0]
    getitem_21: "f32[4, 512, 1]" = var_mean_10[1];  var_mean_10 = None
    sub_15: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_36, getitem_21);  add_36 = getitem_21 = None
    add_37: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
    rsqrt_10: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
    mul_41: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = rsqrt_10 = None
    mul_42: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_41, arg83_1);  mul_41 = arg83_1 = None
    add_38: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_42, arg84_1);  mul_42 = arg84_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_101: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_38, [2048, 768])
    permute_55: "f32[768, 768]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
    addmm_30: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg86_1, view_101, permute_55);  arg86_1 = view_101 = permute_55 = None
    view_102: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_30, [4, 512, 768]);  addmm_30 = None
    mul_43: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_102, 0.125);  view_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_109: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(mul_43, [4, 512, 12, 64]);  mul_43 = None
    permute_60: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_109, [0, 2, 1, 3]);  view_109 = None
    clone_43: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_110: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_43, [48, -1, 64]);  clone_43 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_18: "f32[1, 48, 512, 64]" = torch.ops.aten.unsqueeze.default(view_110, 0);  view_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_103: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_38, [2048, 768])
    permute_56: "f32[768, 768]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
    addmm_31: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg88_1, view_103, permute_56);  arg88_1 = view_103 = permute_56 = None
    view_104: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_31, [4, 512, 768]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_105: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_104, [4, -1, 12, 64]);  view_104 = None
    permute_57: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_105, [0, 2, 1, 3]);  view_105 = None
    clone_41: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_57, memory_format = torch.contiguous_format);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_111: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_41, [48, -1, 64]);  clone_41 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_19: "f32[1, 48, 512, 64]" = torch.ops.aten.unsqueeze.default(view_111, 0);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_106: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_38, [2048, 768])
    permute_58: "f32[768, 768]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
    addmm_32: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg90_1, view_106, permute_58);  arg90_1 = view_106 = permute_58 = None
    view_107: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_32, [4, 512, 768]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_108: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_107, [4, -1, 12, 64]);  view_107 = None
    permute_59: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_108, [0, 2, 1, 3]);  view_108 = None
    clone_42: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_112: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_42, [48, -1, 64]);  clone_42 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_20: "f32[1, 48, 512, 64]" = torch.ops.aten.unsqueeze.default(view_112, 0);  view_112 = None
    _scaled_dot_product_flash_attention_default_6 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_18, unsqueeze_default_19, unsqueeze_default_20, scale = 1.0);  unsqueeze_default_18 = unsqueeze_default_19 = unsqueeze_default_20 = None
    getitem_70: "f32[1, 48, 512, 64]" = _scaled_dot_product_flash_attention_default_6[0];  _scaled_dot_product_flash_attention_default_6 = None
    squeeze_dim_6: "f32[48, 512, 64]" = torch.ops.aten.squeeze.dim(getitem_70, 0);  getitem_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_113: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(squeeze_dim_6, [4, 12, 512, 64]);  squeeze_dim_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_62: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_113, [0, 2, 1, 3]);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_45: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    view_114: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_45, [4, 512, 768]);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_115: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_114, [2048, 768]);  view_114 = None
    permute_63: "f32[768, 768]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
    addmm_33: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg92_1, view_115, permute_63);  arg92_1 = view_115 = permute_63 = None
    view_116: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_33, [4, 512, 768]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:339, code: hidden_states = residual + hidden_states
    add_39: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_38, view_116);  add_38 = view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:340, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_11 = torch.ops.aten.var_mean.correction(add_39, [2], correction = 0, keepdim = True)
    getitem_22: "f32[4, 512, 1]" = var_mean_11[0]
    getitem_23: "f32[4, 512, 1]" = var_mean_11[1];  var_mean_11 = None
    sub_17: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_39, getitem_23);  add_39 = getitem_23 = None
    add_40: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
    rsqrt_11: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
    mul_44: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
    mul_45: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_44, arg93_1);  mul_44 = arg93_1 = None
    add_41: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_45, arg94_1);  mul_45 = arg94_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:343, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_117: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_41, [2048, 768])
    permute_64: "f32[768, 3072]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
    addmm_34: "f32[2048, 3072]" = torch.ops.aten.addmm.default(arg96_1, view_117, permute_64);  arg96_1 = view_117 = permute_64 = None
    view_118: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_34, [4, 512, 3072]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_46: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_118, 0.5)
    mul_47: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_118, 0.7071067811865476);  view_118 = None
    erf_5: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
    add_42: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_48: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_46, add_42);  mul_46 = add_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:345, code: hidden_states = self.fc2(hidden_states)
    view_119: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_48, [2048, 3072]);  mul_48 = None
    permute_65: "f32[3072, 768]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
    addmm_35: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg98_1, view_119, permute_65);  arg98_1 = view_119 = permute_65 = None
    view_120: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_35, [4, 512, 768]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:347, code: hidden_states = residual + hidden_states
    add_43: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_41, view_120);  add_41 = view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:348, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_12 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
    getitem_24: "f32[4, 512, 1]" = var_mean_12[0]
    getitem_25: "f32[4, 512, 1]" = var_mean_12[1];  var_mean_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:1059, code: inputs_embeds = self.embed_tokens(input) * self.embed_scale
    embedding_2: "f32[4, 512, 768]" = torch.ops.aten.embedding.default(arg101_1, arg263_1, 1);  arg101_1 = arg263_1 = None
    mul_51: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(embedding_2, 1.0);  embedding_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:135, code: positions = torch.arange(
    iota_2: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:137, code: ).expand(bsz, -1)
    expand_2: "i64[4, 512]" = torch.ops.aten.expand.default(iota_2, [4, -1]);  iota_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:139, code: return super().forward(positions + self.offset)
    add_47: "i64[4, 512]" = torch.ops.aten.add.Tensor(expand_2, 2);  expand_2 = None
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/sparse.py:163, code: return F.embedding(
    embedding_3: "f32[4, 512, 768]" = torch.ops.aten.embedding.default(arg1_1, add_47);  arg1_1 = add_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:1074, code: hidden_states = inputs_embeds + positions
    add_48: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_51, embedding_3);  mul_51 = embedding_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:1075, code: hidden_states = self.layernorm_embedding(hidden_states)
    var_mean_13 = torch.ops.aten.var_mean.correction(add_48, [2], correction = 0, keepdim = True)
    getitem_26: "f32[4, 512, 1]" = var_mean_13[0]
    getitem_27: "f32[4, 512, 1]" = var_mean_13[1];  var_mean_13 = None
    sub_19: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_48, getitem_27);  add_48 = getitem_27 = None
    add_49: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
    rsqrt_13: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    mul_52: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_13);  sub_19 = rsqrt_13 = None
    mul_53: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_52, arg102_1);  mul_52 = arg102_1 = None
    add_50: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_53, arg103_1);  mul_53 = arg103_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_123: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_50, [2048, 768])
    permute_66: "f32[768, 768]" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
    addmm_36: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg105_1, view_123, permute_66);  arg105_1 = view_123 = permute_66 = None
    view_124: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_36, [4, 512, 768]);  addmm_36 = None
    mul_54: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_124, 0.125);  view_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_131: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(mul_54, [4, 512, 12, 64]);  mul_54 = None
    permute_71: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_131, [0, 2, 1, 3]);  view_131 = None
    clone_52: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_132: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_52, [48, -1, 64]);  clone_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_125: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_50, [2048, 768])
    permute_67: "f32[768, 768]" = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
    addmm_37: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg107_1, view_125, permute_67);  arg107_1 = view_125 = permute_67 = None
    view_126: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_37, [4, 512, 768]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_127: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_126, [4, -1, 12, 64]);  view_126 = None
    permute_68: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_127, [0, 2, 1, 3]);  view_127 = None
    clone_50: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_133: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_50, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_72: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_133, [0, 2, 1]);  view_133 = None
    bmm_12: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_132, permute_72);  view_132 = permute_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:250, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_135: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_12, [4, 12, 512, 512]);  bmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:97, code: mask_cond = torch.arange(mask.size(-1), device=device)
    iota_1: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:98, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    add_46: "i64[512]" = torch.ops.aten.add.Tensor(iota_1, 1)
    view_122: "i64[512, 1]" = torch.ops.aten.reshape.default(add_46, [512, 1]);  add_46 = None
    lt: "b8[512, 512]" = torch.ops.aten.lt.Tensor(iota_1, view_122);  iota_1 = view_122 = None
    full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:96, code: mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    full_default: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:98, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    where: "f32[512, 512]" = torch.ops.aten.where.self(lt, full_default_1, full_default);  lt = full_default_1 = full_default = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:250, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    unsqueeze_2: "f32[1, 512, 512]" = torch.ops.aten.unsqueeze.default(where, 0);  where = None
    unsqueeze_3: "f32[1, 1, 512, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 1);  unsqueeze_2 = None
    expand_3: "f32[4, 1, 512, 512]" = torch.ops.aten.expand.default(unsqueeze_3, [4, 1, 512, 512]);  unsqueeze_3 = None
    add_51: "f32[4, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_135, expand_3);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:251, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_136: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(add_51, [48, 512, 512]);  add_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_6: "f32[48, 512, 1]" = torch.ops.aten.amax.default(view_136, [-1], True)
    sub_20: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(view_136, amax_6);  view_136 = amax_6 = None
    exp_6: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
    sum_7: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_6: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_128: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_50, [2048, 768])
    permute_69: "f32[768, 768]" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
    addmm_38: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg109_1, view_128, permute_69);  arg109_1 = view_128 = permute_69 = None
    view_129: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_38, [4, 512, 768]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_130: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_129, [4, -1, 12, 64]);  view_129 = None
    permute_70: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None
    clone_51: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_134: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_51, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_13: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(div_6, view_134);  div_6 = view_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_137: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_13, [4, 12, 512, 64]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_73: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_54: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
    view_138: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_54, [4, 512, 768]);  clone_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_139: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_138, [2048, 768]);  view_138 = None
    permute_74: "f32[768, 768]" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
    addmm_39: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg111_1, view_139, permute_74);  arg111_1 = view_139 = permute_74 = None
    view_140: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_39, [4, 512, 768]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:435, code: hidden_states = residual + hidden_states
    add_52: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_50, view_140);  add_50 = view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_14 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
    getitem_28: "f32[4, 512, 1]" = var_mean_14[0]
    getitem_29: "f32[4, 512, 1]" = var_mean_14[1];  var_mean_14 = None
    sub_21: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_52, getitem_29);  add_52 = getitem_29 = None
    add_53: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
    rsqrt_14: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    mul_55: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = rsqrt_14 = None
    mul_56: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_55, arg112_1);  mul_55 = arg112_1 = None
    add_54: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_56, arg113_1);  mul_56 = arg113_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_141: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_54, [2048, 768])
    permute_75: "f32[768, 768]" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
    addmm_40: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg115_1, view_141, permute_75);  arg115_1 = view_141 = permute_75 = None
    view_142: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_40, [4, 512, 768]);  addmm_40 = None
    mul_57: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_142, 0.125);  view_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_149: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(mul_57, [4, 512, 12, 64]);  mul_57 = None
    permute_80: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_149, [0, 2, 1, 3]);  view_149 = None
    clone_58: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format);  permute_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_150: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_58, [48, -1, 64]);  clone_58 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_15: "f32[1, 48, 512, 64]" = torch.ops.aten.unsqueeze.default(view_150, 0);  view_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:348, code: hidden_states = self.final_layer_norm(hidden_states)
    sub_18: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_43, getitem_25);  add_43 = getitem_25 = None
    add_44: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
    rsqrt_12: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
    mul_49: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = rsqrt_12 = None
    mul_50: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_49, arg99_1);  mul_49 = arg99_1 = None
    add_45: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_50, arg100_1);  mul_50 = arg100_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_143: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_45, [2048, 768])
    permute_76: "f32[768, 768]" = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
    addmm_41: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg117_1, view_143, permute_76);  arg117_1 = view_143 = permute_76 = None
    view_144: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_41, [4, 512, 768]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_145: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_144, [4, -1, 12, 64]);  view_144 = None
    permute_77: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_145, [0, 2, 1, 3]);  view_145 = None
    clone_56: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_77, memory_format = torch.contiguous_format);  permute_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_151: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_56, [48, -1, 64])
    
    # No stacktrace found for following nodes
    unsqueeze_default_16: "f32[1, 48, 512, 64]" = torch.ops.aten.unsqueeze.default(view_151, 0);  view_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_146: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_45, [2048, 768])
    permute_78: "f32[768, 768]" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
    addmm_42: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg119_1, view_146, permute_78);  arg119_1 = view_146 = permute_78 = None
    view_147: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_42, [4, 512, 768]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_148: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_147, [4, -1, 12, 64]);  view_147 = None
    permute_79: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
    clone_57: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_152: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_57, [48, -1, 64])
    
    # No stacktrace found for following nodes
    unsqueeze_default_17: "f32[1, 48, 512, 64]" = torch.ops.aten.unsqueeze.default(view_152, 0);  view_152 = None
    _scaled_dot_product_flash_attention_default_5 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_15, unsqueeze_default_16, unsqueeze_default_17, scale = 1.0);  unsqueeze_default_15 = unsqueeze_default_16 = unsqueeze_default_17 = None
    getitem_69: "f32[1, 48, 512, 64]" = _scaled_dot_product_flash_attention_default_5[0];  _scaled_dot_product_flash_attention_default_5 = None
    squeeze_dim_5: "f32[48, 512, 64]" = torch.ops.aten.squeeze.dim(getitem_69, 0);  getitem_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_153: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(squeeze_dim_5, [4, 12, 512, 64]);  squeeze_dim_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_82: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_153, [0, 2, 1, 3]);  view_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_60: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
    view_154: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_60, [4, 512, 768]);  clone_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_155: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_154, [2048, 768]);  view_154 = None
    permute_83: "f32[768, 768]" = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
    addmm_43: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg121_1, view_155, permute_83);  arg121_1 = view_155 = permute_83 = None
    view_156: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_43, [4, 512, 768]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:455, code: hidden_states = residual + hidden_states
    add_55: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_54, view_156);  add_54 = view_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:456, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_15 = torch.ops.aten.var_mean.correction(add_55, [2], correction = 0, keepdim = True)
    getitem_30: "f32[4, 512, 1]" = var_mean_15[0]
    getitem_31: "f32[4, 512, 1]" = var_mean_15[1];  var_mean_15 = None
    sub_23: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_55, getitem_31);  add_55 = getitem_31 = None
    add_56: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
    rsqrt_15: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    mul_58: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = rsqrt_15 = None
    mul_59: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_58, arg122_1);  mul_58 = arg122_1 = None
    add_57: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_59, arg123_1);  mul_59 = arg123_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_157: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_57, [2048, 768])
    permute_84: "f32[768, 3072]" = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
    addmm_44: "f32[2048, 3072]" = torch.ops.aten.addmm.default(arg125_1, view_157, permute_84);  arg125_1 = view_157 = permute_84 = None
    view_158: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_44, [4, 512, 3072]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_60: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_158, 0.5)
    mul_61: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_158, 0.7071067811865476);  view_158 = None
    erf_6: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_61);  mul_61 = None
    add_58: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_62: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_60, add_58);  mul_60 = add_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_159: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_62, [2048, 3072]);  mul_62 = None
    permute_85: "f32[3072, 768]" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
    addmm_45: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg127_1, view_159, permute_85);  arg127_1 = view_159 = permute_85 = None
    view_160: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_45, [4, 512, 768]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:467, code: hidden_states = residual + hidden_states
    add_59: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_57, view_160);  add_57 = view_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_16 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
    getitem_32: "f32[4, 512, 1]" = var_mean_16[0]
    getitem_33: "f32[4, 512, 1]" = var_mean_16[1];  var_mean_16 = None
    sub_24: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_59, getitem_33);  add_59 = getitem_33 = None
    add_60: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
    rsqrt_16: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    mul_63: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = rsqrt_16 = None
    mul_64: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_63, arg128_1);  mul_63 = arg128_1 = None
    add_61: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_64, arg129_1);  mul_64 = arg129_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_161: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_61, [2048, 768])
    permute_86: "f32[768, 768]" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
    addmm_46: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg131_1, view_161, permute_86);  arg131_1 = view_161 = permute_86 = None
    view_162: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_46, [4, 512, 768]);  addmm_46 = None
    mul_65: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_162, 0.125);  view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_169: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(mul_65, [4, 512, 12, 64]);  mul_65 = None
    permute_91: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_169, [0, 2, 1, 3]);  view_169 = None
    clone_66: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_91, memory_format = torch.contiguous_format);  permute_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_170: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_66, [48, -1, 64]);  clone_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_163: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_61, [2048, 768])
    permute_87: "f32[768, 768]" = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
    addmm_47: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg133_1, view_163, permute_87);  arg133_1 = view_163 = permute_87 = None
    view_164: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_47, [4, 512, 768]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_165: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_164, [4, -1, 12, 64]);  view_164 = None
    permute_88: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_165, [0, 2, 1, 3]);  view_165 = None
    clone_64: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_88, memory_format = torch.contiguous_format);  permute_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_171: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_64, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_92: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_171, [0, 2, 1]);  view_171 = None
    bmm_16: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_170, permute_92);  view_170 = permute_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:250, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_173: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_16, [4, 12, 512, 512]);  bmm_16 = None
    add_62: "f32[4, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_173, expand_3);  view_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:251, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_174: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(add_62, [48, 512, 512]);  add_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_8: "f32[48, 512, 1]" = torch.ops.aten.amax.default(view_174, [-1], True)
    sub_25: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(view_174, amax_8);  view_174 = amax_8 = None
    exp_8: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
    sum_9: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_8: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_166: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_61, [2048, 768])
    permute_89: "f32[768, 768]" = torch.ops.aten.permute.default(arg134_1, [1, 0]);  arg134_1 = None
    addmm_48: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg135_1, view_166, permute_89);  arg135_1 = view_166 = permute_89 = None
    view_167: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_48, [4, 512, 768]);  addmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_168: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_167, [4, -1, 12, 64]);  view_167 = None
    permute_90: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
    clone_65: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_172: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_65, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_17: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(div_8, view_172);  div_8 = view_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_175: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_17, [4, 12, 512, 64]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_93: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_175, [0, 2, 1, 3]);  view_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_68: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
    view_176: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_68, [4, 512, 768]);  clone_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_177: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_176, [2048, 768]);  view_176 = None
    permute_94: "f32[768, 768]" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
    addmm_49: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg137_1, view_177, permute_94);  arg137_1 = view_177 = permute_94 = None
    view_178: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_49, [4, 512, 768]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:435, code: hidden_states = residual + hidden_states
    add_63: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_61, view_178);  add_61 = view_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_17 = torch.ops.aten.var_mean.correction(add_63, [2], correction = 0, keepdim = True)
    getitem_34: "f32[4, 512, 1]" = var_mean_17[0]
    getitem_35: "f32[4, 512, 1]" = var_mean_17[1];  var_mean_17 = None
    sub_26: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_63, getitem_35);  add_63 = getitem_35 = None
    add_64: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
    rsqrt_17: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
    mul_66: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = rsqrt_17 = None
    mul_67: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_66, arg138_1);  mul_66 = arg138_1 = None
    add_65: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_67, arg139_1);  mul_67 = arg139_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_179: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_65, [2048, 768])
    permute_95: "f32[768, 768]" = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
    addmm_50: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg141_1, view_179, permute_95);  arg141_1 = view_179 = permute_95 = None
    view_180: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_50, [4, 512, 768]);  addmm_50 = None
    mul_68: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_180, 0.125);  view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_187: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(mul_68, [4, 512, 12, 64]);  mul_68 = None
    permute_100: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_187, [0, 2, 1, 3]);  view_187 = None
    clone_72: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_100, memory_format = torch.contiguous_format);  permute_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_188: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_72, [48, -1, 64]);  clone_72 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_12: "f32[1, 48, 512, 64]" = torch.ops.aten.unsqueeze.default(view_188, 0);  view_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_181: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_45, [2048, 768])
    permute_96: "f32[768, 768]" = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
    addmm_51: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg143_1, view_181, permute_96);  arg143_1 = view_181 = permute_96 = None
    view_182: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_51, [4, 512, 768]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_183: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_182, [4, -1, 12, 64]);  view_182 = None
    permute_97: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None
    clone_70: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_97, memory_format = torch.contiguous_format);  permute_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_189: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_70, [48, -1, 64])
    
    # No stacktrace found for following nodes
    unsqueeze_default_13: "f32[1, 48, 512, 64]" = torch.ops.aten.unsqueeze.default(view_189, 0);  view_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_184: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_45, [2048, 768])
    permute_98: "f32[768, 768]" = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
    addmm_52: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg145_1, view_184, permute_98);  arg145_1 = view_184 = permute_98 = None
    view_185: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_52, [4, 512, 768]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_186: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_185, [4, -1, 12, 64]);  view_185 = None
    permute_99: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
    clone_71: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_99, memory_format = torch.contiguous_format);  permute_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_190: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_71, [48, -1, 64])
    
    # No stacktrace found for following nodes
    unsqueeze_default_14: "f32[1, 48, 512, 64]" = torch.ops.aten.unsqueeze.default(view_190, 0);  view_190 = None
    _scaled_dot_product_flash_attention_default_4 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_12, unsqueeze_default_13, unsqueeze_default_14, scale = 1.0);  unsqueeze_default_12 = unsqueeze_default_13 = unsqueeze_default_14 = None
    getitem_68: "f32[1, 48, 512, 64]" = _scaled_dot_product_flash_attention_default_4[0];  _scaled_dot_product_flash_attention_default_4 = None
    squeeze_dim_4: "f32[48, 512, 64]" = torch.ops.aten.squeeze.dim(getitem_68, 0);  getitem_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_191: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(squeeze_dim_4, [4, 12, 512, 64]);  squeeze_dim_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_102: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_191, [0, 2, 1, 3]);  view_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_74: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_102, memory_format = torch.contiguous_format);  permute_102 = None
    view_192: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_74, [4, 512, 768]);  clone_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_193: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_192, [2048, 768]);  view_192 = None
    permute_103: "f32[768, 768]" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
    addmm_53: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg147_1, view_193, permute_103);  arg147_1 = view_193 = permute_103 = None
    view_194: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_53, [4, 512, 768]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:455, code: hidden_states = residual + hidden_states
    add_66: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_65, view_194);  add_65 = view_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:456, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_18 = torch.ops.aten.var_mean.correction(add_66, [2], correction = 0, keepdim = True)
    getitem_36: "f32[4, 512, 1]" = var_mean_18[0]
    getitem_37: "f32[4, 512, 1]" = var_mean_18[1];  var_mean_18 = None
    sub_28: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_66, getitem_37);  add_66 = getitem_37 = None
    add_67: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
    rsqrt_18: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
    mul_69: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_18);  sub_28 = rsqrt_18 = None
    mul_70: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_69, arg148_1);  mul_69 = arg148_1 = None
    add_68: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_70, arg149_1);  mul_70 = arg149_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_195: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_68, [2048, 768])
    permute_104: "f32[768, 3072]" = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
    addmm_54: "f32[2048, 3072]" = torch.ops.aten.addmm.default(arg151_1, view_195, permute_104);  arg151_1 = view_195 = permute_104 = None
    view_196: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_54, [4, 512, 3072]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_71: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_196, 0.5)
    mul_72: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_196, 0.7071067811865476);  view_196 = None
    erf_7: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_72);  mul_72 = None
    add_69: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_73: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_71, add_69);  mul_71 = add_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_197: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_73, [2048, 3072]);  mul_73 = None
    permute_105: "f32[3072, 768]" = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
    addmm_55: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg153_1, view_197, permute_105);  arg153_1 = view_197 = permute_105 = None
    view_198: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_55, [4, 512, 768]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:467, code: hidden_states = residual + hidden_states
    add_70: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_68, view_198);  add_68 = view_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_19 = torch.ops.aten.var_mean.correction(add_70, [2], correction = 0, keepdim = True)
    getitem_38: "f32[4, 512, 1]" = var_mean_19[0]
    getitem_39: "f32[4, 512, 1]" = var_mean_19[1];  var_mean_19 = None
    sub_29: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_70, getitem_39);  add_70 = getitem_39 = None
    add_71: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
    rsqrt_19: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
    mul_74: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = rsqrt_19 = None
    mul_75: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_74, arg154_1);  mul_74 = arg154_1 = None
    add_72: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_75, arg155_1);  mul_75 = arg155_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_199: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_72, [2048, 768])
    permute_106: "f32[768, 768]" = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
    addmm_56: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg157_1, view_199, permute_106);  arg157_1 = view_199 = permute_106 = None
    view_200: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_56, [4, 512, 768]);  addmm_56 = None
    mul_76: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_200, 0.125);  view_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_207: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(mul_76, [4, 512, 12, 64]);  mul_76 = None
    permute_111: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_207, [0, 2, 1, 3]);  view_207 = None
    clone_80: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_111, memory_format = torch.contiguous_format);  permute_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_208: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_80, [48, -1, 64]);  clone_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_201: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_72, [2048, 768])
    permute_107: "f32[768, 768]" = torch.ops.aten.permute.default(arg158_1, [1, 0]);  arg158_1 = None
    addmm_57: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg159_1, view_201, permute_107);  arg159_1 = view_201 = permute_107 = None
    view_202: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_57, [4, 512, 768]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_203: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_202, [4, -1, 12, 64]);  view_202 = None
    permute_108: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_203, [0, 2, 1, 3]);  view_203 = None
    clone_78: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_108, memory_format = torch.contiguous_format);  permute_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_209: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_78, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_112: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_209, [0, 2, 1]);  view_209 = None
    bmm_20: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_208, permute_112);  view_208 = permute_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:250, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_211: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_20, [4, 12, 512, 512]);  bmm_20 = None
    add_73: "f32[4, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_211, expand_3);  view_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:251, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_212: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(add_73, [48, 512, 512]);  add_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_10: "f32[48, 512, 1]" = torch.ops.aten.amax.default(view_212, [-1], True)
    sub_30: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(view_212, amax_10);  view_212 = amax_10 = None
    exp_10: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_30);  sub_30 = None
    sum_11: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_10: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_204: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_72, [2048, 768])
    permute_109: "f32[768, 768]" = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
    addmm_58: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg161_1, view_204, permute_109);  arg161_1 = view_204 = permute_109 = None
    view_205: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_58, [4, 512, 768]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_206: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_205, [4, -1, 12, 64]);  view_205 = None
    permute_110: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
    clone_79: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_110, memory_format = torch.contiguous_format);  permute_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_210: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_79, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_21: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(div_10, view_210);  div_10 = view_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_213: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_21, [4, 12, 512, 64]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_113: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_213, [0, 2, 1, 3]);  view_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_82: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_113, memory_format = torch.contiguous_format);  permute_113 = None
    view_214: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_82, [4, 512, 768]);  clone_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_215: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_214, [2048, 768]);  view_214 = None
    permute_114: "f32[768, 768]" = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
    addmm_59: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg163_1, view_215, permute_114);  arg163_1 = view_215 = permute_114 = None
    view_216: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_59, [4, 512, 768]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:435, code: hidden_states = residual + hidden_states
    add_74: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_72, view_216);  add_72 = view_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_20 = torch.ops.aten.var_mean.correction(add_74, [2], correction = 0, keepdim = True)
    getitem_40: "f32[4, 512, 1]" = var_mean_20[0]
    getitem_41: "f32[4, 512, 1]" = var_mean_20[1];  var_mean_20 = None
    sub_31: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_74, getitem_41);  add_74 = getitem_41 = None
    add_75: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
    rsqrt_20: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
    mul_77: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_20);  sub_31 = rsqrt_20 = None
    mul_78: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_77, arg164_1);  mul_77 = arg164_1 = None
    add_76: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_78, arg165_1);  mul_78 = arg165_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_217: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_76, [2048, 768])
    permute_115: "f32[768, 768]" = torch.ops.aten.permute.default(arg166_1, [1, 0]);  arg166_1 = None
    addmm_60: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg167_1, view_217, permute_115);  arg167_1 = view_217 = permute_115 = None
    view_218: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_60, [4, 512, 768]);  addmm_60 = None
    mul_79: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_218, 0.125);  view_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_225: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(mul_79, [4, 512, 12, 64]);  mul_79 = None
    permute_120: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_225, [0, 2, 1, 3]);  view_225 = None
    clone_86: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_120, memory_format = torch.contiguous_format);  permute_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_226: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_86, [48, -1, 64]);  clone_86 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_9: "f32[1, 48, 512, 64]" = torch.ops.aten.unsqueeze.default(view_226, 0);  view_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_219: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_45, [2048, 768])
    permute_116: "f32[768, 768]" = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
    addmm_61: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg169_1, view_219, permute_116);  arg169_1 = view_219 = permute_116 = None
    view_220: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_61, [4, 512, 768]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_221: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_220, [4, -1, 12, 64]);  view_220 = None
    permute_117: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_221, [0, 2, 1, 3]);  view_221 = None
    clone_84: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_227: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_84, [48, -1, 64])
    
    # No stacktrace found for following nodes
    unsqueeze_default_10: "f32[1, 48, 512, 64]" = torch.ops.aten.unsqueeze.default(view_227, 0);  view_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_222: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_45, [2048, 768])
    permute_118: "f32[768, 768]" = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
    addmm_62: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg171_1, view_222, permute_118);  arg171_1 = view_222 = permute_118 = None
    view_223: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_62, [4, 512, 768]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_224: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_223, [4, -1, 12, 64]);  view_223 = None
    permute_119: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None
    clone_85: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_119, memory_format = torch.contiguous_format);  permute_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_228: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_85, [48, -1, 64])
    
    # No stacktrace found for following nodes
    unsqueeze_default_11: "f32[1, 48, 512, 64]" = torch.ops.aten.unsqueeze.default(view_228, 0);  view_228 = None
    _scaled_dot_product_flash_attention_default_3 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_9, unsqueeze_default_10, unsqueeze_default_11, scale = 1.0);  unsqueeze_default_9 = unsqueeze_default_10 = unsqueeze_default_11 = None
    getitem_67: "f32[1, 48, 512, 64]" = _scaled_dot_product_flash_attention_default_3[0];  _scaled_dot_product_flash_attention_default_3 = None
    squeeze_dim_3: "f32[48, 512, 64]" = torch.ops.aten.squeeze.dim(getitem_67, 0);  getitem_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_229: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(squeeze_dim_3, [4, 12, 512, 64]);  squeeze_dim_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_122: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_229, [0, 2, 1, 3]);  view_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_88: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_122, memory_format = torch.contiguous_format);  permute_122 = None
    view_230: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_88, [4, 512, 768]);  clone_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_231: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_230, [2048, 768]);  view_230 = None
    permute_123: "f32[768, 768]" = torch.ops.aten.permute.default(arg172_1, [1, 0]);  arg172_1 = None
    addmm_63: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg173_1, view_231, permute_123);  arg173_1 = view_231 = permute_123 = None
    view_232: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_63, [4, 512, 768]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:455, code: hidden_states = residual + hidden_states
    add_77: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_76, view_232);  add_76 = view_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:456, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_21 = torch.ops.aten.var_mean.correction(add_77, [2], correction = 0, keepdim = True)
    getitem_42: "f32[4, 512, 1]" = var_mean_21[0]
    getitem_43: "f32[4, 512, 1]" = var_mean_21[1];  var_mean_21 = None
    sub_33: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_77, getitem_43);  add_77 = getitem_43 = None
    add_78: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
    rsqrt_21: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    mul_80: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_21);  sub_33 = rsqrt_21 = None
    mul_81: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_80, arg174_1);  mul_80 = arg174_1 = None
    add_79: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_81, arg175_1);  mul_81 = arg175_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_233: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_79, [2048, 768])
    permute_124: "f32[768, 3072]" = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
    addmm_64: "f32[2048, 3072]" = torch.ops.aten.addmm.default(arg177_1, view_233, permute_124);  arg177_1 = view_233 = permute_124 = None
    view_234: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_64, [4, 512, 3072]);  addmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_82: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_234, 0.5)
    mul_83: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_234, 0.7071067811865476);  view_234 = None
    erf_8: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
    add_80: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_84: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_82, add_80);  mul_82 = add_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_235: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_84, [2048, 3072]);  mul_84 = None
    permute_125: "f32[3072, 768]" = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
    addmm_65: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg179_1, view_235, permute_125);  arg179_1 = view_235 = permute_125 = None
    view_236: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_65, [4, 512, 768]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:467, code: hidden_states = residual + hidden_states
    add_81: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_79, view_236);  add_79 = view_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_22 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
    getitem_44: "f32[4, 512, 1]" = var_mean_22[0]
    getitem_45: "f32[4, 512, 1]" = var_mean_22[1];  var_mean_22 = None
    sub_34: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_81, getitem_45);  add_81 = getitem_45 = None
    add_82: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
    rsqrt_22: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
    mul_85: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_22);  sub_34 = rsqrt_22 = None
    mul_86: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_85, arg180_1);  mul_85 = arg180_1 = None
    add_83: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_86, arg181_1);  mul_86 = arg181_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_237: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_83, [2048, 768])
    permute_126: "f32[768, 768]" = torch.ops.aten.permute.default(arg182_1, [1, 0]);  arg182_1 = None
    addmm_66: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg183_1, view_237, permute_126);  arg183_1 = view_237 = permute_126 = None
    view_238: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_66, [4, 512, 768]);  addmm_66 = None
    mul_87: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_238, 0.125);  view_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_245: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(mul_87, [4, 512, 12, 64]);  mul_87 = None
    permute_131: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_245, [0, 2, 1, 3]);  view_245 = None
    clone_94: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_131, memory_format = torch.contiguous_format);  permute_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_246: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_94, [48, -1, 64]);  clone_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_239: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_83, [2048, 768])
    permute_127: "f32[768, 768]" = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
    addmm_67: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg185_1, view_239, permute_127);  arg185_1 = view_239 = permute_127 = None
    view_240: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_67, [4, 512, 768]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_241: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_240, [4, -1, 12, 64]);  view_240 = None
    permute_128: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_241, [0, 2, 1, 3]);  view_241 = None
    clone_92: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_247: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_92, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_132: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_247, [0, 2, 1]);  view_247 = None
    bmm_24: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_246, permute_132);  view_246 = permute_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:250, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_249: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_24, [4, 12, 512, 512]);  bmm_24 = None
    add_84: "f32[4, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_249, expand_3);  view_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:251, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_250: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(add_84, [48, 512, 512]);  add_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_12: "f32[48, 512, 1]" = torch.ops.aten.amax.default(view_250, [-1], True)
    sub_35: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(view_250, amax_12);  view_250 = amax_12 = None
    exp_12: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_35);  sub_35 = None
    sum_13: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
    div_12: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_242: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_83, [2048, 768])
    permute_129: "f32[768, 768]" = torch.ops.aten.permute.default(arg186_1, [1, 0]);  arg186_1 = None
    addmm_68: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg187_1, view_242, permute_129);  arg187_1 = view_242 = permute_129 = None
    view_243: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_68, [4, 512, 768]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_244: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_243, [4, -1, 12, 64]);  view_243 = None
    permute_130: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_244, [0, 2, 1, 3]);  view_244 = None
    clone_93: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_130, memory_format = torch.contiguous_format);  permute_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_248: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_93, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_25: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(div_12, view_248);  div_12 = view_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_251: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_25, [4, 12, 512, 64]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_133: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_251, [0, 2, 1, 3]);  view_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_96: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_133, memory_format = torch.contiguous_format);  permute_133 = None
    view_252: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_96, [4, 512, 768]);  clone_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_253: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_252, [2048, 768]);  view_252 = None
    permute_134: "f32[768, 768]" = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
    addmm_69: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg189_1, view_253, permute_134);  arg189_1 = view_253 = permute_134 = None
    view_254: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_69, [4, 512, 768]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:435, code: hidden_states = residual + hidden_states
    add_85: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_83, view_254);  add_83 = view_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_23 = torch.ops.aten.var_mean.correction(add_85, [2], correction = 0, keepdim = True)
    getitem_46: "f32[4, 512, 1]" = var_mean_23[0]
    getitem_47: "f32[4, 512, 1]" = var_mean_23[1];  var_mean_23 = None
    sub_36: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_85, getitem_47);  add_85 = getitem_47 = None
    add_86: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
    rsqrt_23: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
    mul_88: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_23);  sub_36 = rsqrt_23 = None
    mul_89: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_88, arg190_1);  mul_88 = arg190_1 = None
    add_87: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_89, arg191_1);  mul_89 = arg191_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_255: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_87, [2048, 768])
    permute_135: "f32[768, 768]" = torch.ops.aten.permute.default(arg192_1, [1, 0]);  arg192_1 = None
    addmm_70: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg193_1, view_255, permute_135);  arg193_1 = view_255 = permute_135 = None
    view_256: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_70, [4, 512, 768]);  addmm_70 = None
    mul_90: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_256, 0.125);  view_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_263: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(mul_90, [4, 512, 12, 64]);  mul_90 = None
    permute_140: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_263, [0, 2, 1, 3]);  view_263 = None
    clone_100: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_140, memory_format = torch.contiguous_format);  permute_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_264: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_100, [48, -1, 64]);  clone_100 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_6: "f32[1, 48, 512, 64]" = torch.ops.aten.unsqueeze.default(view_264, 0);  view_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_257: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_45, [2048, 768])
    permute_136: "f32[768, 768]" = torch.ops.aten.permute.default(arg194_1, [1, 0]);  arg194_1 = None
    addmm_71: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg195_1, view_257, permute_136);  arg195_1 = view_257 = permute_136 = None
    view_258: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_71, [4, 512, 768]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_259: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_258, [4, -1, 12, 64]);  view_258 = None
    permute_137: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_259, [0, 2, 1, 3]);  view_259 = None
    clone_98: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_137, memory_format = torch.contiguous_format);  permute_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_265: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_98, [48, -1, 64])
    
    # No stacktrace found for following nodes
    unsqueeze_default_7: "f32[1, 48, 512, 64]" = torch.ops.aten.unsqueeze.default(view_265, 0);  view_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_260: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_45, [2048, 768])
    permute_138: "f32[768, 768]" = torch.ops.aten.permute.default(arg196_1, [1, 0]);  arg196_1 = None
    addmm_72: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg197_1, view_260, permute_138);  arg197_1 = view_260 = permute_138 = None
    view_261: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_72, [4, 512, 768]);  addmm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_262: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_261, [4, -1, 12, 64]);  view_261 = None
    permute_139: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_262, [0, 2, 1, 3]);  view_262 = None
    clone_99: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_266: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_99, [48, -1, 64])
    
    # No stacktrace found for following nodes
    unsqueeze_default_8: "f32[1, 48, 512, 64]" = torch.ops.aten.unsqueeze.default(view_266, 0);  view_266 = None
    _scaled_dot_product_flash_attention_default_2 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_6, unsqueeze_default_7, unsqueeze_default_8, scale = 1.0);  unsqueeze_default_6 = unsqueeze_default_7 = unsqueeze_default_8 = None
    getitem_66: "f32[1, 48, 512, 64]" = _scaled_dot_product_flash_attention_default_2[0];  _scaled_dot_product_flash_attention_default_2 = None
    squeeze_dim_2: "f32[48, 512, 64]" = torch.ops.aten.squeeze.dim(getitem_66, 0);  getitem_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_267: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(squeeze_dim_2, [4, 12, 512, 64]);  squeeze_dim_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_142: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_267, [0, 2, 1, 3]);  view_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_102: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_142, memory_format = torch.contiguous_format);  permute_142 = None
    view_268: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_102, [4, 512, 768]);  clone_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_269: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_268, [2048, 768]);  view_268 = None
    permute_143: "f32[768, 768]" = torch.ops.aten.permute.default(arg198_1, [1, 0]);  arg198_1 = None
    addmm_73: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg199_1, view_269, permute_143);  arg199_1 = view_269 = permute_143 = None
    view_270: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_73, [4, 512, 768]);  addmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:455, code: hidden_states = residual + hidden_states
    add_88: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_87, view_270);  add_87 = view_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:456, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_24 = torch.ops.aten.var_mean.correction(add_88, [2], correction = 0, keepdim = True)
    getitem_48: "f32[4, 512, 1]" = var_mean_24[0]
    getitem_49: "f32[4, 512, 1]" = var_mean_24[1];  var_mean_24 = None
    sub_38: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_88, getitem_49);  add_88 = getitem_49 = None
    add_89: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
    rsqrt_24: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
    mul_91: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_24);  sub_38 = rsqrt_24 = None
    mul_92: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_91, arg200_1);  mul_91 = arg200_1 = None
    add_90: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_92, arg201_1);  mul_92 = arg201_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_271: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_90, [2048, 768])
    permute_144: "f32[768, 3072]" = torch.ops.aten.permute.default(arg202_1, [1, 0]);  arg202_1 = None
    addmm_74: "f32[2048, 3072]" = torch.ops.aten.addmm.default(arg203_1, view_271, permute_144);  arg203_1 = view_271 = permute_144 = None
    view_272: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_74, [4, 512, 3072]);  addmm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_93: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_272, 0.5)
    mul_94: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_272, 0.7071067811865476);  view_272 = None
    erf_9: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_94);  mul_94 = None
    add_91: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_95: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_93, add_91);  mul_93 = add_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_273: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_95, [2048, 3072]);  mul_95 = None
    permute_145: "f32[3072, 768]" = torch.ops.aten.permute.default(arg204_1, [1, 0]);  arg204_1 = None
    addmm_75: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg205_1, view_273, permute_145);  arg205_1 = view_273 = permute_145 = None
    view_274: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_75, [4, 512, 768]);  addmm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:467, code: hidden_states = residual + hidden_states
    add_92: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_90, view_274);  add_90 = view_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_25 = torch.ops.aten.var_mean.correction(add_92, [2], correction = 0, keepdim = True)
    getitem_50: "f32[4, 512, 1]" = var_mean_25[0]
    getitem_51: "f32[4, 512, 1]" = var_mean_25[1];  var_mean_25 = None
    sub_39: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_92, getitem_51);  add_92 = getitem_51 = None
    add_93: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
    rsqrt_25: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
    mul_96: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_25);  sub_39 = rsqrt_25 = None
    mul_97: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_96, arg206_1);  mul_96 = arg206_1 = None
    add_94: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_97, arg207_1);  mul_97 = arg207_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_275: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_94, [2048, 768])
    permute_146: "f32[768, 768]" = torch.ops.aten.permute.default(arg208_1, [1, 0]);  arg208_1 = None
    addmm_76: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg209_1, view_275, permute_146);  arg209_1 = view_275 = permute_146 = None
    view_276: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_76, [4, 512, 768]);  addmm_76 = None
    mul_98: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_276, 0.125);  view_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_283: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(mul_98, [4, 512, 12, 64]);  mul_98 = None
    permute_151: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_283, [0, 2, 1, 3]);  view_283 = None
    clone_108: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_151, memory_format = torch.contiguous_format);  permute_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_284: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_108, [48, -1, 64]);  clone_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_277: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_94, [2048, 768])
    permute_147: "f32[768, 768]" = torch.ops.aten.permute.default(arg210_1, [1, 0]);  arg210_1 = None
    addmm_77: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg211_1, view_277, permute_147);  arg211_1 = view_277 = permute_147 = None
    view_278: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_77, [4, 512, 768]);  addmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_279: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_278, [4, -1, 12, 64]);  view_278 = None
    permute_148: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_279, [0, 2, 1, 3]);  view_279 = None
    clone_106: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_148, memory_format = torch.contiguous_format);  permute_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_285: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_106, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_152: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_285, [0, 2, 1]);  view_285 = None
    bmm_28: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_284, permute_152);  view_284 = permute_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:250, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_287: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_28, [4, 12, 512, 512]);  bmm_28 = None
    add_95: "f32[4, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_287, expand_3);  view_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:251, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_288: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(add_95, [48, 512, 512]);  add_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_14: "f32[48, 512, 1]" = torch.ops.aten.amax.default(view_288, [-1], True)
    sub_40: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(view_288, amax_14);  view_288 = amax_14 = None
    exp_14: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_40);  sub_40 = None
    sum_15: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
    div_14: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_280: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_94, [2048, 768])
    permute_149: "f32[768, 768]" = torch.ops.aten.permute.default(arg212_1, [1, 0]);  arg212_1 = None
    addmm_78: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg213_1, view_280, permute_149);  arg213_1 = view_280 = permute_149 = None
    view_281: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_78, [4, 512, 768]);  addmm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_282: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_281, [4, -1, 12, 64]);  view_281 = None
    permute_150: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_282, [0, 2, 1, 3]);  view_282 = None
    clone_107: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_286: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_107, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_29: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(div_14, view_286);  div_14 = view_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_289: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_29, [4, 12, 512, 64]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_153: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_289, [0, 2, 1, 3]);  view_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_110: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_153, memory_format = torch.contiguous_format);  permute_153 = None
    view_290: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_110, [4, 512, 768]);  clone_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_291: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_290, [2048, 768]);  view_290 = None
    permute_154: "f32[768, 768]" = torch.ops.aten.permute.default(arg214_1, [1, 0]);  arg214_1 = None
    addmm_79: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg215_1, view_291, permute_154);  arg215_1 = view_291 = permute_154 = None
    view_292: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_79, [4, 512, 768]);  addmm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:435, code: hidden_states = residual + hidden_states
    add_96: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_94, view_292);  add_94 = view_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_26 = torch.ops.aten.var_mean.correction(add_96, [2], correction = 0, keepdim = True)
    getitem_52: "f32[4, 512, 1]" = var_mean_26[0]
    getitem_53: "f32[4, 512, 1]" = var_mean_26[1];  var_mean_26 = None
    sub_41: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_96, getitem_53);  add_96 = getitem_53 = None
    add_97: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
    rsqrt_26: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
    mul_99: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_26);  sub_41 = rsqrt_26 = None
    mul_100: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_99, arg216_1);  mul_99 = arg216_1 = None
    add_98: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_100, arg217_1);  mul_100 = arg217_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_293: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_98, [2048, 768])
    permute_155: "f32[768, 768]" = torch.ops.aten.permute.default(arg218_1, [1, 0]);  arg218_1 = None
    addmm_80: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg219_1, view_293, permute_155);  arg219_1 = view_293 = permute_155 = None
    view_294: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_80, [4, 512, 768]);  addmm_80 = None
    mul_101: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_294, 0.125);  view_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_301: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(mul_101, [4, 512, 12, 64]);  mul_101 = None
    permute_160: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_301, [0, 2, 1, 3]);  view_301 = None
    clone_114: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_160, memory_format = torch.contiguous_format);  permute_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_302: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_114, [48, -1, 64]);  clone_114 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_3: "f32[1, 48, 512, 64]" = torch.ops.aten.unsqueeze.default(view_302, 0);  view_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_295: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_45, [2048, 768])
    permute_156: "f32[768, 768]" = torch.ops.aten.permute.default(arg220_1, [1, 0]);  arg220_1 = None
    addmm_81: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg221_1, view_295, permute_156);  arg221_1 = view_295 = permute_156 = None
    view_296: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_81, [4, 512, 768]);  addmm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_297: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_296, [4, -1, 12, 64]);  view_296 = None
    permute_157: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_297, [0, 2, 1, 3]);  view_297 = None
    clone_112: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_157, memory_format = torch.contiguous_format);  permute_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_303: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_112, [48, -1, 64])
    
    # No stacktrace found for following nodes
    unsqueeze_default_4: "f32[1, 48, 512, 64]" = torch.ops.aten.unsqueeze.default(view_303, 0);  view_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_298: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_45, [2048, 768])
    permute_158: "f32[768, 768]" = torch.ops.aten.permute.default(arg222_1, [1, 0]);  arg222_1 = None
    addmm_82: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg223_1, view_298, permute_158);  arg223_1 = view_298 = permute_158 = None
    view_299: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_82, [4, 512, 768]);  addmm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_300: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_299, [4, -1, 12, 64]);  view_299 = None
    permute_159: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_300, [0, 2, 1, 3]);  view_300 = None
    clone_113: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_159, memory_format = torch.contiguous_format);  permute_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_304: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_113, [48, -1, 64])
    
    # No stacktrace found for following nodes
    unsqueeze_default_5: "f32[1, 48, 512, 64]" = torch.ops.aten.unsqueeze.default(view_304, 0);  view_304 = None
    _scaled_dot_product_flash_attention_default_1 = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default_3, unsqueeze_default_4, unsqueeze_default_5, scale = 1.0);  unsqueeze_default_3 = unsqueeze_default_4 = unsqueeze_default_5 = None
    getitem_65: "f32[1, 48, 512, 64]" = _scaled_dot_product_flash_attention_default_1[0];  _scaled_dot_product_flash_attention_default_1 = None
    squeeze_dim_1: "f32[48, 512, 64]" = torch.ops.aten.squeeze.dim(getitem_65, 0);  getitem_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_305: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(squeeze_dim_1, [4, 12, 512, 64]);  squeeze_dim_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_162: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_305, [0, 2, 1, 3]);  view_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_116: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_162, memory_format = torch.contiguous_format);  permute_162 = None
    view_306: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_116, [4, 512, 768]);  clone_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_307: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_306, [2048, 768]);  view_306 = None
    permute_163: "f32[768, 768]" = torch.ops.aten.permute.default(arg224_1, [1, 0]);  arg224_1 = None
    addmm_83: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg225_1, view_307, permute_163);  arg225_1 = view_307 = permute_163 = None
    view_308: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_83, [4, 512, 768]);  addmm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:455, code: hidden_states = residual + hidden_states
    add_99: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_98, view_308);  add_98 = view_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:456, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_27 = torch.ops.aten.var_mean.correction(add_99, [2], correction = 0, keepdim = True)
    getitem_54: "f32[4, 512, 1]" = var_mean_27[0]
    getitem_55: "f32[4, 512, 1]" = var_mean_27[1];  var_mean_27 = None
    sub_43: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_99, getitem_55);  add_99 = getitem_55 = None
    add_100: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
    rsqrt_27: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
    mul_102: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_27);  sub_43 = rsqrt_27 = None
    mul_103: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_102, arg226_1);  mul_102 = arg226_1 = None
    add_101: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_103, arg227_1);  mul_103 = arg227_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_309: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_101, [2048, 768])
    permute_164: "f32[768, 3072]" = torch.ops.aten.permute.default(arg228_1, [1, 0]);  arg228_1 = None
    addmm_84: "f32[2048, 3072]" = torch.ops.aten.addmm.default(arg229_1, view_309, permute_164);  arg229_1 = view_309 = permute_164 = None
    view_310: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_84, [4, 512, 3072]);  addmm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_104: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_310, 0.5)
    mul_105: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_310, 0.7071067811865476);  view_310 = None
    erf_10: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_105);  mul_105 = None
    add_102: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_106: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_104, add_102);  mul_104 = add_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_311: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_106, [2048, 3072]);  mul_106 = None
    permute_165: "f32[3072, 768]" = torch.ops.aten.permute.default(arg230_1, [1, 0]);  arg230_1 = None
    addmm_85: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg231_1, view_311, permute_165);  arg231_1 = view_311 = permute_165 = None
    view_312: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_85, [4, 512, 768]);  addmm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:467, code: hidden_states = residual + hidden_states
    add_103: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_101, view_312);  add_101 = view_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_28 = torch.ops.aten.var_mean.correction(add_103, [2], correction = 0, keepdim = True)
    getitem_56: "f32[4, 512, 1]" = var_mean_28[0]
    getitem_57: "f32[4, 512, 1]" = var_mean_28[1];  var_mean_28 = None
    sub_44: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_103, getitem_57);  add_103 = getitem_57 = None
    add_104: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
    rsqrt_28: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
    mul_107: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_28);  sub_44 = rsqrt_28 = None
    mul_108: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_107, arg232_1);  mul_107 = arg232_1 = None
    add_105: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_108, arg233_1);  mul_108 = arg233_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_313: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_105, [2048, 768])
    permute_166: "f32[768, 768]" = torch.ops.aten.permute.default(arg234_1, [1, 0]);  arg234_1 = None
    addmm_86: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg235_1, view_313, permute_166);  arg235_1 = view_313 = permute_166 = None
    view_314: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_86, [4, 512, 768]);  addmm_86 = None
    mul_109: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_314, 0.125);  view_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_321: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(mul_109, [4, 512, 12, 64]);  mul_109 = None
    permute_171: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_321, [0, 2, 1, 3]);  view_321 = None
    clone_122: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_171, memory_format = torch.contiguous_format);  permute_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_322: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_122, [48, -1, 64]);  clone_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_315: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_105, [2048, 768])
    permute_167: "f32[768, 768]" = torch.ops.aten.permute.default(arg236_1, [1, 0]);  arg236_1 = None
    addmm_87: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg237_1, view_315, permute_167);  arg237_1 = view_315 = permute_167 = None
    view_316: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_87, [4, 512, 768]);  addmm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_317: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_316, [4, -1, 12, 64]);  view_316 = None
    permute_168: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_317, [0, 2, 1, 3]);  view_317 = None
    clone_120: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_168, memory_format = torch.contiguous_format);  permute_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_323: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_120, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_172: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_323, [0, 2, 1]);  view_323 = None
    bmm_32: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_322, permute_172);  view_322 = permute_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:250, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_325: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_32, [4, 12, 512, 512]);  bmm_32 = None
    add_106: "f32[4, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_325, expand_3);  view_325 = expand_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:251, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_326: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(add_106, [48, 512, 512]);  add_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_16: "f32[48, 512, 1]" = torch.ops.aten.amax.default(view_326, [-1], True)
    sub_45: "f32[48, 512, 512]" = torch.ops.aten.sub.Tensor(view_326, amax_16);  view_326 = amax_16 = None
    exp_16: "f32[48, 512, 512]" = torch.ops.aten.exp.default(sub_45);  sub_45 = None
    sum_17: "f32[48, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
    div_16: "f32[48, 512, 512]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_318: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_105, [2048, 768])
    permute_169: "f32[768, 768]" = torch.ops.aten.permute.default(arg238_1, [1, 0]);  arg238_1 = None
    addmm_88: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg239_1, view_318, permute_169);  arg239_1 = view_318 = permute_169 = None
    view_319: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_88, [4, 512, 768]);  addmm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_320: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_319, [4, -1, 12, 64]);  view_319 = None
    permute_170: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_320, [0, 2, 1, 3]);  view_320 = None
    clone_121: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_170, memory_format = torch.contiguous_format);  permute_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_324: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_121, [48, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_33: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(div_16, view_324);  div_16 = view_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_327: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_33, [4, 12, 512, 64]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_173: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_327, [0, 2, 1, 3]);  view_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_124: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_173, memory_format = torch.contiguous_format);  permute_173 = None
    view_328: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_124, [4, 512, 768]);  clone_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_329: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_328, [2048, 768]);  view_328 = None
    permute_174: "f32[768, 768]" = torch.ops.aten.permute.default(arg240_1, [1, 0]);  arg240_1 = None
    addmm_89: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg241_1, view_329, permute_174);  arg241_1 = view_329 = permute_174 = None
    view_330: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_89, [4, 512, 768]);  addmm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:435, code: hidden_states = residual + hidden_states
    add_107: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_105, view_330);  add_105 = view_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_29 = torch.ops.aten.var_mean.correction(add_107, [2], correction = 0, keepdim = True)
    getitem_58: "f32[4, 512, 1]" = var_mean_29[0]
    getitem_59: "f32[4, 512, 1]" = var_mean_29[1];  var_mean_29 = None
    sub_46: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_107, getitem_59);  add_107 = getitem_59 = None
    add_108: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
    rsqrt_29: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
    mul_110: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_29);  sub_46 = rsqrt_29 = None
    mul_111: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_110, arg242_1);  mul_110 = arg242_1 = None
    add_109: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_111, arg243_1);  mul_111 = arg243_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_331: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_109, [2048, 768])
    permute_175: "f32[768, 768]" = torch.ops.aten.permute.default(arg244_1, [1, 0]);  arg244_1 = None
    addmm_90: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg245_1, view_331, permute_175);  arg245_1 = view_331 = permute_175 = None
    view_332: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_90, [4, 512, 768]);  addmm_90 = None
    mul_112: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_332, 0.125);  view_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_339: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(mul_112, [4, 512, 12, 64]);  mul_112 = None
    permute_180: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_339, [0, 2, 1, 3]);  view_339 = None
    clone_128: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_180, memory_format = torch.contiguous_format);  permute_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_340: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_128, [48, -1, 64]);  clone_128 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default: "f32[1, 48, 512, 64]" = torch.ops.aten.unsqueeze.default(view_340, 0);  view_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:208, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_333: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_45, [2048, 768])
    permute_176: "f32[768, 768]" = torch.ops.aten.permute.default(arg246_1, [1, 0]);  arg246_1 = None
    addmm_91: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg247_1, view_333, permute_176);  arg247_1 = view_333 = permute_176 = None
    view_334: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_91, [4, 512, 768]);  addmm_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_335: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_334, [4, -1, 12, 64]);  view_334 = None
    permute_177: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_335, [0, 2, 1, 3]);  view_335 = None
    clone_126: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_341: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_126, [48, -1, 64])
    
    # No stacktrace found for following nodes
    unsqueeze_default_1: "f32[1, 48, 512, 64]" = torch.ops.aten.unsqueeze.default(view_341, 0);  view_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:209, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_336: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_45, [2048, 768])
    permute_178: "f32[768, 768]" = torch.ops.aten.permute.default(arg248_1, [1, 0]);  arg248_1 = None
    addmm_92: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg249_1, view_336, permute_178);  arg249_1 = view_336 = permute_178 = None
    view_337: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_92, [4, 512, 768]);  addmm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_338: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_337, [4, -1, 12, 64]);  view_337 = None
    permute_179: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_338, [0, 2, 1, 3]);  view_338 = None
    clone_127: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_179, memory_format = torch.contiguous_format);  permute_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_342: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_127, [48, -1, 64])
    
    # No stacktrace found for following nodes
    unsqueeze_default_2: "f32[1, 48, 512, 64]" = torch.ops.aten.unsqueeze.default(view_342, 0);  view_342 = None
    _scaled_dot_product_flash_attention_default = torch.ops.aten._scaled_dot_product_flash_attention.default(unsqueeze_default, unsqueeze_default_1, unsqueeze_default_2, scale = 1.0);  unsqueeze_default = unsqueeze_default_1 = unsqueeze_default_2 = None
    getitem_64: "f32[1, 48, 512, 64]" = _scaled_dot_product_flash_attention_default[0];  _scaled_dot_product_flash_attention_default = None
    squeeze_dim: "f32[48, 512, 64]" = torch.ops.aten.squeeze.dim(getitem_64, 0);  getitem_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_343: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(squeeze_dim, [4, 12, 512, 64]);  squeeze_dim = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_182: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_343, [0, 2, 1, 3]);  view_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_130: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_182, memory_format = torch.contiguous_format);  permute_182 = None
    view_344: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_130, [4, 512, 768]);  clone_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_345: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_344, [2048, 768]);  view_344 = None
    permute_183: "f32[768, 768]" = torch.ops.aten.permute.default(arg250_1, [1, 0]);  arg250_1 = None
    addmm_93: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg251_1, view_345, permute_183);  arg251_1 = view_345 = permute_183 = None
    view_346: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_93, [4, 512, 768]);  addmm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:455, code: hidden_states = residual + hidden_states
    add_110: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_109, view_346);  add_109 = view_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:456, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_30 = torch.ops.aten.var_mean.correction(add_110, [2], correction = 0, keepdim = True)
    getitem_60: "f32[4, 512, 1]" = var_mean_30[0]
    getitem_61: "f32[4, 512, 1]" = var_mean_30[1];  var_mean_30 = None
    sub_48: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_110, getitem_61);  add_110 = getitem_61 = None
    add_111: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
    rsqrt_30: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
    mul_113: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_30);  sub_48 = rsqrt_30 = None
    mul_114: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_113, arg252_1);  mul_113 = arg252_1 = None
    add_112: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_114, arg253_1);  mul_114 = arg253_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_347: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_112, [2048, 768])
    permute_184: "f32[768, 3072]" = torch.ops.aten.permute.default(arg254_1, [1, 0]);  arg254_1 = None
    addmm_94: "f32[2048, 3072]" = torch.ops.aten.addmm.default(arg255_1, view_347, permute_184);  arg255_1 = view_347 = permute_184 = None
    view_348: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_94, [4, 512, 3072]);  addmm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_115: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_348, 0.5)
    mul_116: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_348, 0.7071067811865476);  view_348 = None
    erf_11: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_116);  mul_116 = None
    add_113: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_117: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_115, add_113);  mul_115 = add_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_349: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_117, [2048, 3072]);  mul_117 = None
    permute_185: "f32[3072, 768]" = torch.ops.aten.permute.default(arg256_1, [1, 0]);  arg256_1 = None
    addmm_95: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg257_1, view_349, permute_185);  arg257_1 = view_349 = permute_185 = None
    view_350: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_95, [4, 512, 768]);  addmm_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:467, code: hidden_states = residual + hidden_states
    add_114: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_112, view_350);  add_112 = view_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_31 = torch.ops.aten.var_mean.correction(add_114, [2], correction = 0, keepdim = True)
    getitem_62: "f32[4, 512, 1]" = var_mean_31[0]
    getitem_63: "f32[4, 512, 1]" = var_mean_31[1];  var_mean_31 = None
    sub_49: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_114, getitem_63);  add_114 = getitem_63 = None
    add_115: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
    rsqrt_31: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
    mul_118: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_31);  sub_49 = rsqrt_31 = None
    mul_119: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_118, arg258_1);  mul_118 = arg258_1 = None
    add_116: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_119, arg259_1);  mul_119 = arg259_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:1406, code: lm_logits = self.lm_head(outputs[0])
    view_351: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_116, [2048, 768]);  add_116 = None
    permute_186: "f32[768, 50265]" = torch.ops.aten.permute.default(arg260_1, [1, 0]);  arg260_1 = None
    mm: "f32[2048, 50265]" = torch.ops.aten.mm.default(view_351, permute_186);  view_351 = permute_186 = None
    view_352: "f32[4, 512, 50265]" = torch.ops.aten.reshape.default(mm, [4, 512, 50265]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:1407, code: lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)
    add_117: "f32[4, 512, 50265]" = torch.ops.aten.add.Tensor(view_352, arg261_1);  view_352 = arg261_1 = None
    return (add_117, clone_50, clone_51, clone_56, clone_57, clone_64, clone_65, clone_70, clone_71, clone_78, clone_79, clone_84, clone_85, clone_92, clone_93, clone_98, clone_99, clone_106, clone_107, clone_112, clone_113, clone_120, clone_121, clone_126, clone_127, add_45)
    