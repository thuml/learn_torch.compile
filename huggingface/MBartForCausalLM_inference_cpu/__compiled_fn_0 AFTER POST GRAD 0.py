from __future__ import annotations



def forward(self, arg0_1: "f32[1026, 1024]", arg1_1: "f32[50265, 1024]", arg2_1: "f32[1024]", arg3_1: "f32[1024]", arg4_1: "f32[1024]", arg5_1: "f32[1024]", arg6_1: "f32[1024, 1024]", arg7_1: "f32[1024]", arg8_1: "f32[1024, 1024]", arg9_1: "f32[1024]", arg10_1: "f32[1024, 1024]", arg11_1: "f32[1024]", arg12_1: "f32[1024, 1024]", arg13_1: "f32[1024]", arg14_1: "f32[1024]", arg15_1: "f32[1024]", arg16_1: "f32[4096, 1024]", arg17_1: "f32[4096]", arg18_1: "f32[1024, 4096]", arg19_1: "f32[1024]", arg20_1: "f32[1024]", arg21_1: "f32[1024]", arg22_1: "f32[1024, 1024]", arg23_1: "f32[1024]", arg24_1: "f32[1024, 1024]", arg25_1: "f32[1024]", arg26_1: "f32[1024, 1024]", arg27_1: "f32[1024]", arg28_1: "f32[1024, 1024]", arg29_1: "f32[1024]", arg30_1: "f32[1024]", arg31_1: "f32[1024]", arg32_1: "f32[4096, 1024]", arg33_1: "f32[4096]", arg34_1: "f32[1024, 4096]", arg35_1: "f32[1024]", arg36_1: "f32[1024]", arg37_1: "f32[1024]", arg38_1: "f32[1024, 1024]", arg39_1: "f32[1024]", arg40_1: "f32[1024, 1024]", arg41_1: "f32[1024]", arg42_1: "f32[1024, 1024]", arg43_1: "f32[1024]", arg44_1: "f32[1024, 1024]", arg45_1: "f32[1024]", arg46_1: "f32[1024]", arg47_1: "f32[1024]", arg48_1: "f32[4096, 1024]", arg49_1: "f32[4096]", arg50_1: "f32[1024, 4096]", arg51_1: "f32[1024]", arg52_1: "f32[1024]", arg53_1: "f32[1024]", arg54_1: "f32[1024, 1024]", arg55_1: "f32[1024]", arg56_1: "f32[1024, 1024]", arg57_1: "f32[1024]", arg58_1: "f32[1024, 1024]", arg59_1: "f32[1024]", arg60_1: "f32[1024, 1024]", arg61_1: "f32[1024]", arg62_1: "f32[1024]", arg63_1: "f32[1024]", arg64_1: "f32[4096, 1024]", arg65_1: "f32[4096]", arg66_1: "f32[1024, 4096]", arg67_1: "f32[1024]", arg68_1: "f32[1024]", arg69_1: "f32[1024]", arg70_1: "f32[1024, 1024]", arg71_1: "f32[1024]", arg72_1: "f32[1024, 1024]", arg73_1: "f32[1024]", arg74_1: "f32[1024, 1024]", arg75_1: "f32[1024]", arg76_1: "f32[1024, 1024]", arg77_1: "f32[1024]", arg78_1: "f32[1024]", arg79_1: "f32[1024]", arg80_1: "f32[4096, 1024]", arg81_1: "f32[4096]", arg82_1: "f32[1024, 4096]", arg83_1: "f32[1024]", arg84_1: "f32[1024]", arg85_1: "f32[1024]", arg86_1: "f32[1024, 1024]", arg87_1: "f32[1024]", arg88_1: "f32[1024, 1024]", arg89_1: "f32[1024]", arg90_1: "f32[1024, 1024]", arg91_1: "f32[1024]", arg92_1: "f32[1024, 1024]", arg93_1: "f32[1024]", arg94_1: "f32[1024]", arg95_1: "f32[1024]", arg96_1: "f32[4096, 1024]", arg97_1: "f32[4096]", arg98_1: "f32[1024, 4096]", arg99_1: "f32[1024]", arg100_1: "f32[1024]", arg101_1: "f32[1024]", arg102_1: "f32[1024, 1024]", arg103_1: "f32[1024]", arg104_1: "f32[1024, 1024]", arg105_1: "f32[1024]", arg106_1: "f32[1024, 1024]", arg107_1: "f32[1024]", arg108_1: "f32[1024, 1024]", arg109_1: "f32[1024]", arg110_1: "f32[1024]", arg111_1: "f32[1024]", arg112_1: "f32[4096, 1024]", arg113_1: "f32[4096]", arg114_1: "f32[1024, 4096]", arg115_1: "f32[1024]", arg116_1: "f32[1024]", arg117_1: "f32[1024]", arg118_1: "f32[1024, 1024]", arg119_1: "f32[1024]", arg120_1: "f32[1024, 1024]", arg121_1: "f32[1024]", arg122_1: "f32[1024, 1024]", arg123_1: "f32[1024]", arg124_1: "f32[1024, 1024]", arg125_1: "f32[1024]", arg126_1: "f32[1024]", arg127_1: "f32[1024]", arg128_1: "f32[4096, 1024]", arg129_1: "f32[4096]", arg130_1: "f32[1024, 4096]", arg131_1: "f32[1024]", arg132_1: "f32[1024]", arg133_1: "f32[1024]", arg134_1: "f32[1024, 1024]", arg135_1: "f32[1024]", arg136_1: "f32[1024, 1024]", arg137_1: "f32[1024]", arg138_1: "f32[1024, 1024]", arg139_1: "f32[1024]", arg140_1: "f32[1024, 1024]", arg141_1: "f32[1024]", arg142_1: "f32[1024]", arg143_1: "f32[1024]", arg144_1: "f32[4096, 1024]", arg145_1: "f32[4096]", arg146_1: "f32[1024, 4096]", arg147_1: "f32[1024]", arg148_1: "f32[1024]", arg149_1: "f32[1024]", arg150_1: "f32[1024, 1024]", arg151_1: "f32[1024]", arg152_1: "f32[1024, 1024]", arg153_1: "f32[1024]", arg154_1: "f32[1024, 1024]", arg155_1: "f32[1024]", arg156_1: "f32[1024, 1024]", arg157_1: "f32[1024]", arg158_1: "f32[1024]", arg159_1: "f32[1024]", arg160_1: "f32[4096, 1024]", arg161_1: "f32[4096]", arg162_1: "f32[1024, 4096]", arg163_1: "f32[1024]", arg164_1: "f32[1024]", arg165_1: "f32[1024]", arg166_1: "f32[1024, 1024]", arg167_1: "f32[1024]", arg168_1: "f32[1024, 1024]", arg169_1: "f32[1024]", arg170_1: "f32[1024, 1024]", arg171_1: "f32[1024]", arg172_1: "f32[1024, 1024]", arg173_1: "f32[1024]", arg174_1: "f32[1024]", arg175_1: "f32[1024]", arg176_1: "f32[4096, 1024]", arg177_1: "f32[4096]", arg178_1: "f32[1024, 4096]", arg179_1: "f32[1024]", arg180_1: "f32[1024]", arg181_1: "f32[1024]", arg182_1: "f32[1024, 1024]", arg183_1: "f32[1024]", arg184_1: "f32[1024, 1024]", arg185_1: "f32[1024]", arg186_1: "f32[1024, 1024]", arg187_1: "f32[1024]", arg188_1: "f32[1024, 1024]", arg189_1: "f32[1024]", arg190_1: "f32[1024]", arg191_1: "f32[1024]", arg192_1: "f32[4096, 1024]", arg193_1: "f32[4096]", arg194_1: "f32[1024, 4096]", arg195_1: "f32[1024]", arg196_1: "f32[1024]", arg197_1: "f32[1024]", arg198_1: "f32[50265, 1024]", arg199_1: "i64[1, 1024]", arg200_1: "i64[1, 1024]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:1026, code: input_ids = input_ids.view(-1, input_shape[-1])
    view: "i64[1, 1024]" = torch.ops.aten.reshape.default(arg199_1, [-1, 1024]);  arg199_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:1037, code: inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
    embedding: "f32[1, 1024, 1024]" = torch.ops.aten.embedding.default(arg1_1, view, 1);  arg1_1 = view = None
    mul: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(embedding, 1.0);  embedding = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:130, code: positions = torch.arange(
    iota_1: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:132, code: ).expand(bsz, -1)
    expand_1: "i64[1, 1024]" = torch.ops.aten.expand.default(iota_1, [1, -1]);  iota_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:134, code: return super().forward(positions + self.offset)
    add_1: "i64[1, 1024]" = torch.ops.aten.add.Tensor(expand_1, 2);  expand_1 = None
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/sparse.py:163, code: return F.embedding(
    embedding_1: "f32[1, 1024, 1024]" = torch.ops.aten.embedding.default(arg0_1, add_1);  arg0_1 = add_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:1051, code: hidden_states = inputs_embeds + positions.to(inputs_embeds.device)
    add_2: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul, embedding_1);  mul = embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:1052, code: hidden_states = self.layernorm_embedding(hidden_states)
    var_mean = torch.ops.aten.var_mean.correction(add_2, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 1024, 1]" = var_mean[0]
    getitem_1: "f32[1, 1024, 1]" = var_mean[1];  var_mean = None
    sub: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_2, getitem_1);  add_2 = getitem_1 = None
    add_3: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
    mul_1: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
    mul_2: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_1, arg2_1);  mul_1 = arg2_1 = None
    add_4: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_2, arg3_1);  mul_2 = arg3_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:418, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_1 = torch.ops.aten.var_mean.correction(add_4, [2], correction = 0, keepdim = True)
    getitem_2: "f32[1, 1024, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 1024, 1]" = var_mean_1[1];  var_mean_1 = None
    sub_1: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_4, getitem_3);  getitem_3 = None
    add_5: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
    rsqrt_1: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
    mul_3: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
    mul_4: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_3, arg4_1);  mul_3 = arg4_1 = None
    add_6: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_4, arg5_1);  mul_4 = arg5_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_2: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_6, [1024, 1024])
    permute: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
    addmm: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg7_1, view_2, permute);  arg7_1 = view_2 = permute = None
    view_3: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm, [1, 1024, 1024]);  addmm = None
    mul_5: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_3, 0.125);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_10: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_5, [1, 1024, 16, 64]);  mul_5 = None
    permute_5: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
    clone_3: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_11: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_3, [16, -1, 64]);  clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_4: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_6, [1024, 1024])
    permute_1: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
    addmm_1: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg9_1, view_4, permute_1);  arg9_1 = view_4 = permute_1 = None
    view_5: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_1, [1, 1024, 1024]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_6: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_5, [1, -1, 16, 64]);  view_5 = None
    permute_2: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
    clone_1: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_12: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_1, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_6: "f32[16, 64, 1024]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
    bmm: "f32[16, 1024, 1024]" = torch.ops.aten.bmm.default(view_11, permute_6);  view_11 = permute_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:246, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_14: "f32[1, 16, 1024, 1024]" = torch.ops.aten.reshape.default(bmm, [1, 16, 1024, 1024]);  bmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:90, code: mask_cond = torch.arange(mask.size(-1), device=device)
    iota: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:91, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    add: "i64[1024]" = torch.ops.aten.add.Tensor(iota, 1)
    view_1: "i64[1024, 1]" = torch.ops.aten.reshape.default(add, [1024, 1]);  add = None
    lt: "b8[1024, 1024]" = torch.ops.aten.lt.Tensor(iota, view_1);  iota = view_1 = None
    full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:89, code: mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    full_default: "f32[1024, 1024]" = torch.ops.aten.full.default([1024, 1024], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:91, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    where: "f32[1024, 1024]" = torch.ops.aten.where.self(lt, full_default_1, full_default);  lt = full_default_1 = full_default = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:246, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    unsqueeze_2: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(where, 0);  where = None
    unsqueeze_3: "f32[1, 1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 1);  unsqueeze_2 = None
    expand_2: "f32[1, 1, 1024, 1024]" = torch.ops.aten.expand.default(unsqueeze_3, [1, 1, 1024, 1024]);  unsqueeze_3 = None
    add_7: "f32[1, 16, 1024, 1024]" = torch.ops.aten.add.Tensor(view_14, expand_2);  view_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:247, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_15: "f32[16, 1024, 1024]" = torch.ops.aten.reshape.default(add_7, [16, 1024, 1024]);  add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax: "f32[16, 1024, 1]" = torch.ops.aten.amax.default(view_15, [-1], True)
    sub_2: "f32[16, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_15, amax);  view_15 = amax = None
    exp: "f32[16, 1024, 1024]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
    sum_1: "f32[16, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[16, 1024, 1024]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_7: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_6, [1024, 1024]);  add_6 = None
    permute_3: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
    addmm_2: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg11_1, view_7, permute_3);  arg11_1 = view_7 = permute_3 = None
    view_8: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_2, [1, 1024, 1024]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_9: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_8, [1, -1, 16, 64]);  view_8 = None
    permute_4: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
    clone_2: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_13: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_2, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_1: "f32[16, 1024, 64]" = torch.ops.aten.bmm.default(div, view_13);  div = view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_16: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(bmm_1, [1, 16, 1024, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_7: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_5: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_17: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_5, [1, 1024, 1024]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_18: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_17, [1024, 1024]);  view_17 = None
    permute_8: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
    addmm_3: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg13_1, view_18, permute_8);  arg13_1 = view_18 = permute_8 = None
    view_19: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_3, [1, 1024, 1024]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:432, code: hidden_states = residual + hidden_states
    add_8: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_4, view_19);  add_4 = view_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:459, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
    getitem_4: "f32[1, 1024, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 1024, 1]" = var_mean_2[1];  var_mean_2 = None
    sub_3: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_8, getitem_5);  getitem_5 = None
    add_9: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
    rsqrt_2: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
    mul_6: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
    mul_7: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_6, arg14_1);  mul_6 = arg14_1 = None
    add_10: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_7, arg15_1);  mul_7 = arg15_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_20: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_10, [1024, 1024]);  add_10 = None
    permute_9: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
    addmm_4: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg17_1, view_20, permute_9);  arg17_1 = view_20 = permute_9 = None
    view_21: "f32[1, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_4, [1, 1024, 4096]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_8: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_21, 0.5)
    mul_9: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_21, 0.7071067811865476);  view_21 = None
    erf: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_9);  mul_9 = None
    add_11: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_10: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_8, add_11);  mul_8 = add_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_22: "f32[1024, 4096]" = torch.ops.aten.reshape.default(mul_10, [1024, 4096]);  mul_10 = None
    permute_10: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
    addmm_5: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg19_1, view_22, permute_10);  arg19_1 = view_22 = permute_10 = None
    view_23: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_5, [1, 1024, 1024]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:464, code: hidden_states = residual + hidden_states
    add_12: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_8, view_23);  add_8 = view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:418, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_3 = torch.ops.aten.var_mean.correction(add_12, [2], correction = 0, keepdim = True)
    getitem_6: "f32[1, 1024, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 1024, 1]" = var_mean_3[1];  var_mean_3 = None
    sub_4: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_12, getitem_7);  getitem_7 = None
    add_13: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
    rsqrt_3: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
    mul_11: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = rsqrt_3 = None
    mul_12: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_11, arg20_1);  mul_11 = arg20_1 = None
    add_14: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_12, arg21_1);  mul_12 = arg21_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_24: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_14, [1024, 1024])
    permute_11: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
    addmm_6: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg23_1, view_24, permute_11);  arg23_1 = view_24 = permute_11 = None
    view_25: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_6, [1, 1024, 1024]);  addmm_6 = None
    mul_13: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_25, 0.125);  view_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_32: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_13, [1, 1024, 16, 64]);  mul_13 = None
    permute_16: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None
    clone_11: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_33: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_11, [16, -1, 64]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_26: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_14, [1024, 1024])
    permute_12: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
    addmm_7: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg25_1, view_26, permute_12);  arg25_1 = view_26 = permute_12 = None
    view_27: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_7, [1, 1024, 1024]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_28: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_27, [1, -1, 16, 64]);  view_27 = None
    permute_13: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
    clone_9: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_34: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_9, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_17: "f32[16, 64, 1024]" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
    bmm_2: "f32[16, 1024, 1024]" = torch.ops.aten.bmm.default(view_33, permute_17);  view_33 = permute_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:246, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_36: "f32[1, 16, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_2, [1, 16, 1024, 1024]);  bmm_2 = None
    add_15: "f32[1, 16, 1024, 1024]" = torch.ops.aten.add.Tensor(view_36, expand_2);  view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:247, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_37: "f32[16, 1024, 1024]" = torch.ops.aten.reshape.default(add_15, [16, 1024, 1024]);  add_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_1: "f32[16, 1024, 1]" = torch.ops.aten.amax.default(view_37, [-1], True)
    sub_5: "f32[16, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_37, amax_1);  view_37 = amax_1 = None
    exp_1: "f32[16, 1024, 1024]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
    sum_2: "f32[16, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_1: "f32[16, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_29: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_14, [1024, 1024]);  add_14 = None
    permute_14: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
    addmm_8: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg27_1, view_29, permute_14);  arg27_1 = view_29 = permute_14 = None
    view_30: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_8, [1, 1024, 1024]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_31: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_30, [1, -1, 16, 64]);  view_30 = None
    permute_15: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None
    clone_10: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_35: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_10, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_3: "f32[16, 1024, 64]" = torch.ops.aten.bmm.default(div_1, view_35);  div_1 = view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_38: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(bmm_3, [1, 16, 1024, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_18: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_13: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
    view_39: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_13, [1, 1024, 1024]);  clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_40: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_39, [1024, 1024]);  view_39 = None
    permute_19: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
    addmm_9: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg29_1, view_40, permute_19);  arg29_1 = view_40 = permute_19 = None
    view_41: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_9, [1, 1024, 1024]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:432, code: hidden_states = residual + hidden_states
    add_16: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_12, view_41);  add_12 = view_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:459, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_4 = torch.ops.aten.var_mean.correction(add_16, [2], correction = 0, keepdim = True)
    getitem_8: "f32[1, 1024, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 1024, 1]" = var_mean_4[1];  var_mean_4 = None
    sub_6: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_16, getitem_9);  getitem_9 = None
    add_17: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
    rsqrt_4: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
    mul_14: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
    mul_15: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_14, arg30_1);  mul_14 = arg30_1 = None
    add_18: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_15, arg31_1);  mul_15 = arg31_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_42: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_18, [1024, 1024]);  add_18 = None
    permute_20: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
    addmm_10: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg33_1, view_42, permute_20);  arg33_1 = view_42 = permute_20 = None
    view_43: "f32[1, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_10, [1, 1024, 4096]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_16: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_43, 0.5)
    mul_17: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_43, 0.7071067811865476);  view_43 = None
    erf_1: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_17);  mul_17 = None
    add_19: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_18: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_16, add_19);  mul_16 = add_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_44: "f32[1024, 4096]" = torch.ops.aten.reshape.default(mul_18, [1024, 4096]);  mul_18 = None
    permute_21: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
    addmm_11: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg35_1, view_44, permute_21);  arg35_1 = view_44 = permute_21 = None
    view_45: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_11, [1, 1024, 1024]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:464, code: hidden_states = residual + hidden_states
    add_20: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_16, view_45);  add_16 = view_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:418, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_5 = torch.ops.aten.var_mean.correction(add_20, [2], correction = 0, keepdim = True)
    getitem_10: "f32[1, 1024, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 1024, 1]" = var_mean_5[1];  var_mean_5 = None
    sub_7: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_20, getitem_11);  getitem_11 = None
    add_21: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
    rsqrt_5: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    mul_19: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_5);  sub_7 = rsqrt_5 = None
    mul_20: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_19, arg36_1);  mul_19 = arg36_1 = None
    add_22: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_20, arg37_1);  mul_20 = arg37_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_46: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_22, [1024, 1024])
    permute_22: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
    addmm_12: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg39_1, view_46, permute_22);  arg39_1 = view_46 = permute_22 = None
    view_47: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_12, [1, 1024, 1024]);  addmm_12 = None
    mul_21: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_47, 0.125);  view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_54: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_21, [1, 1024, 16, 64]);  mul_21 = None
    permute_27: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
    clone_19: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_55: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_19, [16, -1, 64]);  clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_48: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_22, [1024, 1024])
    permute_23: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
    addmm_13: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg41_1, view_48, permute_23);  arg41_1 = view_48 = permute_23 = None
    view_49: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_13, [1, 1024, 1024]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_50: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_49, [1, -1, 16, 64]);  view_49 = None
    permute_24: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
    clone_17: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_56: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_17, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_28: "f32[16, 64, 1024]" = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None
    bmm_4: "f32[16, 1024, 1024]" = torch.ops.aten.bmm.default(view_55, permute_28);  view_55 = permute_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:246, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_58: "f32[1, 16, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_4, [1, 16, 1024, 1024]);  bmm_4 = None
    add_23: "f32[1, 16, 1024, 1024]" = torch.ops.aten.add.Tensor(view_58, expand_2);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:247, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_59: "f32[16, 1024, 1024]" = torch.ops.aten.reshape.default(add_23, [16, 1024, 1024]);  add_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_2: "f32[16, 1024, 1]" = torch.ops.aten.amax.default(view_59, [-1], True)
    sub_8: "f32[16, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_59, amax_2);  view_59 = amax_2 = None
    exp_2: "f32[16, 1024, 1024]" = torch.ops.aten.exp.default(sub_8);  sub_8 = None
    sum_3: "f32[16, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_2: "f32[16, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_51: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_22, [1024, 1024]);  add_22 = None
    permute_25: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
    addmm_14: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg43_1, view_51, permute_25);  arg43_1 = view_51 = permute_25 = None
    view_52: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_14, [1, 1024, 1024]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_53: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_52, [1, -1, 16, 64]);  view_52 = None
    permute_26: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
    clone_18: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_57: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_18, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_5: "f32[16, 1024, 64]" = torch.ops.aten.bmm.default(div_2, view_57);  div_2 = view_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_60: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(bmm_5, [1, 16, 1024, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_29: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_60, [0, 2, 1, 3]);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_21: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    view_61: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_21, [1, 1024, 1024]);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_62: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_61, [1024, 1024]);  view_61 = None
    permute_30: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
    addmm_15: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg45_1, view_62, permute_30);  arg45_1 = view_62 = permute_30 = None
    view_63: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_15, [1, 1024, 1024]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:432, code: hidden_states = residual + hidden_states
    add_24: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_20, view_63);  add_20 = view_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:459, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_6 = torch.ops.aten.var_mean.correction(add_24, [2], correction = 0, keepdim = True)
    getitem_12: "f32[1, 1024, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 1024, 1]" = var_mean_6[1];  var_mean_6 = None
    sub_9: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_24, getitem_13);  getitem_13 = None
    add_25: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
    rsqrt_6: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
    mul_22: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = rsqrt_6 = None
    mul_23: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_22, arg46_1);  mul_22 = arg46_1 = None
    add_26: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_23, arg47_1);  mul_23 = arg47_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_64: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_26, [1024, 1024]);  add_26 = None
    permute_31: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
    addmm_16: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg49_1, view_64, permute_31);  arg49_1 = view_64 = permute_31 = None
    view_65: "f32[1, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_16, [1, 1024, 4096]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_24: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_65, 0.5)
    mul_25: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_65, 0.7071067811865476);  view_65 = None
    erf_2: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_25);  mul_25 = None
    add_27: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_26: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_24, add_27);  mul_24 = add_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_66: "f32[1024, 4096]" = torch.ops.aten.reshape.default(mul_26, [1024, 4096]);  mul_26 = None
    permute_32: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
    addmm_17: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg51_1, view_66, permute_32);  arg51_1 = view_66 = permute_32 = None
    view_67: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_17, [1, 1024, 1024]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:464, code: hidden_states = residual + hidden_states
    add_28: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_24, view_67);  add_24 = view_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:418, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_7 = torch.ops.aten.var_mean.correction(add_28, [2], correction = 0, keepdim = True)
    getitem_14: "f32[1, 1024, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 1024, 1]" = var_mean_7[1];  var_mean_7 = None
    sub_10: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_28, getitem_15);  getitem_15 = None
    add_29: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
    rsqrt_7: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
    mul_27: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_7);  sub_10 = rsqrt_7 = None
    mul_28: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_27, arg52_1);  mul_27 = arg52_1 = None
    add_30: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_28, arg53_1);  mul_28 = arg53_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_68: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_30, [1024, 1024])
    permute_33: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
    addmm_18: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg55_1, view_68, permute_33);  arg55_1 = view_68 = permute_33 = None
    view_69: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_18, [1, 1024, 1024]);  addmm_18 = None
    mul_29: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_69, 0.125);  view_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_76: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_29, [1, 1024, 16, 64]);  mul_29 = None
    permute_38: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
    clone_27: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_77: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_27, [16, -1, 64]);  clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_70: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_30, [1024, 1024])
    permute_34: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
    addmm_19: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg57_1, view_70, permute_34);  arg57_1 = view_70 = permute_34 = None
    view_71: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_19, [1, 1024, 1024]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_72: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_71, [1, -1, 16, 64]);  view_71 = None
    permute_35: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
    clone_25: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_78: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_25, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_39: "f32[16, 64, 1024]" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
    bmm_6: "f32[16, 1024, 1024]" = torch.ops.aten.bmm.default(view_77, permute_39);  view_77 = permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:246, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_80: "f32[1, 16, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_6, [1, 16, 1024, 1024]);  bmm_6 = None
    add_31: "f32[1, 16, 1024, 1024]" = torch.ops.aten.add.Tensor(view_80, expand_2);  view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:247, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_81: "f32[16, 1024, 1024]" = torch.ops.aten.reshape.default(add_31, [16, 1024, 1024]);  add_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_3: "f32[16, 1024, 1]" = torch.ops.aten.amax.default(view_81, [-1], True)
    sub_11: "f32[16, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_81, amax_3);  view_81 = amax_3 = None
    exp_3: "f32[16, 1024, 1024]" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
    sum_4: "f32[16, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_3: "f32[16, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_73: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_30, [1024, 1024]);  add_30 = None
    permute_36: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
    addmm_20: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg59_1, view_73, permute_36);  arg59_1 = view_73 = permute_36 = None
    view_74: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_20, [1, 1024, 1024]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_75: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_74, [1, -1, 16, 64]);  view_74 = None
    permute_37: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
    clone_26: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_79: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_26, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_7: "f32[16, 1024, 64]" = torch.ops.aten.bmm.default(div_3, view_79);  div_3 = view_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_82: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(bmm_7, [1, 16, 1024, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_40: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_29: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
    view_83: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_29, [1, 1024, 1024]);  clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_84: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_83, [1024, 1024]);  view_83 = None
    permute_41: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
    addmm_21: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg61_1, view_84, permute_41);  arg61_1 = view_84 = permute_41 = None
    view_85: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_21, [1, 1024, 1024]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:432, code: hidden_states = residual + hidden_states
    add_32: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_28, view_85);  add_28 = view_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:459, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_8 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
    getitem_16: "f32[1, 1024, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 1024, 1]" = var_mean_8[1];  var_mean_8 = None
    sub_12: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_32, getitem_17);  getitem_17 = None
    add_33: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
    rsqrt_8: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
    mul_30: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = rsqrt_8 = None
    mul_31: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_30, arg62_1);  mul_30 = arg62_1 = None
    add_34: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_31, arg63_1);  mul_31 = arg63_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_86: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_34, [1024, 1024]);  add_34 = None
    permute_42: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
    addmm_22: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg65_1, view_86, permute_42);  arg65_1 = view_86 = permute_42 = None
    view_87: "f32[1, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_22, [1, 1024, 4096]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_32: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_87, 0.5)
    mul_33: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_87, 0.7071067811865476);  view_87 = None
    erf_3: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
    add_35: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_34: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_32, add_35);  mul_32 = add_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_88: "f32[1024, 4096]" = torch.ops.aten.reshape.default(mul_34, [1024, 4096]);  mul_34 = None
    permute_43: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
    addmm_23: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg67_1, view_88, permute_43);  arg67_1 = view_88 = permute_43 = None
    view_89: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_23, [1, 1024, 1024]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:464, code: hidden_states = residual + hidden_states
    add_36: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_32, view_89);  add_32 = view_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:418, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_9 = torch.ops.aten.var_mean.correction(add_36, [2], correction = 0, keepdim = True)
    getitem_18: "f32[1, 1024, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 1024, 1]" = var_mean_9[1];  var_mean_9 = None
    sub_13: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_36, getitem_19);  getitem_19 = None
    add_37: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
    rsqrt_9: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
    mul_35: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_9);  sub_13 = rsqrt_9 = None
    mul_36: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_35, arg68_1);  mul_35 = arg68_1 = None
    add_38: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_36, arg69_1);  mul_36 = arg69_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_90: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_38, [1024, 1024])
    permute_44: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
    addmm_24: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg71_1, view_90, permute_44);  arg71_1 = view_90 = permute_44 = None
    view_91: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_24, [1, 1024, 1024]);  addmm_24 = None
    mul_37: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_91, 0.125);  view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_98: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_37, [1, 1024, 16, 64]);  mul_37 = None
    permute_49: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None
    clone_35: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_99: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_35, [16, -1, 64]);  clone_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_92: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_38, [1024, 1024])
    permute_45: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
    addmm_25: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg73_1, view_92, permute_45);  arg73_1 = view_92 = permute_45 = None
    view_93: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_25, [1, 1024, 1024]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_94: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_93, [1, -1, 16, 64]);  view_93 = None
    permute_46: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
    clone_33: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_100: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_33, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_50: "f32[16, 64, 1024]" = torch.ops.aten.permute.default(view_100, [0, 2, 1]);  view_100 = None
    bmm_8: "f32[16, 1024, 1024]" = torch.ops.aten.bmm.default(view_99, permute_50);  view_99 = permute_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:246, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_102: "f32[1, 16, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_8, [1, 16, 1024, 1024]);  bmm_8 = None
    add_39: "f32[1, 16, 1024, 1024]" = torch.ops.aten.add.Tensor(view_102, expand_2);  view_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:247, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_103: "f32[16, 1024, 1024]" = torch.ops.aten.reshape.default(add_39, [16, 1024, 1024]);  add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_4: "f32[16, 1024, 1]" = torch.ops.aten.amax.default(view_103, [-1], True)
    sub_14: "f32[16, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_103, amax_4);  view_103 = amax_4 = None
    exp_4: "f32[16, 1024, 1024]" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
    sum_5: "f32[16, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_4: "f32[16, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_95: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_38, [1024, 1024]);  add_38 = None
    permute_47: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
    addmm_26: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg75_1, view_95, permute_47);  arg75_1 = view_95 = permute_47 = None
    view_96: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_26, [1, 1024, 1024]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_97: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_96, [1, -1, 16, 64]);  view_96 = None
    permute_48: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_97, [0, 2, 1, 3]);  view_97 = None
    clone_34: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_101: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_34, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_9: "f32[16, 1024, 64]" = torch.ops.aten.bmm.default(div_4, view_101);  div_4 = view_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_104: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(bmm_9, [1, 16, 1024, 64]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_51: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_37: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
    view_105: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_37, [1, 1024, 1024]);  clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_106: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_105, [1024, 1024]);  view_105 = None
    permute_52: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
    addmm_27: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg77_1, view_106, permute_52);  arg77_1 = view_106 = permute_52 = None
    view_107: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_27, [1, 1024, 1024]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:432, code: hidden_states = residual + hidden_states
    add_40: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_36, view_107);  add_36 = view_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:459, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_10 = torch.ops.aten.var_mean.correction(add_40, [2], correction = 0, keepdim = True)
    getitem_20: "f32[1, 1024, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 1024, 1]" = var_mean_10[1];  var_mean_10 = None
    sub_15: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_40, getitem_21);  getitem_21 = None
    add_41: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
    rsqrt_10: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
    mul_38: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = rsqrt_10 = None
    mul_39: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_38, arg78_1);  mul_38 = arg78_1 = None
    add_42: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_39, arg79_1);  mul_39 = arg79_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_108: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_42, [1024, 1024]);  add_42 = None
    permute_53: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
    addmm_28: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg81_1, view_108, permute_53);  arg81_1 = view_108 = permute_53 = None
    view_109: "f32[1, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_28, [1, 1024, 4096]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_40: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_109, 0.5)
    mul_41: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_109, 0.7071067811865476);  view_109 = None
    erf_4: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_41);  mul_41 = None
    add_43: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_42: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_40, add_43);  mul_40 = add_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_110: "f32[1024, 4096]" = torch.ops.aten.reshape.default(mul_42, [1024, 4096]);  mul_42 = None
    permute_54: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
    addmm_29: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg83_1, view_110, permute_54);  arg83_1 = view_110 = permute_54 = None
    view_111: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_29, [1, 1024, 1024]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:464, code: hidden_states = residual + hidden_states
    add_44: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_40, view_111);  add_40 = view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:418, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_11 = torch.ops.aten.var_mean.correction(add_44, [2], correction = 0, keepdim = True)
    getitem_22: "f32[1, 1024, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 1024, 1]" = var_mean_11[1];  var_mean_11 = None
    sub_16: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_44, getitem_23);  getitem_23 = None
    add_45: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
    rsqrt_11: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
    mul_43: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_11);  sub_16 = rsqrt_11 = None
    mul_44: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_43, arg84_1);  mul_43 = arg84_1 = None
    add_46: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_44, arg85_1);  mul_44 = arg85_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_112: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_46, [1024, 1024])
    permute_55: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
    addmm_30: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg87_1, view_112, permute_55);  arg87_1 = view_112 = permute_55 = None
    view_113: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_30, [1, 1024, 1024]);  addmm_30 = None
    mul_45: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_113, 0.125);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_120: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_45, [1, 1024, 16, 64]);  mul_45 = None
    permute_60: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
    clone_43: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_121: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_43, [16, -1, 64]);  clone_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_114: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_46, [1024, 1024])
    permute_56: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
    addmm_31: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg89_1, view_114, permute_56);  arg89_1 = view_114 = permute_56 = None
    view_115: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_31, [1, 1024, 1024]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_116: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_115, [1, -1, 16, 64]);  view_115 = None
    permute_57: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_116, [0, 2, 1, 3]);  view_116 = None
    clone_41: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_57, memory_format = torch.contiguous_format);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_122: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_41, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_61: "f32[16, 64, 1024]" = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None
    bmm_10: "f32[16, 1024, 1024]" = torch.ops.aten.bmm.default(view_121, permute_61);  view_121 = permute_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:246, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_124: "f32[1, 16, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_10, [1, 16, 1024, 1024]);  bmm_10 = None
    add_47: "f32[1, 16, 1024, 1024]" = torch.ops.aten.add.Tensor(view_124, expand_2);  view_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:247, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_125: "f32[16, 1024, 1024]" = torch.ops.aten.reshape.default(add_47, [16, 1024, 1024]);  add_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_5: "f32[16, 1024, 1]" = torch.ops.aten.amax.default(view_125, [-1], True)
    sub_17: "f32[16, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_125, amax_5);  view_125 = amax_5 = None
    exp_5: "f32[16, 1024, 1024]" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
    sum_6: "f32[16, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_5: "f32[16, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_117: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_46, [1024, 1024]);  add_46 = None
    permute_58: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
    addmm_32: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg91_1, view_117, permute_58);  arg91_1 = view_117 = permute_58 = None
    view_118: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_32, [1, 1024, 1024]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_119: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_118, [1, -1, 16, 64]);  view_118 = None
    permute_59: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_119, [0, 2, 1, 3]);  view_119 = None
    clone_42: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_123: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_42, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_11: "f32[16, 1024, 64]" = torch.ops.aten.bmm.default(div_5, view_123);  div_5 = view_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_126: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(bmm_11, [1, 16, 1024, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_62: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_45: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    view_127: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_45, [1, 1024, 1024]);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_128: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_127, [1024, 1024]);  view_127 = None
    permute_63: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
    addmm_33: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg93_1, view_128, permute_63);  arg93_1 = view_128 = permute_63 = None
    view_129: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_33, [1, 1024, 1024]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:432, code: hidden_states = residual + hidden_states
    add_48: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_44, view_129);  add_44 = view_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:459, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_12 = torch.ops.aten.var_mean.correction(add_48, [2], correction = 0, keepdim = True)
    getitem_24: "f32[1, 1024, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 1024, 1]" = var_mean_12[1];  var_mean_12 = None
    sub_18: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_48, getitem_25);  getitem_25 = None
    add_49: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
    rsqrt_12: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    mul_46: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = rsqrt_12 = None
    mul_47: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_46, arg94_1);  mul_46 = arg94_1 = None
    add_50: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_47, arg95_1);  mul_47 = arg95_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_130: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_50, [1024, 1024]);  add_50 = None
    permute_64: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
    addmm_34: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg97_1, view_130, permute_64);  arg97_1 = view_130 = permute_64 = None
    view_131: "f32[1, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_34, [1, 1024, 4096]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_48: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_131, 0.5)
    mul_49: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_131, 0.7071067811865476);  view_131 = None
    erf_5: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_49);  mul_49 = None
    add_51: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_50: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_48, add_51);  mul_48 = add_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_132: "f32[1024, 4096]" = torch.ops.aten.reshape.default(mul_50, [1024, 4096]);  mul_50 = None
    permute_65: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
    addmm_35: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg99_1, view_132, permute_65);  arg99_1 = view_132 = permute_65 = None
    view_133: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_35, [1, 1024, 1024]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:464, code: hidden_states = residual + hidden_states
    add_52: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_48, view_133);  add_48 = view_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:418, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_13 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
    getitem_26: "f32[1, 1024, 1]" = var_mean_13[0]
    getitem_27: "f32[1, 1024, 1]" = var_mean_13[1];  var_mean_13 = None
    sub_19: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_52, getitem_27);  getitem_27 = None
    add_53: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
    rsqrt_13: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    mul_51: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_13);  sub_19 = rsqrt_13 = None
    mul_52: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_51, arg100_1);  mul_51 = arg100_1 = None
    add_54: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_52, arg101_1);  mul_52 = arg101_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_134: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_54, [1024, 1024])
    permute_66: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
    addmm_36: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg103_1, view_134, permute_66);  arg103_1 = view_134 = permute_66 = None
    view_135: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_36, [1, 1024, 1024]);  addmm_36 = None
    mul_53: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_135, 0.125);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_142: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_53, [1, 1024, 16, 64]);  mul_53 = None
    permute_71: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_142, [0, 2, 1, 3]);  view_142 = None
    clone_51: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_143: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_51, [16, -1, 64]);  clone_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_136: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_54, [1024, 1024])
    permute_67: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
    addmm_37: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg105_1, view_136, permute_67);  arg105_1 = view_136 = permute_67 = None
    view_137: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_37, [1, 1024, 1024]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_138: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_137, [1, -1, 16, 64]);  view_137 = None
    permute_68: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None
    clone_49: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_144: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_49, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_72: "f32[16, 64, 1024]" = torch.ops.aten.permute.default(view_144, [0, 2, 1]);  view_144 = None
    bmm_12: "f32[16, 1024, 1024]" = torch.ops.aten.bmm.default(view_143, permute_72);  view_143 = permute_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:246, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_146: "f32[1, 16, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_12, [1, 16, 1024, 1024]);  bmm_12 = None
    add_55: "f32[1, 16, 1024, 1024]" = torch.ops.aten.add.Tensor(view_146, expand_2);  view_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:247, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_147: "f32[16, 1024, 1024]" = torch.ops.aten.reshape.default(add_55, [16, 1024, 1024]);  add_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_6: "f32[16, 1024, 1]" = torch.ops.aten.amax.default(view_147, [-1], True)
    sub_20: "f32[16, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_147, amax_6);  view_147 = amax_6 = None
    exp_6: "f32[16, 1024, 1024]" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
    sum_7: "f32[16, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_6: "f32[16, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_139: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_54, [1024, 1024]);  add_54 = None
    permute_69: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
    addmm_38: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg107_1, view_139, permute_69);  arg107_1 = view_139 = permute_69 = None
    view_140: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_38, [1, 1024, 1024]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_141: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_140, [1, -1, 16, 64]);  view_140 = None
    permute_70: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_141, [0, 2, 1, 3]);  view_141 = None
    clone_50: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_145: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_50, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_13: "f32[16, 1024, 64]" = torch.ops.aten.bmm.default(div_6, view_145);  div_6 = view_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_148: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(bmm_13, [1, 16, 1024, 64]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_73: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_53: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
    view_149: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_53, [1, 1024, 1024]);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_150: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_149, [1024, 1024]);  view_149 = None
    permute_74: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
    addmm_39: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg109_1, view_150, permute_74);  arg109_1 = view_150 = permute_74 = None
    view_151: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_39, [1, 1024, 1024]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:432, code: hidden_states = residual + hidden_states
    add_56: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_52, view_151);  add_52 = view_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:459, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_14 = torch.ops.aten.var_mean.correction(add_56, [2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 1024, 1]" = var_mean_14[0]
    getitem_29: "f32[1, 1024, 1]" = var_mean_14[1];  var_mean_14 = None
    sub_21: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_56, getitem_29);  getitem_29 = None
    add_57: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
    rsqrt_14: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
    mul_54: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = rsqrt_14 = None
    mul_55: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_54, arg110_1);  mul_54 = arg110_1 = None
    add_58: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_55, arg111_1);  mul_55 = arg111_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_152: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_58, [1024, 1024]);  add_58 = None
    permute_75: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
    addmm_40: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg113_1, view_152, permute_75);  arg113_1 = view_152 = permute_75 = None
    view_153: "f32[1, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_40, [1, 1024, 4096]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_56: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_153, 0.5)
    mul_57: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_153, 0.7071067811865476);  view_153 = None
    erf_6: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_57);  mul_57 = None
    add_59: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_58: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_56, add_59);  mul_56 = add_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_154: "f32[1024, 4096]" = torch.ops.aten.reshape.default(mul_58, [1024, 4096]);  mul_58 = None
    permute_76: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
    addmm_41: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg115_1, view_154, permute_76);  arg115_1 = view_154 = permute_76 = None
    view_155: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_41, [1, 1024, 1024]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:464, code: hidden_states = residual + hidden_states
    add_60: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_56, view_155);  add_56 = view_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:418, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_15 = torch.ops.aten.var_mean.correction(add_60, [2], correction = 0, keepdim = True)
    getitem_30: "f32[1, 1024, 1]" = var_mean_15[0]
    getitem_31: "f32[1, 1024, 1]" = var_mean_15[1];  var_mean_15 = None
    sub_22: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_60, getitem_31);  getitem_31 = None
    add_61: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
    rsqrt_15: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
    mul_59: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_15);  sub_22 = rsqrt_15 = None
    mul_60: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_59, arg116_1);  mul_59 = arg116_1 = None
    add_62: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_60, arg117_1);  mul_60 = arg117_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_156: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_62, [1024, 1024])
    permute_77: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
    addmm_42: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg119_1, view_156, permute_77);  arg119_1 = view_156 = permute_77 = None
    view_157: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_42, [1, 1024, 1024]);  addmm_42 = None
    mul_61: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_157, 0.125);  view_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_164: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_61, [1, 1024, 16, 64]);  mul_61 = None
    permute_82: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_164, [0, 2, 1, 3]);  view_164 = None
    clone_59: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_165: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_59, [16, -1, 64]);  clone_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_158: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_62, [1024, 1024])
    permute_78: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
    addmm_43: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg121_1, view_158, permute_78);  arg121_1 = view_158 = permute_78 = None
    view_159: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_43, [1, 1024, 1024]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_160: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_159, [1, -1, 16, 64]);  view_159 = None
    permute_79: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_160, [0, 2, 1, 3]);  view_160 = None
    clone_57: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_166: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_57, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_83: "f32[16, 64, 1024]" = torch.ops.aten.permute.default(view_166, [0, 2, 1]);  view_166 = None
    bmm_14: "f32[16, 1024, 1024]" = torch.ops.aten.bmm.default(view_165, permute_83);  view_165 = permute_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:246, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_168: "f32[1, 16, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_14, [1, 16, 1024, 1024]);  bmm_14 = None
    add_63: "f32[1, 16, 1024, 1024]" = torch.ops.aten.add.Tensor(view_168, expand_2);  view_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:247, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_169: "f32[16, 1024, 1024]" = torch.ops.aten.reshape.default(add_63, [16, 1024, 1024]);  add_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_7: "f32[16, 1024, 1]" = torch.ops.aten.amax.default(view_169, [-1], True)
    sub_23: "f32[16, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_169, amax_7);  view_169 = amax_7 = None
    exp_7: "f32[16, 1024, 1024]" = torch.ops.aten.exp.default(sub_23);  sub_23 = None
    sum_8: "f32[16, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_7: "f32[16, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_161: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_62, [1024, 1024]);  add_62 = None
    permute_80: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
    addmm_44: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg123_1, view_161, permute_80);  arg123_1 = view_161 = permute_80 = None
    view_162: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_44, [1, 1024, 1024]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_163: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_162, [1, -1, 16, 64]);  view_162 = None
    permute_81: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_163, [0, 2, 1, 3]);  view_163 = None
    clone_58: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_81, memory_format = torch.contiguous_format);  permute_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_167: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_58, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_15: "f32[16, 1024, 64]" = torch.ops.aten.bmm.default(div_7, view_167);  div_7 = view_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_170: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(bmm_15, [1, 16, 1024, 64]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_84: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_170, [0, 2, 1, 3]);  view_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_61: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
    view_171: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_61, [1, 1024, 1024]);  clone_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_172: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_171, [1024, 1024]);  view_171 = None
    permute_85: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
    addmm_45: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg125_1, view_172, permute_85);  arg125_1 = view_172 = permute_85 = None
    view_173: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_45, [1, 1024, 1024]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:432, code: hidden_states = residual + hidden_states
    add_64: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_60, view_173);  add_60 = view_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:459, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_16 = torch.ops.aten.var_mean.correction(add_64, [2], correction = 0, keepdim = True)
    getitem_32: "f32[1, 1024, 1]" = var_mean_16[0]
    getitem_33: "f32[1, 1024, 1]" = var_mean_16[1];  var_mean_16 = None
    sub_24: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_64, getitem_33);  getitem_33 = None
    add_65: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
    rsqrt_16: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
    mul_62: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = rsqrt_16 = None
    mul_63: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_62, arg126_1);  mul_62 = arg126_1 = None
    add_66: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_63, arg127_1);  mul_63 = arg127_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_174: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_66, [1024, 1024]);  add_66 = None
    permute_86: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
    addmm_46: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg129_1, view_174, permute_86);  arg129_1 = view_174 = permute_86 = None
    view_175: "f32[1, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_46, [1, 1024, 4096]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_64: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_175, 0.5)
    mul_65: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_175, 0.7071067811865476);  view_175 = None
    erf_7: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_65);  mul_65 = None
    add_67: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_66: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_64, add_67);  mul_64 = add_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_176: "f32[1024, 4096]" = torch.ops.aten.reshape.default(mul_66, [1024, 4096]);  mul_66 = None
    permute_87: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
    addmm_47: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg131_1, view_176, permute_87);  arg131_1 = view_176 = permute_87 = None
    view_177: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_47, [1, 1024, 1024]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:464, code: hidden_states = residual + hidden_states
    add_68: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_64, view_177);  add_64 = view_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:418, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_17 = torch.ops.aten.var_mean.correction(add_68, [2], correction = 0, keepdim = True)
    getitem_34: "f32[1, 1024, 1]" = var_mean_17[0]
    getitem_35: "f32[1, 1024, 1]" = var_mean_17[1];  var_mean_17 = None
    sub_25: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_68, getitem_35);  getitem_35 = None
    add_69: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
    rsqrt_17: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
    mul_67: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_17);  sub_25 = rsqrt_17 = None
    mul_68: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_67, arg132_1);  mul_67 = arg132_1 = None
    add_70: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_68, arg133_1);  mul_68 = arg133_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_178: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_70, [1024, 1024])
    permute_88: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg134_1, [1, 0]);  arg134_1 = None
    addmm_48: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg135_1, view_178, permute_88);  arg135_1 = view_178 = permute_88 = None
    view_179: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_48, [1, 1024, 1024]);  addmm_48 = None
    mul_69: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_179, 0.125);  view_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_186: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_69, [1, 1024, 16, 64]);  mul_69 = None
    permute_93: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
    clone_67: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_187: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_67, [16, -1, 64]);  clone_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_180: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_70, [1024, 1024])
    permute_89: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
    addmm_49: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg137_1, view_180, permute_89);  arg137_1 = view_180 = permute_89 = None
    view_181: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_49, [1, 1024, 1024]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_182: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_181, [1, -1, 16, 64]);  view_181 = None
    permute_90: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None
    clone_65: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_188: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_65, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_94: "f32[16, 64, 1024]" = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None
    bmm_16: "f32[16, 1024, 1024]" = torch.ops.aten.bmm.default(view_187, permute_94);  view_187 = permute_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:246, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_190: "f32[1, 16, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_16, [1, 16, 1024, 1024]);  bmm_16 = None
    add_71: "f32[1, 16, 1024, 1024]" = torch.ops.aten.add.Tensor(view_190, expand_2);  view_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:247, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_191: "f32[16, 1024, 1024]" = torch.ops.aten.reshape.default(add_71, [16, 1024, 1024]);  add_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_8: "f32[16, 1024, 1]" = torch.ops.aten.amax.default(view_191, [-1], True)
    sub_26: "f32[16, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_191, amax_8);  view_191 = amax_8 = None
    exp_8: "f32[16, 1024, 1024]" = torch.ops.aten.exp.default(sub_26);  sub_26 = None
    sum_9: "f32[16, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_8: "f32[16, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_183: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_70, [1024, 1024]);  add_70 = None
    permute_91: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
    addmm_50: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg139_1, view_183, permute_91);  arg139_1 = view_183 = permute_91 = None
    view_184: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_50, [1, 1024, 1024]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_185: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_184, [1, -1, 16, 64]);  view_184 = None
    permute_92: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None
    clone_66: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_189: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_66, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_17: "f32[16, 1024, 64]" = torch.ops.aten.bmm.default(div_8, view_189);  div_8 = view_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_192: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(bmm_17, [1, 16, 1024, 64]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_95: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_69: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
    view_193: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_69, [1, 1024, 1024]);  clone_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_194: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_193, [1024, 1024]);  view_193 = None
    permute_96: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
    addmm_51: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg141_1, view_194, permute_96);  arg141_1 = view_194 = permute_96 = None
    view_195: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_51, [1, 1024, 1024]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:432, code: hidden_states = residual + hidden_states
    add_72: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_68, view_195);  add_68 = view_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:459, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_18 = torch.ops.aten.var_mean.correction(add_72, [2], correction = 0, keepdim = True)
    getitem_36: "f32[1, 1024, 1]" = var_mean_18[0]
    getitem_37: "f32[1, 1024, 1]" = var_mean_18[1];  var_mean_18 = None
    sub_27: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_72, getitem_37);  getitem_37 = None
    add_73: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
    rsqrt_18: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
    mul_70: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = rsqrt_18 = None
    mul_71: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_70, arg142_1);  mul_70 = arg142_1 = None
    add_74: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_71, arg143_1);  mul_71 = arg143_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_196: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_74, [1024, 1024]);  add_74 = None
    permute_97: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
    addmm_52: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg145_1, view_196, permute_97);  arg145_1 = view_196 = permute_97 = None
    view_197: "f32[1, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_52, [1, 1024, 4096]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_72: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_197, 0.5)
    mul_73: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_197, 0.7071067811865476);  view_197 = None
    erf_8: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_73);  mul_73 = None
    add_75: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_74: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_72, add_75);  mul_72 = add_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_198: "f32[1024, 4096]" = torch.ops.aten.reshape.default(mul_74, [1024, 4096]);  mul_74 = None
    permute_98: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
    addmm_53: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg147_1, view_198, permute_98);  arg147_1 = view_198 = permute_98 = None
    view_199: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_53, [1, 1024, 1024]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:464, code: hidden_states = residual + hidden_states
    add_76: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_72, view_199);  add_72 = view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:418, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_19 = torch.ops.aten.var_mean.correction(add_76, [2], correction = 0, keepdim = True)
    getitem_38: "f32[1, 1024, 1]" = var_mean_19[0]
    getitem_39: "f32[1, 1024, 1]" = var_mean_19[1];  var_mean_19 = None
    sub_28: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_76, getitem_39);  getitem_39 = None
    add_77: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
    rsqrt_19: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
    mul_75: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_19);  sub_28 = rsqrt_19 = None
    mul_76: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_75, arg148_1);  mul_75 = arg148_1 = None
    add_78: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_76, arg149_1);  mul_76 = arg149_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_200: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_78, [1024, 1024])
    permute_99: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
    addmm_54: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg151_1, view_200, permute_99);  arg151_1 = view_200 = permute_99 = None
    view_201: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_54, [1, 1024, 1024]);  addmm_54 = None
    mul_77: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_201, 0.125);  view_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_208: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_77, [1, 1024, 16, 64]);  mul_77 = None
    permute_104: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None
    clone_75: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_209: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_75, [16, -1, 64]);  clone_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_202: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_78, [1024, 1024])
    permute_100: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
    addmm_55: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg153_1, view_202, permute_100);  arg153_1 = view_202 = permute_100 = None
    view_203: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_55, [1, 1024, 1024]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_204: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_203, [1, -1, 16, 64]);  view_203 = None
    permute_101: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_204, [0, 2, 1, 3]);  view_204 = None
    clone_73: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_101, memory_format = torch.contiguous_format);  permute_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_210: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_73, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_105: "f32[16, 64, 1024]" = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None
    bmm_18: "f32[16, 1024, 1024]" = torch.ops.aten.bmm.default(view_209, permute_105);  view_209 = permute_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:246, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_212: "f32[1, 16, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_18, [1, 16, 1024, 1024]);  bmm_18 = None
    add_79: "f32[1, 16, 1024, 1024]" = torch.ops.aten.add.Tensor(view_212, expand_2);  view_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:247, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_213: "f32[16, 1024, 1024]" = torch.ops.aten.reshape.default(add_79, [16, 1024, 1024]);  add_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_9: "f32[16, 1024, 1]" = torch.ops.aten.amax.default(view_213, [-1], True)
    sub_29: "f32[16, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_213, amax_9);  view_213 = amax_9 = None
    exp_9: "f32[16, 1024, 1024]" = torch.ops.aten.exp.default(sub_29);  sub_29 = None
    sum_10: "f32[16, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_9: "f32[16, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_205: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_78, [1024, 1024]);  add_78 = None
    permute_102: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
    addmm_56: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg155_1, view_205, permute_102);  arg155_1 = view_205 = permute_102 = None
    view_206: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_56, [1, 1024, 1024]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_207: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_206, [1, -1, 16, 64]);  view_206 = None
    permute_103: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_207, [0, 2, 1, 3]);  view_207 = None
    clone_74: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_103, memory_format = torch.contiguous_format);  permute_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_211: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_74, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_19: "f32[16, 1024, 64]" = torch.ops.aten.bmm.default(div_9, view_211);  div_9 = view_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_214: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(bmm_19, [1, 16, 1024, 64]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_106: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_214, [0, 2, 1, 3]);  view_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_77: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
    view_215: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_77, [1, 1024, 1024]);  clone_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_216: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_215, [1024, 1024]);  view_215 = None
    permute_107: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
    addmm_57: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg157_1, view_216, permute_107);  arg157_1 = view_216 = permute_107 = None
    view_217: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_57, [1, 1024, 1024]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:432, code: hidden_states = residual + hidden_states
    add_80: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_76, view_217);  add_76 = view_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:459, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_20 = torch.ops.aten.var_mean.correction(add_80, [2], correction = 0, keepdim = True)
    getitem_40: "f32[1, 1024, 1]" = var_mean_20[0]
    getitem_41: "f32[1, 1024, 1]" = var_mean_20[1];  var_mean_20 = None
    sub_30: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_80, getitem_41);  getitem_41 = None
    add_81: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
    rsqrt_20: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
    mul_78: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = rsqrt_20 = None
    mul_79: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_78, arg158_1);  mul_78 = arg158_1 = None
    add_82: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_79, arg159_1);  mul_79 = arg159_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_218: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_82, [1024, 1024]);  add_82 = None
    permute_108: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
    addmm_58: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg161_1, view_218, permute_108);  arg161_1 = view_218 = permute_108 = None
    view_219: "f32[1, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_58, [1, 1024, 4096]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_80: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_219, 0.5)
    mul_81: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_219, 0.7071067811865476);  view_219 = None
    erf_9: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_81);  mul_81 = None
    add_83: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_82: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_80, add_83);  mul_80 = add_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_220: "f32[1024, 4096]" = torch.ops.aten.reshape.default(mul_82, [1024, 4096]);  mul_82 = None
    permute_109: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
    addmm_59: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg163_1, view_220, permute_109);  arg163_1 = view_220 = permute_109 = None
    view_221: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_59, [1, 1024, 1024]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:464, code: hidden_states = residual + hidden_states
    add_84: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_80, view_221);  add_80 = view_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:418, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_21 = torch.ops.aten.var_mean.correction(add_84, [2], correction = 0, keepdim = True)
    getitem_42: "f32[1, 1024, 1]" = var_mean_21[0]
    getitem_43: "f32[1, 1024, 1]" = var_mean_21[1];  var_mean_21 = None
    sub_31: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_84, getitem_43);  getitem_43 = None
    add_85: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
    rsqrt_21: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
    mul_83: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_21);  sub_31 = rsqrt_21 = None
    mul_84: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_83, arg164_1);  mul_83 = arg164_1 = None
    add_86: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_84, arg165_1);  mul_84 = arg165_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_222: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_86, [1024, 1024])
    permute_110: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg166_1, [1, 0]);  arg166_1 = None
    addmm_60: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg167_1, view_222, permute_110);  arg167_1 = view_222 = permute_110 = None
    view_223: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_60, [1, 1024, 1024]);  addmm_60 = None
    mul_85: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_223, 0.125);  view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_230: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_85, [1, 1024, 16, 64]);  mul_85 = None
    permute_115: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_230, [0, 2, 1, 3]);  view_230 = None
    clone_83: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_231: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_83, [16, -1, 64]);  clone_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_224: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_86, [1024, 1024])
    permute_111: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
    addmm_61: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg169_1, view_224, permute_111);  arg169_1 = view_224 = permute_111 = None
    view_225: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_61, [1, 1024, 1024]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_226: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_225, [1, -1, 16, 64]);  view_225 = None
    permute_112: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
    clone_81: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_112, memory_format = torch.contiguous_format);  permute_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_232: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_81, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_116: "f32[16, 64, 1024]" = torch.ops.aten.permute.default(view_232, [0, 2, 1]);  view_232 = None
    bmm_20: "f32[16, 1024, 1024]" = torch.ops.aten.bmm.default(view_231, permute_116);  view_231 = permute_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:246, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_234: "f32[1, 16, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_20, [1, 16, 1024, 1024]);  bmm_20 = None
    add_87: "f32[1, 16, 1024, 1024]" = torch.ops.aten.add.Tensor(view_234, expand_2);  view_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:247, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_235: "f32[16, 1024, 1024]" = torch.ops.aten.reshape.default(add_87, [16, 1024, 1024]);  add_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_10: "f32[16, 1024, 1]" = torch.ops.aten.amax.default(view_235, [-1], True)
    sub_32: "f32[16, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_235, amax_10);  view_235 = amax_10 = None
    exp_10: "f32[16, 1024, 1024]" = torch.ops.aten.exp.default(sub_32);  sub_32 = None
    sum_11: "f32[16, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_10: "f32[16, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_227: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_86, [1024, 1024]);  add_86 = None
    permute_113: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
    addmm_62: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg171_1, view_227, permute_113);  arg171_1 = view_227 = permute_113 = None
    view_228: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_62, [1, 1024, 1024]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_229: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_228, [1, -1, 16, 64]);  view_228 = None
    permute_114: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_229, [0, 2, 1, 3]);  view_229 = None
    clone_82: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_233: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_82, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_21: "f32[16, 1024, 64]" = torch.ops.aten.bmm.default(div_10, view_233);  div_10 = view_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_236: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(bmm_21, [1, 16, 1024, 64]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_117: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_236, [0, 2, 1, 3]);  view_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_85: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
    view_237: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_85, [1, 1024, 1024]);  clone_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_238: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_237, [1024, 1024]);  view_237 = None
    permute_118: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg172_1, [1, 0]);  arg172_1 = None
    addmm_63: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg173_1, view_238, permute_118);  arg173_1 = view_238 = permute_118 = None
    view_239: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_63, [1, 1024, 1024]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:432, code: hidden_states = residual + hidden_states
    add_88: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_84, view_239);  add_84 = view_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:459, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_22 = torch.ops.aten.var_mean.correction(add_88, [2], correction = 0, keepdim = True)
    getitem_44: "f32[1, 1024, 1]" = var_mean_22[0]
    getitem_45: "f32[1, 1024, 1]" = var_mean_22[1];  var_mean_22 = None
    sub_33: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_88, getitem_45);  getitem_45 = None
    add_89: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
    rsqrt_22: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
    mul_86: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = rsqrt_22 = None
    mul_87: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_86, arg174_1);  mul_86 = arg174_1 = None
    add_90: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_87, arg175_1);  mul_87 = arg175_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_240: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_90, [1024, 1024]);  add_90 = None
    permute_119: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
    addmm_64: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg177_1, view_240, permute_119);  arg177_1 = view_240 = permute_119 = None
    view_241: "f32[1, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_64, [1, 1024, 4096]);  addmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_88: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_241, 0.5)
    mul_89: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_241, 0.7071067811865476);  view_241 = None
    erf_10: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_89);  mul_89 = None
    add_91: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_90: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_88, add_91);  mul_88 = add_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_242: "f32[1024, 4096]" = torch.ops.aten.reshape.default(mul_90, [1024, 4096]);  mul_90 = None
    permute_120: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
    addmm_65: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg179_1, view_242, permute_120);  arg179_1 = view_242 = permute_120 = None
    view_243: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_65, [1, 1024, 1024]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:464, code: hidden_states = residual + hidden_states
    add_92: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_88, view_243);  add_88 = view_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:418, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_23 = torch.ops.aten.var_mean.correction(add_92, [2], correction = 0, keepdim = True)
    getitem_46: "f32[1, 1024, 1]" = var_mean_23[0]
    getitem_47: "f32[1, 1024, 1]" = var_mean_23[1];  var_mean_23 = None
    sub_34: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_92, getitem_47);  getitem_47 = None
    add_93: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
    rsqrt_23: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
    mul_91: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_23);  sub_34 = rsqrt_23 = None
    mul_92: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_91, arg180_1);  mul_91 = arg180_1 = None
    add_94: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_92, arg181_1);  mul_92 = arg181_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:189, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_244: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_94, [1024, 1024])
    permute_121: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg182_1, [1, 0]);  arg182_1 = None
    addmm_66: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg183_1, view_244, permute_121);  arg183_1 = view_244 = permute_121 = None
    view_245: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_66, [1, 1024, 1024]);  addmm_66 = None
    mul_93: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_245, 0.125);  view_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_252: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(mul_93, [1, 1024, 16, 64]);  mul_93 = None
    permute_126: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None
    clone_91: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:228, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_253: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_91, [16, -1, 64]);  clone_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:214, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_246: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_94, [1024, 1024])
    permute_122: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
    addmm_67: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg185_1, view_246, permute_122);  arg185_1 = view_246 = permute_122 = None
    view_247: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_67, [1, 1024, 1024]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_248: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_247, [1, -1, 16, 64]);  view_247 = None
    permute_123: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_248, [0, 2, 1, 3]);  view_248 = None
    clone_89: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_123, memory_format = torch.contiguous_format);  permute_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:229, code: key_states = key_states.reshape(*proj_shape)
    view_254: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_89, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:233, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_127: "f32[16, 64, 1024]" = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None
    bmm_22: "f32[16, 1024, 1024]" = torch.ops.aten.bmm.default(view_253, permute_127);  view_253 = permute_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:246, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_256: "f32[1, 16, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_22, [1, 16, 1024, 1024]);  bmm_22 = None
    add_95: "f32[1, 16, 1024, 1024]" = torch.ops.aten.add.Tensor(view_256, expand_2);  view_256 = expand_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:247, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_257: "f32[16, 1024, 1024]" = torch.ops.aten.reshape.default(add_95, [16, 1024, 1024]);  add_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:249, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_11: "f32[16, 1024, 1]" = torch.ops.aten.amax.default(view_257, [-1], True)
    sub_35: "f32[16, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_257, amax_11);  view_257 = amax_11 = None
    exp_11: "f32[16, 1024, 1024]" = torch.ops.aten.exp.default(sub_35);  sub_35 = None
    sum_12: "f32[16, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_11: "f32[16, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:215, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_249: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_94, [1024, 1024]);  add_94 = None
    permute_124: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg186_1, [1, 0]);  arg186_1 = None
    addmm_68: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg187_1, view_249, permute_124);  arg187_1 = view_249 = permute_124 = None
    view_250: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_68, [1, 1024, 1024]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:169, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_251: "f32[1, 1024, 16, 64]" = torch.ops.aten.reshape.default(view_250, [1, -1, 16, 64]);  view_250 = None
    permute_125: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_251, [0, 2, 1, 3]);  view_251 = None
    clone_90: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:230, code: value_states = value_states.reshape(*proj_shape)
    view_255: "f32[16, 1024, 64]" = torch.ops.aten.reshape.default(clone_90, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:272, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_23: "f32[16, 1024, 64]" = torch.ops.aten.bmm.default(div_11, view_255);  div_11 = view_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:280, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_258: "f32[1, 16, 1024, 64]" = torch.ops.aten.reshape.default(bmm_23, [1, 16, 1024, 64]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:281, code: attn_output = attn_output.transpose(1, 2)
    permute_128: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_258, [0, 2, 1, 3]);  view_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:285, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_93: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
    view_259: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(clone_93, [1, 1024, 1024]);  clone_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:287, code: attn_output = self.out_proj(attn_output)
    view_260: "f32[1024, 1024]" = torch.ops.aten.reshape.default(view_259, [1024, 1024]);  view_259 = None
    permute_129: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
    addmm_69: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg189_1, view_260, permute_129);  arg189_1 = view_260 = permute_129 = None
    view_261: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_69, [1, 1024, 1024]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:432, code: hidden_states = residual + hidden_states
    add_96: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_92, view_261);  add_92 = view_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:459, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_24 = torch.ops.aten.var_mean.correction(add_96, [2], correction = 0, keepdim = True)
    getitem_48: "f32[1, 1024, 1]" = var_mean_24[0]
    getitem_49: "f32[1, 1024, 1]" = var_mean_24[1];  var_mean_24 = None
    sub_36: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_96, getitem_49);  getitem_49 = None
    add_97: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
    rsqrt_24: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
    mul_94: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = rsqrt_24 = None
    mul_95: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_94, arg190_1);  mul_94 = arg190_1 = None
    add_98: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_95, arg191_1);  mul_95 = arg191_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_262: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_98, [1024, 1024]);  add_98 = None
    permute_130: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg192_1, [1, 0]);  arg192_1 = None
    addmm_70: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg193_1, view_262, permute_130);  arg193_1 = view_262 = permute_130 = None
    view_263: "f32[1, 1024, 4096]" = torch.ops.aten.reshape.default(addmm_70, [1, 1024, 4096]);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_96: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_263, 0.5)
    mul_97: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_263, 0.7071067811865476);  view_263 = None
    erf_11: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_97);  mul_97 = None
    add_99: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_98: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_96, add_99);  mul_96 = add_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_264: "f32[1024, 4096]" = torch.ops.aten.reshape.default(mul_98, [1024, 4096]);  mul_98 = None
    permute_131: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg194_1, [1, 0]);  arg194_1 = None
    addmm_71: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg195_1, view_264, permute_131);  arg195_1 = view_264 = permute_131 = None
    view_265: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(addmm_71, [1, 1024, 1024]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:464, code: hidden_states = residual + hidden_states
    add_100: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_96, view_265);  add_96 = view_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:1132, code: hidden_states = self.layer_norm(hidden_states)
    var_mean_25 = torch.ops.aten.var_mean.correction(add_100, [2], correction = 0, keepdim = True)
    getitem_50: "f32[1, 1024, 1]" = var_mean_25[0]
    getitem_51: "f32[1, 1024, 1]" = var_mean_25[1];  var_mean_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:1871, code: loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
    view_269: "i64[1024]" = torch.ops.aten.reshape.default(arg200_1, [-1]);  arg200_1 = None
    ne_1: "b8[1024]" = torch.ops.aten.ne.Scalar(view_269, -100)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:1132, code: hidden_states = self.layer_norm(hidden_states)
    sub_37: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_100, getitem_51);  add_100 = getitem_51 = None
    add_101: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
    rsqrt_25: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
    mul_99: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_25);  sub_37 = rsqrt_25 = None
    mul_100: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_99, arg196_1);  mul_99 = arg196_1 = None
    add_102: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_100, arg197_1);  mul_100 = arg197_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:1865, code: logits = self.lm_head(outputs[0])
    view_266: "f32[1024, 1024]" = torch.ops.aten.reshape.default(add_102, [1024, 1024]);  add_102 = None
    permute_132: "f32[1024, 50265]" = torch.ops.aten.permute.default(arg198_1, [1, 0]);  arg198_1 = None
    mm: "f32[1024, 50265]" = torch.ops.aten.mm.default(view_266, permute_132);  view_266 = permute_132 = None
    view_267: "f32[1, 1024, 50265]" = torch.ops.aten.reshape.default(mm, [1, 1024, 50265]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mbart/modeling_mbart.py:1871, code: loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
    view_268: "f32[1024, 50265]" = torch.ops.aten.reshape.default(view_267, [-1, 50265])
    amax_12: "f32[1024, 1]" = torch.ops.aten.amax.default(view_268, [1], True)
    sub_38: "f32[1024, 50265]" = torch.ops.aten.sub.Tensor(view_268, amax_12);  view_268 = amax_12 = None
    exp_12: "f32[1024, 50265]" = torch.ops.aten.exp.default(sub_38)
    sum_13: "f32[1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [1], True);  exp_12 = None
    log: "f32[1024, 1]" = torch.ops.aten.log.default(sum_13);  sum_13 = None
    sub_39: "f32[1024, 50265]" = torch.ops.aten.sub.Tensor(sub_38, log);  sub_38 = log = None
    ne: "b8[1024]" = torch.ops.aten.ne.Scalar(view_269, -100)
    full_default_2: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where_1: "i64[1024]" = torch.ops.aten.where.self(ne, view_269, full_default_2);  ne = full_default_2 = None
    unsqueeze_4: "i64[1024, 1]" = torch.ops.aten.unsqueeze.default(where_1, 1);  where_1 = None
    gather: "f32[1024, 1]" = torch.ops.aten.gather.default(sub_39, 1, unsqueeze_4);  sub_39 = unsqueeze_4 = None
    squeeze: "f32[1024]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[1024]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
    full_default_3: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where_2: "f32[1024]" = torch.ops.aten.where.self(ne_1, neg, full_default_3);  ne_1 = neg = full_default_3 = None
    sum_15: "f32[]" = torch.ops.aten.sum.default(where_2);  where_2 = None
    ne_2: "b8[1024]" = torch.ops.aten.ne.Scalar(view_269, -100);  view_269 = None
    sum_14: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None
    div_12: "f32[]" = torch.ops.aten.div.Tensor(sum_15, convert_element_type);  sum_15 = convert_element_type = None
    return (div_12, view_267, clone_1, clone_2, clone_9, clone_10, clone_17, clone_18, clone_25, clone_26, clone_33, clone_34, clone_41, clone_42, clone_49, clone_50, clone_57, clone_58, clone_65, clone_66, clone_73, clone_74, clone_81, clone_82, clone_89, clone_90)
    