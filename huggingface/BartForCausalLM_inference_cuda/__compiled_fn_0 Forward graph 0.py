from __future__ import annotations



def forward(self, arg0_1: "f32[1026, 1024]", arg1_1: "f32[50265, 1024]", arg2_1: "f32[1024]", arg3_1: "f32[1024]", arg4_1: "f32[1024, 1024]", arg5_1: "f32[1024]", arg6_1: "f32[1024, 1024]", arg7_1: "f32[1024]", arg8_1: "f32[1024, 1024]", arg9_1: "f32[1024]", arg10_1: "f32[1024, 1024]", arg11_1: "f32[1024]", arg12_1: "f32[1024]", arg13_1: "f32[1024]", arg14_1: "f32[4096, 1024]", arg15_1: "f32[4096]", arg16_1: "f32[1024, 4096]", arg17_1: "f32[1024]", arg18_1: "f32[1024]", arg19_1: "f32[1024]", arg20_1: "f32[1024, 1024]", arg21_1: "f32[1024]", arg22_1: "f32[1024, 1024]", arg23_1: "f32[1024]", arg24_1: "f32[1024, 1024]", arg25_1: "f32[1024]", arg26_1: "f32[1024, 1024]", arg27_1: "f32[1024]", arg28_1: "f32[1024]", arg29_1: "f32[1024]", arg30_1: "f32[4096, 1024]", arg31_1: "f32[4096]", arg32_1: "f32[1024, 4096]", arg33_1: "f32[1024]", arg34_1: "f32[1024]", arg35_1: "f32[1024]", arg36_1: "f32[1024, 1024]", arg37_1: "f32[1024]", arg38_1: "f32[1024, 1024]", arg39_1: "f32[1024]", arg40_1: "f32[1024, 1024]", arg41_1: "f32[1024]", arg42_1: "f32[1024, 1024]", arg43_1: "f32[1024]", arg44_1: "f32[1024]", arg45_1: "f32[1024]", arg46_1: "f32[4096, 1024]", arg47_1: "f32[4096]", arg48_1: "f32[1024, 4096]", arg49_1: "f32[1024]", arg50_1: "f32[1024]", arg51_1: "f32[1024]", arg52_1: "f32[1024, 1024]", arg53_1: "f32[1024]", arg54_1: "f32[1024, 1024]", arg55_1: "f32[1024]", arg56_1: "f32[1024, 1024]", arg57_1: "f32[1024]", arg58_1: "f32[1024, 1024]", arg59_1: "f32[1024]", arg60_1: "f32[1024]", arg61_1: "f32[1024]", arg62_1: "f32[4096, 1024]", arg63_1: "f32[4096]", arg64_1: "f32[1024, 4096]", arg65_1: "f32[1024]", arg66_1: "f32[1024]", arg67_1: "f32[1024]", arg68_1: "f32[1024, 1024]", arg69_1: "f32[1024]", arg70_1: "f32[1024, 1024]", arg71_1: "f32[1024]", arg72_1: "f32[1024, 1024]", arg73_1: "f32[1024]", arg74_1: "f32[1024, 1024]", arg75_1: "f32[1024]", arg76_1: "f32[1024]", arg77_1: "f32[1024]", arg78_1: "f32[4096, 1024]", arg79_1: "f32[4096]", arg80_1: "f32[1024, 4096]", arg81_1: "f32[1024]", arg82_1: "f32[1024]", arg83_1: "f32[1024]", arg84_1: "f32[1024, 1024]", arg85_1: "f32[1024]", arg86_1: "f32[1024, 1024]", arg87_1: "f32[1024]", arg88_1: "f32[1024, 1024]", arg89_1: "f32[1024]", arg90_1: "f32[1024, 1024]", arg91_1: "f32[1024]", arg92_1: "f32[1024]", arg93_1: "f32[1024]", arg94_1: "f32[4096, 1024]", arg95_1: "f32[4096]", arg96_1: "f32[1024, 4096]", arg97_1: "f32[1024]", arg98_1: "f32[1024]", arg99_1: "f32[1024]", arg100_1: "f32[1024, 1024]", arg101_1: "f32[1024]", arg102_1: "f32[1024, 1024]", arg103_1: "f32[1024]", arg104_1: "f32[1024, 1024]", arg105_1: "f32[1024]", arg106_1: "f32[1024, 1024]", arg107_1: "f32[1024]", arg108_1: "f32[1024]", arg109_1: "f32[1024]", arg110_1: "f32[4096, 1024]", arg111_1: "f32[4096]", arg112_1: "f32[1024, 4096]", arg113_1: "f32[1024]", arg114_1: "f32[1024]", arg115_1: "f32[1024]", arg116_1: "f32[1024, 1024]", arg117_1: "f32[1024]", arg118_1: "f32[1024, 1024]", arg119_1: "f32[1024]", arg120_1: "f32[1024, 1024]", arg121_1: "f32[1024]", arg122_1: "f32[1024, 1024]", arg123_1: "f32[1024]", arg124_1: "f32[1024]", arg125_1: "f32[1024]", arg126_1: "f32[4096, 1024]", arg127_1: "f32[4096]", arg128_1: "f32[1024, 4096]", arg129_1: "f32[1024]", arg130_1: "f32[1024]", arg131_1: "f32[1024]", arg132_1: "f32[1024, 1024]", arg133_1: "f32[1024]", arg134_1: "f32[1024, 1024]", arg135_1: "f32[1024]", arg136_1: "f32[1024, 1024]", arg137_1: "f32[1024]", arg138_1: "f32[1024, 1024]", arg139_1: "f32[1024]", arg140_1: "f32[1024]", arg141_1: "f32[1024]", arg142_1: "f32[4096, 1024]", arg143_1: "f32[4096]", arg144_1: "f32[1024, 4096]", arg145_1: "f32[1024]", arg146_1: "f32[1024]", arg147_1: "f32[1024]", arg148_1: "f32[1024, 1024]", arg149_1: "f32[1024]", arg150_1: "f32[1024, 1024]", arg151_1: "f32[1024]", arg152_1: "f32[1024, 1024]", arg153_1: "f32[1024]", arg154_1: "f32[1024, 1024]", arg155_1: "f32[1024]", arg156_1: "f32[1024]", arg157_1: "f32[1024]", arg158_1: "f32[4096, 1024]", arg159_1: "f32[4096]", arg160_1: "f32[1024, 4096]", arg161_1: "f32[1024]", arg162_1: "f32[1024]", arg163_1: "f32[1024]", arg164_1: "f32[1024, 1024]", arg165_1: "f32[1024]", arg166_1: "f32[1024, 1024]", arg167_1: "f32[1024]", arg168_1: "f32[1024, 1024]", arg169_1: "f32[1024]", arg170_1: "f32[1024, 1024]", arg171_1: "f32[1024]", arg172_1: "f32[1024]", arg173_1: "f32[1024]", arg174_1: "f32[4096, 1024]", arg175_1: "f32[4096]", arg176_1: "f32[1024, 4096]", arg177_1: "f32[1024]", arg178_1: "f32[1024]", arg179_1: "f32[1024]", arg180_1: "f32[1024, 1024]", arg181_1: "f32[1024]", arg182_1: "f32[1024, 1024]", arg183_1: "f32[1024]", arg184_1: "f32[1024, 1024]", arg185_1: "f32[1024]", arg186_1: "f32[1024, 1024]", arg187_1: "f32[1024]", arg188_1: "f32[1024]", arg189_1: "f32[1024]", arg190_1: "f32[4096, 1024]", arg191_1: "f32[4096]", arg192_1: "f32[1024, 4096]", arg193_1: "f32[1024]", arg194_1: "f32[1024]", arg195_1: "f32[1024]", arg196_1: "f32[50265, 1024]", arg197_1: "i64[1, 1024]", arg198_1: "i64[1, 1024]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:1059, code: inputs_embeds = self.embed_tokens(input) * self.embed_scale
    embedding: "f32[1, 1024, 1024]" = torch.ops.aten.embedding.default(arg1_1, arg197_1, 1);  arg1_1 = arg197_1 = None
    mul: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(embedding, 1.0);  embedding = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:96, code: mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    full: "f32[1024, 1024]" = torch.ops.aten.full.default([1024, 1024], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:97, code: mask_cond = torch.arange(mask.size(-1), device=device)
    iota: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:98, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    add: "i64[1024]" = torch.ops.aten.add.Tensor(iota, 1)
    view_1: "i64[1024, 1]" = torch.ops.aten.view.default(add, [1024, 1]);  add = None
    lt: "b8[1024, 1024]" = torch.ops.aten.lt.Tensor(iota, view_1);  iota = view_1 = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where: "f32[1024, 1024]" = torch.ops.aten.where.self(lt, scalar_tensor, full);  lt = scalar_tensor = full = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:135, code: positions = torch.arange(
    iota_1: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:137, code: ).expand(bsz, -1)
    expand_1: "i64[1, 1024]" = torch.ops.aten.expand.default(iota_1, [1, -1]);  iota_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:139, code: return super().forward(positions + self.offset)
    add_1: "i64[1, 1024]" = torch.ops.aten.add.Tensor(expand_1, 2);  expand_1 = None
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/sparse.py:163, code: return F.embedding(
    embedding_1: "f32[1, 1024, 1024]" = torch.ops.aten.embedding.default(arg0_1, add_1);  arg0_1 = add_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:1074, code: hidden_states = inputs_embeds + positions
    add_2: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul, embedding_1);  mul = embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:1075, code: hidden_states = self.layernorm_embedding(hidden_states)
    var_mean = torch.ops.aten.var_mean.correction(add_2, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 1024, 1]" = var_mean[0]
    getitem_1: "f32[1, 1024, 1]" = var_mean[1];  var_mean = None
    add_3: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
    sub: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_2, getitem_1);  add_2 = getitem_1 = None
    mul_1: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
    mul_2: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_1, arg2_1);  mul_1 = arg2_1 = None
    add_4: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_2, arg3_1);  mul_2 = arg3_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:1077, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone: "f32[1, 1024, 1024]" = torch.ops.aten.clone.default(add_4);  add_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_2: "f32[1024, 1024]" = torch.ops.aten.view.default(clone, [1024, 1024])
    permute: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
    addmm: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg5_1, view_2, permute);  arg5_1 = view_2 = permute = None
    view_3: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm, [1, 1024, 1024]);  addmm = None
    mul_3: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_3, 0.125);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_4: "f32[1024, 1024]" = torch.ops.aten.view.default(clone, [1024, 1024])
    permute_1: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
    addmm_1: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg7_1, view_4, permute_1);  arg7_1 = view_4 = permute_1 = None
    view_5: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_1, [1, 1024, 1024]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_6: "f32[1, 1024, 16, 64]" = torch.ops.aten.view.default(view_5, [1, -1, 16, 64]);  view_5 = None
    permute_2: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
    clone_1: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_7: "f32[1024, 1024]" = torch.ops.aten.view.default(clone, [1024, 1024])
    permute_3: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
    addmm_2: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg9_1, view_7, permute_3);  arg9_1 = view_7 = permute_3 = None
    view_8: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_2, [1, 1024, 1024]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_9: "f32[1, 1024, 16, 64]" = torch.ops.aten.view.default(view_8, [1, -1, 16, 64]);  view_8 = None
    permute_4: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
    clone_2: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_10: "f32[1, 1024, 16, 64]" = torch.ops.aten.view.default(mul_3, [1, 1024, 16, 64]);  mul_3 = None
    permute_5: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
    clone_3: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_11: "f32[16, 1024, 64]" = torch.ops.aten.view.default(clone_3, [16, -1, 64]);  clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_12: "f32[16, 1024, 64]" = torch.ops.aten.view.default(clone_1, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_13: "f32[16, 1024, 64]" = torch.ops.aten.view.default(clone_2, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_6: "f32[16, 64, 1024]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
    bmm: "f32[16, 1024, 1024]" = torch.ops.aten.bmm.default(view_11, permute_6);  view_11 = permute_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:250, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_14: "f32[1, 16, 1024, 1024]" = torch.ops.aten.view.default(bmm, [1, 16, 1024, 1024]);  bmm = None
    unsqueeze_2: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(where, 0);  where = None
    unsqueeze_3: "f32[1, 1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 1);  unsqueeze_2 = None
    slice_3: "f32[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(unsqueeze_3, 2, 0, 9223372036854775807);  unsqueeze_3 = None
    slice_4: "f32[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_3, 3, 0, 9223372036854775807);  slice_3 = None
    expand_2: "f32[1, 1, 1024, 1024]" = torch.ops.aten.expand.default(slice_4, [1, 1, 1024, 1024]);  slice_4 = None
    add_5: "f32[1, 16, 1024, 1024]" = torch.ops.aten.add.Tensor(view_14, expand_2);  view_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:251, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_15: "f32[16, 1024, 1024]" = torch.ops.aten.view.default(add_5, [16, 1024, 1024]);  add_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax: "f32[16, 1024, 1]" = torch.ops.aten.amax.default(view_15, [-1], True)
    sub_1: "f32[16, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_15, amax);  view_15 = amax = None
    exp: "f32[16, 1024, 1024]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
    sum_1: "f32[16, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[16, 1024, 1024]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:274, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_4: "f32[16, 1024, 1024]" = torch.ops.aten.clone.default(div);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_1: "f32[16, 1024, 64]" = torch.ops.aten.bmm.default(clone_4, view_13);  clone_4 = view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_16: "f32[1, 16, 1024, 64]" = torch.ops.aten.view.default(bmm_1, [1, 16, 1024, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_7: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_5: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_17: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_5, [1, 1024, 1024]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_18: "f32[1024, 1024]" = torch.ops.aten.view.default(view_17, [1024, 1024]);  view_17 = None
    permute_8: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
    addmm_3: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg11_1, view_18, permute_8);  arg11_1 = view_18 = permute_8 = None
    view_19: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_3, [1, 1024, 1024]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:434, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_6: "f32[1, 1024, 1024]" = torch.ops.aten.clone.default(view_19);  view_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:435, code: hidden_states = residual + hidden_states
    add_6: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(clone, clone_6);  clone = clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_1 = torch.ops.aten.var_mean.correction(add_6, [2], correction = 0, keepdim = True)
    getitem_2: "f32[1, 1024, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 1024, 1]" = var_mean_1[1];  var_mean_1 = None
    add_7: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
    rsqrt_1: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
    sub_2: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_6, getitem_3);  add_6 = getitem_3 = None
    mul_4: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
    mul_5: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_4, arg12_1);  mul_4 = arg12_1 = None
    add_8: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_5, arg13_1);  mul_5 = arg13_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_20: "f32[1024, 1024]" = torch.ops.aten.view.default(add_8, [1024, 1024])
    permute_9: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
    addmm_4: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg15_1, view_20, permute_9);  arg15_1 = view_20 = permute_9 = None
    view_21: "f32[1, 1024, 4096]" = torch.ops.aten.view.default(addmm_4, [1, 1024, 4096]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_6: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_21, 0.5)
    mul_7: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_21, 0.7071067811865476);  view_21 = None
    erf: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_7);  mul_7 = None
    add_9: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_8: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_6, add_9);  mul_6 = add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:464, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_7: "f32[1, 1024, 4096]" = torch.ops.aten.clone.default(mul_8);  mul_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_22: "f32[1024, 4096]" = torch.ops.aten.view.default(clone_7, [1024, 4096]);  clone_7 = None
    permute_10: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
    addmm_5: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg17_1, view_22, permute_10);  arg17_1 = view_22 = permute_10 = None
    view_23: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_5, [1, 1024, 1024]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:466, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_8: "f32[1, 1024, 1024]" = torch.ops.aten.clone.default(view_23);  view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:467, code: hidden_states = residual + hidden_states
    add_10: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_8, clone_8);  add_8 = clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_10, [2], correction = 0, keepdim = True)
    getitem_4: "f32[1, 1024, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 1024, 1]" = var_mean_2[1];  var_mean_2 = None
    add_11: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
    rsqrt_2: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_3: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_10, getitem_5);  add_10 = getitem_5 = None
    mul_9: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
    mul_10: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_9, arg18_1);  mul_9 = arg18_1 = None
    add_12: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_10, arg19_1);  mul_10 = arg19_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_24: "f32[1024, 1024]" = torch.ops.aten.view.default(add_12, [1024, 1024])
    permute_11: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
    addmm_6: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg21_1, view_24, permute_11);  arg21_1 = view_24 = permute_11 = None
    view_25: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_6, [1, 1024, 1024]);  addmm_6 = None
    mul_11: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_25, 0.125);  view_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_26: "f32[1024, 1024]" = torch.ops.aten.view.default(add_12, [1024, 1024])
    permute_12: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
    addmm_7: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg23_1, view_26, permute_12);  arg23_1 = view_26 = permute_12 = None
    view_27: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_7, [1, 1024, 1024]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_28: "f32[1, 1024, 16, 64]" = torch.ops.aten.view.default(view_27, [1, -1, 16, 64]);  view_27 = None
    permute_13: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
    clone_9: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_29: "f32[1024, 1024]" = torch.ops.aten.view.default(add_12, [1024, 1024])
    permute_14: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
    addmm_8: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg25_1, view_29, permute_14);  arg25_1 = view_29 = permute_14 = None
    view_30: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_8, [1, 1024, 1024]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_31: "f32[1, 1024, 16, 64]" = torch.ops.aten.view.default(view_30, [1, -1, 16, 64]);  view_30 = None
    permute_15: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None
    clone_10: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_32: "f32[1, 1024, 16, 64]" = torch.ops.aten.view.default(mul_11, [1, 1024, 16, 64]);  mul_11 = None
    permute_16: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None
    clone_11: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_33: "f32[16, 1024, 64]" = torch.ops.aten.view.default(clone_11, [16, -1, 64]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_34: "f32[16, 1024, 64]" = torch.ops.aten.view.default(clone_9, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_35: "f32[16, 1024, 64]" = torch.ops.aten.view.default(clone_10, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_17: "f32[16, 64, 1024]" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
    bmm_2: "f32[16, 1024, 1024]" = torch.ops.aten.bmm.default(view_33, permute_17);  view_33 = permute_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:250, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_36: "f32[1, 16, 1024, 1024]" = torch.ops.aten.view.default(bmm_2, [1, 16, 1024, 1024]);  bmm_2 = None
    add_13: "f32[1, 16, 1024, 1024]" = torch.ops.aten.add.Tensor(view_36, expand_2);  view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:251, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_37: "f32[16, 1024, 1024]" = torch.ops.aten.view.default(add_13, [16, 1024, 1024]);  add_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_1: "f32[16, 1024, 1]" = torch.ops.aten.amax.default(view_37, [-1], True)
    sub_4: "f32[16, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_37, amax_1);  view_37 = amax_1 = None
    exp_1: "f32[16, 1024, 1024]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
    sum_2: "f32[16, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_1: "f32[16, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:274, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_12: "f32[16, 1024, 1024]" = torch.ops.aten.clone.default(div_1);  div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_3: "f32[16, 1024, 64]" = torch.ops.aten.bmm.default(clone_12, view_35);  clone_12 = view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_38: "f32[1, 16, 1024, 64]" = torch.ops.aten.view.default(bmm_3, [1, 16, 1024, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_18: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_13: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
    view_39: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_13, [1, 1024, 1024]);  clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_40: "f32[1024, 1024]" = torch.ops.aten.view.default(view_39, [1024, 1024]);  view_39 = None
    permute_19: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
    addmm_9: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg27_1, view_40, permute_19);  arg27_1 = view_40 = permute_19 = None
    view_41: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_9, [1, 1024, 1024]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:434, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_14: "f32[1, 1024, 1024]" = torch.ops.aten.clone.default(view_41);  view_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:435, code: hidden_states = residual + hidden_states
    add_14: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_12, clone_14);  add_12 = clone_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_3 = torch.ops.aten.var_mean.correction(add_14, [2], correction = 0, keepdim = True)
    getitem_6: "f32[1, 1024, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 1024, 1]" = var_mean_3[1];  var_mean_3 = None
    add_15: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
    rsqrt_3: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
    sub_5: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_14, getitem_7);  add_14 = getitem_7 = None
    mul_12: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
    mul_13: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_12, arg28_1);  mul_12 = arg28_1 = None
    add_16: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_13, arg29_1);  mul_13 = arg29_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_42: "f32[1024, 1024]" = torch.ops.aten.view.default(add_16, [1024, 1024])
    permute_20: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
    addmm_10: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg31_1, view_42, permute_20);  arg31_1 = view_42 = permute_20 = None
    view_43: "f32[1, 1024, 4096]" = torch.ops.aten.view.default(addmm_10, [1, 1024, 4096]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_14: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_43, 0.5)
    mul_15: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_43, 0.7071067811865476);  view_43 = None
    erf_1: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_15);  mul_15 = None
    add_17: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_16: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_14, add_17);  mul_14 = add_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:464, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_15: "f32[1, 1024, 4096]" = torch.ops.aten.clone.default(mul_16);  mul_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_44: "f32[1024, 4096]" = torch.ops.aten.view.default(clone_15, [1024, 4096]);  clone_15 = None
    permute_21: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
    addmm_11: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg33_1, view_44, permute_21);  arg33_1 = view_44 = permute_21 = None
    view_45: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_11, [1, 1024, 1024]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:466, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_16: "f32[1, 1024, 1024]" = torch.ops.aten.clone.default(view_45);  view_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:467, code: hidden_states = residual + hidden_states
    add_18: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_16, clone_16);  add_16 = clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_4 = torch.ops.aten.var_mean.correction(add_18, [2], correction = 0, keepdim = True)
    getitem_8: "f32[1, 1024, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 1024, 1]" = var_mean_4[1];  var_mean_4 = None
    add_19: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
    rsqrt_4: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
    sub_6: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_18, getitem_9);  add_18 = getitem_9 = None
    mul_17: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
    mul_18: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_17, arg34_1);  mul_17 = arg34_1 = None
    add_20: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_18, arg35_1);  mul_18 = arg35_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_46: "f32[1024, 1024]" = torch.ops.aten.view.default(add_20, [1024, 1024])
    permute_22: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
    addmm_12: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg37_1, view_46, permute_22);  arg37_1 = view_46 = permute_22 = None
    view_47: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_12, [1, 1024, 1024]);  addmm_12 = None
    mul_19: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_47, 0.125);  view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_48: "f32[1024, 1024]" = torch.ops.aten.view.default(add_20, [1024, 1024])
    permute_23: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
    addmm_13: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg39_1, view_48, permute_23);  arg39_1 = view_48 = permute_23 = None
    view_49: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_13, [1, 1024, 1024]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_50: "f32[1, 1024, 16, 64]" = torch.ops.aten.view.default(view_49, [1, -1, 16, 64]);  view_49 = None
    permute_24: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
    clone_17: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_51: "f32[1024, 1024]" = torch.ops.aten.view.default(add_20, [1024, 1024])
    permute_25: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
    addmm_14: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg41_1, view_51, permute_25);  arg41_1 = view_51 = permute_25 = None
    view_52: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_14, [1, 1024, 1024]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_53: "f32[1, 1024, 16, 64]" = torch.ops.aten.view.default(view_52, [1, -1, 16, 64]);  view_52 = None
    permute_26: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
    clone_18: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_54: "f32[1, 1024, 16, 64]" = torch.ops.aten.view.default(mul_19, [1, 1024, 16, 64]);  mul_19 = None
    permute_27: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
    clone_19: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_55: "f32[16, 1024, 64]" = torch.ops.aten.view.default(clone_19, [16, -1, 64]);  clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_56: "f32[16, 1024, 64]" = torch.ops.aten.view.default(clone_17, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_57: "f32[16, 1024, 64]" = torch.ops.aten.view.default(clone_18, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_28: "f32[16, 64, 1024]" = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None
    bmm_4: "f32[16, 1024, 1024]" = torch.ops.aten.bmm.default(view_55, permute_28);  view_55 = permute_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:250, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_58: "f32[1, 16, 1024, 1024]" = torch.ops.aten.view.default(bmm_4, [1, 16, 1024, 1024]);  bmm_4 = None
    add_21: "f32[1, 16, 1024, 1024]" = torch.ops.aten.add.Tensor(view_58, expand_2);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:251, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_59: "f32[16, 1024, 1024]" = torch.ops.aten.view.default(add_21, [16, 1024, 1024]);  add_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_2: "f32[16, 1024, 1]" = torch.ops.aten.amax.default(view_59, [-1], True)
    sub_7: "f32[16, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_59, amax_2);  view_59 = amax_2 = None
    exp_2: "f32[16, 1024, 1024]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
    sum_3: "f32[16, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_2: "f32[16, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:274, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_20: "f32[16, 1024, 1024]" = torch.ops.aten.clone.default(div_2);  div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_5: "f32[16, 1024, 64]" = torch.ops.aten.bmm.default(clone_20, view_57);  clone_20 = view_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_60: "f32[1, 16, 1024, 64]" = torch.ops.aten.view.default(bmm_5, [1, 16, 1024, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_29: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_60, [0, 2, 1, 3]);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_21: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    view_61: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_21, [1, 1024, 1024]);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_62: "f32[1024, 1024]" = torch.ops.aten.view.default(view_61, [1024, 1024]);  view_61 = None
    permute_30: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
    addmm_15: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg43_1, view_62, permute_30);  arg43_1 = view_62 = permute_30 = None
    view_63: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_15, [1, 1024, 1024]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:434, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_22: "f32[1, 1024, 1024]" = torch.ops.aten.clone.default(view_63);  view_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:435, code: hidden_states = residual + hidden_states
    add_22: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_20, clone_22);  add_20 = clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_5 = torch.ops.aten.var_mean.correction(add_22, [2], correction = 0, keepdim = True)
    getitem_10: "f32[1, 1024, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 1024, 1]" = var_mean_5[1];  var_mean_5 = None
    add_23: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
    rsqrt_5: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
    sub_8: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_22, getitem_11);  add_22 = getitem_11 = None
    mul_20: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
    mul_21: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_20, arg44_1);  mul_20 = arg44_1 = None
    add_24: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_21, arg45_1);  mul_21 = arg45_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_64: "f32[1024, 1024]" = torch.ops.aten.view.default(add_24, [1024, 1024])
    permute_31: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
    addmm_16: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg47_1, view_64, permute_31);  arg47_1 = view_64 = permute_31 = None
    view_65: "f32[1, 1024, 4096]" = torch.ops.aten.view.default(addmm_16, [1, 1024, 4096]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_22: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_65, 0.5)
    mul_23: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_65, 0.7071067811865476);  view_65 = None
    erf_2: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_23);  mul_23 = None
    add_25: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_24: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_22, add_25);  mul_22 = add_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:464, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_23: "f32[1, 1024, 4096]" = torch.ops.aten.clone.default(mul_24);  mul_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_66: "f32[1024, 4096]" = torch.ops.aten.view.default(clone_23, [1024, 4096]);  clone_23 = None
    permute_32: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
    addmm_17: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg49_1, view_66, permute_32);  arg49_1 = view_66 = permute_32 = None
    view_67: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_17, [1, 1024, 1024]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:466, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_24: "f32[1, 1024, 1024]" = torch.ops.aten.clone.default(view_67);  view_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:467, code: hidden_states = residual + hidden_states
    add_26: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_24, clone_24);  add_24 = clone_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_6 = torch.ops.aten.var_mean.correction(add_26, [2], correction = 0, keepdim = True)
    getitem_12: "f32[1, 1024, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 1024, 1]" = var_mean_6[1];  var_mean_6 = None
    add_27: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
    rsqrt_6: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
    sub_9: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_26, getitem_13);  add_26 = getitem_13 = None
    mul_25: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = rsqrt_6 = None
    mul_26: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_25, arg50_1);  mul_25 = arg50_1 = None
    add_28: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_26, arg51_1);  mul_26 = arg51_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_68: "f32[1024, 1024]" = torch.ops.aten.view.default(add_28, [1024, 1024])
    permute_33: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
    addmm_18: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg53_1, view_68, permute_33);  arg53_1 = view_68 = permute_33 = None
    view_69: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_18, [1, 1024, 1024]);  addmm_18 = None
    mul_27: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_69, 0.125);  view_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_70: "f32[1024, 1024]" = torch.ops.aten.view.default(add_28, [1024, 1024])
    permute_34: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
    addmm_19: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg55_1, view_70, permute_34);  arg55_1 = view_70 = permute_34 = None
    view_71: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_19, [1, 1024, 1024]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_72: "f32[1, 1024, 16, 64]" = torch.ops.aten.view.default(view_71, [1, -1, 16, 64]);  view_71 = None
    permute_35: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
    clone_25: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_73: "f32[1024, 1024]" = torch.ops.aten.view.default(add_28, [1024, 1024])
    permute_36: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
    addmm_20: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg57_1, view_73, permute_36);  arg57_1 = view_73 = permute_36 = None
    view_74: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_20, [1, 1024, 1024]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_75: "f32[1, 1024, 16, 64]" = torch.ops.aten.view.default(view_74, [1, -1, 16, 64]);  view_74 = None
    permute_37: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
    clone_26: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_76: "f32[1, 1024, 16, 64]" = torch.ops.aten.view.default(mul_27, [1, 1024, 16, 64]);  mul_27 = None
    permute_38: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
    clone_27: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_77: "f32[16, 1024, 64]" = torch.ops.aten.view.default(clone_27, [16, -1, 64]);  clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_78: "f32[16, 1024, 64]" = torch.ops.aten.view.default(clone_25, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_79: "f32[16, 1024, 64]" = torch.ops.aten.view.default(clone_26, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_39: "f32[16, 64, 1024]" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
    bmm_6: "f32[16, 1024, 1024]" = torch.ops.aten.bmm.default(view_77, permute_39);  view_77 = permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:250, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_80: "f32[1, 16, 1024, 1024]" = torch.ops.aten.view.default(bmm_6, [1, 16, 1024, 1024]);  bmm_6 = None
    add_29: "f32[1, 16, 1024, 1024]" = torch.ops.aten.add.Tensor(view_80, expand_2);  view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:251, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_81: "f32[16, 1024, 1024]" = torch.ops.aten.view.default(add_29, [16, 1024, 1024]);  add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_3: "f32[16, 1024, 1]" = torch.ops.aten.amax.default(view_81, [-1], True)
    sub_10: "f32[16, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_81, amax_3);  view_81 = amax_3 = None
    exp_3: "f32[16, 1024, 1024]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
    sum_4: "f32[16, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_3: "f32[16, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:274, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_28: "f32[16, 1024, 1024]" = torch.ops.aten.clone.default(div_3);  div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_7: "f32[16, 1024, 64]" = torch.ops.aten.bmm.default(clone_28, view_79);  clone_28 = view_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_82: "f32[1, 16, 1024, 64]" = torch.ops.aten.view.default(bmm_7, [1, 16, 1024, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_40: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_29: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
    view_83: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_29, [1, 1024, 1024]);  clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_84: "f32[1024, 1024]" = torch.ops.aten.view.default(view_83, [1024, 1024]);  view_83 = None
    permute_41: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
    addmm_21: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg59_1, view_84, permute_41);  arg59_1 = view_84 = permute_41 = None
    view_85: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_21, [1, 1024, 1024]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:434, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_30: "f32[1, 1024, 1024]" = torch.ops.aten.clone.default(view_85);  view_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:435, code: hidden_states = residual + hidden_states
    add_30: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_28, clone_30);  add_28 = clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_7 = torch.ops.aten.var_mean.correction(add_30, [2], correction = 0, keepdim = True)
    getitem_14: "f32[1, 1024, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 1024, 1]" = var_mean_7[1];  var_mean_7 = None
    add_31: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
    rsqrt_7: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
    sub_11: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_30, getitem_15);  add_30 = getitem_15 = None
    mul_28: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
    mul_29: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_28, arg60_1);  mul_28 = arg60_1 = None
    add_32: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_29, arg61_1);  mul_29 = arg61_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_86: "f32[1024, 1024]" = torch.ops.aten.view.default(add_32, [1024, 1024])
    permute_42: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
    addmm_22: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg63_1, view_86, permute_42);  arg63_1 = view_86 = permute_42 = None
    view_87: "f32[1, 1024, 4096]" = torch.ops.aten.view.default(addmm_22, [1, 1024, 4096]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_30: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_87, 0.5)
    mul_31: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_87, 0.7071067811865476);  view_87 = None
    erf_3: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_31);  mul_31 = None
    add_33: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_32: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_30, add_33);  mul_30 = add_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:464, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_31: "f32[1, 1024, 4096]" = torch.ops.aten.clone.default(mul_32);  mul_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_88: "f32[1024, 4096]" = torch.ops.aten.view.default(clone_31, [1024, 4096]);  clone_31 = None
    permute_43: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
    addmm_23: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg65_1, view_88, permute_43);  arg65_1 = view_88 = permute_43 = None
    view_89: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_23, [1, 1024, 1024]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:466, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_32: "f32[1, 1024, 1024]" = torch.ops.aten.clone.default(view_89);  view_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:467, code: hidden_states = residual + hidden_states
    add_34: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_32, clone_32);  add_32 = clone_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_8 = torch.ops.aten.var_mean.correction(add_34, [2], correction = 0, keepdim = True)
    getitem_16: "f32[1, 1024, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 1024, 1]" = var_mean_8[1];  var_mean_8 = None
    add_35: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
    rsqrt_8: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
    sub_12: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_34, getitem_17);  add_34 = getitem_17 = None
    mul_33: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = rsqrt_8 = None
    mul_34: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_33, arg66_1);  mul_33 = arg66_1 = None
    add_36: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_34, arg67_1);  mul_34 = arg67_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_90: "f32[1024, 1024]" = torch.ops.aten.view.default(add_36, [1024, 1024])
    permute_44: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
    addmm_24: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg69_1, view_90, permute_44);  arg69_1 = view_90 = permute_44 = None
    view_91: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_24, [1, 1024, 1024]);  addmm_24 = None
    mul_35: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_91, 0.125);  view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_92: "f32[1024, 1024]" = torch.ops.aten.view.default(add_36, [1024, 1024])
    permute_45: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
    addmm_25: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg71_1, view_92, permute_45);  arg71_1 = view_92 = permute_45 = None
    view_93: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_25, [1, 1024, 1024]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_94: "f32[1, 1024, 16, 64]" = torch.ops.aten.view.default(view_93, [1, -1, 16, 64]);  view_93 = None
    permute_46: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
    clone_33: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_95: "f32[1024, 1024]" = torch.ops.aten.view.default(add_36, [1024, 1024])
    permute_47: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
    addmm_26: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg73_1, view_95, permute_47);  arg73_1 = view_95 = permute_47 = None
    view_96: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_26, [1, 1024, 1024]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_97: "f32[1, 1024, 16, 64]" = torch.ops.aten.view.default(view_96, [1, -1, 16, 64]);  view_96 = None
    permute_48: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_97, [0, 2, 1, 3]);  view_97 = None
    clone_34: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_98: "f32[1, 1024, 16, 64]" = torch.ops.aten.view.default(mul_35, [1, 1024, 16, 64]);  mul_35 = None
    permute_49: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None
    clone_35: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_99: "f32[16, 1024, 64]" = torch.ops.aten.view.default(clone_35, [16, -1, 64]);  clone_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_100: "f32[16, 1024, 64]" = torch.ops.aten.view.default(clone_33, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_101: "f32[16, 1024, 64]" = torch.ops.aten.view.default(clone_34, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_50: "f32[16, 64, 1024]" = torch.ops.aten.permute.default(view_100, [0, 2, 1]);  view_100 = None
    bmm_8: "f32[16, 1024, 1024]" = torch.ops.aten.bmm.default(view_99, permute_50);  view_99 = permute_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:250, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_102: "f32[1, 16, 1024, 1024]" = torch.ops.aten.view.default(bmm_8, [1, 16, 1024, 1024]);  bmm_8 = None
    add_37: "f32[1, 16, 1024, 1024]" = torch.ops.aten.add.Tensor(view_102, expand_2);  view_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:251, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_103: "f32[16, 1024, 1024]" = torch.ops.aten.view.default(add_37, [16, 1024, 1024]);  add_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_4: "f32[16, 1024, 1]" = torch.ops.aten.amax.default(view_103, [-1], True)
    sub_13: "f32[16, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_103, amax_4);  view_103 = amax_4 = None
    exp_4: "f32[16, 1024, 1024]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
    sum_5: "f32[16, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_4: "f32[16, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:274, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_36: "f32[16, 1024, 1024]" = torch.ops.aten.clone.default(div_4);  div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_9: "f32[16, 1024, 64]" = torch.ops.aten.bmm.default(clone_36, view_101);  clone_36 = view_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_104: "f32[1, 16, 1024, 64]" = torch.ops.aten.view.default(bmm_9, [1, 16, 1024, 64]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_51: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_37: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
    view_105: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_37, [1, 1024, 1024]);  clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_106: "f32[1024, 1024]" = torch.ops.aten.view.default(view_105, [1024, 1024]);  view_105 = None
    permute_52: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
    addmm_27: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg75_1, view_106, permute_52);  arg75_1 = view_106 = permute_52 = None
    view_107: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_27, [1, 1024, 1024]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:434, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_38: "f32[1, 1024, 1024]" = torch.ops.aten.clone.default(view_107);  view_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:435, code: hidden_states = residual + hidden_states
    add_38: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_36, clone_38);  add_36 = clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_9 = torch.ops.aten.var_mean.correction(add_38, [2], correction = 0, keepdim = True)
    getitem_18: "f32[1, 1024, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 1024, 1]" = var_mean_9[1];  var_mean_9 = None
    add_39: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
    rsqrt_9: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
    sub_14: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_38, getitem_19);  add_38 = getitem_19 = None
    mul_36: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
    mul_37: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_36, arg76_1);  mul_36 = arg76_1 = None
    add_40: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_37, arg77_1);  mul_37 = arg77_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_108: "f32[1024, 1024]" = torch.ops.aten.view.default(add_40, [1024, 1024])
    permute_53: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
    addmm_28: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg79_1, view_108, permute_53);  arg79_1 = view_108 = permute_53 = None
    view_109: "f32[1, 1024, 4096]" = torch.ops.aten.view.default(addmm_28, [1, 1024, 4096]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_38: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_109, 0.5)
    mul_39: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_109, 0.7071067811865476);  view_109 = None
    erf_4: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_39);  mul_39 = None
    add_41: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_40: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_38, add_41);  mul_38 = add_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:464, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_39: "f32[1, 1024, 4096]" = torch.ops.aten.clone.default(mul_40);  mul_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_110: "f32[1024, 4096]" = torch.ops.aten.view.default(clone_39, [1024, 4096]);  clone_39 = None
    permute_54: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
    addmm_29: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg81_1, view_110, permute_54);  arg81_1 = view_110 = permute_54 = None
    view_111: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_29, [1, 1024, 1024]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:466, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_40: "f32[1, 1024, 1024]" = torch.ops.aten.clone.default(view_111);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:467, code: hidden_states = residual + hidden_states
    add_42: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_40, clone_40);  add_40 = clone_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_10 = torch.ops.aten.var_mean.correction(add_42, [2], correction = 0, keepdim = True)
    getitem_20: "f32[1, 1024, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 1024, 1]" = var_mean_10[1];  var_mean_10 = None
    add_43: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
    rsqrt_10: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
    sub_15: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_42, getitem_21);  add_42 = getitem_21 = None
    mul_41: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = rsqrt_10 = None
    mul_42: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_41, arg82_1);  mul_41 = arg82_1 = None
    add_44: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_42, arg83_1);  mul_42 = arg83_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_112: "f32[1024, 1024]" = torch.ops.aten.view.default(add_44, [1024, 1024])
    permute_55: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
    addmm_30: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg85_1, view_112, permute_55);  arg85_1 = view_112 = permute_55 = None
    view_113: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_30, [1, 1024, 1024]);  addmm_30 = None
    mul_43: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_113, 0.125);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_114: "f32[1024, 1024]" = torch.ops.aten.view.default(add_44, [1024, 1024])
    permute_56: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
    addmm_31: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg87_1, view_114, permute_56);  arg87_1 = view_114 = permute_56 = None
    view_115: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_31, [1, 1024, 1024]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_116: "f32[1, 1024, 16, 64]" = torch.ops.aten.view.default(view_115, [1, -1, 16, 64]);  view_115 = None
    permute_57: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_116, [0, 2, 1, 3]);  view_116 = None
    clone_41: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_57, memory_format = torch.contiguous_format);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_117: "f32[1024, 1024]" = torch.ops.aten.view.default(add_44, [1024, 1024])
    permute_58: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
    addmm_32: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg89_1, view_117, permute_58);  arg89_1 = view_117 = permute_58 = None
    view_118: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_32, [1, 1024, 1024]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_119: "f32[1, 1024, 16, 64]" = torch.ops.aten.view.default(view_118, [1, -1, 16, 64]);  view_118 = None
    permute_59: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_119, [0, 2, 1, 3]);  view_119 = None
    clone_42: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_120: "f32[1, 1024, 16, 64]" = torch.ops.aten.view.default(mul_43, [1, 1024, 16, 64]);  mul_43 = None
    permute_60: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
    clone_43: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_121: "f32[16, 1024, 64]" = torch.ops.aten.view.default(clone_43, [16, -1, 64]);  clone_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_122: "f32[16, 1024, 64]" = torch.ops.aten.view.default(clone_41, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_123: "f32[16, 1024, 64]" = torch.ops.aten.view.default(clone_42, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_61: "f32[16, 64, 1024]" = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None
    bmm_10: "f32[16, 1024, 1024]" = torch.ops.aten.bmm.default(view_121, permute_61);  view_121 = permute_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:250, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_124: "f32[1, 16, 1024, 1024]" = torch.ops.aten.view.default(bmm_10, [1, 16, 1024, 1024]);  bmm_10 = None
    add_45: "f32[1, 16, 1024, 1024]" = torch.ops.aten.add.Tensor(view_124, expand_2);  view_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:251, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_125: "f32[16, 1024, 1024]" = torch.ops.aten.view.default(add_45, [16, 1024, 1024]);  add_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_5: "f32[16, 1024, 1]" = torch.ops.aten.amax.default(view_125, [-1], True)
    sub_16: "f32[16, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_125, amax_5);  view_125 = amax_5 = None
    exp_5: "f32[16, 1024, 1024]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
    sum_6: "f32[16, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_5: "f32[16, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:274, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_44: "f32[16, 1024, 1024]" = torch.ops.aten.clone.default(div_5);  div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_11: "f32[16, 1024, 64]" = torch.ops.aten.bmm.default(clone_44, view_123);  clone_44 = view_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_126: "f32[1, 16, 1024, 64]" = torch.ops.aten.view.default(bmm_11, [1, 16, 1024, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_62: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_45: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    view_127: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_45, [1, 1024, 1024]);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_128: "f32[1024, 1024]" = torch.ops.aten.view.default(view_127, [1024, 1024]);  view_127 = None
    permute_63: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
    addmm_33: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg91_1, view_128, permute_63);  arg91_1 = view_128 = permute_63 = None
    view_129: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_33, [1, 1024, 1024]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:434, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_46: "f32[1, 1024, 1024]" = torch.ops.aten.clone.default(view_129);  view_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:435, code: hidden_states = residual + hidden_states
    add_46: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_44, clone_46);  add_44 = clone_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_11 = torch.ops.aten.var_mean.correction(add_46, [2], correction = 0, keepdim = True)
    getitem_22: "f32[1, 1024, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 1024, 1]" = var_mean_11[1];  var_mean_11 = None
    add_47: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
    rsqrt_11: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
    sub_17: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_46, getitem_23);  add_46 = getitem_23 = None
    mul_44: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
    mul_45: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_44, arg92_1);  mul_44 = arg92_1 = None
    add_48: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_45, arg93_1);  mul_45 = arg93_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_130: "f32[1024, 1024]" = torch.ops.aten.view.default(add_48, [1024, 1024])
    permute_64: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
    addmm_34: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg95_1, view_130, permute_64);  arg95_1 = view_130 = permute_64 = None
    view_131: "f32[1, 1024, 4096]" = torch.ops.aten.view.default(addmm_34, [1, 1024, 4096]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_46: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_131, 0.5)
    mul_47: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_131, 0.7071067811865476);  view_131 = None
    erf_5: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
    add_49: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_48: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_46, add_49);  mul_46 = add_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:464, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_47: "f32[1, 1024, 4096]" = torch.ops.aten.clone.default(mul_48);  mul_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_132: "f32[1024, 4096]" = torch.ops.aten.view.default(clone_47, [1024, 4096]);  clone_47 = None
    permute_65: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
    addmm_35: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg97_1, view_132, permute_65);  arg97_1 = view_132 = permute_65 = None
    view_133: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_35, [1, 1024, 1024]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:466, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_48: "f32[1, 1024, 1024]" = torch.ops.aten.clone.default(view_133);  view_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:467, code: hidden_states = residual + hidden_states
    add_50: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_48, clone_48);  add_48 = clone_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_12 = torch.ops.aten.var_mean.correction(add_50, [2], correction = 0, keepdim = True)
    getitem_24: "f32[1, 1024, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 1024, 1]" = var_mean_12[1];  var_mean_12 = None
    add_51: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
    rsqrt_12: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
    sub_18: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_50, getitem_25);  add_50 = getitem_25 = None
    mul_49: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = rsqrt_12 = None
    mul_50: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_49, arg98_1);  mul_49 = arg98_1 = None
    add_52: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_50, arg99_1);  mul_50 = arg99_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_134: "f32[1024, 1024]" = torch.ops.aten.view.default(add_52, [1024, 1024])
    permute_66: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
    addmm_36: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg101_1, view_134, permute_66);  arg101_1 = view_134 = permute_66 = None
    view_135: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_36, [1, 1024, 1024]);  addmm_36 = None
    mul_51: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_135, 0.125);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_136: "f32[1024, 1024]" = torch.ops.aten.view.default(add_52, [1024, 1024])
    permute_67: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
    addmm_37: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg103_1, view_136, permute_67);  arg103_1 = view_136 = permute_67 = None
    view_137: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_37, [1, 1024, 1024]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_138: "f32[1, 1024, 16, 64]" = torch.ops.aten.view.default(view_137, [1, -1, 16, 64]);  view_137 = None
    permute_68: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None
    clone_49: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_139: "f32[1024, 1024]" = torch.ops.aten.view.default(add_52, [1024, 1024])
    permute_69: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
    addmm_38: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg105_1, view_139, permute_69);  arg105_1 = view_139 = permute_69 = None
    view_140: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_38, [1, 1024, 1024]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_141: "f32[1, 1024, 16, 64]" = torch.ops.aten.view.default(view_140, [1, -1, 16, 64]);  view_140 = None
    permute_70: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_141, [0, 2, 1, 3]);  view_141 = None
    clone_50: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_142: "f32[1, 1024, 16, 64]" = torch.ops.aten.view.default(mul_51, [1, 1024, 16, 64]);  mul_51 = None
    permute_71: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_142, [0, 2, 1, 3]);  view_142 = None
    clone_51: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_143: "f32[16, 1024, 64]" = torch.ops.aten.view.default(clone_51, [16, -1, 64]);  clone_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_144: "f32[16, 1024, 64]" = torch.ops.aten.view.default(clone_49, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_145: "f32[16, 1024, 64]" = torch.ops.aten.view.default(clone_50, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_72: "f32[16, 64, 1024]" = torch.ops.aten.permute.default(view_144, [0, 2, 1]);  view_144 = None
    bmm_12: "f32[16, 1024, 1024]" = torch.ops.aten.bmm.default(view_143, permute_72);  view_143 = permute_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:250, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_146: "f32[1, 16, 1024, 1024]" = torch.ops.aten.view.default(bmm_12, [1, 16, 1024, 1024]);  bmm_12 = None
    add_53: "f32[1, 16, 1024, 1024]" = torch.ops.aten.add.Tensor(view_146, expand_2);  view_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:251, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_147: "f32[16, 1024, 1024]" = torch.ops.aten.view.default(add_53, [16, 1024, 1024]);  add_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_6: "f32[16, 1024, 1]" = torch.ops.aten.amax.default(view_147, [-1], True)
    sub_19: "f32[16, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_147, amax_6);  view_147 = amax_6 = None
    exp_6: "f32[16, 1024, 1024]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
    sum_7: "f32[16, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_6: "f32[16, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:274, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_52: "f32[16, 1024, 1024]" = torch.ops.aten.clone.default(div_6);  div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_13: "f32[16, 1024, 64]" = torch.ops.aten.bmm.default(clone_52, view_145);  clone_52 = view_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_148: "f32[1, 16, 1024, 64]" = torch.ops.aten.view.default(bmm_13, [1, 16, 1024, 64]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_73: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_53: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
    view_149: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_53, [1, 1024, 1024]);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_150: "f32[1024, 1024]" = torch.ops.aten.view.default(view_149, [1024, 1024]);  view_149 = None
    permute_74: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
    addmm_39: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg107_1, view_150, permute_74);  arg107_1 = view_150 = permute_74 = None
    view_151: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_39, [1, 1024, 1024]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:434, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_54: "f32[1, 1024, 1024]" = torch.ops.aten.clone.default(view_151);  view_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:435, code: hidden_states = residual + hidden_states
    add_54: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_52, clone_54);  add_52 = clone_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_13 = torch.ops.aten.var_mean.correction(add_54, [2], correction = 0, keepdim = True)
    getitem_26: "f32[1, 1024, 1]" = var_mean_13[0]
    getitem_27: "f32[1, 1024, 1]" = var_mean_13[1];  var_mean_13 = None
    add_55: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
    rsqrt_13: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
    sub_20: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_54, getitem_27);  add_54 = getitem_27 = None
    mul_52: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = rsqrt_13 = None
    mul_53: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_52, arg108_1);  mul_52 = arg108_1 = None
    add_56: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_53, arg109_1);  mul_53 = arg109_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_152: "f32[1024, 1024]" = torch.ops.aten.view.default(add_56, [1024, 1024])
    permute_75: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
    addmm_40: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg111_1, view_152, permute_75);  arg111_1 = view_152 = permute_75 = None
    view_153: "f32[1, 1024, 4096]" = torch.ops.aten.view.default(addmm_40, [1, 1024, 4096]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_54: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_153, 0.5)
    mul_55: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_153, 0.7071067811865476);  view_153 = None
    erf_6: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
    add_57: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_56: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_54, add_57);  mul_54 = add_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:464, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_55: "f32[1, 1024, 4096]" = torch.ops.aten.clone.default(mul_56);  mul_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_154: "f32[1024, 4096]" = torch.ops.aten.view.default(clone_55, [1024, 4096]);  clone_55 = None
    permute_76: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
    addmm_41: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg113_1, view_154, permute_76);  arg113_1 = view_154 = permute_76 = None
    view_155: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_41, [1, 1024, 1024]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:466, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_56: "f32[1, 1024, 1024]" = torch.ops.aten.clone.default(view_155);  view_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:467, code: hidden_states = residual + hidden_states
    add_58: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_56, clone_56);  add_56 = clone_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_14 = torch.ops.aten.var_mean.correction(add_58, [2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 1024, 1]" = var_mean_14[0]
    getitem_29: "f32[1, 1024, 1]" = var_mean_14[1];  var_mean_14 = None
    add_59: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
    rsqrt_14: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
    sub_21: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_58, getitem_29);  add_58 = getitem_29 = None
    mul_57: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = rsqrt_14 = None
    mul_58: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_57, arg114_1);  mul_57 = arg114_1 = None
    add_60: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_58, arg115_1);  mul_58 = arg115_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_156: "f32[1024, 1024]" = torch.ops.aten.view.default(add_60, [1024, 1024])
    permute_77: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
    addmm_42: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg117_1, view_156, permute_77);  arg117_1 = view_156 = permute_77 = None
    view_157: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_42, [1, 1024, 1024]);  addmm_42 = None
    mul_59: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_157, 0.125);  view_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_158: "f32[1024, 1024]" = torch.ops.aten.view.default(add_60, [1024, 1024])
    permute_78: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
    addmm_43: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg119_1, view_158, permute_78);  arg119_1 = view_158 = permute_78 = None
    view_159: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_43, [1, 1024, 1024]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_160: "f32[1, 1024, 16, 64]" = torch.ops.aten.view.default(view_159, [1, -1, 16, 64]);  view_159 = None
    permute_79: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_160, [0, 2, 1, 3]);  view_160 = None
    clone_57: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_161: "f32[1024, 1024]" = torch.ops.aten.view.default(add_60, [1024, 1024])
    permute_80: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
    addmm_44: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg121_1, view_161, permute_80);  arg121_1 = view_161 = permute_80 = None
    view_162: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_44, [1, 1024, 1024]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_163: "f32[1, 1024, 16, 64]" = torch.ops.aten.view.default(view_162, [1, -1, 16, 64]);  view_162 = None
    permute_81: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_163, [0, 2, 1, 3]);  view_163 = None
    clone_58: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_81, memory_format = torch.contiguous_format);  permute_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_164: "f32[1, 1024, 16, 64]" = torch.ops.aten.view.default(mul_59, [1, 1024, 16, 64]);  mul_59 = None
    permute_82: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_164, [0, 2, 1, 3]);  view_164 = None
    clone_59: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_165: "f32[16, 1024, 64]" = torch.ops.aten.view.default(clone_59, [16, -1, 64]);  clone_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_166: "f32[16, 1024, 64]" = torch.ops.aten.view.default(clone_57, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_167: "f32[16, 1024, 64]" = torch.ops.aten.view.default(clone_58, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_83: "f32[16, 64, 1024]" = torch.ops.aten.permute.default(view_166, [0, 2, 1]);  view_166 = None
    bmm_14: "f32[16, 1024, 1024]" = torch.ops.aten.bmm.default(view_165, permute_83);  view_165 = permute_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:250, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_168: "f32[1, 16, 1024, 1024]" = torch.ops.aten.view.default(bmm_14, [1, 16, 1024, 1024]);  bmm_14 = None
    add_61: "f32[1, 16, 1024, 1024]" = torch.ops.aten.add.Tensor(view_168, expand_2);  view_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:251, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_169: "f32[16, 1024, 1024]" = torch.ops.aten.view.default(add_61, [16, 1024, 1024]);  add_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_7: "f32[16, 1024, 1]" = torch.ops.aten.amax.default(view_169, [-1], True)
    sub_22: "f32[16, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_169, amax_7);  view_169 = amax_7 = None
    exp_7: "f32[16, 1024, 1024]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
    sum_8: "f32[16, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_7: "f32[16, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:274, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_60: "f32[16, 1024, 1024]" = torch.ops.aten.clone.default(div_7);  div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_15: "f32[16, 1024, 64]" = torch.ops.aten.bmm.default(clone_60, view_167);  clone_60 = view_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_170: "f32[1, 16, 1024, 64]" = torch.ops.aten.view.default(bmm_15, [1, 16, 1024, 64]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_84: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_170, [0, 2, 1, 3]);  view_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_61: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
    view_171: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_61, [1, 1024, 1024]);  clone_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_172: "f32[1024, 1024]" = torch.ops.aten.view.default(view_171, [1024, 1024]);  view_171 = None
    permute_85: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
    addmm_45: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg123_1, view_172, permute_85);  arg123_1 = view_172 = permute_85 = None
    view_173: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_45, [1, 1024, 1024]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:434, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_62: "f32[1, 1024, 1024]" = torch.ops.aten.clone.default(view_173);  view_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:435, code: hidden_states = residual + hidden_states
    add_62: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_60, clone_62);  add_60 = clone_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_15 = torch.ops.aten.var_mean.correction(add_62, [2], correction = 0, keepdim = True)
    getitem_30: "f32[1, 1024, 1]" = var_mean_15[0]
    getitem_31: "f32[1, 1024, 1]" = var_mean_15[1];  var_mean_15 = None
    add_63: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
    rsqrt_15: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
    sub_23: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_62, getitem_31);  add_62 = getitem_31 = None
    mul_60: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = rsqrt_15 = None
    mul_61: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_60, arg124_1);  mul_60 = arg124_1 = None
    add_64: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_61, arg125_1);  mul_61 = arg125_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_174: "f32[1024, 1024]" = torch.ops.aten.view.default(add_64, [1024, 1024])
    permute_86: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
    addmm_46: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg127_1, view_174, permute_86);  arg127_1 = view_174 = permute_86 = None
    view_175: "f32[1, 1024, 4096]" = torch.ops.aten.view.default(addmm_46, [1, 1024, 4096]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_62: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_175, 0.5)
    mul_63: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_175, 0.7071067811865476);  view_175 = None
    erf_7: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_63);  mul_63 = None
    add_65: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_64: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_62, add_65);  mul_62 = add_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:464, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_63: "f32[1, 1024, 4096]" = torch.ops.aten.clone.default(mul_64);  mul_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_176: "f32[1024, 4096]" = torch.ops.aten.view.default(clone_63, [1024, 4096]);  clone_63 = None
    permute_87: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
    addmm_47: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg129_1, view_176, permute_87);  arg129_1 = view_176 = permute_87 = None
    view_177: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_47, [1, 1024, 1024]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:466, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_64: "f32[1, 1024, 1024]" = torch.ops.aten.clone.default(view_177);  view_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:467, code: hidden_states = residual + hidden_states
    add_66: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_64, clone_64);  add_64 = clone_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_16 = torch.ops.aten.var_mean.correction(add_66, [2], correction = 0, keepdim = True)
    getitem_32: "f32[1, 1024, 1]" = var_mean_16[0]
    getitem_33: "f32[1, 1024, 1]" = var_mean_16[1];  var_mean_16 = None
    add_67: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
    rsqrt_16: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
    sub_24: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_66, getitem_33);  add_66 = getitem_33 = None
    mul_65: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = rsqrt_16 = None
    mul_66: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_65, arg130_1);  mul_65 = arg130_1 = None
    add_68: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_66, arg131_1);  mul_66 = arg131_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_178: "f32[1024, 1024]" = torch.ops.aten.view.default(add_68, [1024, 1024])
    permute_88: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
    addmm_48: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg133_1, view_178, permute_88);  arg133_1 = view_178 = permute_88 = None
    view_179: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_48, [1, 1024, 1024]);  addmm_48 = None
    mul_67: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_179, 0.125);  view_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_180: "f32[1024, 1024]" = torch.ops.aten.view.default(add_68, [1024, 1024])
    permute_89: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg134_1, [1, 0]);  arg134_1 = None
    addmm_49: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg135_1, view_180, permute_89);  arg135_1 = view_180 = permute_89 = None
    view_181: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_49, [1, 1024, 1024]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_182: "f32[1, 1024, 16, 64]" = torch.ops.aten.view.default(view_181, [1, -1, 16, 64]);  view_181 = None
    permute_90: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None
    clone_65: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_183: "f32[1024, 1024]" = torch.ops.aten.view.default(add_68, [1024, 1024])
    permute_91: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
    addmm_50: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg137_1, view_183, permute_91);  arg137_1 = view_183 = permute_91 = None
    view_184: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_50, [1, 1024, 1024]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_185: "f32[1, 1024, 16, 64]" = torch.ops.aten.view.default(view_184, [1, -1, 16, 64]);  view_184 = None
    permute_92: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None
    clone_66: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_186: "f32[1, 1024, 16, 64]" = torch.ops.aten.view.default(mul_67, [1, 1024, 16, 64]);  mul_67 = None
    permute_93: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
    clone_67: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_187: "f32[16, 1024, 64]" = torch.ops.aten.view.default(clone_67, [16, -1, 64]);  clone_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_188: "f32[16, 1024, 64]" = torch.ops.aten.view.default(clone_65, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_189: "f32[16, 1024, 64]" = torch.ops.aten.view.default(clone_66, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_94: "f32[16, 64, 1024]" = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None
    bmm_16: "f32[16, 1024, 1024]" = torch.ops.aten.bmm.default(view_187, permute_94);  view_187 = permute_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:250, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_190: "f32[1, 16, 1024, 1024]" = torch.ops.aten.view.default(bmm_16, [1, 16, 1024, 1024]);  bmm_16 = None
    add_69: "f32[1, 16, 1024, 1024]" = torch.ops.aten.add.Tensor(view_190, expand_2);  view_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:251, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_191: "f32[16, 1024, 1024]" = torch.ops.aten.view.default(add_69, [16, 1024, 1024]);  add_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_8: "f32[16, 1024, 1]" = torch.ops.aten.amax.default(view_191, [-1], True)
    sub_25: "f32[16, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_191, amax_8);  view_191 = amax_8 = None
    exp_8: "f32[16, 1024, 1024]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
    sum_9: "f32[16, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_8: "f32[16, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:274, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_68: "f32[16, 1024, 1024]" = torch.ops.aten.clone.default(div_8);  div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_17: "f32[16, 1024, 64]" = torch.ops.aten.bmm.default(clone_68, view_189);  clone_68 = view_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_192: "f32[1, 16, 1024, 64]" = torch.ops.aten.view.default(bmm_17, [1, 16, 1024, 64]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_95: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_69: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
    view_193: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_69, [1, 1024, 1024]);  clone_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_194: "f32[1024, 1024]" = torch.ops.aten.view.default(view_193, [1024, 1024]);  view_193 = None
    permute_96: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
    addmm_51: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg139_1, view_194, permute_96);  arg139_1 = view_194 = permute_96 = None
    view_195: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_51, [1, 1024, 1024]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:434, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_70: "f32[1, 1024, 1024]" = torch.ops.aten.clone.default(view_195);  view_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:435, code: hidden_states = residual + hidden_states
    add_70: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_68, clone_70);  add_68 = clone_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_17 = torch.ops.aten.var_mean.correction(add_70, [2], correction = 0, keepdim = True)
    getitem_34: "f32[1, 1024, 1]" = var_mean_17[0]
    getitem_35: "f32[1, 1024, 1]" = var_mean_17[1];  var_mean_17 = None
    add_71: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
    rsqrt_17: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
    sub_26: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_70, getitem_35);  add_70 = getitem_35 = None
    mul_68: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = rsqrt_17 = None
    mul_69: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_68, arg140_1);  mul_68 = arg140_1 = None
    add_72: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_69, arg141_1);  mul_69 = arg141_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_196: "f32[1024, 1024]" = torch.ops.aten.view.default(add_72, [1024, 1024])
    permute_97: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
    addmm_52: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg143_1, view_196, permute_97);  arg143_1 = view_196 = permute_97 = None
    view_197: "f32[1, 1024, 4096]" = torch.ops.aten.view.default(addmm_52, [1, 1024, 4096]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_70: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_197, 0.5)
    mul_71: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_197, 0.7071067811865476);  view_197 = None
    erf_8: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_71);  mul_71 = None
    add_73: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_72: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_70, add_73);  mul_70 = add_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:464, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_71: "f32[1, 1024, 4096]" = torch.ops.aten.clone.default(mul_72);  mul_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_198: "f32[1024, 4096]" = torch.ops.aten.view.default(clone_71, [1024, 4096]);  clone_71 = None
    permute_98: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
    addmm_53: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg145_1, view_198, permute_98);  arg145_1 = view_198 = permute_98 = None
    view_199: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_53, [1, 1024, 1024]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:466, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_72: "f32[1, 1024, 1024]" = torch.ops.aten.clone.default(view_199);  view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:467, code: hidden_states = residual + hidden_states
    add_74: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_72, clone_72);  add_72 = clone_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_18 = torch.ops.aten.var_mean.correction(add_74, [2], correction = 0, keepdim = True)
    getitem_36: "f32[1, 1024, 1]" = var_mean_18[0]
    getitem_37: "f32[1, 1024, 1]" = var_mean_18[1];  var_mean_18 = None
    add_75: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
    rsqrt_18: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
    sub_27: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_74, getitem_37);  add_74 = getitem_37 = None
    mul_73: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = rsqrt_18 = None
    mul_74: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_73, arg146_1);  mul_73 = arg146_1 = None
    add_76: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_74, arg147_1);  mul_74 = arg147_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_200: "f32[1024, 1024]" = torch.ops.aten.view.default(add_76, [1024, 1024])
    permute_99: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
    addmm_54: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg149_1, view_200, permute_99);  arg149_1 = view_200 = permute_99 = None
    view_201: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_54, [1, 1024, 1024]);  addmm_54 = None
    mul_75: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_201, 0.125);  view_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_202: "f32[1024, 1024]" = torch.ops.aten.view.default(add_76, [1024, 1024])
    permute_100: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
    addmm_55: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg151_1, view_202, permute_100);  arg151_1 = view_202 = permute_100 = None
    view_203: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_55, [1, 1024, 1024]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_204: "f32[1, 1024, 16, 64]" = torch.ops.aten.view.default(view_203, [1, -1, 16, 64]);  view_203 = None
    permute_101: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_204, [0, 2, 1, 3]);  view_204 = None
    clone_73: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_101, memory_format = torch.contiguous_format);  permute_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_205: "f32[1024, 1024]" = torch.ops.aten.view.default(add_76, [1024, 1024])
    permute_102: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
    addmm_56: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg153_1, view_205, permute_102);  arg153_1 = view_205 = permute_102 = None
    view_206: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_56, [1, 1024, 1024]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_207: "f32[1, 1024, 16, 64]" = torch.ops.aten.view.default(view_206, [1, -1, 16, 64]);  view_206 = None
    permute_103: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_207, [0, 2, 1, 3]);  view_207 = None
    clone_74: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_103, memory_format = torch.contiguous_format);  permute_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_208: "f32[1, 1024, 16, 64]" = torch.ops.aten.view.default(mul_75, [1, 1024, 16, 64]);  mul_75 = None
    permute_104: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None
    clone_75: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_209: "f32[16, 1024, 64]" = torch.ops.aten.view.default(clone_75, [16, -1, 64]);  clone_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_210: "f32[16, 1024, 64]" = torch.ops.aten.view.default(clone_73, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_211: "f32[16, 1024, 64]" = torch.ops.aten.view.default(clone_74, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_105: "f32[16, 64, 1024]" = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None
    bmm_18: "f32[16, 1024, 1024]" = torch.ops.aten.bmm.default(view_209, permute_105);  view_209 = permute_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:250, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_212: "f32[1, 16, 1024, 1024]" = torch.ops.aten.view.default(bmm_18, [1, 16, 1024, 1024]);  bmm_18 = None
    add_77: "f32[1, 16, 1024, 1024]" = torch.ops.aten.add.Tensor(view_212, expand_2);  view_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:251, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_213: "f32[16, 1024, 1024]" = torch.ops.aten.view.default(add_77, [16, 1024, 1024]);  add_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_9: "f32[16, 1024, 1]" = torch.ops.aten.amax.default(view_213, [-1], True)
    sub_28: "f32[16, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_213, amax_9);  view_213 = amax_9 = None
    exp_9: "f32[16, 1024, 1024]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
    sum_10: "f32[16, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_9: "f32[16, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:274, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_76: "f32[16, 1024, 1024]" = torch.ops.aten.clone.default(div_9);  div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_19: "f32[16, 1024, 64]" = torch.ops.aten.bmm.default(clone_76, view_211);  clone_76 = view_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_214: "f32[1, 16, 1024, 64]" = torch.ops.aten.view.default(bmm_19, [1, 16, 1024, 64]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_106: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_214, [0, 2, 1, 3]);  view_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_77: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
    view_215: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_77, [1, 1024, 1024]);  clone_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_216: "f32[1024, 1024]" = torch.ops.aten.view.default(view_215, [1024, 1024]);  view_215 = None
    permute_107: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
    addmm_57: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg155_1, view_216, permute_107);  arg155_1 = view_216 = permute_107 = None
    view_217: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_57, [1, 1024, 1024]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:434, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_78: "f32[1, 1024, 1024]" = torch.ops.aten.clone.default(view_217);  view_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:435, code: hidden_states = residual + hidden_states
    add_78: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_76, clone_78);  add_76 = clone_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_19 = torch.ops.aten.var_mean.correction(add_78, [2], correction = 0, keepdim = True)
    getitem_38: "f32[1, 1024, 1]" = var_mean_19[0]
    getitem_39: "f32[1, 1024, 1]" = var_mean_19[1];  var_mean_19 = None
    add_79: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
    rsqrt_19: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
    sub_29: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_78, getitem_39);  add_78 = getitem_39 = None
    mul_76: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = rsqrt_19 = None
    mul_77: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_76, arg156_1);  mul_76 = arg156_1 = None
    add_80: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_77, arg157_1);  mul_77 = arg157_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_218: "f32[1024, 1024]" = torch.ops.aten.view.default(add_80, [1024, 1024])
    permute_108: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg158_1, [1, 0]);  arg158_1 = None
    addmm_58: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg159_1, view_218, permute_108);  arg159_1 = view_218 = permute_108 = None
    view_219: "f32[1, 1024, 4096]" = torch.ops.aten.view.default(addmm_58, [1, 1024, 4096]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_78: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_219, 0.5)
    mul_79: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_219, 0.7071067811865476);  view_219 = None
    erf_9: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_79);  mul_79 = None
    add_81: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_80: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_78, add_81);  mul_78 = add_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:464, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_79: "f32[1, 1024, 4096]" = torch.ops.aten.clone.default(mul_80);  mul_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_220: "f32[1024, 4096]" = torch.ops.aten.view.default(clone_79, [1024, 4096]);  clone_79 = None
    permute_109: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
    addmm_59: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg161_1, view_220, permute_109);  arg161_1 = view_220 = permute_109 = None
    view_221: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_59, [1, 1024, 1024]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:466, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_80: "f32[1, 1024, 1024]" = torch.ops.aten.clone.default(view_221);  view_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:467, code: hidden_states = residual + hidden_states
    add_82: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_80, clone_80);  add_80 = clone_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_20 = torch.ops.aten.var_mean.correction(add_82, [2], correction = 0, keepdim = True)
    getitem_40: "f32[1, 1024, 1]" = var_mean_20[0]
    getitem_41: "f32[1, 1024, 1]" = var_mean_20[1];  var_mean_20 = None
    add_83: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
    rsqrt_20: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
    sub_30: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_82, getitem_41);  add_82 = getitem_41 = None
    mul_81: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = rsqrt_20 = None
    mul_82: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_81, arg162_1);  mul_81 = arg162_1 = None
    add_84: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_82, arg163_1);  mul_82 = arg163_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_222: "f32[1024, 1024]" = torch.ops.aten.view.default(add_84, [1024, 1024])
    permute_110: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
    addmm_60: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg165_1, view_222, permute_110);  arg165_1 = view_222 = permute_110 = None
    view_223: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_60, [1, 1024, 1024]);  addmm_60 = None
    mul_83: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_223, 0.125);  view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_224: "f32[1024, 1024]" = torch.ops.aten.view.default(add_84, [1024, 1024])
    permute_111: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg166_1, [1, 0]);  arg166_1 = None
    addmm_61: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg167_1, view_224, permute_111);  arg167_1 = view_224 = permute_111 = None
    view_225: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_61, [1, 1024, 1024]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_226: "f32[1, 1024, 16, 64]" = torch.ops.aten.view.default(view_225, [1, -1, 16, 64]);  view_225 = None
    permute_112: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
    clone_81: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_112, memory_format = torch.contiguous_format);  permute_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_227: "f32[1024, 1024]" = torch.ops.aten.view.default(add_84, [1024, 1024])
    permute_113: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
    addmm_62: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg169_1, view_227, permute_113);  arg169_1 = view_227 = permute_113 = None
    view_228: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_62, [1, 1024, 1024]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_229: "f32[1, 1024, 16, 64]" = torch.ops.aten.view.default(view_228, [1, -1, 16, 64]);  view_228 = None
    permute_114: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_229, [0, 2, 1, 3]);  view_229 = None
    clone_82: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_230: "f32[1, 1024, 16, 64]" = torch.ops.aten.view.default(mul_83, [1, 1024, 16, 64]);  mul_83 = None
    permute_115: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_230, [0, 2, 1, 3]);  view_230 = None
    clone_83: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_231: "f32[16, 1024, 64]" = torch.ops.aten.view.default(clone_83, [16, -1, 64]);  clone_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_232: "f32[16, 1024, 64]" = torch.ops.aten.view.default(clone_81, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_233: "f32[16, 1024, 64]" = torch.ops.aten.view.default(clone_82, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_116: "f32[16, 64, 1024]" = torch.ops.aten.permute.default(view_232, [0, 2, 1]);  view_232 = None
    bmm_20: "f32[16, 1024, 1024]" = torch.ops.aten.bmm.default(view_231, permute_116);  view_231 = permute_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:250, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_234: "f32[1, 16, 1024, 1024]" = torch.ops.aten.view.default(bmm_20, [1, 16, 1024, 1024]);  bmm_20 = None
    add_85: "f32[1, 16, 1024, 1024]" = torch.ops.aten.add.Tensor(view_234, expand_2);  view_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:251, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_235: "f32[16, 1024, 1024]" = torch.ops.aten.view.default(add_85, [16, 1024, 1024]);  add_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_10: "f32[16, 1024, 1]" = torch.ops.aten.amax.default(view_235, [-1], True)
    sub_31: "f32[16, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_235, amax_10);  view_235 = amax_10 = None
    exp_10: "f32[16, 1024, 1024]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
    sum_11: "f32[16, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_10: "f32[16, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:274, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_84: "f32[16, 1024, 1024]" = torch.ops.aten.clone.default(div_10);  div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_21: "f32[16, 1024, 64]" = torch.ops.aten.bmm.default(clone_84, view_233);  clone_84 = view_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_236: "f32[1, 16, 1024, 64]" = torch.ops.aten.view.default(bmm_21, [1, 16, 1024, 64]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_117: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_236, [0, 2, 1, 3]);  view_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_85: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
    view_237: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_85, [1, 1024, 1024]);  clone_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_238: "f32[1024, 1024]" = torch.ops.aten.view.default(view_237, [1024, 1024]);  view_237 = None
    permute_118: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
    addmm_63: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg171_1, view_238, permute_118);  arg171_1 = view_238 = permute_118 = None
    view_239: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_63, [1, 1024, 1024]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:434, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_86: "f32[1, 1024, 1024]" = torch.ops.aten.clone.default(view_239);  view_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:435, code: hidden_states = residual + hidden_states
    add_86: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_84, clone_86);  add_84 = clone_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_21 = torch.ops.aten.var_mean.correction(add_86, [2], correction = 0, keepdim = True)
    getitem_42: "f32[1, 1024, 1]" = var_mean_21[0]
    getitem_43: "f32[1, 1024, 1]" = var_mean_21[1];  var_mean_21 = None
    add_87: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
    rsqrt_21: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
    sub_32: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_86, getitem_43);  add_86 = getitem_43 = None
    mul_84: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = rsqrt_21 = None
    mul_85: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_84, arg172_1);  mul_84 = arg172_1 = None
    add_88: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_85, arg173_1);  mul_85 = arg173_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_240: "f32[1024, 1024]" = torch.ops.aten.view.default(add_88, [1024, 1024])
    permute_119: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg174_1, [1, 0]);  arg174_1 = None
    addmm_64: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg175_1, view_240, permute_119);  arg175_1 = view_240 = permute_119 = None
    view_241: "f32[1, 1024, 4096]" = torch.ops.aten.view.default(addmm_64, [1, 1024, 4096]);  addmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_86: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_241, 0.5)
    mul_87: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_241, 0.7071067811865476);  view_241 = None
    erf_10: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_87);  mul_87 = None
    add_89: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_88: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_86, add_89);  mul_86 = add_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:464, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_87: "f32[1, 1024, 4096]" = torch.ops.aten.clone.default(mul_88);  mul_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_242: "f32[1024, 4096]" = torch.ops.aten.view.default(clone_87, [1024, 4096]);  clone_87 = None
    permute_120: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
    addmm_65: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg177_1, view_242, permute_120);  arg177_1 = view_242 = permute_120 = None
    view_243: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_65, [1, 1024, 1024]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:466, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_88: "f32[1, 1024, 1024]" = torch.ops.aten.clone.default(view_243);  view_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:467, code: hidden_states = residual + hidden_states
    add_90: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_88, clone_88);  add_88 = clone_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_22 = torch.ops.aten.var_mean.correction(add_90, [2], correction = 0, keepdim = True)
    getitem_44: "f32[1, 1024, 1]" = var_mean_22[0]
    getitem_45: "f32[1, 1024, 1]" = var_mean_22[1];  var_mean_22 = None
    add_91: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
    rsqrt_22: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
    sub_33: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_90, getitem_45);  add_90 = getitem_45 = None
    mul_89: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = rsqrt_22 = None
    mul_90: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_89, arg178_1);  mul_89 = arg178_1 = None
    add_92: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_90, arg179_1);  mul_90 = arg179_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:193, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_244: "f32[1024, 1024]" = torch.ops.aten.view.default(add_92, [1024, 1024])
    permute_121: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg180_1, [1, 0]);  arg180_1 = None
    addmm_66: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg181_1, view_244, permute_121);  arg181_1 = view_244 = permute_121 = None
    view_245: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_66, [1, 1024, 1024]);  addmm_66 = None
    mul_91: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_245, 0.125);  view_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:218, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_246: "f32[1024, 1024]" = torch.ops.aten.view.default(add_92, [1024, 1024])
    permute_122: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg182_1, [1, 0]);  arg182_1 = None
    addmm_67: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg183_1, view_246, permute_122);  arg183_1 = view_246 = permute_122 = None
    view_247: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_67, [1, 1024, 1024]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_248: "f32[1, 1024, 16, 64]" = torch.ops.aten.view.default(view_247, [1, -1, 16, 64]);  view_247 = None
    permute_123: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_248, [0, 2, 1, 3]);  view_248 = None
    clone_89: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_123, memory_format = torch.contiguous_format);  permute_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:219, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_249: "f32[1024, 1024]" = torch.ops.aten.view.default(add_92, [1024, 1024])
    permute_124: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
    addmm_68: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg185_1, view_249, permute_124);  arg185_1 = view_249 = permute_124 = None
    view_250: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_68, [1, 1024, 1024]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_251: "f32[1, 1024, 16, 64]" = torch.ops.aten.view.default(view_250, [1, -1, 16, 64]);  view_250 = None
    permute_125: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_251, [0, 2, 1, 3]);  view_251 = None
    clone_90: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:173, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_252: "f32[1, 1024, 16, 64]" = torch.ops.aten.view.default(mul_91, [1, 1024, 16, 64]);  mul_91 = None
    permute_126: "f32[1, 16, 1024, 64]" = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None
    clone_91: "f32[1, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:232, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_253: "f32[16, 1024, 64]" = torch.ops.aten.view.default(clone_91, [16, -1, 64]);  clone_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:233, code: key_states = key_states.reshape(*proj_shape)
    view_254: "f32[16, 1024, 64]" = torch.ops.aten.view.default(clone_89, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:234, code: value_states = value_states.reshape(*proj_shape)
    view_255: "f32[16, 1024, 64]" = torch.ops.aten.view.default(clone_90, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:237, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_127: "f32[16, 64, 1024]" = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None
    bmm_22: "f32[16, 1024, 1024]" = torch.ops.aten.bmm.default(view_253, permute_127);  view_253 = permute_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:250, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_256: "f32[1, 16, 1024, 1024]" = torch.ops.aten.view.default(bmm_22, [1, 16, 1024, 1024]);  bmm_22 = None
    add_93: "f32[1, 16, 1024, 1024]" = torch.ops.aten.add.Tensor(view_256, expand_2);  view_256 = expand_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:251, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_257: "f32[16, 1024, 1024]" = torch.ops.aten.view.default(add_93, [16, 1024, 1024]);  add_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:253, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_11: "f32[16, 1024, 1]" = torch.ops.aten.amax.default(view_257, [-1], True)
    sub_34: "f32[16, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_257, amax_11);  view_257 = amax_11 = None
    exp_11: "f32[16, 1024, 1024]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
    sum_12: "f32[16, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_11: "f32[16, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:274, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_92: "f32[16, 1024, 1024]" = torch.ops.aten.clone.default(div_11);  div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:276, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_23: "f32[16, 1024, 64]" = torch.ops.aten.bmm.default(clone_92, view_255);  clone_92 = view_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:284, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_258: "f32[1, 16, 1024, 64]" = torch.ops.aten.view.default(bmm_23, [1, 16, 1024, 64]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:285, code: attn_output = attn_output.transpose(1, 2)
    permute_128: "f32[1, 1024, 16, 64]" = torch.ops.aten.permute.default(view_258, [0, 2, 1, 3]);  view_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:289, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_93: "f32[1, 1024, 16, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
    view_259: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(clone_93, [1, 1024, 1024]);  clone_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:291, code: attn_output = self.out_proj(attn_output)
    view_260: "f32[1024, 1024]" = torch.ops.aten.view.default(view_259, [1024, 1024]);  view_259 = None
    permute_129: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg186_1, [1, 0]);  arg186_1 = None
    addmm_69: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg187_1, view_260, permute_129);  arg187_1 = view_260 = permute_129 = None
    view_261: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_69, [1, 1024, 1024]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:434, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_94: "f32[1, 1024, 1024]" = torch.ops.aten.clone.default(view_261);  view_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:435, code: hidden_states = residual + hidden_states
    add_94: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_92, clone_94);  add_92 = clone_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:436, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_23 = torch.ops.aten.var_mean.correction(add_94, [2], correction = 0, keepdim = True)
    getitem_46: "f32[1, 1024, 1]" = var_mean_23[0]
    getitem_47: "f32[1, 1024, 1]" = var_mean_23[1];  var_mean_23 = None
    add_95: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
    rsqrt_23: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
    sub_35: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_94, getitem_47);  add_94 = getitem_47 = None
    mul_92: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = rsqrt_23 = None
    mul_93: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_92, arg188_1);  mul_92 = arg188_1 = None
    add_96: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_93, arg189_1);  mul_93 = arg189_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:463, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_262: "f32[1024, 1024]" = torch.ops.aten.view.default(add_96, [1024, 1024])
    permute_130: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg190_1, [1, 0]);  arg190_1 = None
    addmm_70: "f32[1024, 4096]" = torch.ops.aten.addmm.default(arg191_1, view_262, permute_130);  arg191_1 = view_262 = permute_130 = None
    view_263: "f32[1, 1024, 4096]" = torch.ops.aten.view.default(addmm_70, [1, 1024, 4096]);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_94: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_263, 0.5)
    mul_95: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_263, 0.7071067811865476);  view_263 = None
    erf_11: "f32[1, 1024, 4096]" = torch.ops.aten.erf.default(mul_95);  mul_95 = None
    add_97: "f32[1, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_96: "f32[1, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_94, add_97);  mul_94 = add_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:464, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_95: "f32[1, 1024, 4096]" = torch.ops.aten.clone.default(mul_96);  mul_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:465, code: hidden_states = self.fc2(hidden_states)
    view_264: "f32[1024, 4096]" = torch.ops.aten.view.default(clone_95, [1024, 4096]);  clone_95 = None
    permute_131: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg192_1, [1, 0]);  arg192_1 = None
    addmm_71: "f32[1024, 1024]" = torch.ops.aten.addmm.default(arg193_1, view_264, permute_131);  arg193_1 = view_264 = permute_131 = None
    view_265: "f32[1, 1024, 1024]" = torch.ops.aten.view.default(addmm_71, [1, 1024, 1024]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:466, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_96: "f32[1, 1024, 1024]" = torch.ops.aten.clone.default(view_265);  view_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:467, code: hidden_states = residual + hidden_states
    add_98: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(add_96, clone_96);  add_96 = clone_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:468, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_24 = torch.ops.aten.var_mean.correction(add_98, [2], correction = 0, keepdim = True)
    getitem_48: "f32[1, 1024, 1]" = var_mean_24[0]
    getitem_49: "f32[1, 1024, 1]" = var_mean_24[1];  var_mean_24 = None
    add_99: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
    rsqrt_24: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
    sub_36: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_98, getitem_49);  add_98 = getitem_49 = None
    mul_97: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = rsqrt_24 = None
    mul_98: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_97, arg194_1);  mul_97 = arg194_1 = None
    add_100: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_98, arg195_1);  mul_98 = arg195_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:1907, code: logits = self.lm_head(outputs[0])
    permute_132: "f32[1024, 50265]" = torch.ops.aten.permute.default(arg196_1, [1, 0]);  arg196_1 = None
    view_266: "f32[1024, 1024]" = torch.ops.aten.view.default(add_100, [1024, 1024]);  add_100 = None
    mm: "f32[1024, 50265]" = torch.ops.aten.mm.default(view_266, permute_132);  view_266 = permute_132 = None
    view_267: "f32[1, 1024, 50265]" = torch.ops.aten.view.default(mm, [1, 1024, 50265]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:1913, code: loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
    view_268: "f32[1024, 50265]" = torch.ops.aten.view.default(view_267, [-1, 50265])
    view_269: "i64[1024]" = torch.ops.aten.view.default(arg198_1, [-1]);  arg198_1 = None
    amax_12: "f32[1024, 1]" = torch.ops.aten.amax.default(view_268, [1], True)
    sub_37: "f32[1024, 50265]" = torch.ops.aten.sub.Tensor(view_268, amax_12);  view_268 = amax_12 = None
    exp_12: "f32[1024, 50265]" = torch.ops.aten.exp.default(sub_37)
    sum_13: "f32[1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [1], True);  exp_12 = None
    log: "f32[1024, 1]" = torch.ops.aten.log.default(sum_13);  sum_13 = None
    sub_38: "f32[1024, 50265]" = torch.ops.aten.sub.Tensor(sub_37, log);  sub_37 = log = None
    ne: "b8[1024]" = torch.ops.aten.ne.Scalar(view_269, -100)
    scalar_tensor_1: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0))
    where_1: "i64[1024]" = torch.ops.aten.where.self(ne, view_269, scalar_tensor_1);  ne = scalar_tensor_1 = None
    unsqueeze_4: "i64[1024, 1]" = torch.ops.aten.unsqueeze.default(where_1, 1);  where_1 = None
    gather: "f32[1024, 1]" = torch.ops.aten.gather.default(sub_38, 1, unsqueeze_4);  sub_38 = unsqueeze_4 = None
    squeeze: "f32[1024]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[1024]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
    ne_1: "b8[1024]" = torch.ops.aten.ne.Scalar(view_269, -100)
    scalar_tensor_2: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_2: "f32[1024]" = torch.ops.aten.where.self(ne_1, neg, scalar_tensor_2);  ne_1 = neg = scalar_tensor_2 = None
    ne_2: "b8[1024]" = torch.ops.aten.ne.Scalar(view_269, -100);  view_269 = None
    sum_14: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None
    sum_15: "f32[]" = torch.ops.aten.sum.default(where_2);  where_2 = None
    div_12: "f32[]" = torch.ops.aten.div.Tensor(sum_15, convert_element_type);  sum_15 = convert_element_type = None
    return (div_12, view_267, clone_1, clone_2, clone_9, clone_10, clone_17, clone_18, clone_25, clone_26, clone_33, clone_34, clone_41, clone_42, clone_49, clone_50, clone_57, clone_58, clone_65, clone_66, clone_73, clone_74, clone_81, clone_82, clone_89, clone_90)
    