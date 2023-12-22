from __future__ import annotations



def forward(self, arg0_1: "f32[1026, 768]", arg1_1: "f32[50005, 768]", arg2_1: "f32[768]", arg3_1: "f32[768]", arg4_1: "f32[768, 768]", arg5_1: "f32[768]", arg6_1: "f32[768, 768]", arg7_1: "f32[768]", arg8_1: "f32[768, 768]", arg9_1: "f32[768]", arg10_1: "f32[768, 768]", arg11_1: "f32[768]", arg12_1: "f32[768]", arg13_1: "f32[768]", arg14_1: "f32[3072, 768]", arg15_1: "f32[3072]", arg16_1: "f32[768, 3072]", arg17_1: "f32[768]", arg18_1: "f32[768]", arg19_1: "f32[768]", arg20_1: "f32[768, 768]", arg21_1: "f32[768]", arg22_1: "f32[768, 768]", arg23_1: "f32[768]", arg24_1: "f32[768, 768]", arg25_1: "f32[768]", arg26_1: "f32[768, 768]", arg27_1: "f32[768]", arg28_1: "f32[768]", arg29_1: "f32[768]", arg30_1: "f32[3072, 768]", arg31_1: "f32[3072]", arg32_1: "f32[768, 3072]", arg33_1: "f32[768]", arg34_1: "f32[768]", arg35_1: "f32[768]", arg36_1: "f32[768, 768]", arg37_1: "f32[768]", arg38_1: "f32[768, 768]", arg39_1: "f32[768]", arg40_1: "f32[768, 768]", arg41_1: "f32[768]", arg42_1: "f32[768, 768]", arg43_1: "f32[768]", arg44_1: "f32[768]", arg45_1: "f32[768]", arg46_1: "f32[3072, 768]", arg47_1: "f32[3072]", arg48_1: "f32[768, 3072]", arg49_1: "f32[768]", arg50_1: "f32[768]", arg51_1: "f32[768]", arg52_1: "f32[768, 768]", arg53_1: "f32[768]", arg54_1: "f32[768, 768]", arg55_1: "f32[768]", arg56_1: "f32[768, 768]", arg57_1: "f32[768]", arg58_1: "f32[768, 768]", arg59_1: "f32[768]", arg60_1: "f32[768]", arg61_1: "f32[768]", arg62_1: "f32[3072, 768]", arg63_1: "f32[3072]", arg64_1: "f32[768, 3072]", arg65_1: "f32[768]", arg66_1: "f32[768]", arg67_1: "f32[768]", arg68_1: "f32[768, 768]", arg69_1: "f32[768]", arg70_1: "f32[768, 768]", arg71_1: "f32[768]", arg72_1: "f32[768, 768]", arg73_1: "f32[768]", arg74_1: "f32[768, 768]", arg75_1: "f32[768]", arg76_1: "f32[768]", arg77_1: "f32[768]", arg78_1: "f32[3072, 768]", arg79_1: "f32[3072]", arg80_1: "f32[768, 3072]", arg81_1: "f32[768]", arg82_1: "f32[768]", arg83_1: "f32[768]", arg84_1: "f32[768, 768]", arg85_1: "f32[768]", arg86_1: "f32[768, 768]", arg87_1: "f32[768]", arg88_1: "f32[768, 768]", arg89_1: "f32[768]", arg90_1: "f32[768, 768]", arg91_1: "f32[768]", arg92_1: "f32[768]", arg93_1: "f32[768]", arg94_1: "f32[3072, 768]", arg95_1: "f32[3072]", arg96_1: "f32[768, 3072]", arg97_1: "f32[768]", arg98_1: "f32[768]", arg99_1: "f32[768]", arg100_1: "f32[50005, 768]", arg101_1: "i64[1, 1024]", arg102_1: "i64[1, 1024]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:1013, code: inputs_embeds = self.embed_tokens(input) * self.embed_scale
    embedding: "f32[1, 1024, 768]" = torch.ops.aten.embedding.default(arg1_1, arg101_1, 1);  arg1_1 = arg101_1 = None
    mul: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(embedding, 27.712812921102035);  embedding = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:129, code: positions = torch.arange(
    iota_1: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:131, code: ).expand(bsz, -1)
    expand_1: "i64[1, 1024]" = torch.ops.aten.expand.default(iota_1, [1, -1]);  iota_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:133, code: return super().forward(positions + self.offset)
    add_1: "i64[1, 1024]" = torch.ops.aten.add.Tensor(expand_1, 2);  expand_1 = None
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/sparse.py:163, code: return F.embedding(
    embedding_1: "f32[1, 1024, 768]" = torch.ops.aten.embedding.default(arg0_1, add_1);  arg0_1 = add_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:1028, code: hidden_states = inputs_embeds + positions
    add_2: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul, embedding_1);  mul = embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:1029, code: hidden_states = self.layernorm_embedding(hidden_states)
    var_mean = torch.ops.aten.var_mean.correction(add_2, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 1024, 1]" = var_mean[0]
    getitem_1: "f32[1, 1024, 1]" = var_mean[1];  var_mean = None
    sub: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_2, getitem_1);  add_2 = getitem_1 = None
    add_3: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
    mul_1: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
    mul_2: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1, arg2_1);  mul_1 = arg2_1 = None
    add_4: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_2, arg3_1);  mul_2 = arg3_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_2: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_4, [1024, 768])
    permute: "f32[768, 768]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
    
    # No stacktrace found for following nodes
    mm_default_35: "f32[1024, 768]" = torch.ops.aten.mm.default(view_2, permute);  view_2 = permute = None
    add_tensor_35: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_35, arg5_1);  mm_default_35 = arg5_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_3: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_35, [1, 1024, 768]);  add_tensor_35 = None
    mul_3: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_3, 0.125);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_10: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(mul_3, [1, 1024, 12, 64]);  mul_3 = None
    permute_5: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
    clone_3: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:227, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_11: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_3, [12, -1, 64]);  clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:213, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_4: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_4, [1024, 768])
    permute_1: "f32[768, 768]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
    
    # No stacktrace found for following nodes
    mm_default_34: "f32[1024, 768]" = torch.ops.aten.mm.default(view_4, permute_1);  view_4 = permute_1 = None
    add_tensor_34: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_34, arg7_1);  mm_default_34 = arg7_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:213, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_5: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_34, [1, 1024, 768]);  add_tensor_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_6: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_5, [1, -1, 12, 64]);  view_5 = None
    permute_2: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
    clone_1: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:228, code: key_states = key_states.reshape(*proj_shape)
    view_12: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_1, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:232, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_6: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
    bmm: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_11, permute_6);  view_11 = permute_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:245, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_14: "f32[1, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm, [1, 12, 1024, 1024]);  bmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:89, code: mask_cond = torch.arange(mask.size(-1), device=device)
    iota: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:90, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    add: "i64[1024]" = torch.ops.aten.add.Tensor(iota, 1)
    view_1: "i64[1024, 1]" = torch.ops.aten.reshape.default(add, [1024, 1]);  add = None
    lt: "b8[1024, 1024]" = torch.ops.aten.lt.Tensor(iota, view_1);  iota = view_1 = None
    full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:88, code: mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    full_default: "f32[1024, 1024]" = torch.ops.aten.full.default([1024, 1024], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:90, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    where: "f32[1024, 1024]" = torch.ops.aten.where.self(lt, full_default_1, full_default);  lt = full_default_1 = full_default = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:245, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    unsqueeze_2: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(where, 0);  where = None
    unsqueeze_3: "f32[1, 1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 1);  unsqueeze_2 = None
    expand_2: "f32[1, 1, 1024, 1024]" = torch.ops.aten.expand.default(unsqueeze_3, [1, 1, 1024, 1024]);  unsqueeze_3 = None
    add_5: "f32[1, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_14, expand_2);  view_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:246, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_15: "f32[12, 1024, 1024]" = torch.ops.aten.reshape.default(add_5, [12, 1024, 1024]);  add_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:248, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax: "f32[12, 1024, 1]" = torch.ops.aten.amax.default(view_15, [-1], True)
    sub_1: "f32[12, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_15, amax);  view_15 = amax = None
    exp: "f32[12, 1024, 1024]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
    sum_1: "f32[12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:214, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_7: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_4, [1024, 768])
    permute_3: "f32[768, 768]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
    
    # No stacktrace found for following nodes
    mm_default_33: "f32[1024, 768]" = torch.ops.aten.mm.default(view_7, permute_3);  view_7 = permute_3 = None
    add_tensor_33: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_33, arg9_1);  mm_default_33 = arg9_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:214, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_8: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_33, [1, 1024, 768]);  add_tensor_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_9: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_8, [1, -1, 12, 64]);  view_8 = None
    permute_4: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
    clone_2: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:229, code: value_states = value_states.reshape(*proj_shape)
    view_13: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_2, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:271, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_1: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(div, view_13);  div = view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:279, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_16: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_1, [1, 12, 1024, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:280, code: attn_output = attn_output.transpose(1, 2)
    permute_7: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:284, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_5: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_17: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_5, [1, 1024, 768]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_18: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_17, [1024, 768]);  view_17 = None
    permute_8: "f32[768, 768]" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
    
    # No stacktrace found for following nodes
    mm_default_32: "f32[1024, 768]" = torch.ops.aten.mm.default(view_18, permute_8);  view_18 = permute_8 = None
    add_tensor_32: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_32, arg11_1);  mm_default_32 = arg11_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_19: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_32, [1, 1024, 768]);  add_tensor_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:432, code: hidden_states = residual + hidden_states
    add_6: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_4, view_19);  add_4 = view_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:433, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_1 = torch.ops.aten.var_mean.correction(add_6, [2], correction = 0, keepdim = True)
    getitem_2: "f32[1, 1024, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 1024, 1]" = var_mean_1[1];  var_mean_1 = None
    sub_2: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_6, getitem_3);  add_6 = getitem_3 = None
    add_7: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
    rsqrt_1: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
    mul_4: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
    mul_5: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_4, arg12_1);  mul_4 = arg12_1 = None
    add_8: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_5, arg13_1);  mul_5 = arg13_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_20: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_8, [1024, 768])
    permute_9: "f32[768, 3072]" = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
    
    # No stacktrace found for following nodes
    mm_default_31: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_20, permute_9);  view_20 = permute_9 = None
    add_tensor_31: "f32[1024, 3072]" = torch.ops.aten.add.Tensor(mm_default_31, arg15_1);  mm_default_31 = arg15_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_21: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(add_tensor_31, [1, 1024, 3072]);  add_tensor_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_6: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_21, 0.5)
    mul_7: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_21, 0.7071067811865476);  view_21 = None
    erf: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_7);  mul_7 = None
    add_9: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_8: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_6, add_9);  mul_6 = add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_22: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_8, [1024, 3072]);  mul_8 = None
    permute_10: "f32[3072, 768]" = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
    
    # No stacktrace found for following nodes
    mm_default_30: "f32[1024, 768]" = torch.ops.aten.mm.default(view_22, permute_10);  view_22 = permute_10 = None
    add_tensor_30: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_30, arg17_1);  mm_default_30 = arg17_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_23: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_30, [1, 1024, 768]);  add_tensor_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:464, code: hidden_states = residual + hidden_states
    add_10: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_8, view_23);  add_8 = view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:465, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_10, [2], correction = 0, keepdim = True)
    getitem_4: "f32[1, 1024, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 1024, 1]" = var_mean_2[1];  var_mean_2 = None
    sub_3: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_10, getitem_5);  add_10 = getitem_5 = None
    add_11: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
    rsqrt_2: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    mul_9: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
    mul_10: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_9, arg18_1);  mul_9 = arg18_1 = None
    add_12: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_10, arg19_1);  mul_10 = arg19_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_24: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_12, [1024, 768])
    permute_11: "f32[768, 768]" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
    
    # No stacktrace found for following nodes
    mm_default_29: "f32[1024, 768]" = torch.ops.aten.mm.default(view_24, permute_11);  view_24 = permute_11 = None
    add_tensor_29: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_29, arg21_1);  mm_default_29 = arg21_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_25: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_29, [1, 1024, 768]);  add_tensor_29 = None
    mul_11: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_25, 0.125);  view_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_32: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(mul_11, [1, 1024, 12, 64]);  mul_11 = None
    permute_16: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None
    clone_11: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:227, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_33: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_11, [12, -1, 64]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:213, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_26: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_12, [1024, 768])
    permute_12: "f32[768, 768]" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
    
    # No stacktrace found for following nodes
    mm_default_28: "f32[1024, 768]" = torch.ops.aten.mm.default(view_26, permute_12);  view_26 = permute_12 = None
    add_tensor_28: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_28, arg23_1);  mm_default_28 = arg23_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:213, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_27: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_28, [1, 1024, 768]);  add_tensor_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_28: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_27, [1, -1, 12, 64]);  view_27 = None
    permute_13: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
    clone_9: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:228, code: key_states = key_states.reshape(*proj_shape)
    view_34: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_9, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:232, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_17: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
    bmm_2: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_33, permute_17);  view_33 = permute_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:245, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_36: "f32[1, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_2, [1, 12, 1024, 1024]);  bmm_2 = None
    add_13: "f32[1, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_36, expand_2);  view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:246, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_37: "f32[12, 1024, 1024]" = torch.ops.aten.reshape.default(add_13, [12, 1024, 1024]);  add_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:248, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_1: "f32[12, 1024, 1]" = torch.ops.aten.amax.default(view_37, [-1], True)
    sub_4: "f32[12, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_37, amax_1);  view_37 = amax_1 = None
    exp_1: "f32[12, 1024, 1024]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
    sum_2: "f32[12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_1: "f32[12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:214, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_29: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_12, [1024, 768])
    permute_14: "f32[768, 768]" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
    
    # No stacktrace found for following nodes
    mm_default_27: "f32[1024, 768]" = torch.ops.aten.mm.default(view_29, permute_14);  view_29 = permute_14 = None
    add_tensor_27: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_27, arg25_1);  mm_default_27 = arg25_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:214, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_30: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_27, [1, 1024, 768]);  add_tensor_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_31: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_30, [1, -1, 12, 64]);  view_30 = None
    permute_15: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None
    clone_10: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:229, code: value_states = value_states.reshape(*proj_shape)
    view_35: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_10, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:271, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_3: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(div_1, view_35);  div_1 = view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:279, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_38: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_3, [1, 12, 1024, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:280, code: attn_output = attn_output.transpose(1, 2)
    permute_18: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:284, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_13: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
    view_39: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_13, [1, 1024, 768]);  clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_40: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_39, [1024, 768]);  view_39 = None
    permute_19: "f32[768, 768]" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
    
    # No stacktrace found for following nodes
    mm_default_26: "f32[1024, 768]" = torch.ops.aten.mm.default(view_40, permute_19);  view_40 = permute_19 = None
    add_tensor_26: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_26, arg27_1);  mm_default_26 = arg27_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_41: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_26, [1, 1024, 768]);  add_tensor_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:432, code: hidden_states = residual + hidden_states
    add_14: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_12, view_41);  add_12 = view_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:433, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_3 = torch.ops.aten.var_mean.correction(add_14, [2], correction = 0, keepdim = True)
    getitem_6: "f32[1, 1024, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 1024, 1]" = var_mean_3[1];  var_mean_3 = None
    sub_5: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_14, getitem_7);  add_14 = getitem_7 = None
    add_15: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
    rsqrt_3: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
    mul_12: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
    mul_13: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_12, arg28_1);  mul_12 = arg28_1 = None
    add_16: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_13, arg29_1);  mul_13 = arg29_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_42: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_16, [1024, 768])
    permute_20: "f32[768, 3072]" = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
    
    # No stacktrace found for following nodes
    mm_default_25: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_42, permute_20);  view_42 = permute_20 = None
    add_tensor_25: "f32[1024, 3072]" = torch.ops.aten.add.Tensor(mm_default_25, arg31_1);  mm_default_25 = arg31_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_43: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(add_tensor_25, [1, 1024, 3072]);  add_tensor_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_14: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_43, 0.5)
    mul_15: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_43, 0.7071067811865476);  view_43 = None
    erf_1: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_15);  mul_15 = None
    add_17: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_16: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_14, add_17);  mul_14 = add_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_44: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_16, [1024, 3072]);  mul_16 = None
    permute_21: "f32[3072, 768]" = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
    
    # No stacktrace found for following nodes
    mm_default_24: "f32[1024, 768]" = torch.ops.aten.mm.default(view_44, permute_21);  view_44 = permute_21 = None
    add_tensor_24: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_24, arg33_1);  mm_default_24 = arg33_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_45: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_24, [1, 1024, 768]);  add_tensor_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:464, code: hidden_states = residual + hidden_states
    add_18: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_16, view_45);  add_16 = view_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:465, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_4 = torch.ops.aten.var_mean.correction(add_18, [2], correction = 0, keepdim = True)
    getitem_8: "f32[1, 1024, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 1024, 1]" = var_mean_4[1];  var_mean_4 = None
    sub_6: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_18, getitem_9);  add_18 = getitem_9 = None
    add_19: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
    rsqrt_4: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
    mul_17: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
    mul_18: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_17, arg34_1);  mul_17 = arg34_1 = None
    add_20: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_18, arg35_1);  mul_18 = arg35_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_46: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_20, [1024, 768])
    permute_22: "f32[768, 768]" = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
    
    # No stacktrace found for following nodes
    mm_default_23: "f32[1024, 768]" = torch.ops.aten.mm.default(view_46, permute_22);  view_46 = permute_22 = None
    add_tensor_23: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_23, arg37_1);  mm_default_23 = arg37_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_47: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_23, [1, 1024, 768]);  add_tensor_23 = None
    mul_19: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_47, 0.125);  view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_54: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(mul_19, [1, 1024, 12, 64]);  mul_19 = None
    permute_27: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
    clone_19: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:227, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_55: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_19, [12, -1, 64]);  clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:213, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_48: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_20, [1024, 768])
    permute_23: "f32[768, 768]" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
    
    # No stacktrace found for following nodes
    mm_default_22: "f32[1024, 768]" = torch.ops.aten.mm.default(view_48, permute_23);  view_48 = permute_23 = None
    add_tensor_22: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_22, arg39_1);  mm_default_22 = arg39_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:213, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_49: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_22, [1, 1024, 768]);  add_tensor_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_50: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_49, [1, -1, 12, 64]);  view_49 = None
    permute_24: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
    clone_17: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:228, code: key_states = key_states.reshape(*proj_shape)
    view_56: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_17, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:232, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_28: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None
    bmm_4: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_55, permute_28);  view_55 = permute_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:245, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_58: "f32[1, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_4, [1, 12, 1024, 1024]);  bmm_4 = None
    add_21: "f32[1, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_58, expand_2);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:246, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_59: "f32[12, 1024, 1024]" = torch.ops.aten.reshape.default(add_21, [12, 1024, 1024]);  add_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:248, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_2: "f32[12, 1024, 1]" = torch.ops.aten.amax.default(view_59, [-1], True)
    sub_7: "f32[12, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_59, amax_2);  view_59 = amax_2 = None
    exp_2: "f32[12, 1024, 1024]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
    sum_3: "f32[12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_2: "f32[12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:214, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_51: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_20, [1024, 768])
    permute_25: "f32[768, 768]" = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
    
    # No stacktrace found for following nodes
    mm_default_21: "f32[1024, 768]" = torch.ops.aten.mm.default(view_51, permute_25);  view_51 = permute_25 = None
    add_tensor_21: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_21, arg41_1);  mm_default_21 = arg41_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:214, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_52: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_21, [1, 1024, 768]);  add_tensor_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_53: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_52, [1, -1, 12, 64]);  view_52 = None
    permute_26: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
    clone_18: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:229, code: value_states = value_states.reshape(*proj_shape)
    view_57: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_18, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:271, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_5: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(div_2, view_57);  div_2 = view_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:279, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_60: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_5, [1, 12, 1024, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:280, code: attn_output = attn_output.transpose(1, 2)
    permute_29: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_60, [0, 2, 1, 3]);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:284, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_21: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    view_61: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_21, [1, 1024, 768]);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_62: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_61, [1024, 768]);  view_61 = None
    permute_30: "f32[768, 768]" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
    
    # No stacktrace found for following nodes
    mm_default_20: "f32[1024, 768]" = torch.ops.aten.mm.default(view_62, permute_30);  view_62 = permute_30 = None
    add_tensor_20: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_20, arg43_1);  mm_default_20 = arg43_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_63: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_20, [1, 1024, 768]);  add_tensor_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:432, code: hidden_states = residual + hidden_states
    add_22: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_20, view_63);  add_20 = view_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:433, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_5 = torch.ops.aten.var_mean.correction(add_22, [2], correction = 0, keepdim = True)
    getitem_10: "f32[1, 1024, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 1024, 1]" = var_mean_5[1];  var_mean_5 = None
    sub_8: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_22, getitem_11);  add_22 = getitem_11 = None
    add_23: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
    rsqrt_5: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
    mul_20: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
    mul_21: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_20, arg44_1);  mul_20 = arg44_1 = None
    add_24: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_21, arg45_1);  mul_21 = arg45_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_64: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_24, [1024, 768])
    permute_31: "f32[768, 3072]" = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
    
    # No stacktrace found for following nodes
    mm_default_19: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_64, permute_31);  view_64 = permute_31 = None
    add_tensor_19: "f32[1024, 3072]" = torch.ops.aten.add.Tensor(mm_default_19, arg47_1);  mm_default_19 = arg47_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_65: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(add_tensor_19, [1, 1024, 3072]);  add_tensor_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_22: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_65, 0.5)
    mul_23: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_65, 0.7071067811865476);  view_65 = None
    erf_2: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_23);  mul_23 = None
    add_25: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_24: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_22, add_25);  mul_22 = add_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_66: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_24, [1024, 3072]);  mul_24 = None
    permute_32: "f32[3072, 768]" = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
    
    # No stacktrace found for following nodes
    mm_default_18: "f32[1024, 768]" = torch.ops.aten.mm.default(view_66, permute_32);  view_66 = permute_32 = None
    add_tensor_18: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_18, arg49_1);  mm_default_18 = arg49_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_67: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_18, [1, 1024, 768]);  add_tensor_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:464, code: hidden_states = residual + hidden_states
    add_26: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_24, view_67);  add_24 = view_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:465, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_6 = torch.ops.aten.var_mean.correction(add_26, [2], correction = 0, keepdim = True)
    getitem_12: "f32[1, 1024, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 1024, 1]" = var_mean_6[1];  var_mean_6 = None
    sub_9: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_26, getitem_13);  add_26 = getitem_13 = None
    add_27: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
    rsqrt_6: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
    mul_25: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = rsqrt_6 = None
    mul_26: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_25, arg50_1);  mul_25 = arg50_1 = None
    add_28: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_26, arg51_1);  mul_26 = arg51_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_68: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_28, [1024, 768])
    permute_33: "f32[768, 768]" = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
    
    # No stacktrace found for following nodes
    mm_default_17: "f32[1024, 768]" = torch.ops.aten.mm.default(view_68, permute_33);  view_68 = permute_33 = None
    add_tensor_17: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_17, arg53_1);  mm_default_17 = arg53_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_69: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_17, [1, 1024, 768]);  add_tensor_17 = None
    mul_27: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_69, 0.125);  view_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_76: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(mul_27, [1, 1024, 12, 64]);  mul_27 = None
    permute_38: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
    clone_27: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:227, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_77: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_27, [12, -1, 64]);  clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:213, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_70: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_28, [1024, 768])
    permute_34: "f32[768, 768]" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
    
    # No stacktrace found for following nodes
    mm_default_16: "f32[1024, 768]" = torch.ops.aten.mm.default(view_70, permute_34);  view_70 = permute_34 = None
    add_tensor_16: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_16, arg55_1);  mm_default_16 = arg55_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:213, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_71: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_16, [1, 1024, 768]);  add_tensor_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_72: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_71, [1, -1, 12, 64]);  view_71 = None
    permute_35: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
    clone_25: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:228, code: key_states = key_states.reshape(*proj_shape)
    view_78: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_25, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:232, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_39: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
    bmm_6: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_77, permute_39);  view_77 = permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:245, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_80: "f32[1, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_6, [1, 12, 1024, 1024]);  bmm_6 = None
    add_29: "f32[1, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_80, expand_2);  view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:246, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_81: "f32[12, 1024, 1024]" = torch.ops.aten.reshape.default(add_29, [12, 1024, 1024]);  add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:248, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_3: "f32[12, 1024, 1]" = torch.ops.aten.amax.default(view_81, [-1], True)
    sub_10: "f32[12, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_81, amax_3);  view_81 = amax_3 = None
    exp_3: "f32[12, 1024, 1024]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
    sum_4: "f32[12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_3: "f32[12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:214, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_73: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_28, [1024, 768])
    permute_36: "f32[768, 768]" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
    
    # No stacktrace found for following nodes
    mm_default_15: "f32[1024, 768]" = torch.ops.aten.mm.default(view_73, permute_36);  view_73 = permute_36 = None
    add_tensor_15: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_15, arg57_1);  mm_default_15 = arg57_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:214, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_74: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_15, [1, 1024, 768]);  add_tensor_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_75: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_74, [1, -1, 12, 64]);  view_74 = None
    permute_37: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
    clone_26: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:229, code: value_states = value_states.reshape(*proj_shape)
    view_79: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_26, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:271, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_7: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(div_3, view_79);  div_3 = view_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:279, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_82: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_7, [1, 12, 1024, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:280, code: attn_output = attn_output.transpose(1, 2)
    permute_40: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:284, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_29: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
    view_83: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_29, [1, 1024, 768]);  clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_84: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_83, [1024, 768]);  view_83 = None
    permute_41: "f32[768, 768]" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
    
    # No stacktrace found for following nodes
    mm_default_14: "f32[1024, 768]" = torch.ops.aten.mm.default(view_84, permute_41);  view_84 = permute_41 = None
    add_tensor_14: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_14, arg59_1);  mm_default_14 = arg59_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_85: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_14, [1, 1024, 768]);  add_tensor_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:432, code: hidden_states = residual + hidden_states
    add_30: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_28, view_85);  add_28 = view_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:433, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_7 = torch.ops.aten.var_mean.correction(add_30, [2], correction = 0, keepdim = True)
    getitem_14: "f32[1, 1024, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 1024, 1]" = var_mean_7[1];  var_mean_7 = None
    sub_11: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_30, getitem_15);  add_30 = getitem_15 = None
    add_31: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
    rsqrt_7: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
    mul_28: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
    mul_29: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_28, arg60_1);  mul_28 = arg60_1 = None
    add_32: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_29, arg61_1);  mul_29 = arg61_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_86: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_32, [1024, 768])
    permute_42: "f32[768, 3072]" = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
    
    # No stacktrace found for following nodes
    mm_default_13: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_86, permute_42);  view_86 = permute_42 = None
    add_tensor_13: "f32[1024, 3072]" = torch.ops.aten.add.Tensor(mm_default_13, arg63_1);  mm_default_13 = arg63_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_87: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(add_tensor_13, [1, 1024, 3072]);  add_tensor_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_30: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_87, 0.5)
    mul_31: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_87, 0.7071067811865476);  view_87 = None
    erf_3: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_31);  mul_31 = None
    add_33: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_32: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_30, add_33);  mul_30 = add_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_88: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_32, [1024, 3072]);  mul_32 = None
    permute_43: "f32[3072, 768]" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
    
    # No stacktrace found for following nodes
    mm_default_12: "f32[1024, 768]" = torch.ops.aten.mm.default(view_88, permute_43);  view_88 = permute_43 = None
    add_tensor_12: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_12, arg65_1);  mm_default_12 = arg65_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_89: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_12, [1, 1024, 768]);  add_tensor_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:464, code: hidden_states = residual + hidden_states
    add_34: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_32, view_89);  add_32 = view_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:465, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_8 = torch.ops.aten.var_mean.correction(add_34, [2], correction = 0, keepdim = True)
    getitem_16: "f32[1, 1024, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 1024, 1]" = var_mean_8[1];  var_mean_8 = None
    sub_12: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_34, getitem_17);  add_34 = getitem_17 = None
    add_35: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
    rsqrt_8: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
    mul_33: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = rsqrt_8 = None
    mul_34: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_33, arg66_1);  mul_33 = arg66_1 = None
    add_36: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_34, arg67_1);  mul_34 = arg67_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_90: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_36, [1024, 768])
    permute_44: "f32[768, 768]" = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
    
    # No stacktrace found for following nodes
    mm_default_11: "f32[1024, 768]" = torch.ops.aten.mm.default(view_90, permute_44);  view_90 = permute_44 = None
    add_tensor_11: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_11, arg69_1);  mm_default_11 = arg69_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_91: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_11, [1, 1024, 768]);  add_tensor_11 = None
    mul_35: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_91, 0.125);  view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_98: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(mul_35, [1, 1024, 12, 64]);  mul_35 = None
    permute_49: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None
    clone_35: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:227, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_99: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_35, [12, -1, 64]);  clone_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:213, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_92: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_36, [1024, 768])
    permute_45: "f32[768, 768]" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
    
    # No stacktrace found for following nodes
    mm_default_10: "f32[1024, 768]" = torch.ops.aten.mm.default(view_92, permute_45);  view_92 = permute_45 = None
    add_tensor_10: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_10, arg71_1);  mm_default_10 = arg71_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:213, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_93: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_10, [1, 1024, 768]);  add_tensor_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_94: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_93, [1, -1, 12, 64]);  view_93 = None
    permute_46: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
    clone_33: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:228, code: key_states = key_states.reshape(*proj_shape)
    view_100: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_33, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:232, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_50: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_100, [0, 2, 1]);  view_100 = None
    bmm_8: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_99, permute_50);  view_99 = permute_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:245, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_102: "f32[1, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_8, [1, 12, 1024, 1024]);  bmm_8 = None
    add_37: "f32[1, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_102, expand_2);  view_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:246, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_103: "f32[12, 1024, 1024]" = torch.ops.aten.reshape.default(add_37, [12, 1024, 1024]);  add_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:248, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_4: "f32[12, 1024, 1]" = torch.ops.aten.amax.default(view_103, [-1], True)
    sub_13: "f32[12, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_103, amax_4);  view_103 = amax_4 = None
    exp_4: "f32[12, 1024, 1024]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
    sum_5: "f32[12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_4: "f32[12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:214, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_95: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_36, [1024, 768])
    permute_47: "f32[768, 768]" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
    
    # No stacktrace found for following nodes
    mm_default_9: "f32[1024, 768]" = torch.ops.aten.mm.default(view_95, permute_47);  view_95 = permute_47 = None
    add_tensor_9: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_9, arg73_1);  mm_default_9 = arg73_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:214, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_96: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_9, [1, 1024, 768]);  add_tensor_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_97: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_96, [1, -1, 12, 64]);  view_96 = None
    permute_48: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_97, [0, 2, 1, 3]);  view_97 = None
    clone_34: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:229, code: value_states = value_states.reshape(*proj_shape)
    view_101: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_34, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:271, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_9: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(div_4, view_101);  div_4 = view_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:279, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_104: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_9, [1, 12, 1024, 64]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:280, code: attn_output = attn_output.transpose(1, 2)
    permute_51: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:284, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_37: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
    view_105: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_37, [1, 1024, 768]);  clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_106: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_105, [1024, 768]);  view_105 = None
    permute_52: "f32[768, 768]" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
    
    # No stacktrace found for following nodes
    mm_default_8: "f32[1024, 768]" = torch.ops.aten.mm.default(view_106, permute_52);  view_106 = permute_52 = None
    add_tensor_8: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_8, arg75_1);  mm_default_8 = arg75_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_107: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_8, [1, 1024, 768]);  add_tensor_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:432, code: hidden_states = residual + hidden_states
    add_38: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_36, view_107);  add_36 = view_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:433, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_9 = torch.ops.aten.var_mean.correction(add_38, [2], correction = 0, keepdim = True)
    getitem_18: "f32[1, 1024, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 1024, 1]" = var_mean_9[1];  var_mean_9 = None
    sub_14: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_38, getitem_19);  add_38 = getitem_19 = None
    add_39: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
    rsqrt_9: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
    mul_36: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
    mul_37: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_36, arg76_1);  mul_36 = arg76_1 = None
    add_40: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_37, arg77_1);  mul_37 = arg77_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_108: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_40, [1024, 768])
    permute_53: "f32[768, 3072]" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
    
    # No stacktrace found for following nodes
    mm_default_7: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_108, permute_53);  view_108 = permute_53 = None
    add_tensor_7: "f32[1024, 3072]" = torch.ops.aten.add.Tensor(mm_default_7, arg79_1);  mm_default_7 = arg79_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_109: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(add_tensor_7, [1, 1024, 3072]);  add_tensor_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_38: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_109, 0.5)
    mul_39: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_109, 0.7071067811865476);  view_109 = None
    erf_4: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_39);  mul_39 = None
    add_41: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_40: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_38, add_41);  mul_38 = add_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_110: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_40, [1024, 3072]);  mul_40 = None
    permute_54: "f32[3072, 768]" = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
    
    # No stacktrace found for following nodes
    mm_default_6: "f32[1024, 768]" = torch.ops.aten.mm.default(view_110, permute_54);  view_110 = permute_54 = None
    add_tensor_6: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_6, arg81_1);  mm_default_6 = arg81_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_111: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_6, [1, 1024, 768]);  add_tensor_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:464, code: hidden_states = residual + hidden_states
    add_42: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_40, view_111);  add_40 = view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:465, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_10 = torch.ops.aten.var_mean.correction(add_42, [2], correction = 0, keepdim = True)
    getitem_20: "f32[1, 1024, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 1024, 1]" = var_mean_10[1];  var_mean_10 = None
    sub_15: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_42, getitem_21);  add_42 = getitem_21 = None
    add_43: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
    rsqrt_10: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
    mul_41: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = rsqrt_10 = None
    mul_42: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_41, arg82_1);  mul_41 = arg82_1 = None
    add_44: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_42, arg83_1);  mul_42 = arg83_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_112: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_44, [1024, 768])
    permute_55: "f32[768, 768]" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
    
    # No stacktrace found for following nodes
    mm_default_5: "f32[1024, 768]" = torch.ops.aten.mm.default(view_112, permute_55);  view_112 = permute_55 = None
    add_tensor_5: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_5, arg85_1);  mm_default_5 = arg85_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:188, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_113: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_5, [1, 1024, 768]);  add_tensor_5 = None
    mul_43: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_113, 0.125);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_120: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(mul_43, [1, 1024, 12, 64]);  mul_43 = None
    permute_60: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
    clone_43: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:227, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_121: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_43, [12, -1, 64]);  clone_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:213, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_114: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_44, [1024, 768])
    permute_56: "f32[768, 768]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
    
    # No stacktrace found for following nodes
    mm_default_4: "f32[1024, 768]" = torch.ops.aten.mm.default(view_114, permute_56);  view_114 = permute_56 = None
    add_tensor_4: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_4, arg87_1);  mm_default_4 = arg87_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:213, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_115: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_4, [1, 1024, 768]);  add_tensor_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_116: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_115, [1, -1, 12, 64]);  view_115 = None
    permute_57: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_116, [0, 2, 1, 3]);  view_116 = None
    clone_41: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_57, memory_format = torch.contiguous_format);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:228, code: key_states = key_states.reshape(*proj_shape)
    view_122: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_41, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:232, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_61: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None
    bmm_10: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_121, permute_61);  view_121 = permute_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:245, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_124: "f32[1, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_10, [1, 12, 1024, 1024]);  bmm_10 = None
    add_45: "f32[1, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_124, expand_2);  view_124 = expand_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:246, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_125: "f32[12, 1024, 1024]" = torch.ops.aten.reshape.default(add_45, [12, 1024, 1024]);  add_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:248, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_5: "f32[12, 1024, 1]" = torch.ops.aten.amax.default(view_125, [-1], True)
    sub_16: "f32[12, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_125, amax_5);  view_125 = amax_5 = None
    exp_5: "f32[12, 1024, 1024]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
    sum_6: "f32[12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_5: "f32[12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:214, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_117: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_44, [1024, 768])
    permute_58: "f32[768, 768]" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
    
    # No stacktrace found for following nodes
    mm_default_3: "f32[1024, 768]" = torch.ops.aten.mm.default(view_117, permute_58);  view_117 = permute_58 = None
    add_tensor_3: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_3, arg89_1);  mm_default_3 = arg89_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:214, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_118: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_3, [1, 1024, 768]);  add_tensor_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:168, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_119: "f32[1, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_118, [1, -1, 12, 64]);  view_118 = None
    permute_59: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_119, [0, 2, 1, 3]);  view_119 = None
    clone_42: "f32[1, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:229, code: value_states = value_states.reshape(*proj_shape)
    view_123: "f32[12, 1024, 64]" = torch.ops.aten.reshape.default(clone_42, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:271, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_11: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(div_5, view_123);  div_5 = view_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:279, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_126: "f32[1, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_11, [1, 12, 1024, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:280, code: attn_output = attn_output.transpose(1, 2)
    permute_62: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:284, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_45: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    view_127: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(clone_45, [1, 1024, 768]);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_128: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_127, [1024, 768]);  view_127 = None
    permute_63: "f32[768, 768]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
    
    # No stacktrace found for following nodes
    mm_default_2: "f32[1024, 768]" = torch.ops.aten.mm.default(view_128, permute_63);  view_128 = permute_63 = None
    add_tensor_2: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_2, arg91_1);  mm_default_2 = arg91_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:286, code: attn_output = self.out_proj(attn_output)
    view_129: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_2, [1, 1024, 768]);  add_tensor_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:432, code: hidden_states = residual + hidden_states
    add_46: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_44, view_129);  add_44 = view_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:433, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_11 = torch.ops.aten.var_mean.correction(add_46, [2], correction = 0, keepdim = True)
    getitem_22: "f32[1, 1024, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 1024, 1]" = var_mean_11[1];  var_mean_11 = None
    sub_17: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_46, getitem_23);  add_46 = getitem_23 = None
    add_47: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
    rsqrt_11: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
    mul_44: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
    mul_45: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_44, arg92_1);  mul_44 = arg92_1 = None
    add_48: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_45, arg93_1);  mul_45 = arg93_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_130: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_48, [1024, 768])
    permute_64: "f32[768, 3072]" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
    
    # No stacktrace found for following nodes
    mm_default_1: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_130, permute_64);  view_130 = permute_64 = None
    add_tensor_1: "f32[1024, 3072]" = torch.ops.aten.add.Tensor(mm_default_1, arg95_1);  mm_default_1 = arg95_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:460, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_131: "f32[1, 1024, 3072]" = torch.ops.aten.reshape.default(add_tensor_1, [1, 1024, 3072]);  add_tensor_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_46: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_131, 0.5)
    mul_47: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_131, 0.7071067811865476);  view_131 = None
    erf_5: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
    add_49: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_48: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_46, add_49);  mul_46 = add_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_132: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_48, [1024, 3072]);  mul_48 = None
    permute_65: "f32[3072, 768]" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
    
    # No stacktrace found for following nodes
    mm_default: "f32[1024, 768]" = torch.ops.aten.mm.default(view_132, permute_65);  view_132 = permute_65 = None
    add_tensor: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default, arg97_1);  mm_default = arg97_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:462, code: hidden_states = self.fc2(hidden_states)
    view_133: "f32[1, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor, [1, 1024, 768]);  add_tensor = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:464, code: hidden_states = residual + hidden_states
    add_50: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_48, view_133);  add_48 = view_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:465, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_12 = torch.ops.aten.var_mean.correction(add_50, [2], correction = 0, keepdim = True)
    getitem_24: "f32[1, 1024, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 1024, 1]" = var_mean_12[1];  var_mean_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:1718, code: loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
    view_137: "i64[1024]" = torch.ops.aten.reshape.default(arg102_1, [-1]);  arg102_1 = None
    ne_1: "b8[1024]" = torch.ops.aten.ne.Scalar(view_137, -100)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:465, code: hidden_states = self.final_layer_norm(hidden_states)
    sub_18: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_50, getitem_25);  add_50 = getitem_25 = None
    add_51: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
    rsqrt_12: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
    mul_49: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = rsqrt_12 = None
    mul_50: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_49, arg98_1);  mul_49 = arg98_1 = None
    add_52: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_50, arg99_1);  mul_50 = arg99_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:1712, code: logits = self.lm_head(outputs[0])
    view_134: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_52, [1024, 768]);  add_52 = None
    permute_66: "f32[768, 50005]" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
    mm: "f32[1024, 50005]" = torch.ops.aten.mm.default(view_134, permute_66);  view_134 = permute_66 = None
    view_135: "f32[1, 1024, 50005]" = torch.ops.aten.reshape.default(mm, [1, 1024, 50005]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/plbart/modeling_plbart.py:1718, code: loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
    view_136: "f32[1024, 50005]" = torch.ops.aten.reshape.default(view_135, [-1, 50005])
    amax_6: "f32[1024, 1]" = torch.ops.aten.amax.default(view_136, [1], True)
    sub_19: "f32[1024, 50005]" = torch.ops.aten.sub.Tensor(view_136, amax_6);  view_136 = amax_6 = None
    exp_6: "f32[1024, 50005]" = torch.ops.aten.exp.default(sub_19)
    sum_7: "f32[1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [1], True);  exp_6 = None
    log: "f32[1024, 1]" = torch.ops.aten.log.default(sum_7);  sum_7 = None
    sub_20: "f32[1024, 50005]" = torch.ops.aten.sub.Tensor(sub_19, log);  sub_19 = log = None
    ne: "b8[1024]" = torch.ops.aten.ne.Scalar(view_137, -100)
    full_default_2: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where_1: "i64[1024]" = torch.ops.aten.where.self(ne, view_137, full_default_2);  ne = full_default_2 = None
    unsqueeze_4: "i64[1024, 1]" = torch.ops.aten.unsqueeze.default(where_1, 1);  where_1 = None
    gather: "f32[1024, 1]" = torch.ops.aten.gather.default(sub_20, 1, unsqueeze_4);  sub_20 = unsqueeze_4 = None
    squeeze: "f32[1024]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[1024]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
    full_default_3: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where_2: "f32[1024]" = torch.ops.aten.where.self(ne_1, neg, full_default_3);  ne_1 = neg = full_default_3 = None
    sum_9: "f32[]" = torch.ops.aten.sum.default(where_2);  where_2 = None
    ne_2: "b8[1024]" = torch.ops.aten.ne.Scalar(view_137, -100);  view_137 = None
    sum_8: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_8, torch.float32);  sum_8 = None
    div_6: "f32[]" = torch.ops.aten.div.Tensor(sum_9, convert_element_type);  sum_9 = convert_element_type = None
    return (div_6, view_135, clone_1, clone_2, clone_9, clone_10, clone_17, clone_18, clone_25, clone_26, clone_33, clone_34, clone_41, clone_42)
    