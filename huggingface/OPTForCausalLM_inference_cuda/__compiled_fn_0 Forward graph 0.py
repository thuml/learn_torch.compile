from __future__ import annotations



def forward(self, arg0_1: "f32[2050, 768]", arg1_1: "f32[50272, 768]", arg2_1: "f32[768]", arg3_1: "f32[768]", arg4_1: "f32[768, 768]", arg5_1: "f32[768]", arg6_1: "f32[768, 768]", arg7_1: "f32[768]", arg8_1: "f32[768, 768]", arg9_1: "f32[768]", arg10_1: "f32[768, 768]", arg11_1: "f32[768]", arg12_1: "f32[768]", arg13_1: "f32[768]", arg14_1: "f32[3072, 768]", arg15_1: "f32[3072]", arg16_1: "f32[768, 3072]", arg17_1: "f32[768]", arg18_1: "f32[768]", arg19_1: "f32[768]", arg20_1: "f32[768, 768]", arg21_1: "f32[768]", arg22_1: "f32[768, 768]", arg23_1: "f32[768]", arg24_1: "f32[768, 768]", arg25_1: "f32[768]", arg26_1: "f32[768, 768]", arg27_1: "f32[768]", arg28_1: "f32[768]", arg29_1: "f32[768]", arg30_1: "f32[3072, 768]", arg31_1: "f32[3072]", arg32_1: "f32[768, 3072]", arg33_1: "f32[768]", arg34_1: "f32[768]", arg35_1: "f32[768]", arg36_1: "f32[768, 768]", arg37_1: "f32[768]", arg38_1: "f32[768, 768]", arg39_1: "f32[768]", arg40_1: "f32[768, 768]", arg41_1: "f32[768]", arg42_1: "f32[768, 768]", arg43_1: "f32[768]", arg44_1: "f32[768]", arg45_1: "f32[768]", arg46_1: "f32[3072, 768]", arg47_1: "f32[3072]", arg48_1: "f32[768, 3072]", arg49_1: "f32[768]", arg50_1: "f32[768]", arg51_1: "f32[768]", arg52_1: "f32[768, 768]", arg53_1: "f32[768]", arg54_1: "f32[768, 768]", arg55_1: "f32[768]", arg56_1: "f32[768, 768]", arg57_1: "f32[768]", arg58_1: "f32[768, 768]", arg59_1: "f32[768]", arg60_1: "f32[768]", arg61_1: "f32[768]", arg62_1: "f32[3072, 768]", arg63_1: "f32[3072]", arg64_1: "f32[768, 3072]", arg65_1: "f32[768]", arg66_1: "f32[768]", arg67_1: "f32[768]", arg68_1: "f32[768, 768]", arg69_1: "f32[768]", arg70_1: "f32[768, 768]", arg71_1: "f32[768]", arg72_1: "f32[768, 768]", arg73_1: "f32[768]", arg74_1: "f32[768, 768]", arg75_1: "f32[768]", arg76_1: "f32[768]", arg77_1: "f32[768]", arg78_1: "f32[3072, 768]", arg79_1: "f32[3072]", arg80_1: "f32[768, 3072]", arg81_1: "f32[768]", arg82_1: "f32[768]", arg83_1: "f32[768]", arg84_1: "f32[768, 768]", arg85_1: "f32[768]", arg86_1: "f32[768, 768]", arg87_1: "f32[768]", arg88_1: "f32[768, 768]", arg89_1: "f32[768]", arg90_1: "f32[768, 768]", arg91_1: "f32[768]", arg92_1: "f32[768]", arg93_1: "f32[768]", arg94_1: "f32[3072, 768]", arg95_1: "f32[3072]", arg96_1: "f32[768, 3072]", arg97_1: "f32[768]", arg98_1: "f32[768]", arg99_1: "f32[768]", arg100_1: "f32[768, 768]", arg101_1: "f32[768]", arg102_1: "f32[768, 768]", arg103_1: "f32[768]", arg104_1: "f32[768, 768]", arg105_1: "f32[768]", arg106_1: "f32[768, 768]", arg107_1: "f32[768]", arg108_1: "f32[768]", arg109_1: "f32[768]", arg110_1: "f32[3072, 768]", arg111_1: "f32[3072]", arg112_1: "f32[768, 3072]", arg113_1: "f32[768]", arg114_1: "f32[768]", arg115_1: "f32[768]", arg116_1: "f32[768, 768]", arg117_1: "f32[768]", arg118_1: "f32[768, 768]", arg119_1: "f32[768]", arg120_1: "f32[768, 768]", arg121_1: "f32[768]", arg122_1: "f32[768, 768]", arg123_1: "f32[768]", arg124_1: "f32[768]", arg125_1: "f32[768]", arg126_1: "f32[3072, 768]", arg127_1: "f32[3072]", arg128_1: "f32[768, 3072]", arg129_1: "f32[768]", arg130_1: "f32[768]", arg131_1: "f32[768]", arg132_1: "f32[768, 768]", arg133_1: "f32[768]", arg134_1: "f32[768, 768]", arg135_1: "f32[768]", arg136_1: "f32[768, 768]", arg137_1: "f32[768]", arg138_1: "f32[768, 768]", arg139_1: "f32[768]", arg140_1: "f32[768]", arg141_1: "f32[768]", arg142_1: "f32[3072, 768]", arg143_1: "f32[3072]", arg144_1: "f32[768, 3072]", arg145_1: "f32[768]", arg146_1: "f32[768]", arg147_1: "f32[768]", arg148_1: "f32[768, 768]", arg149_1: "f32[768]", arg150_1: "f32[768, 768]", arg151_1: "f32[768]", arg152_1: "f32[768, 768]", arg153_1: "f32[768]", arg154_1: "f32[768, 768]", arg155_1: "f32[768]", arg156_1: "f32[768]", arg157_1: "f32[768]", arg158_1: "f32[3072, 768]", arg159_1: "f32[3072]", arg160_1: "f32[768, 3072]", arg161_1: "f32[768]", arg162_1: "f32[768]", arg163_1: "f32[768]", arg164_1: "f32[768, 768]", arg165_1: "f32[768]", arg166_1: "f32[768, 768]", arg167_1: "f32[768]", arg168_1: "f32[768, 768]", arg169_1: "f32[768]", arg170_1: "f32[768, 768]", arg171_1: "f32[768]", arg172_1: "f32[768]", arg173_1: "f32[768]", arg174_1: "f32[3072, 768]", arg175_1: "f32[3072]", arg176_1: "f32[768, 3072]", arg177_1: "f32[768]", arg178_1: "f32[768]", arg179_1: "f32[768]", arg180_1: "f32[768, 768]", arg181_1: "f32[768]", arg182_1: "f32[768, 768]", arg183_1: "f32[768]", arg184_1: "f32[768, 768]", arg185_1: "f32[768]", arg186_1: "f32[768, 768]", arg187_1: "f32[768]", arg188_1: "f32[768]", arg189_1: "f32[768]", arg190_1: "f32[3072, 768]", arg191_1: "f32[3072]", arg192_1: "f32[768, 3072]", arg193_1: "f32[768]", arg194_1: "f32[768]", arg195_1: "f32[768]", arg196_1: "f32[50272, 768]", arg197_1: "i64[1, 2048]", arg198_1: "i64[1, 2048]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:628, code: input_ids = input_ids.view(-1, input_shape[-1])
    view: "i64[1, 2048]" = torch.ops.aten.view.default(arg197_1, [-1, 2048]);  arg197_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:635, code: inputs_embeds = self.embed_tokens(input_ids)
    embedding: "f32[1, 2048, 768]" = torch.ops.aten.embedding.default(arg1_1, view, 1);  arg1_1 = view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:644, code: attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
    full: "f32[1, 2048]" = torch.ops.aten.full.default([1, 2048], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:74, code: mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    full_1: "f32[2048, 2048]" = torch.ops.aten.full.default([2048, 2048], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:75, code: mask_cond = torch.arange(mask.size(-1), device=device)
    iota: "i64[2048]" = torch.ops.prims.iota.default(2048, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:76, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    add: "i64[2048]" = torch.ops.aten.add.Tensor(iota, 1)
    view_1: "i64[2048, 1]" = torch.ops.aten.view.default(add, [2048, 1]);  add = None
    lt: "b8[2048, 2048]" = torch.ops.aten.lt.Tensor(iota, view_1);  iota = view_1 = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where: "f32[2048, 2048]" = torch.ops.aten.where.self(lt, scalar_tensor, full_1);  lt = scalar_tensor = full_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:91, code: expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    slice_3: "f32[1, 2048]" = torch.ops.aten.slice.Tensor(full, 0, 0, 9223372036854775807)
    unsqueeze_2: "f32[1, 1, 2048]" = torch.ops.aten.unsqueeze.default(slice_3, 1);  slice_3 = None
    unsqueeze_3: "f32[1, 1, 1, 2048]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 2);  unsqueeze_2 = None
    slice_4: "f32[1, 1, 1, 2048]" = torch.ops.aten.slice.Tensor(unsqueeze_3, 3, 0, 9223372036854775807);  unsqueeze_3 = None
    expand_1: "f32[1, 1, 2048, 2048]" = torch.ops.aten.expand.default(slice_4, [1, 1, 2048, 2048]);  slice_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:93, code: inverted_mask = 1.0 - expanded_mask
    sub: "f32[1, 1, 2048, 2048]" = torch.ops.aten.sub.Tensor(1.0, expand_1);  expand_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:95, code: return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
    convert_element_type: "b8[1, 1, 2048, 2048]" = torch.ops.prims.convert_element_type.default(sub, torch.bool)
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_1: "f32[1, 1, 2048, 2048]" = torch.ops.aten.where.self(convert_element_type, scalar_tensor_1, sub);  convert_element_type = scalar_tensor_1 = sub = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:551, code: expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
    unsqueeze_4: "f32[1, 2048, 2048]" = torch.ops.aten.unsqueeze.default(where, 0);  where = None
    unsqueeze_5: "f32[1, 1, 2048, 2048]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 1);  unsqueeze_4 = None
    slice_5: "f32[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(unsqueeze_5, 2, 0, 9223372036854775807);  unsqueeze_5 = None
    slice_6: "f32[1, 1, 2048, 2048]" = torch.ops.aten.slice.Tensor(slice_5, 3, 0, 9223372036854775807);  slice_5 = None
    expand_2: "f32[1, 1, 2048, 2048]" = torch.ops.aten.expand.default(slice_6, [1, 1, 2048, 2048]);  slice_6 = None
    add_1: "f32[1, 1, 2048, 2048]" = torch.ops.aten.add.Tensor(where_1, expand_2);  where_1 = expand_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:111, code: attention_mask = attention_mask.long()
    convert_element_type_1: "i64[1, 2048]" = torch.ops.prims.convert_element_type.default(full, torch.int64);  full = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:114, code: positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1
    cumsum: "i64[1, 2048]" = torch.ops.aten.cumsum.default(convert_element_type_1, 1)
    mul: "i64[1, 2048]" = torch.ops.aten.mul.Tensor(cumsum, convert_element_type_1);  cumsum = convert_element_type_1 = None
    sub_1: "i64[1, 2048]" = torch.ops.aten.sub.Tensor(mul, 1);  mul = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:117, code: positions = positions[:, past_key_values_length:]
    slice_7: "i64[1, 2048]" = torch.ops.aten.slice.Tensor(sub_1, 0, 0, 9223372036854775807);  sub_1 = None
    slice_8: "i64[1, 2048]" = torch.ops.aten.slice.Tensor(slice_7, 1, 0, 9223372036854775807);  slice_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:119, code: return super().forward(positions + self.offset)
    add_2: "i64[1, 2048]" = torch.ops.aten.add.Tensor(slice_8, 2);  slice_8 = None
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/sparse.py:163, code: return F.embedding(
    embedding_1: "f32[1, 2048, 768]" = torch.ops.aten.embedding.default(arg0_1, add_2);  arg0_1 = add_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:658, code: hidden_states = inputs_embeds + pos_embeds
    add_3: "f32[1, 2048, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:327, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 2048, 1]" = var_mean[0]
    getitem_1: "f32[1, 2048, 1]" = var_mean[1];  var_mean = None
    add_4: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    sub_2: "f32[1, 2048, 768]" = torch.ops.aten.sub.Tensor(add_3, getitem_1);  getitem_1 = None
    mul_1: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt);  sub_2 = rsqrt = None
    mul_2: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_1, arg2_1);  mul_1 = arg2_1 = None
    add_5: "f32[1, 2048, 768]" = torch.ops.aten.add.Tensor(mul_2, arg3_1);  mul_2 = arg3_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:173, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_2: "f32[2048, 768]" = torch.ops.aten.view.default(add_5, [2048, 768])
    permute: "f32[768, 768]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
    addmm: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg5_1, view_2, permute);  arg5_1 = view_2 = permute = None
    view_3: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm, [1, 2048, 768]);  addmm = None
    mul_3: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(view_3, 0.125);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:191, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_4: "f32[2048, 768]" = torch.ops.aten.view.default(add_5, [2048, 768])
    permute_1: "f32[768, 768]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
    addmm_1: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg7_1, view_4, permute_1);  arg7_1 = view_4 = permute_1 = None
    view_5: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_1, [1, 2048, 768]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_6: "f32[1, 2048, 12, 64]" = torch.ops.aten.view.default(view_5, [1, -1, 12, 64]);  view_5 = None
    permute_2: "f32[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
    clone: "f32[1, 12, 2048, 64]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:192, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_7: "f32[2048, 768]" = torch.ops.aten.view.default(add_5, [2048, 768]);  add_5 = None
    permute_3: "f32[768, 768]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
    addmm_2: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg9_1, view_7, permute_3);  arg9_1 = view_7 = permute_3 = None
    view_8: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_2, [1, 2048, 768]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_9: "f32[1, 2048, 12, 64]" = torch.ops.aten.view.default(view_8, [1, -1, 12, 64]);  view_8 = None
    permute_4: "f32[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
    clone_1: "f32[1, 12, 2048, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_10: "f32[1, 2048, 12, 64]" = torch.ops.aten.view.default(mul_3, [1, 2048, 12, 64]);  mul_3 = None
    permute_5: "f32[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
    clone_2: "f32[1, 12, 2048, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:205, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_11: "f32[12, 2048, 64]" = torch.ops.aten.view.default(clone_2, [12, -1, 64]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:206, code: key_states = key_states.view(*proj_shape)
    view_12: "f32[12, 2048, 64]" = torch.ops.aten.view.default(clone, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:207, code: value_states = value_states.view(*proj_shape)
    view_13: "f32[12, 2048, 64]" = torch.ops.aten.view.default(clone_1, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:210, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_6: "f32[12, 64, 2048]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
    bmm: "f32[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_11, permute_6);  view_11 = permute_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:223, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_14: "f32[1, 12, 2048, 2048]" = torch.ops.aten.view.default(bmm, [1, 12, 2048, 2048]);  bmm = None
    add_6: "f32[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_14, add_1);  view_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:225, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    _tensor_constant0 = self._tensor_constant0
    lift_fresh_copy: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:224, code: attn_weights = torch.max(
    maximum: "f32[1, 12, 2048, 2048]" = torch.ops.aten.maximum.default(add_6, lift_fresh_copy);  add_6 = lift_fresh_copy = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:227, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_15: "f32[12, 2048, 2048]" = torch.ops.aten.view.default(maximum, [12, 2048, 2048]);  maximum = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:233, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax: "f32[12, 2048, 1]" = torch.ops.aten.amax.default(view_15, [-1], True)
    sub_3: "f32[12, 2048, 2048]" = torch.ops.aten.sub.Tensor(view_15, amax);  view_15 = amax = None
    exp: "f32[12, 2048, 2048]" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
    sum_1: "f32[12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:254, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_3: "f32[12, 2048, 2048]" = torch.ops.aten.clone.default(div);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:256, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_1: "f32[12, 2048, 64]" = torch.ops.aten.bmm.default(clone_3, view_13);  clone_3 = view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:264, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_16: "f32[1, 12, 2048, 64]" = torch.ops.aten.view.default(bmm_1, [1, 12, 2048, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:265, code: attn_output = attn_output.transpose(1, 2)
    permute_7: "f32[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:269, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_4: "f32[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_17: "f32[1, 2048, 768]" = torch.ops.aten.view.default(clone_4, [1, 2048, 768]);  clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:271, code: attn_output = self.out_proj(attn_output)
    view_18: "f32[2048, 768]" = torch.ops.aten.view.default(view_17, [2048, 768]);  view_17 = None
    permute_8: "f32[768, 768]" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
    addmm_3: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg11_1, view_18, permute_8);  arg11_1 = view_18 = permute_8 = None
    view_19: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_3, [1, 2048, 768]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:337, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_5: "f32[1, 2048, 768]" = torch.ops.aten.clone.default(view_19);  view_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:338, code: hidden_states = residual + hidden_states
    add_7: "f32[1, 2048, 768]" = torch.ops.aten.add.Tensor(add_3, clone_5);  add_3 = clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:346, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
    view_20: "f32[2048, 768]" = torch.ops.aten.view.default(add_7, [-1, 768]);  add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:351, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_1 = torch.ops.aten.var_mean.correction(view_20, [1], correction = 0, keepdim = True)
    getitem_2: "f32[2048, 1]" = var_mean_1[0]
    getitem_3: "f32[2048, 1]" = var_mean_1[1];  var_mean_1 = None
    add_8: "f32[2048, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
    rsqrt_1: "f32[2048, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
    sub_4: "f32[2048, 768]" = torch.ops.aten.sub.Tensor(view_20, getitem_3);  getitem_3 = None
    mul_4: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_1);  sub_4 = rsqrt_1 = None
    mul_5: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(mul_4, arg12_1);  mul_4 = arg12_1 = None
    add_9: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mul_5, arg13_1);  mul_5 = arg13_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:353, code: hidden_states = self.fc1(hidden_states)
    permute_9: "f32[768, 3072]" = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
    addmm_4: "f32[2048, 3072]" = torch.ops.aten.addmm.default(arg15_1, add_9, permute_9);  arg15_1 = add_9 = permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:354, code: hidden_states = self.activation_fn(hidden_states)
    relu: "f32[2048, 3072]" = torch.ops.aten.relu.default(addmm_4);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:356, code: hidden_states = self.fc2(hidden_states)
    permute_10: "f32[3072, 768]" = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
    addmm_5: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg17_1, relu, permute_10);  arg17_1 = relu = permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:357, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_6: "f32[2048, 768]" = torch.ops.aten.clone.default(addmm_5);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:359, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
    add_10: "f32[2048, 768]" = torch.ops.aten.add.Tensor(view_20, clone_6);  view_20 = clone_6 = None
    view_21: "f32[1, 2048, 768]" = torch.ops.aten.view.default(add_10, [1, 2048, 768]);  add_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:327, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_2 = torch.ops.aten.var_mean.correction(view_21, [2], correction = 0, keepdim = True)
    getitem_4: "f32[1, 2048, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 2048, 1]" = var_mean_2[1];  var_mean_2 = None
    add_11: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
    rsqrt_2: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_5: "f32[1, 2048, 768]" = torch.ops.aten.sub.Tensor(view_21, getitem_5);  getitem_5 = None
    mul_6: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_2);  sub_5 = rsqrt_2 = None
    mul_7: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_6, arg18_1);  mul_6 = arg18_1 = None
    add_12: "f32[1, 2048, 768]" = torch.ops.aten.add.Tensor(mul_7, arg19_1);  mul_7 = arg19_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:173, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_22: "f32[2048, 768]" = torch.ops.aten.view.default(add_12, [2048, 768])
    permute_11: "f32[768, 768]" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
    addmm_6: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg21_1, view_22, permute_11);  arg21_1 = view_22 = permute_11 = None
    view_23: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_6, [1, 2048, 768]);  addmm_6 = None
    mul_8: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(view_23, 0.125);  view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:191, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_24: "f32[2048, 768]" = torch.ops.aten.view.default(add_12, [2048, 768])
    permute_12: "f32[768, 768]" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
    addmm_7: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg23_1, view_24, permute_12);  arg23_1 = view_24 = permute_12 = None
    view_25: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_7, [1, 2048, 768]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_26: "f32[1, 2048, 12, 64]" = torch.ops.aten.view.default(view_25, [1, -1, 12, 64]);  view_25 = None
    permute_13: "f32[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
    clone_7: "f32[1, 12, 2048, 64]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:192, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_27: "f32[2048, 768]" = torch.ops.aten.view.default(add_12, [2048, 768]);  add_12 = None
    permute_14: "f32[768, 768]" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
    addmm_8: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg25_1, view_27, permute_14);  arg25_1 = view_27 = permute_14 = None
    view_28: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_8, [1, 2048, 768]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_29: "f32[1, 2048, 12, 64]" = torch.ops.aten.view.default(view_28, [1, -1, 12, 64]);  view_28 = None
    permute_15: "f32[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
    clone_8: "f32[1, 12, 2048, 64]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_30: "f32[1, 2048, 12, 64]" = torch.ops.aten.view.default(mul_8, [1, 2048, 12, 64]);  mul_8 = None
    permute_16: "f32[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
    clone_9: "f32[1, 12, 2048, 64]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:205, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_31: "f32[12, 2048, 64]" = torch.ops.aten.view.default(clone_9, [12, -1, 64]);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:206, code: key_states = key_states.view(*proj_shape)
    view_32: "f32[12, 2048, 64]" = torch.ops.aten.view.default(clone_7, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:207, code: value_states = value_states.view(*proj_shape)
    view_33: "f32[12, 2048, 64]" = torch.ops.aten.view.default(clone_8, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:210, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_17: "f32[12, 64, 2048]" = torch.ops.aten.permute.default(view_32, [0, 2, 1]);  view_32 = None
    bmm_2: "f32[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_31, permute_17);  view_31 = permute_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:223, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_34: "f32[1, 12, 2048, 2048]" = torch.ops.aten.view.default(bmm_2, [1, 12, 2048, 2048]);  bmm_2 = None
    add_13: "f32[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_34, add_1);  view_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:225, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    _tensor_constant1 = self._tensor_constant1
    lift_fresh_copy_1: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant1);  _tensor_constant1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:224, code: attn_weights = torch.max(
    maximum_1: "f32[1, 12, 2048, 2048]" = torch.ops.aten.maximum.default(add_13, lift_fresh_copy_1);  add_13 = lift_fresh_copy_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:227, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_35: "f32[12, 2048, 2048]" = torch.ops.aten.view.default(maximum_1, [12, 2048, 2048]);  maximum_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:233, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_1: "f32[12, 2048, 1]" = torch.ops.aten.amax.default(view_35, [-1], True)
    sub_6: "f32[12, 2048, 2048]" = torch.ops.aten.sub.Tensor(view_35, amax_1);  view_35 = amax_1 = None
    exp_1: "f32[12, 2048, 2048]" = torch.ops.aten.exp.default(sub_6);  sub_6 = None
    sum_2: "f32[12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_1: "f32[12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:254, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_10: "f32[12, 2048, 2048]" = torch.ops.aten.clone.default(div_1);  div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:256, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_3: "f32[12, 2048, 64]" = torch.ops.aten.bmm.default(clone_10, view_33);  clone_10 = view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:264, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_36: "f32[1, 12, 2048, 64]" = torch.ops.aten.view.default(bmm_3, [1, 12, 2048, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:265, code: attn_output = attn_output.transpose(1, 2)
    permute_18: "f32[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:269, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_11: "f32[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
    view_37: "f32[1, 2048, 768]" = torch.ops.aten.view.default(clone_11, [1, 2048, 768]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:271, code: attn_output = self.out_proj(attn_output)
    view_38: "f32[2048, 768]" = torch.ops.aten.view.default(view_37, [2048, 768]);  view_37 = None
    permute_19: "f32[768, 768]" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
    addmm_9: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg27_1, view_38, permute_19);  arg27_1 = view_38 = permute_19 = None
    view_39: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_9, [1, 2048, 768]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:337, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_12: "f32[1, 2048, 768]" = torch.ops.aten.clone.default(view_39);  view_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:338, code: hidden_states = residual + hidden_states
    add_14: "f32[1, 2048, 768]" = torch.ops.aten.add.Tensor(view_21, clone_12);  view_21 = clone_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:346, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
    view_40: "f32[2048, 768]" = torch.ops.aten.view.default(add_14, [-1, 768]);  add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:351, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_3 = torch.ops.aten.var_mean.correction(view_40, [1], correction = 0, keepdim = True)
    getitem_6: "f32[2048, 1]" = var_mean_3[0]
    getitem_7: "f32[2048, 1]" = var_mean_3[1];  var_mean_3 = None
    add_15: "f32[2048, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
    rsqrt_3: "f32[2048, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
    sub_7: "f32[2048, 768]" = torch.ops.aten.sub.Tensor(view_40, getitem_7);  getitem_7 = None
    mul_9: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_3);  sub_7 = rsqrt_3 = None
    mul_10: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(mul_9, arg28_1);  mul_9 = arg28_1 = None
    add_16: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mul_10, arg29_1);  mul_10 = arg29_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:353, code: hidden_states = self.fc1(hidden_states)
    permute_20: "f32[768, 3072]" = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
    addmm_10: "f32[2048, 3072]" = torch.ops.aten.addmm.default(arg31_1, add_16, permute_20);  arg31_1 = add_16 = permute_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:354, code: hidden_states = self.activation_fn(hidden_states)
    relu_1: "f32[2048, 3072]" = torch.ops.aten.relu.default(addmm_10);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:356, code: hidden_states = self.fc2(hidden_states)
    permute_21: "f32[3072, 768]" = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
    addmm_11: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg33_1, relu_1, permute_21);  arg33_1 = relu_1 = permute_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:357, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_13: "f32[2048, 768]" = torch.ops.aten.clone.default(addmm_11);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:359, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
    add_17: "f32[2048, 768]" = torch.ops.aten.add.Tensor(view_40, clone_13);  view_40 = clone_13 = None
    view_41: "f32[1, 2048, 768]" = torch.ops.aten.view.default(add_17, [1, 2048, 768]);  add_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:327, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_4 = torch.ops.aten.var_mean.correction(view_41, [2], correction = 0, keepdim = True)
    getitem_8: "f32[1, 2048, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 2048, 1]" = var_mean_4[1];  var_mean_4 = None
    add_18: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
    rsqrt_4: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    sub_8: "f32[1, 2048, 768]" = torch.ops.aten.sub.Tensor(view_41, getitem_9);  getitem_9 = None
    mul_11: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_4);  sub_8 = rsqrt_4 = None
    mul_12: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_11, arg34_1);  mul_11 = arg34_1 = None
    add_19: "f32[1, 2048, 768]" = torch.ops.aten.add.Tensor(mul_12, arg35_1);  mul_12 = arg35_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:173, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_42: "f32[2048, 768]" = torch.ops.aten.view.default(add_19, [2048, 768])
    permute_22: "f32[768, 768]" = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
    addmm_12: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg37_1, view_42, permute_22);  arg37_1 = view_42 = permute_22 = None
    view_43: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_12, [1, 2048, 768]);  addmm_12 = None
    mul_13: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(view_43, 0.125);  view_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:191, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_44: "f32[2048, 768]" = torch.ops.aten.view.default(add_19, [2048, 768])
    permute_23: "f32[768, 768]" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
    addmm_13: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg39_1, view_44, permute_23);  arg39_1 = view_44 = permute_23 = None
    view_45: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_13, [1, 2048, 768]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_46: "f32[1, 2048, 12, 64]" = torch.ops.aten.view.default(view_45, [1, -1, 12, 64]);  view_45 = None
    permute_24: "f32[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None
    clone_14: "f32[1, 12, 2048, 64]" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:192, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_47: "f32[2048, 768]" = torch.ops.aten.view.default(add_19, [2048, 768]);  add_19 = None
    permute_25: "f32[768, 768]" = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
    addmm_14: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg41_1, view_47, permute_25);  arg41_1 = view_47 = permute_25 = None
    view_48: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_14, [1, 2048, 768]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_49: "f32[1, 2048, 12, 64]" = torch.ops.aten.view.default(view_48, [1, -1, 12, 64]);  view_48 = None
    permute_26: "f32[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_49, [0, 2, 1, 3]);  view_49 = None
    clone_15: "f32[1, 12, 2048, 64]" = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_50: "f32[1, 2048, 12, 64]" = torch.ops.aten.view.default(mul_13, [1, 2048, 12, 64]);  mul_13 = None
    permute_27: "f32[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
    clone_16: "f32[1, 12, 2048, 64]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:205, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_51: "f32[12, 2048, 64]" = torch.ops.aten.view.default(clone_16, [12, -1, 64]);  clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:206, code: key_states = key_states.view(*proj_shape)
    view_52: "f32[12, 2048, 64]" = torch.ops.aten.view.default(clone_14, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:207, code: value_states = value_states.view(*proj_shape)
    view_53: "f32[12, 2048, 64]" = torch.ops.aten.view.default(clone_15, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:210, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_28: "f32[12, 64, 2048]" = torch.ops.aten.permute.default(view_52, [0, 2, 1]);  view_52 = None
    bmm_4: "f32[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_51, permute_28);  view_51 = permute_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:223, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_54: "f32[1, 12, 2048, 2048]" = torch.ops.aten.view.default(bmm_4, [1, 12, 2048, 2048]);  bmm_4 = None
    add_20: "f32[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_54, add_1);  view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:225, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    _tensor_constant2 = self._tensor_constant2
    lift_fresh_copy_2: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant2);  _tensor_constant2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:224, code: attn_weights = torch.max(
    maximum_2: "f32[1, 12, 2048, 2048]" = torch.ops.aten.maximum.default(add_20, lift_fresh_copy_2);  add_20 = lift_fresh_copy_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:227, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_55: "f32[12, 2048, 2048]" = torch.ops.aten.view.default(maximum_2, [12, 2048, 2048]);  maximum_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:233, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_2: "f32[12, 2048, 1]" = torch.ops.aten.amax.default(view_55, [-1], True)
    sub_9: "f32[12, 2048, 2048]" = torch.ops.aten.sub.Tensor(view_55, amax_2);  view_55 = amax_2 = None
    exp_2: "f32[12, 2048, 2048]" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
    sum_3: "f32[12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_2: "f32[12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:254, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_17: "f32[12, 2048, 2048]" = torch.ops.aten.clone.default(div_2);  div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:256, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_5: "f32[12, 2048, 64]" = torch.ops.aten.bmm.default(clone_17, view_53);  clone_17 = view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:264, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_56: "f32[1, 12, 2048, 64]" = torch.ops.aten.view.default(bmm_5, [1, 12, 2048, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:265, code: attn_output = attn_output.transpose(1, 2)
    permute_29: "f32[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_56, [0, 2, 1, 3]);  view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:269, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_18: "f32[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    view_57: "f32[1, 2048, 768]" = torch.ops.aten.view.default(clone_18, [1, 2048, 768]);  clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:271, code: attn_output = self.out_proj(attn_output)
    view_58: "f32[2048, 768]" = torch.ops.aten.view.default(view_57, [2048, 768]);  view_57 = None
    permute_30: "f32[768, 768]" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
    addmm_15: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg43_1, view_58, permute_30);  arg43_1 = view_58 = permute_30 = None
    view_59: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_15, [1, 2048, 768]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:337, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_19: "f32[1, 2048, 768]" = torch.ops.aten.clone.default(view_59);  view_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:338, code: hidden_states = residual + hidden_states
    add_21: "f32[1, 2048, 768]" = torch.ops.aten.add.Tensor(view_41, clone_19);  view_41 = clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:346, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
    view_60: "f32[2048, 768]" = torch.ops.aten.view.default(add_21, [-1, 768]);  add_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:351, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_5 = torch.ops.aten.var_mean.correction(view_60, [1], correction = 0, keepdim = True)
    getitem_10: "f32[2048, 1]" = var_mean_5[0]
    getitem_11: "f32[2048, 1]" = var_mean_5[1];  var_mean_5 = None
    add_22: "f32[2048, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
    rsqrt_5: "f32[2048, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
    sub_10: "f32[2048, 768]" = torch.ops.aten.sub.Tensor(view_60, getitem_11);  getitem_11 = None
    mul_14: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_5);  sub_10 = rsqrt_5 = None
    mul_15: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(mul_14, arg44_1);  mul_14 = arg44_1 = None
    add_23: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mul_15, arg45_1);  mul_15 = arg45_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:353, code: hidden_states = self.fc1(hidden_states)
    permute_31: "f32[768, 3072]" = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
    addmm_16: "f32[2048, 3072]" = torch.ops.aten.addmm.default(arg47_1, add_23, permute_31);  arg47_1 = add_23 = permute_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:354, code: hidden_states = self.activation_fn(hidden_states)
    relu_2: "f32[2048, 3072]" = torch.ops.aten.relu.default(addmm_16);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:356, code: hidden_states = self.fc2(hidden_states)
    permute_32: "f32[3072, 768]" = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
    addmm_17: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg49_1, relu_2, permute_32);  arg49_1 = relu_2 = permute_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:357, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_20: "f32[2048, 768]" = torch.ops.aten.clone.default(addmm_17);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:359, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
    add_24: "f32[2048, 768]" = torch.ops.aten.add.Tensor(view_60, clone_20);  view_60 = clone_20 = None
    view_61: "f32[1, 2048, 768]" = torch.ops.aten.view.default(add_24, [1, 2048, 768]);  add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:327, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_6 = torch.ops.aten.var_mean.correction(view_61, [2], correction = 0, keepdim = True)
    getitem_12: "f32[1, 2048, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 2048, 1]" = var_mean_6[1];  var_mean_6 = None
    add_25: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
    rsqrt_6: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
    sub_11: "f32[1, 2048, 768]" = torch.ops.aten.sub.Tensor(view_61, getitem_13);  getitem_13 = None
    mul_16: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_6);  sub_11 = rsqrt_6 = None
    mul_17: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_16, arg50_1);  mul_16 = arg50_1 = None
    add_26: "f32[1, 2048, 768]" = torch.ops.aten.add.Tensor(mul_17, arg51_1);  mul_17 = arg51_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:173, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_62: "f32[2048, 768]" = torch.ops.aten.view.default(add_26, [2048, 768])
    permute_33: "f32[768, 768]" = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
    addmm_18: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg53_1, view_62, permute_33);  arg53_1 = view_62 = permute_33 = None
    view_63: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_18, [1, 2048, 768]);  addmm_18 = None
    mul_18: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(view_63, 0.125);  view_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:191, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_64: "f32[2048, 768]" = torch.ops.aten.view.default(add_26, [2048, 768])
    permute_34: "f32[768, 768]" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
    addmm_19: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg55_1, view_64, permute_34);  arg55_1 = view_64 = permute_34 = None
    view_65: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_19, [1, 2048, 768]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_66: "f32[1, 2048, 12, 64]" = torch.ops.aten.view.default(view_65, [1, -1, 12, 64]);  view_65 = None
    permute_35: "f32[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None
    clone_21: "f32[1, 12, 2048, 64]" = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:192, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_67: "f32[2048, 768]" = torch.ops.aten.view.default(add_26, [2048, 768]);  add_26 = None
    permute_36: "f32[768, 768]" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
    addmm_20: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg57_1, view_67, permute_36);  arg57_1 = view_67 = permute_36 = None
    view_68: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_20, [1, 2048, 768]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_69: "f32[1, 2048, 12, 64]" = torch.ops.aten.view.default(view_68, [1, -1, 12, 64]);  view_68 = None
    permute_37: "f32[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_69, [0, 2, 1, 3]);  view_69 = None
    clone_22: "f32[1, 12, 2048, 64]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_70: "f32[1, 2048, 12, 64]" = torch.ops.aten.view.default(mul_18, [1, 2048, 12, 64]);  mul_18 = None
    permute_38: "f32[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None
    clone_23: "f32[1, 12, 2048, 64]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:205, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_71: "f32[12, 2048, 64]" = torch.ops.aten.view.default(clone_23, [12, -1, 64]);  clone_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:206, code: key_states = key_states.view(*proj_shape)
    view_72: "f32[12, 2048, 64]" = torch.ops.aten.view.default(clone_21, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:207, code: value_states = value_states.view(*proj_shape)
    view_73: "f32[12, 2048, 64]" = torch.ops.aten.view.default(clone_22, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:210, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_39: "f32[12, 64, 2048]" = torch.ops.aten.permute.default(view_72, [0, 2, 1]);  view_72 = None
    bmm_6: "f32[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_71, permute_39);  view_71 = permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:223, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_74: "f32[1, 12, 2048, 2048]" = torch.ops.aten.view.default(bmm_6, [1, 12, 2048, 2048]);  bmm_6 = None
    add_27: "f32[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_74, add_1);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:225, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    _tensor_constant3 = self._tensor_constant3
    lift_fresh_copy_3: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant3);  _tensor_constant3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:224, code: attn_weights = torch.max(
    maximum_3: "f32[1, 12, 2048, 2048]" = torch.ops.aten.maximum.default(add_27, lift_fresh_copy_3);  add_27 = lift_fresh_copy_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:227, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_75: "f32[12, 2048, 2048]" = torch.ops.aten.view.default(maximum_3, [12, 2048, 2048]);  maximum_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:233, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_3: "f32[12, 2048, 1]" = torch.ops.aten.amax.default(view_75, [-1], True)
    sub_12: "f32[12, 2048, 2048]" = torch.ops.aten.sub.Tensor(view_75, amax_3);  view_75 = amax_3 = None
    exp_3: "f32[12, 2048, 2048]" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
    sum_4: "f32[12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_3: "f32[12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:254, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_24: "f32[12, 2048, 2048]" = torch.ops.aten.clone.default(div_3);  div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:256, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_7: "f32[12, 2048, 64]" = torch.ops.aten.bmm.default(clone_24, view_73);  clone_24 = view_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:264, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_76: "f32[1, 12, 2048, 64]" = torch.ops.aten.view.default(bmm_7, [1, 12, 2048, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:265, code: attn_output = attn_output.transpose(1, 2)
    permute_40: "f32[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:269, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_25: "f32[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
    view_77: "f32[1, 2048, 768]" = torch.ops.aten.view.default(clone_25, [1, 2048, 768]);  clone_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:271, code: attn_output = self.out_proj(attn_output)
    view_78: "f32[2048, 768]" = torch.ops.aten.view.default(view_77, [2048, 768]);  view_77 = None
    permute_41: "f32[768, 768]" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
    addmm_21: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg59_1, view_78, permute_41);  arg59_1 = view_78 = permute_41 = None
    view_79: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_21, [1, 2048, 768]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:337, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_26: "f32[1, 2048, 768]" = torch.ops.aten.clone.default(view_79);  view_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:338, code: hidden_states = residual + hidden_states
    add_28: "f32[1, 2048, 768]" = torch.ops.aten.add.Tensor(view_61, clone_26);  view_61 = clone_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:346, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
    view_80: "f32[2048, 768]" = torch.ops.aten.view.default(add_28, [-1, 768]);  add_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:351, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_7 = torch.ops.aten.var_mean.correction(view_80, [1], correction = 0, keepdim = True)
    getitem_14: "f32[2048, 1]" = var_mean_7[0]
    getitem_15: "f32[2048, 1]" = var_mean_7[1];  var_mean_7 = None
    add_29: "f32[2048, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
    rsqrt_7: "f32[2048, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
    sub_13: "f32[2048, 768]" = torch.ops.aten.sub.Tensor(view_80, getitem_15);  getitem_15 = None
    mul_19: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_7);  sub_13 = rsqrt_7 = None
    mul_20: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(mul_19, arg60_1);  mul_19 = arg60_1 = None
    add_30: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mul_20, arg61_1);  mul_20 = arg61_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:353, code: hidden_states = self.fc1(hidden_states)
    permute_42: "f32[768, 3072]" = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
    addmm_22: "f32[2048, 3072]" = torch.ops.aten.addmm.default(arg63_1, add_30, permute_42);  arg63_1 = add_30 = permute_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:354, code: hidden_states = self.activation_fn(hidden_states)
    relu_3: "f32[2048, 3072]" = torch.ops.aten.relu.default(addmm_22);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:356, code: hidden_states = self.fc2(hidden_states)
    permute_43: "f32[3072, 768]" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
    addmm_23: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg65_1, relu_3, permute_43);  arg65_1 = relu_3 = permute_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:357, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_27: "f32[2048, 768]" = torch.ops.aten.clone.default(addmm_23);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:359, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
    add_31: "f32[2048, 768]" = torch.ops.aten.add.Tensor(view_80, clone_27);  view_80 = clone_27 = None
    view_81: "f32[1, 2048, 768]" = torch.ops.aten.view.default(add_31, [1, 2048, 768]);  add_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:327, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_8 = torch.ops.aten.var_mean.correction(view_81, [2], correction = 0, keepdim = True)
    getitem_16: "f32[1, 2048, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 2048, 1]" = var_mean_8[1];  var_mean_8 = None
    add_32: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
    rsqrt_8: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    sub_14: "f32[1, 2048, 768]" = torch.ops.aten.sub.Tensor(view_81, getitem_17);  getitem_17 = None
    mul_21: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_8);  sub_14 = rsqrt_8 = None
    mul_22: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_21, arg66_1);  mul_21 = arg66_1 = None
    add_33: "f32[1, 2048, 768]" = torch.ops.aten.add.Tensor(mul_22, arg67_1);  mul_22 = arg67_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:173, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_82: "f32[2048, 768]" = torch.ops.aten.view.default(add_33, [2048, 768])
    permute_44: "f32[768, 768]" = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
    addmm_24: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg69_1, view_82, permute_44);  arg69_1 = view_82 = permute_44 = None
    view_83: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_24, [1, 2048, 768]);  addmm_24 = None
    mul_23: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(view_83, 0.125);  view_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:191, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_84: "f32[2048, 768]" = torch.ops.aten.view.default(add_33, [2048, 768])
    permute_45: "f32[768, 768]" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
    addmm_25: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg71_1, view_84, permute_45);  arg71_1 = view_84 = permute_45 = None
    view_85: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_25, [1, 2048, 768]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_86: "f32[1, 2048, 12, 64]" = torch.ops.aten.view.default(view_85, [1, -1, 12, 64]);  view_85 = None
    permute_46: "f32[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_86, [0, 2, 1, 3]);  view_86 = None
    clone_28: "f32[1, 12, 2048, 64]" = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:192, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_87: "f32[2048, 768]" = torch.ops.aten.view.default(add_33, [2048, 768]);  add_33 = None
    permute_47: "f32[768, 768]" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
    addmm_26: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg73_1, view_87, permute_47);  arg73_1 = view_87 = permute_47 = None
    view_88: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_26, [1, 2048, 768]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_89: "f32[1, 2048, 12, 64]" = torch.ops.aten.view.default(view_88, [1, -1, 12, 64]);  view_88 = None
    permute_48: "f32[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_89, [0, 2, 1, 3]);  view_89 = None
    clone_29: "f32[1, 12, 2048, 64]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_90: "f32[1, 2048, 12, 64]" = torch.ops.aten.view.default(mul_23, [1, 2048, 12, 64]);  mul_23 = None
    permute_49: "f32[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None
    clone_30: "f32[1, 12, 2048, 64]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:205, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_91: "f32[12, 2048, 64]" = torch.ops.aten.view.default(clone_30, [12, -1, 64]);  clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:206, code: key_states = key_states.view(*proj_shape)
    view_92: "f32[12, 2048, 64]" = torch.ops.aten.view.default(clone_28, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:207, code: value_states = value_states.view(*proj_shape)
    view_93: "f32[12, 2048, 64]" = torch.ops.aten.view.default(clone_29, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:210, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_50: "f32[12, 64, 2048]" = torch.ops.aten.permute.default(view_92, [0, 2, 1]);  view_92 = None
    bmm_8: "f32[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_91, permute_50);  view_91 = permute_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:223, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_94: "f32[1, 12, 2048, 2048]" = torch.ops.aten.view.default(bmm_8, [1, 12, 2048, 2048]);  bmm_8 = None
    add_34: "f32[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_94, add_1);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:225, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    _tensor_constant4 = self._tensor_constant4
    lift_fresh_copy_4: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant4);  _tensor_constant4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:224, code: attn_weights = torch.max(
    maximum_4: "f32[1, 12, 2048, 2048]" = torch.ops.aten.maximum.default(add_34, lift_fresh_copy_4);  add_34 = lift_fresh_copy_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:227, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_95: "f32[12, 2048, 2048]" = torch.ops.aten.view.default(maximum_4, [12, 2048, 2048]);  maximum_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:233, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_4: "f32[12, 2048, 1]" = torch.ops.aten.amax.default(view_95, [-1], True)
    sub_15: "f32[12, 2048, 2048]" = torch.ops.aten.sub.Tensor(view_95, amax_4);  view_95 = amax_4 = None
    exp_4: "f32[12, 2048, 2048]" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
    sum_5: "f32[12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_4: "f32[12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:254, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_31: "f32[12, 2048, 2048]" = torch.ops.aten.clone.default(div_4);  div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:256, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_9: "f32[12, 2048, 64]" = torch.ops.aten.bmm.default(clone_31, view_93);  clone_31 = view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:264, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_96: "f32[1, 12, 2048, 64]" = torch.ops.aten.view.default(bmm_9, [1, 12, 2048, 64]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:265, code: attn_output = attn_output.transpose(1, 2)
    permute_51: "f32[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:269, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_32: "f32[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
    view_97: "f32[1, 2048, 768]" = torch.ops.aten.view.default(clone_32, [1, 2048, 768]);  clone_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:271, code: attn_output = self.out_proj(attn_output)
    view_98: "f32[2048, 768]" = torch.ops.aten.view.default(view_97, [2048, 768]);  view_97 = None
    permute_52: "f32[768, 768]" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
    addmm_27: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg75_1, view_98, permute_52);  arg75_1 = view_98 = permute_52 = None
    view_99: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_27, [1, 2048, 768]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:337, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_33: "f32[1, 2048, 768]" = torch.ops.aten.clone.default(view_99);  view_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:338, code: hidden_states = residual + hidden_states
    add_35: "f32[1, 2048, 768]" = torch.ops.aten.add.Tensor(view_81, clone_33);  view_81 = clone_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:346, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
    view_100: "f32[2048, 768]" = torch.ops.aten.view.default(add_35, [-1, 768]);  add_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:351, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_9 = torch.ops.aten.var_mean.correction(view_100, [1], correction = 0, keepdim = True)
    getitem_18: "f32[2048, 1]" = var_mean_9[0]
    getitem_19: "f32[2048, 1]" = var_mean_9[1];  var_mean_9 = None
    add_36: "f32[2048, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
    rsqrt_9: "f32[2048, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    sub_16: "f32[2048, 768]" = torch.ops.aten.sub.Tensor(view_100, getitem_19);  getitem_19 = None
    mul_24: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_9);  sub_16 = rsqrt_9 = None
    mul_25: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(mul_24, arg76_1);  mul_24 = arg76_1 = None
    add_37: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mul_25, arg77_1);  mul_25 = arg77_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:353, code: hidden_states = self.fc1(hidden_states)
    permute_53: "f32[768, 3072]" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
    addmm_28: "f32[2048, 3072]" = torch.ops.aten.addmm.default(arg79_1, add_37, permute_53);  arg79_1 = add_37 = permute_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:354, code: hidden_states = self.activation_fn(hidden_states)
    relu_4: "f32[2048, 3072]" = torch.ops.aten.relu.default(addmm_28);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:356, code: hidden_states = self.fc2(hidden_states)
    permute_54: "f32[3072, 768]" = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
    addmm_29: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg81_1, relu_4, permute_54);  arg81_1 = relu_4 = permute_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:357, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_34: "f32[2048, 768]" = torch.ops.aten.clone.default(addmm_29);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:359, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
    add_38: "f32[2048, 768]" = torch.ops.aten.add.Tensor(view_100, clone_34);  view_100 = clone_34 = None
    view_101: "f32[1, 2048, 768]" = torch.ops.aten.view.default(add_38, [1, 2048, 768]);  add_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:327, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_10 = torch.ops.aten.var_mean.correction(view_101, [2], correction = 0, keepdim = True)
    getitem_20: "f32[1, 2048, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 2048, 1]" = var_mean_10[1];  var_mean_10 = None
    add_39: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
    rsqrt_10: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
    sub_17: "f32[1, 2048, 768]" = torch.ops.aten.sub.Tensor(view_101, getitem_21);  getitem_21 = None
    mul_26: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_10);  sub_17 = rsqrt_10 = None
    mul_27: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_26, arg82_1);  mul_26 = arg82_1 = None
    add_40: "f32[1, 2048, 768]" = torch.ops.aten.add.Tensor(mul_27, arg83_1);  mul_27 = arg83_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:173, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_102: "f32[2048, 768]" = torch.ops.aten.view.default(add_40, [2048, 768])
    permute_55: "f32[768, 768]" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
    addmm_30: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg85_1, view_102, permute_55);  arg85_1 = view_102 = permute_55 = None
    view_103: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_30, [1, 2048, 768]);  addmm_30 = None
    mul_28: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(view_103, 0.125);  view_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:191, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_104: "f32[2048, 768]" = torch.ops.aten.view.default(add_40, [2048, 768])
    permute_56: "f32[768, 768]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
    addmm_31: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg87_1, view_104, permute_56);  arg87_1 = view_104 = permute_56 = None
    view_105: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_31, [1, 2048, 768]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_106: "f32[1, 2048, 12, 64]" = torch.ops.aten.view.default(view_105, [1, -1, 12, 64]);  view_105 = None
    permute_57: "f32[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_106, [0, 2, 1, 3]);  view_106 = None
    clone_35: "f32[1, 12, 2048, 64]" = torch.ops.aten.clone.default(permute_57, memory_format = torch.contiguous_format);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:192, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_107: "f32[2048, 768]" = torch.ops.aten.view.default(add_40, [2048, 768]);  add_40 = None
    permute_58: "f32[768, 768]" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
    addmm_32: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg89_1, view_107, permute_58);  arg89_1 = view_107 = permute_58 = None
    view_108: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_32, [1, 2048, 768]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_109: "f32[1, 2048, 12, 64]" = torch.ops.aten.view.default(view_108, [1, -1, 12, 64]);  view_108 = None
    permute_59: "f32[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_109, [0, 2, 1, 3]);  view_109 = None
    clone_36: "f32[1, 12, 2048, 64]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_110: "f32[1, 2048, 12, 64]" = torch.ops.aten.view.default(mul_28, [1, 2048, 12, 64]);  mul_28 = None
    permute_60: "f32[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_110, [0, 2, 1, 3]);  view_110 = None
    clone_37: "f32[1, 12, 2048, 64]" = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:205, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_111: "f32[12, 2048, 64]" = torch.ops.aten.view.default(clone_37, [12, -1, 64]);  clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:206, code: key_states = key_states.view(*proj_shape)
    view_112: "f32[12, 2048, 64]" = torch.ops.aten.view.default(clone_35, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:207, code: value_states = value_states.view(*proj_shape)
    view_113: "f32[12, 2048, 64]" = torch.ops.aten.view.default(clone_36, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:210, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_61: "f32[12, 64, 2048]" = torch.ops.aten.permute.default(view_112, [0, 2, 1]);  view_112 = None
    bmm_10: "f32[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_111, permute_61);  view_111 = permute_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:223, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_114: "f32[1, 12, 2048, 2048]" = torch.ops.aten.view.default(bmm_10, [1, 12, 2048, 2048]);  bmm_10 = None
    add_41: "f32[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_114, add_1);  view_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:225, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    _tensor_constant5 = self._tensor_constant5
    lift_fresh_copy_5: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant5);  _tensor_constant5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:224, code: attn_weights = torch.max(
    maximum_5: "f32[1, 12, 2048, 2048]" = torch.ops.aten.maximum.default(add_41, lift_fresh_copy_5);  add_41 = lift_fresh_copy_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:227, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_115: "f32[12, 2048, 2048]" = torch.ops.aten.view.default(maximum_5, [12, 2048, 2048]);  maximum_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:233, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_5: "f32[12, 2048, 1]" = torch.ops.aten.amax.default(view_115, [-1], True)
    sub_18: "f32[12, 2048, 2048]" = torch.ops.aten.sub.Tensor(view_115, amax_5);  view_115 = amax_5 = None
    exp_5: "f32[12, 2048, 2048]" = torch.ops.aten.exp.default(sub_18);  sub_18 = None
    sum_6: "f32[12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_5: "f32[12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:254, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_38: "f32[12, 2048, 2048]" = torch.ops.aten.clone.default(div_5);  div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:256, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_11: "f32[12, 2048, 64]" = torch.ops.aten.bmm.default(clone_38, view_113);  clone_38 = view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:264, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_116: "f32[1, 12, 2048, 64]" = torch.ops.aten.view.default(bmm_11, [1, 12, 2048, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:265, code: attn_output = attn_output.transpose(1, 2)
    permute_62: "f32[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_116, [0, 2, 1, 3]);  view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:269, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_39: "f32[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    view_117: "f32[1, 2048, 768]" = torch.ops.aten.view.default(clone_39, [1, 2048, 768]);  clone_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:271, code: attn_output = self.out_proj(attn_output)
    view_118: "f32[2048, 768]" = torch.ops.aten.view.default(view_117, [2048, 768]);  view_117 = None
    permute_63: "f32[768, 768]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
    addmm_33: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg91_1, view_118, permute_63);  arg91_1 = view_118 = permute_63 = None
    view_119: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_33, [1, 2048, 768]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:337, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_40: "f32[1, 2048, 768]" = torch.ops.aten.clone.default(view_119);  view_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:338, code: hidden_states = residual + hidden_states
    add_42: "f32[1, 2048, 768]" = torch.ops.aten.add.Tensor(view_101, clone_40);  view_101 = clone_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:346, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
    view_120: "f32[2048, 768]" = torch.ops.aten.view.default(add_42, [-1, 768]);  add_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:351, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_11 = torch.ops.aten.var_mean.correction(view_120, [1], correction = 0, keepdim = True)
    getitem_22: "f32[2048, 1]" = var_mean_11[0]
    getitem_23: "f32[2048, 1]" = var_mean_11[1];  var_mean_11 = None
    add_43: "f32[2048, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
    rsqrt_11: "f32[2048, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
    sub_19: "f32[2048, 768]" = torch.ops.aten.sub.Tensor(view_120, getitem_23);  getitem_23 = None
    mul_29: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_11);  sub_19 = rsqrt_11 = None
    mul_30: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(mul_29, arg92_1);  mul_29 = arg92_1 = None
    add_44: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mul_30, arg93_1);  mul_30 = arg93_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:353, code: hidden_states = self.fc1(hidden_states)
    permute_64: "f32[768, 3072]" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
    addmm_34: "f32[2048, 3072]" = torch.ops.aten.addmm.default(arg95_1, add_44, permute_64);  arg95_1 = add_44 = permute_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:354, code: hidden_states = self.activation_fn(hidden_states)
    relu_5: "f32[2048, 3072]" = torch.ops.aten.relu.default(addmm_34);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:356, code: hidden_states = self.fc2(hidden_states)
    permute_65: "f32[3072, 768]" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
    addmm_35: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg97_1, relu_5, permute_65);  arg97_1 = relu_5 = permute_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:357, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_41: "f32[2048, 768]" = torch.ops.aten.clone.default(addmm_35);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:359, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
    add_45: "f32[2048, 768]" = torch.ops.aten.add.Tensor(view_120, clone_41);  view_120 = clone_41 = None
    view_121: "f32[1, 2048, 768]" = torch.ops.aten.view.default(add_45, [1, 2048, 768]);  add_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:327, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_12 = torch.ops.aten.var_mean.correction(view_121, [2], correction = 0, keepdim = True)
    getitem_24: "f32[1, 2048, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 2048, 1]" = var_mean_12[1];  var_mean_12 = None
    add_46: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
    rsqrt_12: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
    sub_20: "f32[1, 2048, 768]" = torch.ops.aten.sub.Tensor(view_121, getitem_25);  getitem_25 = None
    mul_31: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_12);  sub_20 = rsqrt_12 = None
    mul_32: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_31, arg98_1);  mul_31 = arg98_1 = None
    add_47: "f32[1, 2048, 768]" = torch.ops.aten.add.Tensor(mul_32, arg99_1);  mul_32 = arg99_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:173, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_122: "f32[2048, 768]" = torch.ops.aten.view.default(add_47, [2048, 768])
    permute_66: "f32[768, 768]" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
    addmm_36: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg101_1, view_122, permute_66);  arg101_1 = view_122 = permute_66 = None
    view_123: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_36, [1, 2048, 768]);  addmm_36 = None
    mul_33: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(view_123, 0.125);  view_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:191, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_124: "f32[2048, 768]" = torch.ops.aten.view.default(add_47, [2048, 768])
    permute_67: "f32[768, 768]" = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
    addmm_37: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg103_1, view_124, permute_67);  arg103_1 = view_124 = permute_67 = None
    view_125: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_37, [1, 2048, 768]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_126: "f32[1, 2048, 12, 64]" = torch.ops.aten.view.default(view_125, [1, -1, 12, 64]);  view_125 = None
    permute_68: "f32[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
    clone_42: "f32[1, 12, 2048, 64]" = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:192, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_127: "f32[2048, 768]" = torch.ops.aten.view.default(add_47, [2048, 768]);  add_47 = None
    permute_69: "f32[768, 768]" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
    addmm_38: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg105_1, view_127, permute_69);  arg105_1 = view_127 = permute_69 = None
    view_128: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_38, [1, 2048, 768]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_129: "f32[1, 2048, 12, 64]" = torch.ops.aten.view.default(view_128, [1, -1, 12, 64]);  view_128 = None
    permute_70: "f32[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_129, [0, 2, 1, 3]);  view_129 = None
    clone_43: "f32[1, 12, 2048, 64]" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_130: "f32[1, 2048, 12, 64]" = torch.ops.aten.view.default(mul_33, [1, 2048, 12, 64]);  mul_33 = None
    permute_71: "f32[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None
    clone_44: "f32[1, 12, 2048, 64]" = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:205, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_131: "f32[12, 2048, 64]" = torch.ops.aten.view.default(clone_44, [12, -1, 64]);  clone_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:206, code: key_states = key_states.view(*proj_shape)
    view_132: "f32[12, 2048, 64]" = torch.ops.aten.view.default(clone_42, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:207, code: value_states = value_states.view(*proj_shape)
    view_133: "f32[12, 2048, 64]" = torch.ops.aten.view.default(clone_43, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:210, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_72: "f32[12, 64, 2048]" = torch.ops.aten.permute.default(view_132, [0, 2, 1]);  view_132 = None
    bmm_12: "f32[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_131, permute_72);  view_131 = permute_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:223, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_134: "f32[1, 12, 2048, 2048]" = torch.ops.aten.view.default(bmm_12, [1, 12, 2048, 2048]);  bmm_12 = None
    add_48: "f32[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_134, add_1);  view_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:225, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    _tensor_constant6 = self._tensor_constant6
    lift_fresh_copy_6: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant6);  _tensor_constant6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:224, code: attn_weights = torch.max(
    maximum_6: "f32[1, 12, 2048, 2048]" = torch.ops.aten.maximum.default(add_48, lift_fresh_copy_6);  add_48 = lift_fresh_copy_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:227, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_135: "f32[12, 2048, 2048]" = torch.ops.aten.view.default(maximum_6, [12, 2048, 2048]);  maximum_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:233, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_6: "f32[12, 2048, 1]" = torch.ops.aten.amax.default(view_135, [-1], True)
    sub_21: "f32[12, 2048, 2048]" = torch.ops.aten.sub.Tensor(view_135, amax_6);  view_135 = amax_6 = None
    exp_6: "f32[12, 2048, 2048]" = torch.ops.aten.exp.default(sub_21);  sub_21 = None
    sum_7: "f32[12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_6: "f32[12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:254, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_45: "f32[12, 2048, 2048]" = torch.ops.aten.clone.default(div_6);  div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:256, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_13: "f32[12, 2048, 64]" = torch.ops.aten.bmm.default(clone_45, view_133);  clone_45 = view_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:264, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_136: "f32[1, 12, 2048, 64]" = torch.ops.aten.view.default(bmm_13, [1, 12, 2048, 64]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:265, code: attn_output = attn_output.transpose(1, 2)
    permute_73: "f32[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:269, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_46: "f32[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
    view_137: "f32[1, 2048, 768]" = torch.ops.aten.view.default(clone_46, [1, 2048, 768]);  clone_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:271, code: attn_output = self.out_proj(attn_output)
    view_138: "f32[2048, 768]" = torch.ops.aten.view.default(view_137, [2048, 768]);  view_137 = None
    permute_74: "f32[768, 768]" = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
    addmm_39: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg107_1, view_138, permute_74);  arg107_1 = view_138 = permute_74 = None
    view_139: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_39, [1, 2048, 768]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:337, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_47: "f32[1, 2048, 768]" = torch.ops.aten.clone.default(view_139);  view_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:338, code: hidden_states = residual + hidden_states
    add_49: "f32[1, 2048, 768]" = torch.ops.aten.add.Tensor(view_121, clone_47);  view_121 = clone_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:346, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
    view_140: "f32[2048, 768]" = torch.ops.aten.view.default(add_49, [-1, 768]);  add_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:351, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_13 = torch.ops.aten.var_mean.correction(view_140, [1], correction = 0, keepdim = True)
    getitem_26: "f32[2048, 1]" = var_mean_13[0]
    getitem_27: "f32[2048, 1]" = var_mean_13[1];  var_mean_13 = None
    add_50: "f32[2048, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
    rsqrt_13: "f32[2048, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
    sub_22: "f32[2048, 768]" = torch.ops.aten.sub.Tensor(view_140, getitem_27);  getitem_27 = None
    mul_34: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_13);  sub_22 = rsqrt_13 = None
    mul_35: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(mul_34, arg108_1);  mul_34 = arg108_1 = None
    add_51: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mul_35, arg109_1);  mul_35 = arg109_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:353, code: hidden_states = self.fc1(hidden_states)
    permute_75: "f32[768, 3072]" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
    addmm_40: "f32[2048, 3072]" = torch.ops.aten.addmm.default(arg111_1, add_51, permute_75);  arg111_1 = add_51 = permute_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:354, code: hidden_states = self.activation_fn(hidden_states)
    relu_6: "f32[2048, 3072]" = torch.ops.aten.relu.default(addmm_40);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:356, code: hidden_states = self.fc2(hidden_states)
    permute_76: "f32[3072, 768]" = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
    addmm_41: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg113_1, relu_6, permute_76);  arg113_1 = relu_6 = permute_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:357, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_48: "f32[2048, 768]" = torch.ops.aten.clone.default(addmm_41);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:359, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
    add_52: "f32[2048, 768]" = torch.ops.aten.add.Tensor(view_140, clone_48);  view_140 = clone_48 = None
    view_141: "f32[1, 2048, 768]" = torch.ops.aten.view.default(add_52, [1, 2048, 768]);  add_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:327, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_14 = torch.ops.aten.var_mean.correction(view_141, [2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 2048, 1]" = var_mean_14[0]
    getitem_29: "f32[1, 2048, 1]" = var_mean_14[1];  var_mean_14 = None
    add_53: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
    rsqrt_14: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    sub_23: "f32[1, 2048, 768]" = torch.ops.aten.sub.Tensor(view_141, getitem_29);  getitem_29 = None
    mul_36: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_14);  sub_23 = rsqrt_14 = None
    mul_37: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_36, arg114_1);  mul_36 = arg114_1 = None
    add_54: "f32[1, 2048, 768]" = torch.ops.aten.add.Tensor(mul_37, arg115_1);  mul_37 = arg115_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:173, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_142: "f32[2048, 768]" = torch.ops.aten.view.default(add_54, [2048, 768])
    permute_77: "f32[768, 768]" = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
    addmm_42: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg117_1, view_142, permute_77);  arg117_1 = view_142 = permute_77 = None
    view_143: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_42, [1, 2048, 768]);  addmm_42 = None
    mul_38: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(view_143, 0.125);  view_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:191, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_144: "f32[2048, 768]" = torch.ops.aten.view.default(add_54, [2048, 768])
    permute_78: "f32[768, 768]" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
    addmm_43: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg119_1, view_144, permute_78);  arg119_1 = view_144 = permute_78 = None
    view_145: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_43, [1, 2048, 768]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_146: "f32[1, 2048, 12, 64]" = torch.ops.aten.view.default(view_145, [1, -1, 12, 64]);  view_145 = None
    permute_79: "f32[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
    clone_49: "f32[1, 12, 2048, 64]" = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:192, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_147: "f32[2048, 768]" = torch.ops.aten.view.default(add_54, [2048, 768]);  add_54 = None
    permute_80: "f32[768, 768]" = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
    addmm_44: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg121_1, view_147, permute_80);  arg121_1 = view_147 = permute_80 = None
    view_148: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_44, [1, 2048, 768]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_149: "f32[1, 2048, 12, 64]" = torch.ops.aten.view.default(view_148, [1, -1, 12, 64]);  view_148 = None
    permute_81: "f32[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_149, [0, 2, 1, 3]);  view_149 = None
    clone_50: "f32[1, 12, 2048, 64]" = torch.ops.aten.clone.default(permute_81, memory_format = torch.contiguous_format);  permute_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_150: "f32[1, 2048, 12, 64]" = torch.ops.aten.view.default(mul_38, [1, 2048, 12, 64]);  mul_38 = None
    permute_82: "f32[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_150, [0, 2, 1, 3]);  view_150 = None
    clone_51: "f32[1, 12, 2048, 64]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:205, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_151: "f32[12, 2048, 64]" = torch.ops.aten.view.default(clone_51, [12, -1, 64]);  clone_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:206, code: key_states = key_states.view(*proj_shape)
    view_152: "f32[12, 2048, 64]" = torch.ops.aten.view.default(clone_49, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:207, code: value_states = value_states.view(*proj_shape)
    view_153: "f32[12, 2048, 64]" = torch.ops.aten.view.default(clone_50, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:210, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_83: "f32[12, 64, 2048]" = torch.ops.aten.permute.default(view_152, [0, 2, 1]);  view_152 = None
    bmm_14: "f32[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_151, permute_83);  view_151 = permute_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:223, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_154: "f32[1, 12, 2048, 2048]" = torch.ops.aten.view.default(bmm_14, [1, 12, 2048, 2048]);  bmm_14 = None
    add_55: "f32[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_154, add_1);  view_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:225, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    _tensor_constant7 = self._tensor_constant7
    lift_fresh_copy_7: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant7);  _tensor_constant7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:224, code: attn_weights = torch.max(
    maximum_7: "f32[1, 12, 2048, 2048]" = torch.ops.aten.maximum.default(add_55, lift_fresh_copy_7);  add_55 = lift_fresh_copy_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:227, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_155: "f32[12, 2048, 2048]" = torch.ops.aten.view.default(maximum_7, [12, 2048, 2048]);  maximum_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:233, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_7: "f32[12, 2048, 1]" = torch.ops.aten.amax.default(view_155, [-1], True)
    sub_24: "f32[12, 2048, 2048]" = torch.ops.aten.sub.Tensor(view_155, amax_7);  view_155 = amax_7 = None
    exp_7: "f32[12, 2048, 2048]" = torch.ops.aten.exp.default(sub_24);  sub_24 = None
    sum_8: "f32[12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_7: "f32[12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:254, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_52: "f32[12, 2048, 2048]" = torch.ops.aten.clone.default(div_7);  div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:256, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_15: "f32[12, 2048, 64]" = torch.ops.aten.bmm.default(clone_52, view_153);  clone_52 = view_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:264, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_156: "f32[1, 12, 2048, 64]" = torch.ops.aten.view.default(bmm_15, [1, 12, 2048, 64]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:265, code: attn_output = attn_output.transpose(1, 2)
    permute_84: "f32[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:269, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_53: "f32[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
    view_157: "f32[1, 2048, 768]" = torch.ops.aten.view.default(clone_53, [1, 2048, 768]);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:271, code: attn_output = self.out_proj(attn_output)
    view_158: "f32[2048, 768]" = torch.ops.aten.view.default(view_157, [2048, 768]);  view_157 = None
    permute_85: "f32[768, 768]" = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
    addmm_45: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg123_1, view_158, permute_85);  arg123_1 = view_158 = permute_85 = None
    view_159: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_45, [1, 2048, 768]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:337, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_54: "f32[1, 2048, 768]" = torch.ops.aten.clone.default(view_159);  view_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:338, code: hidden_states = residual + hidden_states
    add_56: "f32[1, 2048, 768]" = torch.ops.aten.add.Tensor(view_141, clone_54);  view_141 = clone_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:346, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
    view_160: "f32[2048, 768]" = torch.ops.aten.view.default(add_56, [-1, 768]);  add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:351, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_15 = torch.ops.aten.var_mean.correction(view_160, [1], correction = 0, keepdim = True)
    getitem_30: "f32[2048, 1]" = var_mean_15[0]
    getitem_31: "f32[2048, 1]" = var_mean_15[1];  var_mean_15 = None
    add_57: "f32[2048, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
    rsqrt_15: "f32[2048, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
    sub_25: "f32[2048, 768]" = torch.ops.aten.sub.Tensor(view_160, getitem_31);  getitem_31 = None
    mul_39: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_15);  sub_25 = rsqrt_15 = None
    mul_40: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(mul_39, arg124_1);  mul_39 = arg124_1 = None
    add_58: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mul_40, arg125_1);  mul_40 = arg125_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:353, code: hidden_states = self.fc1(hidden_states)
    permute_86: "f32[768, 3072]" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
    addmm_46: "f32[2048, 3072]" = torch.ops.aten.addmm.default(arg127_1, add_58, permute_86);  arg127_1 = add_58 = permute_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:354, code: hidden_states = self.activation_fn(hidden_states)
    relu_7: "f32[2048, 3072]" = torch.ops.aten.relu.default(addmm_46);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:356, code: hidden_states = self.fc2(hidden_states)
    permute_87: "f32[3072, 768]" = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
    addmm_47: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg129_1, relu_7, permute_87);  arg129_1 = relu_7 = permute_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:357, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_55: "f32[2048, 768]" = torch.ops.aten.clone.default(addmm_47);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:359, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
    add_59: "f32[2048, 768]" = torch.ops.aten.add.Tensor(view_160, clone_55);  view_160 = clone_55 = None
    view_161: "f32[1, 2048, 768]" = torch.ops.aten.view.default(add_59, [1, 2048, 768]);  add_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:327, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_16 = torch.ops.aten.var_mean.correction(view_161, [2], correction = 0, keepdim = True)
    getitem_32: "f32[1, 2048, 1]" = var_mean_16[0]
    getitem_33: "f32[1, 2048, 1]" = var_mean_16[1];  var_mean_16 = None
    add_60: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
    rsqrt_16: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    sub_26: "f32[1, 2048, 768]" = torch.ops.aten.sub.Tensor(view_161, getitem_33);  getitem_33 = None
    mul_41: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_16);  sub_26 = rsqrt_16 = None
    mul_42: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_41, arg130_1);  mul_41 = arg130_1 = None
    add_61: "f32[1, 2048, 768]" = torch.ops.aten.add.Tensor(mul_42, arg131_1);  mul_42 = arg131_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:173, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_162: "f32[2048, 768]" = torch.ops.aten.view.default(add_61, [2048, 768])
    permute_88: "f32[768, 768]" = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
    addmm_48: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg133_1, view_162, permute_88);  arg133_1 = view_162 = permute_88 = None
    view_163: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_48, [1, 2048, 768]);  addmm_48 = None
    mul_43: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(view_163, 0.125);  view_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:191, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_164: "f32[2048, 768]" = torch.ops.aten.view.default(add_61, [2048, 768])
    permute_89: "f32[768, 768]" = torch.ops.aten.permute.default(arg134_1, [1, 0]);  arg134_1 = None
    addmm_49: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg135_1, view_164, permute_89);  arg135_1 = view_164 = permute_89 = None
    view_165: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_49, [1, 2048, 768]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_166: "f32[1, 2048, 12, 64]" = torch.ops.aten.view.default(view_165, [1, -1, 12, 64]);  view_165 = None
    permute_90: "f32[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_166, [0, 2, 1, 3]);  view_166 = None
    clone_56: "f32[1, 12, 2048, 64]" = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:192, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_167: "f32[2048, 768]" = torch.ops.aten.view.default(add_61, [2048, 768]);  add_61 = None
    permute_91: "f32[768, 768]" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
    addmm_50: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg137_1, view_167, permute_91);  arg137_1 = view_167 = permute_91 = None
    view_168: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_50, [1, 2048, 768]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_169: "f32[1, 2048, 12, 64]" = torch.ops.aten.view.default(view_168, [1, -1, 12, 64]);  view_168 = None
    permute_92: "f32[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_169, [0, 2, 1, 3]);  view_169 = None
    clone_57: "f32[1, 12, 2048, 64]" = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_170: "f32[1, 2048, 12, 64]" = torch.ops.aten.view.default(mul_43, [1, 2048, 12, 64]);  mul_43 = None
    permute_93: "f32[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_170, [0, 2, 1, 3]);  view_170 = None
    clone_58: "f32[1, 12, 2048, 64]" = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:205, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_171: "f32[12, 2048, 64]" = torch.ops.aten.view.default(clone_58, [12, -1, 64]);  clone_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:206, code: key_states = key_states.view(*proj_shape)
    view_172: "f32[12, 2048, 64]" = torch.ops.aten.view.default(clone_56, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:207, code: value_states = value_states.view(*proj_shape)
    view_173: "f32[12, 2048, 64]" = torch.ops.aten.view.default(clone_57, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:210, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_94: "f32[12, 64, 2048]" = torch.ops.aten.permute.default(view_172, [0, 2, 1]);  view_172 = None
    bmm_16: "f32[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_171, permute_94);  view_171 = permute_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:223, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_174: "f32[1, 12, 2048, 2048]" = torch.ops.aten.view.default(bmm_16, [1, 12, 2048, 2048]);  bmm_16 = None
    add_62: "f32[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_174, add_1);  view_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:225, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    _tensor_constant8 = self._tensor_constant8
    lift_fresh_copy_8: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant8);  _tensor_constant8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:224, code: attn_weights = torch.max(
    maximum_8: "f32[1, 12, 2048, 2048]" = torch.ops.aten.maximum.default(add_62, lift_fresh_copy_8);  add_62 = lift_fresh_copy_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:227, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_175: "f32[12, 2048, 2048]" = torch.ops.aten.view.default(maximum_8, [12, 2048, 2048]);  maximum_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:233, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_8: "f32[12, 2048, 1]" = torch.ops.aten.amax.default(view_175, [-1], True)
    sub_27: "f32[12, 2048, 2048]" = torch.ops.aten.sub.Tensor(view_175, amax_8);  view_175 = amax_8 = None
    exp_8: "f32[12, 2048, 2048]" = torch.ops.aten.exp.default(sub_27);  sub_27 = None
    sum_9: "f32[12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_8: "f32[12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:254, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_59: "f32[12, 2048, 2048]" = torch.ops.aten.clone.default(div_8);  div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:256, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_17: "f32[12, 2048, 64]" = torch.ops.aten.bmm.default(clone_59, view_173);  clone_59 = view_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:264, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_176: "f32[1, 12, 2048, 64]" = torch.ops.aten.view.default(bmm_17, [1, 12, 2048, 64]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:265, code: attn_output = attn_output.transpose(1, 2)
    permute_95: "f32[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_176, [0, 2, 1, 3]);  view_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:269, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_60: "f32[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
    view_177: "f32[1, 2048, 768]" = torch.ops.aten.view.default(clone_60, [1, 2048, 768]);  clone_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:271, code: attn_output = self.out_proj(attn_output)
    view_178: "f32[2048, 768]" = torch.ops.aten.view.default(view_177, [2048, 768]);  view_177 = None
    permute_96: "f32[768, 768]" = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
    addmm_51: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg139_1, view_178, permute_96);  arg139_1 = view_178 = permute_96 = None
    view_179: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_51, [1, 2048, 768]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:337, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_61: "f32[1, 2048, 768]" = torch.ops.aten.clone.default(view_179);  view_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:338, code: hidden_states = residual + hidden_states
    add_63: "f32[1, 2048, 768]" = torch.ops.aten.add.Tensor(view_161, clone_61);  view_161 = clone_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:346, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
    view_180: "f32[2048, 768]" = torch.ops.aten.view.default(add_63, [-1, 768]);  add_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:351, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_17 = torch.ops.aten.var_mean.correction(view_180, [1], correction = 0, keepdim = True)
    getitem_34: "f32[2048, 1]" = var_mean_17[0]
    getitem_35: "f32[2048, 1]" = var_mean_17[1];  var_mean_17 = None
    add_64: "f32[2048, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
    rsqrt_17: "f32[2048, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
    sub_28: "f32[2048, 768]" = torch.ops.aten.sub.Tensor(view_180, getitem_35);  getitem_35 = None
    mul_44: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_17);  sub_28 = rsqrt_17 = None
    mul_45: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(mul_44, arg140_1);  mul_44 = arg140_1 = None
    add_65: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mul_45, arg141_1);  mul_45 = arg141_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:353, code: hidden_states = self.fc1(hidden_states)
    permute_97: "f32[768, 3072]" = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
    addmm_52: "f32[2048, 3072]" = torch.ops.aten.addmm.default(arg143_1, add_65, permute_97);  arg143_1 = add_65 = permute_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:354, code: hidden_states = self.activation_fn(hidden_states)
    relu_8: "f32[2048, 3072]" = torch.ops.aten.relu.default(addmm_52);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:356, code: hidden_states = self.fc2(hidden_states)
    permute_98: "f32[3072, 768]" = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
    addmm_53: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg145_1, relu_8, permute_98);  arg145_1 = relu_8 = permute_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:357, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_62: "f32[2048, 768]" = torch.ops.aten.clone.default(addmm_53);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:359, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
    add_66: "f32[2048, 768]" = torch.ops.aten.add.Tensor(view_180, clone_62);  view_180 = clone_62 = None
    view_181: "f32[1, 2048, 768]" = torch.ops.aten.view.default(add_66, [1, 2048, 768]);  add_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:327, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_18 = torch.ops.aten.var_mean.correction(view_181, [2], correction = 0, keepdim = True)
    getitem_36: "f32[1, 2048, 1]" = var_mean_18[0]
    getitem_37: "f32[1, 2048, 1]" = var_mean_18[1];  var_mean_18 = None
    add_67: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
    rsqrt_18: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
    sub_29: "f32[1, 2048, 768]" = torch.ops.aten.sub.Tensor(view_181, getitem_37);  getitem_37 = None
    mul_46: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_18);  sub_29 = rsqrt_18 = None
    mul_47: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_46, arg146_1);  mul_46 = arg146_1 = None
    add_68: "f32[1, 2048, 768]" = torch.ops.aten.add.Tensor(mul_47, arg147_1);  mul_47 = arg147_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:173, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_182: "f32[2048, 768]" = torch.ops.aten.view.default(add_68, [2048, 768])
    permute_99: "f32[768, 768]" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
    addmm_54: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg149_1, view_182, permute_99);  arg149_1 = view_182 = permute_99 = None
    view_183: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_54, [1, 2048, 768]);  addmm_54 = None
    mul_48: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(view_183, 0.125);  view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:191, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_184: "f32[2048, 768]" = torch.ops.aten.view.default(add_68, [2048, 768])
    permute_100: "f32[768, 768]" = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
    addmm_55: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg151_1, view_184, permute_100);  arg151_1 = view_184 = permute_100 = None
    view_185: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_55, [1, 2048, 768]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_186: "f32[1, 2048, 12, 64]" = torch.ops.aten.view.default(view_185, [1, -1, 12, 64]);  view_185 = None
    permute_101: "f32[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
    clone_63: "f32[1, 12, 2048, 64]" = torch.ops.aten.clone.default(permute_101, memory_format = torch.contiguous_format);  permute_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:192, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_187: "f32[2048, 768]" = torch.ops.aten.view.default(add_68, [2048, 768]);  add_68 = None
    permute_102: "f32[768, 768]" = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
    addmm_56: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg153_1, view_187, permute_102);  arg153_1 = view_187 = permute_102 = None
    view_188: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_56, [1, 2048, 768]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_189: "f32[1, 2048, 12, 64]" = torch.ops.aten.view.default(view_188, [1, -1, 12, 64]);  view_188 = None
    permute_103: "f32[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_189, [0, 2, 1, 3]);  view_189 = None
    clone_64: "f32[1, 12, 2048, 64]" = torch.ops.aten.clone.default(permute_103, memory_format = torch.contiguous_format);  permute_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_190: "f32[1, 2048, 12, 64]" = torch.ops.aten.view.default(mul_48, [1, 2048, 12, 64]);  mul_48 = None
    permute_104: "f32[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_190, [0, 2, 1, 3]);  view_190 = None
    clone_65: "f32[1, 12, 2048, 64]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:205, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_191: "f32[12, 2048, 64]" = torch.ops.aten.view.default(clone_65, [12, -1, 64]);  clone_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:206, code: key_states = key_states.view(*proj_shape)
    view_192: "f32[12, 2048, 64]" = torch.ops.aten.view.default(clone_63, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:207, code: value_states = value_states.view(*proj_shape)
    view_193: "f32[12, 2048, 64]" = torch.ops.aten.view.default(clone_64, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:210, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_105: "f32[12, 64, 2048]" = torch.ops.aten.permute.default(view_192, [0, 2, 1]);  view_192 = None
    bmm_18: "f32[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_191, permute_105);  view_191 = permute_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:223, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_194: "f32[1, 12, 2048, 2048]" = torch.ops.aten.view.default(bmm_18, [1, 12, 2048, 2048]);  bmm_18 = None
    add_69: "f32[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_194, add_1);  view_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:225, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    _tensor_constant9 = self._tensor_constant9
    lift_fresh_copy_9: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant9);  _tensor_constant9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:224, code: attn_weights = torch.max(
    maximum_9: "f32[1, 12, 2048, 2048]" = torch.ops.aten.maximum.default(add_69, lift_fresh_copy_9);  add_69 = lift_fresh_copy_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:227, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_195: "f32[12, 2048, 2048]" = torch.ops.aten.view.default(maximum_9, [12, 2048, 2048]);  maximum_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:233, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_9: "f32[12, 2048, 1]" = torch.ops.aten.amax.default(view_195, [-1], True)
    sub_30: "f32[12, 2048, 2048]" = torch.ops.aten.sub.Tensor(view_195, amax_9);  view_195 = amax_9 = None
    exp_9: "f32[12, 2048, 2048]" = torch.ops.aten.exp.default(sub_30);  sub_30 = None
    sum_10: "f32[12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_9: "f32[12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:254, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_66: "f32[12, 2048, 2048]" = torch.ops.aten.clone.default(div_9);  div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:256, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_19: "f32[12, 2048, 64]" = torch.ops.aten.bmm.default(clone_66, view_193);  clone_66 = view_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:264, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_196: "f32[1, 12, 2048, 64]" = torch.ops.aten.view.default(bmm_19, [1, 12, 2048, 64]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:265, code: attn_output = attn_output.transpose(1, 2)
    permute_106: "f32[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_196, [0, 2, 1, 3]);  view_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:269, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_67: "f32[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
    view_197: "f32[1, 2048, 768]" = torch.ops.aten.view.default(clone_67, [1, 2048, 768]);  clone_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:271, code: attn_output = self.out_proj(attn_output)
    view_198: "f32[2048, 768]" = torch.ops.aten.view.default(view_197, [2048, 768]);  view_197 = None
    permute_107: "f32[768, 768]" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
    addmm_57: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg155_1, view_198, permute_107);  arg155_1 = view_198 = permute_107 = None
    view_199: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_57, [1, 2048, 768]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:337, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_68: "f32[1, 2048, 768]" = torch.ops.aten.clone.default(view_199);  view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:338, code: hidden_states = residual + hidden_states
    add_70: "f32[1, 2048, 768]" = torch.ops.aten.add.Tensor(view_181, clone_68);  view_181 = clone_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:346, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
    view_200: "f32[2048, 768]" = torch.ops.aten.view.default(add_70, [-1, 768]);  add_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:351, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_19 = torch.ops.aten.var_mean.correction(view_200, [1], correction = 0, keepdim = True)
    getitem_38: "f32[2048, 1]" = var_mean_19[0]
    getitem_39: "f32[2048, 1]" = var_mean_19[1];  var_mean_19 = None
    add_71: "f32[2048, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
    rsqrt_19: "f32[2048, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
    sub_31: "f32[2048, 768]" = torch.ops.aten.sub.Tensor(view_200, getitem_39);  getitem_39 = None
    mul_49: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_19);  sub_31 = rsqrt_19 = None
    mul_50: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(mul_49, arg156_1);  mul_49 = arg156_1 = None
    add_72: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mul_50, arg157_1);  mul_50 = arg157_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:353, code: hidden_states = self.fc1(hidden_states)
    permute_108: "f32[768, 3072]" = torch.ops.aten.permute.default(arg158_1, [1, 0]);  arg158_1 = None
    addmm_58: "f32[2048, 3072]" = torch.ops.aten.addmm.default(arg159_1, add_72, permute_108);  arg159_1 = add_72 = permute_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:354, code: hidden_states = self.activation_fn(hidden_states)
    relu_9: "f32[2048, 3072]" = torch.ops.aten.relu.default(addmm_58);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:356, code: hidden_states = self.fc2(hidden_states)
    permute_109: "f32[3072, 768]" = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
    addmm_59: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg161_1, relu_9, permute_109);  arg161_1 = relu_9 = permute_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:357, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_69: "f32[2048, 768]" = torch.ops.aten.clone.default(addmm_59);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:359, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
    add_73: "f32[2048, 768]" = torch.ops.aten.add.Tensor(view_200, clone_69);  view_200 = clone_69 = None
    view_201: "f32[1, 2048, 768]" = torch.ops.aten.view.default(add_73, [1, 2048, 768]);  add_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:327, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_20 = torch.ops.aten.var_mean.correction(view_201, [2], correction = 0, keepdim = True)
    getitem_40: "f32[1, 2048, 1]" = var_mean_20[0]
    getitem_41: "f32[1, 2048, 1]" = var_mean_20[1];  var_mean_20 = None
    add_74: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
    rsqrt_20: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    sub_32: "f32[1, 2048, 768]" = torch.ops.aten.sub.Tensor(view_201, getitem_41);  getitem_41 = None
    mul_51: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_20);  sub_32 = rsqrt_20 = None
    mul_52: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_51, arg162_1);  mul_51 = arg162_1 = None
    add_75: "f32[1, 2048, 768]" = torch.ops.aten.add.Tensor(mul_52, arg163_1);  mul_52 = arg163_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:173, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_202: "f32[2048, 768]" = torch.ops.aten.view.default(add_75, [2048, 768])
    permute_110: "f32[768, 768]" = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
    addmm_60: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg165_1, view_202, permute_110);  arg165_1 = view_202 = permute_110 = None
    view_203: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_60, [1, 2048, 768]);  addmm_60 = None
    mul_53: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(view_203, 0.125);  view_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:191, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_204: "f32[2048, 768]" = torch.ops.aten.view.default(add_75, [2048, 768])
    permute_111: "f32[768, 768]" = torch.ops.aten.permute.default(arg166_1, [1, 0]);  arg166_1 = None
    addmm_61: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg167_1, view_204, permute_111);  arg167_1 = view_204 = permute_111 = None
    view_205: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_61, [1, 2048, 768]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_206: "f32[1, 2048, 12, 64]" = torch.ops.aten.view.default(view_205, [1, -1, 12, 64]);  view_205 = None
    permute_112: "f32[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
    clone_70: "f32[1, 12, 2048, 64]" = torch.ops.aten.clone.default(permute_112, memory_format = torch.contiguous_format);  permute_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:192, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_207: "f32[2048, 768]" = torch.ops.aten.view.default(add_75, [2048, 768]);  add_75 = None
    permute_113: "f32[768, 768]" = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
    addmm_62: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg169_1, view_207, permute_113);  arg169_1 = view_207 = permute_113 = None
    view_208: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_62, [1, 2048, 768]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_209: "f32[1, 2048, 12, 64]" = torch.ops.aten.view.default(view_208, [1, -1, 12, 64]);  view_208 = None
    permute_114: "f32[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_209, [0, 2, 1, 3]);  view_209 = None
    clone_71: "f32[1, 12, 2048, 64]" = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_210: "f32[1, 2048, 12, 64]" = torch.ops.aten.view.default(mul_53, [1, 2048, 12, 64]);  mul_53 = None
    permute_115: "f32[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_210, [0, 2, 1, 3]);  view_210 = None
    clone_72: "f32[1, 12, 2048, 64]" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:205, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_211: "f32[12, 2048, 64]" = torch.ops.aten.view.default(clone_72, [12, -1, 64]);  clone_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:206, code: key_states = key_states.view(*proj_shape)
    view_212: "f32[12, 2048, 64]" = torch.ops.aten.view.default(clone_70, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:207, code: value_states = value_states.view(*proj_shape)
    view_213: "f32[12, 2048, 64]" = torch.ops.aten.view.default(clone_71, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:210, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_116: "f32[12, 64, 2048]" = torch.ops.aten.permute.default(view_212, [0, 2, 1]);  view_212 = None
    bmm_20: "f32[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_211, permute_116);  view_211 = permute_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:223, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_214: "f32[1, 12, 2048, 2048]" = torch.ops.aten.view.default(bmm_20, [1, 12, 2048, 2048]);  bmm_20 = None
    add_76: "f32[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_214, add_1);  view_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:225, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    _tensor_constant10 = self._tensor_constant10
    lift_fresh_copy_10: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant10);  _tensor_constant10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:224, code: attn_weights = torch.max(
    maximum_10: "f32[1, 12, 2048, 2048]" = torch.ops.aten.maximum.default(add_76, lift_fresh_copy_10);  add_76 = lift_fresh_copy_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:227, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_215: "f32[12, 2048, 2048]" = torch.ops.aten.view.default(maximum_10, [12, 2048, 2048]);  maximum_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:233, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_10: "f32[12, 2048, 1]" = torch.ops.aten.amax.default(view_215, [-1], True)
    sub_33: "f32[12, 2048, 2048]" = torch.ops.aten.sub.Tensor(view_215, amax_10);  view_215 = amax_10 = None
    exp_10: "f32[12, 2048, 2048]" = torch.ops.aten.exp.default(sub_33);  sub_33 = None
    sum_11: "f32[12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_10: "f32[12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:254, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_73: "f32[12, 2048, 2048]" = torch.ops.aten.clone.default(div_10);  div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:256, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_21: "f32[12, 2048, 64]" = torch.ops.aten.bmm.default(clone_73, view_213);  clone_73 = view_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:264, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_216: "f32[1, 12, 2048, 64]" = torch.ops.aten.view.default(bmm_21, [1, 12, 2048, 64]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:265, code: attn_output = attn_output.transpose(1, 2)
    permute_117: "f32[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_216, [0, 2, 1, 3]);  view_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:269, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_74: "f32[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
    view_217: "f32[1, 2048, 768]" = torch.ops.aten.view.default(clone_74, [1, 2048, 768]);  clone_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:271, code: attn_output = self.out_proj(attn_output)
    view_218: "f32[2048, 768]" = torch.ops.aten.view.default(view_217, [2048, 768]);  view_217 = None
    permute_118: "f32[768, 768]" = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
    addmm_63: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg171_1, view_218, permute_118);  arg171_1 = view_218 = permute_118 = None
    view_219: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_63, [1, 2048, 768]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:337, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_75: "f32[1, 2048, 768]" = torch.ops.aten.clone.default(view_219);  view_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:338, code: hidden_states = residual + hidden_states
    add_77: "f32[1, 2048, 768]" = torch.ops.aten.add.Tensor(view_201, clone_75);  view_201 = clone_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:346, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
    view_220: "f32[2048, 768]" = torch.ops.aten.view.default(add_77, [-1, 768]);  add_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:351, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_21 = torch.ops.aten.var_mean.correction(view_220, [1], correction = 0, keepdim = True)
    getitem_42: "f32[2048, 1]" = var_mean_21[0]
    getitem_43: "f32[2048, 1]" = var_mean_21[1];  var_mean_21 = None
    add_78: "f32[2048, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
    rsqrt_21: "f32[2048, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    sub_34: "f32[2048, 768]" = torch.ops.aten.sub.Tensor(view_220, getitem_43);  getitem_43 = None
    mul_54: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_21);  sub_34 = rsqrt_21 = None
    mul_55: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(mul_54, arg172_1);  mul_54 = arg172_1 = None
    add_79: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mul_55, arg173_1);  mul_55 = arg173_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:353, code: hidden_states = self.fc1(hidden_states)
    permute_119: "f32[768, 3072]" = torch.ops.aten.permute.default(arg174_1, [1, 0]);  arg174_1 = None
    addmm_64: "f32[2048, 3072]" = torch.ops.aten.addmm.default(arg175_1, add_79, permute_119);  arg175_1 = add_79 = permute_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:354, code: hidden_states = self.activation_fn(hidden_states)
    relu_10: "f32[2048, 3072]" = torch.ops.aten.relu.default(addmm_64);  addmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:356, code: hidden_states = self.fc2(hidden_states)
    permute_120: "f32[3072, 768]" = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
    addmm_65: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg177_1, relu_10, permute_120);  arg177_1 = relu_10 = permute_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:357, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_76: "f32[2048, 768]" = torch.ops.aten.clone.default(addmm_65);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:359, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
    add_80: "f32[2048, 768]" = torch.ops.aten.add.Tensor(view_220, clone_76);  view_220 = clone_76 = None
    view_221: "f32[1, 2048, 768]" = torch.ops.aten.view.default(add_80, [1, 2048, 768]);  add_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:327, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_22 = torch.ops.aten.var_mean.correction(view_221, [2], correction = 0, keepdim = True)
    getitem_44: "f32[1, 2048, 1]" = var_mean_22[0]
    getitem_45: "f32[1, 2048, 1]" = var_mean_22[1];  var_mean_22 = None
    add_81: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
    rsqrt_22: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
    sub_35: "f32[1, 2048, 768]" = torch.ops.aten.sub.Tensor(view_221, getitem_45);  getitem_45 = None
    mul_56: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_22);  sub_35 = rsqrt_22 = None
    mul_57: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_56, arg178_1);  mul_56 = arg178_1 = None
    add_82: "f32[1, 2048, 768]" = torch.ops.aten.add.Tensor(mul_57, arg179_1);  mul_57 = arg179_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:173, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_222: "f32[2048, 768]" = torch.ops.aten.view.default(add_82, [2048, 768])
    permute_121: "f32[768, 768]" = torch.ops.aten.permute.default(arg180_1, [1, 0]);  arg180_1 = None
    addmm_66: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg181_1, view_222, permute_121);  arg181_1 = view_222 = permute_121 = None
    view_223: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_66, [1, 2048, 768]);  addmm_66 = None
    mul_58: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(view_223, 0.125);  view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:191, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_224: "f32[2048, 768]" = torch.ops.aten.view.default(add_82, [2048, 768])
    permute_122: "f32[768, 768]" = torch.ops.aten.permute.default(arg182_1, [1, 0]);  arg182_1 = None
    addmm_67: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg183_1, view_224, permute_122);  arg183_1 = view_224 = permute_122 = None
    view_225: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_67, [1, 2048, 768]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_226: "f32[1, 2048, 12, 64]" = torch.ops.aten.view.default(view_225, [1, -1, 12, 64]);  view_225 = None
    permute_123: "f32[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
    clone_77: "f32[1, 12, 2048, 64]" = torch.ops.aten.clone.default(permute_123, memory_format = torch.contiguous_format);  permute_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:192, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_227: "f32[2048, 768]" = torch.ops.aten.view.default(add_82, [2048, 768]);  add_82 = None
    permute_124: "f32[768, 768]" = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
    addmm_68: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg185_1, view_227, permute_124);  arg185_1 = view_227 = permute_124 = None
    view_228: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_68, [1, 2048, 768]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_229: "f32[1, 2048, 12, 64]" = torch.ops.aten.view.default(view_228, [1, -1, 12, 64]);  view_228 = None
    permute_125: "f32[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_229, [0, 2, 1, 3]);  view_229 = None
    clone_78: "f32[1, 12, 2048, 64]" = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:153, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_230: "f32[1, 2048, 12, 64]" = torch.ops.aten.view.default(mul_58, [1, 2048, 12, 64]);  mul_58 = None
    permute_126: "f32[1, 12, 2048, 64]" = torch.ops.aten.permute.default(view_230, [0, 2, 1, 3]);  view_230 = None
    clone_79: "f32[1, 12, 2048, 64]" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:205, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_231: "f32[12, 2048, 64]" = torch.ops.aten.view.default(clone_79, [12, -1, 64]);  clone_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:206, code: key_states = key_states.view(*proj_shape)
    view_232: "f32[12, 2048, 64]" = torch.ops.aten.view.default(clone_77, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:207, code: value_states = value_states.view(*proj_shape)
    view_233: "f32[12, 2048, 64]" = torch.ops.aten.view.default(clone_78, [12, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:210, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_127: "f32[12, 64, 2048]" = torch.ops.aten.permute.default(view_232, [0, 2, 1]);  view_232 = None
    bmm_22: "f32[12, 2048, 2048]" = torch.ops.aten.bmm.default(view_231, permute_127);  view_231 = permute_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:223, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_234: "f32[1, 12, 2048, 2048]" = torch.ops.aten.view.default(bmm_22, [1, 12, 2048, 2048]);  bmm_22 = None
    add_83: "f32[1, 12, 2048, 2048]" = torch.ops.aten.add.Tensor(view_234, add_1);  view_234 = add_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:225, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    _tensor_constant11 = self._tensor_constant11
    lift_fresh_copy_11: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant11);  _tensor_constant11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:224, code: attn_weights = torch.max(
    maximum_11: "f32[1, 12, 2048, 2048]" = torch.ops.aten.maximum.default(add_83, lift_fresh_copy_11);  add_83 = lift_fresh_copy_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:227, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_235: "f32[12, 2048, 2048]" = torch.ops.aten.view.default(maximum_11, [12, 2048, 2048]);  maximum_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:233, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_11: "f32[12, 2048, 1]" = torch.ops.aten.amax.default(view_235, [-1], True)
    sub_36: "f32[12, 2048, 2048]" = torch.ops.aten.sub.Tensor(view_235, amax_11);  view_235 = amax_11 = None
    exp_11: "f32[12, 2048, 2048]" = torch.ops.aten.exp.default(sub_36);  sub_36 = None
    sum_12: "f32[12, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_11: "f32[12, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:254, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_80: "f32[12, 2048, 2048]" = torch.ops.aten.clone.default(div_11);  div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:256, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_23: "f32[12, 2048, 64]" = torch.ops.aten.bmm.default(clone_80, view_233);  clone_80 = view_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:264, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_236: "f32[1, 12, 2048, 64]" = torch.ops.aten.view.default(bmm_23, [1, 12, 2048, 64]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:265, code: attn_output = attn_output.transpose(1, 2)
    permute_128: "f32[1, 2048, 12, 64]" = torch.ops.aten.permute.default(view_236, [0, 2, 1, 3]);  view_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:269, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_81: "f32[1, 2048, 12, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
    view_237: "f32[1, 2048, 768]" = torch.ops.aten.view.default(clone_81, [1, 2048, 768]);  clone_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:271, code: attn_output = self.out_proj(attn_output)
    view_238: "f32[2048, 768]" = torch.ops.aten.view.default(view_237, [2048, 768]);  view_237 = None
    permute_129: "f32[768, 768]" = torch.ops.aten.permute.default(arg186_1, [1, 0]);  arg186_1 = None
    addmm_69: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg187_1, view_238, permute_129);  arg187_1 = view_238 = permute_129 = None
    view_239: "f32[1, 2048, 768]" = torch.ops.aten.view.default(addmm_69, [1, 2048, 768]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:337, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_82: "f32[1, 2048, 768]" = torch.ops.aten.clone.default(view_239);  view_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:338, code: hidden_states = residual + hidden_states
    add_84: "f32[1, 2048, 768]" = torch.ops.aten.add.Tensor(view_221, clone_82);  view_221 = clone_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:346, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
    view_240: "f32[2048, 768]" = torch.ops.aten.view.default(add_84, [-1, 768]);  add_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:351, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_23 = torch.ops.aten.var_mean.correction(view_240, [1], correction = 0, keepdim = True)
    getitem_46: "f32[2048, 1]" = var_mean_23[0]
    getitem_47: "f32[2048, 1]" = var_mean_23[1];  var_mean_23 = None
    add_85: "f32[2048, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
    rsqrt_23: "f32[2048, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
    sub_37: "f32[2048, 768]" = torch.ops.aten.sub.Tensor(view_240, getitem_47);  getitem_47 = None
    mul_59: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_23);  sub_37 = rsqrt_23 = None
    mul_60: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(mul_59, arg188_1);  mul_59 = arg188_1 = None
    add_86: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mul_60, arg189_1);  mul_60 = arg189_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:353, code: hidden_states = self.fc1(hidden_states)
    permute_130: "f32[768, 3072]" = torch.ops.aten.permute.default(arg190_1, [1, 0]);  arg190_1 = None
    addmm_70: "f32[2048, 3072]" = torch.ops.aten.addmm.default(arg191_1, add_86, permute_130);  arg191_1 = add_86 = permute_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:354, code: hidden_states = self.activation_fn(hidden_states)
    relu_11: "f32[2048, 3072]" = torch.ops.aten.relu.default(addmm_70);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:356, code: hidden_states = self.fc2(hidden_states)
    permute_131: "f32[3072, 768]" = torch.ops.aten.permute.default(arg192_1, [1, 0]);  arg192_1 = None
    addmm_71: "f32[2048, 768]" = torch.ops.aten.addmm.default(arg193_1, relu_11, permute_131);  arg193_1 = relu_11 = permute_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:357, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_83: "f32[2048, 768]" = torch.ops.aten.clone.default(addmm_71);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:359, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
    add_87: "f32[2048, 768]" = torch.ops.aten.add.Tensor(view_240, clone_83);  view_240 = clone_83 = None
    view_241: "f32[1, 2048, 768]" = torch.ops.aten.view.default(add_87, [1, 2048, 768]);  add_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:728, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_24 = torch.ops.aten.var_mean.correction(view_241, [2], correction = 0, keepdim = True)
    getitem_48: "f32[1, 2048, 1]" = var_mean_24[0]
    getitem_49: "f32[1, 2048, 1]" = var_mean_24[1];  var_mean_24 = None
    add_88: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
    rsqrt_24: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
    sub_38: "f32[1, 2048, 768]" = torch.ops.aten.sub.Tensor(view_241, getitem_49);  view_241 = getitem_49 = None
    mul_61: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_24);  sub_38 = rsqrt_24 = None
    mul_62: "f32[1, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_61, arg194_1);  mul_61 = arg194_1 = None
    add_89: "f32[1, 2048, 768]" = torch.ops.aten.add.Tensor(mul_62, arg195_1);  mul_62 = arg195_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:956, code: logits = self.lm_head(outputs[0]).contiguous()
    permute_132: "f32[768, 50272]" = torch.ops.aten.permute.default(arg196_1, [1, 0]);  arg196_1 = None
    view_242: "f32[2048, 768]" = torch.ops.aten.view.default(add_89, [2048, 768]);  add_89 = None
    mm: "f32[2048, 50272]" = torch.ops.aten.mm.default(view_242, permute_132);  view_242 = permute_132 = None
    view_243: "f32[1, 2048, 50272]" = torch.ops.aten.view.default(mm, [1, 2048, 50272]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:963, code: shift_logits = logits[..., :-1, :].contiguous()
    slice_9: "f32[1, 2047, 50272]" = torch.ops.aten.slice.Tensor(view_243, 1, 0, -1)
    slice_10: "f32[1, 2047, 50272]" = torch.ops.aten.slice.Tensor(slice_9, 2, 0, 9223372036854775807);  slice_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:964, code: shift_labels = labels[..., 1:].contiguous()
    slice_11: "i64[1, 2047]" = torch.ops.aten.slice.Tensor(arg198_1, 1, 1, 9223372036854775807);  arg198_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:967, code: loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
    view_244: "f32[2047, 50272]" = torch.ops.aten.view.default(slice_10, [-1, 50272]);  slice_10 = None
    view_245: "i64[2047]" = torch.ops.aten.view.default(slice_11, [-1]);  slice_11 = None
    amax_12: "f32[2047, 1]" = torch.ops.aten.amax.default(view_244, [1], True)
    sub_39: "f32[2047, 50272]" = torch.ops.aten.sub.Tensor(view_244, amax_12);  view_244 = amax_12 = None
    exp_12: "f32[2047, 50272]" = torch.ops.aten.exp.default(sub_39)
    sum_13: "f32[2047, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [1], True);  exp_12 = None
    log: "f32[2047, 1]" = torch.ops.aten.log.default(sum_13);  sum_13 = None
    sub_40: "f32[2047, 50272]" = torch.ops.aten.sub.Tensor(sub_39, log);  sub_39 = log = None
    ne: "b8[2047]" = torch.ops.aten.ne.Scalar(view_245, -100)
    scalar_tensor_2: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0))
    where_2: "i64[2047]" = torch.ops.aten.where.self(ne, view_245, scalar_tensor_2);  ne = scalar_tensor_2 = None
    unsqueeze_6: "i64[2047, 1]" = torch.ops.aten.unsqueeze.default(where_2, 1);  where_2 = None
    gather: "f32[2047, 1]" = torch.ops.aten.gather.default(sub_40, 1, unsqueeze_6);  sub_40 = unsqueeze_6 = None
    squeeze: "f32[2047]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[2047]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
    ne_1: "b8[2047]" = torch.ops.aten.ne.Scalar(view_245, -100)
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_3: "f32[2047]" = torch.ops.aten.where.self(ne_1, neg, scalar_tensor_3);  ne_1 = neg = scalar_tensor_3 = None
    ne_2: "b8[2047]" = torch.ops.aten.ne.Scalar(view_245, -100);  view_245 = None
    sum_14: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
    convert_element_type_2: "f32[]" = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None
    sum_15: "f32[]" = torch.ops.aten.sum.default(where_3);  where_3 = None
    div_12: "f32[]" = torch.ops.aten.div.Tensor(sum_15, convert_element_type_2);  sum_15 = convert_element_type_2 = None
    return (div_12, view_243, clone, clone_1, clone_7, clone_8, clone_14, clone_15, clone_21, clone_22, clone_28, clone_29, clone_35, clone_36, clone_42, clone_43, clone_49, clone_50, clone_56, clone_57, clone_63, clone_64, clone_70, clone_71, clone_77, clone_78)
    