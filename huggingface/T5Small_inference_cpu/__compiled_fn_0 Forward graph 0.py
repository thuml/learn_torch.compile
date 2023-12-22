from __future__ import annotations



def forward(self, arg0_1: "f32[512]", arg1_1: "f32[512]", arg2_1: "f32[512]", arg3_1: "f32[512]", arg4_1: "f32[512]", arg5_1: "f32[512]", arg6_1: "f32[512]", arg7_1: "f32[512]", arg8_1: "f32[512]", arg9_1: "f32[512]", arg10_1: "f32[512]", arg11_1: "f32[512]", arg12_1: "f32[512]", arg13_1: "f32[512]", arg14_1: "f32[512]", arg15_1: "f32[512]", arg16_1: "f32[512]", arg17_1: "f32[512]", arg18_1: "f32[512]", arg19_1: "f32[512]", arg20_1: "f32[512]", arg21_1: "f32[512]", arg22_1: "f32[512]", arg23_1: "f32[512]", arg24_1: "f32[512]", arg25_1: "f32[512]", arg26_1: "f32[512]", arg27_1: "f32[512]", arg28_1: "f32[512]", arg29_1: "f32[512]", arg30_1: "f32[512]", arg31_1: "f32[512]", arg32_1: "f32[32128, 512]", arg33_1: "f32[512, 512]", arg34_1: "f32[512, 512]", arg35_1: "f32[512, 512]", arg36_1: "f32[32, 8]", arg37_1: "f32[512, 512]", arg38_1: "f32[2048, 512]", arg39_1: "f32[512, 2048]", arg40_1: "f32[512, 512]", arg41_1: "f32[512, 512]", arg42_1: "f32[512, 512]", arg43_1: "f32[512, 512]", arg44_1: "f32[2048, 512]", arg45_1: "f32[512, 2048]", arg46_1: "f32[512, 512]", arg47_1: "f32[512, 512]", arg48_1: "f32[512, 512]", arg49_1: "f32[512, 512]", arg50_1: "f32[2048, 512]", arg51_1: "f32[512, 2048]", arg52_1: "f32[512, 512]", arg53_1: "f32[512, 512]", arg54_1: "f32[512, 512]", arg55_1: "f32[512, 512]", arg56_1: "f32[2048, 512]", arg57_1: "f32[512, 2048]", arg58_1: "f32[512, 512]", arg59_1: "f32[512, 512]", arg60_1: "f32[512, 512]", arg61_1: "f32[512, 512]", arg62_1: "f32[2048, 512]", arg63_1: "f32[512, 2048]", arg64_1: "f32[512, 512]", arg65_1: "f32[512, 512]", arg66_1: "f32[512, 512]", arg67_1: "f32[512, 512]", arg68_1: "f32[2048, 512]", arg69_1: "f32[512, 2048]", arg70_1: "f32[512, 512]", arg71_1: "f32[512, 512]", arg72_1: "f32[512, 512]", arg73_1: "f32[32, 8]", arg74_1: "f32[512, 512]", arg75_1: "f32[512, 512]", arg76_1: "f32[512, 512]", arg77_1: "f32[512, 512]", arg78_1: "f32[512, 512]", arg79_1: "f32[2048, 512]", arg80_1: "f32[512, 2048]", arg81_1: "f32[512, 512]", arg82_1: "f32[512, 512]", arg83_1: "f32[512, 512]", arg84_1: "f32[512, 512]", arg85_1: "f32[512, 512]", arg86_1: "f32[512, 512]", arg87_1: "f32[512, 512]", arg88_1: "f32[512, 512]", arg89_1: "f32[2048, 512]", arg90_1: "f32[512, 2048]", arg91_1: "f32[512, 512]", arg92_1: "f32[512, 512]", arg93_1: "f32[512, 512]", arg94_1: "f32[512, 512]", arg95_1: "f32[512, 512]", arg96_1: "f32[512, 512]", arg97_1: "f32[512, 512]", arg98_1: "f32[512, 512]", arg99_1: "f32[2048, 512]", arg100_1: "f32[512, 2048]", arg101_1: "f32[512, 512]", arg102_1: "f32[512, 512]", arg103_1: "f32[512, 512]", arg104_1: "f32[512, 512]", arg105_1: "f32[512, 512]", arg106_1: "f32[512, 512]", arg107_1: "f32[512, 512]", arg108_1: "f32[512, 512]", arg109_1: "f32[2048, 512]", arg110_1: "f32[512, 2048]", arg111_1: "f32[512, 512]", arg112_1: "f32[512, 512]", arg113_1: "f32[512, 512]", arg114_1: "f32[512, 512]", arg115_1: "f32[512, 512]", arg116_1: "f32[512, 512]", arg117_1: "f32[512, 512]", arg118_1: "f32[512, 512]", arg119_1: "f32[2048, 512]", arg120_1: "f32[512, 2048]", arg121_1: "f32[512, 512]", arg122_1: "f32[512, 512]", arg123_1: "f32[512, 512]", arg124_1: "f32[512, 512]", arg125_1: "f32[512, 512]", arg126_1: "f32[512, 512]", arg127_1: "f32[512, 512]", arg128_1: "f32[512, 512]", arg129_1: "f32[2048, 512]", arg130_1: "f32[512, 2048]", arg131_1: "f32[32128, 512]", arg132_1: "i64[1, 1024]", arg133_1: "i64[1, 1024]", arg134_1: "i64[1, 1024]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1011, code: input_ids = input_ids.view(-1, input_shape[-1])
    view: "i64[1, 1024]" = torch.ops.aten.view.default(arg132_1, [-1, 1024]);  arg132_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1021, code: inputs_embeds = self.embed_tokens(input_ids)
    embedding: "f32[1, 1024, 512]" = torch.ops.aten.embedding.default(arg32_1, view);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1033, code: attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
    full: "f32[1, 1024]" = torch.ops.aten.full.default([1, 1024], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:916, code: extended_attention_mask = attention_mask[:, None, None, :]
    slice_1: "f32[1, 1024]" = torch.ops.aten.slice.Tensor(full, 0, 0, 9223372036854775807);  full = None
    unsqueeze: "f32[1, 1, 1024]" = torch.ops.aten.unsqueeze.default(slice_1, 1);  slice_1 = None
    unsqueeze_1: "f32[1, 1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
    slice_2: "f32[1, 1, 1, 1024]" = torch.ops.aten.slice.Tensor(unsqueeze_1, 3, 0, 9223372036854775807);  unsqueeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:928, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    sub: "f32[1, 1, 1, 1024]" = torch.ops.aten.sub.Tensor(1.0, slice_2);  slice_2 = None
    mul: "f32[1, 1, 1, 1024]" = torch.ops.aten.mul.Tensor(sub, -3.4028234663852886e+38);  sub = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1076, code: hidden_states = self.dropout(inputs_embeds)
    clone: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(embedding);  embedding = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_1: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(clone, 2)
    mean: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
    rsqrt: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
    mul_1: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(clone, rsqrt);  rsqrt = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_2: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(arg0_1, mul_1);  arg0_1 = mul_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute: "f32[512, 512]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
    view_1: "f32[1024, 512]" = torch.ops.aten.view.default(mul_2, [1024, 512])
    mm: "f32[1024, 512]" = torch.ops.aten.mm.default(view_1, permute);  view_1 = permute = None
    view_2: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm, [1, 1024, 512]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_3: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_2, [1, -1, 8, 64]);  view_2 = None
    permute_1: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_2: "f32[512, 512]" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
    view_4: "f32[1024, 512]" = torch.ops.aten.view.default(mul_2, [1024, 512])
    mm_1: "f32[1024, 512]" = torch.ops.aten.mm.default(view_4, permute_2);  view_4 = permute_2 = None
    view_5: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_1, [1, 1024, 512]);  mm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_6: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_5, [1, -1, 8, 64]);  view_5 = None
    permute_3: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_4: "f32[512, 512]" = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
    view_7: "f32[1024, 512]" = torch.ops.aten.view.default(mul_2, [1024, 512]);  mul_2 = None
    mm_2: "f32[1024, 512]" = torch.ops.aten.mm.default(view_7, permute_4);  view_7 = permute_4 = None
    view_8: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_2, [1, 1024, 512]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_9: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_8, [1, -1, 8, 64]);  view_8 = None
    permute_5: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_6: "f32[1, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_1, [1, 8, 1024, 64]);  permute_1 = None
    view_10: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand, [8, 1024, 64]);  expand = None
    expand_1: "f32[1, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_6, [1, 8, 64, 1024]);  permute_6 = None
    view_11: "f32[8, 64, 1024]" = torch.ops.aten.view.default(expand_1, [8, 64, 1024]);  expand_1 = None
    bmm: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_10, view_11);  view_10 = view_11 = None
    view_12: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm, [1, 8, 1024, 1024]);  bmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:441, code: context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
    iota: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    slice_3: "i64[1024]" = torch.ops.aten.slice.Tensor(iota, 0, 0, 9223372036854775807);  iota = None
    unsqueeze_2: "i64[1024, 1]" = torch.ops.aten.unsqueeze.default(slice_3, 1);  slice_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:442, code: memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
    iota_1: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_3: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(iota_1, 0);  iota_1 = None
    slice_4: "i64[1, 1024]" = torch.ops.aten.slice.Tensor(unsqueeze_3, 1, 0, 9223372036854775807);  unsqueeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:443, code: relative_position = memory_position - context_position  # shape (query_length, key_length)
    sub_1: "i64[1024, 1024]" = torch.ops.aten.sub.Tensor(slice_4, unsqueeze_2);  slice_4 = unsqueeze_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:414, code: relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
    gt: "b8[1024, 1024]" = torch.ops.aten.gt.Scalar(sub_1, 0)
    convert_element_type: "i64[1024, 1024]" = torch.ops.prims.convert_element_type.default(gt, torch.int64);  gt = None
    mul_3: "i64[1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type, 16);  convert_element_type = None
    add_1: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(mul_3, 0);  mul_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:415, code: relative_position = torch.abs(relative_position)
    abs_1: "i64[1024, 1024]" = torch.ops.aten.abs.default(sub_1);  sub_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:422, code: is_small = relative_position < max_exact
    lt: "b8[1024, 1024]" = torch.ops.aten.lt.Scalar(abs_1, 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:426, code: torch.log(relative_position.float() / max_exact)
    convert_element_type_1: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(abs_1, torch.float32)
    div: "f32[1024, 1024]" = torch.ops.aten.div.Tensor(convert_element_type_1, 8);  convert_element_type_1 = None
    log: "f32[1024, 1024]" = torch.ops.aten.log.default(div);  div = None
    div_1: "f32[1024, 1024]" = torch.ops.aten.div.Tensor(log, 2.772588722239781);  log = None
    mul_4: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(div_1, 8);  div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:429, code: ).to(torch.long)
    convert_element_type_2: "i64[1024, 1024]" = torch.ops.prims.convert_element_type.default(mul_4, torch.int64);  mul_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:425, code: relative_position_if_large = max_exact + (
    add_2: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_2, 8);  convert_element_type_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:431, code: relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
    full_1: "i64[1024, 1024]" = torch.ops.aten.full.default([1024, 1024], 15, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:430, code: relative_position_if_large = torch.min(
    minimum: "i64[1024, 1024]" = torch.ops.aten.minimum.default(add_2, full_1);  add_2 = full_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:434, code: relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
    where: "i64[1024, 1024]" = torch.ops.aten.where.self(lt, abs_1, minimum);  lt = abs_1 = minimum = None
    add_3: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(add_1, where);  add_1 = where = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:450, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
    embedding_1: "f32[1024, 1024, 8]" = torch.ops.aten.embedding.default(arg36_1, add_3);  arg36_1 = add_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:451, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
    permute_7: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(embedding_1, [2, 0, 1]);  embedding_1 = None
    unsqueeze_4: "f32[1, 8, 1024, 1024]" = torch.ops.aten.unsqueeze.default(permute_7, 0);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:552, code: position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)
    add_4: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(unsqueeze_4, mul);  unsqueeze_4 = mul = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_5: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_12, add_4);  view_12 = None
    view_13: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(add_5, [8, 1024, 1024]);  add_5 = None
    view_14: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(view_13, [1, 8, 1024, 1024]);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax: "f32[1, 8, 1024, 1]" = torch.ops.aten.amax.default(view_14, [-1], True)
    sub_2: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_14, amax);  view_14 = amax = None
    exp: "f32[1, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
    sum_1: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div_2: "f32[1, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    clone_1: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(div_2);  div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_2: "f32[1, 8, 1024, 1024]" = torch.ops.aten.expand.default(clone_1, [1, 8, 1024, 1024]);  clone_1 = None
    view_15: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(expand_2, [8, 1024, 1024]);  expand_2 = None
    expand_3: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_5, [1, 8, 1024, 64]);  permute_5 = None
    view_16: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_3, [8, 1024, 64]);  expand_3 = None
    bmm_1: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(view_15, view_16);  view_15 = view_16 = None
    view_17: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_1, [1, 8, 1024, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_8: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_17, [0, 2, 1, 3]);  view_17 = None
    clone_2: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
    view_18: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_2, [1, -1, 512]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_9: "f32[512, 512]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
    view_19: "f32[1024, 512]" = torch.ops.aten.view.default(view_18, [1024, 512]);  view_18 = None
    mm_3: "f32[1024, 512]" = torch.ops.aten.mm.default(view_19, permute_9);  view_19 = permute_9 = None
    view_20: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_3, [1, 1024, 512]);  mm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    clone_3: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(view_20);  view_20 = None
    add_6: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(clone, clone_3);  clone = clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_2: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_6, 2)
    mean_1: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_7: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
    rsqrt_1: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
    mul_5: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_6, rsqrt_1);  rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_6: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(arg1_1, mul_5);  arg1_1 = mul_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_10: "f32[512, 2048]" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
    view_21: "f32[1024, 512]" = torch.ops.aten.view.default(mul_6, [1024, 512]);  mul_6 = None
    mm_4: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_21, permute_10);  view_21 = permute_10 = None
    view_22: "f32[1, 1024, 2048]" = torch.ops.aten.view.default(mm_4, [1, 1024, 2048]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    relu: "f32[1, 1024, 2048]" = torch.ops.aten.relu.default(view_22);  view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    clone_4: "f32[1, 1024, 2048]" = torch.ops.aten.clone.default(relu);  relu = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_11: "f32[2048, 512]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
    view_23: "f32[1024, 2048]" = torch.ops.aten.view.default(clone_4, [1024, 2048]);  clone_4 = None
    mm_5: "f32[1024, 512]" = torch.ops.aten.mm.default(view_23, permute_11);  view_23 = permute_11 = None
    view_24: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_5, [1, 1024, 512]);  mm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    clone_5: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(view_24);  view_24 = None
    add_8: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_6, clone_5);  add_6 = clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_3: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_8, 2)
    mean_2: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_3, [-1], True);  pow_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_9: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_2, 1e-06);  mean_2 = None
    rsqrt_2: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
    mul_7: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_8, rsqrt_2);  rsqrt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_8: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(arg2_1, mul_7);  arg2_1 = mul_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_12: "f32[512, 512]" = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
    view_25: "f32[1024, 512]" = torch.ops.aten.view.default(mul_8, [1024, 512])
    mm_6: "f32[1024, 512]" = torch.ops.aten.mm.default(view_25, permute_12);  view_25 = permute_12 = None
    view_26: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_6, [1, 1024, 512]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_27: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_26, [1, -1, 8, 64]);  view_26 = None
    permute_13: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_14: "f32[512, 512]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
    view_28: "f32[1024, 512]" = torch.ops.aten.view.default(mul_8, [1024, 512])
    mm_7: "f32[1024, 512]" = torch.ops.aten.mm.default(view_28, permute_14);  view_28 = permute_14 = None
    view_29: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_7, [1, 1024, 512]);  mm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_30: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_29, [1, -1, 8, 64]);  view_29 = None
    permute_15: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_16: "f32[512, 512]" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
    view_31: "f32[1024, 512]" = torch.ops.aten.view.default(mul_8, [1024, 512]);  mul_8 = None
    mm_8: "f32[1024, 512]" = torch.ops.aten.mm.default(view_31, permute_16);  view_31 = permute_16 = None
    view_32: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_8, [1, 1024, 512]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_33: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_32, [1, -1, 8, 64]);  view_32 = None
    permute_17: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3]);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_18: "f32[1, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_15, [0, 1, 3, 2]);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_4: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_13, [1, 8, 1024, 64]);  permute_13 = None
    view_34: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_4, [8, 1024, 64]);  expand_4 = None
    expand_5: "f32[1, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_18, [1, 8, 64, 1024]);  permute_18 = None
    view_35: "f32[8, 64, 1024]" = torch.ops.aten.view.default(expand_5, [8, 64, 1024]);  expand_5 = None
    bmm_2: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_34, view_35);  view_34 = view_35 = None
    view_36: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_2, [1, 8, 1024, 1024]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_10: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_36, add_4);  view_36 = None
    view_37: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(add_10, [8, 1024, 1024]);  add_10 = None
    view_38: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(view_37, [1, 8, 1024, 1024]);  view_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_1: "f32[1, 8, 1024, 1]" = torch.ops.aten.amax.default(view_38, [-1], True)
    sub_3: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_38, amax_1);  view_38 = amax_1 = None
    exp_1: "f32[1, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
    sum_2: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_3: "f32[1, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    clone_6: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(div_3);  div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_6: "f32[1, 8, 1024, 1024]" = torch.ops.aten.expand.default(clone_6, [1, 8, 1024, 1024]);  clone_6 = None
    view_39: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(expand_6, [8, 1024, 1024]);  expand_6 = None
    expand_7: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_17, [1, 8, 1024, 64]);  permute_17 = None
    view_40: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_7, [8, 1024, 64]);  expand_7 = None
    bmm_3: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(view_39, view_40);  view_39 = view_40 = None
    view_41: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_3, [1, 8, 1024, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_19: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None
    clone_7: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None
    view_42: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_7, [1, -1, 512]);  clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_20: "f32[512, 512]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
    view_43: "f32[1024, 512]" = torch.ops.aten.view.default(view_42, [1024, 512]);  view_42 = None
    mm_9: "f32[1024, 512]" = torch.ops.aten.mm.default(view_43, permute_20);  view_43 = permute_20 = None
    view_44: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_9, [1, 1024, 512]);  mm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    clone_8: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(view_44);  view_44 = None
    add_11: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_8, clone_8);  add_8 = clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_4: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_11, 2)
    mean_3: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_4, [-1], True);  pow_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_12: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_3, 1e-06);  mean_3 = None
    rsqrt_3: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
    mul_9: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_11, rsqrt_3);  rsqrt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_10: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(arg3_1, mul_9);  arg3_1 = mul_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_21: "f32[512, 2048]" = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
    view_45: "f32[1024, 512]" = torch.ops.aten.view.default(mul_10, [1024, 512]);  mul_10 = None
    mm_10: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_45, permute_21);  view_45 = permute_21 = None
    view_46: "f32[1, 1024, 2048]" = torch.ops.aten.view.default(mm_10, [1, 1024, 2048]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    relu_1: "f32[1, 1024, 2048]" = torch.ops.aten.relu.default(view_46);  view_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    clone_9: "f32[1, 1024, 2048]" = torch.ops.aten.clone.default(relu_1);  relu_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_22: "f32[2048, 512]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
    view_47: "f32[1024, 2048]" = torch.ops.aten.view.default(clone_9, [1024, 2048]);  clone_9 = None
    mm_11: "f32[1024, 512]" = torch.ops.aten.mm.default(view_47, permute_22);  view_47 = permute_22 = None
    view_48: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_11, [1, 1024, 512]);  mm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    clone_10: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(view_48);  view_48 = None
    add_13: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_11, clone_10);  add_11 = clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_5: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_13, 2)
    mean_4: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_5, [-1], True);  pow_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_14: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_4, 1e-06);  mean_4 = None
    rsqrt_4: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
    mul_11: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_13, rsqrt_4);  rsqrt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_12: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(arg4_1, mul_11);  arg4_1 = mul_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_23: "f32[512, 512]" = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
    view_49: "f32[1024, 512]" = torch.ops.aten.view.default(mul_12, [1024, 512])
    mm_12: "f32[1024, 512]" = torch.ops.aten.mm.default(view_49, permute_23);  view_49 = permute_23 = None
    view_50: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_12, [1, 1024, 512]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_51: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_50, [1, -1, 8, 64]);  view_50 = None
    permute_24: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_25: "f32[512, 512]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
    view_52: "f32[1024, 512]" = torch.ops.aten.view.default(mul_12, [1024, 512])
    mm_13: "f32[1024, 512]" = torch.ops.aten.mm.default(view_52, permute_25);  view_52 = permute_25 = None
    view_53: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_13, [1, 1024, 512]);  mm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_54: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_53, [1, -1, 8, 64]);  view_53 = None
    permute_26: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_27: "f32[512, 512]" = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
    view_55: "f32[1024, 512]" = torch.ops.aten.view.default(mul_12, [1024, 512]);  mul_12 = None
    mm_14: "f32[1024, 512]" = torch.ops.aten.mm.default(view_55, permute_27);  view_55 = permute_27 = None
    view_56: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_14, [1, 1024, 512]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_57: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_56, [1, -1, 8, 64]);  view_56 = None
    permute_28: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_57, [0, 2, 1, 3]);  view_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_29: "f32[1, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_26, [0, 1, 3, 2]);  permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_8: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_24, [1, 8, 1024, 64]);  permute_24 = None
    view_58: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_8, [8, 1024, 64]);  expand_8 = None
    expand_9: "f32[1, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_29, [1, 8, 64, 1024]);  permute_29 = None
    view_59: "f32[8, 64, 1024]" = torch.ops.aten.view.default(expand_9, [8, 64, 1024]);  expand_9 = None
    bmm_4: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_58, view_59);  view_58 = view_59 = None
    view_60: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_4, [1, 8, 1024, 1024]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_15: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_60, add_4);  view_60 = None
    view_61: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(add_15, [8, 1024, 1024]);  add_15 = None
    view_62: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(view_61, [1, 8, 1024, 1024]);  view_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_2: "f32[1, 8, 1024, 1]" = torch.ops.aten.amax.default(view_62, [-1], True)
    sub_4: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_62, amax_2);  view_62 = amax_2 = None
    exp_2: "f32[1, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
    sum_3: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_4: "f32[1, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    clone_11: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(div_4);  div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_10: "f32[1, 8, 1024, 1024]" = torch.ops.aten.expand.default(clone_11, [1, 8, 1024, 1024]);  clone_11 = None
    view_63: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(expand_10, [8, 1024, 1024]);  expand_10 = None
    expand_11: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_28, [1, 8, 1024, 64]);  permute_28 = None
    view_64: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_11, [8, 1024, 64]);  expand_11 = None
    bmm_5: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(view_63, view_64);  view_63 = view_64 = None
    view_65: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_5, [1, 8, 1024, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_30: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None
    clone_12: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
    view_66: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_12, [1, -1, 512]);  clone_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_31: "f32[512, 512]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
    view_67: "f32[1024, 512]" = torch.ops.aten.view.default(view_66, [1024, 512]);  view_66 = None
    mm_15: "f32[1024, 512]" = torch.ops.aten.mm.default(view_67, permute_31);  view_67 = permute_31 = None
    view_68: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_15, [1, 1024, 512]);  mm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    clone_13: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(view_68);  view_68 = None
    add_16: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_13, clone_13);  add_13 = clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_6: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_16, 2)
    mean_5: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_6, [-1], True);  pow_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_17: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_5, 1e-06);  mean_5 = None
    rsqrt_5: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
    mul_13: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_16, rsqrt_5);  rsqrt_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_14: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(arg5_1, mul_13);  arg5_1 = mul_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_32: "f32[512, 2048]" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
    view_69: "f32[1024, 512]" = torch.ops.aten.view.default(mul_14, [1024, 512]);  mul_14 = None
    mm_16: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_69, permute_32);  view_69 = permute_32 = None
    view_70: "f32[1, 1024, 2048]" = torch.ops.aten.view.default(mm_16, [1, 1024, 2048]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    relu_2: "f32[1, 1024, 2048]" = torch.ops.aten.relu.default(view_70);  view_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    clone_14: "f32[1, 1024, 2048]" = torch.ops.aten.clone.default(relu_2);  relu_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_33: "f32[2048, 512]" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
    view_71: "f32[1024, 2048]" = torch.ops.aten.view.default(clone_14, [1024, 2048]);  clone_14 = None
    mm_17: "f32[1024, 512]" = torch.ops.aten.mm.default(view_71, permute_33);  view_71 = permute_33 = None
    view_72: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_17, [1, 1024, 512]);  mm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    clone_15: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(view_72);  view_72 = None
    add_18: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_16, clone_15);  add_16 = clone_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_7: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_18, 2)
    mean_6: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_7, [-1], True);  pow_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_19: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_6, 1e-06);  mean_6 = None
    rsqrt_6: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
    mul_15: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_18, rsqrt_6);  rsqrt_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_16: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(arg6_1, mul_15);  arg6_1 = mul_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_34: "f32[512, 512]" = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
    view_73: "f32[1024, 512]" = torch.ops.aten.view.default(mul_16, [1024, 512])
    mm_18: "f32[1024, 512]" = torch.ops.aten.mm.default(view_73, permute_34);  view_73 = permute_34 = None
    view_74: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_18, [1, 1024, 512]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_75: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_74, [1, -1, 8, 64]);  view_74 = None
    permute_35: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_36: "f32[512, 512]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
    view_76: "f32[1024, 512]" = torch.ops.aten.view.default(mul_16, [1024, 512])
    mm_19: "f32[1024, 512]" = torch.ops.aten.mm.default(view_76, permute_36);  view_76 = permute_36 = None
    view_77: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_19, [1, 1024, 512]);  mm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_78: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_77, [1, -1, 8, 64]);  view_77 = None
    permute_37: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_38: "f32[512, 512]" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
    view_79: "f32[1024, 512]" = torch.ops.aten.view.default(mul_16, [1024, 512]);  mul_16 = None
    mm_20: "f32[1024, 512]" = torch.ops.aten.mm.default(view_79, permute_38);  view_79 = permute_38 = None
    view_80: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_20, [1, 1024, 512]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_81: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_80, [1, -1, 8, 64]);  view_80 = None
    permute_39: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_81, [0, 2, 1, 3]);  view_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_40: "f32[1, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_37, [0, 1, 3, 2]);  permute_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_12: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_35, [1, 8, 1024, 64]);  permute_35 = None
    view_82: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_12, [8, 1024, 64]);  expand_12 = None
    expand_13: "f32[1, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_40, [1, 8, 64, 1024]);  permute_40 = None
    view_83: "f32[8, 64, 1024]" = torch.ops.aten.view.default(expand_13, [8, 64, 1024]);  expand_13 = None
    bmm_6: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_82, view_83);  view_82 = view_83 = None
    view_84: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_6, [1, 8, 1024, 1024]);  bmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_20: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_84, add_4);  view_84 = None
    view_85: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(add_20, [8, 1024, 1024]);  add_20 = None
    view_86: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(view_85, [1, 8, 1024, 1024]);  view_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_3: "f32[1, 8, 1024, 1]" = torch.ops.aten.amax.default(view_86, [-1], True)
    sub_5: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_86, amax_3);  view_86 = amax_3 = None
    exp_3: "f32[1, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
    sum_4: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_5: "f32[1, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    clone_16: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(div_5);  div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_14: "f32[1, 8, 1024, 1024]" = torch.ops.aten.expand.default(clone_16, [1, 8, 1024, 1024]);  clone_16 = None
    view_87: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(expand_14, [8, 1024, 1024]);  expand_14 = None
    expand_15: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_39, [1, 8, 1024, 64]);  permute_39 = None
    view_88: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_15, [8, 1024, 64]);  expand_15 = None
    bmm_7: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(view_87, view_88);  view_87 = view_88 = None
    view_89: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_7, [1, 8, 1024, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_41: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_89, [0, 2, 1, 3]);  view_89 = None
    clone_17: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_41, memory_format = torch.contiguous_format);  permute_41 = None
    view_90: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_17, [1, -1, 512]);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_42: "f32[512, 512]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
    view_91: "f32[1024, 512]" = torch.ops.aten.view.default(view_90, [1024, 512]);  view_90 = None
    mm_21: "f32[1024, 512]" = torch.ops.aten.mm.default(view_91, permute_42);  view_91 = permute_42 = None
    view_92: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_21, [1, 1024, 512]);  mm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    clone_18: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(view_92);  view_92 = None
    add_21: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_18, clone_18);  add_18 = clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_8: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_21, 2)
    mean_7: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_8, [-1], True);  pow_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_22: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_7, 1e-06);  mean_7 = None
    rsqrt_7: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
    mul_17: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_21, rsqrt_7);  rsqrt_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_18: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(arg7_1, mul_17);  arg7_1 = mul_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_43: "f32[512, 2048]" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
    view_93: "f32[1024, 512]" = torch.ops.aten.view.default(mul_18, [1024, 512]);  mul_18 = None
    mm_22: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_93, permute_43);  view_93 = permute_43 = None
    view_94: "f32[1, 1024, 2048]" = torch.ops.aten.view.default(mm_22, [1, 1024, 2048]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    relu_3: "f32[1, 1024, 2048]" = torch.ops.aten.relu.default(view_94);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    clone_19: "f32[1, 1024, 2048]" = torch.ops.aten.clone.default(relu_3);  relu_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_44: "f32[2048, 512]" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
    view_95: "f32[1024, 2048]" = torch.ops.aten.view.default(clone_19, [1024, 2048]);  clone_19 = None
    mm_23: "f32[1024, 512]" = torch.ops.aten.mm.default(view_95, permute_44);  view_95 = permute_44 = None
    view_96: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_23, [1, 1024, 512]);  mm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    clone_20: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(view_96);  view_96 = None
    add_23: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_21, clone_20);  add_21 = clone_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_9: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_23, 2)
    mean_8: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_9, [-1], True);  pow_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_24: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_8, 1e-06);  mean_8 = None
    rsqrt_8: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
    mul_19: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_23, rsqrt_8);  rsqrt_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_20: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(arg8_1, mul_19);  arg8_1 = mul_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_45: "f32[512, 512]" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
    view_97: "f32[1024, 512]" = torch.ops.aten.view.default(mul_20, [1024, 512])
    mm_24: "f32[1024, 512]" = torch.ops.aten.mm.default(view_97, permute_45);  view_97 = permute_45 = None
    view_98: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_24, [1, 1024, 512]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_99: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_98, [1, -1, 8, 64]);  view_98 = None
    permute_46: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_99, [0, 2, 1, 3]);  view_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_47: "f32[512, 512]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
    view_100: "f32[1024, 512]" = torch.ops.aten.view.default(mul_20, [1024, 512])
    mm_25: "f32[1024, 512]" = torch.ops.aten.mm.default(view_100, permute_47);  view_100 = permute_47 = None
    view_101: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_25, [1, 1024, 512]);  mm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_102: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_101, [1, -1, 8, 64]);  view_101 = None
    permute_48: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_49: "f32[512, 512]" = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
    view_103: "f32[1024, 512]" = torch.ops.aten.view.default(mul_20, [1024, 512]);  mul_20 = None
    mm_26: "f32[1024, 512]" = torch.ops.aten.mm.default(view_103, permute_49);  view_103 = permute_49 = None
    view_104: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_26, [1, 1024, 512]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_105: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_104, [1, -1, 8, 64]);  view_104 = None
    permute_50: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_105, [0, 2, 1, 3]);  view_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_51: "f32[1, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_48, [0, 1, 3, 2]);  permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_16: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_46, [1, 8, 1024, 64]);  permute_46 = None
    view_106: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_16, [8, 1024, 64]);  expand_16 = None
    expand_17: "f32[1, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_51, [1, 8, 64, 1024]);  permute_51 = None
    view_107: "f32[8, 64, 1024]" = torch.ops.aten.view.default(expand_17, [8, 64, 1024]);  expand_17 = None
    bmm_8: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_106, view_107);  view_106 = view_107 = None
    view_108: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_8, [1, 8, 1024, 1024]);  bmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_25: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_108, add_4);  view_108 = None
    view_109: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(add_25, [8, 1024, 1024]);  add_25 = None
    view_110: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(view_109, [1, 8, 1024, 1024]);  view_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_4: "f32[1, 8, 1024, 1]" = torch.ops.aten.amax.default(view_110, [-1], True)
    sub_6: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_110, amax_4);  view_110 = amax_4 = None
    exp_4: "f32[1, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_6);  sub_6 = None
    sum_5: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_6: "f32[1, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    clone_21: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(div_6);  div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_18: "f32[1, 8, 1024, 1024]" = torch.ops.aten.expand.default(clone_21, [1, 8, 1024, 1024]);  clone_21 = None
    view_111: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(expand_18, [8, 1024, 1024]);  expand_18 = None
    expand_19: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_50, [1, 8, 1024, 64]);  permute_50 = None
    view_112: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_19, [8, 1024, 64]);  expand_19 = None
    bmm_9: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(view_111, view_112);  view_111 = view_112 = None
    view_113: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_9, [1, 8, 1024, 64]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_52: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_113, [0, 2, 1, 3]);  view_113 = None
    clone_22: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None
    view_114: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_22, [1, -1, 512]);  clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_53: "f32[512, 512]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
    view_115: "f32[1024, 512]" = torch.ops.aten.view.default(view_114, [1024, 512]);  view_114 = None
    mm_27: "f32[1024, 512]" = torch.ops.aten.mm.default(view_115, permute_53);  view_115 = permute_53 = None
    view_116: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_27, [1, 1024, 512]);  mm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    clone_23: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(view_116);  view_116 = None
    add_26: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_23, clone_23);  add_23 = clone_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_10: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_26, 2)
    mean_9: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_10, [-1], True);  pow_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_27: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_9, 1e-06);  mean_9 = None
    rsqrt_9: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
    mul_21: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_26, rsqrt_9);  rsqrt_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_22: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(arg9_1, mul_21);  arg9_1 = mul_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_54: "f32[512, 2048]" = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
    view_117: "f32[1024, 512]" = torch.ops.aten.view.default(mul_22, [1024, 512]);  mul_22 = None
    mm_28: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_117, permute_54);  view_117 = permute_54 = None
    view_118: "f32[1, 1024, 2048]" = torch.ops.aten.view.default(mm_28, [1, 1024, 2048]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    relu_4: "f32[1, 1024, 2048]" = torch.ops.aten.relu.default(view_118);  view_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    clone_24: "f32[1, 1024, 2048]" = torch.ops.aten.clone.default(relu_4);  relu_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_55: "f32[2048, 512]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
    view_119: "f32[1024, 2048]" = torch.ops.aten.view.default(clone_24, [1024, 2048]);  clone_24 = None
    mm_29: "f32[1024, 512]" = torch.ops.aten.mm.default(view_119, permute_55);  view_119 = permute_55 = None
    view_120: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_29, [1, 1024, 512]);  mm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    clone_25: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(view_120);  view_120 = None
    add_28: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_26, clone_25);  add_26 = clone_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_11: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_28, 2)
    mean_10: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_11, [-1], True);  pow_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_29: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_10, 1e-06);  mean_10 = None
    rsqrt_10: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
    mul_23: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_28, rsqrt_10);  rsqrt_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_24: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(arg10_1, mul_23);  arg10_1 = mul_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_56: "f32[512, 512]" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
    view_121: "f32[1024, 512]" = torch.ops.aten.view.default(mul_24, [1024, 512])
    mm_30: "f32[1024, 512]" = torch.ops.aten.mm.default(view_121, permute_56);  view_121 = permute_56 = None
    view_122: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_30, [1, 1024, 512]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_123: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_122, [1, -1, 8, 64]);  view_122 = None
    permute_57: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_123, [0, 2, 1, 3]);  view_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_58: "f32[512, 512]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
    view_124: "f32[1024, 512]" = torch.ops.aten.view.default(mul_24, [1024, 512])
    mm_31: "f32[1024, 512]" = torch.ops.aten.mm.default(view_124, permute_58);  view_124 = permute_58 = None
    view_125: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_31, [1, 1024, 512]);  mm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_126: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_125, [1, -1, 8, 64]);  view_125 = None
    permute_59: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_60: "f32[512, 512]" = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
    view_127: "f32[1024, 512]" = torch.ops.aten.view.default(mul_24, [1024, 512]);  mul_24 = None
    mm_32: "f32[1024, 512]" = torch.ops.aten.mm.default(view_127, permute_60);  view_127 = permute_60 = None
    view_128: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_32, [1, 1024, 512]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_129: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_128, [1, -1, 8, 64]);  view_128 = None
    permute_61: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_129, [0, 2, 1, 3]);  view_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_62: "f32[1, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_59, [0, 1, 3, 2]);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_20: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_57, [1, 8, 1024, 64]);  permute_57 = None
    view_130: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_20, [8, 1024, 64]);  expand_20 = None
    expand_21: "f32[1, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_62, [1, 8, 64, 1024]);  permute_62 = None
    view_131: "f32[8, 64, 1024]" = torch.ops.aten.view.default(expand_21, [8, 64, 1024]);  expand_21 = None
    bmm_10: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_130, view_131);  view_130 = view_131 = None
    view_132: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_10, [1, 8, 1024, 1024]);  bmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_30: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_132, add_4);  view_132 = add_4 = None
    view_133: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(add_30, [8, 1024, 1024]);  add_30 = None
    view_134: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(view_133, [1, 8, 1024, 1024]);  view_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_5: "f32[1, 8, 1024, 1]" = torch.ops.aten.amax.default(view_134, [-1], True)
    sub_7: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_134, amax_5);  view_134 = amax_5 = None
    exp_5: "f32[1, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
    sum_6: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_7: "f32[1, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    clone_26: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(div_7);  div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_22: "f32[1, 8, 1024, 1024]" = torch.ops.aten.expand.default(clone_26, [1, 8, 1024, 1024]);  clone_26 = None
    view_135: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(expand_22, [8, 1024, 1024]);  expand_22 = None
    expand_23: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_61, [1, 8, 1024, 64]);  permute_61 = None
    view_136: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_23, [8, 1024, 64]);  expand_23 = None
    bmm_11: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(view_135, view_136);  view_135 = view_136 = None
    view_137: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_11, [1, 8, 1024, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_63: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None
    clone_27: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format);  permute_63 = None
    view_138: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_27, [1, -1, 512]);  clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_64: "f32[512, 512]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
    view_139: "f32[1024, 512]" = torch.ops.aten.view.default(view_138, [1024, 512]);  view_138 = None
    mm_33: "f32[1024, 512]" = torch.ops.aten.mm.default(view_139, permute_64);  view_139 = permute_64 = None
    view_140: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_33, [1, 1024, 512]);  mm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    clone_28: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(view_140);  view_140 = None
    add_31: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_28, clone_28);  add_28 = clone_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_12: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_31, 2)
    mean_11: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_12, [-1], True);  pow_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_32: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_11, 1e-06);  mean_11 = None
    rsqrt_11: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    mul_25: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_31, rsqrt_11);  rsqrt_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_26: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(arg11_1, mul_25);  arg11_1 = mul_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_65: "f32[512, 2048]" = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
    view_141: "f32[1024, 512]" = torch.ops.aten.view.default(mul_26, [1024, 512]);  mul_26 = None
    mm_34: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_141, permute_65);  view_141 = permute_65 = None
    view_142: "f32[1, 1024, 2048]" = torch.ops.aten.view.default(mm_34, [1, 1024, 2048]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    relu_5: "f32[1, 1024, 2048]" = torch.ops.aten.relu.default(view_142);  view_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    clone_29: "f32[1, 1024, 2048]" = torch.ops.aten.clone.default(relu_5);  relu_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_66: "f32[2048, 512]" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
    view_143: "f32[1024, 2048]" = torch.ops.aten.view.default(clone_29, [1024, 2048]);  clone_29 = None
    mm_35: "f32[1024, 512]" = torch.ops.aten.mm.default(view_143, permute_66);  view_143 = permute_66 = None
    view_144: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_35, [1, 1024, 512]);  mm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    clone_30: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(view_144);  view_144 = None
    add_33: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_31, clone_30);  add_31 = clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_13: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_33, 2)
    mean_12: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_13, [-1], True);  pow_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_34: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_12, 1e-06);  mean_12 = None
    rsqrt_12: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
    mul_27: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_33, rsqrt_12);  add_33 = rsqrt_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_28: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(arg12_1, mul_27);  arg12_1 = mul_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1166, code: hidden_states = self.dropout(hidden_states)
    clone_31: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(mul_28);  mul_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1011, code: input_ids = input_ids.view(-1, input_shape[-1])
    view_145: "i64[1, 1024]" = torch.ops.aten.view.default(arg134_1, [-1, 1024]);  arg134_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1021, code: inputs_embeds = self.embed_tokens(input_ids)
    embedding_2: "f32[1, 1024, 512]" = torch.ops.aten.embedding.default(arg32_1, view_145);  arg32_1 = view_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1033, code: attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
    full_2: "f32[1, 1024]" = torch.ops.aten.full.default([1, 1024], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1036, code: encoder_attention_mask = torch.ones(
    full_3: "i64[1, 1024]" = torch.ops.aten.full.default([1, 1024], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:860, code: seq_ids = torch.arange(seq_length, device=device)
    iota_2: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:861, code: causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
    unsqueeze_5: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(iota_2, 0)
    unsqueeze_6: "i64[1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 1);  unsqueeze_5 = None
    slice_5: "i64[1, 1, 1024]" = torch.ops.aten.slice.Tensor(unsqueeze_6, 2, 0, 9223372036854775807);  unsqueeze_6 = None
    repeat: "i64[1, 1024, 1024]" = torch.ops.aten.repeat.default(slice_5, [1, 1024, 1]);  slice_5 = None
    unsqueeze_7: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(iota_2, 0);  iota_2 = None
    slice_6: "i64[1, 1024]" = torch.ops.aten.slice.Tensor(unsqueeze_7, 1, 0, 9223372036854775807);  unsqueeze_7 = None
    unsqueeze_8: "i64[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(slice_6, 2);  slice_6 = None
    le: "b8[1, 1024, 1024]" = torch.ops.aten.le.Tensor(repeat, unsqueeze_8);  repeat = unsqueeze_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:864, code: causal_mask = causal_mask.to(attention_mask.dtype)
    convert_element_type_3: "f32[1, 1024, 1024]" = torch.ops.prims.convert_element_type.default(le, torch.float32);  le = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:876, code: extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
    slice_7: "f32[1, 1024, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_3, 0, 0, 9223372036854775807);  convert_element_type_3 = None
    unsqueeze_9: "f32[1, 1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(slice_7, 1);  slice_7 = None
    slice_8: "f32[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(unsqueeze_9, 2, 0, 9223372036854775807);  unsqueeze_9 = None
    slice_9: "f32[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_8, 3, 0, 9223372036854775807);  slice_8 = None
    slice_10: "f32[1, 1024]" = torch.ops.aten.slice.Tensor(full_2, 0, 0, 9223372036854775807);  full_2 = None
    unsqueeze_10: "f32[1, 1, 1024]" = torch.ops.aten.unsqueeze.default(slice_10, 1);  slice_10 = None
    unsqueeze_11: "f32[1, 1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 2);  unsqueeze_10 = None
    slice_11: "f32[1, 1, 1, 1024]" = torch.ops.aten.slice.Tensor(unsqueeze_11, 3, 0, 9223372036854775807);  unsqueeze_11 = None
    mul_29: "f32[1, 1, 1024, 1024]" = torch.ops.aten.mul.Tensor(slice_9, slice_11);  slice_9 = slice_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:928, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    sub_8: "f32[1, 1, 1024, 1024]" = torch.ops.aten.sub.Tensor(1.0, mul_29);  mul_29 = None
    mul_30: "f32[1, 1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_8, -3.4028234663852886e+38);  sub_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:840, code: encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
    slice_12: "i64[1, 1024]" = torch.ops.aten.slice.Tensor(full_3, 0, 0, 9223372036854775807);  full_3 = None
    unsqueeze_12: "i64[1, 1, 1024]" = torch.ops.aten.unsqueeze.default(slice_12, 1);  slice_12 = None
    unsqueeze_13: "i64[1, 1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, 2);  unsqueeze_12 = None
    slice_13: "i64[1, 1, 1, 1024]" = torch.ops.aten.slice.Tensor(unsqueeze_13, 3, 0, 9223372036854775807);  unsqueeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:846, code: encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
    convert_element_type_4: "f32[1, 1, 1, 1024]" = torch.ops.prims.convert_element_type.default(slice_13, torch.float32);  slice_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:847, code: encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(self.dtype).min
    sub_9: "f32[1, 1, 1, 1024]" = torch.ops.aten.sub.Tensor(1.0, convert_element_type_4);  convert_element_type_4 = None
    mul_31: "f32[1, 1, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_9, -3.4028234663852886e+38);  sub_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1076, code: hidden_states = self.dropout(inputs_embeds)
    clone_32: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(embedding_2);  embedding_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_14: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(clone_32, 2)
    mean_13: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_14, [-1], True);  pow_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_35: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_13, 1e-06);  mean_13 = None
    rsqrt_13: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
    mul_32: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(clone_32, rsqrt_13);  rsqrt_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_33: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(arg13_1, mul_32);  arg13_1 = mul_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_67: "f32[512, 512]" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
    view_146: "f32[1024, 512]" = torch.ops.aten.view.default(mul_33, [1024, 512])
    mm_36: "f32[1024, 512]" = torch.ops.aten.mm.default(view_146, permute_67);  view_146 = permute_67 = None
    view_147: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_36, [1, 1024, 512]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_148: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_147, [1, -1, 8, 64]);  view_147 = None
    permute_68: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_69: "f32[512, 512]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
    view_149: "f32[1024, 512]" = torch.ops.aten.view.default(mul_33, [1024, 512])
    mm_37: "f32[1024, 512]" = torch.ops.aten.mm.default(view_149, permute_69);  view_149 = permute_69 = None
    view_150: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_37, [1, 1024, 512]);  mm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_151: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_150, [1, -1, 8, 64]);  view_150 = None
    permute_70: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_151, [0, 2, 1, 3]);  view_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_71: "f32[512, 512]" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
    view_152: "f32[1024, 512]" = torch.ops.aten.view.default(mul_33, [1024, 512]);  mul_33 = None
    mm_38: "f32[1024, 512]" = torch.ops.aten.mm.default(view_152, permute_71);  view_152 = permute_71 = None
    view_153: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_38, [1, 1024, 512]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_154: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_153, [1, -1, 8, 64]);  view_153 = None
    permute_72: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_154, [0, 2, 1, 3]);  view_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_73: "f32[1, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_70, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_24: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_68, [1, 8, 1024, 64]);  permute_68 = None
    view_155: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_24, [8, 1024, 64]);  expand_24 = None
    expand_25: "f32[1, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_73, [1, 8, 64, 1024]);  permute_73 = None
    view_156: "f32[8, 64, 1024]" = torch.ops.aten.view.default(expand_25, [8, 64, 1024]);  expand_25 = None
    bmm_12: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_155, view_156);  view_155 = view_156 = None
    view_157: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_12, [1, 8, 1024, 1024]);  bmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:441, code: context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
    iota_3: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    slice_14: "i64[1024]" = torch.ops.aten.slice.Tensor(iota_3, 0, 0, 9223372036854775807);  iota_3 = None
    unsqueeze_14: "i64[1024, 1]" = torch.ops.aten.unsqueeze.default(slice_14, 1);  slice_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:442, code: memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
    iota_4: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_15: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(iota_4, 0);  iota_4 = None
    slice_15: "i64[1, 1024]" = torch.ops.aten.slice.Tensor(unsqueeze_15, 1, 0, 9223372036854775807);  unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:443, code: relative_position = memory_position - context_position  # shape (query_length, key_length)
    sub_10: "i64[1024, 1024]" = torch.ops.aten.sub.Tensor(slice_15, unsqueeze_14);  slice_15 = unsqueeze_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:417, code: relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
    full_4: "i64[1024, 1024]" = torch.ops.aten.full.default([1024, 1024], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    minimum_1: "i64[1024, 1024]" = torch.ops.aten.minimum.default(sub_10, full_4);  sub_10 = full_4 = None
    neg: "i64[1024, 1024]" = torch.ops.aten.neg.default(minimum_1);  minimum_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:422, code: is_small = relative_position < max_exact
    lt_1: "b8[1024, 1024]" = torch.ops.aten.lt.Scalar(neg, 16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:426, code: torch.log(relative_position.float() / max_exact)
    convert_element_type_5: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(neg, torch.float32)
    div_8: "f32[1024, 1024]" = torch.ops.aten.div.Tensor(convert_element_type_5, 16);  convert_element_type_5 = None
    log_1: "f32[1024, 1024]" = torch.ops.aten.log.default(div_8);  div_8 = None
    div_9: "f32[1024, 1024]" = torch.ops.aten.div.Tensor(log_1, 2.0794415416798357);  log_1 = None
    mul_34: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(div_9, 16);  div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:429, code: ).to(torch.long)
    convert_element_type_6: "i64[1024, 1024]" = torch.ops.prims.convert_element_type.default(mul_34, torch.int64);  mul_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:425, code: relative_position_if_large = max_exact + (
    add_36: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_6, 16);  convert_element_type_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:431, code: relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
    full_5: "i64[1024, 1024]" = torch.ops.aten.full.default([1024, 1024], 31, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:430, code: relative_position_if_large = torch.min(
    minimum_2: "i64[1024, 1024]" = torch.ops.aten.minimum.default(add_36, full_5);  add_36 = full_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:434, code: relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
    where_1: "i64[1024, 1024]" = torch.ops.aten.where.self(lt_1, neg, minimum_2);  lt_1 = neg = minimum_2 = None
    add_37: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(where_1, 0);  where_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:450, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
    embedding_3: "f32[1024, 1024, 8]" = torch.ops.aten.embedding.default(arg73_1, add_37);  arg73_1 = add_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:451, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
    permute_74: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(embedding_3, [2, 0, 1]);  embedding_3 = None
    unsqueeze_16: "f32[1, 8, 1024, 1024]" = torch.ops.aten.unsqueeze.default(permute_74, 0);  permute_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:552, code: position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)
    add_38: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(unsqueeze_16, mul_30);  unsqueeze_16 = mul_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_39: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_157, add_38);  view_157 = None
    view_158: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(add_39, [8, 1024, 1024]);  add_39 = None
    view_159: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(view_158, [1, 8, 1024, 1024]);  view_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_6: "f32[1, 8, 1024, 1]" = torch.ops.aten.amax.default(view_159, [-1], True)
    sub_11: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_159, amax_6);  view_159 = amax_6 = None
    exp_6: "f32[1, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
    sum_7: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_10: "f32[1, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    clone_33: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(div_10);  div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_26: "f32[1, 8, 1024, 1024]" = torch.ops.aten.expand.default(clone_33, [1, 8, 1024, 1024]);  clone_33 = None
    view_160: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(expand_26, [8, 1024, 1024]);  expand_26 = None
    expand_27: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_72, [1, 8, 1024, 64])
    view_161: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_27, [8, 1024, 64]);  expand_27 = None
    bmm_13: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(view_160, view_161);  view_160 = view_161 = None
    view_162: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_13, [1, 8, 1024, 64]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_75: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
    clone_34: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_75, memory_format = torch.contiguous_format);  permute_75 = None
    view_163: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_34, [1, -1, 512]);  clone_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_76: "f32[512, 512]" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
    view_164: "f32[1024, 512]" = torch.ops.aten.view.default(view_163, [1024, 512]);  view_163 = None
    mm_39: "f32[1024, 512]" = torch.ops.aten.mm.default(view_164, permute_76);  view_164 = permute_76 = None
    view_165: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_39, [1, 1024, 512]);  mm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    clone_35: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(view_165);  view_165 = None
    add_40: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(clone_32, clone_35);  clone_32 = clone_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_15: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_40, 2)
    mean_14: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_15, [-1], True);  pow_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_41: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_14, 1e-06);  mean_14 = None
    rsqrt_14: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
    mul_35: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_40, rsqrt_14);  rsqrt_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_36: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(arg14_1, mul_35);  arg14_1 = mul_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_77: "f32[512, 512]" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
    view_166: "f32[1024, 512]" = torch.ops.aten.view.default(mul_36, [1024, 512]);  mul_36 = None
    mm_40: "f32[1024, 512]" = torch.ops.aten.mm.default(view_166, permute_77);  view_166 = permute_77 = None
    view_167: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_40, [1, 1024, 512]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_168: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_167, [1, -1, 8, 64]);  view_167 = None
    permute_78: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_79: "f32[512, 512]" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
    view_169: "f32[1024, 512]" = torch.ops.aten.view.default(clone_31, [1024, 512])
    mm_41: "f32[1024, 512]" = torch.ops.aten.mm.default(view_169, permute_79);  view_169 = permute_79 = None
    view_170: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_41, [1, 1024, 512]);  mm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_171: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_170, [1, -1, 8, 64]);  view_170 = None
    permute_80: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_171, [0, 2, 1, 3]);  view_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_81: "f32[512, 512]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
    view_172: "f32[1024, 512]" = torch.ops.aten.view.default(clone_31, [1024, 512])
    mm_42: "f32[1024, 512]" = torch.ops.aten.mm.default(view_172, permute_81);  view_172 = permute_81 = None
    view_173: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_42, [1, 1024, 512]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_174: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_173, [1, -1, 8, 64]);  view_173 = None
    permute_82: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_174, [0, 2, 1, 3]);  view_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_83: "f32[1, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_80, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_28: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_78, [1, 8, 1024, 64]);  permute_78 = None
    view_175: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_28, [8, 1024, 64]);  expand_28 = None
    expand_29: "f32[1, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_83, [1, 8, 64, 1024]);  permute_83 = None
    view_176: "f32[8, 64, 1024]" = torch.ops.aten.view.default(expand_29, [8, 64, 1024]);  expand_29 = None
    bmm_14: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_175, view_176);  view_175 = view_176 = None
    view_177: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_14, [1, 8, 1024, 1024]);  bmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:538, code: position_bias = torch.zeros(
    full_6: "f32[1, 8, 1024, 1024]" = torch.ops.aten.full.default([1, 8, 1024, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:552, code: position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)
    add_42: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(full_6, mul_31);  full_6 = mul_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_43: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_177, add_42);  view_177 = None
    view_178: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(add_43, [8, 1024, 1024]);  add_43 = None
    view_179: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(view_178, [1, 8, 1024, 1024]);  view_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_7: "f32[1, 8, 1024, 1]" = torch.ops.aten.amax.default(view_179, [-1], True)
    sub_12: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_179, amax_7);  view_179 = amax_7 = None
    exp_7: "f32[1, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
    sum_8: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_11: "f32[1, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    clone_36: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(div_11);  div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_30: "f32[1, 8, 1024, 1024]" = torch.ops.aten.expand.default(clone_36, [1, 8, 1024, 1024]);  clone_36 = None
    view_180: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(expand_30, [8, 1024, 1024]);  expand_30 = None
    expand_31: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_82, [1, 8, 1024, 64])
    view_181: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_31, [8, 1024, 64]);  expand_31 = None
    bmm_15: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(view_180, view_181);  view_180 = view_181 = None
    view_182: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_15, [1, 8, 1024, 64]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_84: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None
    clone_37: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
    view_183: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_37, [1, -1, 512]);  clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_85: "f32[512, 512]" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
    view_184: "f32[1024, 512]" = torch.ops.aten.view.default(view_183, [1024, 512]);  view_183 = None
    mm_43: "f32[1024, 512]" = torch.ops.aten.mm.default(view_184, permute_85);  view_184 = permute_85 = None
    view_185: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_43, [1, 1024, 512]);  mm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:647, code: layer_output = hidden_states + self.dropout(attention_output[0])
    clone_38: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(view_185);  view_185 = None
    add_44: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_40, clone_38);  add_40 = clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_16: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_44, 2)
    mean_15: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_16, [-1], True);  pow_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_45: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_15, 1e-06);  mean_15 = None
    rsqrt_15: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
    mul_37: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_44, rsqrt_15);  rsqrt_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_38: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(arg15_1, mul_37);  arg15_1 = mul_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_86: "f32[512, 2048]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
    view_186: "f32[1024, 512]" = torch.ops.aten.view.default(mul_38, [1024, 512]);  mul_38 = None
    mm_44: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_186, permute_86);  view_186 = permute_86 = None
    view_187: "f32[1, 1024, 2048]" = torch.ops.aten.view.default(mm_44, [1, 1024, 2048]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    relu_6: "f32[1, 1024, 2048]" = torch.ops.aten.relu.default(view_187);  view_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    clone_39: "f32[1, 1024, 2048]" = torch.ops.aten.clone.default(relu_6);  relu_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_87: "f32[2048, 512]" = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
    view_188: "f32[1024, 2048]" = torch.ops.aten.view.default(clone_39, [1024, 2048]);  clone_39 = None
    mm_45: "f32[1024, 512]" = torch.ops.aten.mm.default(view_188, permute_87);  view_188 = permute_87 = None
    view_189: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_45, [1, 1024, 512]);  mm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    clone_40: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(view_189);  view_189 = None
    add_46: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_44, clone_40);  add_44 = clone_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_17: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_46, 2)
    mean_16: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_17, [-1], True);  pow_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_47: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_16, 1e-06);  mean_16 = None
    rsqrt_16: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
    mul_39: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_46, rsqrt_16);  rsqrt_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_40: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(arg16_1, mul_39);  arg16_1 = mul_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_88: "f32[512, 512]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
    view_190: "f32[1024, 512]" = torch.ops.aten.view.default(mul_40, [1024, 512])
    mm_46: "f32[1024, 512]" = torch.ops.aten.mm.default(view_190, permute_88);  view_190 = permute_88 = None
    view_191: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_46, [1, 1024, 512]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_192: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_191, [1, -1, 8, 64]);  view_191 = None
    permute_89: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_90: "f32[512, 512]" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
    view_193: "f32[1024, 512]" = torch.ops.aten.view.default(mul_40, [1024, 512])
    mm_47: "f32[1024, 512]" = torch.ops.aten.mm.default(view_193, permute_90);  view_193 = permute_90 = None
    view_194: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_47, [1, 1024, 512]);  mm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_195: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_194, [1, -1, 8, 64]);  view_194 = None
    permute_91: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_195, [0, 2, 1, 3]);  view_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_92: "f32[512, 512]" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
    view_196: "f32[1024, 512]" = torch.ops.aten.view.default(mul_40, [1024, 512]);  mul_40 = None
    mm_48: "f32[1024, 512]" = torch.ops.aten.mm.default(view_196, permute_92);  view_196 = permute_92 = None
    view_197: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_48, [1, 1024, 512]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_198: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_197, [1, -1, 8, 64]);  view_197 = None
    permute_93: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_198, [0, 2, 1, 3]);  view_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_94: "f32[1, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_91, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_32: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_89, [1, 8, 1024, 64]);  permute_89 = None
    view_199: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_32, [8, 1024, 64]);  expand_32 = None
    expand_33: "f32[1, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_94, [1, 8, 64, 1024]);  permute_94 = None
    view_200: "f32[8, 64, 1024]" = torch.ops.aten.view.default(expand_33, [8, 64, 1024]);  expand_33 = None
    bmm_16: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_199, view_200);  view_199 = view_200 = None
    view_201: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_16, [1, 8, 1024, 1024]);  bmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_48: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_201, add_38);  view_201 = None
    view_202: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(add_48, [8, 1024, 1024]);  add_48 = None
    view_203: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(view_202, [1, 8, 1024, 1024]);  view_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_8: "f32[1, 8, 1024, 1]" = torch.ops.aten.amax.default(view_203, [-1], True)
    sub_13: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_203, amax_8);  view_203 = amax_8 = None
    exp_8: "f32[1, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
    sum_9: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_12: "f32[1, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    clone_41: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(div_12);  div_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_34: "f32[1, 8, 1024, 1024]" = torch.ops.aten.expand.default(clone_41, [1, 8, 1024, 1024]);  clone_41 = None
    view_204: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(expand_34, [8, 1024, 1024]);  expand_34 = None
    expand_35: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_93, [1, 8, 1024, 64])
    view_205: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_35, [8, 1024, 64]);  expand_35 = None
    bmm_17: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(view_204, view_205);  view_204 = view_205 = None
    view_206: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_17, [1, 8, 1024, 64]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_95: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
    clone_42: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
    view_207: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_42, [1, -1, 512]);  clone_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_96: "f32[512, 512]" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
    view_208: "f32[1024, 512]" = torch.ops.aten.view.default(view_207, [1024, 512]);  view_207 = None
    mm_49: "f32[1024, 512]" = torch.ops.aten.mm.default(view_208, permute_96);  view_208 = permute_96 = None
    view_209: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_49, [1, 1024, 512]);  mm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    clone_43: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(view_209);  view_209 = None
    add_49: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_46, clone_43);  add_46 = clone_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_18: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_49, 2)
    mean_17: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_18, [-1], True);  pow_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_50: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_17, 1e-06);  mean_17 = None
    rsqrt_17: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
    mul_41: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_49, rsqrt_17);  rsqrt_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_42: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(arg17_1, mul_41);  arg17_1 = mul_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_97: "f32[512, 512]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
    view_210: "f32[1024, 512]" = torch.ops.aten.view.default(mul_42, [1024, 512]);  mul_42 = None
    mm_50: "f32[1024, 512]" = torch.ops.aten.mm.default(view_210, permute_97);  view_210 = permute_97 = None
    view_211: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_50, [1, 1024, 512]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_212: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_211, [1, -1, 8, 64]);  view_211 = None
    permute_98: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_99: "f32[512, 512]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
    view_213: "f32[1024, 512]" = torch.ops.aten.view.default(clone_31, [1024, 512])
    mm_51: "f32[1024, 512]" = torch.ops.aten.mm.default(view_213, permute_99);  view_213 = permute_99 = None
    view_214: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_51, [1, 1024, 512]);  mm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_215: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_214, [1, -1, 8, 64]);  view_214 = None
    permute_100: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_215, [0, 2, 1, 3]);  view_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_101: "f32[512, 512]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
    view_216: "f32[1024, 512]" = torch.ops.aten.view.default(clone_31, [1024, 512])
    mm_52: "f32[1024, 512]" = torch.ops.aten.mm.default(view_216, permute_101);  view_216 = permute_101 = None
    view_217: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_52, [1, 1024, 512]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_218: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_217, [1, -1, 8, 64]);  view_217 = None
    permute_102: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_218, [0, 2, 1, 3]);  view_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_103: "f32[1, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_100, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_36: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_98, [1, 8, 1024, 64]);  permute_98 = None
    view_219: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_36, [8, 1024, 64]);  expand_36 = None
    expand_37: "f32[1, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_103, [1, 8, 64, 1024]);  permute_103 = None
    view_220: "f32[8, 64, 1024]" = torch.ops.aten.view.default(expand_37, [8, 64, 1024]);  expand_37 = None
    bmm_18: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_219, view_220);  view_219 = view_220 = None
    view_221: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_18, [1, 8, 1024, 1024]);  bmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_51: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_221, add_42);  view_221 = None
    view_222: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(add_51, [8, 1024, 1024]);  add_51 = None
    view_223: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(view_222, [1, 8, 1024, 1024]);  view_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_9: "f32[1, 8, 1024, 1]" = torch.ops.aten.amax.default(view_223, [-1], True)
    sub_14: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_223, amax_9);  view_223 = amax_9 = None
    exp_9: "f32[1, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
    sum_10: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_13: "f32[1, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    clone_44: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(div_13);  div_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_38: "f32[1, 8, 1024, 1024]" = torch.ops.aten.expand.default(clone_44, [1, 8, 1024, 1024]);  clone_44 = None
    view_224: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(expand_38, [8, 1024, 1024]);  expand_38 = None
    expand_39: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_102, [1, 8, 1024, 64])
    view_225: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_39, [8, 1024, 64]);  expand_39 = None
    bmm_19: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(view_224, view_225);  view_224 = view_225 = None
    view_226: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_19, [1, 8, 1024, 64]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_104: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
    clone_45: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
    view_227: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_45, [1, -1, 512]);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_105: "f32[512, 512]" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
    view_228: "f32[1024, 512]" = torch.ops.aten.view.default(view_227, [1024, 512]);  view_227 = None
    mm_53: "f32[1024, 512]" = torch.ops.aten.mm.default(view_228, permute_105);  view_228 = permute_105 = None
    view_229: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_53, [1, 1024, 512]);  mm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:647, code: layer_output = hidden_states + self.dropout(attention_output[0])
    clone_46: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(view_229);  view_229 = None
    add_52: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_49, clone_46);  add_49 = clone_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_19: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_52, 2)
    mean_18: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_19, [-1], True);  pow_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_53: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_18, 1e-06);  mean_18 = None
    rsqrt_18: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    mul_43: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_52, rsqrt_18);  rsqrt_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_44: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(arg18_1, mul_43);  arg18_1 = mul_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_106: "f32[512, 2048]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
    view_230: "f32[1024, 512]" = torch.ops.aten.view.default(mul_44, [1024, 512]);  mul_44 = None
    mm_54: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_230, permute_106);  view_230 = permute_106 = None
    view_231: "f32[1, 1024, 2048]" = torch.ops.aten.view.default(mm_54, [1, 1024, 2048]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    relu_7: "f32[1, 1024, 2048]" = torch.ops.aten.relu.default(view_231);  view_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    clone_47: "f32[1, 1024, 2048]" = torch.ops.aten.clone.default(relu_7);  relu_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_107: "f32[2048, 512]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
    view_232: "f32[1024, 2048]" = torch.ops.aten.view.default(clone_47, [1024, 2048]);  clone_47 = None
    mm_55: "f32[1024, 512]" = torch.ops.aten.mm.default(view_232, permute_107);  view_232 = permute_107 = None
    view_233: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_55, [1, 1024, 512]);  mm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    clone_48: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(view_233);  view_233 = None
    add_54: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_52, clone_48);  add_52 = clone_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_20: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_54, 2)
    mean_19: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_20, [-1], True);  pow_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_55: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_19, 1e-06);  mean_19 = None
    rsqrt_19: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
    mul_45: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_54, rsqrt_19);  rsqrt_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_46: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(arg19_1, mul_45);  arg19_1 = mul_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_108: "f32[512, 512]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
    view_234: "f32[1024, 512]" = torch.ops.aten.view.default(mul_46, [1024, 512])
    mm_56: "f32[1024, 512]" = torch.ops.aten.mm.default(view_234, permute_108);  view_234 = permute_108 = None
    view_235: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_56, [1, 1024, 512]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_236: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_235, [1, -1, 8, 64]);  view_235 = None
    permute_109: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_236, [0, 2, 1, 3]);  view_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_110: "f32[512, 512]" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
    view_237: "f32[1024, 512]" = torch.ops.aten.view.default(mul_46, [1024, 512])
    mm_57: "f32[1024, 512]" = torch.ops.aten.mm.default(view_237, permute_110);  view_237 = permute_110 = None
    view_238: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_57, [1, 1024, 512]);  mm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_239: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_238, [1, -1, 8, 64]);  view_238 = None
    permute_111: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_239, [0, 2, 1, 3]);  view_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_112: "f32[512, 512]" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
    view_240: "f32[1024, 512]" = torch.ops.aten.view.default(mul_46, [1024, 512]);  mul_46 = None
    mm_58: "f32[1024, 512]" = torch.ops.aten.mm.default(view_240, permute_112);  view_240 = permute_112 = None
    view_241: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_58, [1, 1024, 512]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_242: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_241, [1, -1, 8, 64]);  view_241 = None
    permute_113: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_242, [0, 2, 1, 3]);  view_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_114: "f32[1, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_111, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_40: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_109, [1, 8, 1024, 64]);  permute_109 = None
    view_243: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_40, [8, 1024, 64]);  expand_40 = None
    expand_41: "f32[1, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_114, [1, 8, 64, 1024]);  permute_114 = None
    view_244: "f32[8, 64, 1024]" = torch.ops.aten.view.default(expand_41, [8, 64, 1024]);  expand_41 = None
    bmm_20: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_243, view_244);  view_243 = view_244 = None
    view_245: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_20, [1, 8, 1024, 1024]);  bmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_56: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_245, add_38);  view_245 = None
    view_246: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(add_56, [8, 1024, 1024]);  add_56 = None
    view_247: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(view_246, [1, 8, 1024, 1024]);  view_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_10: "f32[1, 8, 1024, 1]" = torch.ops.aten.amax.default(view_247, [-1], True)
    sub_15: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_247, amax_10);  view_247 = amax_10 = None
    exp_10: "f32[1, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
    sum_11: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_14: "f32[1, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    clone_49: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(div_14);  div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_42: "f32[1, 8, 1024, 1024]" = torch.ops.aten.expand.default(clone_49, [1, 8, 1024, 1024]);  clone_49 = None
    view_248: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(expand_42, [8, 1024, 1024]);  expand_42 = None
    expand_43: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_113, [1, 8, 1024, 64])
    view_249: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_43, [8, 1024, 64]);  expand_43 = None
    bmm_21: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(view_248, view_249);  view_248 = view_249 = None
    view_250: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_21, [1, 8, 1024, 64]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_115: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None
    clone_50: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
    view_251: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_50, [1, -1, 512]);  clone_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_116: "f32[512, 512]" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
    view_252: "f32[1024, 512]" = torch.ops.aten.view.default(view_251, [1024, 512]);  view_251 = None
    mm_59: "f32[1024, 512]" = torch.ops.aten.mm.default(view_252, permute_116);  view_252 = permute_116 = None
    view_253: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_59, [1, 1024, 512]);  mm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    clone_51: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(view_253);  view_253 = None
    add_57: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_54, clone_51);  add_54 = clone_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_21: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_57, 2)
    mean_20: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_21, [-1], True);  pow_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_58: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_20, 1e-06);  mean_20 = None
    rsqrt_20: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
    mul_47: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_57, rsqrt_20);  rsqrt_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_48: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(arg20_1, mul_47);  arg20_1 = mul_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_117: "f32[512, 512]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
    view_254: "f32[1024, 512]" = torch.ops.aten.view.default(mul_48, [1024, 512]);  mul_48 = None
    mm_60: "f32[1024, 512]" = torch.ops.aten.mm.default(view_254, permute_117);  view_254 = permute_117 = None
    view_255: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_60, [1, 1024, 512]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_256: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_255, [1, -1, 8, 64]);  view_255 = None
    permute_118: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_256, [0, 2, 1, 3]);  view_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_119: "f32[512, 512]" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
    view_257: "f32[1024, 512]" = torch.ops.aten.view.default(clone_31, [1024, 512])
    mm_61: "f32[1024, 512]" = torch.ops.aten.mm.default(view_257, permute_119);  view_257 = permute_119 = None
    view_258: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_61, [1, 1024, 512]);  mm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_259: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_258, [1, -1, 8, 64]);  view_258 = None
    permute_120: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_259, [0, 2, 1, 3]);  view_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_121: "f32[512, 512]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
    view_260: "f32[1024, 512]" = torch.ops.aten.view.default(clone_31, [1024, 512])
    mm_62: "f32[1024, 512]" = torch.ops.aten.mm.default(view_260, permute_121);  view_260 = permute_121 = None
    view_261: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_62, [1, 1024, 512]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_262: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_261, [1, -1, 8, 64]);  view_261 = None
    permute_122: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_262, [0, 2, 1, 3]);  view_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_123: "f32[1, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_120, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_44: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_118, [1, 8, 1024, 64]);  permute_118 = None
    view_263: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_44, [8, 1024, 64]);  expand_44 = None
    expand_45: "f32[1, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_123, [1, 8, 64, 1024]);  permute_123 = None
    view_264: "f32[8, 64, 1024]" = torch.ops.aten.view.default(expand_45, [8, 64, 1024]);  expand_45 = None
    bmm_22: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_263, view_264);  view_263 = view_264 = None
    view_265: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_22, [1, 8, 1024, 1024]);  bmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_59: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_265, add_42);  view_265 = None
    view_266: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(add_59, [8, 1024, 1024]);  add_59 = None
    view_267: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(view_266, [1, 8, 1024, 1024]);  view_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_11: "f32[1, 8, 1024, 1]" = torch.ops.aten.amax.default(view_267, [-1], True)
    sub_16: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_267, amax_11);  view_267 = amax_11 = None
    exp_11: "f32[1, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
    sum_12: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_15: "f32[1, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    clone_52: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(div_15);  div_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_46: "f32[1, 8, 1024, 1024]" = torch.ops.aten.expand.default(clone_52, [1, 8, 1024, 1024]);  clone_52 = None
    view_268: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(expand_46, [8, 1024, 1024]);  expand_46 = None
    expand_47: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_122, [1, 8, 1024, 64])
    view_269: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_47, [8, 1024, 64]);  expand_47 = None
    bmm_23: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(view_268, view_269);  view_268 = view_269 = None
    view_270: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_23, [1, 8, 1024, 64]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_124: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_270, [0, 2, 1, 3]);  view_270 = None
    clone_53: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_124, memory_format = torch.contiguous_format);  permute_124 = None
    view_271: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_53, [1, -1, 512]);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_125: "f32[512, 512]" = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
    view_272: "f32[1024, 512]" = torch.ops.aten.view.default(view_271, [1024, 512]);  view_271 = None
    mm_63: "f32[1024, 512]" = torch.ops.aten.mm.default(view_272, permute_125);  view_272 = permute_125 = None
    view_273: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_63, [1, 1024, 512]);  mm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:647, code: layer_output = hidden_states + self.dropout(attention_output[0])
    clone_54: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(view_273);  view_273 = None
    add_60: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_57, clone_54);  add_57 = clone_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_22: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_60, 2)
    mean_21: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_22, [-1], True);  pow_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_61: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_21, 1e-06);  mean_21 = None
    rsqrt_21: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
    mul_49: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_60, rsqrt_21);  rsqrt_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_50: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(arg21_1, mul_49);  arg21_1 = mul_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_126: "f32[512, 2048]" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
    view_274: "f32[1024, 512]" = torch.ops.aten.view.default(mul_50, [1024, 512]);  mul_50 = None
    mm_64: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_274, permute_126);  view_274 = permute_126 = None
    view_275: "f32[1, 1024, 2048]" = torch.ops.aten.view.default(mm_64, [1, 1024, 2048]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    relu_8: "f32[1, 1024, 2048]" = torch.ops.aten.relu.default(view_275);  view_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    clone_55: "f32[1, 1024, 2048]" = torch.ops.aten.clone.default(relu_8);  relu_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_127: "f32[2048, 512]" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
    view_276: "f32[1024, 2048]" = torch.ops.aten.view.default(clone_55, [1024, 2048]);  clone_55 = None
    mm_65: "f32[1024, 512]" = torch.ops.aten.mm.default(view_276, permute_127);  view_276 = permute_127 = None
    view_277: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_65, [1, 1024, 512]);  mm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    clone_56: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(view_277);  view_277 = None
    add_62: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_60, clone_56);  add_60 = clone_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_23: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_62, 2)
    mean_22: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_23, [-1], True);  pow_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_63: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_22, 1e-06);  mean_22 = None
    rsqrt_22: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
    mul_51: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_62, rsqrt_22);  rsqrt_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_52: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(arg22_1, mul_51);  arg22_1 = mul_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_128: "f32[512, 512]" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
    view_278: "f32[1024, 512]" = torch.ops.aten.view.default(mul_52, [1024, 512])
    mm_66: "f32[1024, 512]" = torch.ops.aten.mm.default(view_278, permute_128);  view_278 = permute_128 = None
    view_279: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_66, [1, 1024, 512]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_280: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_279, [1, -1, 8, 64]);  view_279 = None
    permute_129: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_280, [0, 2, 1, 3]);  view_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_130: "f32[512, 512]" = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
    view_281: "f32[1024, 512]" = torch.ops.aten.view.default(mul_52, [1024, 512])
    mm_67: "f32[1024, 512]" = torch.ops.aten.mm.default(view_281, permute_130);  view_281 = permute_130 = None
    view_282: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_67, [1, 1024, 512]);  mm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_283: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_282, [1, -1, 8, 64]);  view_282 = None
    permute_131: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_283, [0, 2, 1, 3]);  view_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_132: "f32[512, 512]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
    view_284: "f32[1024, 512]" = torch.ops.aten.view.default(mul_52, [1024, 512]);  mul_52 = None
    mm_68: "f32[1024, 512]" = torch.ops.aten.mm.default(view_284, permute_132);  view_284 = permute_132 = None
    view_285: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_68, [1, 1024, 512]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_286: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_285, [1, -1, 8, 64]);  view_285 = None
    permute_133: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_286, [0, 2, 1, 3]);  view_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_134: "f32[1, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_131, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_48: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_129, [1, 8, 1024, 64]);  permute_129 = None
    view_287: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_48, [8, 1024, 64]);  expand_48 = None
    expand_49: "f32[1, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_134, [1, 8, 64, 1024]);  permute_134 = None
    view_288: "f32[8, 64, 1024]" = torch.ops.aten.view.default(expand_49, [8, 64, 1024]);  expand_49 = None
    bmm_24: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_287, view_288);  view_287 = view_288 = None
    view_289: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_24, [1, 8, 1024, 1024]);  bmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_64: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_289, add_38);  view_289 = None
    view_290: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(add_64, [8, 1024, 1024]);  add_64 = None
    view_291: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(view_290, [1, 8, 1024, 1024]);  view_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_12: "f32[1, 8, 1024, 1]" = torch.ops.aten.amax.default(view_291, [-1], True)
    sub_17: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_291, amax_12);  view_291 = amax_12 = None
    exp_12: "f32[1, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
    sum_13: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
    div_16: "f32[1, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    clone_57: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(div_16);  div_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_50: "f32[1, 8, 1024, 1024]" = torch.ops.aten.expand.default(clone_57, [1, 8, 1024, 1024]);  clone_57 = None
    view_292: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(expand_50, [8, 1024, 1024]);  expand_50 = None
    expand_51: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_133, [1, 8, 1024, 64])
    view_293: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_51, [8, 1024, 64]);  expand_51 = None
    bmm_25: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(view_292, view_293);  view_292 = view_293 = None
    view_294: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_25, [1, 8, 1024, 64]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_135: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_294, [0, 2, 1, 3]);  view_294 = None
    clone_58: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_135, memory_format = torch.contiguous_format);  permute_135 = None
    view_295: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_58, [1, -1, 512]);  clone_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_136: "f32[512, 512]" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
    view_296: "f32[1024, 512]" = torch.ops.aten.view.default(view_295, [1024, 512]);  view_295 = None
    mm_69: "f32[1024, 512]" = torch.ops.aten.mm.default(view_296, permute_136);  view_296 = permute_136 = None
    view_297: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_69, [1, 1024, 512]);  mm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    clone_59: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(view_297);  view_297 = None
    add_65: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_62, clone_59);  add_62 = clone_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_24: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_65, 2)
    mean_23: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_24, [-1], True);  pow_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_66: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_23, 1e-06);  mean_23 = None
    rsqrt_23: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    mul_53: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_65, rsqrt_23);  rsqrt_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_54: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(arg23_1, mul_53);  arg23_1 = mul_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_137: "f32[512, 512]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
    view_298: "f32[1024, 512]" = torch.ops.aten.view.default(mul_54, [1024, 512]);  mul_54 = None
    mm_70: "f32[1024, 512]" = torch.ops.aten.mm.default(view_298, permute_137);  view_298 = permute_137 = None
    view_299: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_70, [1, 1024, 512]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_300: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_299, [1, -1, 8, 64]);  view_299 = None
    permute_138: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_300, [0, 2, 1, 3]);  view_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_139: "f32[512, 512]" = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
    view_301: "f32[1024, 512]" = torch.ops.aten.view.default(clone_31, [1024, 512])
    mm_71: "f32[1024, 512]" = torch.ops.aten.mm.default(view_301, permute_139);  view_301 = permute_139 = None
    view_302: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_71, [1, 1024, 512]);  mm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_303: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_302, [1, -1, 8, 64]);  view_302 = None
    permute_140: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_303, [0, 2, 1, 3]);  view_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_141: "f32[512, 512]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
    view_304: "f32[1024, 512]" = torch.ops.aten.view.default(clone_31, [1024, 512])
    mm_72: "f32[1024, 512]" = torch.ops.aten.mm.default(view_304, permute_141);  view_304 = permute_141 = None
    view_305: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_72, [1, 1024, 512]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_306: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_305, [1, -1, 8, 64]);  view_305 = None
    permute_142: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_306, [0, 2, 1, 3]);  view_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_143: "f32[1, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_140, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_52: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_138, [1, 8, 1024, 64]);  permute_138 = None
    view_307: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_52, [8, 1024, 64]);  expand_52 = None
    expand_53: "f32[1, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_143, [1, 8, 64, 1024]);  permute_143 = None
    view_308: "f32[8, 64, 1024]" = torch.ops.aten.view.default(expand_53, [8, 64, 1024]);  expand_53 = None
    bmm_26: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_307, view_308);  view_307 = view_308 = None
    view_309: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_26, [1, 8, 1024, 1024]);  bmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_67: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_309, add_42);  view_309 = None
    view_310: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(add_67, [8, 1024, 1024]);  add_67 = None
    view_311: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(view_310, [1, 8, 1024, 1024]);  view_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_13: "f32[1, 8, 1024, 1]" = torch.ops.aten.amax.default(view_311, [-1], True)
    sub_18: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_311, amax_13);  view_311 = amax_13 = None
    exp_13: "f32[1, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_18);  sub_18 = None
    sum_14: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
    div_17: "f32[1, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    clone_60: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(div_17);  div_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_54: "f32[1, 8, 1024, 1024]" = torch.ops.aten.expand.default(clone_60, [1, 8, 1024, 1024]);  clone_60 = None
    view_312: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(expand_54, [8, 1024, 1024]);  expand_54 = None
    expand_55: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_142, [1, 8, 1024, 64])
    view_313: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_55, [8, 1024, 64]);  expand_55 = None
    bmm_27: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(view_312, view_313);  view_312 = view_313 = None
    view_314: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_27, [1, 8, 1024, 64]);  bmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_144: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_314, [0, 2, 1, 3]);  view_314 = None
    clone_61: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_144, memory_format = torch.contiguous_format);  permute_144 = None
    view_315: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_61, [1, -1, 512]);  clone_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_145: "f32[512, 512]" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
    view_316: "f32[1024, 512]" = torch.ops.aten.view.default(view_315, [1024, 512]);  view_315 = None
    mm_73: "f32[1024, 512]" = torch.ops.aten.mm.default(view_316, permute_145);  view_316 = permute_145 = None
    view_317: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_73, [1, 1024, 512]);  mm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:647, code: layer_output = hidden_states + self.dropout(attention_output[0])
    clone_62: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(view_317);  view_317 = None
    add_68: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_65, clone_62);  add_65 = clone_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_25: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_68, 2)
    mean_24: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_25, [-1], True);  pow_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_69: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_24, 1e-06);  mean_24 = None
    rsqrt_24: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
    mul_55: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_68, rsqrt_24);  rsqrt_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_56: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(arg24_1, mul_55);  arg24_1 = mul_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_146: "f32[512, 2048]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
    view_318: "f32[1024, 512]" = torch.ops.aten.view.default(mul_56, [1024, 512]);  mul_56 = None
    mm_74: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_318, permute_146);  view_318 = permute_146 = None
    view_319: "f32[1, 1024, 2048]" = torch.ops.aten.view.default(mm_74, [1, 1024, 2048]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    relu_9: "f32[1, 1024, 2048]" = torch.ops.aten.relu.default(view_319);  view_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    clone_63: "f32[1, 1024, 2048]" = torch.ops.aten.clone.default(relu_9);  relu_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_147: "f32[2048, 512]" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
    view_320: "f32[1024, 2048]" = torch.ops.aten.view.default(clone_63, [1024, 2048]);  clone_63 = None
    mm_75: "f32[1024, 512]" = torch.ops.aten.mm.default(view_320, permute_147);  view_320 = permute_147 = None
    view_321: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_75, [1, 1024, 512]);  mm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    clone_64: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(view_321);  view_321 = None
    add_70: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_68, clone_64);  add_68 = clone_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_26: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_70, 2)
    mean_25: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_26, [-1], True);  pow_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_71: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_25, 1e-06);  mean_25 = None
    rsqrt_25: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
    mul_57: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_70, rsqrt_25);  rsqrt_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_58: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(arg25_1, mul_57);  arg25_1 = mul_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_148: "f32[512, 512]" = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
    view_322: "f32[1024, 512]" = torch.ops.aten.view.default(mul_58, [1024, 512])
    mm_76: "f32[1024, 512]" = torch.ops.aten.mm.default(view_322, permute_148);  view_322 = permute_148 = None
    view_323: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_76, [1, 1024, 512]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_324: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_323, [1, -1, 8, 64]);  view_323 = None
    permute_149: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_324, [0, 2, 1, 3]);  view_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_150: "f32[512, 512]" = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
    view_325: "f32[1024, 512]" = torch.ops.aten.view.default(mul_58, [1024, 512])
    mm_77: "f32[1024, 512]" = torch.ops.aten.mm.default(view_325, permute_150);  view_325 = permute_150 = None
    view_326: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_77, [1, 1024, 512]);  mm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_327: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_326, [1, -1, 8, 64]);  view_326 = None
    permute_151: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_327, [0, 2, 1, 3]);  view_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_152: "f32[512, 512]" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
    view_328: "f32[1024, 512]" = torch.ops.aten.view.default(mul_58, [1024, 512]);  mul_58 = None
    mm_78: "f32[1024, 512]" = torch.ops.aten.mm.default(view_328, permute_152);  view_328 = permute_152 = None
    view_329: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_78, [1, 1024, 512]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_330: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_329, [1, -1, 8, 64]);  view_329 = None
    permute_153: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_330, [0, 2, 1, 3]);  view_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_154: "f32[1, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_151, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_56: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_149, [1, 8, 1024, 64]);  permute_149 = None
    view_331: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_56, [8, 1024, 64]);  expand_56 = None
    expand_57: "f32[1, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_154, [1, 8, 64, 1024]);  permute_154 = None
    view_332: "f32[8, 64, 1024]" = torch.ops.aten.view.default(expand_57, [8, 64, 1024]);  expand_57 = None
    bmm_28: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_331, view_332);  view_331 = view_332 = None
    view_333: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_28, [1, 8, 1024, 1024]);  bmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_72: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_333, add_38);  view_333 = None
    view_334: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(add_72, [8, 1024, 1024]);  add_72 = None
    view_335: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(view_334, [1, 8, 1024, 1024]);  view_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_14: "f32[1, 8, 1024, 1]" = torch.ops.aten.amax.default(view_335, [-1], True)
    sub_19: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_335, amax_14);  view_335 = amax_14 = None
    exp_14: "f32[1, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
    sum_15: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
    div_18: "f32[1, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    clone_65: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(div_18);  div_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_58: "f32[1, 8, 1024, 1024]" = torch.ops.aten.expand.default(clone_65, [1, 8, 1024, 1024]);  clone_65 = None
    view_336: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(expand_58, [8, 1024, 1024]);  expand_58 = None
    expand_59: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_153, [1, 8, 1024, 64])
    view_337: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_59, [8, 1024, 64]);  expand_59 = None
    bmm_29: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(view_336, view_337);  view_336 = view_337 = None
    view_338: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_29, [1, 8, 1024, 64]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_155: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_338, [0, 2, 1, 3]);  view_338 = None
    clone_66: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_155, memory_format = torch.contiguous_format);  permute_155 = None
    view_339: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_66, [1, -1, 512]);  clone_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_156: "f32[512, 512]" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
    view_340: "f32[1024, 512]" = torch.ops.aten.view.default(view_339, [1024, 512]);  view_339 = None
    mm_79: "f32[1024, 512]" = torch.ops.aten.mm.default(view_340, permute_156);  view_340 = permute_156 = None
    view_341: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_79, [1, 1024, 512]);  mm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    clone_67: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(view_341);  view_341 = None
    add_73: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_70, clone_67);  add_70 = clone_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_27: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_73, 2)
    mean_26: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_27, [-1], True);  pow_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_74: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_26, 1e-06);  mean_26 = None
    rsqrt_26: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    mul_59: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_73, rsqrt_26);  rsqrt_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_60: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(arg26_1, mul_59);  arg26_1 = mul_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_157: "f32[512, 512]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
    view_342: "f32[1024, 512]" = torch.ops.aten.view.default(mul_60, [1024, 512]);  mul_60 = None
    mm_80: "f32[1024, 512]" = torch.ops.aten.mm.default(view_342, permute_157);  view_342 = permute_157 = None
    view_343: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_80, [1, 1024, 512]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_344: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_343, [1, -1, 8, 64]);  view_343 = None
    permute_158: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_344, [0, 2, 1, 3]);  view_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_159: "f32[512, 512]" = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
    view_345: "f32[1024, 512]" = torch.ops.aten.view.default(clone_31, [1024, 512])
    mm_81: "f32[1024, 512]" = torch.ops.aten.mm.default(view_345, permute_159);  view_345 = permute_159 = None
    view_346: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_81, [1, 1024, 512]);  mm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_347: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_346, [1, -1, 8, 64]);  view_346 = None
    permute_160: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_347, [0, 2, 1, 3]);  view_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_161: "f32[512, 512]" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
    view_348: "f32[1024, 512]" = torch.ops.aten.view.default(clone_31, [1024, 512])
    mm_82: "f32[1024, 512]" = torch.ops.aten.mm.default(view_348, permute_161);  view_348 = permute_161 = None
    view_349: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_82, [1, 1024, 512]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_350: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_349, [1, -1, 8, 64]);  view_349 = None
    permute_162: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_350, [0, 2, 1, 3]);  view_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_163: "f32[1, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_160, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_60: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_158, [1, 8, 1024, 64]);  permute_158 = None
    view_351: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_60, [8, 1024, 64]);  expand_60 = None
    expand_61: "f32[1, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_163, [1, 8, 64, 1024]);  permute_163 = None
    view_352: "f32[8, 64, 1024]" = torch.ops.aten.view.default(expand_61, [8, 64, 1024]);  expand_61 = None
    bmm_30: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_351, view_352);  view_351 = view_352 = None
    view_353: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_30, [1, 8, 1024, 1024]);  bmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_75: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_353, add_42);  view_353 = None
    view_354: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(add_75, [8, 1024, 1024]);  add_75 = None
    view_355: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(view_354, [1, 8, 1024, 1024]);  view_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_15: "f32[1, 8, 1024, 1]" = torch.ops.aten.amax.default(view_355, [-1], True)
    sub_20: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_355, amax_15);  view_355 = amax_15 = None
    exp_15: "f32[1, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
    sum_16: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
    div_19: "f32[1, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    clone_68: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(div_19);  div_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_62: "f32[1, 8, 1024, 1024]" = torch.ops.aten.expand.default(clone_68, [1, 8, 1024, 1024]);  clone_68 = None
    view_356: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(expand_62, [8, 1024, 1024]);  expand_62 = None
    expand_63: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_162, [1, 8, 1024, 64])
    view_357: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_63, [8, 1024, 64]);  expand_63 = None
    bmm_31: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(view_356, view_357);  view_356 = view_357 = None
    view_358: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_31, [1, 8, 1024, 64]);  bmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_164: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_358, [0, 2, 1, 3]);  view_358 = None
    clone_69: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_164, memory_format = torch.contiguous_format);  permute_164 = None
    view_359: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_69, [1, -1, 512]);  clone_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_165: "f32[512, 512]" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
    view_360: "f32[1024, 512]" = torch.ops.aten.view.default(view_359, [1024, 512]);  view_359 = None
    mm_83: "f32[1024, 512]" = torch.ops.aten.mm.default(view_360, permute_165);  view_360 = permute_165 = None
    view_361: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_83, [1, 1024, 512]);  mm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:647, code: layer_output = hidden_states + self.dropout(attention_output[0])
    clone_70: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(view_361);  view_361 = None
    add_76: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_73, clone_70);  add_73 = clone_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_28: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_76, 2)
    mean_27: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_28, [-1], True);  pow_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_77: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_27, 1e-06);  mean_27 = None
    rsqrt_27: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
    mul_61: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_76, rsqrt_27);  rsqrt_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_62: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(arg27_1, mul_61);  arg27_1 = mul_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_166: "f32[512, 2048]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
    view_362: "f32[1024, 512]" = torch.ops.aten.view.default(mul_62, [1024, 512]);  mul_62 = None
    mm_84: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_362, permute_166);  view_362 = permute_166 = None
    view_363: "f32[1, 1024, 2048]" = torch.ops.aten.view.default(mm_84, [1, 1024, 2048]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    relu_10: "f32[1, 1024, 2048]" = torch.ops.aten.relu.default(view_363);  view_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    clone_71: "f32[1, 1024, 2048]" = torch.ops.aten.clone.default(relu_10);  relu_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_167: "f32[2048, 512]" = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
    view_364: "f32[1024, 2048]" = torch.ops.aten.view.default(clone_71, [1024, 2048]);  clone_71 = None
    mm_85: "f32[1024, 512]" = torch.ops.aten.mm.default(view_364, permute_167);  view_364 = permute_167 = None
    view_365: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_85, [1, 1024, 512]);  mm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    clone_72: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(view_365);  view_365 = None
    add_78: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_76, clone_72);  add_76 = clone_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_29: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_78, 2)
    mean_28: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_29, [-1], True);  pow_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_79: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_28, 1e-06);  mean_28 = None
    rsqrt_28: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
    mul_63: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_78, rsqrt_28);  rsqrt_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_64: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(arg28_1, mul_63);  arg28_1 = mul_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_168: "f32[512, 512]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
    view_366: "f32[1024, 512]" = torch.ops.aten.view.default(mul_64, [1024, 512])
    mm_86: "f32[1024, 512]" = torch.ops.aten.mm.default(view_366, permute_168);  view_366 = permute_168 = None
    view_367: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_86, [1, 1024, 512]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_368: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_367, [1, -1, 8, 64]);  view_367 = None
    permute_169: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_368, [0, 2, 1, 3]);  view_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_170: "f32[512, 512]" = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
    view_369: "f32[1024, 512]" = torch.ops.aten.view.default(mul_64, [1024, 512])
    mm_87: "f32[1024, 512]" = torch.ops.aten.mm.default(view_369, permute_170);  view_369 = permute_170 = None
    view_370: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_87, [1, 1024, 512]);  mm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_371: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_370, [1, -1, 8, 64]);  view_370 = None
    permute_171: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_371, [0, 2, 1, 3]);  view_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:498, code: hidden_states = shape(proj_layer(hidden_states))
    permute_172: "f32[512, 512]" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
    view_372: "f32[1024, 512]" = torch.ops.aten.view.default(mul_64, [1024, 512]);  mul_64 = None
    mm_88: "f32[1024, 512]" = torch.ops.aten.mm.default(view_372, permute_172);  view_372 = permute_172 = None
    view_373: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_88, [1, 1024, 512]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_374: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_373, [1, -1, 8, 64]);  view_373 = None
    permute_173: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_374, [0, 2, 1, 3]);  view_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_174: "f32[1, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_171, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_64: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_169, [1, 8, 1024, 64]);  permute_169 = None
    view_375: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_64, [8, 1024, 64]);  expand_64 = None
    expand_65: "f32[1, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_174, [1, 8, 64, 1024]);  permute_174 = None
    view_376: "f32[8, 64, 1024]" = torch.ops.aten.view.default(expand_65, [8, 64, 1024]);  expand_65 = None
    bmm_32: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_375, view_376);  view_375 = view_376 = None
    view_377: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_32, [1, 8, 1024, 1024]);  bmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_80: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_377, add_38);  view_377 = add_38 = None
    view_378: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(add_80, [8, 1024, 1024]);  add_80 = None
    view_379: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(view_378, [1, 8, 1024, 1024]);  view_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_16: "f32[1, 8, 1024, 1]" = torch.ops.aten.amax.default(view_379, [-1], True)
    sub_21: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_379, amax_16);  view_379 = amax_16 = None
    exp_16: "f32[1, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_21);  sub_21 = None
    sum_17: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
    div_20: "f32[1, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    clone_73: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(div_20);  div_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_66: "f32[1, 8, 1024, 1024]" = torch.ops.aten.expand.default(clone_73, [1, 8, 1024, 1024]);  clone_73 = None
    view_380: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(expand_66, [8, 1024, 1024]);  expand_66 = None
    expand_67: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_173, [1, 8, 1024, 64])
    view_381: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_67, [8, 1024, 64]);  expand_67 = None
    bmm_33: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(view_380, view_381);  view_380 = view_381 = None
    view_382: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_33, [1, 8, 1024, 64]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_175: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_382, [0, 2, 1, 3]);  view_382 = None
    clone_74: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_175, memory_format = torch.contiguous_format);  permute_175 = None
    view_383: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_74, [1, -1, 512]);  clone_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_176: "f32[512, 512]" = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
    view_384: "f32[1024, 512]" = torch.ops.aten.view.default(view_383, [1024, 512]);  view_383 = None
    mm_89: "f32[1024, 512]" = torch.ops.aten.mm.default(view_384, permute_176);  view_384 = permute_176 = None
    view_385: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_89, [1, 1024, 512]);  mm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:611, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    clone_75: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(view_385);  view_385 = None
    add_81: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_78, clone_75);  add_78 = clone_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_30: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_81, 2)
    mean_29: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_30, [-1], True);  pow_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_82: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_29, 1e-06);  mean_29 = None
    rsqrt_29: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
    mul_65: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_81, rsqrt_29);  rsqrt_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_66: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(arg29_1, mul_65);  arg29_1 = mul_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:521, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_177: "f32[512, 512]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
    view_386: "f32[1024, 512]" = torch.ops.aten.view.default(mul_66, [1024, 512]);  mul_66 = None
    mm_90: "f32[1024, 512]" = torch.ops.aten.mm.default(view_386, permute_177);  view_386 = permute_177 = None
    view_387: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_90, [1, 1024, 512]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_388: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_387, [1, -1, 8, 64]);  view_387 = None
    permute_178: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_388, [0, 2, 1, 3]);  view_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_179: "f32[512, 512]" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
    view_389: "f32[1024, 512]" = torch.ops.aten.view.default(clone_31, [1024, 512])
    mm_91: "f32[1024, 512]" = torch.ops.aten.mm.default(view_389, permute_179);  view_389 = permute_179 = None
    view_390: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_91, [1, 1024, 512]);  mm_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_391: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_390, [1, -1, 8, 64]);  view_390 = None
    permute_180: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_391, [0, 2, 1, 3]);  view_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:502, code: hidden_states = shape(proj_layer(key_value_states))
    permute_181: "f32[512, 512]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
    view_392: "f32[1024, 512]" = torch.ops.aten.view.default(clone_31, [1024, 512])
    mm_92: "f32[1024, 512]" = torch.ops.aten.mm.default(view_392, permute_181);  view_392 = permute_181 = None
    view_393: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_92, [1, 1024, 512]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:487, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_394: "f32[1, 1024, 8, 64]" = torch.ops.aten.view.default(view_393, [1, -1, 8, 64]);  view_393 = None
    permute_182: "f32[1, 8, 1024, 64]" = torch.ops.aten.permute.default(view_394, [0, 2, 1, 3]);  view_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:533, code: query_states, key_states.transpose(3, 2)
    permute_183: "f32[1, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_180, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:532, code: scores = torch.matmul(
    expand_68: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_178, [1, 8, 1024, 64]);  permute_178 = None
    view_395: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_68, [8, 1024, 64]);  expand_68 = None
    expand_69: "f32[1, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_183, [1, 8, 64, 1024]);  permute_183 = None
    view_396: "f32[8, 64, 1024]" = torch.ops.aten.view.default(expand_69, [8, 64, 1024]);  expand_69 = None
    bmm_34: "f32[8, 1024, 1024]" = torch.ops.aten.bmm.default(view_395, view_396);  view_395 = view_396 = None
    view_397: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_34, [1, 8, 1024, 1024]);  bmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:561, code: scores += position_bias_masked
    add_83: "f32[1, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_397, add_42);  view_397 = add_42 = None
    view_398: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(add_83, [8, 1024, 1024]);  add_83 = None
    view_399: "f32[1, 8, 1024, 1024]" = torch.ops.aten.view.default(view_398, [1, 8, 1024, 1024]);  view_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:562, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_17: "f32[1, 8, 1024, 1]" = torch.ops.aten.amax.default(view_399, [-1], True)
    sub_22: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(view_399, amax_17);  view_399 = amax_17 = None
    exp_17: "f32[1, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
    sum_18: "f32[1, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
    div_21: "f32[1, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:565, code: attn_weights = nn.functional.dropout(
    clone_76: "f32[1, 8, 1024, 1024]" = torch.ops.aten.clone.default(div_21);  div_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:573, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_70: "f32[1, 8, 1024, 1024]" = torch.ops.aten.expand.default(clone_76, [1, 8, 1024, 1024]);  clone_76 = None
    view_400: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(expand_70, [8, 1024, 1024]);  expand_70 = None
    expand_71: "f32[1, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_182, [1, 8, 1024, 64])
    view_401: "f32[8, 1024, 64]" = torch.ops.aten.view.default(expand_71, [8, 1024, 64]);  expand_71 = None
    bmm_35: "f32[8, 1024, 64]" = torch.ops.aten.bmm.default(view_400, view_401);  view_400 = view_401 = None
    view_402: "f32[1, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_35, [1, 8, 1024, 64]);  bmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:491, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_184: "f32[1, 1024, 8, 64]" = torch.ops.aten.permute.default(view_402, [0, 2, 1, 3]);  view_402 = None
    clone_77: "f32[1, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_184, memory_format = torch.contiguous_format);  permute_184 = None
    view_403: "f32[1, 1024, 512]" = torch.ops.aten.view.default(clone_77, [1, -1, 512]);  clone_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:574, code: attn_output = self.o(attn_output)
    permute_185: "f32[512, 512]" = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
    view_404: "f32[1024, 512]" = torch.ops.aten.view.default(view_403, [1024, 512]);  view_403 = None
    mm_93: "f32[1024, 512]" = torch.ops.aten.mm.default(view_404, permute_185);  view_404 = permute_185 = None
    view_405: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_93, [1, 1024, 512]);  mm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:647, code: layer_output = hidden_states + self.dropout(attention_output[0])
    clone_78: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(view_405);  view_405 = None
    add_84: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_81, clone_78);  add_81 = clone_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_31: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_84, 2)
    mean_30: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_31, [-1], True);  pow_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_85: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_30, 1e-06);  mean_30 = None
    rsqrt_30: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
    mul_67: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_84, rsqrt_30);  rsqrt_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_68: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(arg30_1, mul_67);  arg30_1 = mul_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:289, code: hidden_states = self.wi(hidden_states)
    permute_186: "f32[512, 2048]" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
    view_406: "f32[1024, 512]" = torch.ops.aten.view.default(mul_68, [1024, 512]);  mul_68 = None
    mm_94: "f32[1024, 2048]" = torch.ops.aten.mm.default(view_406, permute_186);  view_406 = permute_186 = None
    view_407: "f32[1, 1024, 2048]" = torch.ops.aten.view.default(mm_94, [1, 1024, 2048]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:290, code: hidden_states = self.act(hidden_states)
    relu_11: "f32[1, 1024, 2048]" = torch.ops.aten.relu.default(view_407);  view_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:291, code: hidden_states = self.dropout(hidden_states)
    clone_79: "f32[1, 1024, 2048]" = torch.ops.aten.clone.default(relu_11);  relu_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:298, code: hidden_states = self.wo(hidden_states)
    permute_187: "f32[2048, 512]" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
    view_408: "f32[1024, 2048]" = torch.ops.aten.view.default(clone_79, [1024, 2048]);  clone_79 = None
    mm_95: "f32[1024, 512]" = torch.ops.aten.mm.default(view_408, permute_187);  view_408 = permute_187 = None
    view_409: "f32[1, 1024, 512]" = torch.ops.aten.view.default(mm_95, [1, 1024, 512]);  mm_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:345, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    clone_80: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(view_409);  view_409 = None
    add_86: "f32[1, 1024, 512]" = torch.ops.aten.add.Tensor(add_84, clone_80);  add_84 = clone_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:254, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_32: "f32[1, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_86, 2)
    mean_31: "f32[1, 1024, 1]" = torch.ops.aten.mean.dim(pow_32, [-1], True);  pow_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:255, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_87: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(mean_31, 1e-06);  mean_31 = None
    rsqrt_31: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
    mul_69: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(add_86, rsqrt_31);  add_86 = rsqrt_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:261, code: return self.weight * hidden_states
    mul_70: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(arg31_1, mul_69);  arg31_1 = mul_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1166, code: hidden_states = self.dropout(hidden_states)
    clone_81: "f32[1, 1024, 512]" = torch.ops.aten.clone.default(mul_70);  mul_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1772, code: sequence_output = sequence_output * (self.model_dim**-0.5)
    mul_71: "f32[1, 1024, 512]" = torch.ops.aten.mul.Tensor(clone_81, 0.04419417382415922);  clone_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1774, code: lm_logits = self.lm_head(sequence_output)
    permute_188: "f32[512, 32128]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
    view_410: "f32[1024, 512]" = torch.ops.aten.view.default(mul_71, [1024, 512]);  mul_71 = None
    mm_96: "f32[1024, 32128]" = torch.ops.aten.mm.default(view_410, permute_188);  view_410 = permute_188 = None
    view_411: "f32[1, 1024, 32128]" = torch.ops.aten.view.default(mm_96, [1, 1024, 32128]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/t5/modeling_t5.py:1781, code: loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
    view_412: "f32[1024, 32128]" = torch.ops.aten.view.default(view_411, [-1, 32128])
    view_413: "i64[1024]" = torch.ops.aten.view.default(arg133_1, [-1]);  arg133_1 = None
    amax_18: "f32[1024, 1]" = torch.ops.aten.amax.default(view_412, [1], True)
    sub_23: "f32[1024, 32128]" = torch.ops.aten.sub.Tensor(view_412, amax_18);  view_412 = amax_18 = None
    exp_18: "f32[1024, 32128]" = torch.ops.aten.exp.default(sub_23)
    sum_19: "f32[1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [1], True);  exp_18 = None
    log_2: "f32[1024, 1]" = torch.ops.aten.log.default(sum_19);  sum_19 = None
    sub_24: "f32[1024, 32128]" = torch.ops.aten.sub.Tensor(sub_23, log_2);  sub_23 = log_2 = None
    ne: "b8[1024]" = torch.ops.aten.ne.Scalar(view_413, -100)
    scalar_tensor: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'))
    where_2: "i64[1024]" = torch.ops.aten.where.self(ne, view_413, scalar_tensor);  ne = scalar_tensor = None
    unsqueeze_17: "i64[1024, 1]" = torch.ops.aten.unsqueeze.default(where_2, 1);  where_2 = None
    gather: "f32[1024, 1]" = torch.ops.aten.gather.default(sub_24, 1, unsqueeze_17);  sub_24 = unsqueeze_17 = None
    squeeze: "f32[1024]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg_1: "f32[1024]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
    ne_1: "b8[1024]" = torch.ops.aten.ne.Scalar(view_413, -100)
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_3: "f32[1024]" = torch.ops.aten.where.self(ne_1, neg_1, scalar_tensor_1);  ne_1 = neg_1 = scalar_tensor_1 = None
    ne_2: "b8[1024]" = torch.ops.aten.ne.Scalar(view_413, -100);  view_413 = None
    sum_20: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
    convert_element_type_7: "f32[]" = torch.ops.prims.convert_element_type.default(sum_20, torch.float32);  sum_20 = None
    sum_21: "f32[]" = torch.ops.aten.sum.default(where_3);  where_3 = None
    div_22: "f32[]" = torch.ops.aten.div.Tensor(sum_21, convert_element_type_7);  sum_21 = convert_element_type_7 = None
    return (div_22, view_411, permute_70, permute_72, permute_80, permute_82, permute_91, permute_93, permute_100, permute_102, permute_111, permute_113, permute_120, permute_122, permute_131, permute_133, permute_140, permute_142, permute_151, permute_153, permute_160, permute_162, permute_171, permute_173, permute_180, permute_182, clone_31)
    