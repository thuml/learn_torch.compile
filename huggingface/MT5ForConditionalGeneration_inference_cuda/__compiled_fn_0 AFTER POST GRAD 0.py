from __future__ import annotations



def forward(self, arg0_1: "f32[512]", arg1_1: "f32[512]", arg2_1: "f32[512]", arg3_1: "f32[512]", arg4_1: "f32[512]", arg5_1: "f32[512]", arg6_1: "f32[512]", arg7_1: "f32[512]", arg8_1: "f32[512]", arg9_1: "f32[512]", arg10_1: "f32[512]", arg11_1: "f32[512]", arg12_1: "f32[512]", arg13_1: "f32[512]", arg14_1: "f32[512]", arg15_1: "f32[512]", arg16_1: "f32[512]", arg17_1: "f32[512]", arg18_1: "f32[512]", arg19_1: "f32[512]", arg20_1: "f32[512]", arg21_1: "f32[512]", arg22_1: "f32[512]", arg23_1: "f32[512]", arg24_1: "f32[512]", arg25_1: "f32[512]", arg26_1: "f32[512]", arg27_1: "f32[512]", arg28_1: "f32[512]", arg29_1: "f32[512]", arg30_1: "f32[512]", arg31_1: "f32[512]", arg32_1: "f32[512]", arg33_1: "f32[512]", arg34_1: "f32[512]", arg35_1: "f32[512]", arg36_1: "f32[512]", arg37_1: "f32[512]", arg38_1: "f32[512]", arg39_1: "f32[512]", arg40_1: "f32[512]", arg41_1: "f32[512]", arg42_1: "f32[250112, 512]", arg43_1: "f32[384, 512]", arg44_1: "f32[384, 512]", arg45_1: "f32[384, 512]", arg46_1: "f32[32, 6]", arg47_1: "f32[512, 384]", arg48_1: "f32[1024, 512]", arg49_1: "f32[1024, 512]", arg50_1: "f32[512, 1024]", arg51_1: "f32[384, 512]", arg52_1: "f32[384, 512]", arg53_1: "f32[384, 512]", arg54_1: "f32[512, 384]", arg55_1: "f32[1024, 512]", arg56_1: "f32[1024, 512]", arg57_1: "f32[512, 1024]", arg58_1: "f32[384, 512]", arg59_1: "f32[384, 512]", arg60_1: "f32[384, 512]", arg61_1: "f32[512, 384]", arg62_1: "f32[1024, 512]", arg63_1: "f32[1024, 512]", arg64_1: "f32[512, 1024]", arg65_1: "f32[384, 512]", arg66_1: "f32[384, 512]", arg67_1: "f32[384, 512]", arg68_1: "f32[512, 384]", arg69_1: "f32[1024, 512]", arg70_1: "f32[1024, 512]", arg71_1: "f32[512, 1024]", arg72_1: "f32[384, 512]", arg73_1: "f32[384, 512]", arg74_1: "f32[384, 512]", arg75_1: "f32[512, 384]", arg76_1: "f32[1024, 512]", arg77_1: "f32[1024, 512]", arg78_1: "f32[512, 1024]", arg79_1: "f32[384, 512]", arg80_1: "f32[384, 512]", arg81_1: "f32[384, 512]", arg82_1: "f32[512, 384]", arg83_1: "f32[1024, 512]", arg84_1: "f32[1024, 512]", arg85_1: "f32[512, 1024]", arg86_1: "f32[384, 512]", arg87_1: "f32[384, 512]", arg88_1: "f32[384, 512]", arg89_1: "f32[512, 384]", arg90_1: "f32[1024, 512]", arg91_1: "f32[1024, 512]", arg92_1: "f32[512, 1024]", arg93_1: "f32[384, 512]", arg94_1: "f32[384, 512]", arg95_1: "f32[384, 512]", arg96_1: "f32[512, 384]", arg97_1: "f32[1024, 512]", arg98_1: "f32[1024, 512]", arg99_1: "f32[512, 1024]", arg100_1: "f32[384, 512]", arg101_1: "f32[384, 512]", arg102_1: "f32[384, 512]", arg103_1: "f32[32, 6]", arg104_1: "f32[512, 384]", arg105_1: "f32[384, 512]", arg106_1: "f32[384, 512]", arg107_1: "f32[384, 512]", arg108_1: "f32[512, 384]", arg109_1: "f32[1024, 512]", arg110_1: "f32[1024, 512]", arg111_1: "f32[512, 1024]", arg112_1: "f32[384, 512]", arg113_1: "f32[384, 512]", arg114_1: "f32[384, 512]", arg115_1: "f32[512, 384]", arg116_1: "f32[384, 512]", arg117_1: "f32[384, 512]", arg118_1: "f32[384, 512]", arg119_1: "f32[512, 384]", arg120_1: "f32[1024, 512]", arg121_1: "f32[1024, 512]", arg122_1: "f32[512, 1024]", arg123_1: "f32[384, 512]", arg124_1: "f32[384, 512]", arg125_1: "f32[384, 512]", arg126_1: "f32[512, 384]", arg127_1: "f32[384, 512]", arg128_1: "f32[384, 512]", arg129_1: "f32[384, 512]", arg130_1: "f32[512, 384]", arg131_1: "f32[1024, 512]", arg132_1: "f32[1024, 512]", arg133_1: "f32[512, 1024]", arg134_1: "f32[384, 512]", arg135_1: "f32[384, 512]", arg136_1: "f32[384, 512]", arg137_1: "f32[512, 384]", arg138_1: "f32[384, 512]", arg139_1: "f32[384, 512]", arg140_1: "f32[384, 512]", arg141_1: "f32[512, 384]", arg142_1: "f32[1024, 512]", arg143_1: "f32[1024, 512]", arg144_1: "f32[512, 1024]", arg145_1: "f32[384, 512]", arg146_1: "f32[384, 512]", arg147_1: "f32[384, 512]", arg148_1: "f32[512, 384]", arg149_1: "f32[384, 512]", arg150_1: "f32[384, 512]", arg151_1: "f32[384, 512]", arg152_1: "f32[512, 384]", arg153_1: "f32[1024, 512]", arg154_1: "f32[1024, 512]", arg155_1: "f32[512, 1024]", arg156_1: "f32[384, 512]", arg157_1: "f32[384, 512]", arg158_1: "f32[384, 512]", arg159_1: "f32[512, 384]", arg160_1: "f32[384, 512]", arg161_1: "f32[384, 512]", arg162_1: "f32[384, 512]", arg163_1: "f32[512, 384]", arg164_1: "f32[1024, 512]", arg165_1: "f32[1024, 512]", arg166_1: "f32[512, 1024]", arg167_1: "f32[384, 512]", arg168_1: "f32[384, 512]", arg169_1: "f32[384, 512]", arg170_1: "f32[512, 384]", arg171_1: "f32[384, 512]", arg172_1: "f32[384, 512]", arg173_1: "f32[384, 512]", arg174_1: "f32[512, 384]", arg175_1: "f32[1024, 512]", arg176_1: "f32[1024, 512]", arg177_1: "f32[512, 1024]", arg178_1: "f32[384, 512]", arg179_1: "f32[384, 512]", arg180_1: "f32[384, 512]", arg181_1: "f32[512, 384]", arg182_1: "f32[384, 512]", arg183_1: "f32[384, 512]", arg184_1: "f32[384, 512]", arg185_1: "f32[512, 384]", arg186_1: "f32[1024, 512]", arg187_1: "f32[1024, 512]", arg188_1: "f32[512, 1024]", arg189_1: "f32[250112, 512]", arg190_1: "i64[1, 128]", arg191_1: "i64[1, 128]", arg192_1: "i64[1, 128]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:1006, code: attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
    full: "f32[1, 128]" = torch.ops.aten.full.default([1, 128], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:916, code: extended_attention_mask = attention_mask[:, None, None, :]
    unsqueeze: "f32[1, 1, 128]" = torch.ops.aten.unsqueeze.default(full, 1);  full = None
    unsqueeze_1: "f32[1, 1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:928, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    sub: "f32[1, 1, 1, 128]" = torch.ops.aten.sub.Tensor(1.0, unsqueeze_1);  unsqueeze_1 = None
    full_default: "f32[1, 1, 1, 128]" = torch.ops.aten.full.default([1, 1, 1, 128], -0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:1006, code: attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
    full_2: "f32[1, 128]" = torch.ops.aten.full.default([1, 128], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:876, code: extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
    unsqueeze_10: "f32[1, 1, 128]" = torch.ops.aten.unsqueeze.default(full_2, 1);  full_2 = None
    unsqueeze_11: "f32[1, 1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 2);  unsqueeze_10 = None
    full_default_2: "f32[1, 1, 1, 128]" = torch.ops.aten.full.default([1, 1, 1, 128], 1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:1009, code: encoder_attention_mask = torch.ones(
    full_3: "i64[1, 128]" = torch.ops.aten.full.default([1, 128], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:840, code: encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
    unsqueeze_12: "i64[1, 1, 128]" = torch.ops.aten.unsqueeze.default(full_3, 1);  full_3 = None
    unsqueeze_13: "i64[1, 1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, 2);  unsqueeze_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:846, code: encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
    convert_element_type_4: "f32[1, 1, 1, 128]" = torch.ops.prims.convert_element_type.default(unsqueeze_13, torch.float32);  unsqueeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:847, code: encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(self.dtype).min
    sub_11: "f32[1, 1, 1, 128]" = torch.ops.aten.sub.Tensor(1.0, convert_element_type_4);  convert_element_type_4 = None
    mul_79: "f32[1, 1, 1, 128]" = torch.ops.aten.mul.Tensor(sub_11, -3.4028234663852886e+38);  sub_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:399, code: position_bias = torch.zeros(
    full_6: "f32[1, 6, 128, 128]" = torch.ops.aten.full.default([1, 6, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:413, code: position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)
    full_default_5: "f32[1, 6, 128, 128]" = torch.ops.aten.full.default([1, 6, 128, 128], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:1806, code: loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
    view_581: "i64[128]" = torch.ops.aten.reshape.default(arg191_1, [-1]);  arg191_1 = None
    ne_1: "b8[128]" = torch.ops.aten.ne.Scalar(view_581, -100)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:984, code: input_ids = input_ids.view(-1, input_shape[-1])
    view_209: "i64[1, 128]" = torch.ops.aten.reshape.default(arg192_1, [-1, 128]);  arg192_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:994, code: inputs_embeds = self.embed_tokens(input_ids)
    embedding_2: "f32[1, 128, 512]" = torch.ops.aten.embedding.default(arg42_1, view_209);  view_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_26: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(embedding_2, 2)
    mean_17: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_26, [-1], True);  pow_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_61: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_17, 1e-06);  mean_17 = None
    rsqrt_17: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
    mul_80: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(embedding_2, rsqrt_17);  rsqrt_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_81: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(arg17_1, mul_80);  arg17_1 = mul_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_210: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_81, [128, 512])
    permute_97: "f32[512, 384]" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
    mm_56: "f32[128, 384]" = torch.ops.aten.mm.default(view_210, permute_97);  view_210 = permute_97 = None
    view_211: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_56, [1, 128, 384]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_212: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_211, [1, -1, 6, 64]);  view_211 = None
    permute_98: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_32: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_98, [1, 6, 128, 64]);  permute_98 = None
    view_219: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_32, [6, 128, 64]);  expand_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_213: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_81, [128, 512])
    permute_99: "f32[512, 384]" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
    mm_57: "f32[128, 384]" = torch.ops.aten.mm.default(view_213, permute_99);  view_213 = permute_99 = None
    view_214: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_57, [1, 128, 384]);  mm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_215: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_214, [1, -1, 6, 64]);  view_214 = None
    permute_100: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_215, [0, 2, 1, 3]);  view_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_103: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_100, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_33: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_103, [1, 6, 64, 128]);  permute_103 = None
    view_220: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_33, [6, 64, 128]);  expand_33 = None
    bmm_16: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_219, view_220);  view_219 = view_220 = None
    view_221: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_16, [1, 6, 128, 128]);  bmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:303, code: memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
    iota_4: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_15: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(iota_4, 0);  iota_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:302, code: context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
    iota_3: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_14: "i64[128, 1]" = torch.ops.aten.unsqueeze.default(iota_3, 1);  iota_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:304, code: relative_position = memory_position - context_position  # shape (query_length, key_length)
    sub_12: "i64[128, 128]" = torch.ops.aten.sub.Tensor(unsqueeze_15, unsqueeze_14);  unsqueeze_15 = unsqueeze_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:278, code: relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
    full_default_3: "i64[128, 128]" = torch.ops.aten.full.default([128, 128], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    minimum_1: "i64[128, 128]" = torch.ops.aten.minimum.default(sub_12, full_default_3);  sub_12 = full_default_3 = None
    neg: "i64[128, 128]" = torch.ops.aten.neg.default(minimum_1);  minimum_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:283, code: is_small = relative_position < max_exact
    lt_1: "b8[128, 128]" = torch.ops.aten.lt.Scalar(neg, 16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:287, code: torch.log(relative_position.float() / max_exact)
    convert_element_type_5: "f32[128, 128]" = torch.ops.prims.convert_element_type.default(neg, torch.float32)
    div_10: "f32[128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_5, 16);  convert_element_type_5 = None
    log_1: "f32[128, 128]" = torch.ops.aten.log.default(div_10);  div_10 = None
    div_11: "f32[128, 128]" = torch.ops.aten.div.Tensor(log_1, 2.0794415416798357);  log_1 = None
    mul_82: "f32[128, 128]" = torch.ops.aten.mul.Tensor(div_11, 16);  div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:290, code: ).to(torch.long)
    convert_element_type_6: "i64[128, 128]" = torch.ops.prims.convert_element_type.default(mul_82, torch.int64);  mul_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:286, code: relative_position_if_large = max_exact + (
    add_62: "i64[128, 128]" = torch.ops.aten.add.Tensor(convert_element_type_6, 16);  convert_element_type_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:292, code: relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
    full_default_4: "i64[128, 128]" = torch.ops.aten.full.default([128, 128], 31, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:291, code: relative_position_if_large = torch.min(
    minimum_2: "i64[128, 128]" = torch.ops.aten.minimum.default(add_62, full_default_4);  add_62 = full_default_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:295, code: relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
    where_1: "i64[128, 128]" = torch.ops.aten.where.self(lt_1, neg, minimum_2);  lt_1 = neg = minimum_2 = None
    add_63: "i64[128, 128]" = torch.ops.aten.add.Tensor(where_1, 0);  where_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:311, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
    embedding_3: "f32[128, 128, 6]" = torch.ops.aten.embedding.default(arg103_1, add_63);  arg103_1 = add_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:312, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
    permute_104: "f32[6, 128, 128]" = torch.ops.aten.permute.default(embedding_3, [2, 0, 1]);  embedding_3 = None
    unsqueeze_16: "f32[1, 6, 128, 128]" = torch.ops.aten.unsqueeze.default(permute_104, 0);  permute_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:860, code: seq_ids = torch.arange(seq_length, device=device)
    iota_2: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:861, code: causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
    unsqueeze_5: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(iota_2, 0)
    unsqueeze_6: "i64[1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 1);  unsqueeze_5 = None
    repeat: "i64[1, 128, 128]" = torch.ops.aten.repeat.default(unsqueeze_6, [1, 128, 1]);  unsqueeze_6 = None
    unsqueeze_7: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(iota_2, 0);  iota_2 = None
    unsqueeze_8: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 2);  unsqueeze_7 = None
    le: "b8[1, 128, 128]" = torch.ops.aten.le.Tensor(repeat, unsqueeze_8);  repeat = unsqueeze_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:864, code: causal_mask = causal_mask.to(attention_mask.dtype)
    convert_element_type_3: "f32[1, 128, 128]" = torch.ops.prims.convert_element_type.default(le, torch.float32);  le = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:876, code: extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
    unsqueeze_9: "f32[1, 1, 128, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1);  convert_element_type_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:928, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    sub_10: "f32[1, 1, 128, 128]" = torch.ops.aten.sub.Tensor(1.0, unsqueeze_9);  unsqueeze_9 = None
    mul_78: "f32[1, 1, 128, 128]" = torch.ops.aten.mul.Tensor(sub_10, -3.4028234663852886e+38);  sub_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:413, code: position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)
    add_64: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(unsqueeze_16, mul_78);  unsqueeze_16 = mul_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_65: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_221, add_64);  view_221 = None
    view_222: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(add_65, [6, 128, 128]);  add_65 = None
    view_223: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_222, [1, 6, 128, 128]);  view_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_8: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_223, [-1], True)
    sub_13: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_223, amax_8);  view_223 = amax_8 = None
    exp_8: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
    sum_9: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_12: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_34: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(div_12, [1, 6, 128, 128]);  div_12 = None
    view_224: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_34, [6, 128, 128]);  expand_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_216: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_81, [128, 512]);  mul_81 = None
    permute_101: "f32[512, 384]" = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
    mm_58: "f32[128, 384]" = torch.ops.aten.mm.default(view_216, permute_101);  view_216 = permute_101 = None
    view_217: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_58, [1, 128, 384]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_218: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_217, [1, -1, 6, 64]);  view_217 = None
    permute_102: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_218, [0, 2, 1, 3]);  view_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_35: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_102, [1, 6, 128, 64])
    view_225: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_35, [6, 128, 64]);  expand_35 = None
    bmm_17: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_224, view_225);  view_224 = view_225 = None
    view_226: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_17, [1, 6, 128, 64]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_105: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
    clone_44: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_105, memory_format = torch.contiguous_format);  permute_105 = None
    view_227: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_44, [1, -1, 384]);  clone_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_228: "f32[128, 384]" = torch.ops.aten.reshape.default(view_227, [128, 384]);  view_227 = None
    permute_106: "f32[384, 512]" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
    mm_59: "f32[128, 512]" = torch.ops.aten.mm.default(view_228, permute_106);  view_228 = permute_106 = None
    view_229: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_59, [1, 128, 512]);  mm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    add_66: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(embedding_2, view_229);  embedding_2 = view_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_27: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_66, 2)
    mean_18: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_27, [-1], True);  pow_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_67: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_18, 1e-06);  mean_18 = None
    rsqrt_18: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
    mul_83: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_66, rsqrt_18);  rsqrt_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_84: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(arg18_1, mul_83);  arg18_1 = mul_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_230: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_84, [128, 512]);  mul_84 = None
    permute_107: "f32[512, 384]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
    mm_60: "f32[128, 384]" = torch.ops.aten.mm.default(view_230, permute_107);  view_230 = permute_107 = None
    view_231: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_60, [1, 128, 384]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_232: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_231, [1, -1, 6, 64]);  view_231 = None
    permute_108: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_232, [0, 2, 1, 3]);  view_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_36: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_108, [1, 6, 128, 64]);  permute_108 = None
    view_239: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_36, [6, 128, 64]);  expand_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:984, code: input_ids = input_ids.view(-1, input_shape[-1])
    view: "i64[1, 128]" = torch.ops.aten.reshape.default(arg190_1, [-1, 128]);  arg190_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:994, code: inputs_embeds = self.embed_tokens(input_ids)
    embedding: "f32[1, 128, 512]" = torch.ops.aten.embedding.default(arg42_1, view);  arg42_1 = view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_1: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(embedding, 2)
    mean: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
    rsqrt: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
    mul_1: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(embedding, rsqrt);  rsqrt = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_2: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(arg0_1, mul_1);  arg0_1 = mul_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_1: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_2, [128, 512])
    permute: "f32[512, 384]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
    mm: "f32[128, 384]" = torch.ops.aten.mm.default(view_1, permute);  view_1 = permute = None
    view_2: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm, [1, 128, 384]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_3: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_2, [1, -1, 6, 64]);  view_2 = None
    permute_1: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_1, [1, 6, 128, 64]);  permute_1 = None
    view_10: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand, [6, 128, 64]);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_4: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_2, [128, 512])
    permute_2: "f32[512, 384]" = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
    mm_1: "f32[128, 384]" = torch.ops.aten.mm.default(view_4, permute_2);  view_4 = permute_2 = None
    view_5: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_1, [1, 128, 384]);  mm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_6: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_5, [1, -1, 6, 64]);  view_5 = None
    permute_3: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_6: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_1: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_6, [1, 6, 64, 128]);  permute_6 = None
    view_11: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_1, [6, 64, 128]);  expand_1 = None
    bmm: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_10, view_11);  view_10 = view_11 = None
    view_12: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm, [1, 6, 128, 128]);  bmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:303, code: memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
    iota_1: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_3: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(iota_1, 0);  iota_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:302, code: context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
    iota: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_2: "i64[128, 1]" = torch.ops.aten.unsqueeze.default(iota, 1);  iota = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:304, code: relative_position = memory_position - context_position  # shape (query_length, key_length)
    sub_1: "i64[128, 128]" = torch.ops.aten.sub.Tensor(unsqueeze_3, unsqueeze_2);  unsqueeze_3 = unsqueeze_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:275, code: relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
    gt: "b8[128, 128]" = torch.ops.aten.gt.Scalar(sub_1, 0)
    convert_element_type: "i64[128, 128]" = torch.ops.prims.convert_element_type.default(gt, torch.int64);  gt = None
    mul_3: "i64[128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type, 16);  convert_element_type = None
    add_1: "i64[128, 128]" = torch.ops.aten.add.Tensor(mul_3, 0);  mul_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:276, code: relative_position = torch.abs(relative_position)
    abs_1: "i64[128, 128]" = torch.ops.aten.abs.default(sub_1);  sub_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:283, code: is_small = relative_position < max_exact
    lt: "b8[128, 128]" = torch.ops.aten.lt.Scalar(abs_1, 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:287, code: torch.log(relative_position.float() / max_exact)
    convert_element_type_1: "f32[128, 128]" = torch.ops.prims.convert_element_type.default(abs_1, torch.float32)
    div: "f32[128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_1, 8);  convert_element_type_1 = None
    log: "f32[128, 128]" = torch.ops.aten.log.default(div);  div = None
    div_1: "f32[128, 128]" = torch.ops.aten.div.Tensor(log, 2.772588722239781);  log = None
    mul_4: "f32[128, 128]" = torch.ops.aten.mul.Tensor(div_1, 8);  div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:290, code: ).to(torch.long)
    convert_element_type_2: "i64[128, 128]" = torch.ops.prims.convert_element_type.default(mul_4, torch.int64);  mul_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:286, code: relative_position_if_large = max_exact + (
    add_2: "i64[128, 128]" = torch.ops.aten.add.Tensor(convert_element_type_2, 8);  convert_element_type_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:292, code: relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
    full_default_1: "i64[128, 128]" = torch.ops.aten.full.default([128, 128], 15, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:291, code: relative_position_if_large = torch.min(
    minimum: "i64[128, 128]" = torch.ops.aten.minimum.default(add_2, full_default_1);  add_2 = full_default_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:295, code: relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
    where: "i64[128, 128]" = torch.ops.aten.where.self(lt, abs_1, minimum);  lt = abs_1 = minimum = None
    add_3: "i64[128, 128]" = torch.ops.aten.add.Tensor(add_1, where);  add_1 = where = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:311, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
    embedding_1: "f32[128, 128, 6]" = torch.ops.aten.embedding.default(arg46_1, add_3);  arg46_1 = add_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:312, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
    permute_7: "f32[6, 128, 128]" = torch.ops.aten.permute.default(embedding_1, [2, 0, 1]);  embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:413, code: position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)
    unsqueeze_4: "f32[1, 6, 128, 128]" = torch.ops.aten.unsqueeze.default(permute_7, 0);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_5: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_12, unsqueeze_4);  view_12 = None
    view_13: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(add_5, [6, 128, 128]);  add_5 = None
    view_14: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_13, [1, 6, 128, 128]);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_14, [-1], True)
    sub_2: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_14, amax);  view_14 = amax = None
    exp: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
    sum_1: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div_2: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_2: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(div_2, [1, 6, 128, 128]);  div_2 = None
    view_15: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_2, [6, 128, 128]);  expand_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_7: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_2, [128, 512]);  mul_2 = None
    permute_4: "f32[512, 384]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
    mm_2: "f32[128, 384]" = torch.ops.aten.mm.default(view_7, permute_4);  view_7 = permute_4 = None
    view_8: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_2, [1, 128, 384]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_9: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_8, [1, -1, 6, 64]);  view_8 = None
    permute_5: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_3: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_5, [1, 6, 128, 64]);  permute_5 = None
    view_16: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_3, [6, 128, 64]);  expand_3 = None
    bmm_1: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_15, view_16);  view_15 = view_16 = None
    view_17: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_1, [1, 6, 128, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_8: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_17, [0, 2, 1, 3]);  view_17 = None
    clone_2: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
    view_18: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_2, [1, -1, 384]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_19: "f32[128, 384]" = torch.ops.aten.reshape.default(view_18, [128, 384]);  view_18 = None
    permute_9: "f32[384, 512]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
    mm_3: "f32[128, 512]" = torch.ops.aten.mm.default(view_19, permute_9);  view_19 = permute_9 = None
    view_20: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_3, [1, 128, 512]);  mm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    add_6: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(embedding, view_20);  embedding = view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_2: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_6, 2)
    mean_1: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_7: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
    rsqrt_1: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
    mul_5: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_6, rsqrt_1);  rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_6: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(arg1_1, mul_5);  arg1_1 = mul_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_21: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_6, [128, 512])
    permute_10: "f32[512, 1024]" = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
    mm_4: "f32[128, 1024]" = torch.ops.aten.mm.default(view_21, permute_10);  view_21 = permute_10 = None
    view_22: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_4, [1, 128, 1024]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_7: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_22, 0.5)
    pow_3: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_22, 3.0)
    mul_8: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
    add_8: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_22, mul_8);  view_22 = mul_8 = None
    mul_9: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_8, 0.7978845608028654);  add_8 = None
    tanh: "f32[1, 128, 1024]" = torch.ops.aten.tanh.default(mul_9);  mul_9 = None
    add_9: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
    mul_10: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_7, add_9);  mul_7 = add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_23: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_6, [128, 512]);  mul_6 = None
    permute_11: "f32[512, 1024]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
    mm_5: "f32[128, 1024]" = torch.ops.aten.mm.default(view_23, permute_11);  view_23 = permute_11 = None
    view_24: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_5, [1, 128, 1024]);  mm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_11: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_10, view_24);  mul_10 = view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    view_25: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_11, [128, 1024]);  mul_11 = None
    permute_12: "f32[1024, 512]" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
    mm_6: "f32[128, 512]" = torch.ops.aten.mm.default(view_25, permute_12);  view_25 = permute_12 = None
    view_26: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_6, [1, 128, 512]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    add_10: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_6, view_26);  add_6 = view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_4: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_10, 2)
    mean_2: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_4, [-1], True);  pow_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_11: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_2, 1e-06);  mean_2 = None
    rsqrt_2: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    mul_12: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_10, rsqrt_2);  rsqrt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_13: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(arg2_1, mul_12);  arg2_1 = mul_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_27: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_13, [128, 512])
    permute_13: "f32[512, 384]" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
    mm_7: "f32[128, 384]" = torch.ops.aten.mm.default(view_27, permute_13);  view_27 = permute_13 = None
    view_28: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_7, [1, 128, 384]);  mm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_29: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_28, [1, -1, 6, 64]);  view_28 = None
    permute_14: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_4: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_14, [1, 6, 128, 64]);  permute_14 = None
    view_36: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_4, [6, 128, 64]);  expand_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_30: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_13, [128, 512])
    permute_15: "f32[512, 384]" = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
    mm_8: "f32[128, 384]" = torch.ops.aten.mm.default(view_30, permute_15);  view_30 = permute_15 = None
    view_31: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_8, [1, 128, 384]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_32: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_31, [1, -1, 6, 64]);  view_31 = None
    permute_16: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_19: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_16, [0, 1, 3, 2]);  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_5: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_19, [1, 6, 64, 128]);  permute_19 = None
    view_37: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_5, [6, 64, 128]);  expand_5 = None
    bmm_2: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_36, view_37);  view_36 = view_37 = None
    view_38: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_2, [1, 6, 128, 128]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_12: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_38, unsqueeze_4);  view_38 = None
    view_39: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(add_12, [6, 128, 128]);  add_12 = None
    view_40: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_39, [1, 6, 128, 128]);  view_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_1: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_40, [-1], True)
    sub_3: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_40, amax_1);  view_40 = amax_1 = None
    exp_1: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
    sum_2: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_3: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_6: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(div_3, [1, 6, 128, 128]);  div_3 = None
    view_41: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_6, [6, 128, 128]);  expand_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_33: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_13, [128, 512]);  mul_13 = None
    permute_17: "f32[512, 384]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
    mm_9: "f32[128, 384]" = torch.ops.aten.mm.default(view_33, permute_17);  view_33 = permute_17 = None
    view_34: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_9, [1, 128, 384]);  mm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_35: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_34, [1, -1, 6, 64]);  view_34 = None
    permute_18: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_35, [0, 2, 1, 3]);  view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_7: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_18, [1, 6, 128, 64]);  permute_18 = None
    view_42: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_7, [6, 128, 64]);  expand_7 = None
    bmm_3: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_41, view_42);  view_41 = view_42 = None
    view_43: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_3, [1, 6, 128, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_20: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_43, [0, 2, 1, 3]);  view_43 = None
    clone_7: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_20, memory_format = torch.contiguous_format);  permute_20 = None
    view_44: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_7, [1, -1, 384]);  clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_45: "f32[128, 384]" = torch.ops.aten.reshape.default(view_44, [128, 384]);  view_44 = None
    permute_21: "f32[384, 512]" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
    mm_10: "f32[128, 512]" = torch.ops.aten.mm.default(view_45, permute_21);  view_45 = permute_21 = None
    view_46: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_10, [1, 128, 512]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    add_13: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_10, view_46);  add_10 = view_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_5: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_13, 2)
    mean_3: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_5, [-1], True);  pow_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_14: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_3, 1e-06);  mean_3 = None
    rsqrt_3: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
    mul_14: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_13, rsqrt_3);  rsqrt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_15: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(arg3_1, mul_14);  arg3_1 = mul_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_47: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_15, [128, 512])
    permute_22: "f32[512, 1024]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
    mm_11: "f32[128, 1024]" = torch.ops.aten.mm.default(view_47, permute_22);  view_47 = permute_22 = None
    view_48: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_11, [1, 128, 1024]);  mm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_16: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_48, 0.5)
    pow_6: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_48, 3.0)
    mul_17: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
    add_15: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_48, mul_17);  view_48 = mul_17 = None
    mul_18: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_15, 0.7978845608028654);  add_15 = None
    tanh_1: "f32[1, 128, 1024]" = torch.ops.aten.tanh.default(mul_18);  mul_18 = None
    add_16: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
    mul_19: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_16, add_16);  mul_16 = add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_49: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_15, [128, 512]);  mul_15 = None
    permute_23: "f32[512, 1024]" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
    mm_12: "f32[128, 1024]" = torch.ops.aten.mm.default(view_49, permute_23);  view_49 = permute_23 = None
    view_50: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_12, [1, 128, 1024]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_20: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_19, view_50);  mul_19 = view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    view_51: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_20, [128, 1024]);  mul_20 = None
    permute_24: "f32[1024, 512]" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
    mm_13: "f32[128, 512]" = torch.ops.aten.mm.default(view_51, permute_24);  view_51 = permute_24 = None
    view_52: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_13, [1, 128, 512]);  mm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    add_17: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_13, view_52);  add_13 = view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_7: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_17, 2)
    mean_4: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_7, [-1], True);  pow_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_18: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_4, 1e-06);  mean_4 = None
    rsqrt_4: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    mul_21: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_17, rsqrt_4);  rsqrt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_22: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(arg4_1, mul_21);  arg4_1 = mul_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_53: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_22, [128, 512])
    permute_25: "f32[512, 384]" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
    mm_14: "f32[128, 384]" = torch.ops.aten.mm.default(view_53, permute_25);  view_53 = permute_25 = None
    view_54: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_14, [1, 128, 384]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_55: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_54, [1, -1, 6, 64]);  view_54 = None
    permute_26: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_55, [0, 2, 1, 3]);  view_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_8: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_26, [1, 6, 128, 64]);  permute_26 = None
    view_62: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_8, [6, 128, 64]);  expand_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_56: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_22, [128, 512])
    permute_27: "f32[512, 384]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
    mm_15: "f32[128, 384]" = torch.ops.aten.mm.default(view_56, permute_27);  view_56 = permute_27 = None
    view_57: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_15, [1, 128, 384]);  mm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_58: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_57, [1, -1, 6, 64]);  view_57 = None
    permute_28: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_31: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_28, [0, 1, 3, 2]);  permute_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_9: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_31, [1, 6, 64, 128]);  permute_31 = None
    view_63: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_9, [6, 64, 128]);  expand_9 = None
    bmm_4: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_62, view_63);  view_62 = view_63 = None
    view_64: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_4, [1, 6, 128, 128]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_19: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_64, unsqueeze_4);  view_64 = None
    view_65: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(add_19, [6, 128, 128]);  add_19 = None
    view_66: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_65, [1, 6, 128, 128]);  view_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_2: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_66, [-1], True)
    sub_4: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_66, amax_2);  view_66 = amax_2 = None
    exp_2: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
    sum_3: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_4: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_10: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(div_4, [1, 6, 128, 128]);  div_4 = None
    view_67: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_10, [6, 128, 128]);  expand_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_59: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_22, [128, 512]);  mul_22 = None
    permute_29: "f32[512, 384]" = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
    mm_16: "f32[128, 384]" = torch.ops.aten.mm.default(view_59, permute_29);  view_59 = permute_29 = None
    view_60: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_16, [1, 128, 384]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_61: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_60, [1, -1, 6, 64]);  view_60 = None
    permute_30: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_61, [0, 2, 1, 3]);  view_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_11: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_30, [1, 6, 128, 64]);  permute_30 = None
    view_68: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_11, [6, 128, 64]);  expand_11 = None
    bmm_5: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_67, view_68);  view_67 = view_68 = None
    view_69: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_5, [1, 6, 128, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_32: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_69, [0, 2, 1, 3]);  view_69 = None
    clone_12: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_32, memory_format = torch.contiguous_format);  permute_32 = None
    view_70: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_12, [1, -1, 384]);  clone_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_71: "f32[128, 384]" = torch.ops.aten.reshape.default(view_70, [128, 384]);  view_70 = None
    permute_33: "f32[384, 512]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
    mm_17: "f32[128, 512]" = torch.ops.aten.mm.default(view_71, permute_33);  view_71 = permute_33 = None
    view_72: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_17, [1, 128, 512]);  mm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    add_20: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_17, view_72);  add_17 = view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_8: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_20, 2)
    mean_5: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_8, [-1], True);  pow_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_21: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_5, 1e-06);  mean_5 = None
    rsqrt_5: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    mul_23: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_20, rsqrt_5);  rsqrt_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_24: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(arg5_1, mul_23);  arg5_1 = mul_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_73: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_24, [128, 512])
    permute_34: "f32[512, 1024]" = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
    mm_18: "f32[128, 1024]" = torch.ops.aten.mm.default(view_73, permute_34);  view_73 = permute_34 = None
    view_74: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_18, [1, 128, 1024]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_25: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_74, 0.5)
    pow_9: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_74, 3.0)
    mul_26: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
    add_22: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_74, mul_26);  view_74 = mul_26 = None
    mul_27: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_22, 0.7978845608028654);  add_22 = None
    tanh_2: "f32[1, 128, 1024]" = torch.ops.aten.tanh.default(mul_27);  mul_27 = None
    add_23: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
    mul_28: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_25, add_23);  mul_25 = add_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_75: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_24, [128, 512]);  mul_24 = None
    permute_35: "f32[512, 1024]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
    mm_19: "f32[128, 1024]" = torch.ops.aten.mm.default(view_75, permute_35);  view_75 = permute_35 = None
    view_76: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_19, [1, 128, 1024]);  mm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_29: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_28, view_76);  mul_28 = view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    view_77: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_29, [128, 1024]);  mul_29 = None
    permute_36: "f32[1024, 512]" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
    mm_20: "f32[128, 512]" = torch.ops.aten.mm.default(view_77, permute_36);  view_77 = permute_36 = None
    view_78: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_20, [1, 128, 512]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    add_24: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_20, view_78);  add_20 = view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_10: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_24, 2)
    mean_6: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_10, [-1], True);  pow_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_25: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_6, 1e-06);  mean_6 = None
    rsqrt_6: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
    mul_30: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_24, rsqrt_6);  rsqrt_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_31: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(arg6_1, mul_30);  arg6_1 = mul_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_79: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_31, [128, 512])
    permute_37: "f32[512, 384]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
    mm_21: "f32[128, 384]" = torch.ops.aten.mm.default(view_79, permute_37);  view_79 = permute_37 = None
    view_80: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_21, [1, 128, 384]);  mm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_81: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_80, [1, -1, 6, 64]);  view_80 = None
    permute_38: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_81, [0, 2, 1, 3]);  view_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_12: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_38, [1, 6, 128, 64]);  permute_38 = None
    view_88: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_12, [6, 128, 64]);  expand_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_82: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_31, [128, 512])
    permute_39: "f32[512, 384]" = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
    mm_22: "f32[128, 384]" = torch.ops.aten.mm.default(view_82, permute_39);  view_82 = permute_39 = None
    view_83: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_22, [1, 128, 384]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_84: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_83, [1, -1, 6, 64]);  view_83 = None
    permute_40: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_84, [0, 2, 1, 3]);  view_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_43: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_40, [0, 1, 3, 2]);  permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_13: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_43, [1, 6, 64, 128]);  permute_43 = None
    view_89: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_13, [6, 64, 128]);  expand_13 = None
    bmm_6: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_88, view_89);  view_88 = view_89 = None
    view_90: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_6, [1, 6, 128, 128]);  bmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_26: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_90, unsqueeze_4);  view_90 = None
    view_91: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(add_26, [6, 128, 128]);  add_26 = None
    view_92: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_91, [1, 6, 128, 128]);  view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_3: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_92, [-1], True)
    sub_5: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_92, amax_3);  view_92 = amax_3 = None
    exp_3: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
    sum_4: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_5: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_14: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(div_5, [1, 6, 128, 128]);  div_5 = None
    view_93: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_14, [6, 128, 128]);  expand_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_85: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_31, [128, 512]);  mul_31 = None
    permute_41: "f32[512, 384]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
    mm_23: "f32[128, 384]" = torch.ops.aten.mm.default(view_85, permute_41);  view_85 = permute_41 = None
    view_86: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_23, [1, 128, 384]);  mm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_87: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_86, [1, -1, 6, 64]);  view_86 = None
    permute_42: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_87, [0, 2, 1, 3]);  view_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_15: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_42, [1, 6, 128, 64]);  permute_42 = None
    view_94: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_15, [6, 128, 64]);  expand_15 = None
    bmm_7: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_93, view_94);  view_93 = view_94 = None
    view_95: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_7, [1, 6, 128, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_44: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None
    clone_17: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_44, memory_format = torch.contiguous_format);  permute_44 = None
    view_96: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_17, [1, -1, 384]);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_97: "f32[128, 384]" = torch.ops.aten.reshape.default(view_96, [128, 384]);  view_96 = None
    permute_45: "f32[384, 512]" = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
    mm_24: "f32[128, 512]" = torch.ops.aten.mm.default(view_97, permute_45);  view_97 = permute_45 = None
    view_98: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_24, [1, 128, 512]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    add_27: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_24, view_98);  add_24 = view_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_11: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_27, 2)
    mean_7: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_11, [-1], True);  pow_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_28: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_7, 1e-06);  mean_7 = None
    rsqrt_7: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    mul_32: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_27, rsqrt_7);  rsqrt_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_33: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(arg7_1, mul_32);  arg7_1 = mul_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_99: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_33, [128, 512])
    permute_46: "f32[512, 1024]" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
    mm_25: "f32[128, 1024]" = torch.ops.aten.mm.default(view_99, permute_46);  view_99 = permute_46 = None
    view_100: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_25, [1, 128, 1024]);  mm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_34: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_100, 0.5)
    pow_12: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_100, 3.0)
    mul_35: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
    add_29: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_100, mul_35);  view_100 = mul_35 = None
    mul_36: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_29, 0.7978845608028654);  add_29 = None
    tanh_3: "f32[1, 128, 1024]" = torch.ops.aten.tanh.default(mul_36);  mul_36 = None
    add_30: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
    mul_37: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_34, add_30);  mul_34 = add_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_101: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_33, [128, 512]);  mul_33 = None
    permute_47: "f32[512, 1024]" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
    mm_26: "f32[128, 1024]" = torch.ops.aten.mm.default(view_101, permute_47);  view_101 = permute_47 = None
    view_102: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_26, [1, 128, 1024]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_38: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_37, view_102);  mul_37 = view_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    view_103: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_38, [128, 1024]);  mul_38 = None
    permute_48: "f32[1024, 512]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
    mm_27: "f32[128, 512]" = torch.ops.aten.mm.default(view_103, permute_48);  view_103 = permute_48 = None
    view_104: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_27, [1, 128, 512]);  mm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    add_31: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_27, view_104);  add_27 = view_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_13: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_31, 2)
    mean_8: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_13, [-1], True);  pow_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_32: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_8, 1e-06);  mean_8 = None
    rsqrt_8: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    mul_39: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_31, rsqrt_8);  rsqrt_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_40: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(arg8_1, mul_39);  arg8_1 = mul_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_105: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_40, [128, 512])
    permute_49: "f32[512, 384]" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
    mm_28: "f32[128, 384]" = torch.ops.aten.mm.default(view_105, permute_49);  view_105 = permute_49 = None
    view_106: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_28, [1, 128, 384]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_107: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_106, [1, -1, 6, 64]);  view_106 = None
    permute_50: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_107, [0, 2, 1, 3]);  view_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_16: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_50, [1, 6, 128, 64]);  permute_50 = None
    view_114: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_16, [6, 128, 64]);  expand_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_108: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_40, [128, 512])
    permute_51: "f32[512, 384]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
    mm_29: "f32[128, 384]" = torch.ops.aten.mm.default(view_108, permute_51);  view_108 = permute_51 = None
    view_109: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_29, [1, 128, 384]);  mm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_110: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_109, [1, -1, 6, 64]);  view_109 = None
    permute_52: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_110, [0, 2, 1, 3]);  view_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_55: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_52, [0, 1, 3, 2]);  permute_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_17: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_55, [1, 6, 64, 128]);  permute_55 = None
    view_115: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_17, [6, 64, 128]);  expand_17 = None
    bmm_8: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_114, view_115);  view_114 = view_115 = None
    view_116: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_8, [1, 6, 128, 128]);  bmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_33: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_116, unsqueeze_4);  view_116 = None
    view_117: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(add_33, [6, 128, 128]);  add_33 = None
    view_118: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_117, [1, 6, 128, 128]);  view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_4: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_118, [-1], True)
    sub_6: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_118, amax_4);  view_118 = amax_4 = None
    exp_4: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_6);  sub_6 = None
    sum_5: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_6: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_18: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(div_6, [1, 6, 128, 128]);  div_6 = None
    view_119: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_18, [6, 128, 128]);  expand_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_111: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_40, [128, 512]);  mul_40 = None
    permute_53: "f32[512, 384]" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
    mm_30: "f32[128, 384]" = torch.ops.aten.mm.default(view_111, permute_53);  view_111 = permute_53 = None
    view_112: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_30, [1, 128, 384]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_113: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_112, [1, -1, 6, 64]);  view_112 = None
    permute_54: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_113, [0, 2, 1, 3]);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_19: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_54, [1, 6, 128, 64]);  permute_54 = None
    view_120: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_19, [6, 128, 64]);  expand_19 = None
    bmm_9: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_119, view_120);  view_119 = view_120 = None
    view_121: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_9, [1, 6, 128, 64]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_56: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_121, [0, 2, 1, 3]);  view_121 = None
    clone_22: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_56, memory_format = torch.contiguous_format);  permute_56 = None
    view_122: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_22, [1, -1, 384]);  clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_123: "f32[128, 384]" = torch.ops.aten.reshape.default(view_122, [128, 384]);  view_122 = None
    permute_57: "f32[384, 512]" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
    mm_31: "f32[128, 512]" = torch.ops.aten.mm.default(view_123, permute_57);  view_123 = permute_57 = None
    view_124: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_31, [1, 128, 512]);  mm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    add_34: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_31, view_124);  add_31 = view_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_14: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_34, 2)
    mean_9: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_14, [-1], True);  pow_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_35: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_9, 1e-06);  mean_9 = None
    rsqrt_9: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
    mul_41: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_34, rsqrt_9);  rsqrt_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_42: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(arg9_1, mul_41);  arg9_1 = mul_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_125: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_42, [128, 512])
    permute_58: "f32[512, 1024]" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
    mm_32: "f32[128, 1024]" = torch.ops.aten.mm.default(view_125, permute_58);  view_125 = permute_58 = None
    view_126: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_32, [1, 128, 1024]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_43: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_126, 0.5)
    pow_15: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_126, 3.0)
    mul_44: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_15, 0.044715);  pow_15 = None
    add_36: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_126, mul_44);  view_126 = mul_44 = None
    mul_45: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_36, 0.7978845608028654);  add_36 = None
    tanh_4: "f32[1, 128, 1024]" = torch.ops.aten.tanh.default(mul_45);  mul_45 = None
    add_37: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
    mul_46: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_43, add_37);  mul_43 = add_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_127: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_42, [128, 512]);  mul_42 = None
    permute_59: "f32[512, 1024]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
    mm_33: "f32[128, 1024]" = torch.ops.aten.mm.default(view_127, permute_59);  view_127 = permute_59 = None
    view_128: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_33, [1, 128, 1024]);  mm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_47: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_46, view_128);  mul_46 = view_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    view_129: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_47, [128, 1024]);  mul_47 = None
    permute_60: "f32[1024, 512]" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
    mm_34: "f32[128, 512]" = torch.ops.aten.mm.default(view_129, permute_60);  view_129 = permute_60 = None
    view_130: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_34, [1, 128, 512]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    add_38: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_34, view_130);  add_34 = view_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_16: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_38, 2)
    mean_10: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_16, [-1], True);  pow_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_39: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_10, 1e-06);  mean_10 = None
    rsqrt_10: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
    mul_48: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_38, rsqrt_10);  rsqrt_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_49: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(arg10_1, mul_48);  arg10_1 = mul_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_131: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_49, [128, 512])
    permute_61: "f32[512, 384]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
    mm_35: "f32[128, 384]" = torch.ops.aten.mm.default(view_131, permute_61);  view_131 = permute_61 = None
    view_132: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_35, [1, 128, 384]);  mm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_133: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_132, [1, -1, 6, 64]);  view_132 = None
    permute_62: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_133, [0, 2, 1, 3]);  view_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_20: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_62, [1, 6, 128, 64]);  permute_62 = None
    view_140: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_20, [6, 128, 64]);  expand_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_134: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_49, [128, 512])
    permute_63: "f32[512, 384]" = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
    mm_36: "f32[128, 384]" = torch.ops.aten.mm.default(view_134, permute_63);  view_134 = permute_63 = None
    view_135: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_36, [1, 128, 384]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_136: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_135, [1, -1, 6, 64]);  view_135 = None
    permute_64: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_67: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_64, [0, 1, 3, 2]);  permute_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_21: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_67, [1, 6, 64, 128]);  permute_67 = None
    view_141: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_21, [6, 64, 128]);  expand_21 = None
    bmm_10: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_140, view_141);  view_140 = view_141 = None
    view_142: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_10, [1, 6, 128, 128]);  bmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_40: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_142, unsqueeze_4);  view_142 = None
    view_143: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(add_40, [6, 128, 128]);  add_40 = None
    view_144: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_143, [1, 6, 128, 128]);  view_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_5: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_144, [-1], True)
    sub_7: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_144, amax_5);  view_144 = amax_5 = None
    exp_5: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
    sum_6: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_7: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_22: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(div_7, [1, 6, 128, 128]);  div_7 = None
    view_145: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_22, [6, 128, 128]);  expand_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_137: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_49, [128, 512]);  mul_49 = None
    permute_65: "f32[512, 384]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
    mm_37: "f32[128, 384]" = torch.ops.aten.mm.default(view_137, permute_65);  view_137 = permute_65 = None
    view_138: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_37, [1, 128, 384]);  mm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_139: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_138, [1, -1, 6, 64]);  view_138 = None
    permute_66: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_139, [0, 2, 1, 3]);  view_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_23: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_66, [1, 6, 128, 64]);  permute_66 = None
    view_146: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_23, [6, 128, 64]);  expand_23 = None
    bmm_11: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_145, view_146);  view_145 = view_146 = None
    view_147: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_11, [1, 6, 128, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_68: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_147, [0, 2, 1, 3]);  view_147 = None
    clone_27: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
    view_148: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_27, [1, -1, 384]);  clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_149: "f32[128, 384]" = torch.ops.aten.reshape.default(view_148, [128, 384]);  view_148 = None
    permute_69: "f32[384, 512]" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
    mm_38: "f32[128, 512]" = torch.ops.aten.mm.default(view_149, permute_69);  view_149 = permute_69 = None
    view_150: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_38, [1, 128, 512]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    add_41: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_38, view_150);  add_38 = view_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_17: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_41, 2)
    mean_11: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_17, [-1], True);  pow_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_42: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_11, 1e-06);  mean_11 = None
    rsqrt_11: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    mul_50: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_41, rsqrt_11);  rsqrt_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_51: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(arg11_1, mul_50);  arg11_1 = mul_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_151: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_51, [128, 512])
    permute_70: "f32[512, 1024]" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
    mm_39: "f32[128, 1024]" = torch.ops.aten.mm.default(view_151, permute_70);  view_151 = permute_70 = None
    view_152: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_39, [1, 128, 1024]);  mm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_52: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_152, 0.5)
    pow_18: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_152, 3.0)
    mul_53: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_18, 0.044715);  pow_18 = None
    add_43: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_152, mul_53);  view_152 = mul_53 = None
    mul_54: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_43, 0.7978845608028654);  add_43 = None
    tanh_5: "f32[1, 128, 1024]" = torch.ops.aten.tanh.default(mul_54);  mul_54 = None
    add_44: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
    mul_55: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_52, add_44);  mul_52 = add_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_153: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_51, [128, 512]);  mul_51 = None
    permute_71: "f32[512, 1024]" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
    mm_40: "f32[128, 1024]" = torch.ops.aten.mm.default(view_153, permute_71);  view_153 = permute_71 = None
    view_154: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_40, [1, 128, 1024]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_56: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_55, view_154);  mul_55 = view_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    view_155: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_56, [128, 1024]);  mul_56 = None
    permute_72: "f32[1024, 512]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
    mm_41: "f32[128, 512]" = torch.ops.aten.mm.default(view_155, permute_72);  view_155 = permute_72 = None
    view_156: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_41, [1, 128, 512]);  mm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    add_45: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_41, view_156);  add_41 = view_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_19: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_45, 2)
    mean_12: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_19, [-1], True);  pow_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_46: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_12, 1e-06);  mean_12 = None
    rsqrt_12: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
    mul_57: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_45, rsqrt_12);  rsqrt_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_58: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(arg12_1, mul_57);  arg12_1 = mul_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_157: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_58, [128, 512])
    permute_73: "f32[512, 384]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
    mm_42: "f32[128, 384]" = torch.ops.aten.mm.default(view_157, permute_73);  view_157 = permute_73 = None
    view_158: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_42, [1, 128, 384]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_159: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_158, [1, -1, 6, 64]);  view_158 = None
    permute_74: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_159, [0, 2, 1, 3]);  view_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_24: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_74, [1, 6, 128, 64]);  permute_74 = None
    view_166: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_24, [6, 128, 64]);  expand_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_160: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_58, [128, 512])
    permute_75: "f32[512, 384]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
    mm_43: "f32[128, 384]" = torch.ops.aten.mm.default(view_160, permute_75);  view_160 = permute_75 = None
    view_161: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_43, [1, 128, 384]);  mm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_162: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_161, [1, -1, 6, 64]);  view_161 = None
    permute_76: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_79: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_76, [0, 1, 3, 2]);  permute_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_25: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_79, [1, 6, 64, 128]);  permute_79 = None
    view_167: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_25, [6, 64, 128]);  expand_25 = None
    bmm_12: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_166, view_167);  view_166 = view_167 = None
    view_168: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_12, [1, 6, 128, 128]);  bmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_47: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_168, unsqueeze_4);  view_168 = None
    view_169: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(add_47, [6, 128, 128]);  add_47 = None
    view_170: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_169, [1, 6, 128, 128]);  view_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_6: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_170, [-1], True)
    sub_8: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_170, amax_6);  view_170 = amax_6 = None
    exp_6: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_8);  sub_8 = None
    sum_7: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_8: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_26: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(div_8, [1, 6, 128, 128]);  div_8 = None
    view_171: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_26, [6, 128, 128]);  expand_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_163: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_58, [128, 512]);  mul_58 = None
    permute_77: "f32[512, 384]" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
    mm_44: "f32[128, 384]" = torch.ops.aten.mm.default(view_163, permute_77);  view_163 = permute_77 = None
    view_164: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_44, [1, 128, 384]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_165: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_164, [1, -1, 6, 64]);  view_164 = None
    permute_78: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_165, [0, 2, 1, 3]);  view_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_27: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_78, [1, 6, 128, 64]);  permute_78 = None
    view_172: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_27, [6, 128, 64]);  expand_27 = None
    bmm_13: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_171, view_172);  view_171 = view_172 = None
    view_173: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_13, [1, 6, 128, 64]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_80: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_173, [0, 2, 1, 3]);  view_173 = None
    clone_32: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format);  permute_80 = None
    view_174: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_32, [1, -1, 384]);  clone_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_175: "f32[128, 384]" = torch.ops.aten.reshape.default(view_174, [128, 384]);  view_174 = None
    permute_81: "f32[384, 512]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
    mm_45: "f32[128, 512]" = torch.ops.aten.mm.default(view_175, permute_81);  view_175 = permute_81 = None
    view_176: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_45, [1, 128, 512]);  mm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    add_48: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_45, view_176);  add_45 = view_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_20: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_48, 2)
    mean_13: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_20, [-1], True);  pow_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_49: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_13, 1e-06);  mean_13 = None
    rsqrt_13: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    mul_59: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_48, rsqrt_13);  rsqrt_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_60: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(arg13_1, mul_59);  arg13_1 = mul_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_177: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_60, [128, 512])
    permute_82: "f32[512, 1024]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
    mm_46: "f32[128, 1024]" = torch.ops.aten.mm.default(view_177, permute_82);  view_177 = permute_82 = None
    view_178: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_46, [1, 128, 1024]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_61: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_178, 0.5)
    pow_21: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_178, 3.0)
    mul_62: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_21, 0.044715);  pow_21 = None
    add_50: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_178, mul_62);  view_178 = mul_62 = None
    mul_63: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_50, 0.7978845608028654);  add_50 = None
    tanh_6: "f32[1, 128, 1024]" = torch.ops.aten.tanh.default(mul_63);  mul_63 = None
    add_51: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
    mul_64: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_61, add_51);  mul_61 = add_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_179: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_60, [128, 512]);  mul_60 = None
    permute_83: "f32[512, 1024]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
    mm_47: "f32[128, 1024]" = torch.ops.aten.mm.default(view_179, permute_83);  view_179 = permute_83 = None
    view_180: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_47, [1, 128, 1024]);  mm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_65: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_64, view_180);  mul_64 = view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    view_181: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_65, [128, 1024]);  mul_65 = None
    permute_84: "f32[1024, 512]" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
    mm_48: "f32[128, 512]" = torch.ops.aten.mm.default(view_181, permute_84);  view_181 = permute_84 = None
    view_182: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_48, [1, 128, 512]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    add_52: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_48, view_182);  add_48 = view_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_22: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_52, 2)
    mean_14: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_22, [-1], True);  pow_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_53: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_14, 1e-06);  mean_14 = None
    rsqrt_14: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    mul_66: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_52, rsqrt_14);  rsqrt_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_67: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(arg14_1, mul_66);  arg14_1 = mul_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_183: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_67, [128, 512])
    permute_85: "f32[512, 384]" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
    mm_49: "f32[128, 384]" = torch.ops.aten.mm.default(view_183, permute_85);  view_183 = permute_85 = None
    view_184: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_49, [1, 128, 384]);  mm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_185: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_184, [1, -1, 6, 64]);  view_184 = None
    permute_86: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_28: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_86, [1, 6, 128, 64]);  permute_86 = None
    view_192: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_28, [6, 128, 64]);  expand_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_186: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_67, [128, 512])
    permute_87: "f32[512, 384]" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
    mm_50: "f32[128, 384]" = torch.ops.aten.mm.default(view_186, permute_87);  view_186 = permute_87 = None
    view_187: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_50, [1, 128, 384]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_188: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_187, [1, -1, 6, 64]);  view_187 = None
    permute_88: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_188, [0, 2, 1, 3]);  view_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_91: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_88, [0, 1, 3, 2]);  permute_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_29: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_91, [1, 6, 64, 128]);  permute_91 = None
    view_193: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_29, [6, 64, 128]);  expand_29 = None
    bmm_14: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_192, view_193);  view_192 = view_193 = None
    view_194: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_14, [1, 6, 128, 128]);  bmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_54: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_194, unsqueeze_4);  view_194 = unsqueeze_4 = None
    view_195: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(add_54, [6, 128, 128]);  add_54 = None
    view_196: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_195, [1, 6, 128, 128]);  view_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_7: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_196, [-1], True)
    sub_9: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_196, amax_7);  view_196 = amax_7 = None
    exp_7: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
    sum_8: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_9: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_30: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(div_9, [1, 6, 128, 128]);  div_9 = None
    view_197: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_30, [6, 128, 128]);  expand_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_189: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_67, [128, 512]);  mul_67 = None
    permute_89: "f32[512, 384]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
    mm_51: "f32[128, 384]" = torch.ops.aten.mm.default(view_189, permute_89);  view_189 = permute_89 = None
    view_190: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_51, [1, 128, 384]);  mm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_191: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_190, [1, -1, 6, 64]);  view_190 = None
    permute_90: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_191, [0, 2, 1, 3]);  view_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_31: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_90, [1, 6, 128, 64]);  permute_90 = None
    view_198: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_31, [6, 128, 64]);  expand_31 = None
    bmm_15: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_197, view_198);  view_197 = view_198 = None
    view_199: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_15, [1, 6, 128, 64]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_92: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_199, [0, 2, 1, 3]);  view_199 = None
    clone_37: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
    view_200: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_37, [1, -1, 384]);  clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_201: "f32[128, 384]" = torch.ops.aten.reshape.default(view_200, [128, 384]);  view_200 = None
    permute_93: "f32[384, 512]" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
    mm_52: "f32[128, 512]" = torch.ops.aten.mm.default(view_201, permute_93);  view_201 = permute_93 = None
    view_202: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_52, [1, 128, 512]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    add_55: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_52, view_202);  add_52 = view_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_23: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_55, 2)
    mean_15: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_23, [-1], True);  pow_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_56: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_15, 1e-06);  mean_15 = None
    rsqrt_15: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    mul_68: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_55, rsqrt_15);  rsqrt_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_69: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(arg15_1, mul_68);  arg15_1 = mul_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_203: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_69, [128, 512])
    permute_94: "f32[512, 1024]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
    mm_53: "f32[128, 1024]" = torch.ops.aten.mm.default(view_203, permute_94);  view_203 = permute_94 = None
    view_204: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_53, [1, 128, 1024]);  mm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_70: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_204, 0.5)
    pow_24: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_204, 3.0)
    mul_71: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_24, 0.044715);  pow_24 = None
    add_57: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_204, mul_71);  view_204 = mul_71 = None
    mul_72: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_57, 0.7978845608028654);  add_57 = None
    tanh_7: "f32[1, 128, 1024]" = torch.ops.aten.tanh.default(mul_72);  mul_72 = None
    add_58: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
    mul_73: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_70, add_58);  mul_70 = add_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_205: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_69, [128, 512]);  mul_69 = None
    permute_95: "f32[512, 1024]" = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
    mm_54: "f32[128, 1024]" = torch.ops.aten.mm.default(view_205, permute_95);  view_205 = permute_95 = None
    view_206: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_54, [1, 128, 1024]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_74: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_73, view_206);  mul_73 = view_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    view_207: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_74, [128, 1024]);  mul_74 = None
    permute_96: "f32[1024, 512]" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
    mm_55: "f32[128, 512]" = torch.ops.aten.mm.default(view_207, permute_96);  view_207 = permute_96 = None
    view_208: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_55, [1, 128, 512]);  mm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    add_59: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_55, view_208);  add_55 = view_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_25: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_59, 2)
    mean_16: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_25, [-1], True);  pow_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_60: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_16, 1e-06);  mean_16 = None
    rsqrt_16: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    mul_75: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_59, rsqrt_16);  add_59 = rsqrt_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_76: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(arg16_1, mul_75);  arg16_1 = mul_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    view_233: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_76, [128, 512])
    permute_109: "f32[512, 384]" = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
    mm_61: "f32[128, 384]" = torch.ops.aten.mm.default(view_233, permute_109);  view_233 = permute_109 = None
    view_234: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_61, [1, 128, 384]);  mm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_235: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_234, [1, -1, 6, 64]);  view_234 = None
    permute_110: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_235, [0, 2, 1, 3]);  view_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_113: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_110, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_37: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_113, [1, 6, 64, 128]);  permute_113 = None
    view_240: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_37, [6, 64, 128]);  expand_37 = None
    bmm_18: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_239, view_240);  view_239 = view_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    view_241: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_18, [1, 6, 128, 128]);  bmm_18 = None
    view_242: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(view_241, [6, 128, 128]);  view_241 = None
    view_243: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_242, [1, 6, 128, 128]);  view_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_9: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_243, [-1], True)
    sub_14: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_243, amax_9);  view_243 = amax_9 = None
    exp_9: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
    sum_10: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_13: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_38: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(div_13, [1, 6, 128, 128]);  div_13 = None
    view_244: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_38, [6, 128, 128]);  expand_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    view_236: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_76, [128, 512])
    permute_111: "f32[512, 384]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
    mm_62: "f32[128, 384]" = torch.ops.aten.mm.default(view_236, permute_111);  view_236 = permute_111 = None
    view_237: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_62, [1, 128, 384]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_238: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_237, [1, -1, 6, 64]);  view_237 = None
    permute_112: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_238, [0, 2, 1, 3]);  view_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_39: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_112, [1, 6, 128, 64])
    view_245: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_39, [6, 128, 64]);  expand_39 = None
    bmm_19: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_244, view_245);  view_244 = view_245 = None
    view_246: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_19, [1, 6, 128, 64]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_114: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None
    clone_47: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
    view_247: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_47, [1, -1, 384]);  clone_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_248: "f32[128, 384]" = torch.ops.aten.reshape.default(view_247, [128, 384]);  view_247 = None
    permute_115: "f32[384, 512]" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
    mm_63: "f32[128, 512]" = torch.ops.aten.mm.default(view_248, permute_115);  view_248 = permute_115 = None
    view_249: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_63, [1, 128, 512]);  mm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:510, code: layer_output = hidden_states + self.dropout(attention_output[0])
    add_70: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_66, view_249);  add_66 = view_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_28: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_70, 2)
    mean_19: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_28, [-1], True);  pow_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_71: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_19, 1e-06);  mean_19 = None
    rsqrt_19: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
    mul_85: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_70, rsqrt_19);  rsqrt_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_86: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(arg19_1, mul_85);  arg19_1 = mul_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_250: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_86, [128, 512])
    permute_116: "f32[512, 1024]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
    mm_64: "f32[128, 1024]" = torch.ops.aten.mm.default(view_250, permute_116);  view_250 = permute_116 = None
    view_251: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_64, [1, 128, 1024]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_87: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_251, 0.5)
    pow_29: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_251, 3.0)
    mul_88: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_29, 0.044715);  pow_29 = None
    add_72: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_251, mul_88);  view_251 = mul_88 = None
    mul_89: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_72, 0.7978845608028654);  add_72 = None
    tanh_8: "f32[1, 128, 1024]" = torch.ops.aten.tanh.default(mul_89);  mul_89 = None
    add_73: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
    mul_90: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_87, add_73);  mul_87 = add_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_252: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_86, [128, 512]);  mul_86 = None
    permute_117: "f32[512, 1024]" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
    mm_65: "f32[128, 1024]" = torch.ops.aten.mm.default(view_252, permute_117);  view_252 = permute_117 = None
    view_253: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_65, [1, 128, 1024]);  mm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_91: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_90, view_253);  mul_90 = view_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    view_254: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_91, [128, 1024]);  mul_91 = None
    permute_118: "f32[1024, 512]" = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
    mm_66: "f32[128, 512]" = torch.ops.aten.mm.default(view_254, permute_118);  view_254 = permute_118 = None
    view_255: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_66, [1, 128, 512]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    add_74: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_70, view_255);  add_70 = view_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_30: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_74, 2)
    mean_20: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_30, [-1], True);  pow_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_75: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_20, 1e-06);  mean_20 = None
    rsqrt_20: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
    mul_92: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_74, rsqrt_20);  rsqrt_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_93: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(arg20_1, mul_92);  arg20_1 = mul_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_256: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_93, [128, 512])
    permute_119: "f32[512, 384]" = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
    mm_67: "f32[128, 384]" = torch.ops.aten.mm.default(view_256, permute_119);  view_256 = permute_119 = None
    view_257: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_67, [1, 128, 384]);  mm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_258: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_257, [1, -1, 6, 64]);  view_257 = None
    permute_120: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_258, [0, 2, 1, 3]);  view_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_40: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_120, [1, 6, 128, 64]);  permute_120 = None
    view_265: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_40, [6, 128, 64]);  expand_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_259: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_93, [128, 512])
    permute_121: "f32[512, 384]" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
    mm_68: "f32[128, 384]" = torch.ops.aten.mm.default(view_259, permute_121);  view_259 = permute_121 = None
    view_260: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_68, [1, 128, 384]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_261: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_260, [1, -1, 6, 64]);  view_260 = None
    permute_122: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_261, [0, 2, 1, 3]);  view_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_125: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_122, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_41: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_125, [1, 6, 64, 128]);  permute_125 = None
    view_266: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_41, [6, 64, 128]);  expand_41 = None
    bmm_20: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_265, view_266);  view_265 = view_266 = None
    view_267: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_20, [1, 6, 128, 128]);  bmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_76: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_267, add_64);  view_267 = None
    view_268: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(add_76, [6, 128, 128]);  add_76 = None
    view_269: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_268, [1, 6, 128, 128]);  view_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_10: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_269, [-1], True)
    sub_15: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_269, amax_10);  view_269 = amax_10 = None
    exp_10: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
    sum_11: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_14: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_42: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(div_14, [1, 6, 128, 128]);  div_14 = None
    view_270: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_42, [6, 128, 128]);  expand_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_262: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_93, [128, 512]);  mul_93 = None
    permute_123: "f32[512, 384]" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
    mm_69: "f32[128, 384]" = torch.ops.aten.mm.default(view_262, permute_123);  view_262 = permute_123 = None
    view_263: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_69, [1, 128, 384]);  mm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_264: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_263, [1, -1, 6, 64]);  view_263 = None
    permute_124: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_264, [0, 2, 1, 3]);  view_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_43: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_124, [1, 6, 128, 64])
    view_271: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_43, [6, 128, 64]);  expand_43 = None
    bmm_21: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_270, view_271);  view_270 = view_271 = None
    view_272: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_21, [1, 6, 128, 64]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_126: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_272, [0, 2, 1, 3]);  view_272 = None
    clone_52: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
    view_273: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_52, [1, -1, 384]);  clone_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_274: "f32[128, 384]" = torch.ops.aten.reshape.default(view_273, [128, 384]);  view_273 = None
    permute_127: "f32[384, 512]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
    mm_70: "f32[128, 512]" = torch.ops.aten.mm.default(view_274, permute_127);  view_274 = permute_127 = None
    view_275: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_70, [1, 128, 512]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    add_77: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_74, view_275);  add_74 = view_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_31: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_77, 2)
    mean_21: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_31, [-1], True);  pow_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_78: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_21, 1e-06);  mean_21 = None
    rsqrt_21: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    mul_94: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_77, rsqrt_21);  rsqrt_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_95: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(arg21_1, mul_94);  arg21_1 = mul_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_276: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_95, [128, 512]);  mul_95 = None
    permute_128: "f32[512, 384]" = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
    mm_71: "f32[128, 384]" = torch.ops.aten.mm.default(view_276, permute_128);  view_276 = permute_128 = None
    view_277: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_71, [1, 128, 384]);  mm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_278: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_277, [1, -1, 6, 64]);  view_277 = None
    permute_129: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_278, [0, 2, 1, 3]);  view_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_44: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_129, [1, 6, 128, 64]);  permute_129 = None
    view_285: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_44, [6, 128, 64]);  expand_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    view_279: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_76, [128, 512])
    permute_130: "f32[512, 384]" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
    mm_72: "f32[128, 384]" = torch.ops.aten.mm.default(view_279, permute_130);  view_279 = permute_130 = None
    view_280: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_72, [1, 128, 384]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_281: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_280, [1, -1, 6, 64]);  view_280 = None
    permute_131: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_281, [0, 2, 1, 3]);  view_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_134: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_131, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_45: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_134, [1, 6, 64, 128]);  permute_134 = None
    view_286: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_45, [6, 64, 128]);  expand_45 = None
    bmm_22: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_285, view_286);  view_285 = view_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    view_287: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_22, [1, 6, 128, 128]);  bmm_22 = None
    view_288: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(view_287, [6, 128, 128]);  view_287 = None
    view_289: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_288, [1, 6, 128, 128]);  view_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_11: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_289, [-1], True)
    sub_16: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_289, amax_11);  view_289 = amax_11 = None
    exp_11: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
    sum_12: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_15: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_46: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(div_15, [1, 6, 128, 128]);  div_15 = None
    view_290: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_46, [6, 128, 128]);  expand_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    view_282: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_76, [128, 512])
    permute_132: "f32[512, 384]" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
    mm_73: "f32[128, 384]" = torch.ops.aten.mm.default(view_282, permute_132);  view_282 = permute_132 = None
    view_283: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_73, [1, 128, 384]);  mm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_284: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_283, [1, -1, 6, 64]);  view_283 = None
    permute_133: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_284, [0, 2, 1, 3]);  view_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_47: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_133, [1, 6, 128, 64])
    view_291: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_47, [6, 128, 64]);  expand_47 = None
    bmm_23: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_290, view_291);  view_290 = view_291 = None
    view_292: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_23, [1, 6, 128, 64]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_135: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_292, [0, 2, 1, 3]);  view_292 = None
    clone_55: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_135, memory_format = torch.contiguous_format);  permute_135 = None
    view_293: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_55, [1, -1, 384]);  clone_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_294: "f32[128, 384]" = torch.ops.aten.reshape.default(view_293, [128, 384]);  view_293 = None
    permute_136: "f32[384, 512]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
    mm_74: "f32[128, 512]" = torch.ops.aten.mm.default(view_294, permute_136);  view_294 = permute_136 = None
    view_295: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_74, [1, 128, 512]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:510, code: layer_output = hidden_states + self.dropout(attention_output[0])
    add_80: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_77, view_295);  add_77 = view_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_32: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_80, 2)
    mean_22: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_32, [-1], True);  pow_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_81: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_22, 1e-06);  mean_22 = None
    rsqrt_22: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
    mul_96: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_80, rsqrt_22);  rsqrt_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_97: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(arg22_1, mul_96);  arg22_1 = mul_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_296: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_97, [128, 512])
    permute_137: "f32[512, 1024]" = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
    mm_75: "f32[128, 1024]" = torch.ops.aten.mm.default(view_296, permute_137);  view_296 = permute_137 = None
    view_297: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_75, [1, 128, 1024]);  mm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_98: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_297, 0.5)
    pow_33: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_297, 3.0)
    mul_99: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_33, 0.044715);  pow_33 = None
    add_82: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_297, mul_99);  view_297 = mul_99 = None
    mul_100: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_82, 0.7978845608028654);  add_82 = None
    tanh_9: "f32[1, 128, 1024]" = torch.ops.aten.tanh.default(mul_100);  mul_100 = None
    add_83: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
    mul_101: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_98, add_83);  mul_98 = add_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_298: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_97, [128, 512]);  mul_97 = None
    permute_138: "f32[512, 1024]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
    mm_76: "f32[128, 1024]" = torch.ops.aten.mm.default(view_298, permute_138);  view_298 = permute_138 = None
    view_299: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_76, [1, 128, 1024]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_102: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_101, view_299);  mul_101 = view_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    view_300: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_102, [128, 1024]);  mul_102 = None
    permute_139: "f32[1024, 512]" = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
    mm_77: "f32[128, 512]" = torch.ops.aten.mm.default(view_300, permute_139);  view_300 = permute_139 = None
    view_301: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_77, [1, 128, 512]);  mm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    add_84: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_80, view_301);  add_80 = view_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_34: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_84, 2)
    mean_23: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_34, [-1], True);  pow_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_85: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_23, 1e-06);  mean_23 = None
    rsqrt_23: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
    mul_103: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_84, rsqrt_23);  rsqrt_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_104: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(arg23_1, mul_103);  arg23_1 = mul_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_302: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_104, [128, 512])
    permute_140: "f32[512, 384]" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
    mm_78: "f32[128, 384]" = torch.ops.aten.mm.default(view_302, permute_140);  view_302 = permute_140 = None
    view_303: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_78, [1, 128, 384]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_304: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_303, [1, -1, 6, 64]);  view_303 = None
    permute_141: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_304, [0, 2, 1, 3]);  view_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_48: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_141, [1, 6, 128, 64]);  permute_141 = None
    view_311: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_48, [6, 128, 64]);  expand_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_305: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_104, [128, 512])
    permute_142: "f32[512, 384]" = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
    mm_79: "f32[128, 384]" = torch.ops.aten.mm.default(view_305, permute_142);  view_305 = permute_142 = None
    view_306: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_79, [1, 128, 384]);  mm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_307: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_306, [1, -1, 6, 64]);  view_306 = None
    permute_143: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_307, [0, 2, 1, 3]);  view_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_146: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_143, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_49: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_146, [1, 6, 64, 128]);  permute_146 = None
    view_312: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_49, [6, 64, 128]);  expand_49 = None
    bmm_24: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_311, view_312);  view_311 = view_312 = None
    view_313: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_24, [1, 6, 128, 128]);  bmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_86: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_313, add_64);  view_313 = None
    view_314: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(add_86, [6, 128, 128]);  add_86 = None
    view_315: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_314, [1, 6, 128, 128]);  view_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_12: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_315, [-1], True)
    sub_17: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_315, amax_12);  view_315 = amax_12 = None
    exp_12: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
    sum_13: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
    div_16: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_50: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(div_16, [1, 6, 128, 128]);  div_16 = None
    view_316: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_50, [6, 128, 128]);  expand_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_308: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_104, [128, 512]);  mul_104 = None
    permute_144: "f32[512, 384]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
    mm_80: "f32[128, 384]" = torch.ops.aten.mm.default(view_308, permute_144);  view_308 = permute_144 = None
    view_309: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_80, [1, 128, 384]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_310: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_309, [1, -1, 6, 64]);  view_309 = None
    permute_145: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_310, [0, 2, 1, 3]);  view_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_51: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_145, [1, 6, 128, 64])
    view_317: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_51, [6, 128, 64]);  expand_51 = None
    bmm_25: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_316, view_317);  view_316 = view_317 = None
    view_318: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_25, [1, 6, 128, 64]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_147: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_318, [0, 2, 1, 3]);  view_318 = None
    clone_60: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_147, memory_format = torch.contiguous_format);  permute_147 = None
    view_319: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_60, [1, -1, 384]);  clone_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_320: "f32[128, 384]" = torch.ops.aten.reshape.default(view_319, [128, 384]);  view_319 = None
    permute_148: "f32[384, 512]" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
    mm_81: "f32[128, 512]" = torch.ops.aten.mm.default(view_320, permute_148);  view_320 = permute_148 = None
    view_321: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_81, [1, 128, 512]);  mm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    add_87: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_84, view_321);  add_84 = view_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_35: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_87, 2)
    mean_24: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_35, [-1], True);  pow_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_88: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_24, 1e-06);  mean_24 = None
    rsqrt_24: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
    mul_105: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_87, rsqrt_24);  rsqrt_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_106: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(arg24_1, mul_105);  arg24_1 = mul_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_322: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_106, [128, 512]);  mul_106 = None
    permute_149: "f32[512, 384]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
    mm_82: "f32[128, 384]" = torch.ops.aten.mm.default(view_322, permute_149);  view_322 = permute_149 = None
    view_323: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_82, [1, 128, 384]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_324: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_323, [1, -1, 6, 64]);  view_323 = None
    permute_150: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_324, [0, 2, 1, 3]);  view_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_52: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_150, [1, 6, 128, 64]);  permute_150 = None
    view_331: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_52, [6, 128, 64]);  expand_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    view_325: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_76, [128, 512])
    permute_151: "f32[512, 384]" = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
    mm_83: "f32[128, 384]" = torch.ops.aten.mm.default(view_325, permute_151);  view_325 = permute_151 = None
    view_326: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_83, [1, 128, 384]);  mm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_327: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_326, [1, -1, 6, 64]);  view_326 = None
    permute_152: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_327, [0, 2, 1, 3]);  view_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_155: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_152, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_53: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_155, [1, 6, 64, 128]);  permute_155 = None
    view_332: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_53, [6, 64, 128]);  expand_53 = None
    bmm_26: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_331, view_332);  view_331 = view_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    view_333: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_26, [1, 6, 128, 128]);  bmm_26 = None
    view_334: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(view_333, [6, 128, 128]);  view_333 = None
    view_335: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_334, [1, 6, 128, 128]);  view_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_13: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_335, [-1], True)
    sub_18: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_335, amax_13);  view_335 = amax_13 = None
    exp_13: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_18);  sub_18 = None
    sum_14: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
    div_17: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_54: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(div_17, [1, 6, 128, 128]);  div_17 = None
    view_336: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_54, [6, 128, 128]);  expand_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    view_328: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_76, [128, 512])
    permute_153: "f32[512, 384]" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
    mm_84: "f32[128, 384]" = torch.ops.aten.mm.default(view_328, permute_153);  view_328 = permute_153 = None
    view_329: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_84, [1, 128, 384]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_330: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_329, [1, -1, 6, 64]);  view_329 = None
    permute_154: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_330, [0, 2, 1, 3]);  view_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_55: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_154, [1, 6, 128, 64])
    view_337: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_55, [6, 128, 64]);  expand_55 = None
    bmm_27: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_336, view_337);  view_336 = view_337 = None
    view_338: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_27, [1, 6, 128, 64]);  bmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_156: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_338, [0, 2, 1, 3]);  view_338 = None
    clone_63: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_156, memory_format = torch.contiguous_format);  permute_156 = None
    view_339: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_63, [1, -1, 384]);  clone_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_340: "f32[128, 384]" = torch.ops.aten.reshape.default(view_339, [128, 384]);  view_339 = None
    permute_157: "f32[384, 512]" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
    mm_85: "f32[128, 512]" = torch.ops.aten.mm.default(view_340, permute_157);  view_340 = permute_157 = None
    view_341: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_85, [1, 128, 512]);  mm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:510, code: layer_output = hidden_states + self.dropout(attention_output[0])
    add_90: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_87, view_341);  add_87 = view_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_36: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_90, 2)
    mean_25: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_36, [-1], True);  pow_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_91: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_25, 1e-06);  mean_25 = None
    rsqrt_25: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
    mul_107: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_90, rsqrt_25);  rsqrt_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_108: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(arg25_1, mul_107);  arg25_1 = mul_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_342: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_108, [128, 512])
    permute_158: "f32[512, 1024]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
    mm_86: "f32[128, 1024]" = torch.ops.aten.mm.default(view_342, permute_158);  view_342 = permute_158 = None
    view_343: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_86, [1, 128, 1024]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_109: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_343, 0.5)
    pow_37: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_343, 3.0)
    mul_110: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_37, 0.044715);  pow_37 = None
    add_92: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_343, mul_110);  view_343 = mul_110 = None
    mul_111: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_92, 0.7978845608028654);  add_92 = None
    tanh_10: "f32[1, 128, 1024]" = torch.ops.aten.tanh.default(mul_111);  mul_111 = None
    add_93: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
    mul_112: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_109, add_93);  mul_109 = add_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_344: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_108, [128, 512]);  mul_108 = None
    permute_159: "f32[512, 1024]" = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
    mm_87: "f32[128, 1024]" = torch.ops.aten.mm.default(view_344, permute_159);  view_344 = permute_159 = None
    view_345: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_87, [1, 128, 1024]);  mm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_113: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_112, view_345);  mul_112 = view_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    view_346: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_113, [128, 1024]);  mul_113 = None
    permute_160: "f32[1024, 512]" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
    mm_88: "f32[128, 512]" = torch.ops.aten.mm.default(view_346, permute_160);  view_346 = permute_160 = None
    view_347: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_88, [1, 128, 512]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    add_94: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_90, view_347);  add_90 = view_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_38: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_94, 2)
    mean_26: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_38, [-1], True);  pow_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_95: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_26, 1e-06);  mean_26 = None
    rsqrt_26: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
    mul_114: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_94, rsqrt_26);  rsqrt_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_115: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(arg26_1, mul_114);  arg26_1 = mul_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_348: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_115, [128, 512])
    permute_161: "f32[512, 384]" = torch.ops.aten.permute.default(arg134_1, [1, 0]);  arg134_1 = None
    mm_89: "f32[128, 384]" = torch.ops.aten.mm.default(view_348, permute_161);  view_348 = permute_161 = None
    view_349: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_89, [1, 128, 384]);  mm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_350: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_349, [1, -1, 6, 64]);  view_349 = None
    permute_162: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_350, [0, 2, 1, 3]);  view_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_56: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_162, [1, 6, 128, 64]);  permute_162 = None
    view_357: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_56, [6, 128, 64]);  expand_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_351: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_115, [128, 512])
    permute_163: "f32[512, 384]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
    mm_90: "f32[128, 384]" = torch.ops.aten.mm.default(view_351, permute_163);  view_351 = permute_163 = None
    view_352: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_90, [1, 128, 384]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_353: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_352, [1, -1, 6, 64]);  view_352 = None
    permute_164: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_353, [0, 2, 1, 3]);  view_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_167: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_164, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_57: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_167, [1, 6, 64, 128]);  permute_167 = None
    view_358: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_57, [6, 64, 128]);  expand_57 = None
    bmm_28: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_357, view_358);  view_357 = view_358 = None
    view_359: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_28, [1, 6, 128, 128]);  bmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_96: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_359, add_64);  view_359 = None
    view_360: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(add_96, [6, 128, 128]);  add_96 = None
    view_361: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_360, [1, 6, 128, 128]);  view_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_14: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_361, [-1], True)
    sub_19: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_361, amax_14);  view_361 = amax_14 = None
    exp_14: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
    sum_15: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
    div_18: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_58: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(div_18, [1, 6, 128, 128]);  div_18 = None
    view_362: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_58, [6, 128, 128]);  expand_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_354: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_115, [128, 512]);  mul_115 = None
    permute_165: "f32[512, 384]" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
    mm_91: "f32[128, 384]" = torch.ops.aten.mm.default(view_354, permute_165);  view_354 = permute_165 = None
    view_355: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_91, [1, 128, 384]);  mm_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_356: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_355, [1, -1, 6, 64]);  view_355 = None
    permute_166: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_356, [0, 2, 1, 3]);  view_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_59: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_166, [1, 6, 128, 64])
    view_363: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_59, [6, 128, 64]);  expand_59 = None
    bmm_29: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_362, view_363);  view_362 = view_363 = None
    view_364: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_29, [1, 6, 128, 64]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_168: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_364, [0, 2, 1, 3]);  view_364 = None
    clone_68: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_168, memory_format = torch.contiguous_format);  permute_168 = None
    view_365: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_68, [1, -1, 384]);  clone_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_366: "f32[128, 384]" = torch.ops.aten.reshape.default(view_365, [128, 384]);  view_365 = None
    permute_169: "f32[384, 512]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
    mm_92: "f32[128, 512]" = torch.ops.aten.mm.default(view_366, permute_169);  view_366 = permute_169 = None
    view_367: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_92, [1, 128, 512]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    add_97: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_94, view_367);  add_94 = view_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_39: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_97, 2)
    mean_27: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_39, [-1], True);  pow_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_98: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_27, 1e-06);  mean_27 = None
    rsqrt_27: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
    mul_116: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_97, rsqrt_27);  rsqrt_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_117: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(arg27_1, mul_116);  arg27_1 = mul_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_368: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_117, [128, 512]);  mul_117 = None
    permute_170: "f32[512, 384]" = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
    mm_93: "f32[128, 384]" = torch.ops.aten.mm.default(view_368, permute_170);  view_368 = permute_170 = None
    view_369: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_93, [1, 128, 384]);  mm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_370: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_369, [1, -1, 6, 64]);  view_369 = None
    permute_171: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_370, [0, 2, 1, 3]);  view_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_60: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_171, [1, 6, 128, 64]);  permute_171 = None
    view_377: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_60, [6, 128, 64]);  expand_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    view_371: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_76, [128, 512])
    permute_172: "f32[512, 384]" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
    mm_94: "f32[128, 384]" = torch.ops.aten.mm.default(view_371, permute_172);  view_371 = permute_172 = None
    view_372: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_94, [1, 128, 384]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_373: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_372, [1, -1, 6, 64]);  view_372 = None
    permute_173: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_373, [0, 2, 1, 3]);  view_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_176: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_173, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_61: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_176, [1, 6, 64, 128]);  permute_176 = None
    view_378: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_61, [6, 64, 128]);  expand_61 = None
    bmm_30: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_377, view_378);  view_377 = view_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    view_379: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_30, [1, 6, 128, 128]);  bmm_30 = None
    view_380: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(view_379, [6, 128, 128]);  view_379 = None
    view_381: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_380, [1, 6, 128, 128]);  view_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_15: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_381, [-1], True)
    sub_20: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_381, amax_15);  view_381 = amax_15 = None
    exp_15: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
    sum_16: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
    div_19: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_62: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(div_19, [1, 6, 128, 128]);  div_19 = None
    view_382: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_62, [6, 128, 128]);  expand_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    view_374: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_76, [128, 512])
    permute_174: "f32[512, 384]" = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
    mm_95: "f32[128, 384]" = torch.ops.aten.mm.default(view_374, permute_174);  view_374 = permute_174 = None
    view_375: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_95, [1, 128, 384]);  mm_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_376: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_375, [1, -1, 6, 64]);  view_375 = None
    permute_175: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_376, [0, 2, 1, 3]);  view_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_63: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_175, [1, 6, 128, 64])
    view_383: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_63, [6, 128, 64]);  expand_63 = None
    bmm_31: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_382, view_383);  view_382 = view_383 = None
    view_384: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_31, [1, 6, 128, 64]);  bmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_177: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_384, [0, 2, 1, 3]);  view_384 = None
    clone_71: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None
    view_385: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_71, [1, -1, 384]);  clone_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_386: "f32[128, 384]" = torch.ops.aten.reshape.default(view_385, [128, 384]);  view_385 = None
    permute_178: "f32[384, 512]" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
    mm_96: "f32[128, 512]" = torch.ops.aten.mm.default(view_386, permute_178);  view_386 = permute_178 = None
    view_387: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_96, [1, 128, 512]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:510, code: layer_output = hidden_states + self.dropout(attention_output[0])
    add_100: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_97, view_387);  add_97 = view_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_40: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_100, 2)
    mean_28: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_40, [-1], True);  pow_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_101: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_28, 1e-06);  mean_28 = None
    rsqrt_28: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
    mul_118: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_100, rsqrt_28);  rsqrt_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_119: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(arg28_1, mul_118);  arg28_1 = mul_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_388: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_119, [128, 512])
    permute_179: "f32[512, 1024]" = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
    mm_97: "f32[128, 1024]" = torch.ops.aten.mm.default(view_388, permute_179);  view_388 = permute_179 = None
    view_389: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_97, [1, 128, 1024]);  mm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_120: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_389, 0.5)
    pow_41: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_389, 3.0)
    mul_121: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_41, 0.044715);  pow_41 = None
    add_102: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_389, mul_121);  view_389 = mul_121 = None
    mul_122: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_102, 0.7978845608028654);  add_102 = None
    tanh_11: "f32[1, 128, 1024]" = torch.ops.aten.tanh.default(mul_122);  mul_122 = None
    add_103: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
    mul_123: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_120, add_103);  mul_120 = add_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_390: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_119, [128, 512]);  mul_119 = None
    permute_180: "f32[512, 1024]" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
    mm_98: "f32[128, 1024]" = torch.ops.aten.mm.default(view_390, permute_180);  view_390 = permute_180 = None
    view_391: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_98, [1, 128, 1024]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_124: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_123, view_391);  mul_123 = view_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    view_392: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_124, [128, 1024]);  mul_124 = None
    permute_181: "f32[1024, 512]" = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
    mm_99: "f32[128, 512]" = torch.ops.aten.mm.default(view_392, permute_181);  view_392 = permute_181 = None
    view_393: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_99, [1, 128, 512]);  mm_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    add_104: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_100, view_393);  add_100 = view_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_42: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_104, 2)
    mean_29: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_42, [-1], True);  pow_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_105: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_29, 1e-06);  mean_29 = None
    rsqrt_29: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
    mul_125: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_104, rsqrt_29);  rsqrt_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_126: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(arg29_1, mul_125);  arg29_1 = mul_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_394: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_126, [128, 512])
    permute_182: "f32[512, 384]" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
    mm_100: "f32[128, 384]" = torch.ops.aten.mm.default(view_394, permute_182);  view_394 = permute_182 = None
    view_395: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_100, [1, 128, 384]);  mm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_396: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_395, [1, -1, 6, 64]);  view_395 = None
    permute_183: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_396, [0, 2, 1, 3]);  view_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_64: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_183, [1, 6, 128, 64]);  permute_183 = None
    view_403: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_64, [6, 128, 64]);  expand_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_397: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_126, [128, 512])
    permute_184: "f32[512, 384]" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
    mm_101: "f32[128, 384]" = torch.ops.aten.mm.default(view_397, permute_184);  view_397 = permute_184 = None
    view_398: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_101, [1, 128, 384]);  mm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_399: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_398, [1, -1, 6, 64]);  view_398 = None
    permute_185: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_399, [0, 2, 1, 3]);  view_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_188: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_185, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_65: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_188, [1, 6, 64, 128]);  permute_188 = None
    view_404: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_65, [6, 64, 128]);  expand_65 = None
    bmm_32: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_403, view_404);  view_403 = view_404 = None
    view_405: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_32, [1, 6, 128, 128]);  bmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_106: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_405, add_64);  view_405 = None
    view_406: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(add_106, [6, 128, 128]);  add_106 = None
    view_407: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_406, [1, 6, 128, 128]);  view_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_16: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_407, [-1], True)
    sub_21: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_407, amax_16);  view_407 = amax_16 = None
    exp_16: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_21);  sub_21 = None
    sum_17: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
    div_20: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_66: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(div_20, [1, 6, 128, 128]);  div_20 = None
    view_408: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_66, [6, 128, 128]);  expand_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_400: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_126, [128, 512]);  mul_126 = None
    permute_186: "f32[512, 384]" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
    mm_102: "f32[128, 384]" = torch.ops.aten.mm.default(view_400, permute_186);  view_400 = permute_186 = None
    view_401: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_102, [1, 128, 384]);  mm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_402: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_401, [1, -1, 6, 64]);  view_401 = None
    permute_187: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_402, [0, 2, 1, 3]);  view_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_67: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_187, [1, 6, 128, 64])
    view_409: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_67, [6, 128, 64]);  expand_67 = None
    bmm_33: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_408, view_409);  view_408 = view_409 = None
    view_410: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_33, [1, 6, 128, 64]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_189: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_410, [0, 2, 1, 3]);  view_410 = None
    clone_76: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None
    view_411: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_76, [1, -1, 384]);  clone_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_412: "f32[128, 384]" = torch.ops.aten.reshape.default(view_411, [128, 384]);  view_411 = None
    permute_190: "f32[384, 512]" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
    mm_103: "f32[128, 512]" = torch.ops.aten.mm.default(view_412, permute_190);  view_412 = permute_190 = None
    view_413: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_103, [1, 128, 512]);  mm_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    add_107: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_104, view_413);  add_104 = view_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_43: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_107, 2)
    mean_30: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_43, [-1], True);  pow_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_108: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_30, 1e-06);  mean_30 = None
    rsqrt_30: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
    mul_127: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_107, rsqrt_30);  rsqrt_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_128: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(arg30_1, mul_127);  arg30_1 = mul_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_414: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_128, [128, 512]);  mul_128 = None
    permute_191: "f32[512, 384]" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
    mm_104: "f32[128, 384]" = torch.ops.aten.mm.default(view_414, permute_191);  view_414 = permute_191 = None
    view_415: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_104, [1, 128, 384]);  mm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_416: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_415, [1, -1, 6, 64]);  view_415 = None
    permute_192: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_416, [0, 2, 1, 3]);  view_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_68: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_192, [1, 6, 128, 64]);  permute_192 = None
    view_423: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_68, [6, 128, 64]);  expand_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    view_417: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_76, [128, 512])
    permute_193: "f32[512, 384]" = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
    mm_105: "f32[128, 384]" = torch.ops.aten.mm.default(view_417, permute_193);  view_417 = permute_193 = None
    view_418: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_105, [1, 128, 384]);  mm_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_419: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_418, [1, -1, 6, 64]);  view_418 = None
    permute_194: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_419, [0, 2, 1, 3]);  view_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_197: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_194, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_69: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_197, [1, 6, 64, 128]);  permute_197 = None
    view_424: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_69, [6, 64, 128]);  expand_69 = None
    bmm_34: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_423, view_424);  view_423 = view_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    view_425: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_34, [1, 6, 128, 128]);  bmm_34 = None
    view_426: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(view_425, [6, 128, 128]);  view_425 = None
    view_427: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_426, [1, 6, 128, 128]);  view_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_17: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_427, [-1], True)
    sub_22: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_427, amax_17);  view_427 = amax_17 = None
    exp_17: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
    sum_18: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
    div_21: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_70: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(div_21, [1, 6, 128, 128]);  div_21 = None
    view_428: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_70, [6, 128, 128]);  expand_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    view_420: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_76, [128, 512])
    permute_195: "f32[512, 384]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
    mm_106: "f32[128, 384]" = torch.ops.aten.mm.default(view_420, permute_195);  view_420 = permute_195 = None
    view_421: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_106, [1, 128, 384]);  mm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_422: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_421, [1, -1, 6, 64]);  view_421 = None
    permute_196: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_422, [0, 2, 1, 3]);  view_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_71: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_196, [1, 6, 128, 64])
    view_429: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_71, [6, 128, 64]);  expand_71 = None
    bmm_35: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_428, view_429);  view_428 = view_429 = None
    view_430: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_35, [1, 6, 128, 64]);  bmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_198: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_430, [0, 2, 1, 3]);  view_430 = None
    clone_79: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_198, memory_format = torch.contiguous_format);  permute_198 = None
    view_431: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_79, [1, -1, 384]);  clone_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_432: "f32[128, 384]" = torch.ops.aten.reshape.default(view_431, [128, 384]);  view_431 = None
    permute_199: "f32[384, 512]" = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
    mm_107: "f32[128, 512]" = torch.ops.aten.mm.default(view_432, permute_199);  view_432 = permute_199 = None
    view_433: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_107, [1, 128, 512]);  mm_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:510, code: layer_output = hidden_states + self.dropout(attention_output[0])
    add_110: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_107, view_433);  add_107 = view_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_44: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_110, 2)
    mean_31: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_44, [-1], True);  pow_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_111: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_31, 1e-06);  mean_31 = None
    rsqrt_31: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
    mul_129: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_110, rsqrt_31);  rsqrt_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_130: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(arg31_1, mul_129);  arg31_1 = mul_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_434: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_130, [128, 512])
    permute_200: "f32[512, 1024]" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
    mm_108: "f32[128, 1024]" = torch.ops.aten.mm.default(view_434, permute_200);  view_434 = permute_200 = None
    view_435: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_108, [1, 128, 1024]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_131: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_435, 0.5)
    pow_45: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_435, 3.0)
    mul_132: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_45, 0.044715);  pow_45 = None
    add_112: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_435, mul_132);  view_435 = mul_132 = None
    mul_133: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_112, 0.7978845608028654);  add_112 = None
    tanh_12: "f32[1, 128, 1024]" = torch.ops.aten.tanh.default(mul_133);  mul_133 = None
    add_113: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_12, 1.0);  tanh_12 = None
    mul_134: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_131, add_113);  mul_131 = add_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_436: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_130, [128, 512]);  mul_130 = None
    permute_201: "f32[512, 1024]" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
    mm_109: "f32[128, 1024]" = torch.ops.aten.mm.default(view_436, permute_201);  view_436 = permute_201 = None
    view_437: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_109, [1, 128, 1024]);  mm_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_135: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_134, view_437);  mul_134 = view_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    view_438: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_135, [128, 1024]);  mul_135 = None
    permute_202: "f32[1024, 512]" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
    mm_110: "f32[128, 512]" = torch.ops.aten.mm.default(view_438, permute_202);  view_438 = permute_202 = None
    view_439: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_110, [1, 128, 512]);  mm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    add_114: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_110, view_439);  add_110 = view_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_46: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_114, 2)
    mean_32: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_46, [-1], True);  pow_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_115: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_32, 1e-06);  mean_32 = None
    rsqrt_32: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
    mul_136: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_114, rsqrt_32);  rsqrt_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_137: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(arg32_1, mul_136);  arg32_1 = mul_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_440: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_137, [128, 512])
    permute_203: "f32[512, 384]" = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
    mm_111: "f32[128, 384]" = torch.ops.aten.mm.default(view_440, permute_203);  view_440 = permute_203 = None
    view_441: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_111, [1, 128, 384]);  mm_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_442: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_441, [1, -1, 6, 64]);  view_441 = None
    permute_204: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_442, [0, 2, 1, 3]);  view_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_72: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_204, [1, 6, 128, 64]);  permute_204 = None
    view_449: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_72, [6, 128, 64]);  expand_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_443: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_137, [128, 512])
    permute_205: "f32[512, 384]" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
    mm_112: "f32[128, 384]" = torch.ops.aten.mm.default(view_443, permute_205);  view_443 = permute_205 = None
    view_444: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_112, [1, 128, 384]);  mm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_445: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_444, [1, -1, 6, 64]);  view_444 = None
    permute_206: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_445, [0, 2, 1, 3]);  view_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_209: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_206, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_73: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_209, [1, 6, 64, 128]);  permute_209 = None
    view_450: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_73, [6, 64, 128]);  expand_73 = None
    bmm_36: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_449, view_450);  view_449 = view_450 = None
    view_451: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_36, [1, 6, 128, 128]);  bmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_116: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_451, add_64);  view_451 = None
    view_452: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(add_116, [6, 128, 128]);  add_116 = None
    view_453: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_452, [1, 6, 128, 128]);  view_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_18: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_453, [-1], True)
    sub_23: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_453, amax_18);  view_453 = amax_18 = None
    exp_18: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_23);  sub_23 = None
    sum_19: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
    div_22: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_74: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(div_22, [1, 6, 128, 128]);  div_22 = None
    view_454: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_74, [6, 128, 128]);  expand_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_446: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_137, [128, 512]);  mul_137 = None
    permute_207: "f32[512, 384]" = torch.ops.aten.permute.default(arg158_1, [1, 0]);  arg158_1 = None
    mm_113: "f32[128, 384]" = torch.ops.aten.mm.default(view_446, permute_207);  view_446 = permute_207 = None
    view_447: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_113, [1, 128, 384]);  mm_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_448: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_447, [1, -1, 6, 64]);  view_447 = None
    permute_208: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_448, [0, 2, 1, 3]);  view_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_75: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_208, [1, 6, 128, 64])
    view_455: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_75, [6, 128, 64]);  expand_75 = None
    bmm_37: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_454, view_455);  view_454 = view_455 = None
    view_456: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_37, [1, 6, 128, 64]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_210: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_456, [0, 2, 1, 3]);  view_456 = None
    clone_84: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_210, memory_format = torch.contiguous_format);  permute_210 = None
    view_457: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_84, [1, -1, 384]);  clone_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_458: "f32[128, 384]" = torch.ops.aten.reshape.default(view_457, [128, 384]);  view_457 = None
    permute_211: "f32[384, 512]" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
    mm_114: "f32[128, 512]" = torch.ops.aten.mm.default(view_458, permute_211);  view_458 = permute_211 = None
    view_459: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_114, [1, 128, 512]);  mm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    add_117: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_114, view_459);  add_114 = view_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_47: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_117, 2)
    mean_33: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_47, [-1], True);  pow_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_118: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_33, 1e-06);  mean_33 = None
    rsqrt_33: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
    mul_138: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_117, rsqrt_33);  rsqrt_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_139: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(arg33_1, mul_138);  arg33_1 = mul_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_460: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_139, [128, 512]);  mul_139 = None
    permute_212: "f32[512, 384]" = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
    mm_115: "f32[128, 384]" = torch.ops.aten.mm.default(view_460, permute_212);  view_460 = permute_212 = None
    view_461: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_115, [1, 128, 384]);  mm_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_462: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_461, [1, -1, 6, 64]);  view_461 = None
    permute_213: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_462, [0, 2, 1, 3]);  view_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_76: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_213, [1, 6, 128, 64]);  permute_213 = None
    view_469: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_76, [6, 128, 64]);  expand_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    view_463: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_76, [128, 512])
    permute_214: "f32[512, 384]" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
    mm_116: "f32[128, 384]" = torch.ops.aten.mm.default(view_463, permute_214);  view_463 = permute_214 = None
    view_464: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_116, [1, 128, 384]);  mm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_465: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_464, [1, -1, 6, 64]);  view_464 = None
    permute_215: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_465, [0, 2, 1, 3]);  view_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_218: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_215, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_77: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_218, [1, 6, 64, 128]);  permute_218 = None
    view_470: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_77, [6, 64, 128]);  expand_77 = None
    bmm_38: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_469, view_470);  view_469 = view_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    view_471: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_38, [1, 6, 128, 128]);  bmm_38 = None
    view_472: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(view_471, [6, 128, 128]);  view_471 = None
    view_473: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_472, [1, 6, 128, 128]);  view_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_19: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_473, [-1], True)
    sub_24: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_473, amax_19);  view_473 = amax_19 = None
    exp_19: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_24);  sub_24 = None
    sum_20: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
    div_23: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_78: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(div_23, [1, 6, 128, 128]);  div_23 = None
    view_474: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_78, [6, 128, 128]);  expand_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    view_466: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_76, [128, 512])
    permute_216: "f32[512, 384]" = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
    mm_117: "f32[128, 384]" = torch.ops.aten.mm.default(view_466, permute_216);  view_466 = permute_216 = None
    view_467: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_117, [1, 128, 384]);  mm_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_468: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_467, [1, -1, 6, 64]);  view_467 = None
    permute_217: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_468, [0, 2, 1, 3]);  view_468 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_79: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_217, [1, 6, 128, 64])
    view_475: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_79, [6, 128, 64]);  expand_79 = None
    bmm_39: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_474, view_475);  view_474 = view_475 = None
    view_476: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_39, [1, 6, 128, 64]);  bmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_219: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_476, [0, 2, 1, 3]);  view_476 = None
    clone_87: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_219, memory_format = torch.contiguous_format);  permute_219 = None
    view_477: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_87, [1, -1, 384]);  clone_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_478: "f32[128, 384]" = torch.ops.aten.reshape.default(view_477, [128, 384]);  view_477 = None
    permute_220: "f32[384, 512]" = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
    mm_118: "f32[128, 512]" = torch.ops.aten.mm.default(view_478, permute_220);  view_478 = permute_220 = None
    view_479: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_118, [1, 128, 512]);  mm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:510, code: layer_output = hidden_states + self.dropout(attention_output[0])
    add_120: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_117, view_479);  add_117 = view_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_48: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_120, 2)
    mean_34: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_48, [-1], True);  pow_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_121: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_34, 1e-06);  mean_34 = None
    rsqrt_34: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_121);  add_121 = None
    mul_140: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_120, rsqrt_34);  rsqrt_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_141: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(arg34_1, mul_140);  arg34_1 = mul_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_480: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_141, [128, 512])
    permute_221: "f32[512, 1024]" = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
    mm_119: "f32[128, 1024]" = torch.ops.aten.mm.default(view_480, permute_221);  view_480 = permute_221 = None
    view_481: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_119, [1, 128, 1024]);  mm_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_142: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_481, 0.5)
    pow_49: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_481, 3.0)
    mul_143: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_49, 0.044715);  pow_49 = None
    add_122: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_481, mul_143);  view_481 = mul_143 = None
    mul_144: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_122, 0.7978845608028654);  add_122 = None
    tanh_13: "f32[1, 128, 1024]" = torch.ops.aten.tanh.default(mul_144);  mul_144 = None
    add_123: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_13, 1.0);  tanh_13 = None
    mul_145: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_142, add_123);  mul_142 = add_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_482: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_141, [128, 512]);  mul_141 = None
    permute_222: "f32[512, 1024]" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
    mm_120: "f32[128, 1024]" = torch.ops.aten.mm.default(view_482, permute_222);  view_482 = permute_222 = None
    view_483: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_120, [1, 128, 1024]);  mm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_146: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_145, view_483);  mul_145 = view_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    view_484: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_146, [128, 1024]);  mul_146 = None
    permute_223: "f32[1024, 512]" = torch.ops.aten.permute.default(arg166_1, [1, 0]);  arg166_1 = None
    mm_121: "f32[128, 512]" = torch.ops.aten.mm.default(view_484, permute_223);  view_484 = permute_223 = None
    view_485: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_121, [1, 128, 512]);  mm_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    add_124: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_120, view_485);  add_120 = view_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_50: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_124, 2)
    mean_35: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_50, [-1], True);  pow_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_125: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_35, 1e-06);  mean_35 = None
    rsqrt_35: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_125);  add_125 = None
    mul_147: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_124, rsqrt_35);  rsqrt_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_148: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(arg35_1, mul_147);  arg35_1 = mul_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_486: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_148, [128, 512])
    permute_224: "f32[512, 384]" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
    mm_122: "f32[128, 384]" = torch.ops.aten.mm.default(view_486, permute_224);  view_486 = permute_224 = None
    view_487: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_122, [1, 128, 384]);  mm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_488: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_487, [1, -1, 6, 64]);  view_487 = None
    permute_225: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_488, [0, 2, 1, 3]);  view_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_80: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_225, [1, 6, 128, 64]);  permute_225 = None
    view_495: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_80, [6, 128, 64]);  expand_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_489: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_148, [128, 512])
    permute_226: "f32[512, 384]" = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
    mm_123: "f32[128, 384]" = torch.ops.aten.mm.default(view_489, permute_226);  view_489 = permute_226 = None
    view_490: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_123, [1, 128, 384]);  mm_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_491: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_490, [1, -1, 6, 64]);  view_490 = None
    permute_227: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_491, [0, 2, 1, 3]);  view_491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_230: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_227, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_81: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_230, [1, 6, 64, 128]);  permute_230 = None
    view_496: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_81, [6, 64, 128]);  expand_81 = None
    bmm_40: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_495, view_496);  view_495 = view_496 = None
    view_497: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_40, [1, 6, 128, 128]);  bmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_126: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_497, add_64);  view_497 = None
    view_498: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(add_126, [6, 128, 128]);  add_126 = None
    view_499: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_498, [1, 6, 128, 128]);  view_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_20: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_499, [-1], True)
    sub_25: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_499, amax_20);  view_499 = amax_20 = None
    exp_20: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
    sum_21: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
    div_24: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_82: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(div_24, [1, 6, 128, 128]);  div_24 = None
    view_500: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_82, [6, 128, 128]);  expand_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_492: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_148, [128, 512]);  mul_148 = None
    permute_228: "f32[512, 384]" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
    mm_124: "f32[128, 384]" = torch.ops.aten.mm.default(view_492, permute_228);  view_492 = permute_228 = None
    view_493: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_124, [1, 128, 384]);  mm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_494: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_493, [1, -1, 6, 64]);  view_493 = None
    permute_229: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_494, [0, 2, 1, 3]);  view_494 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_83: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_229, [1, 6, 128, 64])
    view_501: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_83, [6, 128, 64]);  expand_83 = None
    bmm_41: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_500, view_501);  view_500 = view_501 = None
    view_502: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_41, [1, 6, 128, 64]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_231: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_502, [0, 2, 1, 3]);  view_502 = None
    clone_92: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_231, memory_format = torch.contiguous_format);  permute_231 = None
    view_503: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_92, [1, -1, 384]);  clone_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_504: "f32[128, 384]" = torch.ops.aten.reshape.default(view_503, [128, 384]);  view_503 = None
    permute_232: "f32[384, 512]" = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
    mm_125: "f32[128, 512]" = torch.ops.aten.mm.default(view_504, permute_232);  view_504 = permute_232 = None
    view_505: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_125, [1, 128, 512]);  mm_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    add_127: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_124, view_505);  add_124 = view_505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_51: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_127, 2)
    mean_36: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_51, [-1], True);  pow_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_128: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_36, 1e-06);  mean_36 = None
    rsqrt_36: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
    mul_149: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_127, rsqrt_36);  rsqrt_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_150: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(arg36_1, mul_149);  arg36_1 = mul_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_506: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_150, [128, 512]);  mul_150 = None
    permute_233: "f32[512, 384]" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
    mm_126: "f32[128, 384]" = torch.ops.aten.mm.default(view_506, permute_233);  view_506 = permute_233 = None
    view_507: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_126, [1, 128, 384]);  mm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_508: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_507, [1, -1, 6, 64]);  view_507 = None
    permute_234: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_508, [0, 2, 1, 3]);  view_508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_84: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_234, [1, 6, 128, 64]);  permute_234 = None
    view_515: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_84, [6, 128, 64]);  expand_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    view_509: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_76, [128, 512])
    permute_235: "f32[512, 384]" = torch.ops.aten.permute.default(arg172_1, [1, 0]);  arg172_1 = None
    mm_127: "f32[128, 384]" = torch.ops.aten.mm.default(view_509, permute_235);  view_509 = permute_235 = None
    view_510: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_127, [1, 128, 384]);  mm_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_511: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_510, [1, -1, 6, 64]);  view_510 = None
    permute_236: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_511, [0, 2, 1, 3]);  view_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_239: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_236, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_85: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_239, [1, 6, 64, 128]);  permute_239 = None
    view_516: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_85, [6, 64, 128]);  expand_85 = None
    bmm_42: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_515, view_516);  view_515 = view_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    view_517: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_42, [1, 6, 128, 128]);  bmm_42 = None
    view_518: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(view_517, [6, 128, 128]);  view_517 = None
    view_519: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_518, [1, 6, 128, 128]);  view_518 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_21: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_519, [-1], True)
    sub_26: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_519, amax_21);  view_519 = amax_21 = None
    exp_21: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_26);  sub_26 = None
    sum_22: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
    div_25: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_86: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(div_25, [1, 6, 128, 128]);  div_25 = None
    view_520: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_86, [6, 128, 128]);  expand_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    view_512: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_76, [128, 512])
    permute_237: "f32[512, 384]" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
    mm_128: "f32[128, 384]" = torch.ops.aten.mm.default(view_512, permute_237);  view_512 = permute_237 = None
    view_513: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_128, [1, 128, 384]);  mm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_514: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_513, [1, -1, 6, 64]);  view_513 = None
    permute_238: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_514, [0, 2, 1, 3]);  view_514 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_87: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_238, [1, 6, 128, 64])
    view_521: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_87, [6, 128, 64]);  expand_87 = None
    bmm_43: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_520, view_521);  view_520 = view_521 = None
    view_522: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_43, [1, 6, 128, 64]);  bmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_240: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_522, [0, 2, 1, 3]);  view_522 = None
    clone_95: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_240, memory_format = torch.contiguous_format);  permute_240 = None
    view_523: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_95, [1, -1, 384]);  clone_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_524: "f32[128, 384]" = torch.ops.aten.reshape.default(view_523, [128, 384]);  view_523 = None
    permute_241: "f32[384, 512]" = torch.ops.aten.permute.default(arg174_1, [1, 0]);  arg174_1 = None
    mm_129: "f32[128, 512]" = torch.ops.aten.mm.default(view_524, permute_241);  view_524 = permute_241 = None
    view_525: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_129, [1, 128, 512]);  mm_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:510, code: layer_output = hidden_states + self.dropout(attention_output[0])
    add_130: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_127, view_525);  add_127 = view_525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_52: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_130, 2)
    mean_37: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_52, [-1], True);  pow_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_131: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_37, 1e-06);  mean_37 = None
    rsqrt_37: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_131);  add_131 = None
    mul_151: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_130, rsqrt_37);  rsqrt_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_152: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(arg37_1, mul_151);  arg37_1 = mul_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_526: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_152, [128, 512])
    permute_242: "f32[512, 1024]" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
    mm_130: "f32[128, 1024]" = torch.ops.aten.mm.default(view_526, permute_242);  view_526 = permute_242 = None
    view_527: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_130, [1, 128, 1024]);  mm_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_153: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_527, 0.5)
    pow_53: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_527, 3.0)
    mul_154: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_53, 0.044715);  pow_53 = None
    add_132: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_527, mul_154);  view_527 = mul_154 = None
    mul_155: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_132, 0.7978845608028654);  add_132 = None
    tanh_14: "f32[1, 128, 1024]" = torch.ops.aten.tanh.default(mul_155);  mul_155 = None
    add_133: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_14, 1.0);  tanh_14 = None
    mul_156: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_153, add_133);  mul_153 = add_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_528: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_152, [128, 512]);  mul_152 = None
    permute_243: "f32[512, 1024]" = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
    mm_131: "f32[128, 1024]" = torch.ops.aten.mm.default(view_528, permute_243);  view_528 = permute_243 = None
    view_529: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_131, [1, 128, 1024]);  mm_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_157: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_156, view_529);  mul_156 = view_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    view_530: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_157, [128, 1024]);  mul_157 = None
    permute_244: "f32[1024, 512]" = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
    mm_132: "f32[128, 512]" = torch.ops.aten.mm.default(view_530, permute_244);  view_530 = permute_244 = None
    view_531: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_132, [1, 128, 512]);  mm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    add_134: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_130, view_531);  add_130 = view_531 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_54: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_134, 2)
    mean_38: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_54, [-1], True);  pow_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_135: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_38, 1e-06);  mean_38 = None
    rsqrt_38: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_135);  add_135 = None
    mul_158: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_134, rsqrt_38);  rsqrt_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_159: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(arg38_1, mul_158);  arg38_1 = mul_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_532: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_159, [128, 512])
    permute_245: "f32[512, 384]" = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
    mm_133: "f32[128, 384]" = torch.ops.aten.mm.default(view_532, permute_245);  view_532 = permute_245 = None
    view_533: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_133, [1, 128, 384]);  mm_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_534: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_533, [1, -1, 6, 64]);  view_533 = None
    permute_246: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_534, [0, 2, 1, 3]);  view_534 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_88: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_246, [1, 6, 128, 64]);  permute_246 = None
    view_541: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_88, [6, 128, 64]);  expand_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_535: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_159, [128, 512])
    permute_247: "f32[512, 384]" = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
    mm_134: "f32[128, 384]" = torch.ops.aten.mm.default(view_535, permute_247);  view_535 = permute_247 = None
    view_536: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_134, [1, 128, 384]);  mm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_537: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_536, [1, -1, 6, 64]);  view_536 = None
    permute_248: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_537, [0, 2, 1, 3]);  view_537 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_251: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_248, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_89: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_251, [1, 6, 64, 128]);  permute_251 = None
    view_542: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_89, [6, 64, 128]);  expand_89 = None
    bmm_44: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_541, view_542);  view_541 = view_542 = None
    view_543: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_44, [1, 6, 128, 128]);  bmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_136: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_543, add_64);  view_543 = add_64 = None
    view_544: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(add_136, [6, 128, 128]);  add_136 = None
    view_545: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_544, [1, 6, 128, 128]);  view_544 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_22: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_545, [-1], True)
    sub_27: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_545, amax_22);  view_545 = amax_22 = None
    exp_22: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_27);  sub_27 = None
    sum_23: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
    div_26: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_90: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(div_26, [1, 6, 128, 128]);  div_26 = None
    view_546: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_90, [6, 128, 128]);  expand_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    view_538: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_159, [128, 512]);  mul_159 = None
    permute_249: "f32[512, 384]" = torch.ops.aten.permute.default(arg180_1, [1, 0]);  arg180_1 = None
    mm_135: "f32[128, 384]" = torch.ops.aten.mm.default(view_538, permute_249);  view_538 = permute_249 = None
    view_539: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_135, [1, 128, 384]);  mm_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_540: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_539, [1, -1, 6, 64]);  view_539 = None
    permute_250: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_540, [0, 2, 1, 3]);  view_540 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_91: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_250, [1, 6, 128, 64])
    view_547: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_91, [6, 128, 64]);  expand_91 = None
    bmm_45: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_546, view_547);  view_546 = view_547 = None
    view_548: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_45, [1, 6, 128, 64]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_252: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_548, [0, 2, 1, 3]);  view_548 = None
    clone_100: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_252, memory_format = torch.contiguous_format);  permute_252 = None
    view_549: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_100, [1, -1, 384]);  clone_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_550: "f32[128, 384]" = torch.ops.aten.reshape.default(view_549, [128, 384]);  view_549 = None
    permute_253: "f32[384, 512]" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
    mm_136: "f32[128, 512]" = torch.ops.aten.mm.default(view_550, permute_253);  view_550 = permute_253 = None
    view_551: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_136, [1, 128, 512]);  mm_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    add_137: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_134, view_551);  add_134 = view_551 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_55: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_137, 2)
    mean_39: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_55, [-1], True);  pow_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_138: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_39, 1e-06);  mean_39 = None
    rsqrt_39: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
    mul_160: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_137, rsqrt_39);  rsqrt_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_161: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(arg39_1, mul_160);  arg39_1 = mul_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    view_552: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_161, [128, 512]);  mul_161 = None
    permute_254: "f32[512, 384]" = torch.ops.aten.permute.default(arg182_1, [1, 0]);  arg182_1 = None
    mm_137: "f32[128, 384]" = torch.ops.aten.mm.default(view_552, permute_254);  view_552 = permute_254 = None
    view_553: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_137, [1, 128, 384]);  mm_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_554: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_553, [1, -1, 6, 64]);  view_553 = None
    permute_255: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_554, [0, 2, 1, 3]);  view_554 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_92: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_255, [1, 6, 128, 64]);  permute_255 = None
    view_561: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_92, [6, 128, 64]);  expand_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    view_555: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_76, [128, 512])
    permute_256: "f32[512, 384]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
    mm_138: "f32[128, 384]" = torch.ops.aten.mm.default(view_555, permute_256);  view_555 = permute_256 = None
    view_556: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_138, [1, 128, 384]);  mm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_557: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_556, [1, -1, 6, 64]);  view_556 = None
    permute_257: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_557, [0, 2, 1, 3]);  view_557 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_260: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_257, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_93: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_260, [1, 6, 64, 128]);  permute_260 = None
    view_562: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_93, [6, 64, 128]);  expand_93 = None
    bmm_46: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_561, view_562);  view_561 = view_562 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    view_563: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_46, [1, 6, 128, 128]);  bmm_46 = None
    view_564: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(view_563, [6, 128, 128]);  view_563 = None
    view_565: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_564, [1, 6, 128, 128]);  view_564 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_23: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_565, [-1], True)
    sub_28: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_565, amax_23);  view_565 = amax_23 = None
    exp_23: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
    sum_24: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
    div_27: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_94: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(div_27, [1, 6, 128, 128]);  div_27 = None
    view_566: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_94, [6, 128, 128]);  expand_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    view_558: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_76, [128, 512])
    permute_258: "f32[512, 384]" = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
    mm_139: "f32[128, 384]" = torch.ops.aten.mm.default(view_558, permute_258);  view_558 = permute_258 = None
    view_559: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_139, [1, 128, 384]);  mm_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_560: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_559, [1, -1, 6, 64]);  view_559 = None
    permute_259: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_560, [0, 2, 1, 3]);  view_560 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_95: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_259, [1, 6, 128, 64])
    view_567: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_95, [6, 128, 64]);  expand_95 = None
    bmm_47: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_566, view_567);  view_566 = view_567 = None
    view_568: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_47, [1, 6, 128, 64]);  bmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_261: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_568, [0, 2, 1, 3]);  view_568 = None
    clone_103: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_261, memory_format = torch.contiguous_format);  permute_261 = None
    view_569: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_103, [1, -1, 384]);  clone_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    view_570: "f32[128, 384]" = torch.ops.aten.reshape.default(view_569, [128, 384]);  view_569 = None
    permute_262: "f32[384, 512]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
    mm_140: "f32[128, 512]" = torch.ops.aten.mm.default(view_570, permute_262);  view_570 = permute_262 = None
    view_571: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_140, [1, 128, 512]);  mm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:510, code: layer_output = hidden_states + self.dropout(attention_output[0])
    add_140: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_137, view_571);  add_137 = view_571 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_56: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_140, 2)
    mean_40: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_56, [-1], True);  pow_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_141: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_40, 1e-06);  mean_40 = None
    rsqrt_40: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
    mul_162: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_140, rsqrt_40);  rsqrt_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_163: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(arg40_1, mul_162);  arg40_1 = mul_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    view_572: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_163, [128, 512])
    permute_263: "f32[512, 1024]" = torch.ops.aten.permute.default(arg186_1, [1, 0]);  arg186_1 = None
    mm_141: "f32[128, 1024]" = torch.ops.aten.mm.default(view_572, permute_263);  view_572 = permute_263 = None
    view_573: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_141, [1, 128, 1024]);  mm_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_164: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_573, 0.5)
    pow_57: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_573, 3.0)
    mul_165: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_57, 0.044715);  pow_57 = None
    add_142: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_573, mul_165);  view_573 = mul_165 = None
    mul_166: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_142, 0.7978845608028654);  add_142 = None
    tanh_15: "f32[1, 128, 1024]" = torch.ops.aten.tanh.default(mul_166);  mul_166 = None
    add_143: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_15, 1.0);  tanh_15 = None
    mul_167: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_164, add_143);  mul_164 = add_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    view_574: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_163, [128, 512]);  mul_163 = None
    permute_264: "f32[512, 1024]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
    mm_142: "f32[128, 1024]" = torch.ops.aten.mm.default(view_574, permute_264);  view_574 = permute_264 = None
    view_575: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_142, [1, 128, 1024]);  mm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_168: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_167, view_575);  mul_167 = view_575 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    view_576: "f32[128, 1024]" = torch.ops.aten.reshape.default(mul_168, [128, 1024]);  mul_168 = None
    permute_265: "f32[1024, 512]" = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
    mm_143: "f32[128, 512]" = torch.ops.aten.mm.default(view_576, permute_265);  view_576 = permute_265 = None
    view_577: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_143, [1, 128, 512]);  mm_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    add_144: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_140, view_577);  add_140 = view_577 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_58: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_144, 2)
    mean_41: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_58, [-1], True);  pow_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_145: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_41, 1e-06);  mean_41 = None
    rsqrt_41: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_145);  add_145 = None
    mul_169: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_144, rsqrt_41);  add_144 = rsqrt_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_170: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(arg41_1, mul_169);  arg41_1 = mul_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:1799, code: lm_logits = self.lm_head(sequence_output)
    view_578: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_170, [128, 512]);  mul_170 = None
    permute_266: "f32[512, 250112]" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
    mm_144: "f32[128, 250112]" = torch.ops.aten.mm.default(view_578, permute_266);  view_578 = permute_266 = None
    view_579: "f32[1, 128, 250112]" = torch.ops.aten.reshape.default(mm_144, [1, 128, 250112]);  mm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:1806, code: loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
    view_580: "f32[128, 250112]" = torch.ops.aten.reshape.default(view_579, [-1, 250112])
    amax_24: "f32[128, 1]" = torch.ops.aten.amax.default(view_580, [1], True)
    sub_29: "f32[128, 250112]" = torch.ops.aten.sub.Tensor(view_580, amax_24);  view_580 = amax_24 = None
    exp_24: "f32[128, 250112]" = torch.ops.aten.exp.default(sub_29)
    sum_25: "f32[128, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
    log_2: "f32[128, 1]" = torch.ops.aten.log.default(sum_25);  sum_25 = None
    sub_30: "f32[128, 250112]" = torch.ops.aten.sub.Tensor(sub_29, log_2);  sub_29 = log_2 = None
    ne: "b8[128]" = torch.ops.aten.ne.Scalar(view_581, -100)
    full_default_6: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where_2: "i64[128]" = torch.ops.aten.where.self(ne, view_581, full_default_6);  ne = full_default_6 = None
    unsqueeze_17: "i64[128, 1]" = torch.ops.aten.unsqueeze.default(where_2, 1);  where_2 = None
    gather: "f32[128, 1]" = torch.ops.aten.gather.default(sub_30, 1, unsqueeze_17);  sub_30 = unsqueeze_17 = None
    squeeze: "f32[128]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg_1: "f32[128]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
    full_default_7: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where_3: "f32[128]" = torch.ops.aten.where.self(ne_1, neg_1, full_default_7);  ne_1 = neg_1 = full_default_7 = None
    sum_27: "f32[]" = torch.ops.aten.sum.default(where_3);  where_3 = None
    ne_2: "b8[128]" = torch.ops.aten.ne.Scalar(view_581, -100);  view_581 = None
    sum_26: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
    convert_element_type_7: "f32[]" = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
    div_28: "f32[]" = torch.ops.aten.div.Tensor(sum_27, convert_element_type_7);  sum_27 = convert_element_type_7 = None
    return (div_28, view_579, permute_100, permute_102, permute_110, permute_112, permute_122, permute_124, permute_131, permute_133, permute_143, permute_145, permute_152, permute_154, permute_164, permute_166, permute_173, permute_175, permute_185, permute_187, permute_194, permute_196, permute_206, permute_208, permute_215, permute_217, permute_227, permute_229, permute_236, permute_238, permute_248, permute_250, permute_257, permute_259, mul_76)
    