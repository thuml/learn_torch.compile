from __future__ import annotations



def forward(self, arg0_1: "f32[2304]", arg1_1: "f32[768, 2304]", arg2_1: "f32[768]", arg3_1: "f32[768, 768]", arg4_1: "f32[3072]", arg5_1: "f32[768, 3072]", arg6_1: "f32[768]", arg7_1: "f32[3072, 768]", arg8_1: "f32[2304]", arg9_1: "f32[768, 2304]", arg10_1: "f32[768]", arg11_1: "f32[768, 768]", arg12_1: "f32[3072]", arg13_1: "f32[768, 3072]", arg14_1: "f32[768]", arg15_1: "f32[3072, 768]", arg16_1: "f32[2304]", arg17_1: "f32[768, 2304]", arg18_1: "f32[768]", arg19_1: "f32[768, 768]", arg20_1: "f32[3072]", arg21_1: "f32[768, 3072]", arg22_1: "f32[768]", arg23_1: "f32[3072, 768]", arg24_1: "f32[2304]", arg25_1: "f32[768, 2304]", arg26_1: "f32[768]", arg27_1: "f32[768, 768]", arg28_1: "f32[3072]", arg29_1: "f32[768, 3072]", arg30_1: "f32[768]", arg31_1: "f32[3072, 768]", arg32_1: "f32[2304]", arg33_1: "f32[768, 2304]", arg34_1: "f32[768]", arg35_1: "f32[768, 768]", arg36_1: "f32[3072]", arg37_1: "f32[768, 3072]", arg38_1: "f32[768]", arg39_1: "f32[3072, 768]", arg40_1: "f32[2304]", arg41_1: "f32[768, 2304]", arg42_1: "f32[768]", arg43_1: "f32[768, 768]", arg44_1: "f32[3072]", arg45_1: "f32[768, 3072]", arg46_1: "f32[768]", arg47_1: "f32[3072, 768]", arg48_1: "f32[2304]", arg49_1: "f32[768, 2304]", arg50_1: "f32[768]", arg51_1: "f32[768, 768]", arg52_1: "f32[3072]", arg53_1: "f32[768, 3072]", arg54_1: "f32[768]", arg55_1: "f32[3072, 768]", arg56_1: "f32[2304]", arg57_1: "f32[768, 2304]", arg58_1: "f32[768]", arg59_1: "f32[768, 768]", arg60_1: "f32[3072]", arg61_1: "f32[768, 3072]", arg62_1: "f32[768]", arg63_1: "f32[3072, 768]", arg64_1: "f32[2304]", arg65_1: "f32[768, 2304]", arg66_1: "f32[768]", arg67_1: "f32[768, 768]", arg68_1: "f32[3072]", arg69_1: "f32[768, 3072]", arg70_1: "f32[768]", arg71_1: "f32[3072, 768]", arg72_1: "f32[2304]", arg73_1: "f32[768, 2304]", arg74_1: "f32[768]", arg75_1: "f32[768, 768]", arg76_1: "f32[3072]", arg77_1: "f32[768, 3072]", arg78_1: "f32[768]", arg79_1: "f32[3072, 768]", arg80_1: "f32[2304]", arg81_1: "f32[768, 2304]", arg82_1: "f32[768]", arg83_1: "f32[768, 768]", arg84_1: "f32[3072]", arg85_1: "f32[768, 3072]", arg86_1: "f32[768]", arg87_1: "f32[3072, 768]", arg88_1: "f32[2304]", arg89_1: "f32[768, 2304]", arg90_1: "f32[768]", arg91_1: "f32[768, 768]", arg92_1: "f32[3072]", arg93_1: "f32[768, 3072]", arg94_1: "f32[768]", arg95_1: "f32[3072, 768]", arg96_1: "f32[50257, 768]", arg97_1: "f32[1024, 768]", arg98_1: "f32[768]", arg99_1: "f32[768]", arg100_1: "f32[768]", arg101_1: "f32[768]", arg102_1: "f32[768]", arg103_1: "f32[768]", arg104_1: "f32[768]", arg105_1: "f32[768]", arg106_1: "f32[768]", arg107_1: "f32[768]", arg108_1: "f32[768]", arg109_1: "f32[768]", arg110_1: "f32[768]", arg111_1: "f32[768]", arg112_1: "f32[768]", arg113_1: "f32[768]", arg114_1: "f32[768]", arg115_1: "f32[768]", arg116_1: "f32[768]", arg117_1: "f32[768]", arg118_1: "f32[768]", arg119_1: "f32[768]", arg120_1: "f32[768]", arg121_1: "f32[768]", arg122_1: "f32[768]", arg123_1: "f32[768]", arg124_1: "f32[768]", arg125_1: "f32[768]", arg126_1: "f32[768]", arg127_1: "f32[768]", arg128_1: "f32[768]", arg129_1: "f32[768]", arg130_1: "f32[768]", arg131_1: "f32[768]", arg132_1: "f32[768]", arg133_1: "f32[768]", arg134_1: "f32[768]", arg135_1: "f32[768]", arg136_1: "f32[768]", arg137_1: "f32[768]", arg138_1: "f32[768]", arg139_1: "f32[768]", arg140_1: "f32[768]", arg141_1: "f32[768]", arg142_1: "f32[768]", arg143_1: "f32[768]", arg144_1: "f32[768]", arg145_1: "f32[768]", arg146_1: "f32[768]", arg147_1: "f32[768]", arg148_1: "f32[50257, 768]", arg149_1: "b8[1, 1, 1024, 1024]", arg150_1: "b8[1, 1, 1024, 1024]", arg151_1: "b8[1, 1, 1024, 1024]", arg152_1: "b8[1, 1, 1024, 1024]", arg153_1: "b8[1, 1, 1024, 1024]", arg154_1: "b8[1, 1, 1024, 1024]", arg155_1: "b8[1, 1, 1024, 1024]", arg156_1: "b8[1, 1, 1024, 1024]", arg157_1: "b8[1, 1, 1024, 1024]", arg158_1: "b8[1, 1, 1024, 1024]", arg159_1: "b8[1, 1, 1024, 1024]", arg160_1: "b8[1, 1, 1024, 1024]", arg161_1: "i64[2, 1024]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:781, code: input_ids = input_ids.view(-1, input_shape[-1])
    view: "i64[2, 1024]" = torch.ops.aten.reshape.default(arg161_1, [-1, 1024]);  arg161_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:843, code: inputs_embeds = self.wte(input_ids)
    embedding: "f32[2, 1024, 768]" = torch.ops.aten.embedding.default(arg96_1, view);  arg96_1 = view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:802, code: position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
    iota: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:803, code: position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
    unsqueeze: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
    view_1: "i64[1, 1024]" = torch.ops.aten.reshape.default(unsqueeze, [-1, 1024]);  unsqueeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:844, code: position_embeds = self.wpe(position_ids)
    embedding_1: "f32[1, 1024, 768]" = torch.ops.aten.embedding.default(arg97_1, view_1);  arg97_1 = view_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:845, code: hidden_states = inputs_embeds + position_embeds
    add: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
    getitem: "f32[2, 1024, 1]" = var_mean[0]
    getitem_1: "f32[2, 1024, 1]" = var_mean[1];  var_mean = None
    sub: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(add, getitem_1);  getitem_1 = None
    add_1: "f32[2, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[2, 1024, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    mul: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
    mul_1: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul, arg98_1);  mul = arg98_1 = None
    add_2: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_1, arg99_1);  mul_1 = arg99_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_2: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_2, [-1, 768]);  add_2 = None
    addmm: "f32[2048, 2304]" = torch.ops.aten.addmm.default(arg0_1, view_2, arg1_1);  arg0_1 = view_2 = arg1_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_3: "f32[2, 1024, 2304]" = torch.ops.aten.reshape.default(addmm, [2, 1024, 2304]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes = torch.ops.aten.split_with_sizes.default(view_3, [768, 768, 768], 2);  view_3 = None
    getitem_2: "f32[2, 1024, 768]" = split_with_sizes[0]
    getitem_3: "f32[2, 1024, 768]" = split_with_sizes[1]
    getitem_4: "f32[2, 1024, 768]" = split_with_sizes[2];  split_with_sizes = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_1: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
    device_put: "f32[]" = torch.ops.prims.device_put.default(full_1, device(type='cuda', index=0));  full_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_4: "f32[2, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_2, [2, 1024, 12, 64]);  getitem_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand: "f32[2, 12, 1024, 64]" = torch.ops.aten.expand.default(permute, [2, 12, 1024, 64]);  permute = None
    clone_1: "f32[2, 12, 1024, 64]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
    view_7: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(clone_1, [24, 1024, 64]);  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_5: "f32[2, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_3, [2, 1024, 12, 64]);  getitem_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_1: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_3: "f32[2, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_1, [0, 1, 3, 2])
    expand_1: "f32[2, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_3, [2, 12, 64, 1024]);  permute_3 = None
    clone_2: "f32[2, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
    view_8: "f32[24, 64, 1024]" = torch.ops.aten.reshape.default(clone_2, [24, 64, 1024]);  clone_2 = None
    bmm: "f32[24, 1024, 1024]" = torch.ops.aten.bmm.default(view_7, view_8);  view_7 = view_8 = None
    view_9: "f32[2, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm, [2, 12, 1024, 1024]);  bmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_default: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    div: "f32[2, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_9, full_default);  view_9 = full_default = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_1: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where: "f32[2, 12, 1024, 1024]" = torch.ops.aten.where.self(arg149_1, div, full_default_1);  arg149_1 = div = full_default_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax: "f32[2, 12, 1024, 1]" = torch.ops.aten.amax.default(where, [-1], True)
    sub_1: "f32[2, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where, amax);  where = amax = None
    exp: "f32[2, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
    sum_1: "f32[2, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div_1: "f32[2, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_2: "f32[2, 12, 1024, 1024]" = torch.ops.aten.expand.default(div_1, [2, 12, 1024, 1024]);  div_1 = None
    view_10: "f32[24, 1024, 1024]" = torch.ops.aten.reshape.default(expand_2, [24, 1024, 1024]);  expand_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_6: "f32[2, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_4, [2, 1024, 12, 64]);  getitem_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_2: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_3: "f32[2, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_2, [2, 12, 1024, 64])
    clone_4: "f32[2, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
    view_11: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(clone_4, [24, 1024, 64]);  clone_4 = None
    bmm_1: "f32[24, 1024, 64]" = torch.ops.aten.bmm.default(view_10, view_11);  view_10 = view_11 = None
    view_12: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_1, [2, 12, 1024, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_4: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None
    clone_5: "f32[2, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_13: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(clone_5, [2, 1024, 768]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_14: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_13, [-1, 768]);  view_13 = None
    
    # No stacktrace found for following nodes
    mm_default_36: "f32[2048, 768]" = torch.ops.aten.mm.default(view_14, arg3_1);  view_14 = arg3_1 = None
    add_tensor_35: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_36, arg2_1);  mm_default_36 = arg2_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_15: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_35, [2, 1024, 768]);  add_tensor_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_3: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(view_15, add);  view_15 = add = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
    getitem_5: "f32[2, 1024, 1]" = var_mean_1[0]
    getitem_6: "f32[2, 1024, 1]" = var_mean_1[1];  var_mean_1 = None
    sub_2: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(add_3, getitem_6);  getitem_6 = None
    add_4: "f32[2, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_5, 1e-05);  getitem_5 = None
    rsqrt_1: "f32[2, 1024, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    mul_2: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
    mul_3: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2, arg100_1);  mul_2 = arg100_1 = None
    add_5: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_3, arg101_1);  mul_3 = arg101_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_16: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_5, [-1, 768]);  add_5 = None
    
    # No stacktrace found for following nodes
    mm_default_35: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_16, arg5_1);  view_16 = arg5_1 = None
    add_tensor_34: "f32[2048, 3072]" = torch.ops.aten.add.Tensor(mm_default_35, arg4_1);  mm_default_35 = arg4_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_17: "f32[2, 1024, 3072]" = torch.ops.aten.reshape.default(add_tensor_34, [2, 1024, 3072]);  add_tensor_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_4: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_17, 0.5)
    pow_1: "f32[2, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_17, 3.0)
    mul_5: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
    add_6: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(view_17, mul_5);  view_17 = mul_5 = None
    mul_6: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_6, 0.7978845608028654);  add_6 = None
    tanh: "f32[2, 1024, 3072]" = torch.ops.aten.tanh.default(mul_6);  mul_6 = None
    add_7: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
    mul_7: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_4, add_7);  mul_4 = add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_18: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_7, [-1, 3072]);  mul_7 = None
    
    # No stacktrace found for following nodes
    mm_default_34: "f32[2048, 768]" = torch.ops.aten.mm.default(view_18, arg7_1);  view_18 = arg7_1 = None
    add_tensor_33: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_34, arg6_1);  mm_default_34 = arg6_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_19: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_33, [2, 1024, 768]);  add_tensor_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_8: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(add_3, view_19);  add_3 = view_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
    getitem_7: "f32[2, 1024, 1]" = var_mean_2[0]
    getitem_8: "f32[2, 1024, 1]" = var_mean_2[1];  var_mean_2 = None
    sub_3: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(add_8, getitem_8);  getitem_8 = None
    add_9: "f32[2, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_7, 1e-05);  getitem_7 = None
    rsqrt_2: "f32[2, 1024, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
    mul_8: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
    mul_9: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_8, arg102_1);  mul_8 = arg102_1 = None
    add_10: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_9, arg103_1);  mul_9 = arg103_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_20: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_10, [-1, 768]);  add_10 = None
    addmm_4: "f32[2048, 2304]" = torch.ops.aten.addmm.default(arg8_1, view_20, arg9_1);  arg8_1 = view_20 = arg9_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_21: "f32[2, 1024, 2304]" = torch.ops.aten.reshape.default(addmm_4, [2, 1024, 2304]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(view_21, [768, 768, 768], 2);  view_21 = None
    getitem_9: "f32[2, 1024, 768]" = split_with_sizes_1[0]
    getitem_10: "f32[2, 1024, 768]" = split_with_sizes_1[1]
    getitem_11: "f32[2, 1024, 768]" = split_with_sizes_1[2];  split_with_sizes_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_3: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
    device_put_1: "f32[]" = torch.ops.prims.device_put.default(full_3, device(type='cuda', index=0));  full_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_22: "f32[2, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_9, [2, 1024, 12, 64]);  getitem_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_5: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(view_22, [0, 2, 1, 3]);  view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_4: "f32[2, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_5, [2, 12, 1024, 64]);  permute_5 = None
    clone_8: "f32[2, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
    view_25: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(clone_8, [24, 1024, 64]);  clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_23: "f32[2, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_10, [2, 1024, 12, 64]);  getitem_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_6: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(view_23, [0, 2, 1, 3]);  view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_8: "f32[2, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_6, [0, 1, 3, 2])
    expand_5: "f32[2, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_8, [2, 12, 64, 1024]);  permute_8 = None
    clone_9: "f32[2, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
    view_26: "f32[24, 64, 1024]" = torch.ops.aten.reshape.default(clone_9, [24, 64, 1024]);  clone_9 = None
    bmm_2: "f32[24, 1024, 1024]" = torch.ops.aten.bmm.default(view_25, view_26);  view_25 = view_26 = None
    view_27: "f32[2, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_2, [2, 12, 1024, 1024]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_default_2: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    div_2: "f32[2, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_27, full_default_2);  view_27 = full_default_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_3: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_1: "f32[2, 12, 1024, 1024]" = torch.ops.aten.where.self(arg150_1, div_2, full_default_3);  arg150_1 = div_2 = full_default_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_1: "f32[2, 12, 1024, 1]" = torch.ops.aten.amax.default(where_1, [-1], True)
    sub_4: "f32[2, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where_1, amax_1);  where_1 = amax_1 = None
    exp_1: "f32[2, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
    sum_2: "f32[2, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_3: "f32[2, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_6: "f32[2, 12, 1024, 1024]" = torch.ops.aten.expand.default(div_3, [2, 12, 1024, 1024]);  div_3 = None
    view_28: "f32[24, 1024, 1024]" = torch.ops.aten.reshape.default(expand_6, [24, 1024, 1024]);  expand_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_24: "f32[2, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_11, [2, 1024, 12, 64]);  getitem_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_7: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_7: "f32[2, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_7, [2, 12, 1024, 64])
    clone_11: "f32[2, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
    view_29: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(clone_11, [24, 1024, 64]);  clone_11 = None
    bmm_3: "f32[24, 1024, 64]" = torch.ops.aten.bmm.default(view_28, view_29);  view_28 = view_29 = None
    view_30: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_3, [2, 12, 1024, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_9: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
    clone_12: "f32[2, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_9, memory_format = torch.contiguous_format);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_31: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(clone_12, [2, 1024, 768]);  clone_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_32: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_31, [-1, 768]);  view_31 = None
    
    # No stacktrace found for following nodes
    mm_default_33: "f32[2048, 768]" = torch.ops.aten.mm.default(view_32, arg11_1);  view_32 = arg11_1 = None
    add_tensor_32: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_33, arg10_1);  mm_default_33 = arg10_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_33: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_32, [2, 1024, 768]);  add_tensor_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_11: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(view_33, add_8);  view_33 = add_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_3 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
    getitem_12: "f32[2, 1024, 1]" = var_mean_3[0]
    getitem_13: "f32[2, 1024, 1]" = var_mean_3[1];  var_mean_3 = None
    sub_5: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(add_11, getitem_13);  getitem_13 = None
    add_12: "f32[2, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
    rsqrt_3: "f32[2, 1024, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
    mul_10: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
    mul_11: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_10, arg104_1);  mul_10 = arg104_1 = None
    add_13: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_11, arg105_1);  mul_11 = arg105_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_34: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_13, [-1, 768]);  add_13 = None
    
    # No stacktrace found for following nodes
    mm_default_32: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_34, arg13_1);  view_34 = arg13_1 = None
    add_tensor_31: "f32[2048, 3072]" = torch.ops.aten.add.Tensor(mm_default_32, arg12_1);  mm_default_32 = arg12_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_35: "f32[2, 1024, 3072]" = torch.ops.aten.reshape.default(add_tensor_31, [2, 1024, 3072]);  add_tensor_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_12: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_35, 0.5)
    pow_2: "f32[2, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_35, 3.0)
    mul_13: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
    add_14: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(view_35, mul_13);  view_35 = mul_13 = None
    mul_14: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_14, 0.7978845608028654);  add_14 = None
    tanh_1: "f32[2, 1024, 3072]" = torch.ops.aten.tanh.default(mul_14);  mul_14 = None
    add_15: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
    mul_15: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_12, add_15);  mul_12 = add_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_36: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_15, [-1, 3072]);  mul_15 = None
    
    # No stacktrace found for following nodes
    mm_default_31: "f32[2048, 768]" = torch.ops.aten.mm.default(view_36, arg15_1);  view_36 = arg15_1 = None
    add_tensor_30: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_31, arg14_1);  mm_default_31 = arg14_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_37: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_30, [2, 1024, 768]);  add_tensor_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_16: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(add_11, view_37);  add_11 = view_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_4 = torch.ops.aten.var_mean.correction(add_16, [2], correction = 0, keepdim = True)
    getitem_14: "f32[2, 1024, 1]" = var_mean_4[0]
    getitem_15: "f32[2, 1024, 1]" = var_mean_4[1];  var_mean_4 = None
    sub_6: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(add_16, getitem_15);  getitem_15 = None
    add_17: "f32[2, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
    rsqrt_4: "f32[2, 1024, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
    mul_16: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
    mul_17: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_16, arg106_1);  mul_16 = arg106_1 = None
    add_18: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_17, arg107_1);  mul_17 = arg107_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_38: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_18, [-1, 768]);  add_18 = None
    addmm_8: "f32[2048, 2304]" = torch.ops.aten.addmm.default(arg16_1, view_38, arg17_1);  arg16_1 = view_38 = arg17_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_39: "f32[2, 1024, 2304]" = torch.ops.aten.reshape.default(addmm_8, [2, 1024, 2304]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_2 = torch.ops.aten.split_with_sizes.default(view_39, [768, 768, 768], 2);  view_39 = None
    getitem_16: "f32[2, 1024, 768]" = split_with_sizes_2[0]
    getitem_17: "f32[2, 1024, 768]" = split_with_sizes_2[1]
    getitem_18: "f32[2, 1024, 768]" = split_with_sizes_2[2];  split_with_sizes_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_5: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
    device_put_2: "f32[]" = torch.ops.prims.device_put.default(full_5, device(type='cuda', index=0));  full_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_40: "f32[2, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_16, [2, 1024, 12, 64]);  getitem_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_10: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_8: "f32[2, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_10, [2, 12, 1024, 64]);  permute_10 = None
    clone_15: "f32[2, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
    view_43: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(clone_15, [24, 1024, 64]);  clone_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_41: "f32[2, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_17, [2, 1024, 12, 64]);  getitem_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_11: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_13: "f32[2, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_11, [0, 1, 3, 2])
    expand_9: "f32[2, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_13, [2, 12, 64, 1024]);  permute_13 = None
    clone_16: "f32[2, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
    view_44: "f32[24, 64, 1024]" = torch.ops.aten.reshape.default(clone_16, [24, 64, 1024]);  clone_16 = None
    bmm_4: "f32[24, 1024, 1024]" = torch.ops.aten.bmm.default(view_43, view_44);  view_43 = view_44 = None
    view_45: "f32[2, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_4, [2, 12, 1024, 1024]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_default_4: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    div_4: "f32[2, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_45, full_default_4);  view_45 = full_default_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_5: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_2: "f32[2, 12, 1024, 1024]" = torch.ops.aten.where.self(arg151_1, div_4, full_default_5);  arg151_1 = div_4 = full_default_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_2: "f32[2, 12, 1024, 1]" = torch.ops.aten.amax.default(where_2, [-1], True)
    sub_7: "f32[2, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where_2, amax_2);  where_2 = amax_2 = None
    exp_2: "f32[2, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
    sum_3: "f32[2, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_5: "f32[2, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_10: "f32[2, 12, 1024, 1024]" = torch.ops.aten.expand.default(div_5, [2, 12, 1024, 1024]);  div_5 = None
    view_46: "f32[24, 1024, 1024]" = torch.ops.aten.reshape.default(expand_10, [24, 1024, 1024]);  expand_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_42: "f32[2, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_18, [2, 1024, 12, 64]);  getitem_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_12: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_11: "f32[2, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_12, [2, 12, 1024, 64])
    clone_18: "f32[2, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
    view_47: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(clone_18, [24, 1024, 64]);  clone_18 = None
    bmm_5: "f32[24, 1024, 64]" = torch.ops.aten.bmm.default(view_46, view_47);  view_46 = view_47 = None
    view_48: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_5, [2, 12, 1024, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_14: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
    clone_19: "f32[2, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_49: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(clone_19, [2, 1024, 768]);  clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_50: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_49, [-1, 768]);  view_49 = None
    
    # No stacktrace found for following nodes
    mm_default_30: "f32[2048, 768]" = torch.ops.aten.mm.default(view_50, arg19_1);  view_50 = arg19_1 = None
    add_tensor_29: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_30, arg18_1);  mm_default_30 = arg18_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_51: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_29, [2, 1024, 768]);  add_tensor_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_19: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(view_51, add_16);  view_51 = add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_5 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
    getitem_19: "f32[2, 1024, 1]" = var_mean_5[0]
    getitem_20: "f32[2, 1024, 1]" = var_mean_5[1];  var_mean_5 = None
    sub_8: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(add_19, getitem_20);  getitem_20 = None
    add_20: "f32[2, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_19, 1e-05);  getitem_19 = None
    rsqrt_5: "f32[2, 1024, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
    mul_18: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
    mul_19: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_18, arg108_1);  mul_18 = arg108_1 = None
    add_21: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_19, arg109_1);  mul_19 = arg109_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_52: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_21, [-1, 768]);  add_21 = None
    
    # No stacktrace found for following nodes
    mm_default_29: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_52, arg21_1);  view_52 = arg21_1 = None
    add_tensor_28: "f32[2048, 3072]" = torch.ops.aten.add.Tensor(mm_default_29, arg20_1);  mm_default_29 = arg20_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_53: "f32[2, 1024, 3072]" = torch.ops.aten.reshape.default(add_tensor_28, [2, 1024, 3072]);  add_tensor_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_20: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_53, 0.5)
    pow_3: "f32[2, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_53, 3.0)
    mul_21: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
    add_22: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(view_53, mul_21);  view_53 = mul_21 = None
    mul_22: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_22, 0.7978845608028654);  add_22 = None
    tanh_2: "f32[2, 1024, 3072]" = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
    add_23: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
    mul_23: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_20, add_23);  mul_20 = add_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_54: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_23, [-1, 3072]);  mul_23 = None
    
    # No stacktrace found for following nodes
    mm_default_28: "f32[2048, 768]" = torch.ops.aten.mm.default(view_54, arg23_1);  view_54 = arg23_1 = None
    add_tensor_27: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_28, arg22_1);  mm_default_28 = arg22_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_55: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_27, [2, 1024, 768]);  add_tensor_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_24: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(add_19, view_55);  add_19 = view_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_6 = torch.ops.aten.var_mean.correction(add_24, [2], correction = 0, keepdim = True)
    getitem_21: "f32[2, 1024, 1]" = var_mean_6[0]
    getitem_22: "f32[2, 1024, 1]" = var_mean_6[1];  var_mean_6 = None
    sub_9: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(add_24, getitem_22);  getitem_22 = None
    add_25: "f32[2, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_21, 1e-05);  getitem_21 = None
    rsqrt_6: "f32[2, 1024, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
    mul_24: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = rsqrt_6 = None
    mul_25: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_24, arg110_1);  mul_24 = arg110_1 = None
    add_26: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_25, arg111_1);  mul_25 = arg111_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_56: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_26, [-1, 768]);  add_26 = None
    addmm_12: "f32[2048, 2304]" = torch.ops.aten.addmm.default(arg24_1, view_56, arg25_1);  arg24_1 = view_56 = arg25_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_57: "f32[2, 1024, 2304]" = torch.ops.aten.reshape.default(addmm_12, [2, 1024, 2304]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_3 = torch.ops.aten.split_with_sizes.default(view_57, [768, 768, 768], 2);  view_57 = None
    getitem_23: "f32[2, 1024, 768]" = split_with_sizes_3[0]
    getitem_24: "f32[2, 1024, 768]" = split_with_sizes_3[1]
    getitem_25: "f32[2, 1024, 768]" = split_with_sizes_3[2];  split_with_sizes_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_7: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
    device_put_3: "f32[]" = torch.ops.prims.device_put.default(full_7, device(type='cuda', index=0));  full_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_58: "f32[2, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_23, [2, 1024, 12, 64]);  getitem_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_15: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_12: "f32[2, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_15, [2, 12, 1024, 64]);  permute_15 = None
    clone_22: "f32[2, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
    view_61: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(clone_22, [24, 1024, 64]);  clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_59: "f32[2, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_24, [2, 1024, 12, 64]);  getitem_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_16: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(view_59, [0, 2, 1, 3]);  view_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_18: "f32[2, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_16, [0, 1, 3, 2])
    expand_13: "f32[2, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_18, [2, 12, 64, 1024]);  permute_18 = None
    clone_23: "f32[2, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
    view_62: "f32[24, 64, 1024]" = torch.ops.aten.reshape.default(clone_23, [24, 64, 1024]);  clone_23 = None
    bmm_6: "f32[24, 1024, 1024]" = torch.ops.aten.bmm.default(view_61, view_62);  view_61 = view_62 = None
    view_63: "f32[2, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_6, [2, 12, 1024, 1024]);  bmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_default_6: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    div_6: "f32[2, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_63, full_default_6);  view_63 = full_default_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_7: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_3: "f32[2, 12, 1024, 1024]" = torch.ops.aten.where.self(arg152_1, div_6, full_default_7);  arg152_1 = div_6 = full_default_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_3: "f32[2, 12, 1024, 1]" = torch.ops.aten.amax.default(where_3, [-1], True)
    sub_10: "f32[2, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where_3, amax_3);  where_3 = amax_3 = None
    exp_3: "f32[2, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
    sum_4: "f32[2, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_7: "f32[2, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_14: "f32[2, 12, 1024, 1024]" = torch.ops.aten.expand.default(div_7, [2, 12, 1024, 1024]);  div_7 = None
    view_64: "f32[24, 1024, 1024]" = torch.ops.aten.reshape.default(expand_14, [24, 1024, 1024]);  expand_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_60: "f32[2, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_25, [2, 1024, 12, 64]);  getitem_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_17: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(view_60, [0, 2, 1, 3]);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_15: "f32[2, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_17, [2, 12, 1024, 64])
    clone_25: "f32[2, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
    view_65: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(clone_25, [24, 1024, 64]);  clone_25 = None
    bmm_7: "f32[24, 1024, 64]" = torch.ops.aten.bmm.default(view_64, view_65);  view_64 = view_65 = None
    view_66: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_7, [2, 12, 1024, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_19: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None
    clone_26: "f32[2, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_67: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(clone_26, [2, 1024, 768]);  clone_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_68: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_67, [-1, 768]);  view_67 = None
    
    # No stacktrace found for following nodes
    mm_default_27: "f32[2048, 768]" = torch.ops.aten.mm.default(view_68, arg27_1);  view_68 = arg27_1 = None
    add_tensor_26: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_27, arg26_1);  mm_default_27 = arg26_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_69: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_26, [2, 1024, 768]);  add_tensor_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_27: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(view_69, add_24);  view_69 = add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_7 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
    getitem_26: "f32[2, 1024, 1]" = var_mean_7[0]
    getitem_27: "f32[2, 1024, 1]" = var_mean_7[1];  var_mean_7 = None
    sub_11: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(add_27, getitem_27);  getitem_27 = None
    add_28: "f32[2, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
    rsqrt_7: "f32[2, 1024, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    mul_26: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
    mul_27: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_26, arg112_1);  mul_26 = arg112_1 = None
    add_29: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_27, arg113_1);  mul_27 = arg113_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_70: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_29, [-1, 768]);  add_29 = None
    
    # No stacktrace found for following nodes
    mm_default_26: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_70, arg29_1);  view_70 = arg29_1 = None
    add_tensor_25: "f32[2048, 3072]" = torch.ops.aten.add.Tensor(mm_default_26, arg28_1);  mm_default_26 = arg28_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_71: "f32[2, 1024, 3072]" = torch.ops.aten.reshape.default(add_tensor_25, [2, 1024, 3072]);  add_tensor_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_28: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_71, 0.5)
    pow_4: "f32[2, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_71, 3.0)
    mul_29: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
    add_30: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(view_71, mul_29);  view_71 = mul_29 = None
    mul_30: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_30, 0.7978845608028654);  add_30 = None
    tanh_3: "f32[2, 1024, 3072]" = torch.ops.aten.tanh.default(mul_30);  mul_30 = None
    add_31: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
    mul_31: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_28, add_31);  mul_28 = add_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_72: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_31, [-1, 3072]);  mul_31 = None
    
    # No stacktrace found for following nodes
    mm_default_25: "f32[2048, 768]" = torch.ops.aten.mm.default(view_72, arg31_1);  view_72 = arg31_1 = None
    add_tensor_24: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_25, arg30_1);  mm_default_25 = arg30_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_73: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_24, [2, 1024, 768]);  add_tensor_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_32: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(add_27, view_73);  add_27 = view_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_8 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
    getitem_28: "f32[2, 1024, 1]" = var_mean_8[0]
    getitem_29: "f32[2, 1024, 1]" = var_mean_8[1];  var_mean_8 = None
    sub_12: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(add_32, getitem_29);  getitem_29 = None
    add_33: "f32[2, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
    rsqrt_8: "f32[2, 1024, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
    mul_32: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = rsqrt_8 = None
    mul_33: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_32, arg114_1);  mul_32 = arg114_1 = None
    add_34: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_33, arg115_1);  mul_33 = arg115_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_74: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_34, [-1, 768]);  add_34 = None
    addmm_16: "f32[2048, 2304]" = torch.ops.aten.addmm.default(arg32_1, view_74, arg33_1);  arg32_1 = view_74 = arg33_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_75: "f32[2, 1024, 2304]" = torch.ops.aten.reshape.default(addmm_16, [2, 1024, 2304]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_4 = torch.ops.aten.split_with_sizes.default(view_75, [768, 768, 768], 2);  view_75 = None
    getitem_30: "f32[2, 1024, 768]" = split_with_sizes_4[0]
    getitem_31: "f32[2, 1024, 768]" = split_with_sizes_4[1]
    getitem_32: "f32[2, 1024, 768]" = split_with_sizes_4[2];  split_with_sizes_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_9: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
    device_put_4: "f32[]" = torch.ops.prims.device_put.default(full_9, device(type='cuda', index=0));  full_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_76: "f32[2, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_30, [2, 1024, 12, 64]);  getitem_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_20: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_16: "f32[2, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_20, [2, 12, 1024, 64]);  permute_20 = None
    clone_29: "f32[2, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
    view_79: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(clone_29, [24, 1024, 64]);  clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_77: "f32[2, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_31, [2, 1024, 12, 64]);  getitem_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_21: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(view_77, [0, 2, 1, 3]);  view_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_23: "f32[2, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_21, [0, 1, 3, 2])
    expand_17: "f32[2, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_23, [2, 12, 64, 1024]);  permute_23 = None
    clone_30: "f32[2, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
    view_80: "f32[24, 64, 1024]" = torch.ops.aten.reshape.default(clone_30, [24, 64, 1024]);  clone_30 = None
    bmm_8: "f32[24, 1024, 1024]" = torch.ops.aten.bmm.default(view_79, view_80);  view_79 = view_80 = None
    view_81: "f32[2, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_8, [2, 12, 1024, 1024]);  bmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_default_8: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    div_8: "f32[2, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_81, full_default_8);  view_81 = full_default_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_9: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_4: "f32[2, 12, 1024, 1024]" = torch.ops.aten.where.self(arg153_1, div_8, full_default_9);  arg153_1 = div_8 = full_default_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_4: "f32[2, 12, 1024, 1]" = torch.ops.aten.amax.default(where_4, [-1], True)
    sub_13: "f32[2, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where_4, amax_4);  where_4 = amax_4 = None
    exp_4: "f32[2, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
    sum_5: "f32[2, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_9: "f32[2, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_18: "f32[2, 12, 1024, 1024]" = torch.ops.aten.expand.default(div_9, [2, 12, 1024, 1024]);  div_9 = None
    view_82: "f32[24, 1024, 1024]" = torch.ops.aten.reshape.default(expand_18, [24, 1024, 1024]);  expand_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_78: "f32[2, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_32, [2, 1024, 12, 64]);  getitem_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_22: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_19: "f32[2, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_22, [2, 12, 1024, 64])
    clone_32: "f32[2, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
    view_83: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(clone_32, [24, 1024, 64]);  clone_32 = None
    bmm_9: "f32[24, 1024, 64]" = torch.ops.aten.bmm.default(view_82, view_83);  view_82 = view_83 = None
    view_84: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_9, [2, 12, 1024, 64]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_24: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_84, [0, 2, 1, 3]);  view_84 = None
    clone_33: "f32[2, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_85: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(clone_33, [2, 1024, 768]);  clone_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_86: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_85, [-1, 768]);  view_85 = None
    
    # No stacktrace found for following nodes
    mm_default_24: "f32[2048, 768]" = torch.ops.aten.mm.default(view_86, arg35_1);  view_86 = arg35_1 = None
    add_tensor_23: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_24, arg34_1);  mm_default_24 = arg34_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_87: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_23, [2, 1024, 768]);  add_tensor_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_35: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(view_87, add_32);  view_87 = add_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_9 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
    getitem_33: "f32[2, 1024, 1]" = var_mean_9[0]
    getitem_34: "f32[2, 1024, 1]" = var_mean_9[1];  var_mean_9 = None
    sub_14: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_34);  getitem_34 = None
    add_36: "f32[2, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_33, 1e-05);  getitem_33 = None
    rsqrt_9: "f32[2, 1024, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    mul_34: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
    mul_35: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_34, arg116_1);  mul_34 = arg116_1 = None
    add_37: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_35, arg117_1);  mul_35 = arg117_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_88: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_37, [-1, 768]);  add_37 = None
    
    # No stacktrace found for following nodes
    mm_default_23: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_88, arg37_1);  view_88 = arg37_1 = None
    add_tensor_22: "f32[2048, 3072]" = torch.ops.aten.add.Tensor(mm_default_23, arg36_1);  mm_default_23 = arg36_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_89: "f32[2, 1024, 3072]" = torch.ops.aten.reshape.default(add_tensor_22, [2, 1024, 3072]);  add_tensor_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_36: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_89, 0.5)
    pow_5: "f32[2, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_89, 3.0)
    mul_37: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
    add_38: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(view_89, mul_37);  view_89 = mul_37 = None
    mul_38: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_38, 0.7978845608028654);  add_38 = None
    tanh_4: "f32[2, 1024, 3072]" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
    add_39: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
    mul_39: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_36, add_39);  mul_36 = add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_90: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_39, [-1, 3072]);  mul_39 = None
    
    # No stacktrace found for following nodes
    mm_default_22: "f32[2048, 768]" = torch.ops.aten.mm.default(view_90, arg39_1);  view_90 = arg39_1 = None
    add_tensor_21: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_22, arg38_1);  mm_default_22 = arg38_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_91: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_21, [2, 1024, 768]);  add_tensor_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_40: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(add_35, view_91);  add_35 = view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_10 = torch.ops.aten.var_mean.correction(add_40, [2], correction = 0, keepdim = True)
    getitem_35: "f32[2, 1024, 1]" = var_mean_10[0]
    getitem_36: "f32[2, 1024, 1]" = var_mean_10[1];  var_mean_10 = None
    sub_15: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(add_40, getitem_36);  getitem_36 = None
    add_41: "f32[2, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_35, 1e-05);  getitem_35 = None
    rsqrt_10: "f32[2, 1024, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
    mul_40: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = rsqrt_10 = None
    mul_41: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_40, arg118_1);  mul_40 = arg118_1 = None
    add_42: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_41, arg119_1);  mul_41 = arg119_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_92: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_42, [-1, 768]);  add_42 = None
    addmm_20: "f32[2048, 2304]" = torch.ops.aten.addmm.default(arg40_1, view_92, arg41_1);  arg40_1 = view_92 = arg41_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_93: "f32[2, 1024, 2304]" = torch.ops.aten.reshape.default(addmm_20, [2, 1024, 2304]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_5 = torch.ops.aten.split_with_sizes.default(view_93, [768, 768, 768], 2);  view_93 = None
    getitem_37: "f32[2, 1024, 768]" = split_with_sizes_5[0]
    getitem_38: "f32[2, 1024, 768]" = split_with_sizes_5[1]
    getitem_39: "f32[2, 1024, 768]" = split_with_sizes_5[2];  split_with_sizes_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_11: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
    device_put_5: "f32[]" = torch.ops.prims.device_put.default(full_11, device(type='cuda', index=0));  full_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_94: "f32[2, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_37, [2, 1024, 12, 64]);  getitem_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_25: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_20: "f32[2, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_25, [2, 12, 1024, 64]);  permute_25 = None
    clone_36: "f32[2, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
    view_97: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(clone_36, [24, 1024, 64]);  clone_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_95: "f32[2, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_38, [2, 1024, 12, 64]);  getitem_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_26: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_28: "f32[2, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_26, [0, 1, 3, 2])
    expand_21: "f32[2, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_28, [2, 12, 64, 1024]);  permute_28 = None
    clone_37: "f32[2, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
    view_98: "f32[24, 64, 1024]" = torch.ops.aten.reshape.default(clone_37, [24, 64, 1024]);  clone_37 = None
    bmm_10: "f32[24, 1024, 1024]" = torch.ops.aten.bmm.default(view_97, view_98);  view_97 = view_98 = None
    view_99: "f32[2, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_10, [2, 12, 1024, 1024]);  bmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_default_10: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    div_10: "f32[2, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_99, full_default_10);  view_99 = full_default_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_11: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_5: "f32[2, 12, 1024, 1024]" = torch.ops.aten.where.self(arg154_1, div_10, full_default_11);  arg154_1 = div_10 = full_default_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_5: "f32[2, 12, 1024, 1]" = torch.ops.aten.amax.default(where_5, [-1], True)
    sub_16: "f32[2, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where_5, amax_5);  where_5 = amax_5 = None
    exp_5: "f32[2, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
    sum_6: "f32[2, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_11: "f32[2, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_22: "f32[2, 12, 1024, 1024]" = torch.ops.aten.expand.default(div_11, [2, 12, 1024, 1024]);  div_11 = None
    view_100: "f32[24, 1024, 1024]" = torch.ops.aten.reshape.default(expand_22, [24, 1024, 1024]);  expand_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_96: "f32[2, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_39, [2, 1024, 12, 64]);  getitem_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_27: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_23: "f32[2, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_27, [2, 12, 1024, 64])
    clone_39: "f32[2, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
    view_101: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(clone_39, [24, 1024, 64]);  clone_39 = None
    bmm_11: "f32[24, 1024, 64]" = torch.ops.aten.bmm.default(view_100, view_101);  view_100 = view_101 = None
    view_102: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_11, [2, 12, 1024, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_29: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
    clone_40: "f32[2, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_103: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(clone_40, [2, 1024, 768]);  clone_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_104: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_103, [-1, 768]);  view_103 = None
    
    # No stacktrace found for following nodes
    mm_default_21: "f32[2048, 768]" = torch.ops.aten.mm.default(view_104, arg43_1);  view_104 = arg43_1 = None
    add_tensor_20: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_21, arg42_1);  mm_default_21 = arg42_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_105: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_20, [2, 1024, 768]);  add_tensor_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_43: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(view_105, add_40);  view_105 = add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_11 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
    getitem_40: "f32[2, 1024, 1]" = var_mean_11[0]
    getitem_41: "f32[2, 1024, 1]" = var_mean_11[1];  var_mean_11 = None
    sub_17: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(add_43, getitem_41);  getitem_41 = None
    add_44: "f32[2, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
    rsqrt_11: "f32[2, 1024, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
    mul_42: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
    mul_43: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_42, arg120_1);  mul_42 = arg120_1 = None
    add_45: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_43, arg121_1);  mul_43 = arg121_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_106: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_45, [-1, 768]);  add_45 = None
    
    # No stacktrace found for following nodes
    mm_default_20: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_106, arg45_1);  view_106 = arg45_1 = None
    add_tensor_19: "f32[2048, 3072]" = torch.ops.aten.add.Tensor(mm_default_20, arg44_1);  mm_default_20 = arg44_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_107: "f32[2, 1024, 3072]" = torch.ops.aten.reshape.default(add_tensor_19, [2, 1024, 3072]);  add_tensor_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_44: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.5)
    pow_6: "f32[2, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_107, 3.0)
    mul_45: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
    add_46: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(view_107, mul_45);  view_107 = mul_45 = None
    mul_46: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_46, 0.7978845608028654);  add_46 = None
    tanh_5: "f32[2, 1024, 3072]" = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
    add_47: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
    mul_47: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_44, add_47);  mul_44 = add_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_108: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_47, [-1, 3072]);  mul_47 = None
    
    # No stacktrace found for following nodes
    mm_default_19: "f32[2048, 768]" = torch.ops.aten.mm.default(view_108, arg47_1);  view_108 = arg47_1 = None
    add_tensor_18: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_19, arg46_1);  mm_default_19 = arg46_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_109: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_18, [2, 1024, 768]);  add_tensor_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_48: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(add_43, view_109);  add_43 = view_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_12 = torch.ops.aten.var_mean.correction(add_48, [2], correction = 0, keepdim = True)
    getitem_42: "f32[2, 1024, 1]" = var_mean_12[0]
    getitem_43: "f32[2, 1024, 1]" = var_mean_12[1];  var_mean_12 = None
    sub_18: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(add_48, getitem_43);  getitem_43 = None
    add_49: "f32[2, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
    rsqrt_12: "f32[2, 1024, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    mul_48: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = rsqrt_12 = None
    mul_49: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_48, arg122_1);  mul_48 = arg122_1 = None
    add_50: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_49, arg123_1);  mul_49 = arg123_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_110: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_50, [-1, 768]);  add_50 = None
    addmm_24: "f32[2048, 2304]" = torch.ops.aten.addmm.default(arg48_1, view_110, arg49_1);  arg48_1 = view_110 = arg49_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_111: "f32[2, 1024, 2304]" = torch.ops.aten.reshape.default(addmm_24, [2, 1024, 2304]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_6 = torch.ops.aten.split_with_sizes.default(view_111, [768, 768, 768], 2);  view_111 = None
    getitem_44: "f32[2, 1024, 768]" = split_with_sizes_6[0]
    getitem_45: "f32[2, 1024, 768]" = split_with_sizes_6[1]
    getitem_46: "f32[2, 1024, 768]" = split_with_sizes_6[2];  split_with_sizes_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_13: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
    device_put_6: "f32[]" = torch.ops.prims.device_put.default(full_13, device(type='cuda', index=0));  full_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_112: "f32[2, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_44, [2, 1024, 12, 64]);  getitem_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_30: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_24: "f32[2, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_30, [2, 12, 1024, 64]);  permute_30 = None
    clone_43: "f32[2, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
    view_115: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(clone_43, [24, 1024, 64]);  clone_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_113: "f32[2, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_45, [2, 1024, 12, 64]);  getitem_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_31: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(view_113, [0, 2, 1, 3]);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_33: "f32[2, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_31, [0, 1, 3, 2])
    expand_25: "f32[2, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_33, [2, 12, 64, 1024]);  permute_33 = None
    clone_44: "f32[2, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
    view_116: "f32[24, 64, 1024]" = torch.ops.aten.reshape.default(clone_44, [24, 64, 1024]);  clone_44 = None
    bmm_12: "f32[24, 1024, 1024]" = torch.ops.aten.bmm.default(view_115, view_116);  view_115 = view_116 = None
    view_117: "f32[2, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_12, [2, 12, 1024, 1024]);  bmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_default_12: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    div_12: "f32[2, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_117, full_default_12);  view_117 = full_default_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_13: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_6: "f32[2, 12, 1024, 1024]" = torch.ops.aten.where.self(arg155_1, div_12, full_default_13);  arg155_1 = div_12 = full_default_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_6: "f32[2, 12, 1024, 1]" = torch.ops.aten.amax.default(where_6, [-1], True)
    sub_19: "f32[2, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where_6, amax_6);  where_6 = amax_6 = None
    exp_6: "f32[2, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
    sum_7: "f32[2, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_13: "f32[2, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_26: "f32[2, 12, 1024, 1024]" = torch.ops.aten.expand.default(div_13, [2, 12, 1024, 1024]);  div_13 = None
    view_118: "f32[24, 1024, 1024]" = torch.ops.aten.reshape.default(expand_26, [24, 1024, 1024]);  expand_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_114: "f32[2, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_46, [2, 1024, 12, 64]);  getitem_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_32: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_27: "f32[2, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_32, [2, 12, 1024, 64])
    clone_46: "f32[2, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
    view_119: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(clone_46, [24, 1024, 64]);  clone_46 = None
    bmm_13: "f32[24, 1024, 64]" = torch.ops.aten.bmm.default(view_118, view_119);  view_118 = view_119 = None
    view_120: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_13, [2, 12, 1024, 64]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_34: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
    clone_47: "f32[2, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format);  permute_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_121: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(clone_47, [2, 1024, 768]);  clone_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_122: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_121, [-1, 768]);  view_121 = None
    
    # No stacktrace found for following nodes
    mm_default_18: "f32[2048, 768]" = torch.ops.aten.mm.default(view_122, arg51_1);  view_122 = arg51_1 = None
    add_tensor_17: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_18, arg50_1);  mm_default_18 = arg50_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_123: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_17, [2, 1024, 768]);  add_tensor_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_51: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(view_123, add_48);  view_123 = add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_13 = torch.ops.aten.var_mean.correction(add_51, [2], correction = 0, keepdim = True)
    getitem_47: "f32[2, 1024, 1]" = var_mean_13[0]
    getitem_48: "f32[2, 1024, 1]" = var_mean_13[1];  var_mean_13 = None
    sub_20: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(add_51, getitem_48);  getitem_48 = None
    add_52: "f32[2, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_47, 1e-05);  getitem_47 = None
    rsqrt_13: "f32[2, 1024, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    mul_50: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = rsqrt_13 = None
    mul_51: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_50, arg124_1);  mul_50 = arg124_1 = None
    add_53: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_51, arg125_1);  mul_51 = arg125_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_124: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_53, [-1, 768]);  add_53 = None
    
    # No stacktrace found for following nodes
    mm_default_17: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_124, arg53_1);  view_124 = arg53_1 = None
    add_tensor_16: "f32[2048, 3072]" = torch.ops.aten.add.Tensor(mm_default_17, arg52_1);  mm_default_17 = arg52_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_125: "f32[2, 1024, 3072]" = torch.ops.aten.reshape.default(add_tensor_16, [2, 1024, 3072]);  add_tensor_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_52: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_125, 0.5)
    pow_7: "f32[2, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_125, 3.0)
    mul_53: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
    add_54: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(view_125, mul_53);  view_125 = mul_53 = None
    mul_54: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_54, 0.7978845608028654);  add_54 = None
    tanh_6: "f32[2, 1024, 3072]" = torch.ops.aten.tanh.default(mul_54);  mul_54 = None
    add_55: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
    mul_55: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_52, add_55);  mul_52 = add_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_126: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_55, [-1, 3072]);  mul_55 = None
    
    # No stacktrace found for following nodes
    mm_default_16: "f32[2048, 768]" = torch.ops.aten.mm.default(view_126, arg55_1);  view_126 = arg55_1 = None
    add_tensor_15: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_16, arg54_1);  mm_default_16 = arg54_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_127: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_15, [2, 1024, 768]);  add_tensor_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_56: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(add_51, view_127);  add_51 = view_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_14 = torch.ops.aten.var_mean.correction(add_56, [2], correction = 0, keepdim = True)
    getitem_49: "f32[2, 1024, 1]" = var_mean_14[0]
    getitem_50: "f32[2, 1024, 1]" = var_mean_14[1];  var_mean_14 = None
    sub_21: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(add_56, getitem_50);  getitem_50 = None
    add_57: "f32[2, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_49, 1e-05);  getitem_49 = None
    rsqrt_14: "f32[2, 1024, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
    mul_56: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = rsqrt_14 = None
    mul_57: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_56, arg126_1);  mul_56 = arg126_1 = None
    add_58: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_57, arg127_1);  mul_57 = arg127_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_128: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_58, [-1, 768]);  add_58 = None
    addmm_28: "f32[2048, 2304]" = torch.ops.aten.addmm.default(arg56_1, view_128, arg57_1);  arg56_1 = view_128 = arg57_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_129: "f32[2, 1024, 2304]" = torch.ops.aten.reshape.default(addmm_28, [2, 1024, 2304]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_7 = torch.ops.aten.split_with_sizes.default(view_129, [768, 768, 768], 2);  view_129 = None
    getitem_51: "f32[2, 1024, 768]" = split_with_sizes_7[0]
    getitem_52: "f32[2, 1024, 768]" = split_with_sizes_7[1]
    getitem_53: "f32[2, 1024, 768]" = split_with_sizes_7[2];  split_with_sizes_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_15: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
    device_put_7: "f32[]" = torch.ops.prims.device_put.default(full_15, device(type='cuda', index=0));  full_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_130: "f32[2, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_51, [2, 1024, 12, 64]);  getitem_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_35: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_28: "f32[2, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_35, [2, 12, 1024, 64]);  permute_35 = None
    clone_50: "f32[2, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
    view_133: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(clone_50, [24, 1024, 64]);  clone_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_131: "f32[2, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_52, [2, 1024, 12, 64]);  getitem_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_36: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(view_131, [0, 2, 1, 3]);  view_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_38: "f32[2, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_36, [0, 1, 3, 2])
    expand_29: "f32[2, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_38, [2, 12, 64, 1024]);  permute_38 = None
    clone_51: "f32[2, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
    view_134: "f32[24, 64, 1024]" = torch.ops.aten.reshape.default(clone_51, [24, 64, 1024]);  clone_51 = None
    bmm_14: "f32[24, 1024, 1024]" = torch.ops.aten.bmm.default(view_133, view_134);  view_133 = view_134 = None
    view_135: "f32[2, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_14, [2, 12, 1024, 1024]);  bmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_default_14: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    div_14: "f32[2, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_135, full_default_14);  view_135 = full_default_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_15: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_7: "f32[2, 12, 1024, 1024]" = torch.ops.aten.where.self(arg156_1, div_14, full_default_15);  arg156_1 = div_14 = full_default_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_7: "f32[2, 12, 1024, 1]" = torch.ops.aten.amax.default(where_7, [-1], True)
    sub_22: "f32[2, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where_7, amax_7);  where_7 = amax_7 = None
    exp_7: "f32[2, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
    sum_8: "f32[2, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_15: "f32[2, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_30: "f32[2, 12, 1024, 1024]" = torch.ops.aten.expand.default(div_15, [2, 12, 1024, 1024]);  div_15 = None
    view_136: "f32[24, 1024, 1024]" = torch.ops.aten.reshape.default(expand_30, [24, 1024, 1024]);  expand_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_132: "f32[2, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_53, [2, 1024, 12, 64]);  getitem_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_37: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(view_132, [0, 2, 1, 3]);  view_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_31: "f32[2, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_37, [2, 12, 1024, 64])
    clone_53: "f32[2, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
    view_137: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(clone_53, [24, 1024, 64]);  clone_53 = None
    bmm_15: "f32[24, 1024, 64]" = torch.ops.aten.bmm.default(view_136, view_137);  view_136 = view_137 = None
    view_138: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_15, [2, 12, 1024, 64]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_39: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None
    clone_54: "f32[2, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_39, memory_format = torch.contiguous_format);  permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_139: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(clone_54, [2, 1024, 768]);  clone_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_140: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_139, [-1, 768]);  view_139 = None
    
    # No stacktrace found for following nodes
    mm_default_15: "f32[2048, 768]" = torch.ops.aten.mm.default(view_140, arg59_1);  view_140 = arg59_1 = None
    add_tensor_14: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_15, arg58_1);  mm_default_15 = arg58_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_141: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_14, [2, 1024, 768]);  add_tensor_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_59: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(view_141, add_56);  view_141 = add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_15 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
    getitem_54: "f32[2, 1024, 1]" = var_mean_15[0]
    getitem_55: "f32[2, 1024, 1]" = var_mean_15[1];  var_mean_15 = None
    sub_23: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(add_59, getitem_55);  getitem_55 = None
    add_60: "f32[2, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
    rsqrt_15: "f32[2, 1024, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    mul_58: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = rsqrt_15 = None
    mul_59: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_58, arg128_1);  mul_58 = arg128_1 = None
    add_61: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_59, arg129_1);  mul_59 = arg129_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_142: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_61, [-1, 768]);  add_61 = None
    
    # No stacktrace found for following nodes
    mm_default_14: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_142, arg61_1);  view_142 = arg61_1 = None
    add_tensor_13: "f32[2048, 3072]" = torch.ops.aten.add.Tensor(mm_default_14, arg60_1);  mm_default_14 = arg60_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_143: "f32[2, 1024, 3072]" = torch.ops.aten.reshape.default(add_tensor_13, [2, 1024, 3072]);  add_tensor_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_60: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_143, 0.5)
    pow_8: "f32[2, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_143, 3.0)
    mul_61: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
    add_62: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(view_143, mul_61);  view_143 = mul_61 = None
    mul_62: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_62, 0.7978845608028654);  add_62 = None
    tanh_7: "f32[2, 1024, 3072]" = torch.ops.aten.tanh.default(mul_62);  mul_62 = None
    add_63: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
    mul_63: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_60, add_63);  mul_60 = add_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_144: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_63, [-1, 3072]);  mul_63 = None
    
    # No stacktrace found for following nodes
    mm_default_13: "f32[2048, 768]" = torch.ops.aten.mm.default(view_144, arg63_1);  view_144 = arg63_1 = None
    add_tensor_12: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_13, arg62_1);  mm_default_13 = arg62_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_145: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_12, [2, 1024, 768]);  add_tensor_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_64: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(add_59, view_145);  add_59 = view_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_16 = torch.ops.aten.var_mean.correction(add_64, [2], correction = 0, keepdim = True)
    getitem_56: "f32[2, 1024, 1]" = var_mean_16[0]
    getitem_57: "f32[2, 1024, 1]" = var_mean_16[1];  var_mean_16 = None
    sub_24: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(add_64, getitem_57);  getitem_57 = None
    add_65: "f32[2, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
    rsqrt_16: "f32[2, 1024, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
    mul_64: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = rsqrt_16 = None
    mul_65: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_64, arg130_1);  mul_64 = arg130_1 = None
    add_66: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_65, arg131_1);  mul_65 = arg131_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_146: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_66, [-1, 768]);  add_66 = None
    addmm_32: "f32[2048, 2304]" = torch.ops.aten.addmm.default(arg64_1, view_146, arg65_1);  arg64_1 = view_146 = arg65_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_147: "f32[2, 1024, 2304]" = torch.ops.aten.reshape.default(addmm_32, [2, 1024, 2304]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_8 = torch.ops.aten.split_with_sizes.default(view_147, [768, 768, 768], 2);  view_147 = None
    getitem_58: "f32[2, 1024, 768]" = split_with_sizes_8[0]
    getitem_59: "f32[2, 1024, 768]" = split_with_sizes_8[1]
    getitem_60: "f32[2, 1024, 768]" = split_with_sizes_8[2];  split_with_sizes_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_17: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
    device_put_8: "f32[]" = torch.ops.prims.device_put.default(full_17, device(type='cuda', index=0));  full_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_148: "f32[2, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_58, [2, 1024, 12, 64]);  getitem_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_40: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_32: "f32[2, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_40, [2, 12, 1024, 64]);  permute_40 = None
    clone_57: "f32[2, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
    view_151: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(clone_57, [24, 1024, 64]);  clone_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_149: "f32[2, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_59, [2, 1024, 12, 64]);  getitem_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_41: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(view_149, [0, 2, 1, 3]);  view_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_43: "f32[2, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_41, [0, 1, 3, 2])
    expand_33: "f32[2, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_43, [2, 12, 64, 1024]);  permute_43 = None
    clone_58: "f32[2, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
    view_152: "f32[24, 64, 1024]" = torch.ops.aten.reshape.default(clone_58, [24, 64, 1024]);  clone_58 = None
    bmm_16: "f32[24, 1024, 1024]" = torch.ops.aten.bmm.default(view_151, view_152);  view_151 = view_152 = None
    view_153: "f32[2, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_16, [2, 12, 1024, 1024]);  bmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_default_16: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    div_16: "f32[2, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_153, full_default_16);  view_153 = full_default_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_17: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_8: "f32[2, 12, 1024, 1024]" = torch.ops.aten.where.self(arg157_1, div_16, full_default_17);  arg157_1 = div_16 = full_default_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_8: "f32[2, 12, 1024, 1]" = torch.ops.aten.amax.default(where_8, [-1], True)
    sub_25: "f32[2, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where_8, amax_8);  where_8 = amax_8 = None
    exp_8: "f32[2, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
    sum_9: "f32[2, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_17: "f32[2, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_34: "f32[2, 12, 1024, 1024]" = torch.ops.aten.expand.default(div_17, [2, 12, 1024, 1024]);  div_17 = None
    view_154: "f32[24, 1024, 1024]" = torch.ops.aten.reshape.default(expand_34, [24, 1024, 1024]);  expand_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_150: "f32[2, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_60, [2, 1024, 12, 64]);  getitem_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_42: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(view_150, [0, 2, 1, 3]);  view_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_35: "f32[2, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_42, [2, 12, 1024, 64])
    clone_60: "f32[2, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
    view_155: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(clone_60, [24, 1024, 64]);  clone_60 = None
    bmm_17: "f32[24, 1024, 64]" = torch.ops.aten.bmm.default(view_154, view_155);  view_154 = view_155 = None
    view_156: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_17, [2, 12, 1024, 64]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_44: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None
    clone_61: "f32[2, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_44, memory_format = torch.contiguous_format);  permute_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_157: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(clone_61, [2, 1024, 768]);  clone_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_158: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_157, [-1, 768]);  view_157 = None
    
    # No stacktrace found for following nodes
    mm_default_12: "f32[2048, 768]" = torch.ops.aten.mm.default(view_158, arg67_1);  view_158 = arg67_1 = None
    add_tensor_11: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_12, arg66_1);  mm_default_12 = arg66_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_159: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_11, [2, 1024, 768]);  add_tensor_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_67: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(view_159, add_64);  view_159 = add_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_17 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
    getitem_61: "f32[2, 1024, 1]" = var_mean_17[0]
    getitem_62: "f32[2, 1024, 1]" = var_mean_17[1];  var_mean_17 = None
    sub_26: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(add_67, getitem_62);  getitem_62 = None
    add_68: "f32[2, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_61, 1e-05);  getitem_61 = None
    rsqrt_17: "f32[2, 1024, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
    mul_66: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = rsqrt_17 = None
    mul_67: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_66, arg132_1);  mul_66 = arg132_1 = None
    add_69: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_67, arg133_1);  mul_67 = arg133_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_160: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_69, [-1, 768]);  add_69 = None
    
    # No stacktrace found for following nodes
    mm_default_11: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_160, arg69_1);  view_160 = arg69_1 = None
    add_tensor_10: "f32[2048, 3072]" = torch.ops.aten.add.Tensor(mm_default_11, arg68_1);  mm_default_11 = arg68_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_161: "f32[2, 1024, 3072]" = torch.ops.aten.reshape.default(add_tensor_10, [2, 1024, 3072]);  add_tensor_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_68: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_161, 0.5)
    pow_9: "f32[2, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_161, 3.0)
    mul_69: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
    add_70: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(view_161, mul_69);  view_161 = mul_69 = None
    mul_70: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_70, 0.7978845608028654);  add_70 = None
    tanh_8: "f32[2, 1024, 3072]" = torch.ops.aten.tanh.default(mul_70);  mul_70 = None
    add_71: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
    mul_71: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_68, add_71);  mul_68 = add_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_162: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_71, [-1, 3072]);  mul_71 = None
    
    # No stacktrace found for following nodes
    mm_default_10: "f32[2048, 768]" = torch.ops.aten.mm.default(view_162, arg71_1);  view_162 = arg71_1 = None
    add_tensor_9: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_10, arg70_1);  mm_default_10 = arg70_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_163: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_9, [2, 1024, 768]);  add_tensor_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_72: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(add_67, view_163);  add_67 = view_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_18 = torch.ops.aten.var_mean.correction(add_72, [2], correction = 0, keepdim = True)
    getitem_63: "f32[2, 1024, 1]" = var_mean_18[0]
    getitem_64: "f32[2, 1024, 1]" = var_mean_18[1];  var_mean_18 = None
    sub_27: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(add_72, getitem_64);  getitem_64 = None
    add_73: "f32[2, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_63, 1e-05);  getitem_63 = None
    rsqrt_18: "f32[2, 1024, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
    mul_72: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = rsqrt_18 = None
    mul_73: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_72, arg134_1);  mul_72 = arg134_1 = None
    add_74: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_73, arg135_1);  mul_73 = arg135_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_164: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_74, [-1, 768]);  add_74 = None
    addmm_36: "f32[2048, 2304]" = torch.ops.aten.addmm.default(arg72_1, view_164, arg73_1);  arg72_1 = view_164 = arg73_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_165: "f32[2, 1024, 2304]" = torch.ops.aten.reshape.default(addmm_36, [2, 1024, 2304]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_9 = torch.ops.aten.split_with_sizes.default(view_165, [768, 768, 768], 2);  view_165 = None
    getitem_65: "f32[2, 1024, 768]" = split_with_sizes_9[0]
    getitem_66: "f32[2, 1024, 768]" = split_with_sizes_9[1]
    getitem_67: "f32[2, 1024, 768]" = split_with_sizes_9[2];  split_with_sizes_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_19: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
    device_put_9: "f32[]" = torch.ops.prims.device_put.default(full_19, device(type='cuda', index=0));  full_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_166: "f32[2, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_65, [2, 1024, 12, 64]);  getitem_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_45: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(view_166, [0, 2, 1, 3]);  view_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_36: "f32[2, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_45, [2, 12, 1024, 64]);  permute_45 = None
    clone_64: "f32[2, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
    view_169: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(clone_64, [24, 1024, 64]);  clone_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_167: "f32[2, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_66, [2, 1024, 12, 64]);  getitem_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_46: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(view_167, [0, 2, 1, 3]);  view_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_48: "f32[2, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_46, [0, 1, 3, 2])
    expand_37: "f32[2, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_48, [2, 12, 64, 1024]);  permute_48 = None
    clone_65: "f32[2, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
    view_170: "f32[24, 64, 1024]" = torch.ops.aten.reshape.default(clone_65, [24, 64, 1024]);  clone_65 = None
    bmm_18: "f32[24, 1024, 1024]" = torch.ops.aten.bmm.default(view_169, view_170);  view_169 = view_170 = None
    view_171: "f32[2, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_18, [2, 12, 1024, 1024]);  bmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_default_18: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    div_18: "f32[2, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_171, full_default_18);  view_171 = full_default_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_19: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_9: "f32[2, 12, 1024, 1024]" = torch.ops.aten.where.self(arg158_1, div_18, full_default_19);  arg158_1 = div_18 = full_default_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_9: "f32[2, 12, 1024, 1]" = torch.ops.aten.amax.default(where_9, [-1], True)
    sub_28: "f32[2, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where_9, amax_9);  where_9 = amax_9 = None
    exp_9: "f32[2, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
    sum_10: "f32[2, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_19: "f32[2, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_38: "f32[2, 12, 1024, 1024]" = torch.ops.aten.expand.default(div_19, [2, 12, 1024, 1024]);  div_19 = None
    view_172: "f32[24, 1024, 1024]" = torch.ops.aten.reshape.default(expand_38, [24, 1024, 1024]);  expand_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_168: "f32[2, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_67, [2, 1024, 12, 64]);  getitem_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_47: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_39: "f32[2, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_47, [2, 12, 1024, 64])
    clone_67: "f32[2, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
    view_173: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(clone_67, [24, 1024, 64]);  clone_67 = None
    bmm_19: "f32[24, 1024, 64]" = torch.ops.aten.bmm.default(view_172, view_173);  view_172 = view_173 = None
    view_174: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_19, [2, 12, 1024, 64]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_49: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_174, [0, 2, 1, 3]);  view_174 = None
    clone_68: "f32[2, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_175: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(clone_68, [2, 1024, 768]);  clone_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_176: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_175, [-1, 768]);  view_175 = None
    
    # No stacktrace found for following nodes
    mm_default_9: "f32[2048, 768]" = torch.ops.aten.mm.default(view_176, arg75_1);  view_176 = arg75_1 = None
    add_tensor_8: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_9, arg74_1);  mm_default_9 = arg74_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_177: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_8, [2, 1024, 768]);  add_tensor_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_75: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(view_177, add_72);  view_177 = add_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_19 = torch.ops.aten.var_mean.correction(add_75, [2], correction = 0, keepdim = True)
    getitem_68: "f32[2, 1024, 1]" = var_mean_19[0]
    getitem_69: "f32[2, 1024, 1]" = var_mean_19[1];  var_mean_19 = None
    sub_29: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(add_75, getitem_69);  getitem_69 = None
    add_76: "f32[2, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05);  getitem_68 = None
    rsqrt_19: "f32[2, 1024, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
    mul_74: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = rsqrt_19 = None
    mul_75: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_74, arg136_1);  mul_74 = arg136_1 = None
    add_77: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_75, arg137_1);  mul_75 = arg137_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_178: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_77, [-1, 768]);  add_77 = None
    
    # No stacktrace found for following nodes
    mm_default_8: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_178, arg77_1);  view_178 = arg77_1 = None
    add_tensor_7: "f32[2048, 3072]" = torch.ops.aten.add.Tensor(mm_default_8, arg76_1);  mm_default_8 = arg76_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_179: "f32[2, 1024, 3072]" = torch.ops.aten.reshape.default(add_tensor_7, [2, 1024, 3072]);  add_tensor_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_76: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_179, 0.5)
    pow_10: "f32[2, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_179, 3.0)
    mul_77: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
    add_78: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(view_179, mul_77);  view_179 = mul_77 = None
    mul_78: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_78, 0.7978845608028654);  add_78 = None
    tanh_9: "f32[2, 1024, 3072]" = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
    add_79: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
    mul_79: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_76, add_79);  mul_76 = add_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_180: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_79, [-1, 3072]);  mul_79 = None
    
    # No stacktrace found for following nodes
    mm_default_7: "f32[2048, 768]" = torch.ops.aten.mm.default(view_180, arg79_1);  view_180 = arg79_1 = None
    add_tensor_6: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_7, arg78_1);  mm_default_7 = arg78_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_181: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_6, [2, 1024, 768]);  add_tensor_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_80: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(add_75, view_181);  add_75 = view_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_20 = torch.ops.aten.var_mean.correction(add_80, [2], correction = 0, keepdim = True)
    getitem_70: "f32[2, 1024, 1]" = var_mean_20[0]
    getitem_71: "f32[2, 1024, 1]" = var_mean_20[1];  var_mean_20 = None
    sub_30: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(add_80, getitem_71);  getitem_71 = None
    add_81: "f32[2, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05);  getitem_70 = None
    rsqrt_20: "f32[2, 1024, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
    mul_80: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = rsqrt_20 = None
    mul_81: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_80, arg138_1);  mul_80 = arg138_1 = None
    add_82: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_81, arg139_1);  mul_81 = arg139_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_182: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_82, [-1, 768]);  add_82 = None
    addmm_40: "f32[2048, 2304]" = torch.ops.aten.addmm.default(arg80_1, view_182, arg81_1);  arg80_1 = view_182 = arg81_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_183: "f32[2, 1024, 2304]" = torch.ops.aten.reshape.default(addmm_40, [2, 1024, 2304]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_10 = torch.ops.aten.split_with_sizes.default(view_183, [768, 768, 768], 2);  view_183 = None
    getitem_72: "f32[2, 1024, 768]" = split_with_sizes_10[0]
    getitem_73: "f32[2, 1024, 768]" = split_with_sizes_10[1]
    getitem_74: "f32[2, 1024, 768]" = split_with_sizes_10[2];  split_with_sizes_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_21: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
    device_put_10: "f32[]" = torch.ops.prims.device_put.default(full_21, device(type='cuda', index=0));  full_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_184: "f32[2, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_72, [2, 1024, 12, 64]);  getitem_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_50: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_40: "f32[2, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_50, [2, 12, 1024, 64]);  permute_50 = None
    clone_71: "f32[2, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
    view_187: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(clone_71, [24, 1024, 64]);  clone_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_185: "f32[2, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_73, [2, 1024, 12, 64]);  getitem_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_51: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_53: "f32[2, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_51, [0, 1, 3, 2])
    expand_41: "f32[2, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_53, [2, 12, 64, 1024]);  permute_53 = None
    clone_72: "f32[2, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
    view_188: "f32[24, 64, 1024]" = torch.ops.aten.reshape.default(clone_72, [24, 64, 1024]);  clone_72 = None
    bmm_20: "f32[24, 1024, 1024]" = torch.ops.aten.bmm.default(view_187, view_188);  view_187 = view_188 = None
    view_189: "f32[2, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_20, [2, 12, 1024, 1024]);  bmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_default_20: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    div_20: "f32[2, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_189, full_default_20);  view_189 = full_default_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_21: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_10: "f32[2, 12, 1024, 1024]" = torch.ops.aten.where.self(arg159_1, div_20, full_default_21);  arg159_1 = div_20 = full_default_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_10: "f32[2, 12, 1024, 1]" = torch.ops.aten.amax.default(where_10, [-1], True)
    sub_31: "f32[2, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where_10, amax_10);  where_10 = amax_10 = None
    exp_10: "f32[2, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
    sum_11: "f32[2, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_21: "f32[2, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_42: "f32[2, 12, 1024, 1024]" = torch.ops.aten.expand.default(div_21, [2, 12, 1024, 1024]);  div_21 = None
    view_190: "f32[24, 1024, 1024]" = torch.ops.aten.reshape.default(expand_42, [24, 1024, 1024]);  expand_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_186: "f32[2, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_74, [2, 1024, 12, 64]);  getitem_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_52: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_43: "f32[2, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_52, [2, 12, 1024, 64])
    clone_74: "f32[2, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
    view_191: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(clone_74, [24, 1024, 64]);  clone_74 = None
    bmm_21: "f32[24, 1024, 64]" = torch.ops.aten.bmm.default(view_190, view_191);  view_190 = view_191 = None
    view_192: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_21, [2, 12, 1024, 64]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_54: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None
    clone_75: "f32[2, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_54, memory_format = torch.contiguous_format);  permute_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_193: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(clone_75, [2, 1024, 768]);  clone_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_194: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_193, [-1, 768]);  view_193 = None
    
    # No stacktrace found for following nodes
    mm_default_6: "f32[2048, 768]" = torch.ops.aten.mm.default(view_194, arg83_1);  view_194 = arg83_1 = None
    add_tensor_5: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_6, arg82_1);  mm_default_6 = arg82_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_195: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_5, [2, 1024, 768]);  add_tensor_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_83: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(view_195, add_80);  view_195 = add_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_21 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
    getitem_75: "f32[2, 1024, 1]" = var_mean_21[0]
    getitem_76: "f32[2, 1024, 1]" = var_mean_21[1];  var_mean_21 = None
    sub_32: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(add_83, getitem_76);  getitem_76 = None
    add_84: "f32[2, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_75, 1e-05);  getitem_75 = None
    rsqrt_21: "f32[2, 1024, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
    mul_82: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = rsqrt_21 = None
    mul_83: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_82, arg140_1);  mul_82 = arg140_1 = None
    add_85: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_83, arg141_1);  mul_83 = arg141_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_196: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_85, [-1, 768]);  add_85 = None
    
    # No stacktrace found for following nodes
    mm_default_5: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_196, arg85_1);  view_196 = arg85_1 = None
    add_tensor_4: "f32[2048, 3072]" = torch.ops.aten.add.Tensor(mm_default_5, arg84_1);  mm_default_5 = arg84_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_197: "f32[2, 1024, 3072]" = torch.ops.aten.reshape.default(add_tensor_4, [2, 1024, 3072]);  add_tensor_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_84: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_197, 0.5)
    pow_11: "f32[2, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_197, 3.0)
    mul_85: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
    add_86: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(view_197, mul_85);  view_197 = mul_85 = None
    mul_86: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_86, 0.7978845608028654);  add_86 = None
    tanh_10: "f32[2, 1024, 3072]" = torch.ops.aten.tanh.default(mul_86);  mul_86 = None
    add_87: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
    mul_87: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_84, add_87);  mul_84 = add_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_198: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_87, [-1, 3072]);  mul_87 = None
    
    # No stacktrace found for following nodes
    mm_default_4: "f32[2048, 768]" = torch.ops.aten.mm.default(view_198, arg87_1);  view_198 = arg87_1 = None
    add_tensor_3: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_4, arg86_1);  mm_default_4 = arg86_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_199: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_3, [2, 1024, 768]);  add_tensor_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_88: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(add_83, view_199);  add_83 = view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_22 = torch.ops.aten.var_mean.correction(add_88, [2], correction = 0, keepdim = True)
    getitem_77: "f32[2, 1024, 1]" = var_mean_22[0]
    getitem_78: "f32[2, 1024, 1]" = var_mean_22[1];  var_mean_22 = None
    sub_33: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(add_88, getitem_78);  getitem_78 = None
    add_89: "f32[2, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_77, 1e-05);  getitem_77 = None
    rsqrt_22: "f32[2, 1024, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
    mul_88: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = rsqrt_22 = None
    mul_89: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_88, arg142_1);  mul_88 = arg142_1 = None
    add_90: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_89, arg143_1);  mul_89 = arg143_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_200: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_90, [-1, 768]);  add_90 = None
    addmm_44: "f32[2048, 2304]" = torch.ops.aten.addmm.default(arg88_1, view_200, arg89_1);  arg88_1 = view_200 = arg89_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_201: "f32[2, 1024, 2304]" = torch.ops.aten.reshape.default(addmm_44, [2, 1024, 2304]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_11 = torch.ops.aten.split_with_sizes.default(view_201, [768, 768, 768], 2);  view_201 = None
    getitem_79: "f32[2, 1024, 768]" = split_with_sizes_11[0]
    getitem_80: "f32[2, 1024, 768]" = split_with_sizes_11[1]
    getitem_81: "f32[2, 1024, 768]" = split_with_sizes_11[2];  split_with_sizes_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_23: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
    device_put_11: "f32[]" = torch.ops.prims.device_put.default(full_23, device(type='cuda', index=0));  full_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_202: "f32[2, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_79, [2, 1024, 12, 64]);  getitem_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_55: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(view_202, [0, 2, 1, 3]);  view_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_44: "f32[2, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_55, [2, 12, 1024, 64]);  permute_55 = None
    clone_78: "f32[2, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
    view_205: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(clone_78, [24, 1024, 64]);  clone_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_203: "f32[2, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_80, [2, 1024, 12, 64]);  getitem_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_56: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(view_203, [0, 2, 1, 3]);  view_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_58: "f32[2, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_56, [0, 1, 3, 2])
    expand_45: "f32[2, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_58, [2, 12, 64, 1024]);  permute_58 = None
    clone_79: "f32[2, 12, 64, 1024]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
    view_206: "f32[24, 64, 1024]" = torch.ops.aten.reshape.default(clone_79, [24, 64, 1024]);  clone_79 = None
    bmm_22: "f32[24, 1024, 1024]" = torch.ops.aten.bmm.default(view_205, view_206);  view_205 = view_206 = None
    view_207: "f32[2, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_22, [2, 12, 1024, 1024]);  bmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_default_22: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    div_22: "f32[2, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_207, full_default_22);  view_207 = full_default_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_23: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_11: "f32[2, 12, 1024, 1024]" = torch.ops.aten.where.self(arg160_1, div_22, full_default_23);  arg160_1 = div_22 = full_default_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_11: "f32[2, 12, 1024, 1]" = torch.ops.aten.amax.default(where_11, [-1], True)
    sub_34: "f32[2, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where_11, amax_11);  where_11 = amax_11 = None
    exp_11: "f32[2, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
    sum_12: "f32[2, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_23: "f32[2, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_46: "f32[2, 12, 1024, 1024]" = torch.ops.aten.expand.default(div_23, [2, 12, 1024, 1024]);  div_23 = None
    view_208: "f32[24, 1024, 1024]" = torch.ops.aten.reshape.default(expand_46, [24, 1024, 1024]);  expand_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_204: "f32[2, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_81, [2, 1024, 12, 64]);  getitem_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_57: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(view_204, [0, 2, 1, 3]);  view_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_47: "f32[2, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_57, [2, 12, 1024, 64])
    clone_81: "f32[2, 12, 1024, 64]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
    view_209: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(clone_81, [24, 1024, 64]);  clone_81 = None
    bmm_23: "f32[24, 1024, 64]" = torch.ops.aten.bmm.default(view_208, view_209);  view_208 = view_209 = None
    view_210: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_23, [2, 12, 1024, 64]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_59: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_210, [0, 2, 1, 3]);  view_210 = None
    clone_82: "f32[2, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_211: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(clone_82, [2, 1024, 768]);  clone_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_212: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_211, [-1, 768]);  view_211 = None
    
    # No stacktrace found for following nodes
    mm_default_3: "f32[2048, 768]" = torch.ops.aten.mm.default(view_212, arg91_1);  view_212 = arg91_1 = None
    add_tensor_2: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_3, arg90_1);  mm_default_3 = arg90_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_213: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_2, [2, 1024, 768]);  add_tensor_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_91: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(view_213, add_88);  view_213 = add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_23 = torch.ops.aten.var_mean.correction(add_91, [2], correction = 0, keepdim = True)
    getitem_82: "f32[2, 1024, 1]" = var_mean_23[0]
    getitem_83: "f32[2, 1024, 1]" = var_mean_23[1];  var_mean_23 = None
    sub_35: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(add_91, getitem_83);  getitem_83 = None
    add_92: "f32[2, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
    rsqrt_23: "f32[2, 1024, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
    mul_90: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = rsqrt_23 = None
    mul_91: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_90, arg144_1);  mul_90 = arg144_1 = None
    add_93: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_91, arg145_1);  mul_91 = arg145_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_214: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_93, [-1, 768]);  add_93 = None
    
    # No stacktrace found for following nodes
    mm_default_2: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_214, arg93_1);  view_214 = arg93_1 = None
    add_tensor_1: "f32[2048, 3072]" = torch.ops.aten.add.Tensor(mm_default_2, arg92_1);  mm_default_2 = arg92_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_215: "f32[2, 1024, 3072]" = torch.ops.aten.reshape.default(add_tensor_1, [2, 1024, 3072]);  add_tensor_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_92: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_215, 0.5)
    pow_12: "f32[2, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_215, 3.0)
    mul_93: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
    add_94: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(view_215, mul_93);  view_215 = mul_93 = None
    mul_94: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_94, 0.7978845608028654);  add_94 = None
    tanh_11: "f32[2, 1024, 3072]" = torch.ops.aten.tanh.default(mul_94);  mul_94 = None
    add_95: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
    mul_95: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_92, add_95);  mul_92 = add_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_216: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_95, [-1, 3072]);  mul_95 = None
    
    # No stacktrace found for following nodes
    mm_default_1: "f32[2048, 768]" = torch.ops.aten.mm.default(view_216, arg95_1);  view_216 = arg95_1 = None
    add_tensor: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_1, arg94_1);  mm_default_1 = arg94_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_217: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor, [2, 1024, 768]);  add_tensor = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_96: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(add_91, view_217);  add_91 = view_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:926, code: hidden_states = self.ln_f(hidden_states)
    var_mean_24 = torch.ops.aten.var_mean.correction(add_96, [2], correction = 0, keepdim = True)
    getitem_84: "f32[2, 1024, 1]" = var_mean_24[0]
    getitem_85: "f32[2, 1024, 1]" = var_mean_24[1];  var_mean_24 = None
    sub_36: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(add_96, getitem_85);  add_96 = getitem_85 = None
    add_97: "f32[2, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05);  getitem_84 = None
    rsqrt_24: "f32[2, 1024, 1]" = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
    mul_96: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = rsqrt_24 = None
    mul_97: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_96, arg146_1);  mul_96 = arg146_1 = None
    add_98: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_97, arg147_1);  mul_97 = arg147_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:928, code: hidden_states = hidden_states.view(output_shape)
    view_218: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(add_98, [-1, 1024, 768]);  add_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1098, code: lm_logits = self.lm_head(hidden_states)
    view_219: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_218, [2048, 768]);  view_218 = None
    permute_60: "f32[768, 50257]" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
    
    # No stacktrace found for following nodes
    full_default_24: "f32[768, 3]" = torch.ops.aten.full.default([768, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    cat_default: "f32[768, 50260]" = torch.ops.aten.cat.default([permute_60, full_default_24], 1);  permute_60 = full_default_24 = None
    mm_default: "f32[2048, 50260]" = torch.ops.aten.mm.default(view_219, cat_default);  view_219 = cat_default = None
    slice_tensor: "f32[2048, 50260]" = torch.ops.aten.slice.Tensor(mm_default, 0, 0, 9223372036854775807);  mm_default = None
    slice_tensor_1: "f32[2048, 50257]" = torch.ops.aten.slice.Tensor(slice_tensor, 1, 0, -3);  slice_tensor = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1098, code: lm_logits = self.lm_head(hidden_states)
    view_220: "f32[2, 1024, 50257]" = torch.ops.aten.reshape.default(slice_tensor_1, [2, 1024, 50257]);  slice_tensor_1 = None
    return (view_220, permute_1, permute_2, permute_6, permute_7, permute_11, permute_12, permute_16, permute_17, permute_21, permute_22, permute_26, permute_27, permute_31, permute_32, permute_36, permute_37, permute_41, permute_42, permute_46, permute_47, permute_51, permute_52, permute_56, permute_57)
    