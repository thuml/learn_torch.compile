from __future__ import annotations



def forward(self, arg0_1: "f32[2304]", arg1_1: "f32[768, 2304]", arg2_1: "f32[768]", arg3_1: "f32[768, 768]", arg4_1: "f32[3072]", arg5_1: "f32[768, 3072]", arg6_1: "f32[768]", arg7_1: "f32[3072, 768]", arg8_1: "f32[2304]", arg9_1: "f32[768, 2304]", arg10_1: "f32[768]", arg11_1: "f32[768, 768]", arg12_1: "f32[3072]", arg13_1: "f32[768, 3072]", arg14_1: "f32[768]", arg15_1: "f32[3072, 768]", arg16_1: "f32[2304]", arg17_1: "f32[768, 2304]", arg18_1: "f32[768]", arg19_1: "f32[768, 768]", arg20_1: "f32[3072]", arg21_1: "f32[768, 3072]", arg22_1: "f32[768]", arg23_1: "f32[3072, 768]", arg24_1: "f32[2304]", arg25_1: "f32[768, 2304]", arg26_1: "f32[768]", arg27_1: "f32[768, 768]", arg28_1: "f32[3072]", arg29_1: "f32[768, 3072]", arg30_1: "f32[768]", arg31_1: "f32[3072, 768]", arg32_1: "f32[2304]", arg33_1: "f32[768, 2304]", arg34_1: "f32[768]", arg35_1: "f32[768, 768]", arg36_1: "f32[3072]", arg37_1: "f32[768, 3072]", arg38_1: "f32[768]", arg39_1: "f32[3072, 768]", arg40_1: "f32[2304]", arg41_1: "f32[768, 2304]", arg42_1: "f32[768]", arg43_1: "f32[768, 768]", arg44_1: "f32[3072]", arg45_1: "f32[768, 3072]", arg46_1: "f32[768]", arg47_1: "f32[3072, 768]", arg48_1: "f32[2304]", arg49_1: "f32[768, 2304]", arg50_1: "f32[768]", arg51_1: "f32[768, 768]", arg52_1: "f32[3072]", arg53_1: "f32[768, 3072]", arg54_1: "f32[768]", arg55_1: "f32[3072, 768]", arg56_1: "f32[2304]", arg57_1: "f32[768, 2304]", arg58_1: "f32[768]", arg59_1: "f32[768, 768]", arg60_1: "f32[3072]", arg61_1: "f32[768, 3072]", arg62_1: "f32[768]", arg63_1: "f32[3072, 768]", arg64_1: "f32[2304]", arg65_1: "f32[768, 2304]", arg66_1: "f32[768]", arg67_1: "f32[768, 768]", arg68_1: "f32[3072]", arg69_1: "f32[768, 3072]", arg70_1: "f32[768]", arg71_1: "f32[3072, 768]", arg72_1: "f32[2304]", arg73_1: "f32[768, 2304]", arg74_1: "f32[768]", arg75_1: "f32[768, 768]", arg76_1: "f32[3072]", arg77_1: "f32[768, 3072]", arg78_1: "f32[768]", arg79_1: "f32[3072, 768]", arg80_1: "f32[2304]", arg81_1: "f32[768, 2304]", arg82_1: "f32[768]", arg83_1: "f32[768, 768]", arg84_1: "f32[3072]", arg85_1: "f32[768, 3072]", arg86_1: "f32[768]", arg87_1: "f32[3072, 768]", arg88_1: "f32[2304]", arg89_1: "f32[768, 2304]", arg90_1: "f32[768]", arg91_1: "f32[768, 768]", arg92_1: "f32[3072]", arg93_1: "f32[768, 3072]", arg94_1: "f32[768]", arg95_1: "f32[3072, 768]", arg96_1: "f32[50257, 768]", arg97_1: "f32[1024, 768]", arg98_1: "f32[768]", arg99_1: "f32[768]", arg100_1: "f32[768]", arg101_1: "f32[768]", arg102_1: "f32[768]", arg103_1: "f32[768]", arg104_1: "f32[768]", arg105_1: "f32[768]", arg106_1: "f32[768]", arg107_1: "f32[768]", arg108_1: "f32[768]", arg109_1: "f32[768]", arg110_1: "f32[768]", arg111_1: "f32[768]", arg112_1: "f32[768]", arg113_1: "f32[768]", arg114_1: "f32[768]", arg115_1: "f32[768]", arg116_1: "f32[768]", arg117_1: "f32[768]", arg118_1: "f32[768]", arg119_1: "f32[768]", arg120_1: "f32[768]", arg121_1: "f32[768]", arg122_1: "f32[768]", arg123_1: "f32[768]", arg124_1: "f32[768]", arg125_1: "f32[768]", arg126_1: "f32[768]", arg127_1: "f32[768]", arg128_1: "f32[768]", arg129_1: "f32[768]", arg130_1: "f32[768]", arg131_1: "f32[768]", arg132_1: "f32[768]", arg133_1: "f32[768]", arg134_1: "f32[768]", arg135_1: "f32[768]", arg136_1: "f32[768]", arg137_1: "f32[768]", arg138_1: "f32[768]", arg139_1: "f32[768]", arg140_1: "f32[768]", arg141_1: "f32[768]", arg142_1: "f32[768]", arg143_1: "f32[768]", arg144_1: "f32[768]", arg145_1: "f32[768]", arg146_1: "f32[768]", arg147_1: "f32[768]", arg148_1: "f32[2, 768]", arg149_1: "b8[1, 1, 1024, 1024]", arg150_1: "b8[1, 1, 1024, 1024]", arg151_1: "b8[1, 1, 1024, 1024]", arg152_1: "b8[1, 1, 1024, 1024]", arg153_1: "b8[1, 1, 1024, 1024]", arg154_1: "b8[1, 1, 1024, 1024]", arg155_1: "b8[1, 1, 1024, 1024]", arg156_1: "b8[1, 1, 1024, 1024]", arg157_1: "b8[1, 1, 1024, 1024]", arg158_1: "b8[1, 1, 1024, 1024]", arg159_1: "b8[1, 1, 1024, 1024]", arg160_1: "b8[1, 1, 1024, 1024]", arg161_1: "i64[1, 1024]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:781, code: input_ids = input_ids.view(-1, input_shape[-1])
    view: "i64[1, 1024]" = torch.ops.aten.view.default(arg161_1, [-1, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:802, code: position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
    iota: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:803, code: position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
    unsqueeze: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
    view_1: "i64[1, 1024]" = torch.ops.aten.view.default(unsqueeze, [-1, 1024]);  unsqueeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:843, code: inputs_embeds = self.wte(input_ids)
    embedding: "f32[1, 1024, 768]" = torch.ops.aten.embedding.default(arg96_1, view);  arg96_1 = view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:844, code: position_embeds = self.wpe(position_ids)
    embedding_1: "f32[1, 1024, 768]" = torch.ops.aten.embedding.default(arg97_1, view_1);  arg97_1 = view_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:845, code: hidden_states = inputs_embeds + position_embeds
    add: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:851, code: hidden_states = self.drop(hidden_states)
    clone: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(add);  add = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean = torch.ops.aten.var_mean.correction(clone, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 1024, 1]" = var_mean[0]
    getitem_1: "f32[1, 1024, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(clone, getitem_1);  getitem_1 = None
    mul: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
    mul_1: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul, arg98_1);  mul = arg98_1 = None
    add_2: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_1, arg99_1);  mul_1 = arg99_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_2: "f32[1024, 768]" = torch.ops.aten.view.default(add_2, [-1, 768]);  add_2 = None
    addmm: "f32[1024, 2304]" = torch.ops.aten.addmm.default(arg0_1, view_2, arg1_1);  arg0_1 = view_2 = arg1_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_3: "f32[1, 1024, 2304]" = torch.ops.aten.view.default(addmm, [1, 1024, 2304]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes = torch.ops.aten.split_with_sizes.default(view_3, [768, 768, 768], 2);  view_3 = None
    getitem_2: "f32[1, 1024, 768]" = split_with_sizes[0]
    getitem_3: "f32[1, 1024, 768]" = split_with_sizes[1]
    getitem_4: "f32[1, 1024, 768]" = split_with_sizes[2];  split_with_sizes = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_4: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_2, [1, 1024, 12, 64]);  getitem_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_5: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_3, [1, 1024, 12, 64]);  getitem_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_1: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_6: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_4, [1, 1024, 12, 64]);  getitem_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_2: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_3: "f32[1, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_1, [0, 1, 3, 2])
    expand: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute, [1, 12, 1024, 64]);  permute = None
    view_7: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand, [12, 1024, 64]);  expand = None
    expand_1: "f32[1, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_3, [1, 12, 64, 1024]);  permute_3 = None
    view_8: "f32[12, 64, 1024]" = torch.ops.aten.view.default(expand_1, [12, 64, 1024]);  expand_1 = None
    bmm: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_7, view_8);  view_7 = view_8 = None
    view_9: "f32[1, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm, [1, 12, 1024, 1024]);  bmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    div: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_9, full);  view_9 = full = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_1: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(arg149_1, 0, 0, 9223372036854775807);  arg149_1 = None
    slice_2: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 9223372036854775807);  slice_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_1: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_2, div, full_1);  slice_2 = div = full_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax: "f32[1, 12, 1024, 1]" = torch.ops.aten.amax.default(where, [-1], True)
    sub_1: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where, amax);  where = amax = None
    exp: "f32[1, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
    sum_1: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div_1: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    clone_1: "f32[1, 12, 1024, 1024]" = torch.ops.aten.clone.default(div_1);  div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_2: "f32[1, 12, 1024, 1024]" = torch.ops.aten.expand.default(clone_1, [1, 12, 1024, 1024]);  clone_1 = None
    view_10: "f32[12, 1024, 1024]" = torch.ops.aten.view.default(expand_2, [12, 1024, 1024]);  expand_2 = None
    expand_3: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_2, [1, 12, 1024, 64])
    view_11: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_3, [12, 1024, 64]);  expand_3 = None
    bmm_1: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_10, view_11);  view_10 = view_11 = None
    view_12: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_1, [1, 12, 1024, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_4: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None
    clone_2: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_13: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_2, [1, 1024, 768]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_14: "f32[1024, 768]" = torch.ops.aten.view.default(view_13, [-1, 768]);  view_13 = None
    addmm_1: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg2_1, view_14, arg3_1);  arg2_1 = view_14 = arg3_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_15: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_1, [1, 1024, 768]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    clone_3: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_15);  view_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_3: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(clone_3, clone);  clone_3 = clone = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
    getitem_5: "f32[1, 1024, 1]" = var_mean_1[0]
    getitem_6: "f32[1, 1024, 1]" = var_mean_1[1];  var_mean_1 = None
    add_4: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_5, 1e-05);  getitem_5 = None
    rsqrt_1: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    sub_2: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_3, getitem_6);  getitem_6 = None
    mul_2: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
    mul_3: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2, arg100_1);  mul_2 = arg100_1 = None
    add_5: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_3, arg101_1);  mul_3 = arg101_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_16: "f32[1024, 768]" = torch.ops.aten.view.default(add_5, [-1, 768]);  add_5 = None
    addmm_2: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg4_1, view_16, arg5_1);  arg4_1 = view_16 = arg5_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_17: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_2, [1, 1024, 3072]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_4: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_17, 0.5)
    pow_1: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_17, 3.0)
    mul_5: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
    add_6: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(view_17, mul_5);  view_17 = mul_5 = None
    mul_6: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_6, 0.7978845608028654);  add_6 = None
    tanh: "f32[1, 1024, 3072]" = torch.ops.aten.tanh.default(mul_6);  mul_6 = None
    add_7: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
    mul_7: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_4, add_7);  mul_4 = add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_18: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_7, [-1, 3072]);  mul_7 = None
    addmm_3: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg6_1, view_18, arg7_1);  arg6_1 = view_18 = arg7_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_19: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_3, [1, 1024, 768]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    clone_4: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_19);  view_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_8: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_3, clone_4);  add_3 = clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
    getitem_7: "f32[1, 1024, 1]" = var_mean_2[0]
    getitem_8: "f32[1, 1024, 1]" = var_mean_2[1];  var_mean_2 = None
    add_9: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_7, 1e-05);  getitem_7 = None
    rsqrt_2: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
    sub_3: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_8, getitem_8);  getitem_8 = None
    mul_8: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
    mul_9: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_8, arg102_1);  mul_8 = arg102_1 = None
    add_10: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_9, arg103_1);  mul_9 = arg103_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_20: "f32[1024, 768]" = torch.ops.aten.view.default(add_10, [-1, 768]);  add_10 = None
    addmm_4: "f32[1024, 2304]" = torch.ops.aten.addmm.default(arg8_1, view_20, arg9_1);  arg8_1 = view_20 = arg9_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_21: "f32[1, 1024, 2304]" = torch.ops.aten.view.default(addmm_4, [1, 1024, 2304]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(view_21, [768, 768, 768], 2);  view_21 = None
    getitem_9: "f32[1, 1024, 768]" = split_with_sizes_1[0]
    getitem_10: "f32[1, 1024, 768]" = split_with_sizes_1[1]
    getitem_11: "f32[1, 1024, 768]" = split_with_sizes_1[2];  split_with_sizes_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_22: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_9, [1, 1024, 12, 64]);  getitem_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_5: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_22, [0, 2, 1, 3]);  view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_23: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_10, [1, 1024, 12, 64]);  getitem_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_6: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_23, [0, 2, 1, 3]);  view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_24: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_11, [1, 1024, 12, 64]);  getitem_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_7: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_8: "f32[1, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_6, [0, 1, 3, 2])
    expand_4: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_5, [1, 12, 1024, 64]);  permute_5 = None
    view_25: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_4, [12, 1024, 64]);  expand_4 = None
    expand_5: "f32[1, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_8, [1, 12, 64, 1024]);  permute_8 = None
    view_26: "f32[12, 64, 1024]" = torch.ops.aten.view.default(expand_5, [12, 64, 1024]);  expand_5 = None
    bmm_2: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_25, view_26);  view_25 = view_26 = None
    view_27: "f32[1, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_2, [1, 12, 1024, 1024]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_2: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    div_2: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_27, full_2);  view_27 = full_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_3: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(arg150_1, 0, 0, 9223372036854775807);  arg150_1 = None
    slice_4: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_3, 1, 0, 9223372036854775807);  slice_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_3: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_1: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_4, div_2, full_3);  slice_4 = div_2 = full_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_1: "f32[1, 12, 1024, 1]" = torch.ops.aten.amax.default(where_1, [-1], True)
    sub_4: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where_1, amax_1);  where_1 = amax_1 = None
    exp_1: "f32[1, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
    sum_2: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_3: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    clone_5: "f32[1, 12, 1024, 1024]" = torch.ops.aten.clone.default(div_3);  div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_6: "f32[1, 12, 1024, 1024]" = torch.ops.aten.expand.default(clone_5, [1, 12, 1024, 1024]);  clone_5 = None
    view_28: "f32[12, 1024, 1024]" = torch.ops.aten.view.default(expand_6, [12, 1024, 1024]);  expand_6 = None
    expand_7: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_7, [1, 12, 1024, 64])
    view_29: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_7, [12, 1024, 64]);  expand_7 = None
    bmm_3: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_28, view_29);  view_28 = view_29 = None
    view_30: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_3, [1, 12, 1024, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_9: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
    clone_6: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_9, memory_format = torch.contiguous_format);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_31: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_6, [1, 1024, 768]);  clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_32: "f32[1024, 768]" = torch.ops.aten.view.default(view_31, [-1, 768]);  view_31 = None
    addmm_5: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg10_1, view_32, arg11_1);  arg10_1 = view_32 = arg11_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_33: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_5, [1, 1024, 768]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    clone_7: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_33);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_11: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(clone_7, add_8);  clone_7 = add_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_3 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
    getitem_12: "f32[1, 1024, 1]" = var_mean_3[0]
    getitem_13: "f32[1, 1024, 1]" = var_mean_3[1];  var_mean_3 = None
    add_12: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
    rsqrt_3: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
    sub_5: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_11, getitem_13);  getitem_13 = None
    mul_10: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
    mul_11: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_10, arg104_1);  mul_10 = arg104_1 = None
    add_13: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_11, arg105_1);  mul_11 = arg105_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_34: "f32[1024, 768]" = torch.ops.aten.view.default(add_13, [-1, 768]);  add_13 = None
    addmm_6: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg12_1, view_34, arg13_1);  arg12_1 = view_34 = arg13_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_35: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_6, [1, 1024, 3072]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_12: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_35, 0.5)
    pow_2: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_35, 3.0)
    mul_13: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
    add_14: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(view_35, mul_13);  view_35 = mul_13 = None
    mul_14: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_14, 0.7978845608028654);  add_14 = None
    tanh_1: "f32[1, 1024, 3072]" = torch.ops.aten.tanh.default(mul_14);  mul_14 = None
    add_15: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
    mul_15: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_12, add_15);  mul_12 = add_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_36: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_15, [-1, 3072]);  mul_15 = None
    addmm_7: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg14_1, view_36, arg15_1);  arg14_1 = view_36 = arg15_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_37: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_7, [1, 1024, 768]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    clone_8: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_37);  view_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_16: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_11, clone_8);  add_11 = clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_4 = torch.ops.aten.var_mean.correction(add_16, [2], correction = 0, keepdim = True)
    getitem_14: "f32[1, 1024, 1]" = var_mean_4[0]
    getitem_15: "f32[1, 1024, 1]" = var_mean_4[1];  var_mean_4 = None
    add_17: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
    rsqrt_4: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
    sub_6: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_16, getitem_15);  getitem_15 = None
    mul_16: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
    mul_17: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_16, arg106_1);  mul_16 = arg106_1 = None
    add_18: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_17, arg107_1);  mul_17 = arg107_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_38: "f32[1024, 768]" = torch.ops.aten.view.default(add_18, [-1, 768]);  add_18 = None
    addmm_8: "f32[1024, 2304]" = torch.ops.aten.addmm.default(arg16_1, view_38, arg17_1);  arg16_1 = view_38 = arg17_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_39: "f32[1, 1024, 2304]" = torch.ops.aten.view.default(addmm_8, [1, 1024, 2304]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_2 = torch.ops.aten.split_with_sizes.default(view_39, [768, 768, 768], 2);  view_39 = None
    getitem_16: "f32[1, 1024, 768]" = split_with_sizes_2[0]
    getitem_17: "f32[1, 1024, 768]" = split_with_sizes_2[1]
    getitem_18: "f32[1, 1024, 768]" = split_with_sizes_2[2];  split_with_sizes_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_40: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_16, [1, 1024, 12, 64]);  getitem_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_10: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_41: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_17, [1, 1024, 12, 64]);  getitem_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_11: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_42: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_18, [1, 1024, 12, 64]);  getitem_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_12: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_13: "f32[1, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_11, [0, 1, 3, 2])
    expand_8: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_10, [1, 12, 1024, 64]);  permute_10 = None
    view_43: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_8, [12, 1024, 64]);  expand_8 = None
    expand_9: "f32[1, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_13, [1, 12, 64, 1024]);  permute_13 = None
    view_44: "f32[12, 64, 1024]" = torch.ops.aten.view.default(expand_9, [12, 64, 1024]);  expand_9 = None
    bmm_4: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_43, view_44);  view_43 = view_44 = None
    view_45: "f32[1, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_4, [1, 12, 1024, 1024]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_4: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    div_4: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_45, full_4);  view_45 = full_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_5: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(arg151_1, 0, 0, 9223372036854775807);  arg151_1 = None
    slice_6: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_5, 1, 0, 9223372036854775807);  slice_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_5: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_2: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_6, div_4, full_5);  slice_6 = div_4 = full_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_2: "f32[1, 12, 1024, 1]" = torch.ops.aten.amax.default(where_2, [-1], True)
    sub_7: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where_2, amax_2);  where_2 = amax_2 = None
    exp_2: "f32[1, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
    sum_3: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_5: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    clone_9: "f32[1, 12, 1024, 1024]" = torch.ops.aten.clone.default(div_5);  div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_10: "f32[1, 12, 1024, 1024]" = torch.ops.aten.expand.default(clone_9, [1, 12, 1024, 1024]);  clone_9 = None
    view_46: "f32[12, 1024, 1024]" = torch.ops.aten.view.default(expand_10, [12, 1024, 1024]);  expand_10 = None
    expand_11: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_12, [1, 12, 1024, 64])
    view_47: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_11, [12, 1024, 64]);  expand_11 = None
    bmm_5: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_46, view_47);  view_46 = view_47 = None
    view_48: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_5, [1, 12, 1024, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_14: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
    clone_10: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_49: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_10, [1, 1024, 768]);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_50: "f32[1024, 768]" = torch.ops.aten.view.default(view_49, [-1, 768]);  view_49 = None
    addmm_9: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg18_1, view_50, arg19_1);  arg18_1 = view_50 = arg19_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_51: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_9, [1, 1024, 768]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    clone_11: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_51);  view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_19: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(clone_11, add_16);  clone_11 = add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_5 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
    getitem_19: "f32[1, 1024, 1]" = var_mean_5[0]
    getitem_20: "f32[1, 1024, 1]" = var_mean_5[1];  var_mean_5 = None
    add_20: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_19, 1e-05);  getitem_19 = None
    rsqrt_5: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
    sub_8: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_19, getitem_20);  getitem_20 = None
    mul_18: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
    mul_19: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_18, arg108_1);  mul_18 = arg108_1 = None
    add_21: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_19, arg109_1);  mul_19 = arg109_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_52: "f32[1024, 768]" = torch.ops.aten.view.default(add_21, [-1, 768]);  add_21 = None
    addmm_10: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg20_1, view_52, arg21_1);  arg20_1 = view_52 = arg21_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_53: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_10, [1, 1024, 3072]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_20: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_53, 0.5)
    pow_3: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_53, 3.0)
    mul_21: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
    add_22: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(view_53, mul_21);  view_53 = mul_21 = None
    mul_22: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_22, 0.7978845608028654);  add_22 = None
    tanh_2: "f32[1, 1024, 3072]" = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
    add_23: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
    mul_23: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_20, add_23);  mul_20 = add_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_54: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_23, [-1, 3072]);  mul_23 = None
    addmm_11: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg22_1, view_54, arg23_1);  arg22_1 = view_54 = arg23_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_55: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_11, [1, 1024, 768]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    clone_12: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_55);  view_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_24: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_19, clone_12);  add_19 = clone_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_6 = torch.ops.aten.var_mean.correction(add_24, [2], correction = 0, keepdim = True)
    getitem_21: "f32[1, 1024, 1]" = var_mean_6[0]
    getitem_22: "f32[1, 1024, 1]" = var_mean_6[1];  var_mean_6 = None
    add_25: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_21, 1e-05);  getitem_21 = None
    rsqrt_6: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
    sub_9: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_24, getitem_22);  getitem_22 = None
    mul_24: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = rsqrt_6 = None
    mul_25: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_24, arg110_1);  mul_24 = arg110_1 = None
    add_26: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_25, arg111_1);  mul_25 = arg111_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_56: "f32[1024, 768]" = torch.ops.aten.view.default(add_26, [-1, 768]);  add_26 = None
    addmm_12: "f32[1024, 2304]" = torch.ops.aten.addmm.default(arg24_1, view_56, arg25_1);  arg24_1 = view_56 = arg25_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_57: "f32[1, 1024, 2304]" = torch.ops.aten.view.default(addmm_12, [1, 1024, 2304]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_3 = torch.ops.aten.split_with_sizes.default(view_57, [768, 768, 768], 2);  view_57 = None
    getitem_23: "f32[1, 1024, 768]" = split_with_sizes_3[0]
    getitem_24: "f32[1, 1024, 768]" = split_with_sizes_3[1]
    getitem_25: "f32[1, 1024, 768]" = split_with_sizes_3[2];  split_with_sizes_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_58: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_23, [1, 1024, 12, 64]);  getitem_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_15: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_59: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_24, [1, 1024, 12, 64]);  getitem_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_16: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_59, [0, 2, 1, 3]);  view_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_60: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_25, [1, 1024, 12, 64]);  getitem_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_17: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_60, [0, 2, 1, 3]);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_18: "f32[1, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_16, [0, 1, 3, 2])
    expand_12: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_15, [1, 12, 1024, 64]);  permute_15 = None
    view_61: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_12, [12, 1024, 64]);  expand_12 = None
    expand_13: "f32[1, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_18, [1, 12, 64, 1024]);  permute_18 = None
    view_62: "f32[12, 64, 1024]" = torch.ops.aten.view.default(expand_13, [12, 64, 1024]);  expand_13 = None
    bmm_6: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_61, view_62);  view_61 = view_62 = None
    view_63: "f32[1, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_6, [1, 12, 1024, 1024]);  bmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_6: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    div_6: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_63, full_6);  view_63 = full_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_7: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(arg152_1, 0, 0, 9223372036854775807);  arg152_1 = None
    slice_8: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_7, 1, 0, 9223372036854775807);  slice_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_7: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_3: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_8, div_6, full_7);  slice_8 = div_6 = full_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_3: "f32[1, 12, 1024, 1]" = torch.ops.aten.amax.default(where_3, [-1], True)
    sub_10: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where_3, amax_3);  where_3 = amax_3 = None
    exp_3: "f32[1, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
    sum_4: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_7: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    clone_13: "f32[1, 12, 1024, 1024]" = torch.ops.aten.clone.default(div_7);  div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_14: "f32[1, 12, 1024, 1024]" = torch.ops.aten.expand.default(clone_13, [1, 12, 1024, 1024]);  clone_13 = None
    view_64: "f32[12, 1024, 1024]" = torch.ops.aten.view.default(expand_14, [12, 1024, 1024]);  expand_14 = None
    expand_15: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_17, [1, 12, 1024, 64])
    view_65: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_15, [12, 1024, 64]);  expand_15 = None
    bmm_7: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_64, view_65);  view_64 = view_65 = None
    view_66: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_7, [1, 12, 1024, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_19: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None
    clone_14: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_67: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_14, [1, 1024, 768]);  clone_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_68: "f32[1024, 768]" = torch.ops.aten.view.default(view_67, [-1, 768]);  view_67 = None
    addmm_13: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg26_1, view_68, arg27_1);  arg26_1 = view_68 = arg27_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_69: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_13, [1, 1024, 768]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    clone_15: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_69);  view_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_27: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(clone_15, add_24);  clone_15 = add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_7 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
    getitem_26: "f32[1, 1024, 1]" = var_mean_7[0]
    getitem_27: "f32[1, 1024, 1]" = var_mean_7[1];  var_mean_7 = None
    add_28: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
    rsqrt_7: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    sub_11: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_27, getitem_27);  getitem_27 = None
    mul_26: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
    mul_27: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_26, arg112_1);  mul_26 = arg112_1 = None
    add_29: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_27, arg113_1);  mul_27 = arg113_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_70: "f32[1024, 768]" = torch.ops.aten.view.default(add_29, [-1, 768]);  add_29 = None
    addmm_14: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg28_1, view_70, arg29_1);  arg28_1 = view_70 = arg29_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_71: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_14, [1, 1024, 3072]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_28: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_71, 0.5)
    pow_4: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_71, 3.0)
    mul_29: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
    add_30: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(view_71, mul_29);  view_71 = mul_29 = None
    mul_30: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_30, 0.7978845608028654);  add_30 = None
    tanh_3: "f32[1, 1024, 3072]" = torch.ops.aten.tanh.default(mul_30);  mul_30 = None
    add_31: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
    mul_31: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_28, add_31);  mul_28 = add_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_72: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_31, [-1, 3072]);  mul_31 = None
    addmm_15: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg30_1, view_72, arg31_1);  arg30_1 = view_72 = arg31_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_73: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_15, [1, 1024, 768]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    clone_16: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_73);  view_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_32: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_27, clone_16);  add_27 = clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_8 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 1024, 1]" = var_mean_8[0]
    getitem_29: "f32[1, 1024, 1]" = var_mean_8[1];  var_mean_8 = None
    add_33: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
    rsqrt_8: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
    sub_12: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_32, getitem_29);  getitem_29 = None
    mul_32: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = rsqrt_8 = None
    mul_33: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_32, arg114_1);  mul_32 = arg114_1 = None
    add_34: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_33, arg115_1);  mul_33 = arg115_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_74: "f32[1024, 768]" = torch.ops.aten.view.default(add_34, [-1, 768]);  add_34 = None
    addmm_16: "f32[1024, 2304]" = torch.ops.aten.addmm.default(arg32_1, view_74, arg33_1);  arg32_1 = view_74 = arg33_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_75: "f32[1, 1024, 2304]" = torch.ops.aten.view.default(addmm_16, [1, 1024, 2304]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_4 = torch.ops.aten.split_with_sizes.default(view_75, [768, 768, 768], 2);  view_75 = None
    getitem_30: "f32[1, 1024, 768]" = split_with_sizes_4[0]
    getitem_31: "f32[1, 1024, 768]" = split_with_sizes_4[1]
    getitem_32: "f32[1, 1024, 768]" = split_with_sizes_4[2];  split_with_sizes_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_76: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_30, [1, 1024, 12, 64]);  getitem_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_20: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_77: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_31, [1, 1024, 12, 64]);  getitem_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_21: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_77, [0, 2, 1, 3]);  view_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_78: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_32, [1, 1024, 12, 64]);  getitem_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_22: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_23: "f32[1, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_21, [0, 1, 3, 2])
    expand_16: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_20, [1, 12, 1024, 64]);  permute_20 = None
    view_79: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_16, [12, 1024, 64]);  expand_16 = None
    expand_17: "f32[1, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_23, [1, 12, 64, 1024]);  permute_23 = None
    view_80: "f32[12, 64, 1024]" = torch.ops.aten.view.default(expand_17, [12, 64, 1024]);  expand_17 = None
    bmm_8: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_79, view_80);  view_79 = view_80 = None
    view_81: "f32[1, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_8, [1, 12, 1024, 1024]);  bmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_8: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    div_8: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_81, full_8);  view_81 = full_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_9: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(arg153_1, 0, 0, 9223372036854775807);  arg153_1 = None
    slice_10: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_9, 1, 0, 9223372036854775807);  slice_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_9: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_4: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_10, div_8, full_9);  slice_10 = div_8 = full_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_4: "f32[1, 12, 1024, 1]" = torch.ops.aten.amax.default(where_4, [-1], True)
    sub_13: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where_4, amax_4);  where_4 = amax_4 = None
    exp_4: "f32[1, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
    sum_5: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_9: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    clone_17: "f32[1, 12, 1024, 1024]" = torch.ops.aten.clone.default(div_9);  div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_18: "f32[1, 12, 1024, 1024]" = torch.ops.aten.expand.default(clone_17, [1, 12, 1024, 1024]);  clone_17 = None
    view_82: "f32[12, 1024, 1024]" = torch.ops.aten.view.default(expand_18, [12, 1024, 1024]);  expand_18 = None
    expand_19: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_22, [1, 12, 1024, 64])
    view_83: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_19, [12, 1024, 64]);  expand_19 = None
    bmm_9: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_82, view_83);  view_82 = view_83 = None
    view_84: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_9, [1, 12, 1024, 64]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_24: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_84, [0, 2, 1, 3]);  view_84 = None
    clone_18: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_85: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_18, [1, 1024, 768]);  clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_86: "f32[1024, 768]" = torch.ops.aten.view.default(view_85, [-1, 768]);  view_85 = None
    addmm_17: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg34_1, view_86, arg35_1);  arg34_1 = view_86 = arg35_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_87: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_17, [1, 1024, 768]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    clone_19: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_87);  view_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_35: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(clone_19, add_32);  clone_19 = add_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_9 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
    getitem_33: "f32[1, 1024, 1]" = var_mean_9[0]
    getitem_34: "f32[1, 1024, 1]" = var_mean_9[1];  var_mean_9 = None
    add_36: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_33, 1e-05);  getitem_33 = None
    rsqrt_9: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    sub_14: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_34);  getitem_34 = None
    mul_34: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
    mul_35: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_34, arg116_1);  mul_34 = arg116_1 = None
    add_37: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_35, arg117_1);  mul_35 = arg117_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_88: "f32[1024, 768]" = torch.ops.aten.view.default(add_37, [-1, 768]);  add_37 = None
    addmm_18: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg36_1, view_88, arg37_1);  arg36_1 = view_88 = arg37_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_89: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_18, [1, 1024, 3072]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_36: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_89, 0.5)
    pow_5: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_89, 3.0)
    mul_37: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
    add_38: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(view_89, mul_37);  view_89 = mul_37 = None
    mul_38: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_38, 0.7978845608028654);  add_38 = None
    tanh_4: "f32[1, 1024, 3072]" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
    add_39: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
    mul_39: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_36, add_39);  mul_36 = add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_90: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_39, [-1, 3072]);  mul_39 = None
    addmm_19: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg38_1, view_90, arg39_1);  arg38_1 = view_90 = arg39_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_91: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_19, [1, 1024, 768]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    clone_20: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_91);  view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_40: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_35, clone_20);  add_35 = clone_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_10 = torch.ops.aten.var_mean.correction(add_40, [2], correction = 0, keepdim = True)
    getitem_35: "f32[1, 1024, 1]" = var_mean_10[0]
    getitem_36: "f32[1, 1024, 1]" = var_mean_10[1];  var_mean_10 = None
    add_41: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_35, 1e-05);  getitem_35 = None
    rsqrt_10: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
    sub_15: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_40, getitem_36);  getitem_36 = None
    mul_40: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = rsqrt_10 = None
    mul_41: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_40, arg118_1);  mul_40 = arg118_1 = None
    add_42: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_41, arg119_1);  mul_41 = arg119_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_92: "f32[1024, 768]" = torch.ops.aten.view.default(add_42, [-1, 768]);  add_42 = None
    addmm_20: "f32[1024, 2304]" = torch.ops.aten.addmm.default(arg40_1, view_92, arg41_1);  arg40_1 = view_92 = arg41_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_93: "f32[1, 1024, 2304]" = torch.ops.aten.view.default(addmm_20, [1, 1024, 2304]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_5 = torch.ops.aten.split_with_sizes.default(view_93, [768, 768, 768], 2);  view_93 = None
    getitem_37: "f32[1, 1024, 768]" = split_with_sizes_5[0]
    getitem_38: "f32[1, 1024, 768]" = split_with_sizes_5[1]
    getitem_39: "f32[1, 1024, 768]" = split_with_sizes_5[2];  split_with_sizes_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_94: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_37, [1, 1024, 12, 64]);  getitem_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_25: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_95: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_38, [1, 1024, 12, 64]);  getitem_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_26: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_96: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_39, [1, 1024, 12, 64]);  getitem_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_27: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_28: "f32[1, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_26, [0, 1, 3, 2])
    expand_20: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_25, [1, 12, 1024, 64]);  permute_25 = None
    view_97: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_20, [12, 1024, 64]);  expand_20 = None
    expand_21: "f32[1, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_28, [1, 12, 64, 1024]);  permute_28 = None
    view_98: "f32[12, 64, 1024]" = torch.ops.aten.view.default(expand_21, [12, 64, 1024]);  expand_21 = None
    bmm_10: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_97, view_98);  view_97 = view_98 = None
    view_99: "f32[1, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_10, [1, 12, 1024, 1024]);  bmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_10: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    div_10: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_99, full_10);  view_99 = full_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_11: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(arg154_1, 0, 0, 9223372036854775807);  arg154_1 = None
    slice_12: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_11, 1, 0, 9223372036854775807);  slice_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_11: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_5: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_12, div_10, full_11);  slice_12 = div_10 = full_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_5: "f32[1, 12, 1024, 1]" = torch.ops.aten.amax.default(where_5, [-1], True)
    sub_16: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where_5, amax_5);  where_5 = amax_5 = None
    exp_5: "f32[1, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
    sum_6: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_11: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    clone_21: "f32[1, 12, 1024, 1024]" = torch.ops.aten.clone.default(div_11);  div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_22: "f32[1, 12, 1024, 1024]" = torch.ops.aten.expand.default(clone_21, [1, 12, 1024, 1024]);  clone_21 = None
    view_100: "f32[12, 1024, 1024]" = torch.ops.aten.view.default(expand_22, [12, 1024, 1024]);  expand_22 = None
    expand_23: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_27, [1, 12, 1024, 64])
    view_101: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_23, [12, 1024, 64]);  expand_23 = None
    bmm_11: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_100, view_101);  view_100 = view_101 = None
    view_102: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_11, [1, 12, 1024, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_29: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
    clone_22: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_103: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_22, [1, 1024, 768]);  clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_104: "f32[1024, 768]" = torch.ops.aten.view.default(view_103, [-1, 768]);  view_103 = None
    addmm_21: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg42_1, view_104, arg43_1);  arg42_1 = view_104 = arg43_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_105: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_21, [1, 1024, 768]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    clone_23: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_105);  view_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_43: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(clone_23, add_40);  clone_23 = add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_11 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
    getitem_40: "f32[1, 1024, 1]" = var_mean_11[0]
    getitem_41: "f32[1, 1024, 1]" = var_mean_11[1];  var_mean_11 = None
    add_44: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
    rsqrt_11: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
    sub_17: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_43, getitem_41);  getitem_41 = None
    mul_42: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
    mul_43: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_42, arg120_1);  mul_42 = arg120_1 = None
    add_45: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_43, arg121_1);  mul_43 = arg121_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_106: "f32[1024, 768]" = torch.ops.aten.view.default(add_45, [-1, 768]);  add_45 = None
    addmm_22: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg44_1, view_106, arg45_1);  arg44_1 = view_106 = arg45_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_107: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_22, [1, 1024, 3072]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_44: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.5)
    pow_6: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_107, 3.0)
    mul_45: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
    add_46: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(view_107, mul_45);  view_107 = mul_45 = None
    mul_46: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_46, 0.7978845608028654);  add_46 = None
    tanh_5: "f32[1, 1024, 3072]" = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
    add_47: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
    mul_47: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_44, add_47);  mul_44 = add_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_108: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_47, [-1, 3072]);  mul_47 = None
    addmm_23: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg46_1, view_108, arg47_1);  arg46_1 = view_108 = arg47_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_109: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_23, [1, 1024, 768]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    clone_24: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_109);  view_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_48: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_43, clone_24);  add_43 = clone_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_12 = torch.ops.aten.var_mean.correction(add_48, [2], correction = 0, keepdim = True)
    getitem_42: "f32[1, 1024, 1]" = var_mean_12[0]
    getitem_43: "f32[1, 1024, 1]" = var_mean_12[1];  var_mean_12 = None
    add_49: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
    rsqrt_12: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    sub_18: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_48, getitem_43);  getitem_43 = None
    mul_48: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = rsqrt_12 = None
    mul_49: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_48, arg122_1);  mul_48 = arg122_1 = None
    add_50: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_49, arg123_1);  mul_49 = arg123_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_110: "f32[1024, 768]" = torch.ops.aten.view.default(add_50, [-1, 768]);  add_50 = None
    addmm_24: "f32[1024, 2304]" = torch.ops.aten.addmm.default(arg48_1, view_110, arg49_1);  arg48_1 = view_110 = arg49_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_111: "f32[1, 1024, 2304]" = torch.ops.aten.view.default(addmm_24, [1, 1024, 2304]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_6 = torch.ops.aten.split_with_sizes.default(view_111, [768, 768, 768], 2);  view_111 = None
    getitem_44: "f32[1, 1024, 768]" = split_with_sizes_6[0]
    getitem_45: "f32[1, 1024, 768]" = split_with_sizes_6[1]
    getitem_46: "f32[1, 1024, 768]" = split_with_sizes_6[2];  split_with_sizes_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_112: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_44, [1, 1024, 12, 64]);  getitem_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_30: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_113: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_45, [1, 1024, 12, 64]);  getitem_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_31: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_113, [0, 2, 1, 3]);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_114: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_46, [1, 1024, 12, 64]);  getitem_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_32: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_33: "f32[1, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_31, [0, 1, 3, 2])
    expand_24: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_30, [1, 12, 1024, 64]);  permute_30 = None
    view_115: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_24, [12, 1024, 64]);  expand_24 = None
    expand_25: "f32[1, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_33, [1, 12, 64, 1024]);  permute_33 = None
    view_116: "f32[12, 64, 1024]" = torch.ops.aten.view.default(expand_25, [12, 64, 1024]);  expand_25 = None
    bmm_12: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_115, view_116);  view_115 = view_116 = None
    view_117: "f32[1, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_12, [1, 12, 1024, 1024]);  bmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_12: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    div_12: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_117, full_12);  view_117 = full_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_13: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(arg155_1, 0, 0, 9223372036854775807);  arg155_1 = None
    slice_14: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_13, 1, 0, 9223372036854775807);  slice_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_13: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_6: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_14, div_12, full_13);  slice_14 = div_12 = full_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_6: "f32[1, 12, 1024, 1]" = torch.ops.aten.amax.default(where_6, [-1], True)
    sub_19: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where_6, amax_6);  where_6 = amax_6 = None
    exp_6: "f32[1, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
    sum_7: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_13: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    clone_25: "f32[1, 12, 1024, 1024]" = torch.ops.aten.clone.default(div_13);  div_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_26: "f32[1, 12, 1024, 1024]" = torch.ops.aten.expand.default(clone_25, [1, 12, 1024, 1024]);  clone_25 = None
    view_118: "f32[12, 1024, 1024]" = torch.ops.aten.view.default(expand_26, [12, 1024, 1024]);  expand_26 = None
    expand_27: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_32, [1, 12, 1024, 64])
    view_119: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_27, [12, 1024, 64]);  expand_27 = None
    bmm_13: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_118, view_119);  view_118 = view_119 = None
    view_120: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_13, [1, 12, 1024, 64]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_34: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
    clone_26: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format);  permute_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_121: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_26, [1, 1024, 768]);  clone_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_122: "f32[1024, 768]" = torch.ops.aten.view.default(view_121, [-1, 768]);  view_121 = None
    addmm_25: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg50_1, view_122, arg51_1);  arg50_1 = view_122 = arg51_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_123: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_25, [1, 1024, 768]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    clone_27: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_123);  view_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_51: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(clone_27, add_48);  clone_27 = add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_13 = torch.ops.aten.var_mean.correction(add_51, [2], correction = 0, keepdim = True)
    getitem_47: "f32[1, 1024, 1]" = var_mean_13[0]
    getitem_48: "f32[1, 1024, 1]" = var_mean_13[1];  var_mean_13 = None
    add_52: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_47, 1e-05);  getitem_47 = None
    rsqrt_13: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    sub_20: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_51, getitem_48);  getitem_48 = None
    mul_50: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = rsqrt_13 = None
    mul_51: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_50, arg124_1);  mul_50 = arg124_1 = None
    add_53: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_51, arg125_1);  mul_51 = arg125_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_124: "f32[1024, 768]" = torch.ops.aten.view.default(add_53, [-1, 768]);  add_53 = None
    addmm_26: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg52_1, view_124, arg53_1);  arg52_1 = view_124 = arg53_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_125: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_26, [1, 1024, 3072]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_52: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_125, 0.5)
    pow_7: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_125, 3.0)
    mul_53: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
    add_54: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(view_125, mul_53);  view_125 = mul_53 = None
    mul_54: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_54, 0.7978845608028654);  add_54 = None
    tanh_6: "f32[1, 1024, 3072]" = torch.ops.aten.tanh.default(mul_54);  mul_54 = None
    add_55: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
    mul_55: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_52, add_55);  mul_52 = add_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_126: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_55, [-1, 3072]);  mul_55 = None
    addmm_27: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg54_1, view_126, arg55_1);  arg54_1 = view_126 = arg55_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_127: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_27, [1, 1024, 768]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    clone_28: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_127);  view_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_56: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_51, clone_28);  add_51 = clone_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_14 = torch.ops.aten.var_mean.correction(add_56, [2], correction = 0, keepdim = True)
    getitem_49: "f32[1, 1024, 1]" = var_mean_14[0]
    getitem_50: "f32[1, 1024, 1]" = var_mean_14[1];  var_mean_14 = None
    add_57: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_49, 1e-05);  getitem_49 = None
    rsqrt_14: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
    sub_21: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_56, getitem_50);  getitem_50 = None
    mul_56: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = rsqrt_14 = None
    mul_57: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_56, arg126_1);  mul_56 = arg126_1 = None
    add_58: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_57, arg127_1);  mul_57 = arg127_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_128: "f32[1024, 768]" = torch.ops.aten.view.default(add_58, [-1, 768]);  add_58 = None
    addmm_28: "f32[1024, 2304]" = torch.ops.aten.addmm.default(arg56_1, view_128, arg57_1);  arg56_1 = view_128 = arg57_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_129: "f32[1, 1024, 2304]" = torch.ops.aten.view.default(addmm_28, [1, 1024, 2304]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_7 = torch.ops.aten.split_with_sizes.default(view_129, [768, 768, 768], 2);  view_129 = None
    getitem_51: "f32[1, 1024, 768]" = split_with_sizes_7[0]
    getitem_52: "f32[1, 1024, 768]" = split_with_sizes_7[1]
    getitem_53: "f32[1, 1024, 768]" = split_with_sizes_7[2];  split_with_sizes_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_130: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_51, [1, 1024, 12, 64]);  getitem_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_35: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_131: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_52, [1, 1024, 12, 64]);  getitem_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_36: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_131, [0, 2, 1, 3]);  view_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_132: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_53, [1, 1024, 12, 64]);  getitem_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_37: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_132, [0, 2, 1, 3]);  view_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_38: "f32[1, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_36, [0, 1, 3, 2])
    expand_28: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_35, [1, 12, 1024, 64]);  permute_35 = None
    view_133: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_28, [12, 1024, 64]);  expand_28 = None
    expand_29: "f32[1, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_38, [1, 12, 64, 1024]);  permute_38 = None
    view_134: "f32[12, 64, 1024]" = torch.ops.aten.view.default(expand_29, [12, 64, 1024]);  expand_29 = None
    bmm_14: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_133, view_134);  view_133 = view_134 = None
    view_135: "f32[1, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_14, [1, 12, 1024, 1024]);  bmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_14: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    div_14: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_135, full_14);  view_135 = full_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_15: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(arg156_1, 0, 0, 9223372036854775807);  arg156_1 = None
    slice_16: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_15, 1, 0, 9223372036854775807);  slice_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_15: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_7: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_16, div_14, full_15);  slice_16 = div_14 = full_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_7: "f32[1, 12, 1024, 1]" = torch.ops.aten.amax.default(where_7, [-1], True)
    sub_22: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where_7, amax_7);  where_7 = amax_7 = None
    exp_7: "f32[1, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
    sum_8: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_15: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    clone_29: "f32[1, 12, 1024, 1024]" = torch.ops.aten.clone.default(div_15);  div_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_30: "f32[1, 12, 1024, 1024]" = torch.ops.aten.expand.default(clone_29, [1, 12, 1024, 1024]);  clone_29 = None
    view_136: "f32[12, 1024, 1024]" = torch.ops.aten.view.default(expand_30, [12, 1024, 1024]);  expand_30 = None
    expand_31: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_37, [1, 12, 1024, 64])
    view_137: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_31, [12, 1024, 64]);  expand_31 = None
    bmm_15: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_136, view_137);  view_136 = view_137 = None
    view_138: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_15, [1, 12, 1024, 64]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_39: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None
    clone_30: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_39, memory_format = torch.contiguous_format);  permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_139: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_30, [1, 1024, 768]);  clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_140: "f32[1024, 768]" = torch.ops.aten.view.default(view_139, [-1, 768]);  view_139 = None
    addmm_29: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg58_1, view_140, arg59_1);  arg58_1 = view_140 = arg59_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_141: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_29, [1, 1024, 768]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    clone_31: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_141);  view_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_59: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(clone_31, add_56);  clone_31 = add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_15 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
    getitem_54: "f32[1, 1024, 1]" = var_mean_15[0]
    getitem_55: "f32[1, 1024, 1]" = var_mean_15[1];  var_mean_15 = None
    add_60: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
    rsqrt_15: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    sub_23: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_59, getitem_55);  getitem_55 = None
    mul_58: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = rsqrt_15 = None
    mul_59: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_58, arg128_1);  mul_58 = arg128_1 = None
    add_61: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_59, arg129_1);  mul_59 = arg129_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_142: "f32[1024, 768]" = torch.ops.aten.view.default(add_61, [-1, 768]);  add_61 = None
    addmm_30: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg60_1, view_142, arg61_1);  arg60_1 = view_142 = arg61_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_143: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_30, [1, 1024, 3072]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_60: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_143, 0.5)
    pow_8: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_143, 3.0)
    mul_61: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
    add_62: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(view_143, mul_61);  view_143 = mul_61 = None
    mul_62: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_62, 0.7978845608028654);  add_62 = None
    tanh_7: "f32[1, 1024, 3072]" = torch.ops.aten.tanh.default(mul_62);  mul_62 = None
    add_63: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
    mul_63: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_60, add_63);  mul_60 = add_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_144: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_63, [-1, 3072]);  mul_63 = None
    addmm_31: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg62_1, view_144, arg63_1);  arg62_1 = view_144 = arg63_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_145: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_31, [1, 1024, 768]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    clone_32: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_145);  view_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_64: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_59, clone_32);  add_59 = clone_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_16 = torch.ops.aten.var_mean.correction(add_64, [2], correction = 0, keepdim = True)
    getitem_56: "f32[1, 1024, 1]" = var_mean_16[0]
    getitem_57: "f32[1, 1024, 1]" = var_mean_16[1];  var_mean_16 = None
    add_65: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
    rsqrt_16: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
    sub_24: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_64, getitem_57);  getitem_57 = None
    mul_64: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = rsqrt_16 = None
    mul_65: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_64, arg130_1);  mul_64 = arg130_1 = None
    add_66: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_65, arg131_1);  mul_65 = arg131_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_146: "f32[1024, 768]" = torch.ops.aten.view.default(add_66, [-1, 768]);  add_66 = None
    addmm_32: "f32[1024, 2304]" = torch.ops.aten.addmm.default(arg64_1, view_146, arg65_1);  arg64_1 = view_146 = arg65_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_147: "f32[1, 1024, 2304]" = torch.ops.aten.view.default(addmm_32, [1, 1024, 2304]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_8 = torch.ops.aten.split_with_sizes.default(view_147, [768, 768, 768], 2);  view_147 = None
    getitem_58: "f32[1, 1024, 768]" = split_with_sizes_8[0]
    getitem_59: "f32[1, 1024, 768]" = split_with_sizes_8[1]
    getitem_60: "f32[1, 1024, 768]" = split_with_sizes_8[2];  split_with_sizes_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_148: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_58, [1, 1024, 12, 64]);  getitem_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_40: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_149: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_59, [1, 1024, 12, 64]);  getitem_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_41: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_149, [0, 2, 1, 3]);  view_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_150: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_60, [1, 1024, 12, 64]);  getitem_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_42: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_150, [0, 2, 1, 3]);  view_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_43: "f32[1, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_41, [0, 1, 3, 2])
    expand_32: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_40, [1, 12, 1024, 64]);  permute_40 = None
    view_151: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_32, [12, 1024, 64]);  expand_32 = None
    expand_33: "f32[1, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_43, [1, 12, 64, 1024]);  permute_43 = None
    view_152: "f32[12, 64, 1024]" = torch.ops.aten.view.default(expand_33, [12, 64, 1024]);  expand_33 = None
    bmm_16: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_151, view_152);  view_151 = view_152 = None
    view_153: "f32[1, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_16, [1, 12, 1024, 1024]);  bmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_16: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    div_16: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_153, full_16);  view_153 = full_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_17: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(arg157_1, 0, 0, 9223372036854775807);  arg157_1 = None
    slice_18: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_17, 1, 0, 9223372036854775807);  slice_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_17: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_8: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_18, div_16, full_17);  slice_18 = div_16 = full_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_8: "f32[1, 12, 1024, 1]" = torch.ops.aten.amax.default(where_8, [-1], True)
    sub_25: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where_8, amax_8);  where_8 = amax_8 = None
    exp_8: "f32[1, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
    sum_9: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_17: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    clone_33: "f32[1, 12, 1024, 1024]" = torch.ops.aten.clone.default(div_17);  div_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_34: "f32[1, 12, 1024, 1024]" = torch.ops.aten.expand.default(clone_33, [1, 12, 1024, 1024]);  clone_33 = None
    view_154: "f32[12, 1024, 1024]" = torch.ops.aten.view.default(expand_34, [12, 1024, 1024]);  expand_34 = None
    expand_35: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_42, [1, 12, 1024, 64])
    view_155: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_35, [12, 1024, 64]);  expand_35 = None
    bmm_17: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_154, view_155);  view_154 = view_155 = None
    view_156: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_17, [1, 12, 1024, 64]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_44: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None
    clone_34: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_44, memory_format = torch.contiguous_format);  permute_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_157: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_34, [1, 1024, 768]);  clone_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_158: "f32[1024, 768]" = torch.ops.aten.view.default(view_157, [-1, 768]);  view_157 = None
    addmm_33: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg66_1, view_158, arg67_1);  arg66_1 = view_158 = arg67_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_159: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_33, [1, 1024, 768]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    clone_35: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_159);  view_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_67: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(clone_35, add_64);  clone_35 = add_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_17 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
    getitem_61: "f32[1, 1024, 1]" = var_mean_17[0]
    getitem_62: "f32[1, 1024, 1]" = var_mean_17[1];  var_mean_17 = None
    add_68: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_61, 1e-05);  getitem_61 = None
    rsqrt_17: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
    sub_26: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_67, getitem_62);  getitem_62 = None
    mul_66: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = rsqrt_17 = None
    mul_67: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_66, arg132_1);  mul_66 = arg132_1 = None
    add_69: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_67, arg133_1);  mul_67 = arg133_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_160: "f32[1024, 768]" = torch.ops.aten.view.default(add_69, [-1, 768]);  add_69 = None
    addmm_34: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg68_1, view_160, arg69_1);  arg68_1 = view_160 = arg69_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_161: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_34, [1, 1024, 3072]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_68: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_161, 0.5)
    pow_9: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_161, 3.0)
    mul_69: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
    add_70: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(view_161, mul_69);  view_161 = mul_69 = None
    mul_70: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_70, 0.7978845608028654);  add_70 = None
    tanh_8: "f32[1, 1024, 3072]" = torch.ops.aten.tanh.default(mul_70);  mul_70 = None
    add_71: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
    mul_71: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_68, add_71);  mul_68 = add_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_162: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_71, [-1, 3072]);  mul_71 = None
    addmm_35: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg70_1, view_162, arg71_1);  arg70_1 = view_162 = arg71_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_163: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_35, [1, 1024, 768]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    clone_36: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_163);  view_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_72: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_67, clone_36);  add_67 = clone_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_18 = torch.ops.aten.var_mean.correction(add_72, [2], correction = 0, keepdim = True)
    getitem_63: "f32[1, 1024, 1]" = var_mean_18[0]
    getitem_64: "f32[1, 1024, 1]" = var_mean_18[1];  var_mean_18 = None
    add_73: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_63, 1e-05);  getitem_63 = None
    rsqrt_18: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
    sub_27: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_72, getitem_64);  getitem_64 = None
    mul_72: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = rsqrt_18 = None
    mul_73: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_72, arg134_1);  mul_72 = arg134_1 = None
    add_74: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_73, arg135_1);  mul_73 = arg135_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_164: "f32[1024, 768]" = torch.ops.aten.view.default(add_74, [-1, 768]);  add_74 = None
    addmm_36: "f32[1024, 2304]" = torch.ops.aten.addmm.default(arg72_1, view_164, arg73_1);  arg72_1 = view_164 = arg73_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_165: "f32[1, 1024, 2304]" = torch.ops.aten.view.default(addmm_36, [1, 1024, 2304]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_9 = torch.ops.aten.split_with_sizes.default(view_165, [768, 768, 768], 2);  view_165 = None
    getitem_65: "f32[1, 1024, 768]" = split_with_sizes_9[0]
    getitem_66: "f32[1, 1024, 768]" = split_with_sizes_9[1]
    getitem_67: "f32[1, 1024, 768]" = split_with_sizes_9[2];  split_with_sizes_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_166: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_65, [1, 1024, 12, 64]);  getitem_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_45: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_166, [0, 2, 1, 3]);  view_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_167: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_66, [1, 1024, 12, 64]);  getitem_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_46: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_167, [0, 2, 1, 3]);  view_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_168: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_67, [1, 1024, 12, 64]);  getitem_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_47: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_48: "f32[1, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_46, [0, 1, 3, 2])
    expand_36: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_45, [1, 12, 1024, 64]);  permute_45 = None
    view_169: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_36, [12, 1024, 64]);  expand_36 = None
    expand_37: "f32[1, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_48, [1, 12, 64, 1024]);  permute_48 = None
    view_170: "f32[12, 64, 1024]" = torch.ops.aten.view.default(expand_37, [12, 64, 1024]);  expand_37 = None
    bmm_18: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_169, view_170);  view_169 = view_170 = None
    view_171: "f32[1, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_18, [1, 12, 1024, 1024]);  bmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_18: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    div_18: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_171, full_18);  view_171 = full_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_19: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(arg158_1, 0, 0, 9223372036854775807);  arg158_1 = None
    slice_20: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_19, 1, 0, 9223372036854775807);  slice_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_19: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_9: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_20, div_18, full_19);  slice_20 = div_18 = full_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_9: "f32[1, 12, 1024, 1]" = torch.ops.aten.amax.default(where_9, [-1], True)
    sub_28: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where_9, amax_9);  where_9 = amax_9 = None
    exp_9: "f32[1, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
    sum_10: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_19: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    clone_37: "f32[1, 12, 1024, 1024]" = torch.ops.aten.clone.default(div_19);  div_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_38: "f32[1, 12, 1024, 1024]" = torch.ops.aten.expand.default(clone_37, [1, 12, 1024, 1024]);  clone_37 = None
    view_172: "f32[12, 1024, 1024]" = torch.ops.aten.view.default(expand_38, [12, 1024, 1024]);  expand_38 = None
    expand_39: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_47, [1, 12, 1024, 64])
    view_173: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_39, [12, 1024, 64]);  expand_39 = None
    bmm_19: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_172, view_173);  view_172 = view_173 = None
    view_174: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_19, [1, 12, 1024, 64]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_49: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_174, [0, 2, 1, 3]);  view_174 = None
    clone_38: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_175: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_38, [1, 1024, 768]);  clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_176: "f32[1024, 768]" = torch.ops.aten.view.default(view_175, [-1, 768]);  view_175 = None
    addmm_37: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg74_1, view_176, arg75_1);  arg74_1 = view_176 = arg75_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_177: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_37, [1, 1024, 768]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    clone_39: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_177);  view_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_75: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(clone_39, add_72);  clone_39 = add_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_19 = torch.ops.aten.var_mean.correction(add_75, [2], correction = 0, keepdim = True)
    getitem_68: "f32[1, 1024, 1]" = var_mean_19[0]
    getitem_69: "f32[1, 1024, 1]" = var_mean_19[1];  var_mean_19 = None
    add_76: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05);  getitem_68 = None
    rsqrt_19: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
    sub_29: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_75, getitem_69);  getitem_69 = None
    mul_74: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = rsqrt_19 = None
    mul_75: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_74, arg136_1);  mul_74 = arg136_1 = None
    add_77: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_75, arg137_1);  mul_75 = arg137_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_178: "f32[1024, 768]" = torch.ops.aten.view.default(add_77, [-1, 768]);  add_77 = None
    addmm_38: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg76_1, view_178, arg77_1);  arg76_1 = view_178 = arg77_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_179: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_38, [1, 1024, 3072]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_76: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_179, 0.5)
    pow_10: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_179, 3.0)
    mul_77: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
    add_78: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(view_179, mul_77);  view_179 = mul_77 = None
    mul_78: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_78, 0.7978845608028654);  add_78 = None
    tanh_9: "f32[1, 1024, 3072]" = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
    add_79: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
    mul_79: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_76, add_79);  mul_76 = add_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_180: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_79, [-1, 3072]);  mul_79 = None
    addmm_39: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg78_1, view_180, arg79_1);  arg78_1 = view_180 = arg79_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_181: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_39, [1, 1024, 768]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    clone_40: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_181);  view_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_80: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_75, clone_40);  add_75 = clone_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_20 = torch.ops.aten.var_mean.correction(add_80, [2], correction = 0, keepdim = True)
    getitem_70: "f32[1, 1024, 1]" = var_mean_20[0]
    getitem_71: "f32[1, 1024, 1]" = var_mean_20[1];  var_mean_20 = None
    add_81: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05);  getitem_70 = None
    rsqrt_20: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
    sub_30: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_80, getitem_71);  getitem_71 = None
    mul_80: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = rsqrt_20 = None
    mul_81: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_80, arg138_1);  mul_80 = arg138_1 = None
    add_82: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_81, arg139_1);  mul_81 = arg139_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_182: "f32[1024, 768]" = torch.ops.aten.view.default(add_82, [-1, 768]);  add_82 = None
    addmm_40: "f32[1024, 2304]" = torch.ops.aten.addmm.default(arg80_1, view_182, arg81_1);  arg80_1 = view_182 = arg81_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_183: "f32[1, 1024, 2304]" = torch.ops.aten.view.default(addmm_40, [1, 1024, 2304]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_10 = torch.ops.aten.split_with_sizes.default(view_183, [768, 768, 768], 2);  view_183 = None
    getitem_72: "f32[1, 1024, 768]" = split_with_sizes_10[0]
    getitem_73: "f32[1, 1024, 768]" = split_with_sizes_10[1]
    getitem_74: "f32[1, 1024, 768]" = split_with_sizes_10[2];  split_with_sizes_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_184: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_72, [1, 1024, 12, 64]);  getitem_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_50: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_185: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_73, [1, 1024, 12, 64]);  getitem_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_51: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_186: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_74, [1, 1024, 12, 64]);  getitem_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_52: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_53: "f32[1, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_51, [0, 1, 3, 2])
    expand_40: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_50, [1, 12, 1024, 64]);  permute_50 = None
    view_187: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_40, [12, 1024, 64]);  expand_40 = None
    expand_41: "f32[1, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_53, [1, 12, 64, 1024]);  permute_53 = None
    view_188: "f32[12, 64, 1024]" = torch.ops.aten.view.default(expand_41, [12, 64, 1024]);  expand_41 = None
    bmm_20: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_187, view_188);  view_187 = view_188 = None
    view_189: "f32[1, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_20, [1, 12, 1024, 1024]);  bmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_20: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    div_20: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_189, full_20);  view_189 = full_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_21: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(arg159_1, 0, 0, 9223372036854775807);  arg159_1 = None
    slice_22: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_21, 1, 0, 9223372036854775807);  slice_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_21: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_10: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_22, div_20, full_21);  slice_22 = div_20 = full_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_10: "f32[1, 12, 1024, 1]" = torch.ops.aten.amax.default(where_10, [-1], True)
    sub_31: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where_10, amax_10);  where_10 = amax_10 = None
    exp_10: "f32[1, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
    sum_11: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_21: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    clone_41: "f32[1, 12, 1024, 1024]" = torch.ops.aten.clone.default(div_21);  div_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_42: "f32[1, 12, 1024, 1024]" = torch.ops.aten.expand.default(clone_41, [1, 12, 1024, 1024]);  clone_41 = None
    view_190: "f32[12, 1024, 1024]" = torch.ops.aten.view.default(expand_42, [12, 1024, 1024]);  expand_42 = None
    expand_43: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_52, [1, 12, 1024, 64])
    view_191: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_43, [12, 1024, 64]);  expand_43 = None
    bmm_21: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_190, view_191);  view_190 = view_191 = None
    view_192: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_21, [1, 12, 1024, 64]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_54: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None
    clone_42: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_54, memory_format = torch.contiguous_format);  permute_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_193: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_42, [1, 1024, 768]);  clone_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_194: "f32[1024, 768]" = torch.ops.aten.view.default(view_193, [-1, 768]);  view_193 = None
    addmm_41: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg82_1, view_194, arg83_1);  arg82_1 = view_194 = arg83_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_195: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_41, [1, 1024, 768]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    clone_43: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_195);  view_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_83: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(clone_43, add_80);  clone_43 = add_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_21 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
    getitem_75: "f32[1, 1024, 1]" = var_mean_21[0]
    getitem_76: "f32[1, 1024, 1]" = var_mean_21[1];  var_mean_21 = None
    add_84: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_75, 1e-05);  getitem_75 = None
    rsqrt_21: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
    sub_32: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_83, getitem_76);  getitem_76 = None
    mul_82: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = rsqrt_21 = None
    mul_83: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_82, arg140_1);  mul_82 = arg140_1 = None
    add_85: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_83, arg141_1);  mul_83 = arg141_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_196: "f32[1024, 768]" = torch.ops.aten.view.default(add_85, [-1, 768]);  add_85 = None
    addmm_42: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg84_1, view_196, arg85_1);  arg84_1 = view_196 = arg85_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_197: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_42, [1, 1024, 3072]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_84: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_197, 0.5)
    pow_11: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_197, 3.0)
    mul_85: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
    add_86: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(view_197, mul_85);  view_197 = mul_85 = None
    mul_86: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_86, 0.7978845608028654);  add_86 = None
    tanh_10: "f32[1, 1024, 3072]" = torch.ops.aten.tanh.default(mul_86);  mul_86 = None
    add_87: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
    mul_87: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_84, add_87);  mul_84 = add_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_198: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_87, [-1, 3072]);  mul_87 = None
    addmm_43: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg86_1, view_198, arg87_1);  arg86_1 = view_198 = arg87_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_199: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_43, [1, 1024, 768]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    clone_44: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_199);  view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_88: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_83, clone_44);  add_83 = clone_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_22 = torch.ops.aten.var_mean.correction(add_88, [2], correction = 0, keepdim = True)
    getitem_77: "f32[1, 1024, 1]" = var_mean_22[0]
    getitem_78: "f32[1, 1024, 1]" = var_mean_22[1];  var_mean_22 = None
    add_89: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_77, 1e-05);  getitem_77 = None
    rsqrt_22: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
    sub_33: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_88, getitem_78);  getitem_78 = None
    mul_88: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = rsqrt_22 = None
    mul_89: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_88, arg142_1);  mul_88 = arg142_1 = None
    add_90: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_89, arg143_1);  mul_89 = arg143_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_200: "f32[1024, 768]" = torch.ops.aten.view.default(add_90, [-1, 768]);  add_90 = None
    addmm_44: "f32[1024, 2304]" = torch.ops.aten.addmm.default(arg88_1, view_200, arg89_1);  arg88_1 = view_200 = arg89_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_201: "f32[1, 1024, 2304]" = torch.ops.aten.view.default(addmm_44, [1, 1024, 2304]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_11 = torch.ops.aten.split_with_sizes.default(view_201, [768, 768, 768], 2);  view_201 = None
    getitem_79: "f32[1, 1024, 768]" = split_with_sizes_11[0]
    getitem_80: "f32[1, 1024, 768]" = split_with_sizes_11[1]
    getitem_81: "f32[1, 1024, 768]" = split_with_sizes_11[2];  split_with_sizes_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_202: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_79, [1, 1024, 12, 64]);  getitem_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_55: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_202, [0, 2, 1, 3]);  view_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_203: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_80, [1, 1024, 12, 64]);  getitem_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_56: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_203, [0, 2, 1, 3]);  view_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_204: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_81, [1, 1024, 12, 64]);  getitem_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_57: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_204, [0, 2, 1, 3]);  view_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_58: "f32[1, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_56, [0, 1, 3, 2])
    expand_44: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_55, [1, 12, 1024, 64]);  permute_55 = None
    view_205: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_44, [12, 1024, 64]);  expand_44 = None
    expand_45: "f32[1, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_58, [1, 12, 64, 1024]);  permute_58 = None
    view_206: "f32[12, 64, 1024]" = torch.ops.aten.view.default(expand_45, [12, 64, 1024]);  expand_45 = None
    bmm_22: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_205, view_206);  view_205 = view_206 = None
    view_207: "f32[1, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_22, [1, 12, 1024, 1024]);  bmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_22: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    div_22: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_207, full_22);  view_207 = full_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_23: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(arg160_1, 0, 0, 9223372036854775807);  arg160_1 = None
    slice_24: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_23, 1, 0, 9223372036854775807);  slice_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_23: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_11: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_24, div_22, full_23);  slice_24 = div_22 = full_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_11: "f32[1, 12, 1024, 1]" = torch.ops.aten.amax.default(where_11, [-1], True)
    sub_34: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where_11, amax_11);  where_11 = amax_11 = None
    exp_11: "f32[1, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
    sum_12: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_23: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    clone_45: "f32[1, 12, 1024, 1024]" = torch.ops.aten.clone.default(div_23);  div_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_46: "f32[1, 12, 1024, 1024]" = torch.ops.aten.expand.default(clone_45, [1, 12, 1024, 1024]);  clone_45 = None
    view_208: "f32[12, 1024, 1024]" = torch.ops.aten.view.default(expand_46, [12, 1024, 1024]);  expand_46 = None
    expand_47: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_57, [1, 12, 1024, 64])
    view_209: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_47, [12, 1024, 64]);  expand_47 = None
    bmm_23: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_208, view_209);  view_208 = view_209 = None
    view_210: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_23, [1, 12, 1024, 64]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_59: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_210, [0, 2, 1, 3]);  view_210 = None
    clone_46: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_211: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_46, [1, 1024, 768]);  clone_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_212: "f32[1024, 768]" = torch.ops.aten.view.default(view_211, [-1, 768]);  view_211 = None
    addmm_45: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg90_1, view_212, arg91_1);  arg90_1 = view_212 = arg91_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_213: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_45, [1, 1024, 768]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    clone_47: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_213);  view_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_91: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(clone_47, add_88);  clone_47 = add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_23 = torch.ops.aten.var_mean.correction(add_91, [2], correction = 0, keepdim = True)
    getitem_82: "f32[1, 1024, 1]" = var_mean_23[0]
    getitem_83: "f32[1, 1024, 1]" = var_mean_23[1];  var_mean_23 = None
    add_92: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
    rsqrt_23: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
    sub_35: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_91, getitem_83);  getitem_83 = None
    mul_90: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = rsqrt_23 = None
    mul_91: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_90, arg144_1);  mul_90 = arg144_1 = None
    add_93: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_91, arg145_1);  mul_91 = arg145_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_214: "f32[1024, 768]" = torch.ops.aten.view.default(add_93, [-1, 768]);  add_93 = None
    addmm_46: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg92_1, view_214, arg93_1);  arg92_1 = view_214 = arg93_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_215: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_46, [1, 1024, 3072]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_92: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_215, 0.5)
    pow_12: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_215, 3.0)
    mul_93: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
    add_94: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(view_215, mul_93);  view_215 = mul_93 = None
    mul_94: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_94, 0.7978845608028654);  add_94 = None
    tanh_11: "f32[1, 1024, 3072]" = torch.ops.aten.tanh.default(mul_94);  mul_94 = None
    add_95: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
    mul_95: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_92, add_95);  mul_92 = add_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_216: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_95, [-1, 3072]);  mul_95 = None
    addmm_47: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg94_1, view_216, arg95_1);  arg94_1 = view_216 = arg95_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_217: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_47, [1, 1024, 768]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    clone_48: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(view_217);  view_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_96: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_91, clone_48);  add_91 = clone_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:926, code: hidden_states = self.ln_f(hidden_states)
    var_mean_24 = torch.ops.aten.var_mean.correction(add_96, [2], correction = 0, keepdim = True)
    getitem_84: "f32[1, 1024, 1]" = var_mean_24[0]
    getitem_85: "f32[1, 1024, 1]" = var_mean_24[1];  var_mean_24 = None
    add_97: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05);  getitem_84 = None
    rsqrt_24: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
    sub_36: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_96, getitem_85);  add_96 = getitem_85 = None
    mul_96: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = rsqrt_24 = None
    mul_97: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_96, arg146_1);  mul_96 = arg146_1 = None
    add_98: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_97, arg147_1);  mul_97 = arg147_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:928, code: hidden_states = hidden_states.view(output_shape)
    view_218: "f32[1, 1024, 768]" = torch.ops.aten.view.default(add_98, [-1, 1024, 768]);  add_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1433, code: logits = self.score(hidden_states)
    permute_60: "f32[768, 2]" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
    view_219: "f32[1024, 768]" = torch.ops.aten.view.default(view_218, [1024, 768])
    mm: "f32[1024, 2]" = torch.ops.aten.mm.default(view_219, permute_60);  view_219 = permute_60 = None
    view_220: "f32[1, 1024, 2]" = torch.ops.aten.view.default(mm, [1, 1024, 2]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1447, code: sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1).to(
    eq: "b8[1, 1024]" = torch.ops.aten.eq.Scalar(arg161_1, 0);  arg161_1 = None
    convert_element_type: "i64[1, 1024]" = torch.ops.prims.convert_element_type.default(eq, torch.int64);  eq = None
    argmax: "i64[1]" = torch.ops.aten.argmax.default(convert_element_type, -1);  convert_element_type = None
    sub_37: "i64[1]" = torch.ops.aten.sub.Tensor(argmax, 1);  argmax = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1457, code: pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
    iota_1: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    index: "f32[1, 2]" = torch.ops.aten.index.Tensor(view_220, [iota_1, sub_37]);  view_220 = iota_1 = sub_37 = None
    return (view_218, permute_1, permute_2, permute_6, permute_7, permute_11, permute_12, permute_16, permute_17, permute_21, permute_22, permute_26, permute_27, permute_31, permute_32, permute_36, permute_37, permute_41, permute_42, permute_46, permute_47, permute_51, permute_52, permute_56, permute_57, index)
    