from __future__ import annotations



def forward(self, arg0_1: "f32[2304]", arg1_1: "f32[768, 2304]", arg2_1: "f32[768]", arg3_1: "f32[768, 768]", arg4_1: "f32[3072]", arg5_1: "f32[768, 3072]", arg6_1: "f32[768]", arg7_1: "f32[3072, 768]", arg8_1: "f32[2304]", arg9_1: "f32[768, 2304]", arg10_1: "f32[768]", arg11_1: "f32[768, 768]", arg12_1: "f32[3072]", arg13_1: "f32[768, 3072]", arg14_1: "f32[768]", arg15_1: "f32[3072, 768]", arg16_1: "f32[2304]", arg17_1: "f32[768, 2304]", arg18_1: "f32[768]", arg19_1: "f32[768, 768]", arg20_1: "f32[3072]", arg21_1: "f32[768, 3072]", arg22_1: "f32[768]", arg23_1: "f32[3072, 768]", arg24_1: "f32[2304]", arg25_1: "f32[768, 2304]", arg26_1: "f32[768]", arg27_1: "f32[768, 768]", arg28_1: "f32[3072]", arg29_1: "f32[768, 3072]", arg30_1: "f32[768]", arg31_1: "f32[3072, 768]", arg32_1: "f32[2304]", arg33_1: "f32[768, 2304]", arg34_1: "f32[768]", arg35_1: "f32[768, 768]", arg36_1: "f32[3072]", arg37_1: "f32[768, 3072]", arg38_1: "f32[768]", arg39_1: "f32[3072, 768]", arg40_1: "f32[2304]", arg41_1: "f32[768, 2304]", arg42_1: "f32[768]", arg43_1: "f32[768, 768]", arg44_1: "f32[3072]", arg45_1: "f32[768, 3072]", arg46_1: "f32[768]", arg47_1: "f32[3072, 768]", arg48_1: "f32[50257, 768]", arg49_1: "f32[1024, 768]", arg50_1: "f32[768]", arg51_1: "f32[768]", arg52_1: "f32[768]", arg53_1: "f32[768]", arg54_1: "f32[768]", arg55_1: "f32[768]", arg56_1: "f32[768]", arg57_1: "f32[768]", arg58_1: "f32[768]", arg59_1: "f32[768]", arg60_1: "f32[768]", arg61_1: "f32[768]", arg62_1: "f32[768]", arg63_1: "f32[768]", arg64_1: "f32[768]", arg65_1: "f32[768]", arg66_1: "f32[768]", arg67_1: "f32[768]", arg68_1: "f32[768]", arg69_1: "f32[768]", arg70_1: "f32[768]", arg71_1: "f32[768]", arg72_1: "f32[768]", arg73_1: "f32[768]", arg74_1: "f32[768]", arg75_1: "f32[768]", arg76_1: "f32[50257, 768]", arg77_1: "b8[1, 1, 1024, 1024]", arg78_1: "b8[1, 1, 1024, 1024]", arg79_1: "b8[1, 1, 1024, 1024]", arg80_1: "b8[1, 1, 1024, 1024]", arg81_1: "b8[1, 1, 1024, 1024]", arg82_1: "b8[1, 1, 1024, 1024]", arg83_1: "i64[1, 512]", arg84_1: "i64[1, 512]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:781, code: input_ids = input_ids.view(-1, input_shape[-1])
    view: "i64[1, 512]" = torch.ops.aten.reshape.default(arg83_1, [-1, 512]);  arg83_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:843, code: inputs_embeds = self.wte(input_ids)
    embedding: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg48_1, view);  arg48_1 = view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:802, code: position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
    iota: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:803, code: position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
    unsqueeze: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
    view_1: "i64[1, 512]" = torch.ops.aten.reshape.default(unsqueeze, [-1, 512]);  unsqueeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:844, code: position_embeds = self.wpe(position_ids)
    embedding_1: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg49_1, view_1);  arg49_1 = view_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:845, code: hidden_states = inputs_embeds + position_embeds
    add: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 512, 1]" = var_mean[0]
    getitem_1: "f32[1, 512, 1]" = var_mean[1];  var_mean = None
    sub: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add, getitem_1);  getitem_1 = None
    add_1: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    mul: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
    mul_1: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul, arg50_1);  mul = arg50_1 = None
    add_2: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_1, arg51_1);  mul_1 = arg51_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_2: "f32[512, 768]" = torch.ops.aten.reshape.default(add_2, [-1, 768]);  add_2 = None
    addmm: "f32[512, 2304]" = torch.ops.aten.addmm.default(arg0_1, view_2, arg1_1);  arg0_1 = view_2 = arg1_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_3: "f32[1, 512, 2304]" = torch.ops.aten.reshape.default(addmm, [1, 512, 2304]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes = torch.ops.aten.split_with_sizes.default(view_3, [768, 768, 768], 2);  view_3 = None
    getitem_2: "f32[1, 512, 768]" = split_with_sizes[0]
    getitem_3: "f32[1, 512, 768]" = split_with_sizes[1]
    getitem_4: "f32[1, 512, 768]" = split_with_sizes[2];  split_with_sizes = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_3: "b8[1, 1, 512, 1024]" = torch.ops.aten.slice.Tensor(arg77_1, 2, 0, 512);  arg77_1 = None
    slice_4: "b8[1, 1, 512, 512]" = torch.ops.aten.slice.Tensor(slice_3, 3, 0, 512);  slice_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_4: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_2, [1, 512, 12, 64]);  getitem_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute, [1, 12, 512, 64]);  permute = None
    view_7: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand, [12, 512, 64]);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_5: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_3, [1, 512, 12, 64]);  getitem_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_1: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_3: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_1, [0, 1, 3, 2])
    expand_1: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_3, [1, 12, 64, 512]);  permute_3 = None
    view_8: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_1, [12, 64, 512]);  expand_1 = None
    bmm: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_7, view_8);  view_7 = view_8 = None
    view_9: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm, [1, 12, 512, 512]);  bmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_default: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    div: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_9, full_default);  view_9 = full_default = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_1: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(slice_4, div, full_default_1);  slice_4 = div = full_default_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where, [-1], True)
    sub_1: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where, amax);  where = amax = None
    exp: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
    sum_1: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div_1: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_2: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(div_1, [1, 12, 512, 512]);  div_1 = None
    view_10: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_2, [12, 512, 512]);  expand_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_6: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_4, [1, 512, 12, 64]);  getitem_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_2: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_3: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_2, [1, 12, 512, 64])
    view_11: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_3, [12, 512, 64]);  expand_3 = None
    bmm_1: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_10, view_11);  view_10 = view_11 = None
    view_12: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_1, [1, 12, 512, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_4: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None
    clone_2: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_13: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_2, [1, 512, 768]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_14: "f32[512, 768]" = torch.ops.aten.reshape.default(view_13, [-1, 768]);  view_13 = None
    addmm_1: "f32[512, 768]" = torch.ops.aten.addmm.default(arg2_1, view_14, arg3_1);  arg2_1 = view_14 = arg3_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_15: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_1, [1, 512, 768]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_3: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_15, add);  view_15 = add = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
    getitem_5: "f32[1, 512, 1]" = var_mean_1[0]
    getitem_6: "f32[1, 512, 1]" = var_mean_1[1];  var_mean_1 = None
    sub_2: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_3, getitem_6);  getitem_6 = None
    add_4: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_5, 1e-05);  getitem_5 = None
    rsqrt_1: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    mul_2: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
    mul_3: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_2, arg52_1);  mul_2 = arg52_1 = None
    add_5: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_3, arg53_1);  mul_3 = arg53_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_16: "f32[512, 768]" = torch.ops.aten.reshape.default(add_5, [-1, 768]);  add_5 = None
    addmm_2: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg4_1, view_16, arg5_1);  arg4_1 = view_16 = arg5_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_17: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_2, [1, 512, 3072]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_4: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_17, 0.5)
    pow_1: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_17, 3.0)
    mul_5: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
    add_6: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_17, mul_5);  view_17 = mul_5 = None
    mul_6: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_6, 0.7978845608028654);  add_6 = None
    tanh: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_6);  mul_6 = None
    add_7: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
    mul_7: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_4, add_7);  mul_4 = add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_18: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_7, [-1, 3072]);  mul_7 = None
    addmm_3: "f32[512, 768]" = torch.ops.aten.addmm.default(arg6_1, view_18, arg7_1);  arg6_1 = view_18 = arg7_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_19: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_3, [1, 512, 768]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_8: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_3, view_19);  add_3 = view_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
    getitem_7: "f32[1, 512, 1]" = var_mean_2[0]
    getitem_8: "f32[1, 512, 1]" = var_mean_2[1];  var_mean_2 = None
    sub_3: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_8, getitem_8);  getitem_8 = None
    add_9: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_7, 1e-05);  getitem_7 = None
    rsqrt_2: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
    mul_8: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
    mul_9: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_8, arg54_1);  mul_8 = arg54_1 = None
    add_10: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_9, arg55_1);  mul_9 = arg55_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_20: "f32[512, 768]" = torch.ops.aten.reshape.default(add_10, [-1, 768]);  add_10 = None
    addmm_4: "f32[512, 2304]" = torch.ops.aten.addmm.default(arg8_1, view_20, arg9_1);  arg8_1 = view_20 = arg9_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_21: "f32[1, 512, 2304]" = torch.ops.aten.reshape.default(addmm_4, [1, 512, 2304]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(view_21, [768, 768, 768], 2);  view_21 = None
    getitem_9: "f32[1, 512, 768]" = split_with_sizes_1[0]
    getitem_10: "f32[1, 512, 768]" = split_with_sizes_1[1]
    getitem_11: "f32[1, 512, 768]" = split_with_sizes_1[2];  split_with_sizes_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_7: "b8[1, 1, 512, 1024]" = torch.ops.aten.slice.Tensor(arg78_1, 2, 0, 512);  arg78_1 = None
    slice_8: "b8[1, 1, 512, 512]" = torch.ops.aten.slice.Tensor(slice_7, 3, 0, 512);  slice_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_22: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_9, [1, 512, 12, 64]);  getitem_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_5: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_22, [0, 2, 1, 3]);  view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_4: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_5, [1, 12, 512, 64]);  permute_5 = None
    view_25: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_4, [12, 512, 64]);  expand_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_23: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_10, [1, 512, 12, 64]);  getitem_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_6: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_23, [0, 2, 1, 3]);  view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_8: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_6, [0, 1, 3, 2])
    expand_5: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_8, [1, 12, 64, 512]);  permute_8 = None
    view_26: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_5, [12, 64, 512]);  expand_5 = None
    bmm_2: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_25, view_26);  view_25 = view_26 = None
    view_27: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_2, [1, 12, 512, 512]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_default_2: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    div_2: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_27, full_default_2);  view_27 = full_default_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_3: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_1: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(slice_8, div_2, full_default_3);  slice_8 = div_2 = full_default_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_1: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where_1, [-1], True)
    sub_4: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where_1, amax_1);  where_1 = amax_1 = None
    exp_1: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
    sum_2: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_3: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_6: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(div_3, [1, 12, 512, 512]);  div_3 = None
    view_28: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_6, [12, 512, 512]);  expand_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_24: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_11, [1, 512, 12, 64]);  getitem_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_7: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_7: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_7, [1, 12, 512, 64])
    view_29: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_7, [12, 512, 64]);  expand_7 = None
    bmm_3: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_28, view_29);  view_28 = view_29 = None
    view_30: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_3, [1, 12, 512, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_9: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
    clone_6: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_9, memory_format = torch.contiguous_format);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_31: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_6, [1, 512, 768]);  clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_32: "f32[512, 768]" = torch.ops.aten.reshape.default(view_31, [-1, 768]);  view_31 = None
    addmm_5: "f32[512, 768]" = torch.ops.aten.addmm.default(arg10_1, view_32, arg11_1);  arg10_1 = view_32 = arg11_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_33: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_5, [1, 512, 768]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_11: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_33, add_8);  view_33 = add_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_3 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
    getitem_12: "f32[1, 512, 1]" = var_mean_3[0]
    getitem_13: "f32[1, 512, 1]" = var_mean_3[1];  var_mean_3 = None
    sub_5: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_11, getitem_13);  getitem_13 = None
    add_12: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
    rsqrt_3: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
    mul_10: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
    mul_11: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_10, arg56_1);  mul_10 = arg56_1 = None
    add_13: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_11, arg57_1);  mul_11 = arg57_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_34: "f32[512, 768]" = torch.ops.aten.reshape.default(add_13, [-1, 768]);  add_13 = None
    addmm_6: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg12_1, view_34, arg13_1);  arg12_1 = view_34 = arg13_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_35: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_6, [1, 512, 3072]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_12: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_35, 0.5)
    pow_2: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_35, 3.0)
    mul_13: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
    add_14: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_35, mul_13);  view_35 = mul_13 = None
    mul_14: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_14, 0.7978845608028654);  add_14 = None
    tanh_1: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_14);  mul_14 = None
    add_15: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
    mul_15: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_12, add_15);  mul_12 = add_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_36: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_15, [-1, 3072]);  mul_15 = None
    addmm_7: "f32[512, 768]" = torch.ops.aten.addmm.default(arg14_1, view_36, arg15_1);  arg14_1 = view_36 = arg15_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_37: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_7, [1, 512, 768]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_16: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_11, view_37);  add_11 = view_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_4 = torch.ops.aten.var_mean.correction(add_16, [2], correction = 0, keepdim = True)
    getitem_14: "f32[1, 512, 1]" = var_mean_4[0]
    getitem_15: "f32[1, 512, 1]" = var_mean_4[1];  var_mean_4 = None
    sub_6: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_16, getitem_15);  getitem_15 = None
    add_17: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
    rsqrt_4: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
    mul_16: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
    mul_17: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_16, arg58_1);  mul_16 = arg58_1 = None
    add_18: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_17, arg59_1);  mul_17 = arg59_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_38: "f32[512, 768]" = torch.ops.aten.reshape.default(add_18, [-1, 768]);  add_18 = None
    addmm_8: "f32[512, 2304]" = torch.ops.aten.addmm.default(arg16_1, view_38, arg17_1);  arg16_1 = view_38 = arg17_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_39: "f32[1, 512, 2304]" = torch.ops.aten.reshape.default(addmm_8, [1, 512, 2304]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_2 = torch.ops.aten.split_with_sizes.default(view_39, [768, 768, 768], 2);  view_39 = None
    getitem_16: "f32[1, 512, 768]" = split_with_sizes_2[0]
    getitem_17: "f32[1, 512, 768]" = split_with_sizes_2[1]
    getitem_18: "f32[1, 512, 768]" = split_with_sizes_2[2];  split_with_sizes_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_11: "b8[1, 1, 512, 1024]" = torch.ops.aten.slice.Tensor(arg79_1, 2, 0, 512);  arg79_1 = None
    slice_12: "b8[1, 1, 512, 512]" = torch.ops.aten.slice.Tensor(slice_11, 3, 0, 512);  slice_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_40: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_16, [1, 512, 12, 64]);  getitem_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_10: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_8: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_10, [1, 12, 512, 64]);  permute_10 = None
    view_43: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_8, [12, 512, 64]);  expand_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_41: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_17, [1, 512, 12, 64]);  getitem_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_11: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_13: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_11, [0, 1, 3, 2])
    expand_9: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_13, [1, 12, 64, 512]);  permute_13 = None
    view_44: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_9, [12, 64, 512]);  expand_9 = None
    bmm_4: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_43, view_44);  view_43 = view_44 = None
    view_45: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_4, [1, 12, 512, 512]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_default_4: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    div_4: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_45, full_default_4);  view_45 = full_default_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_5: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_2: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(slice_12, div_4, full_default_5);  slice_12 = div_4 = full_default_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_2: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where_2, [-1], True)
    sub_7: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where_2, amax_2);  where_2 = amax_2 = None
    exp_2: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
    sum_3: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_5: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_10: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(div_5, [1, 12, 512, 512]);  div_5 = None
    view_46: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_10, [12, 512, 512]);  expand_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_42: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_18, [1, 512, 12, 64]);  getitem_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_12: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_11: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_12, [1, 12, 512, 64])
    view_47: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_11, [12, 512, 64]);  expand_11 = None
    bmm_5: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_46, view_47);  view_46 = view_47 = None
    view_48: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_5, [1, 12, 512, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_14: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
    clone_10: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_49: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_10, [1, 512, 768]);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_50: "f32[512, 768]" = torch.ops.aten.reshape.default(view_49, [-1, 768]);  view_49 = None
    addmm_9: "f32[512, 768]" = torch.ops.aten.addmm.default(arg18_1, view_50, arg19_1);  arg18_1 = view_50 = arg19_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_51: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_9, [1, 512, 768]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_19: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_51, add_16);  view_51 = add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_5 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
    getitem_19: "f32[1, 512, 1]" = var_mean_5[0]
    getitem_20: "f32[1, 512, 1]" = var_mean_5[1];  var_mean_5 = None
    sub_8: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_19, getitem_20);  getitem_20 = None
    add_20: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_19, 1e-05);  getitem_19 = None
    rsqrt_5: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
    mul_18: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
    mul_19: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_18, arg60_1);  mul_18 = arg60_1 = None
    add_21: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_19, arg61_1);  mul_19 = arg61_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_52: "f32[512, 768]" = torch.ops.aten.reshape.default(add_21, [-1, 768]);  add_21 = None
    addmm_10: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg20_1, view_52, arg21_1);  arg20_1 = view_52 = arg21_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_53: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_10, [1, 512, 3072]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_20: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_53, 0.5)
    pow_3: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_53, 3.0)
    mul_21: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
    add_22: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_53, mul_21);  view_53 = mul_21 = None
    mul_22: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_22, 0.7978845608028654);  add_22 = None
    tanh_2: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
    add_23: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
    mul_23: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_20, add_23);  mul_20 = add_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_54: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_23, [-1, 3072]);  mul_23 = None
    addmm_11: "f32[512, 768]" = torch.ops.aten.addmm.default(arg22_1, view_54, arg23_1);  arg22_1 = view_54 = arg23_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_55: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_11, [1, 512, 768]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_24: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_19, view_55);  add_19 = view_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_6 = torch.ops.aten.var_mean.correction(add_24, [2], correction = 0, keepdim = True)
    getitem_21: "f32[1, 512, 1]" = var_mean_6[0]
    getitem_22: "f32[1, 512, 1]" = var_mean_6[1];  var_mean_6 = None
    sub_9: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_24, getitem_22);  getitem_22 = None
    add_25: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_21, 1e-05);  getitem_21 = None
    rsqrt_6: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
    mul_24: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = rsqrt_6 = None
    mul_25: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_24, arg62_1);  mul_24 = arg62_1 = None
    add_26: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_25, arg63_1);  mul_25 = arg63_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_56: "f32[512, 768]" = torch.ops.aten.reshape.default(add_26, [-1, 768]);  add_26 = None
    addmm_12: "f32[512, 2304]" = torch.ops.aten.addmm.default(arg24_1, view_56, arg25_1);  arg24_1 = view_56 = arg25_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_57: "f32[1, 512, 2304]" = torch.ops.aten.reshape.default(addmm_12, [1, 512, 2304]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_3 = torch.ops.aten.split_with_sizes.default(view_57, [768, 768, 768], 2);  view_57 = None
    getitem_23: "f32[1, 512, 768]" = split_with_sizes_3[0]
    getitem_24: "f32[1, 512, 768]" = split_with_sizes_3[1]
    getitem_25: "f32[1, 512, 768]" = split_with_sizes_3[2];  split_with_sizes_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_15: "b8[1, 1, 512, 1024]" = torch.ops.aten.slice.Tensor(arg80_1, 2, 0, 512);  arg80_1 = None
    slice_16: "b8[1, 1, 512, 512]" = torch.ops.aten.slice.Tensor(slice_15, 3, 0, 512);  slice_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_58: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_23, [1, 512, 12, 64]);  getitem_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_15: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_12: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_15, [1, 12, 512, 64]);  permute_15 = None
    view_61: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_12, [12, 512, 64]);  expand_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_59: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_24, [1, 512, 12, 64]);  getitem_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_16: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_59, [0, 2, 1, 3]);  view_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_18: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_16, [0, 1, 3, 2])
    expand_13: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_18, [1, 12, 64, 512]);  permute_18 = None
    view_62: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_13, [12, 64, 512]);  expand_13 = None
    bmm_6: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_61, view_62);  view_61 = view_62 = None
    view_63: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_6, [1, 12, 512, 512]);  bmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_default_6: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    div_6: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_63, full_default_6);  view_63 = full_default_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_7: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_3: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(slice_16, div_6, full_default_7);  slice_16 = div_6 = full_default_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_3: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where_3, [-1], True)
    sub_10: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where_3, amax_3);  where_3 = amax_3 = None
    exp_3: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
    sum_4: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_7: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_14: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(div_7, [1, 12, 512, 512]);  div_7 = None
    view_64: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_14, [12, 512, 512]);  expand_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_60: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_25, [1, 512, 12, 64]);  getitem_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_17: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_60, [0, 2, 1, 3]);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_15: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_17, [1, 12, 512, 64])
    view_65: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_15, [12, 512, 64]);  expand_15 = None
    bmm_7: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_64, view_65);  view_64 = view_65 = None
    view_66: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_7, [1, 12, 512, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_19: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None
    clone_14: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_67: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_14, [1, 512, 768]);  clone_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_68: "f32[512, 768]" = torch.ops.aten.reshape.default(view_67, [-1, 768]);  view_67 = None
    addmm_13: "f32[512, 768]" = torch.ops.aten.addmm.default(arg26_1, view_68, arg27_1);  arg26_1 = view_68 = arg27_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_69: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_13, [1, 512, 768]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_27: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_69, add_24);  view_69 = add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_7 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
    getitem_26: "f32[1, 512, 1]" = var_mean_7[0]
    getitem_27: "f32[1, 512, 1]" = var_mean_7[1];  var_mean_7 = None
    sub_11: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_27, getitem_27);  getitem_27 = None
    add_28: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
    rsqrt_7: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    mul_26: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
    mul_27: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_26, arg64_1);  mul_26 = arg64_1 = None
    add_29: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_27, arg65_1);  mul_27 = arg65_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_70: "f32[512, 768]" = torch.ops.aten.reshape.default(add_29, [-1, 768]);  add_29 = None
    addmm_14: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg28_1, view_70, arg29_1);  arg28_1 = view_70 = arg29_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_71: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_14, [1, 512, 3072]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_28: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_71, 0.5)
    pow_4: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_71, 3.0)
    mul_29: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
    add_30: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_71, mul_29);  view_71 = mul_29 = None
    mul_30: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_30, 0.7978845608028654);  add_30 = None
    tanh_3: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_30);  mul_30 = None
    add_31: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
    mul_31: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_28, add_31);  mul_28 = add_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_72: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_31, [-1, 3072]);  mul_31 = None
    addmm_15: "f32[512, 768]" = torch.ops.aten.addmm.default(arg30_1, view_72, arg31_1);  arg30_1 = view_72 = arg31_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_73: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_15, [1, 512, 768]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_32: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_27, view_73);  add_27 = view_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_8 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 512, 1]" = var_mean_8[0]
    getitem_29: "f32[1, 512, 1]" = var_mean_8[1];  var_mean_8 = None
    sub_12: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_32, getitem_29);  getitem_29 = None
    add_33: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
    rsqrt_8: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
    mul_32: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = rsqrt_8 = None
    mul_33: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_32, arg66_1);  mul_32 = arg66_1 = None
    add_34: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_33, arg67_1);  mul_33 = arg67_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_74: "f32[512, 768]" = torch.ops.aten.reshape.default(add_34, [-1, 768]);  add_34 = None
    addmm_16: "f32[512, 2304]" = torch.ops.aten.addmm.default(arg32_1, view_74, arg33_1);  arg32_1 = view_74 = arg33_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_75: "f32[1, 512, 2304]" = torch.ops.aten.reshape.default(addmm_16, [1, 512, 2304]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_4 = torch.ops.aten.split_with_sizes.default(view_75, [768, 768, 768], 2);  view_75 = None
    getitem_30: "f32[1, 512, 768]" = split_with_sizes_4[0]
    getitem_31: "f32[1, 512, 768]" = split_with_sizes_4[1]
    getitem_32: "f32[1, 512, 768]" = split_with_sizes_4[2];  split_with_sizes_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_19: "b8[1, 1, 512, 1024]" = torch.ops.aten.slice.Tensor(arg81_1, 2, 0, 512);  arg81_1 = None
    slice_20: "b8[1, 1, 512, 512]" = torch.ops.aten.slice.Tensor(slice_19, 3, 0, 512);  slice_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_76: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_30, [1, 512, 12, 64]);  getitem_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_20: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_16: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_20, [1, 12, 512, 64]);  permute_20 = None
    view_79: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_16, [12, 512, 64]);  expand_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_77: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_31, [1, 512, 12, 64]);  getitem_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_21: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_77, [0, 2, 1, 3]);  view_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_23: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_21, [0, 1, 3, 2])
    expand_17: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_23, [1, 12, 64, 512]);  permute_23 = None
    view_80: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_17, [12, 64, 512]);  expand_17 = None
    bmm_8: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_79, view_80);  view_79 = view_80 = None
    view_81: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_8, [1, 12, 512, 512]);  bmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_default_8: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    div_8: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_81, full_default_8);  view_81 = full_default_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_9: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_4: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(slice_20, div_8, full_default_9);  slice_20 = div_8 = full_default_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_4: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where_4, [-1], True)
    sub_13: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where_4, amax_4);  where_4 = amax_4 = None
    exp_4: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
    sum_5: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_9: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_18: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(div_9, [1, 12, 512, 512]);  div_9 = None
    view_82: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_18, [12, 512, 512]);  expand_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_78: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_32, [1, 512, 12, 64]);  getitem_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_22: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_19: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_22, [1, 12, 512, 64])
    view_83: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_19, [12, 512, 64]);  expand_19 = None
    bmm_9: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_82, view_83);  view_82 = view_83 = None
    view_84: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_9, [1, 12, 512, 64]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_24: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_84, [0, 2, 1, 3]);  view_84 = None
    clone_18: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_85: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_18, [1, 512, 768]);  clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_86: "f32[512, 768]" = torch.ops.aten.reshape.default(view_85, [-1, 768]);  view_85 = None
    addmm_17: "f32[512, 768]" = torch.ops.aten.addmm.default(arg34_1, view_86, arg35_1);  arg34_1 = view_86 = arg35_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_87: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_17, [1, 512, 768]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_35: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_87, add_32);  view_87 = add_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_9 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
    getitem_33: "f32[1, 512, 1]" = var_mean_9[0]
    getitem_34: "f32[1, 512, 1]" = var_mean_9[1];  var_mean_9 = None
    sub_14: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_34);  getitem_34 = None
    add_36: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_33, 1e-05);  getitem_33 = None
    rsqrt_9: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    mul_34: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
    mul_35: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_34, arg68_1);  mul_34 = arg68_1 = None
    add_37: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_35, arg69_1);  mul_35 = arg69_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_88: "f32[512, 768]" = torch.ops.aten.reshape.default(add_37, [-1, 768]);  add_37 = None
    addmm_18: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg36_1, view_88, arg37_1);  arg36_1 = view_88 = arg37_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_89: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_18, [1, 512, 3072]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_36: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_89, 0.5)
    pow_5: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_89, 3.0)
    mul_37: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
    add_38: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_89, mul_37);  view_89 = mul_37 = None
    mul_38: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_38, 0.7978845608028654);  add_38 = None
    tanh_4: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
    add_39: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
    mul_39: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_36, add_39);  mul_36 = add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_90: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_39, [-1, 3072]);  mul_39 = None
    addmm_19: "f32[512, 768]" = torch.ops.aten.addmm.default(arg38_1, view_90, arg39_1);  arg38_1 = view_90 = arg39_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_91: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_19, [1, 512, 768]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_40: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_35, view_91);  add_35 = view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_10 = torch.ops.aten.var_mean.correction(add_40, [2], correction = 0, keepdim = True)
    getitem_35: "f32[1, 512, 1]" = var_mean_10[0]
    getitem_36: "f32[1, 512, 1]" = var_mean_10[1];  var_mean_10 = None
    sub_15: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_40, getitem_36);  getitem_36 = None
    add_41: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_35, 1e-05);  getitem_35 = None
    rsqrt_10: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
    mul_40: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = rsqrt_10 = None
    mul_41: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_40, arg70_1);  mul_40 = arg70_1 = None
    add_42: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_41, arg71_1);  mul_41 = arg71_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_92: "f32[512, 768]" = torch.ops.aten.reshape.default(add_42, [-1, 768]);  add_42 = None
    addmm_20: "f32[512, 2304]" = torch.ops.aten.addmm.default(arg40_1, view_92, arg41_1);  arg40_1 = view_92 = arg41_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_93: "f32[1, 512, 2304]" = torch.ops.aten.reshape.default(addmm_20, [1, 512, 2304]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_5 = torch.ops.aten.split_with_sizes.default(view_93, [768, 768, 768], 2);  view_93 = None
    getitem_37: "f32[1, 512, 768]" = split_with_sizes_5[0]
    getitem_38: "f32[1, 512, 768]" = split_with_sizes_5[1]
    getitem_39: "f32[1, 512, 768]" = split_with_sizes_5[2];  split_with_sizes_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_23: "b8[1, 1, 512, 1024]" = torch.ops.aten.slice.Tensor(arg82_1, 2, 0, 512);  arg82_1 = None
    slice_24: "b8[1, 1, 512, 512]" = torch.ops.aten.slice.Tensor(slice_23, 3, 0, 512);  slice_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_94: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_37, [1, 512, 12, 64]);  getitem_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_25: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_20: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_25, [1, 12, 512, 64]);  permute_25 = None
    view_97: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_20, [12, 512, 64]);  expand_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_95: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_38, [1, 512, 12, 64]);  getitem_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_26: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_28: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_26, [0, 1, 3, 2])
    expand_21: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_28, [1, 12, 64, 512]);  permute_28 = None
    view_98: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_21, [12, 64, 512]);  expand_21 = None
    bmm_10: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_97, view_98);  view_97 = view_98 = None
    view_99: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_10, [1, 12, 512, 512]);  bmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_default_10: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    div_10: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_99, full_default_10);  view_99 = full_default_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_11: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_5: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(slice_24, div_10, full_default_11);  slice_24 = div_10 = full_default_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_5: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where_5, [-1], True)
    sub_16: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where_5, amax_5);  where_5 = amax_5 = None
    exp_5: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
    sum_6: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_11: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_22: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(div_11, [1, 12, 512, 512]);  div_11 = None
    view_100: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_22, [12, 512, 512]);  expand_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_96: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(getitem_39, [1, 512, 12, 64]);  getitem_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_27: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_23: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_27, [1, 12, 512, 64])
    view_101: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_23, [12, 512, 64]);  expand_23 = None
    bmm_11: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_100, view_101);  view_100 = view_101 = None
    view_102: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_11, [1, 12, 512, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_29: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
    clone_22: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_103: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_22, [1, 512, 768]);  clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_104: "f32[512, 768]" = torch.ops.aten.reshape.default(view_103, [-1, 768]);  view_103 = None
    addmm_21: "f32[512, 768]" = torch.ops.aten.addmm.default(arg42_1, view_104, arg43_1);  arg42_1 = view_104 = arg43_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_105: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_21, [1, 512, 768]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_43: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_105, add_40);  view_105 = add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_11 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
    getitem_40: "f32[1, 512, 1]" = var_mean_11[0]
    getitem_41: "f32[1, 512, 1]" = var_mean_11[1];  var_mean_11 = None
    sub_17: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_43, getitem_41);  getitem_41 = None
    add_44: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
    rsqrt_11: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
    mul_42: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
    mul_43: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_42, arg72_1);  mul_42 = arg72_1 = None
    add_45: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_43, arg73_1);  mul_43 = arg73_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_106: "f32[512, 768]" = torch.ops.aten.reshape.default(add_45, [-1, 768]);  add_45 = None
    addmm_22: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg44_1, view_106, arg45_1);  arg44_1 = view_106 = arg45_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_107: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_22, [1, 512, 3072]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_44: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.5)
    pow_6: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_107, 3.0)
    mul_45: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
    add_46: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_107, mul_45);  view_107 = mul_45 = None
    mul_46: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_46, 0.7978845608028654);  add_46 = None
    tanh_5: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
    add_47: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
    mul_47: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_44, add_47);  mul_44 = add_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_108: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_47, [-1, 3072]);  mul_47 = None
    addmm_23: "f32[512, 768]" = torch.ops.aten.addmm.default(arg46_1, view_108, arg47_1);  arg46_1 = view_108 = arg47_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_109: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_23, [1, 512, 768]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_48: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_43, view_109);  add_43 = view_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:926, code: hidden_states = self.ln_f(hidden_states)
    var_mean_12 = torch.ops.aten.var_mean.correction(add_48, [2], correction = 0, keepdim = True)
    getitem_42: "f32[1, 512, 1]" = var_mean_12[0]
    getitem_43: "f32[1, 512, 1]" = var_mean_12[1];  var_mean_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1106, code: shift_labels = labels[..., 1:].contiguous()
    slice_27: "i64[1, 511]" = torch.ops.aten.slice.Tensor(arg84_1, 1, 1, 9223372036854775807);  arg84_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1109, code: loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    view_114: "i64[511]" = torch.ops.aten.reshape.default(slice_27, [-1]);  slice_27 = None
    ne_1: "b8[511]" = torch.ops.aten.ne.Scalar(view_114, -100)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:926, code: hidden_states = self.ln_f(hidden_states)
    sub_18: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_48, getitem_43);  add_48 = getitem_43 = None
    add_49: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
    rsqrt_12: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    mul_48: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = rsqrt_12 = None
    mul_49: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_48, arg74_1);  mul_48 = arg74_1 = None
    add_50: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_49, arg75_1);  mul_49 = arg75_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:928, code: hidden_states = hidden_states.view(output_shape)
    view_110: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_50, [-1, 512, 768]);  add_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1098, code: lm_logits = self.lm_head(hidden_states)
    view_111: "f32[512, 768]" = torch.ops.aten.reshape.default(view_110, [512, 768]);  view_110 = None
    permute_30: "f32[768, 50257]" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
    mm: "f32[512, 50257]" = torch.ops.aten.mm.default(view_111, permute_30);  view_111 = permute_30 = None
    view_112: "f32[1, 512, 50257]" = torch.ops.aten.reshape.default(mm, [1, 512, 50257]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1105, code: shift_logits = lm_logits[..., :-1, :].contiguous()
    slice_25: "f32[1, 511, 50257]" = torch.ops.aten.slice.Tensor(view_112, 1, 0, -1)
    slice_26: "f32[1, 511, 50257]" = torch.ops.aten.slice.Tensor(slice_25, 2, 0, 9223372036854775807);  slice_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1109, code: loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    view_113: "f32[511, 50257]" = torch.ops.aten.reshape.default(slice_26, [-1, 50257]);  slice_26 = None
    amax_6: "f32[511, 1]" = torch.ops.aten.amax.default(view_113, [1], True)
    sub_19: "f32[511, 50257]" = torch.ops.aten.sub.Tensor(view_113, amax_6);  view_113 = amax_6 = None
    exp_6: "f32[511, 50257]" = torch.ops.aten.exp.default(sub_19)
    sum_7: "f32[511, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [1], True);  exp_6 = None
    log: "f32[511, 1]" = torch.ops.aten.log.default(sum_7);  sum_7 = None
    sub_20: "f32[511, 50257]" = torch.ops.aten.sub.Tensor(sub_19, log);  sub_19 = log = None
    ne: "b8[511]" = torch.ops.aten.ne.Scalar(view_114, -100)
    full_default_12: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where_6: "i64[511]" = torch.ops.aten.where.self(ne, view_114, full_default_12);  ne = full_default_12 = None
    unsqueeze_1: "i64[511, 1]" = torch.ops.aten.unsqueeze.default(where_6, 1);  where_6 = None
    gather: "f32[511, 1]" = torch.ops.aten.gather.default(sub_20, 1, unsqueeze_1);  sub_20 = unsqueeze_1 = None
    squeeze: "f32[511]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[511]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
    full_default_13: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where_7: "f32[511]" = torch.ops.aten.where.self(ne_1, neg, full_default_13);  ne_1 = neg = full_default_13 = None
    sum_9: "f32[]" = torch.ops.aten.sum.default(where_7);  where_7 = None
    ne_2: "b8[511]" = torch.ops.aten.ne.Scalar(view_114, -100);  view_114 = None
    sum_8: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_8, torch.float32);  sum_8 = None
    div_12: "f32[]" = torch.ops.aten.div.Tensor(sum_9, convert_element_type);  sum_9 = convert_element_type = None
    return (div_12, view_112, permute_1, permute_2, permute_6, permute_7, permute_11, permute_12, permute_16, permute_17, permute_21, permute_22, permute_26, permute_27)
    