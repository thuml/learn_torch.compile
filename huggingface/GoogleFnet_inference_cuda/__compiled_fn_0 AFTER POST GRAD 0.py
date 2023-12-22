from __future__ import annotations



def forward(self, arg0_1: "f32[32000, 768]", arg1_1: "f32[4, 768]", arg2_1: "f32[512, 768]", arg3_1: "f32[768]", arg4_1: "f32[768]", arg5_1: "f32[768, 768]", arg6_1: "f32[768]", arg7_1: "f32[768]", arg8_1: "f32[768]", arg9_1: "f32[3072, 768]", arg10_1: "f32[3072]", arg11_1: "f32[768, 3072]", arg12_1: "f32[768]", arg13_1: "f32[768]", arg14_1: "f32[768]", arg15_1: "f32[768]", arg16_1: "f32[768]", arg17_1: "f32[3072, 768]", arg18_1: "f32[3072]", arg19_1: "f32[768, 3072]", arg20_1: "f32[768]", arg21_1: "f32[768]", arg22_1: "f32[768]", arg23_1: "f32[768]", arg24_1: "f32[768]", arg25_1: "f32[3072, 768]", arg26_1: "f32[3072]", arg27_1: "f32[768, 3072]", arg28_1: "f32[768]", arg29_1: "f32[768]", arg30_1: "f32[768]", arg31_1: "f32[768]", arg32_1: "f32[768]", arg33_1: "f32[3072, 768]", arg34_1: "f32[3072]", arg35_1: "f32[768, 3072]", arg36_1: "f32[768]", arg37_1: "f32[768]", arg38_1: "f32[768]", arg39_1: "f32[768]", arg40_1: "f32[768]", arg41_1: "f32[3072, 768]", arg42_1: "f32[3072]", arg43_1: "f32[768, 3072]", arg44_1: "f32[768]", arg45_1: "f32[768]", arg46_1: "f32[768]", arg47_1: "f32[768]", arg48_1: "f32[768]", arg49_1: "f32[3072, 768]", arg50_1: "f32[3072]", arg51_1: "f32[768, 3072]", arg52_1: "f32[768]", arg53_1: "f32[768]", arg54_1: "f32[768]", arg55_1: "f32[768]", arg56_1: "f32[768]", arg57_1: "f32[3072, 768]", arg58_1: "f32[3072]", arg59_1: "f32[768, 3072]", arg60_1: "f32[768]", arg61_1: "f32[768]", arg62_1: "f32[768]", arg63_1: "f32[768]", arg64_1: "f32[768]", arg65_1: "f32[3072, 768]", arg66_1: "f32[3072]", arg67_1: "f32[768, 3072]", arg68_1: "f32[768]", arg69_1: "f32[768]", arg70_1: "f32[768]", arg71_1: "f32[768]", arg72_1: "f32[768]", arg73_1: "f32[3072, 768]", arg74_1: "f32[3072]", arg75_1: "f32[768, 3072]", arg76_1: "f32[768]", arg77_1: "f32[768]", arg78_1: "f32[768]", arg79_1: "f32[768]", arg80_1: "f32[768]", arg81_1: "f32[3072, 768]", arg82_1: "f32[3072]", arg83_1: "f32[768, 3072]", arg84_1: "f32[768]", arg85_1: "f32[768]", arg86_1: "f32[768]", arg87_1: "f32[768]", arg88_1: "f32[768]", arg89_1: "f32[3072, 768]", arg90_1: "f32[3072]", arg91_1: "f32[768, 3072]", arg92_1: "f32[768]", arg93_1: "f32[768]", arg94_1: "f32[768]", arg95_1: "f32[768]", arg96_1: "f32[768]", arg97_1: "f32[3072, 768]", arg98_1: "f32[3072]", arg99_1: "f32[768, 3072]", arg100_1: "f32[768]", arg101_1: "f32[768]", arg102_1: "f32[768]", arg103_1: "f32[768, 768]", arg104_1: "f32[768]", arg105_1: "f32[768, 768]", arg106_1: "f32[768]", arg107_1: "f32[768]", arg108_1: "f32[768]", arg109_1: "f32[32000, 768]", arg110_1: "f32[32000]", arg111_1: "i64[1, 512]", arg112_1: "i64[1, 512]", arg113_1: "i64[1, 512]", arg114_1: "i64[1, 512]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:148, code: inputs_embeds = self.word_embeddings(input_ids)
    embedding: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg0_1, arg113_1, 3);  arg0_1 = arg113_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:587, code: buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
    expand: "i64[1, 512]" = torch.ops.aten.expand.default(arg111_1, [1, 512]);  arg111_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:149, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    embedding_1: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg1_1, expand);  arg1_1 = expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:151, code: embeddings = inputs_embeds + token_type_embeddings
    add: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:153, code: position_embeddings = self.position_embeddings(position_ids)
    embedding_2: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg2_1, arg112_1);  arg2_1 = arg112_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:154, code: embeddings += position_embeddings
    add_1: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:155, code: embeddings = self.LayerNorm(embeddings)
    var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 512, 1]" = var_mean[0]
    getitem_1: "f32[1, 512, 1]" = var_mean[1];  var_mean = None
    sub: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
    add_2: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
    rsqrt: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
    mul: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
    mul_1: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul, arg3_1);  mul = arg3_1 = None
    add_3: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_1, arg4_1);  mul_1 = arg4_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:156, code: embeddings = self.projection(embeddings)
    view: "f32[512, 768]" = torch.ops.aten.reshape.default(add_3, [512, 768]);  add_3 = None
    permute: "f32[768, 768]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
    
    # No stacktrace found for following nodes
    mm_default_25: "f32[512, 768]" = torch.ops.aten.mm.default(view, permute);  view = permute = None
    add_tensor_25: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_25, arg6_1);  mm_default_25 = arg6_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:156, code: embeddings = self.projection(embeddings)
    view_1: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_25, [1, 512, 768]);  add_tensor_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    convert_element_type: "c64[1, 512, 768]" = torch.ops.prims.convert_element_type.default(view_1, torch.complex64)
    _fft_c2c: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(convert_element_type, [1, 2], 0, True);  convert_element_type = None
    view_as_real: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c);  _fft_c2c = None
    select: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real, 3, 0);  view_as_real = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    add_4: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_1, select);  view_1 = select = None
    var_mean_1 = torch.ops.aten.var_mean.correction(add_4, [2], correction = 0, keepdim = True)
    getitem_2: "f32[1, 512, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 512, 1]" = var_mean_1[1];  var_mean_1 = None
    sub_1: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_4, getitem_3);  add_4 = getitem_3 = None
    add_5: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
    rsqrt_1: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
    mul_2: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
    mul_3: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_2, arg7_1);  mul_2 = arg7_1 = None
    add_6: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_3, arg8_1);  mul_3 = arg8_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_2: "f32[512, 768]" = torch.ops.aten.reshape.default(add_6, [512, 768])
    permute_1: "f32[768, 3072]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
    
    # No stacktrace found for following nodes
    mm_default_24: "f32[512, 3072]" = torch.ops.aten.mm.default(view_2, permute_1);  view_2 = permute_1 = None
    add_tensor_24: "f32[512, 3072]" = torch.ops.aten.add.Tensor(mm_default_24, arg10_1);  mm_default_24 = arg10_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_3: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(add_tensor_24, [1, 512, 3072]);  add_tensor_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_4: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_3, 0.5)
    pow_1: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_3, 3.0)
    mul_5: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
    add_7: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_3, mul_5);  view_3 = mul_5 = None
    mul_6: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_7, 0.7978845608028654);  add_7 = None
    tanh: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_6);  mul_6 = None
    add_8: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
    mul_7: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_4, add_8);  mul_4 = add_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_4: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_7, [512, 3072]);  mul_7 = None
    permute_2: "f32[3072, 768]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
    
    # No stacktrace found for following nodes
    mm_default_23: "f32[512, 768]" = torch.ops.aten.mm.default(view_4, permute_2);  view_4 = permute_2 = None
    add_tensor_23: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_23, arg12_1);  mm_default_23 = arg12_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_5: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_23, [1, 512, 768]);  add_tensor_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_9: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_5, add_6);  view_5 = add_6 = None
    var_mean_2 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
    getitem_4: "f32[1, 512, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 512, 1]" = var_mean_2[1];  var_mean_2 = None
    sub_2: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_9, getitem_5);  add_9 = getitem_5 = None
    add_10: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
    rsqrt_2: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
    mul_8: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
    mul_9: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_8, arg13_1);  mul_8 = arg13_1 = None
    add_11: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_9, arg14_1);  mul_9 = arg14_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    convert_element_type_1: "c64[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_11, torch.complex64)
    _fft_c2c_1: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(convert_element_type_1, [1, 2], 0, True);  convert_element_type_1 = None
    view_as_real_1: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_1);  _fft_c2c_1 = None
    select_1: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_1, 3, 0);  view_as_real_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    add_12: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_11, select_1);  add_11 = select_1 = None
    var_mean_3 = torch.ops.aten.var_mean.correction(add_12, [2], correction = 0, keepdim = True)
    getitem_6: "f32[1, 512, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 512, 1]" = var_mean_3[1];  var_mean_3 = None
    sub_3: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_12, getitem_7);  add_12 = getitem_7 = None
    add_13: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
    rsqrt_3: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
    mul_10: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = rsqrt_3 = None
    mul_11: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_10, arg15_1);  mul_10 = arg15_1 = None
    add_14: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_11, arg16_1);  mul_11 = arg16_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_6: "f32[512, 768]" = torch.ops.aten.reshape.default(add_14, [512, 768])
    permute_3: "f32[768, 3072]" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
    
    # No stacktrace found for following nodes
    mm_default_22: "f32[512, 3072]" = torch.ops.aten.mm.default(view_6, permute_3);  view_6 = permute_3 = None
    add_tensor_22: "f32[512, 3072]" = torch.ops.aten.add.Tensor(mm_default_22, arg18_1);  mm_default_22 = arg18_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_7: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(add_tensor_22, [1, 512, 3072]);  add_tensor_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_12: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_7, 0.5)
    pow_2: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_7, 3.0)
    mul_13: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
    add_15: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_7, mul_13);  view_7 = mul_13 = None
    mul_14: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_15, 0.7978845608028654);  add_15 = None
    tanh_1: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_14);  mul_14 = None
    add_16: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
    mul_15: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_12, add_16);  mul_12 = add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_8: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_15, [512, 3072]);  mul_15 = None
    permute_4: "f32[3072, 768]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
    
    # No stacktrace found for following nodes
    mm_default_21: "f32[512, 768]" = torch.ops.aten.mm.default(view_8, permute_4);  view_8 = permute_4 = None
    add_tensor_21: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_21, arg20_1);  mm_default_21 = arg20_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_9: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_21, [1, 512, 768]);  add_tensor_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_17: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_9, add_14);  view_9 = add_14 = None
    var_mean_4 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
    getitem_8: "f32[1, 512, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 512, 1]" = var_mean_4[1];  var_mean_4 = None
    sub_4: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_17, getitem_9);  add_17 = getitem_9 = None
    add_18: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
    rsqrt_4: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    mul_16: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = rsqrt_4 = None
    mul_17: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_16, arg21_1);  mul_16 = arg21_1 = None
    add_19: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_17, arg22_1);  mul_17 = arg22_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    convert_element_type_2: "c64[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_19, torch.complex64)
    _fft_c2c_2: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(convert_element_type_2, [1, 2], 0, True);  convert_element_type_2 = None
    view_as_real_2: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_2);  _fft_c2c_2 = None
    select_2: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_2, 3, 0);  view_as_real_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    add_20: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_19, select_2);  add_19 = select_2 = None
    var_mean_5 = torch.ops.aten.var_mean.correction(add_20, [2], correction = 0, keepdim = True)
    getitem_10: "f32[1, 512, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 512, 1]" = var_mean_5[1];  var_mean_5 = None
    sub_5: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_20, getitem_11);  add_20 = getitem_11 = None
    add_21: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
    rsqrt_5: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    mul_18: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = rsqrt_5 = None
    mul_19: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_18, arg23_1);  mul_18 = arg23_1 = None
    add_22: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_19, arg24_1);  mul_19 = arg24_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_10: "f32[512, 768]" = torch.ops.aten.reshape.default(add_22, [512, 768])
    permute_5: "f32[768, 3072]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
    
    # No stacktrace found for following nodes
    mm_default_20: "f32[512, 3072]" = torch.ops.aten.mm.default(view_10, permute_5);  view_10 = permute_5 = None
    add_tensor_20: "f32[512, 3072]" = torch.ops.aten.add.Tensor(mm_default_20, arg26_1);  mm_default_20 = arg26_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_11: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(add_tensor_20, [1, 512, 3072]);  add_tensor_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_20: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_11, 0.5)
    pow_3: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_11, 3.0)
    mul_21: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
    add_23: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_11, mul_21);  view_11 = mul_21 = None
    mul_22: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_23, 0.7978845608028654);  add_23 = None
    tanh_2: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
    add_24: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
    mul_23: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_20, add_24);  mul_20 = add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_12: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_23, [512, 3072]);  mul_23 = None
    permute_6: "f32[3072, 768]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
    
    # No stacktrace found for following nodes
    mm_default_19: "f32[512, 768]" = torch.ops.aten.mm.default(view_12, permute_6);  view_12 = permute_6 = None
    add_tensor_19: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_19, arg28_1);  mm_default_19 = arg28_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_13: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_19, [1, 512, 768]);  add_tensor_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_25: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_13, add_22);  view_13 = add_22 = None
    var_mean_6 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
    getitem_12: "f32[1, 512, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 512, 1]" = var_mean_6[1];  var_mean_6 = None
    sub_6: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_25, getitem_13);  add_25 = getitem_13 = None
    add_26: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
    rsqrt_6: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
    mul_24: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = rsqrt_6 = None
    mul_25: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_24, arg29_1);  mul_24 = arg29_1 = None
    add_27: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_25, arg30_1);  mul_25 = arg30_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    convert_element_type_3: "c64[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_27, torch.complex64)
    _fft_c2c_3: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(convert_element_type_3, [1, 2], 0, True);  convert_element_type_3 = None
    view_as_real_3: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_3);  _fft_c2c_3 = None
    select_3: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_3, 3, 0);  view_as_real_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    add_28: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_27, select_3);  add_27 = select_3 = None
    var_mean_7 = torch.ops.aten.var_mean.correction(add_28, [2], correction = 0, keepdim = True)
    getitem_14: "f32[1, 512, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 512, 1]" = var_mean_7[1];  var_mean_7 = None
    sub_7: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_28, getitem_15);  add_28 = getitem_15 = None
    add_29: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
    rsqrt_7: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
    mul_26: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = rsqrt_7 = None
    mul_27: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_26, arg31_1);  mul_26 = arg31_1 = None
    add_30: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_27, arg32_1);  mul_27 = arg32_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_14: "f32[512, 768]" = torch.ops.aten.reshape.default(add_30, [512, 768])
    permute_7: "f32[768, 3072]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
    
    # No stacktrace found for following nodes
    mm_default_18: "f32[512, 3072]" = torch.ops.aten.mm.default(view_14, permute_7);  view_14 = permute_7 = None
    add_tensor_18: "f32[512, 3072]" = torch.ops.aten.add.Tensor(mm_default_18, arg34_1);  mm_default_18 = arg34_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_15: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(add_tensor_18, [1, 512, 3072]);  add_tensor_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_28: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_15, 0.5)
    pow_4: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_15, 3.0)
    mul_29: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
    add_31: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_15, mul_29);  view_15 = mul_29 = None
    mul_30: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_31, 0.7978845608028654);  add_31 = None
    tanh_3: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_30);  mul_30 = None
    add_32: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
    mul_31: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_28, add_32);  mul_28 = add_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_16: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_31, [512, 3072]);  mul_31 = None
    permute_8: "f32[3072, 768]" = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
    
    # No stacktrace found for following nodes
    mm_default_17: "f32[512, 768]" = torch.ops.aten.mm.default(view_16, permute_8);  view_16 = permute_8 = None
    add_tensor_17: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_17, arg36_1);  mm_default_17 = arg36_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_17: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_17, [1, 512, 768]);  add_tensor_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_33: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_17, add_30);  view_17 = add_30 = None
    var_mean_8 = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
    getitem_16: "f32[1, 512, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 512, 1]" = var_mean_8[1];  var_mean_8 = None
    sub_8: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_33, getitem_17);  add_33 = getitem_17 = None
    add_34: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
    rsqrt_8: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
    mul_32: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = rsqrt_8 = None
    mul_33: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_32, arg37_1);  mul_32 = arg37_1 = None
    add_35: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_33, arg38_1);  mul_33 = arg38_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    convert_element_type_4: "c64[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_35, torch.complex64)
    _fft_c2c_4: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(convert_element_type_4, [1, 2], 0, True);  convert_element_type_4 = None
    view_as_real_4: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_4);  _fft_c2c_4 = None
    select_4: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_4, 3, 0);  view_as_real_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    add_36: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_35, select_4);  add_35 = select_4 = None
    var_mean_9 = torch.ops.aten.var_mean.correction(add_36, [2], correction = 0, keepdim = True)
    getitem_18: "f32[1, 512, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 512, 1]" = var_mean_9[1];  var_mean_9 = None
    sub_9: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_36, getitem_19);  add_36 = getitem_19 = None
    add_37: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
    rsqrt_9: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
    mul_34: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = rsqrt_9 = None
    mul_35: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_34, arg39_1);  mul_34 = arg39_1 = None
    add_38: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_35, arg40_1);  mul_35 = arg40_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_18: "f32[512, 768]" = torch.ops.aten.reshape.default(add_38, [512, 768])
    permute_9: "f32[768, 3072]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
    
    # No stacktrace found for following nodes
    mm_default_16: "f32[512, 3072]" = torch.ops.aten.mm.default(view_18, permute_9);  view_18 = permute_9 = None
    add_tensor_16: "f32[512, 3072]" = torch.ops.aten.add.Tensor(mm_default_16, arg42_1);  mm_default_16 = arg42_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_19: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(add_tensor_16, [1, 512, 3072]);  add_tensor_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_36: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.5)
    pow_5: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_19, 3.0)
    mul_37: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
    add_39: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_19, mul_37);  view_19 = mul_37 = None
    mul_38: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_39, 0.7978845608028654);  add_39 = None
    tanh_4: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
    add_40: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
    mul_39: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_36, add_40);  mul_36 = add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_20: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_39, [512, 3072]);  mul_39 = None
    permute_10: "f32[3072, 768]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
    
    # No stacktrace found for following nodes
    mm_default_15: "f32[512, 768]" = torch.ops.aten.mm.default(view_20, permute_10);  view_20 = permute_10 = None
    add_tensor_15: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_15, arg44_1);  mm_default_15 = arg44_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_21: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_15, [1, 512, 768]);  add_tensor_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_41: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_21, add_38);  view_21 = add_38 = None
    var_mean_10 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
    getitem_20: "f32[1, 512, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 512, 1]" = var_mean_10[1];  var_mean_10 = None
    sub_10: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_41, getitem_21);  add_41 = getitem_21 = None
    add_42: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
    rsqrt_10: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    mul_40: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = rsqrt_10 = None
    mul_41: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_40, arg45_1);  mul_40 = arg45_1 = None
    add_43: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_41, arg46_1);  mul_41 = arg46_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    convert_element_type_5: "c64[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_43, torch.complex64)
    _fft_c2c_5: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(convert_element_type_5, [1, 2], 0, True);  convert_element_type_5 = None
    view_as_real_5: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_5);  _fft_c2c_5 = None
    select_5: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_5, 3, 0);  view_as_real_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    add_44: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_43, select_5);  add_43 = select_5 = None
    var_mean_11 = torch.ops.aten.var_mean.correction(add_44, [2], correction = 0, keepdim = True)
    getitem_22: "f32[1, 512, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 512, 1]" = var_mean_11[1];  var_mean_11 = None
    sub_11: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_44, getitem_23);  add_44 = getitem_23 = None
    add_45: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
    rsqrt_11: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
    mul_42: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = rsqrt_11 = None
    mul_43: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_42, arg47_1);  mul_42 = arg47_1 = None
    add_46: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_43, arg48_1);  mul_43 = arg48_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_22: "f32[512, 768]" = torch.ops.aten.reshape.default(add_46, [512, 768])
    permute_11: "f32[768, 3072]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
    
    # No stacktrace found for following nodes
    mm_default_14: "f32[512, 3072]" = torch.ops.aten.mm.default(view_22, permute_11);  view_22 = permute_11 = None
    add_tensor_14: "f32[512, 3072]" = torch.ops.aten.add.Tensor(mm_default_14, arg50_1);  mm_default_14 = arg50_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_23: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(add_tensor_14, [1, 512, 3072]);  add_tensor_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_44: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_23, 0.5)
    pow_6: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_23, 3.0)
    mul_45: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
    add_47: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_23, mul_45);  view_23 = mul_45 = None
    mul_46: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_47, 0.7978845608028654);  add_47 = None
    tanh_5: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
    add_48: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
    mul_47: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_44, add_48);  mul_44 = add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_24: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_47, [512, 3072]);  mul_47 = None
    permute_12: "f32[3072, 768]" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
    
    # No stacktrace found for following nodes
    mm_default_13: "f32[512, 768]" = torch.ops.aten.mm.default(view_24, permute_12);  view_24 = permute_12 = None
    add_tensor_13: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_13, arg52_1);  mm_default_13 = arg52_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_25: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_13, [1, 512, 768]);  add_tensor_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_49: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_25, add_46);  view_25 = add_46 = None
    var_mean_12 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
    getitem_24: "f32[1, 512, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 512, 1]" = var_mean_12[1];  var_mean_12 = None
    sub_12: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_49, getitem_25);  add_49 = getitem_25 = None
    add_50: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
    rsqrt_12: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
    mul_48: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = rsqrt_12 = None
    mul_49: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_48, arg53_1);  mul_48 = arg53_1 = None
    add_51: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_49, arg54_1);  mul_49 = arg54_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    convert_element_type_6: "c64[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_51, torch.complex64)
    _fft_c2c_6: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(convert_element_type_6, [1, 2], 0, True);  convert_element_type_6 = None
    view_as_real_6: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_6);  _fft_c2c_6 = None
    select_6: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_6, 3, 0);  view_as_real_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    add_52: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_51, select_6);  add_51 = select_6 = None
    var_mean_13 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
    getitem_26: "f32[1, 512, 1]" = var_mean_13[0]
    getitem_27: "f32[1, 512, 1]" = var_mean_13[1];  var_mean_13 = None
    sub_13: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_52, getitem_27);  add_52 = getitem_27 = None
    add_53: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
    rsqrt_13: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    mul_50: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = rsqrt_13 = None
    mul_51: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_50, arg55_1);  mul_50 = arg55_1 = None
    add_54: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_51, arg56_1);  mul_51 = arg56_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_26: "f32[512, 768]" = torch.ops.aten.reshape.default(add_54, [512, 768])
    permute_13: "f32[768, 3072]" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
    
    # No stacktrace found for following nodes
    mm_default_12: "f32[512, 3072]" = torch.ops.aten.mm.default(view_26, permute_13);  view_26 = permute_13 = None
    add_tensor_12: "f32[512, 3072]" = torch.ops.aten.add.Tensor(mm_default_12, arg58_1);  mm_default_12 = arg58_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_27: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(add_tensor_12, [1, 512, 3072]);  add_tensor_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_52: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_27, 0.5)
    pow_7: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_27, 3.0)
    mul_53: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
    add_55: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_27, mul_53);  view_27 = mul_53 = None
    mul_54: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_55, 0.7978845608028654);  add_55 = None
    tanh_6: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_54);  mul_54 = None
    add_56: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
    mul_55: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_52, add_56);  mul_52 = add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_28: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_55, [512, 3072]);  mul_55 = None
    permute_14: "f32[3072, 768]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
    
    # No stacktrace found for following nodes
    mm_default_11: "f32[512, 768]" = torch.ops.aten.mm.default(view_28, permute_14);  view_28 = permute_14 = None
    add_tensor_11: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_11, arg60_1);  mm_default_11 = arg60_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_29: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_11, [1, 512, 768]);  add_tensor_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_57: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_29, add_54);  view_29 = add_54 = None
    var_mean_14 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 512, 1]" = var_mean_14[0]
    getitem_29: "f32[1, 512, 1]" = var_mean_14[1];  var_mean_14 = None
    sub_14: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_57, getitem_29);  add_57 = getitem_29 = None
    add_58: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
    rsqrt_14: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
    mul_56: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = rsqrt_14 = None
    mul_57: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_56, arg61_1);  mul_56 = arg61_1 = None
    add_59: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_57, arg62_1);  mul_57 = arg62_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    convert_element_type_7: "c64[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_59, torch.complex64)
    _fft_c2c_7: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(convert_element_type_7, [1, 2], 0, True);  convert_element_type_7 = None
    view_as_real_7: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_7);  _fft_c2c_7 = None
    select_7: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_7, 3, 0);  view_as_real_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    add_60: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_59, select_7);  add_59 = select_7 = None
    var_mean_15 = torch.ops.aten.var_mean.correction(add_60, [2], correction = 0, keepdim = True)
    getitem_30: "f32[1, 512, 1]" = var_mean_15[0]
    getitem_31: "f32[1, 512, 1]" = var_mean_15[1];  var_mean_15 = None
    sub_15: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_60, getitem_31);  add_60 = getitem_31 = None
    add_61: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
    rsqrt_15: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
    mul_58: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = rsqrt_15 = None
    mul_59: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_58, arg63_1);  mul_58 = arg63_1 = None
    add_62: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_59, arg64_1);  mul_59 = arg64_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_30: "f32[512, 768]" = torch.ops.aten.reshape.default(add_62, [512, 768])
    permute_15: "f32[768, 3072]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
    
    # No stacktrace found for following nodes
    mm_default_10: "f32[512, 3072]" = torch.ops.aten.mm.default(view_30, permute_15);  view_30 = permute_15 = None
    add_tensor_10: "f32[512, 3072]" = torch.ops.aten.add.Tensor(mm_default_10, arg66_1);  mm_default_10 = arg66_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_31: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(add_tensor_10, [1, 512, 3072]);  add_tensor_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_60: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_31, 0.5)
    pow_8: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_31, 3.0)
    mul_61: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
    add_63: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_31, mul_61);  view_31 = mul_61 = None
    mul_62: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_63, 0.7978845608028654);  add_63 = None
    tanh_7: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_62);  mul_62 = None
    add_64: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
    mul_63: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_60, add_64);  mul_60 = add_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_32: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_63, [512, 3072]);  mul_63 = None
    permute_16: "f32[3072, 768]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
    
    # No stacktrace found for following nodes
    mm_default_9: "f32[512, 768]" = torch.ops.aten.mm.default(view_32, permute_16);  view_32 = permute_16 = None
    add_tensor_9: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_9, arg68_1);  mm_default_9 = arg68_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_33: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_9, [1, 512, 768]);  add_tensor_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_65: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_33, add_62);  view_33 = add_62 = None
    var_mean_16 = torch.ops.aten.var_mean.correction(add_65, [2], correction = 0, keepdim = True)
    getitem_32: "f32[1, 512, 1]" = var_mean_16[0]
    getitem_33: "f32[1, 512, 1]" = var_mean_16[1];  var_mean_16 = None
    sub_16: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_65, getitem_33);  add_65 = getitem_33 = None
    add_66: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
    rsqrt_16: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    mul_64: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = rsqrt_16 = None
    mul_65: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_64, arg69_1);  mul_64 = arg69_1 = None
    add_67: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_65, arg70_1);  mul_65 = arg70_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    convert_element_type_8: "c64[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_67, torch.complex64)
    _fft_c2c_8: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(convert_element_type_8, [1, 2], 0, True);  convert_element_type_8 = None
    view_as_real_8: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_8);  _fft_c2c_8 = None
    select_8: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_8, 3, 0);  view_as_real_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    add_68: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_67, select_8);  add_67 = select_8 = None
    var_mean_17 = torch.ops.aten.var_mean.correction(add_68, [2], correction = 0, keepdim = True)
    getitem_34: "f32[1, 512, 1]" = var_mean_17[0]
    getitem_35: "f32[1, 512, 1]" = var_mean_17[1];  var_mean_17 = None
    sub_17: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_68, getitem_35);  add_68 = getitem_35 = None
    add_69: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
    rsqrt_17: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
    mul_66: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = rsqrt_17 = None
    mul_67: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_66, arg71_1);  mul_66 = arg71_1 = None
    add_70: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_67, arg72_1);  mul_67 = arg72_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_34: "f32[512, 768]" = torch.ops.aten.reshape.default(add_70, [512, 768])
    permute_17: "f32[768, 3072]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
    
    # No stacktrace found for following nodes
    mm_default_8: "f32[512, 3072]" = torch.ops.aten.mm.default(view_34, permute_17);  view_34 = permute_17 = None
    add_tensor_8: "f32[512, 3072]" = torch.ops.aten.add.Tensor(mm_default_8, arg74_1);  mm_default_8 = arg74_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_35: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(add_tensor_8, [1, 512, 3072]);  add_tensor_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_68: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_35, 0.5)
    pow_9: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_35, 3.0)
    mul_69: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
    add_71: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_35, mul_69);  view_35 = mul_69 = None
    mul_70: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_71, 0.7978845608028654);  add_71 = None
    tanh_8: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_70);  mul_70 = None
    add_72: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
    mul_71: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_68, add_72);  mul_68 = add_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_36: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_71, [512, 3072]);  mul_71 = None
    permute_18: "f32[3072, 768]" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
    
    # No stacktrace found for following nodes
    mm_default_7: "f32[512, 768]" = torch.ops.aten.mm.default(view_36, permute_18);  view_36 = permute_18 = None
    add_tensor_7: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_7, arg76_1);  mm_default_7 = arg76_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_37: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_7, [1, 512, 768]);  add_tensor_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_73: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_37, add_70);  view_37 = add_70 = None
    var_mean_18 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
    getitem_36: "f32[1, 512, 1]" = var_mean_18[0]
    getitem_37: "f32[1, 512, 1]" = var_mean_18[1];  var_mean_18 = None
    sub_18: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_73, getitem_37);  add_73 = getitem_37 = None
    add_74: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
    rsqrt_18: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    mul_72: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = rsqrt_18 = None
    mul_73: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_72, arg77_1);  mul_72 = arg77_1 = None
    add_75: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_73, arg78_1);  mul_73 = arg78_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    convert_element_type_9: "c64[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_75, torch.complex64)
    _fft_c2c_9: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(convert_element_type_9, [1, 2], 0, True);  convert_element_type_9 = None
    view_as_real_9: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_9);  _fft_c2c_9 = None
    select_9: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_9, 3, 0);  view_as_real_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    add_76: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_75, select_9);  add_75 = select_9 = None
    var_mean_19 = torch.ops.aten.var_mean.correction(add_76, [2], correction = 0, keepdim = True)
    getitem_38: "f32[1, 512, 1]" = var_mean_19[0]
    getitem_39: "f32[1, 512, 1]" = var_mean_19[1];  var_mean_19 = None
    sub_19: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_76, getitem_39);  add_76 = getitem_39 = None
    add_77: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
    rsqrt_19: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
    mul_74: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = rsqrt_19 = None
    mul_75: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_74, arg79_1);  mul_74 = arg79_1 = None
    add_78: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_75, arg80_1);  mul_75 = arg80_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_38: "f32[512, 768]" = torch.ops.aten.reshape.default(add_78, [512, 768])
    permute_19: "f32[768, 3072]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
    
    # No stacktrace found for following nodes
    mm_default_6: "f32[512, 3072]" = torch.ops.aten.mm.default(view_38, permute_19);  view_38 = permute_19 = None
    add_tensor_6: "f32[512, 3072]" = torch.ops.aten.add.Tensor(mm_default_6, arg82_1);  mm_default_6 = arg82_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_39: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(add_tensor_6, [1, 512, 3072]);  add_tensor_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_76: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_39, 0.5)
    pow_10: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_39, 3.0)
    mul_77: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
    add_79: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_39, mul_77);  view_39 = mul_77 = None
    mul_78: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_79, 0.7978845608028654);  add_79 = None
    tanh_9: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
    add_80: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
    mul_79: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_76, add_80);  mul_76 = add_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_40: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_79, [512, 3072]);  mul_79 = None
    permute_20: "f32[3072, 768]" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
    
    # No stacktrace found for following nodes
    mm_default_5: "f32[512, 768]" = torch.ops.aten.mm.default(view_40, permute_20);  view_40 = permute_20 = None
    add_tensor_5: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_5, arg84_1);  mm_default_5 = arg84_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_41: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_5, [1, 512, 768]);  add_tensor_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_81: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_41, add_78);  view_41 = add_78 = None
    var_mean_20 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
    getitem_40: "f32[1, 512, 1]" = var_mean_20[0]
    getitem_41: "f32[1, 512, 1]" = var_mean_20[1];  var_mean_20 = None
    sub_20: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_81, getitem_41);  add_81 = getitem_41 = None
    add_82: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
    rsqrt_20: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
    mul_80: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = rsqrt_20 = None
    mul_81: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_80, arg85_1);  mul_80 = arg85_1 = None
    add_83: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_81, arg86_1);  mul_81 = arg86_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    convert_element_type_10: "c64[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_83, torch.complex64)
    _fft_c2c_10: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(convert_element_type_10, [1, 2], 0, True);  convert_element_type_10 = None
    view_as_real_10: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_10);  _fft_c2c_10 = None
    select_10: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_10, 3, 0);  view_as_real_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    add_84: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_83, select_10);  add_83 = select_10 = None
    var_mean_21 = torch.ops.aten.var_mean.correction(add_84, [2], correction = 0, keepdim = True)
    getitem_42: "f32[1, 512, 1]" = var_mean_21[0]
    getitem_43: "f32[1, 512, 1]" = var_mean_21[1];  var_mean_21 = None
    sub_21: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_84, getitem_43);  add_84 = getitem_43 = None
    add_85: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
    rsqrt_21: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
    mul_82: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = rsqrt_21 = None
    mul_83: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_82, arg87_1);  mul_82 = arg87_1 = None
    add_86: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_83, arg88_1);  mul_83 = arg88_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_42: "f32[512, 768]" = torch.ops.aten.reshape.default(add_86, [512, 768])
    permute_21: "f32[768, 3072]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
    
    # No stacktrace found for following nodes
    mm_default_4: "f32[512, 3072]" = torch.ops.aten.mm.default(view_42, permute_21);  view_42 = permute_21 = None
    add_tensor_4: "f32[512, 3072]" = torch.ops.aten.add.Tensor(mm_default_4, arg90_1);  mm_default_4 = arg90_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_43: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(add_tensor_4, [1, 512, 3072]);  add_tensor_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_84: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_43, 0.5)
    pow_11: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_43, 3.0)
    mul_85: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
    add_87: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_43, mul_85);  view_43 = mul_85 = None
    mul_86: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_87, 0.7978845608028654);  add_87 = None
    tanh_10: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_86);  mul_86 = None
    add_88: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
    mul_87: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_84, add_88);  mul_84 = add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_44: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_87, [512, 3072]);  mul_87 = None
    permute_22: "f32[3072, 768]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
    
    # No stacktrace found for following nodes
    mm_default_3: "f32[512, 768]" = torch.ops.aten.mm.default(view_44, permute_22);  view_44 = permute_22 = None
    add_tensor_3: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_3, arg92_1);  mm_default_3 = arg92_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_45: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_3, [1, 512, 768]);  add_tensor_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_89: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_45, add_86);  view_45 = add_86 = None
    var_mean_22 = torch.ops.aten.var_mean.correction(add_89, [2], correction = 0, keepdim = True)
    getitem_44: "f32[1, 512, 1]" = var_mean_22[0]
    getitem_45: "f32[1, 512, 1]" = var_mean_22[1];  var_mean_22 = None
    sub_22: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_89, getitem_45);  add_89 = getitem_45 = None
    add_90: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
    rsqrt_22: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
    mul_88: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = rsqrt_22 = None
    mul_89: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_88, arg93_1);  mul_88 = arg93_1 = None
    add_91: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_89, arg94_1);  mul_89 = arg94_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    convert_element_type_11: "c64[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_91, torch.complex64)
    _fft_c2c_11: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(convert_element_type_11, [1, 2], 0, True);  convert_element_type_11 = None
    view_as_real_11: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_11);  _fft_c2c_11 = None
    select_11: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_11, 3, 0);  view_as_real_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    add_92: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_91, select_11);  add_91 = select_11 = None
    var_mean_23 = torch.ops.aten.var_mean.correction(add_92, [2], correction = 0, keepdim = True)
    getitem_46: "f32[1, 512, 1]" = var_mean_23[0]
    getitem_47: "f32[1, 512, 1]" = var_mean_23[1];  var_mean_23 = None
    sub_23: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_92, getitem_47);  add_92 = getitem_47 = None
    add_93: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
    rsqrt_23: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
    mul_90: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = rsqrt_23 = None
    mul_91: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_90, arg95_1);  mul_90 = arg95_1 = None
    add_94: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_91, arg96_1);  mul_91 = arg96_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_46: "f32[512, 768]" = torch.ops.aten.reshape.default(add_94, [512, 768])
    permute_23: "f32[768, 3072]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
    
    # No stacktrace found for following nodes
    mm_default_2: "f32[512, 3072]" = torch.ops.aten.mm.default(view_46, permute_23);  view_46 = permute_23 = None
    add_tensor_2: "f32[512, 3072]" = torch.ops.aten.add.Tensor(mm_default_2, arg98_1);  mm_default_2 = arg98_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_47: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(add_tensor_2, [1, 512, 3072]);  add_tensor_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_92: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_47, 0.5)
    pow_12: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_47, 3.0)
    mul_93: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
    add_95: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_47, mul_93);  view_47 = mul_93 = None
    mul_94: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_95, 0.7978845608028654);  add_95 = None
    tanh_11: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_94);  mul_94 = None
    add_96: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
    mul_95: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_92, add_96);  mul_92 = add_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_48: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_95, [512, 3072]);  mul_95 = None
    permute_24: "f32[3072, 768]" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
    
    # No stacktrace found for following nodes
    mm_default_1: "f32[512, 768]" = torch.ops.aten.mm.default(view_48, permute_24);  view_48 = permute_24 = None
    add_tensor_1: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_1, arg100_1);  mm_default_1 = arg100_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_49: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_1, [1, 512, 768]);  add_tensor_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_97: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_49, add_94);  view_49 = add_94 = None
    var_mean_24 = torch.ops.aten.var_mean.correction(add_97, [2], correction = 0, keepdim = True)
    getitem_48: "f32[1, 512, 1]" = var_mean_24[0]
    getitem_49: "f32[1, 512, 1]" = var_mean_24[1];  var_mean_24 = None
    sub_24: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_97, getitem_49);  add_97 = getitem_49 = None
    add_98: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
    rsqrt_24: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
    mul_96: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = rsqrt_24 = None
    mul_97: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_96, arg101_1);  mul_96 = arg101_1 = None
    add_99: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_97, arg102_1);  mul_97 = arg102_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:345, code: hidden_states = self.dense(hidden_states)
    view_50: "f32[512, 768]" = torch.ops.aten.reshape.default(add_99, [512, 768]);  add_99 = None
    permute_26: "f32[768, 768]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
    
    # No stacktrace found for following nodes
    mm_default: "f32[512, 768]" = torch.ops.aten.mm.default(view_50, permute_26);  view_50 = permute_26 = None
    add_tensor: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default, arg106_1);  mm_default = arg106_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:345, code: hidden_states = self.dense(hidden_states)
    view_51: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor, [1, 512, 768]);  add_tensor = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_98: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_51, 0.5)
    pow_13: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(view_51, 3.0)
    mul_99: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(pow_13, 0.044715);  pow_13 = None
    add_100: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_51, mul_99);  view_51 = mul_99 = None
    mul_100: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_100, 0.7978845608028654);  add_100 = None
    tanh_13: "f32[1, 512, 768]" = torch.ops.aten.tanh.default(mul_100);  mul_100 = None
    add_101: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(tanh_13, 1.0);  tanh_13 = None
    mul_101: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_98, add_101);  mul_98 = add_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:347, code: hidden_states = self.LayerNorm(hidden_states)
    var_mean_25 = torch.ops.aten.var_mean.correction(mul_101, [2], correction = 0, keepdim = True)
    getitem_50: "f32[1, 512, 1]" = var_mean_25[0]
    getitem_51: "f32[1, 512, 1]" = var_mean_25[1];  var_mean_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:775, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    view_55: "i64[512]" = torch.ops.aten.reshape.default(arg114_1, [-1]);  arg114_1 = None
    ne_1: "b8[512]" = torch.ops.aten.ne.Scalar(view_55, -100)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:347, code: hidden_states = self.LayerNorm(hidden_states)
    sub_25: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_101, getitem_51);  mul_101 = getitem_51 = None
    add_102: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
    rsqrt_25: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
    mul_102: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = rsqrt_25 = None
    mul_103: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_102, arg107_1);  mul_102 = arg107_1 = None
    add_103: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_103, arg108_1);  mul_103 = arg108_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:365, code: hidden_states = self.decoder(hidden_states)
    view_52: "f32[512, 768]" = torch.ops.aten.reshape.default(add_103, [512, 768]);  add_103 = None
    permute_27: "f32[768, 32000]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
    addmm_27: "f32[512, 32000]" = torch.ops.aten.addmm.default(arg110_1, view_52, permute_27);  arg110_1 = view_52 = permute_27 = None
    view_53: "f32[1, 512, 32000]" = torch.ops.aten.reshape.default(addmm_27, [1, 512, 32000]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:775, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    view_54: "f32[512, 32000]" = torch.ops.aten.reshape.default(view_53, [-1, 32000])
    amax: "f32[512, 1]" = torch.ops.aten.amax.default(view_54, [1], True)
    sub_26: "f32[512, 32000]" = torch.ops.aten.sub.Tensor(view_54, amax);  view_54 = amax = None
    exp: "f32[512, 32000]" = torch.ops.aten.exp.default(sub_26)
    sum_1: "f32[512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
    log: "f32[512, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
    sub_27: "f32[512, 32000]" = torch.ops.aten.sub.Tensor(sub_26, log);  sub_26 = log = None
    ne: "b8[512]" = torch.ops.aten.ne.Scalar(view_55, -100)
    full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where: "i64[512]" = torch.ops.aten.where.self(ne, view_55, full_default);  ne = full_default = None
    unsqueeze: "i64[512, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
    gather: "f32[512, 1]" = torch.ops.aten.gather.default(sub_27, 1, unsqueeze);  sub_27 = unsqueeze = None
    squeeze: "f32[512]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[512]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
    full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where_1: "f32[512]" = torch.ops.aten.where.self(ne_1, neg, full_default_1);  ne_1 = neg = full_default_1 = None
    sum_3: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
    ne_2: "b8[512]" = torch.ops.aten.ne.Scalar(view_55, -100);  view_55 = None
    sum_2: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
    convert_element_type_12: "f32[]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
    div: "f32[]" = torch.ops.aten.div.Tensor(sum_3, convert_element_type_12);  sum_3 = convert_element_type_12 = None
    return (div, view_53)
    