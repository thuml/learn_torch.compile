from __future__ import annotations



def forward(self, primals_1: "f32[32000, 768]", primals_2: "f32[4, 768]", primals_3: "f32[512, 768]", primals_4: "f32[768]", primals_5: "f32[768]", primals_6: "f32[768, 768]", primals_7: "f32[768]", primals_8: "f32[768]", primals_9: "f32[768]", primals_10: "f32[3072, 768]", primals_11: "f32[3072]", primals_12: "f32[768, 3072]", primals_13: "f32[768]", primals_14: "f32[768]", primals_15: "f32[768]", primals_16: "f32[768]", primals_17: "f32[768]", primals_18: "f32[3072, 768]", primals_19: "f32[3072]", primals_20: "f32[768, 3072]", primals_21: "f32[768]", primals_22: "f32[768]", primals_23: "f32[768]", primals_24: "f32[768]", primals_25: "f32[768]", primals_26: "f32[3072, 768]", primals_27: "f32[3072]", primals_28: "f32[768, 3072]", primals_29: "f32[768]", primals_30: "f32[768]", primals_31: "f32[768]", primals_32: "f32[768]", primals_33: "f32[768]", primals_34: "f32[3072, 768]", primals_35: "f32[3072]", primals_36: "f32[768, 3072]", primals_37: "f32[768]", primals_38: "f32[768]", primals_39: "f32[768]", primals_40: "f32[768]", primals_41: "f32[768]", primals_42: "f32[3072, 768]", primals_43: "f32[3072]", primals_44: "f32[768, 3072]", primals_45: "f32[768]", primals_46: "f32[768]", primals_47: "f32[768]", primals_48: "f32[768]", primals_49: "f32[768]", primals_50: "f32[3072, 768]", primals_51: "f32[3072]", primals_52: "f32[768, 3072]", primals_53: "f32[768]", primals_54: "f32[768]", primals_55: "f32[768]", primals_56: "f32[768]", primals_57: "f32[768]", primals_58: "f32[3072, 768]", primals_59: "f32[3072]", primals_60: "f32[768, 3072]", primals_61: "f32[768]", primals_62: "f32[768]", primals_63: "f32[768]", primals_64: "f32[768]", primals_65: "f32[768]", primals_66: "f32[3072, 768]", primals_67: "f32[3072]", primals_68: "f32[768, 3072]", primals_69: "f32[768]", primals_70: "f32[768]", primals_71: "f32[768]", primals_72: "f32[768]", primals_73: "f32[768]", primals_74: "f32[3072, 768]", primals_75: "f32[3072]", primals_76: "f32[768, 3072]", primals_77: "f32[768]", primals_78: "f32[768]", primals_79: "f32[768]", primals_80: "f32[768]", primals_81: "f32[768]", primals_82: "f32[3072, 768]", primals_83: "f32[3072]", primals_84: "f32[768, 3072]", primals_85: "f32[768]", primals_86: "f32[768]", primals_87: "f32[768]", primals_88: "f32[768]", primals_89: "f32[768]", primals_90: "f32[3072, 768]", primals_91: "f32[3072]", primals_92: "f32[768, 3072]", primals_93: "f32[768]", primals_94: "f32[768]", primals_95: "f32[768]", primals_96: "f32[768]", primals_97: "f32[768]", primals_98: "f32[3072, 768]", primals_99: "f32[3072]", primals_100: "f32[768, 3072]", primals_101: "f32[768]", primals_102: "f32[768]", primals_103: "f32[768]", primals_104: "f32[768, 768]", primals_105: "f32[768]", primals_106: "f32[768, 768]", primals_107: "f32[768]", primals_108: "f32[768]", primals_109: "f32[768]", primals_110: "f32[32000, 768]", primals_111: "f32[32000]", primals_112: "i64[1, 512]", primals_113: "i64[1, 512]", primals_114: "i64[1, 512]", primals_115: "i64[1, 512]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:586, code: buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
    slice_1: "i64[1, 512]" = torch.ops.aten.slice.Tensor(primals_112, 0, 0, 9223372036854775807);  primals_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:587, code: buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
    expand: "i64[1, 512]" = torch.ops.aten.expand.default(slice_1, [1, 512]);  slice_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:134, code: position_ids = self.position_ids[:, :seq_length]
    slice_2: "i64[1, 512]" = torch.ops.aten.slice.Tensor(primals_113, 0, 0, 9223372036854775807);  primals_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:148, code: inputs_embeds = self.word_embeddings(input_ids)
    embedding: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(primals_1, primals_114, 3);  primals_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:149, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    embedding_1: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(primals_2, expand);  primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:151, code: embeddings = inputs_embeds + token_type_embeddings
    add: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:153, code: position_embeddings = self.position_embeddings(position_ids)
    embedding_2: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(primals_3, slice_2);  primals_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:154, code: embeddings += position_embeddings
    add_1: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:155, code: embeddings = self.LayerNorm(embeddings)
    var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 512, 1]" = var_mean[0]
    getitem_1: "f32[1, 512, 1]" = var_mean[1];  var_mean = None
    add_2: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
    rsqrt: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
    sub: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
    mul: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    mul_1: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul, primals_4)
    add_3: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_1, primals_5);  mul_1 = primals_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:156, code: embeddings = self.projection(embeddings)
    view: "f32[512, 768]" = torch.ops.aten.reshape.default(add_3, [512, 768]);  add_3 = None
    permute: "f32[768, 768]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
    addmm: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_7, view, permute);  primals_7 = None
    view_1: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm, [1, 512, 768]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:157, code: embeddings = self.dropout(embeddings)
    native_dropout = torch.ops.aten.native_dropout.default(view_1, 0.1, True);  view_1 = None
    getitem_2: "f32[1, 512, 768]" = native_dropout[0]
    getitem_3: "b8[1, 512, 768]" = native_dropout[1];  native_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    convert_element_type: "c64[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_2, torch.complex64)
    _fft_c2c: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(convert_element_type, [1, 2], 0, True);  convert_element_type = None
    view_as_real: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c);  _fft_c2c = None
    select: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real, 3, 0);  view_as_real = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    add_4: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_2, select);  getitem_2 = select = None
    var_mean_1 = torch.ops.aten.var_mean.correction(add_4, [2], correction = 0, keepdim = True)
    getitem_4: "f32[1, 512, 1]" = var_mean_1[0]
    getitem_5: "f32[1, 512, 1]" = var_mean_1[1];  var_mean_1 = None
    add_5: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
    rsqrt_1: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
    sub_1: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_4, getitem_5);  add_4 = getitem_5 = None
    mul_2: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    mul_3: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_2, primals_8)
    add_6: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_3, primals_9);  mul_3 = primals_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_2: "f32[512, 768]" = torch.ops.aten.reshape.default(add_6, [512, 768])
    permute_1: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
    addmm_1: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_11, view_2, permute_1);  primals_11 = None
    view_3: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_1, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_4: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_3, 0.5)
    pow_1: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_3, 3.0)
    mul_5: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
    add_7: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_3, mul_5);  view_3 = mul_5 = None
    mul_6: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_7, 0.7978845608028654);  add_7 = None
    tanh: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_6);  mul_6 = None
    add_8: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0)
    mul_7: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_4, add_8);  mul_4 = add_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_4: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_7, [512, 3072]);  mul_7 = None
    permute_2: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
    addmm_2: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_13, view_4, permute_2);  primals_13 = None
    view_5: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_2, [1, 512, 768]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    native_dropout_1 = torch.ops.aten.native_dropout.default(view_5, 0.1, True);  view_5 = None
    getitem_6: "f32[1, 512, 768]" = native_dropout_1[0]
    getitem_7: "b8[1, 512, 768]" = native_dropout_1[1];  native_dropout_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_9: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_6, add_6);  getitem_6 = add_6 = None
    var_mean_2 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
    getitem_8: "f32[1, 512, 1]" = var_mean_2[0]
    getitem_9: "f32[1, 512, 1]" = var_mean_2[1];  var_mean_2 = None
    add_10: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
    rsqrt_2: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
    sub_2: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_9, getitem_9);  add_9 = getitem_9 = None
    mul_8: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    mul_9: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_8, primals_14)
    add_11: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_9, primals_15);  mul_9 = primals_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    convert_element_type_1: "c64[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_11, torch.complex64)
    _fft_c2c_1: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(convert_element_type_1, [1, 2], 0, True);  convert_element_type_1 = None
    view_as_real_1: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_1);  _fft_c2c_1 = None
    select_1: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_1, 3, 0);  view_as_real_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    add_12: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_11, select_1);  add_11 = select_1 = None
    var_mean_3 = torch.ops.aten.var_mean.correction(add_12, [2], correction = 0, keepdim = True)
    getitem_10: "f32[1, 512, 1]" = var_mean_3[0]
    getitem_11: "f32[1, 512, 1]" = var_mean_3[1];  var_mean_3 = None
    add_13: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
    rsqrt_3: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
    sub_3: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_12, getitem_11);  add_12 = getitem_11 = None
    mul_10: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    mul_11: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_10, primals_16)
    add_14: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_11, primals_17);  mul_11 = primals_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_6: "f32[512, 768]" = torch.ops.aten.reshape.default(add_14, [512, 768])
    permute_3: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
    addmm_3: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_19, view_6, permute_3);  primals_19 = None
    view_7: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_3, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_12: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_7, 0.5)
    pow_2: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_7, 3.0)
    mul_13: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
    add_15: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_7, mul_13);  view_7 = mul_13 = None
    mul_14: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_15, 0.7978845608028654);  add_15 = None
    tanh_1: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_14);  mul_14 = None
    add_16: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_1, 1.0)
    mul_15: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_12, add_16);  mul_12 = add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_8: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_15, [512, 3072]);  mul_15 = None
    permute_4: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_20, [1, 0]);  primals_20 = None
    addmm_4: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_21, view_8, permute_4);  primals_21 = None
    view_9: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_4, [1, 512, 768]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    native_dropout_2 = torch.ops.aten.native_dropout.default(view_9, 0.1, True);  view_9 = None
    getitem_12: "f32[1, 512, 768]" = native_dropout_2[0]
    getitem_13: "b8[1, 512, 768]" = native_dropout_2[1];  native_dropout_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_17: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_12, add_14);  getitem_12 = add_14 = None
    var_mean_4 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
    getitem_14: "f32[1, 512, 1]" = var_mean_4[0]
    getitem_15: "f32[1, 512, 1]" = var_mean_4[1];  var_mean_4 = None
    add_18: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
    rsqrt_4: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    sub_4: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_17, getitem_15);  add_17 = getitem_15 = None
    mul_16: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    mul_17: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_16, primals_22)
    add_19: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_17, primals_23);  mul_17 = primals_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    convert_element_type_2: "c64[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_19, torch.complex64)
    _fft_c2c_2: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(convert_element_type_2, [1, 2], 0, True);  convert_element_type_2 = None
    view_as_real_2: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_2);  _fft_c2c_2 = None
    select_2: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_2, 3, 0);  view_as_real_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    add_20: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_19, select_2);  add_19 = select_2 = None
    var_mean_5 = torch.ops.aten.var_mean.correction(add_20, [2], correction = 0, keepdim = True)
    getitem_16: "f32[1, 512, 1]" = var_mean_5[0]
    getitem_17: "f32[1, 512, 1]" = var_mean_5[1];  var_mean_5 = None
    add_21: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
    rsqrt_5: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_5: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_20, getitem_17);  add_20 = getitem_17 = None
    mul_18: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    mul_19: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_18, primals_24)
    add_22: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_19, primals_25);  mul_19 = primals_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_10: "f32[512, 768]" = torch.ops.aten.reshape.default(add_22, [512, 768])
    permute_5: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_26, [1, 0]);  primals_26 = None
    addmm_5: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_27, view_10, permute_5);  primals_27 = None
    view_11: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_5, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_20: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_11, 0.5)
    pow_3: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_11, 3.0)
    mul_21: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
    add_23: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_11, mul_21);  view_11 = mul_21 = None
    mul_22: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_23, 0.7978845608028654);  add_23 = None
    tanh_2: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
    add_24: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_2, 1.0)
    mul_23: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_20, add_24);  mul_20 = add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_12: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_23, [512, 3072]);  mul_23 = None
    permute_6: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_28, [1, 0]);  primals_28 = None
    addmm_6: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_29, view_12, permute_6);  primals_29 = None
    view_13: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_6, [1, 512, 768]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    native_dropout_3 = torch.ops.aten.native_dropout.default(view_13, 0.1, True);  view_13 = None
    getitem_18: "f32[1, 512, 768]" = native_dropout_3[0]
    getitem_19: "b8[1, 512, 768]" = native_dropout_3[1];  native_dropout_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_25: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_18, add_22);  getitem_18 = add_22 = None
    var_mean_6 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
    getitem_20: "f32[1, 512, 1]" = var_mean_6[0]
    getitem_21: "f32[1, 512, 1]" = var_mean_6[1];  var_mean_6 = None
    add_26: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
    rsqrt_6: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
    sub_6: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_25, getitem_21);  add_25 = getitem_21 = None
    mul_24: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    mul_25: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_24, primals_30)
    add_27: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_25, primals_31);  mul_25 = primals_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    convert_element_type_3: "c64[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_27, torch.complex64)
    _fft_c2c_3: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(convert_element_type_3, [1, 2], 0, True);  convert_element_type_3 = None
    view_as_real_3: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_3);  _fft_c2c_3 = None
    select_3: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_3, 3, 0);  view_as_real_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    add_28: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_27, select_3);  add_27 = select_3 = None
    var_mean_7 = torch.ops.aten.var_mean.correction(add_28, [2], correction = 0, keepdim = True)
    getitem_22: "f32[1, 512, 1]" = var_mean_7[0]
    getitem_23: "f32[1, 512, 1]" = var_mean_7[1];  var_mean_7 = None
    add_29: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
    rsqrt_7: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
    sub_7: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_28, getitem_23);  add_28 = getitem_23 = None
    mul_26: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    mul_27: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_26, primals_32)
    add_30: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_27, primals_33);  mul_27 = primals_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_14: "f32[512, 768]" = torch.ops.aten.reshape.default(add_30, [512, 768])
    permute_7: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
    addmm_7: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_35, view_14, permute_7);  primals_35 = None
    view_15: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_7, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_28: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_15, 0.5)
    pow_4: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_15, 3.0)
    mul_29: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
    add_31: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_15, mul_29);  view_15 = mul_29 = None
    mul_30: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_31, 0.7978845608028654);  add_31 = None
    tanh_3: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_30);  mul_30 = None
    add_32: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_3, 1.0)
    mul_31: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_28, add_32);  mul_28 = add_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_16: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_31, [512, 3072]);  mul_31 = None
    permute_8: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_36, [1, 0]);  primals_36 = None
    addmm_8: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_37, view_16, permute_8);  primals_37 = None
    view_17: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_8, [1, 512, 768]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    native_dropout_4 = torch.ops.aten.native_dropout.default(view_17, 0.1, True);  view_17 = None
    getitem_24: "f32[1, 512, 768]" = native_dropout_4[0]
    getitem_25: "b8[1, 512, 768]" = native_dropout_4[1];  native_dropout_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_33: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_24, add_30);  getitem_24 = add_30 = None
    var_mean_8 = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
    getitem_26: "f32[1, 512, 1]" = var_mean_8[0]
    getitem_27: "f32[1, 512, 1]" = var_mean_8[1];  var_mean_8 = None
    add_34: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
    rsqrt_8: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
    sub_8: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_33, getitem_27);  add_33 = getitem_27 = None
    mul_32: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    mul_33: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_32, primals_38)
    add_35: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_33, primals_39);  mul_33 = primals_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    convert_element_type_4: "c64[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_35, torch.complex64)
    _fft_c2c_4: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(convert_element_type_4, [1, 2], 0, True);  convert_element_type_4 = None
    view_as_real_4: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_4);  _fft_c2c_4 = None
    select_4: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_4, 3, 0);  view_as_real_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    add_36: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_35, select_4);  add_35 = select_4 = None
    var_mean_9 = torch.ops.aten.var_mean.correction(add_36, [2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 512, 1]" = var_mean_9[0]
    getitem_29: "f32[1, 512, 1]" = var_mean_9[1];  var_mean_9 = None
    add_37: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
    rsqrt_9: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
    sub_9: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_36, getitem_29);  add_36 = getitem_29 = None
    mul_34: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    mul_35: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_34, primals_40)
    add_38: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_35, primals_41);  mul_35 = primals_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_18: "f32[512, 768]" = torch.ops.aten.reshape.default(add_38, [512, 768])
    permute_9: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
    addmm_9: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_43, view_18, permute_9);  primals_43 = None
    view_19: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_9, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_36: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.5)
    pow_5: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_19, 3.0)
    mul_37: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
    add_39: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_19, mul_37);  view_19 = mul_37 = None
    mul_38: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_39, 0.7978845608028654);  add_39 = None
    tanh_4: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
    add_40: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_4, 1.0)
    mul_39: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_36, add_40);  mul_36 = add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_20: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_39, [512, 3072]);  mul_39 = None
    permute_10: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_44, [1, 0]);  primals_44 = None
    addmm_10: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_45, view_20, permute_10);  primals_45 = None
    view_21: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_10, [1, 512, 768]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    native_dropout_5 = torch.ops.aten.native_dropout.default(view_21, 0.1, True);  view_21 = None
    getitem_30: "f32[1, 512, 768]" = native_dropout_5[0]
    getitem_31: "b8[1, 512, 768]" = native_dropout_5[1];  native_dropout_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_41: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_30, add_38);  getitem_30 = add_38 = None
    var_mean_10 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
    getitem_32: "f32[1, 512, 1]" = var_mean_10[0]
    getitem_33: "f32[1, 512, 1]" = var_mean_10[1];  var_mean_10 = None
    add_42: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
    rsqrt_10: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_10: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_41, getitem_33);  add_41 = getitem_33 = None
    mul_40: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    mul_41: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_40, primals_46)
    add_43: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_41, primals_47);  mul_41 = primals_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    convert_element_type_5: "c64[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_43, torch.complex64)
    _fft_c2c_5: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(convert_element_type_5, [1, 2], 0, True);  convert_element_type_5 = None
    view_as_real_5: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_5);  _fft_c2c_5 = None
    select_5: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_5, 3, 0);  view_as_real_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    add_44: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_43, select_5);  add_43 = select_5 = None
    var_mean_11 = torch.ops.aten.var_mean.correction(add_44, [2], correction = 0, keepdim = True)
    getitem_34: "f32[1, 512, 1]" = var_mean_11[0]
    getitem_35: "f32[1, 512, 1]" = var_mean_11[1];  var_mean_11 = None
    add_45: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
    rsqrt_11: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
    sub_11: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_44, getitem_35);  add_44 = getitem_35 = None
    mul_42: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    mul_43: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_42, primals_48)
    add_46: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_43, primals_49);  mul_43 = primals_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_22: "f32[512, 768]" = torch.ops.aten.reshape.default(add_46, [512, 768])
    permute_11: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_50, [1, 0]);  primals_50 = None
    addmm_11: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_51, view_22, permute_11);  primals_51 = None
    view_23: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_11, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_44: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_23, 0.5)
    pow_6: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_23, 3.0)
    mul_45: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
    add_47: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_23, mul_45);  view_23 = mul_45 = None
    mul_46: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_47, 0.7978845608028654);  add_47 = None
    tanh_5: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
    add_48: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_5, 1.0)
    mul_47: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_44, add_48);  mul_44 = add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_24: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_47, [512, 3072]);  mul_47 = None
    permute_12: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_52, [1, 0]);  primals_52 = None
    addmm_12: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_53, view_24, permute_12);  primals_53 = None
    view_25: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_12, [1, 512, 768]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    native_dropout_6 = torch.ops.aten.native_dropout.default(view_25, 0.1, True);  view_25 = None
    getitem_36: "f32[1, 512, 768]" = native_dropout_6[0]
    getitem_37: "b8[1, 512, 768]" = native_dropout_6[1];  native_dropout_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_49: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_36, add_46);  getitem_36 = add_46 = None
    var_mean_12 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
    getitem_38: "f32[1, 512, 1]" = var_mean_12[0]
    getitem_39: "f32[1, 512, 1]" = var_mean_12[1];  var_mean_12 = None
    add_50: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
    rsqrt_12: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
    sub_12: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_49, getitem_39);  add_49 = getitem_39 = None
    mul_48: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    mul_49: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_48, primals_54)
    add_51: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_49, primals_55);  mul_49 = primals_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    convert_element_type_6: "c64[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_51, torch.complex64)
    _fft_c2c_6: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(convert_element_type_6, [1, 2], 0, True);  convert_element_type_6 = None
    view_as_real_6: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_6);  _fft_c2c_6 = None
    select_6: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_6, 3, 0);  view_as_real_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    add_52: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_51, select_6);  add_51 = select_6 = None
    var_mean_13 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
    getitem_40: "f32[1, 512, 1]" = var_mean_13[0]
    getitem_41: "f32[1, 512, 1]" = var_mean_13[1];  var_mean_13 = None
    add_53: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
    rsqrt_13: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    sub_13: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_52, getitem_41);  add_52 = getitem_41 = None
    mul_50: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    mul_51: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_50, primals_56)
    add_54: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_51, primals_57);  mul_51 = primals_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_26: "f32[512, 768]" = torch.ops.aten.reshape.default(add_54, [512, 768])
    permute_13: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
    addmm_13: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_59, view_26, permute_13);  primals_59 = None
    view_27: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_13, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_52: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_27, 0.5)
    pow_7: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_27, 3.0)
    mul_53: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
    add_55: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_27, mul_53);  view_27 = mul_53 = None
    mul_54: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_55, 0.7978845608028654);  add_55 = None
    tanh_6: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_54);  mul_54 = None
    add_56: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_6, 1.0)
    mul_55: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_52, add_56);  mul_52 = add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_28: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_55, [512, 3072]);  mul_55 = None
    permute_14: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
    addmm_14: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_61, view_28, permute_14);  primals_61 = None
    view_29: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_14, [1, 512, 768]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    native_dropout_7 = torch.ops.aten.native_dropout.default(view_29, 0.1, True);  view_29 = None
    getitem_42: "f32[1, 512, 768]" = native_dropout_7[0]
    getitem_43: "b8[1, 512, 768]" = native_dropout_7[1];  native_dropout_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_57: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_42, add_54);  getitem_42 = add_54 = None
    var_mean_14 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
    getitem_44: "f32[1, 512, 1]" = var_mean_14[0]
    getitem_45: "f32[1, 512, 1]" = var_mean_14[1];  var_mean_14 = None
    add_58: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
    rsqrt_14: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
    sub_14: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_57, getitem_45);  add_57 = getitem_45 = None
    mul_56: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    mul_57: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_56, primals_62)
    add_59: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_57, primals_63);  mul_57 = primals_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    convert_element_type_7: "c64[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_59, torch.complex64)
    _fft_c2c_7: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(convert_element_type_7, [1, 2], 0, True);  convert_element_type_7 = None
    view_as_real_7: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_7);  _fft_c2c_7 = None
    select_7: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_7, 3, 0);  view_as_real_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    add_60: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_59, select_7);  add_59 = select_7 = None
    var_mean_15 = torch.ops.aten.var_mean.correction(add_60, [2], correction = 0, keepdim = True)
    getitem_46: "f32[1, 512, 1]" = var_mean_15[0]
    getitem_47: "f32[1, 512, 1]" = var_mean_15[1];  var_mean_15 = None
    add_61: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
    rsqrt_15: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
    sub_15: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_60, getitem_47);  add_60 = getitem_47 = None
    mul_58: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    mul_59: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_58, primals_64)
    add_62: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_59, primals_65);  mul_59 = primals_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_30: "f32[512, 768]" = torch.ops.aten.reshape.default(add_62, [512, 768])
    permute_15: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
    addmm_15: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_67, view_30, permute_15);  primals_67 = None
    view_31: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_15, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_60: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_31, 0.5)
    pow_8: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_31, 3.0)
    mul_61: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
    add_63: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_31, mul_61);  view_31 = mul_61 = None
    mul_62: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_63, 0.7978845608028654);  add_63 = None
    tanh_7: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_62);  mul_62 = None
    add_64: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_7, 1.0)
    mul_63: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_60, add_64);  mul_60 = add_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_32: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_63, [512, 3072]);  mul_63 = None
    permute_16: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_68, [1, 0]);  primals_68 = None
    addmm_16: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_69, view_32, permute_16);  primals_69 = None
    view_33: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_16, [1, 512, 768]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    native_dropout_8 = torch.ops.aten.native_dropout.default(view_33, 0.1, True);  view_33 = None
    getitem_48: "f32[1, 512, 768]" = native_dropout_8[0]
    getitem_49: "b8[1, 512, 768]" = native_dropout_8[1];  native_dropout_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_65: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_48, add_62);  getitem_48 = add_62 = None
    var_mean_16 = torch.ops.aten.var_mean.correction(add_65, [2], correction = 0, keepdim = True)
    getitem_50: "f32[1, 512, 1]" = var_mean_16[0]
    getitem_51: "f32[1, 512, 1]" = var_mean_16[1];  var_mean_16 = None
    add_66: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
    rsqrt_16: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    sub_16: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_65, getitem_51);  add_65 = getitem_51 = None
    mul_64: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    mul_65: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_64, primals_70)
    add_67: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_65, primals_71);  mul_65 = primals_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    convert_element_type_8: "c64[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_67, torch.complex64)
    _fft_c2c_8: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(convert_element_type_8, [1, 2], 0, True);  convert_element_type_8 = None
    view_as_real_8: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_8);  _fft_c2c_8 = None
    select_8: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_8, 3, 0);  view_as_real_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    add_68: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_67, select_8);  add_67 = select_8 = None
    var_mean_17 = torch.ops.aten.var_mean.correction(add_68, [2], correction = 0, keepdim = True)
    getitem_52: "f32[1, 512, 1]" = var_mean_17[0]
    getitem_53: "f32[1, 512, 1]" = var_mean_17[1];  var_mean_17 = None
    add_69: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-12);  getitem_52 = None
    rsqrt_17: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
    sub_17: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_68, getitem_53);  add_68 = getitem_53 = None
    mul_66: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    mul_67: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_66, primals_72)
    add_70: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_67, primals_73);  mul_67 = primals_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_34: "f32[512, 768]" = torch.ops.aten.reshape.default(add_70, [512, 768])
    permute_17: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_74, [1, 0]);  primals_74 = None
    addmm_17: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_75, view_34, permute_17);  primals_75 = None
    view_35: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_17, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_68: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_35, 0.5)
    pow_9: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_35, 3.0)
    mul_69: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
    add_71: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_35, mul_69);  view_35 = mul_69 = None
    mul_70: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_71, 0.7978845608028654);  add_71 = None
    tanh_8: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_70);  mul_70 = None
    add_72: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_8, 1.0)
    mul_71: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_68, add_72);  mul_68 = add_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_36: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_71, [512, 3072]);  mul_71 = None
    permute_18: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_76, [1, 0]);  primals_76 = None
    addmm_18: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_77, view_36, permute_18);  primals_77 = None
    view_37: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_18, [1, 512, 768]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    native_dropout_9 = torch.ops.aten.native_dropout.default(view_37, 0.1, True);  view_37 = None
    getitem_54: "f32[1, 512, 768]" = native_dropout_9[0]
    getitem_55: "b8[1, 512, 768]" = native_dropout_9[1];  native_dropout_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_73: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_54, add_70);  getitem_54 = add_70 = None
    var_mean_18 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
    getitem_56: "f32[1, 512, 1]" = var_mean_18[0]
    getitem_57: "f32[1, 512, 1]" = var_mean_18[1];  var_mean_18 = None
    add_74: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-12);  getitem_56 = None
    rsqrt_18: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    sub_18: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_73, getitem_57);  add_73 = getitem_57 = None
    mul_72: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    mul_73: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_72, primals_78)
    add_75: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_73, primals_79);  mul_73 = primals_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    convert_element_type_9: "c64[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_75, torch.complex64)
    _fft_c2c_9: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(convert_element_type_9, [1, 2], 0, True);  convert_element_type_9 = None
    view_as_real_9: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_9);  _fft_c2c_9 = None
    select_9: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_9, 3, 0);  view_as_real_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    add_76: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_75, select_9);  add_75 = select_9 = None
    var_mean_19 = torch.ops.aten.var_mean.correction(add_76, [2], correction = 0, keepdim = True)
    getitem_58: "f32[1, 512, 1]" = var_mean_19[0]
    getitem_59: "f32[1, 512, 1]" = var_mean_19[1];  var_mean_19 = None
    add_77: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-12);  getitem_58 = None
    rsqrt_19: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
    sub_19: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_76, getitem_59);  add_76 = getitem_59 = None
    mul_74: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    mul_75: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_74, primals_80)
    add_78: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_75, primals_81);  mul_75 = primals_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_38: "f32[512, 768]" = torch.ops.aten.reshape.default(add_78, [512, 768])
    permute_19: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_82, [1, 0]);  primals_82 = None
    addmm_19: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_83, view_38, permute_19);  primals_83 = None
    view_39: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_19, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_76: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_39, 0.5)
    pow_10: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_39, 3.0)
    mul_77: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
    add_79: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_39, mul_77);  view_39 = mul_77 = None
    mul_78: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_79, 0.7978845608028654);  add_79 = None
    tanh_9: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
    add_80: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_9, 1.0)
    mul_79: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_76, add_80);  mul_76 = add_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_40: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_79, [512, 3072]);  mul_79 = None
    permute_20: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_84, [1, 0]);  primals_84 = None
    addmm_20: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_85, view_40, permute_20);  primals_85 = None
    view_41: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_20, [1, 512, 768]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    native_dropout_10 = torch.ops.aten.native_dropout.default(view_41, 0.1, True);  view_41 = None
    getitem_60: "f32[1, 512, 768]" = native_dropout_10[0]
    getitem_61: "b8[1, 512, 768]" = native_dropout_10[1];  native_dropout_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_81: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_60, add_78);  getitem_60 = add_78 = None
    var_mean_20 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
    getitem_62: "f32[1, 512, 1]" = var_mean_20[0]
    getitem_63: "f32[1, 512, 1]" = var_mean_20[1];  var_mean_20 = None
    add_82: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-12);  getitem_62 = None
    rsqrt_20: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
    sub_20: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_81, getitem_63);  add_81 = getitem_63 = None
    mul_80: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    mul_81: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_80, primals_86)
    add_83: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_81, primals_87);  mul_81 = primals_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    convert_element_type_10: "c64[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_83, torch.complex64)
    _fft_c2c_10: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(convert_element_type_10, [1, 2], 0, True);  convert_element_type_10 = None
    view_as_real_10: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_10);  _fft_c2c_10 = None
    select_10: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_10, 3, 0);  view_as_real_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    add_84: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_83, select_10);  add_83 = select_10 = None
    var_mean_21 = torch.ops.aten.var_mean.correction(add_84, [2], correction = 0, keepdim = True)
    getitem_64: "f32[1, 512, 1]" = var_mean_21[0]
    getitem_65: "f32[1, 512, 1]" = var_mean_21[1];  var_mean_21 = None
    add_85: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-12);  getitem_64 = None
    rsqrt_21: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
    sub_21: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_84, getitem_65);  add_84 = getitem_65 = None
    mul_82: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    mul_83: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_82, primals_88)
    add_86: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_83, primals_89);  mul_83 = primals_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_42: "f32[512, 768]" = torch.ops.aten.reshape.default(add_86, [512, 768])
    permute_21: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_90, [1, 0]);  primals_90 = None
    addmm_21: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_91, view_42, permute_21);  primals_91 = None
    view_43: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_21, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_84: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_43, 0.5)
    pow_11: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_43, 3.0)
    mul_85: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
    add_87: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_43, mul_85);  view_43 = mul_85 = None
    mul_86: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_87, 0.7978845608028654);  add_87 = None
    tanh_10: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_86);  mul_86 = None
    add_88: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_10, 1.0)
    mul_87: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_84, add_88);  mul_84 = add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_44: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_87, [512, 3072]);  mul_87 = None
    permute_22: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
    addmm_22: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_93, view_44, permute_22);  primals_93 = None
    view_45: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_22, [1, 512, 768]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    native_dropout_11 = torch.ops.aten.native_dropout.default(view_45, 0.1, True);  view_45 = None
    getitem_66: "f32[1, 512, 768]" = native_dropout_11[0]
    getitem_67: "b8[1, 512, 768]" = native_dropout_11[1];  native_dropout_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_89: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_66, add_86);  getitem_66 = add_86 = None
    var_mean_22 = torch.ops.aten.var_mean.correction(add_89, [2], correction = 0, keepdim = True)
    getitem_68: "f32[1, 512, 1]" = var_mean_22[0]
    getitem_69: "f32[1, 512, 1]" = var_mean_22[1];  var_mean_22 = None
    add_90: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-12);  getitem_68 = None
    rsqrt_22: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
    sub_22: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_89, getitem_69);  add_89 = getitem_69 = None
    mul_88: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    mul_89: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_88, primals_94)
    add_91: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_89, primals_95);  mul_89 = primals_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    convert_element_type_11: "c64[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_91, torch.complex64)
    _fft_c2c_11: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(convert_element_type_11, [1, 2], 0, True);  convert_element_type_11 = None
    view_as_real_11: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_11);  _fft_c2c_11 = None
    select_11: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_11, 3, 0);  view_as_real_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    add_92: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_91, select_11);  add_91 = select_11 = None
    var_mean_23 = torch.ops.aten.var_mean.correction(add_92, [2], correction = 0, keepdim = True)
    getitem_70: "f32[1, 512, 1]" = var_mean_23[0]
    getitem_71: "f32[1, 512, 1]" = var_mean_23[1];  var_mean_23 = None
    add_93: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-12);  getitem_70 = None
    rsqrt_23: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
    sub_23: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_92, getitem_71);  add_92 = getitem_71 = None
    mul_90: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
    mul_91: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_90, primals_96)
    add_94: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_91, primals_97);  mul_91 = primals_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_46: "f32[512, 768]" = torch.ops.aten.reshape.default(add_94, [512, 768])
    permute_23: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_98, [1, 0]);  primals_98 = None
    addmm_23: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_99, view_46, permute_23);  primals_99 = None
    view_47: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_23, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_92: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_47, 0.5)
    pow_12: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_47, 3.0)
    mul_93: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
    add_95: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_47, mul_93);  view_47 = mul_93 = None
    mul_94: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_95, 0.7978845608028654);  add_95 = None
    tanh_11: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_94);  mul_94 = None
    add_96: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_11, 1.0)
    mul_95: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_92, add_96);  mul_92 = add_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_48: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_95, [512, 3072]);  mul_95 = None
    permute_24: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_100, [1, 0]);  primals_100 = None
    addmm_24: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_101, view_48, permute_24);  primals_101 = None
    view_49: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_24, [1, 512, 768]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    native_dropout_12 = torch.ops.aten.native_dropout.default(view_49, 0.1, True);  view_49 = None
    getitem_72: "f32[1, 512, 768]" = native_dropout_12[0]
    getitem_73: "b8[1, 512, 768]" = native_dropout_12[1];  native_dropout_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_97: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_72, add_94);  getitem_72 = add_94 = None
    var_mean_24 = torch.ops.aten.var_mean.correction(add_97, [2], correction = 0, keepdim = True)
    getitem_74: "f32[1, 512, 1]" = var_mean_24[0]
    getitem_75: "f32[1, 512, 1]" = var_mean_24[1];  var_mean_24 = None
    add_98: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-12);  getitem_74 = None
    rsqrt_24: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
    sub_24: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_97, getitem_75);  add_97 = getitem_75 = None
    mul_96: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
    mul_97: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_96, primals_102)
    add_99: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_97, primals_103);  mul_97 = primals_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:345, code: hidden_states = self.dense(hidden_states)
    view_50: "f32[512, 768]" = torch.ops.aten.reshape.default(add_99, [512, 768]);  add_99 = None
    permute_26: "f32[768, 768]" = torch.ops.aten.permute.default(primals_106, [1, 0]);  primals_106 = None
    addmm_26: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_107, view_50, permute_26);  primals_107 = None
    view_51: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_26, [1, 512, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_98: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_51, 0.5)
    pow_13: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(view_51, 3.0)
    mul_99: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(pow_13, 0.044715);  pow_13 = None
    add_100: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_51, mul_99);  view_51 = mul_99 = None
    mul_100: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_100, 0.7978845608028654);  add_100 = None
    tanh_13: "f32[1, 512, 768]" = torch.ops.aten.tanh.default(mul_100);  mul_100 = None
    add_101: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(tanh_13, 1.0)
    mul_101: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_98, add_101);  mul_98 = add_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:347, code: hidden_states = self.LayerNorm(hidden_states)
    var_mean_25 = torch.ops.aten.var_mean.correction(mul_101, [2], correction = 0, keepdim = True)
    getitem_76: "f32[1, 512, 1]" = var_mean_25[0]
    getitem_77: "f32[1, 512, 1]" = var_mean_25[1];  var_mean_25 = None
    add_102: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-12);  getitem_76 = None
    rsqrt_25: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
    sub_25: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_101, getitem_77);  mul_101 = None
    mul_102: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
    mul_103: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_102, primals_108);  mul_102 = None
    add_103: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_103, primals_109);  mul_103 = primals_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:365, code: hidden_states = self.decoder(hidden_states)
    view_52: "f32[512, 768]" = torch.ops.aten.reshape.default(add_103, [512, 768]);  add_103 = None
    permute_27: "f32[768, 32000]" = torch.ops.aten.permute.default(primals_110, [1, 0]);  primals_110 = None
    addmm_27: "f32[512, 32000]" = torch.ops.aten.addmm.default(primals_111, view_52, permute_27);  primals_111 = None
    view_53: "f32[1, 512, 32000]" = torch.ops.aten.reshape.default(addmm_27, [1, 512, 32000]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:775, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    view_54: "f32[512, 32000]" = torch.ops.aten.reshape.default(view_53, [-1, 32000])
    view_55: "i64[512]" = torch.ops.aten.reshape.default(primals_115, [-1])
    amax: "f32[512, 1]" = torch.ops.aten.amax.default(view_54, [1], True)
    sub_26: "f32[512, 32000]" = torch.ops.aten.sub.Tensor(view_54, amax);  view_54 = amax = None
    exp: "f32[512, 32000]" = torch.ops.aten.exp.default(sub_26)
    sum_1: "f32[512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
    log: "f32[512, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
    sub_27: "f32[512, 32000]" = torch.ops.aten.sub.Tensor(sub_26, log);  sub_26 = log = None
    ne: "b8[512]" = torch.ops.aten.ne.Scalar(view_55, -100)
    full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where: "i64[512]" = torch.ops.aten.where.self(ne, view_55, full_default);  view_55 = full_default = None
    unsqueeze: "i64[512, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
    gather: "f32[512, 1]" = torch.ops.aten.gather.default(sub_27, 1, unsqueeze);  unsqueeze = None
    squeeze: "f32[512]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[512]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
    full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where_1: "f32[512]" = torch.ops.aten.where.self(ne, neg, full_default_1);  neg = full_default_1 = None
    sum_2: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
    convert_element_type_12: "f32[]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
    sum_3: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
    div: "f32[]" = torch.ops.aten.div.Tensor(sum_3, convert_element_type_12);  sum_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:365, code: hidden_states = self.decoder(hidden_states)
    permute_28: "f32[32000, 768]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:345, code: hidden_states = self.dense(hidden_states)
    permute_32: "f32[768, 768]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_3: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    permute_36: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    permute_40: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    div_4: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_5: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    permute_44: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    permute_48: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    div_6: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_7: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    permute_52: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    permute_56: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    div_8: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_9: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    permute_60: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    permute_64: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    div_10: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_11: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    permute_68: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    permute_72: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    div_12: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_13: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    permute_76: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    permute_80: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    div_14: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_15: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    permute_84: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    permute_88: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    div_16: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_17: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    permute_92: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    permute_96: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    div_18: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_19: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    permute_100: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    permute_104: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    div_20: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_21: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    permute_108: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    permute_112: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    div_22: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_23: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    permute_116: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    permute_120: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    div_24: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_25: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    permute_124: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    permute_128: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    div_26: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:156, code: embeddings = self.projection(embeddings)
    permute_132: "f32[768, 768]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:155, code: embeddings = self.LayerNorm(embeddings)
    div_27: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
    return [div, view_53, primals_4, primals_8, primals_14, primals_16, primals_22, primals_24, primals_30, primals_32, primals_38, primals_40, primals_46, primals_48, primals_54, primals_56, primals_62, primals_64, primals_70, primals_72, primals_78, primals_80, primals_86, primals_88, primals_94, primals_96, primals_102, primals_108, primals_114, primals_115, expand, slice_2, mul, view, getitem_3, mul_2, view_2, addmm_1, tanh, view_4, getitem_7, mul_8, mul_10, view_6, addmm_3, tanh_1, view_8, getitem_13, mul_16, mul_18, view_10, addmm_5, tanh_2, view_12, getitem_19, mul_24, mul_26, view_14, addmm_7, tanh_3, view_16, getitem_25, mul_32, mul_34, view_18, addmm_9, tanh_4, view_20, getitem_31, mul_40, mul_42, view_22, addmm_11, tanh_5, view_24, getitem_37, mul_48, mul_50, view_26, addmm_13, tanh_6, view_28, getitem_43, mul_56, mul_58, view_30, addmm_15, tanh_7, view_32, getitem_49, mul_64, mul_66, view_34, addmm_17, tanh_8, view_36, getitem_55, mul_72, mul_74, view_38, addmm_19, tanh_9, view_40, getitem_61, mul_80, mul_82, view_42, addmm_21, tanh_10, view_44, getitem_67, mul_88, mul_90, view_46, addmm_23, tanh_11, view_48, getitem_73, mul_96, view_50, addmm_26, tanh_13, getitem_77, rsqrt_25, view_52, sub_27, convert_element_type_12, permute_28, permute_32, div_3, permute_36, permute_40, div_4, div_5, permute_44, permute_48, div_6, div_7, permute_52, permute_56, div_8, div_9, permute_60, permute_64, div_10, div_11, permute_68, permute_72, div_12, div_13, permute_76, permute_80, div_14, div_15, permute_84, permute_88, div_16, div_17, permute_92, permute_96, div_18, div_19, permute_100, permute_104, div_20, div_21, permute_108, permute_112, div_22, div_23, permute_116, permute_120, div_24, div_25, permute_124, permute_128, div_26, permute_132, div_27]
    