from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[32000, 768]"; primals_2: "f32[4, 768]"; primals_3: "f32[512, 768]"; primals_4: "f32[768]"; primals_5: "f32[768]"; primals_6: "f32[768, 768]"; primals_7: "f32[768]"; primals_8: "f32[768]"; primals_9: "f32[768]"; primals_10: "f32[3072, 768]"; primals_11: "f32[3072]"; primals_12: "f32[768, 3072]"; primals_13: "f32[768]"; primals_14: "f32[768]"; primals_15: "f32[768]"; primals_16: "f32[768]"; primals_17: "f32[768]"; primals_18: "f32[3072, 768]"; primals_19: "f32[3072]"; primals_20: "f32[768, 3072]"; primals_21: "f32[768]"; primals_22: "f32[768]"; primals_23: "f32[768]"; primals_24: "f32[768]"; primals_25: "f32[768]"; primals_26: "f32[3072, 768]"; primals_27: "f32[3072]"; primals_28: "f32[768, 3072]"; primals_29: "f32[768]"; primals_30: "f32[768]"; primals_31: "f32[768]"; primals_32: "f32[768]"; primals_33: "f32[768]"; primals_34: "f32[3072, 768]"; primals_35: "f32[3072]"; primals_36: "f32[768, 3072]"; primals_37: "f32[768]"; primals_38: "f32[768]"; primals_39: "f32[768]"; primals_40: "f32[768]"; primals_41: "f32[768]"; primals_42: "f32[3072, 768]"; primals_43: "f32[3072]"; primals_44: "f32[768, 3072]"; primals_45: "f32[768]"; primals_46: "f32[768]"; primals_47: "f32[768]"; primals_48: "f32[768]"; primals_49: "f32[768]"; primals_50: "f32[3072, 768]"; primals_51: "f32[3072]"; primals_52: "f32[768, 3072]"; primals_53: "f32[768]"; primals_54: "f32[768]"; primals_55: "f32[768]"; primals_56: "f32[768]"; primals_57: "f32[768]"; primals_58: "f32[3072, 768]"; primals_59: "f32[3072]"; primals_60: "f32[768, 3072]"; primals_61: "f32[768]"; primals_62: "f32[768]"; primals_63: "f32[768]"; primals_64: "f32[768]"; primals_65: "f32[768]"; primals_66: "f32[3072, 768]"; primals_67: "f32[3072]"; primals_68: "f32[768, 3072]"; primals_69: "f32[768]"; primals_70: "f32[768]"; primals_71: "f32[768]"; primals_72: "f32[768]"; primals_73: "f32[768]"; primals_74: "f32[3072, 768]"; primals_75: "f32[3072]"; primals_76: "f32[768, 3072]"; primals_77: "f32[768]"; primals_78: "f32[768]"; primals_79: "f32[768]"; primals_80: "f32[768]"; primals_81: "f32[768]"; primals_82: "f32[3072, 768]"; primals_83: "f32[3072]"; primals_84: "f32[768, 3072]"; primals_85: "f32[768]"; primals_86: "f32[768]"; primals_87: "f32[768]"; primals_88: "f32[768]"; primals_89: "f32[768]"; primals_90: "f32[3072, 768]"; primals_91: "f32[3072]"; primals_92: "f32[768, 3072]"; primals_93: "f32[768]"; primals_94: "f32[768]"; primals_95: "f32[768]"; primals_96: "f32[768]"; primals_97: "f32[768]"; primals_98: "f32[3072, 768]"; primals_99: "f32[3072]"; primals_100: "f32[768, 3072]"; primals_101: "f32[768]"; primals_102: "f32[768]"; primals_103: "f32[768]"; primals_104: "f32[768, 768]"; primals_105: "f32[768]"; primals_106: "f32[768, 768]"; primals_107: "f32[768]"; primals_108: "f32[768]"; primals_109: "f32[768]"; primals_110: "f32[32000, 768]"; primals_111: "f32[32000]"; primals_112: "i64[1, 512]"; primals_113: "i64[1, 512]"; primals_114: "i64[1, 512]"; primals_115: "i64[1, 512]"; tangents_1: "f32[]"; tangents_2: "f32[1, 512, 32000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, tangents_1, tangents_2, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
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
    sub: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_1, getitem_1)
    mul: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    mul_1: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul, primals_4);  mul = None
    add_3: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_1, primals_5);  mul_1 = primals_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:156, code: embeddings = self.projection(embeddings)
    view: "f32[512, 768]" = torch.ops.aten.view.default(add_3, [512, 768]);  add_3 = None
    permute: "f32[768, 768]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
    addmm: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_7, view, permute);  primals_7 = None
    view_1: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm, [1, 512, 768]);  addmm = None
    
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
    sub_1: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_4, getitem_5)
    mul_2: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    mul_3: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_2, primals_8);  mul_2 = None
    add_6: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_3, primals_9);  mul_3 = primals_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_2: "f32[512, 768]" = torch.ops.aten.view.default(add_6, [512, 768])
    permute_1: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
    addmm_1: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_11, view_2, permute_1);  primals_11 = None
    view_3: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_1, [1, 512, 3072]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_4: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_3, 0.5)
    pow_1: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_3, 3.0)
    mul_5: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
    add_7: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_3, mul_5);  mul_5 = None
    mul_6: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_7, 0.7978845608028654);  add_7 = None
    tanh: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_6);  mul_6 = None
    alias: "f32[1, 512, 3072]" = torch.ops.aten.alias.default(tanh)
    add_8: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
    mul_7: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_4, add_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_4: "f32[512, 3072]" = torch.ops.aten.view.default(mul_7, [512, 3072]);  mul_7 = None
    permute_2: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
    addmm_2: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_13, view_4, permute_2);  primals_13 = None
    view_5: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_2, [1, 512, 768]);  addmm_2 = None
    
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
    sub_2: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_9, getitem_9)
    mul_8: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    mul_9: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_8, primals_14);  mul_8 = None
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
    sub_3: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_12, getitem_11)
    mul_10: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    mul_11: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_10, primals_16);  mul_10 = None
    add_14: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_11, primals_17);  mul_11 = primals_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_6: "f32[512, 768]" = torch.ops.aten.view.default(add_14, [512, 768])
    permute_3: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
    addmm_3: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_19, view_6, permute_3);  primals_19 = None
    view_7: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_3, [1, 512, 3072]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_12: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_7, 0.5)
    pow_2: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_7, 3.0)
    mul_13: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
    add_15: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_7, mul_13);  mul_13 = None
    mul_14: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_15, 0.7978845608028654);  add_15 = None
    tanh_1: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_14);  mul_14 = None
    alias_1: "f32[1, 512, 3072]" = torch.ops.aten.alias.default(tanh_1)
    add_16: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
    mul_15: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_12, add_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_8: "f32[512, 3072]" = torch.ops.aten.view.default(mul_15, [512, 3072]);  mul_15 = None
    permute_4: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_20, [1, 0]);  primals_20 = None
    addmm_4: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_21, view_8, permute_4);  primals_21 = None
    view_9: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_4, [1, 512, 768]);  addmm_4 = None
    
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
    sub_4: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_17, getitem_15)
    mul_16: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    mul_17: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_16, primals_22);  mul_16 = None
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
    sub_5: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_20, getitem_17)
    mul_18: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    mul_19: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_18, primals_24);  mul_18 = None
    add_22: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_19, primals_25);  mul_19 = primals_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_10: "f32[512, 768]" = torch.ops.aten.view.default(add_22, [512, 768])
    permute_5: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_26, [1, 0]);  primals_26 = None
    addmm_5: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_27, view_10, permute_5);  primals_27 = None
    view_11: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_5, [1, 512, 3072]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_20: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_11, 0.5)
    pow_3: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_11, 3.0)
    mul_21: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
    add_23: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_11, mul_21);  mul_21 = None
    mul_22: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_23, 0.7978845608028654);  add_23 = None
    tanh_2: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
    alias_2: "f32[1, 512, 3072]" = torch.ops.aten.alias.default(tanh_2)
    add_24: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
    mul_23: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_20, add_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_12: "f32[512, 3072]" = torch.ops.aten.view.default(mul_23, [512, 3072]);  mul_23 = None
    permute_6: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_28, [1, 0]);  primals_28 = None
    addmm_6: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_29, view_12, permute_6);  primals_29 = None
    view_13: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_6, [1, 512, 768]);  addmm_6 = None
    
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
    sub_6: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_25, getitem_21)
    mul_24: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    mul_25: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_24, primals_30);  mul_24 = None
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
    sub_7: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_28, getitem_23)
    mul_26: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    mul_27: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_26, primals_32);  mul_26 = None
    add_30: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_27, primals_33);  mul_27 = primals_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_14: "f32[512, 768]" = torch.ops.aten.view.default(add_30, [512, 768])
    permute_7: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
    addmm_7: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_35, view_14, permute_7);  primals_35 = None
    view_15: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_7, [1, 512, 3072]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_28: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_15, 0.5)
    pow_4: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_15, 3.0)
    mul_29: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
    add_31: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_15, mul_29);  mul_29 = None
    mul_30: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_31, 0.7978845608028654);  add_31 = None
    tanh_3: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_30);  mul_30 = None
    alias_3: "f32[1, 512, 3072]" = torch.ops.aten.alias.default(tanh_3)
    add_32: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
    mul_31: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_28, add_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_16: "f32[512, 3072]" = torch.ops.aten.view.default(mul_31, [512, 3072]);  mul_31 = None
    permute_8: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_36, [1, 0]);  primals_36 = None
    addmm_8: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_37, view_16, permute_8);  primals_37 = None
    view_17: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_8, [1, 512, 768]);  addmm_8 = None
    
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
    sub_8: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_33, getitem_27)
    mul_32: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    mul_33: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_32, primals_38);  mul_32 = None
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
    sub_9: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_36, getitem_29)
    mul_34: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    mul_35: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_34, primals_40);  mul_34 = None
    add_38: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_35, primals_41);  mul_35 = primals_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_18: "f32[512, 768]" = torch.ops.aten.view.default(add_38, [512, 768])
    permute_9: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
    addmm_9: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_43, view_18, permute_9);  primals_43 = None
    view_19: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_9, [1, 512, 3072]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_36: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.5)
    pow_5: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_19, 3.0)
    mul_37: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
    add_39: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_19, mul_37);  mul_37 = None
    mul_38: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_39, 0.7978845608028654);  add_39 = None
    tanh_4: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
    alias_4: "f32[1, 512, 3072]" = torch.ops.aten.alias.default(tanh_4)
    add_40: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
    mul_39: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_36, add_40)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_20: "f32[512, 3072]" = torch.ops.aten.view.default(mul_39, [512, 3072]);  mul_39 = None
    permute_10: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_44, [1, 0]);  primals_44 = None
    addmm_10: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_45, view_20, permute_10);  primals_45 = None
    view_21: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_10, [1, 512, 768]);  addmm_10 = None
    
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
    sub_10: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_41, getitem_33)
    mul_40: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    mul_41: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_40, primals_46);  mul_40 = None
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
    sub_11: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_44, getitem_35)
    mul_42: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    mul_43: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_42, primals_48);  mul_42 = None
    add_46: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_43, primals_49);  mul_43 = primals_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_22: "f32[512, 768]" = torch.ops.aten.view.default(add_46, [512, 768])
    permute_11: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_50, [1, 0]);  primals_50 = None
    addmm_11: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_51, view_22, permute_11);  primals_51 = None
    view_23: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_11, [1, 512, 3072]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_44: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_23, 0.5)
    pow_6: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_23, 3.0)
    mul_45: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
    add_47: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_23, mul_45);  mul_45 = None
    mul_46: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_47, 0.7978845608028654);  add_47 = None
    tanh_5: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
    alias_5: "f32[1, 512, 3072]" = torch.ops.aten.alias.default(tanh_5)
    add_48: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
    mul_47: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_44, add_48)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_24: "f32[512, 3072]" = torch.ops.aten.view.default(mul_47, [512, 3072]);  mul_47 = None
    permute_12: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_52, [1, 0]);  primals_52 = None
    addmm_12: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_53, view_24, permute_12);  primals_53 = None
    view_25: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_12, [1, 512, 768]);  addmm_12 = None
    
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
    sub_12: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_49, getitem_39)
    mul_48: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    mul_49: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_48, primals_54);  mul_48 = None
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
    sub_13: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_52, getitem_41)
    mul_50: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    mul_51: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_50, primals_56);  mul_50 = None
    add_54: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_51, primals_57);  mul_51 = primals_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_26: "f32[512, 768]" = torch.ops.aten.view.default(add_54, [512, 768])
    permute_13: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
    addmm_13: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_59, view_26, permute_13);  primals_59 = None
    view_27: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_13, [1, 512, 3072]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_52: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_27, 0.5)
    pow_7: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_27, 3.0)
    mul_53: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
    add_55: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_27, mul_53);  mul_53 = None
    mul_54: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_55, 0.7978845608028654);  add_55 = None
    tanh_6: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_54);  mul_54 = None
    alias_6: "f32[1, 512, 3072]" = torch.ops.aten.alias.default(tanh_6)
    add_56: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
    mul_55: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_52, add_56)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_28: "f32[512, 3072]" = torch.ops.aten.view.default(mul_55, [512, 3072]);  mul_55 = None
    permute_14: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
    addmm_14: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_61, view_28, permute_14);  primals_61 = None
    view_29: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_14, [1, 512, 768]);  addmm_14 = None
    
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
    sub_14: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_57, getitem_45)
    mul_56: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    mul_57: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_56, primals_62);  mul_56 = None
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
    sub_15: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_60, getitem_47)
    mul_58: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    mul_59: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_58, primals_64);  mul_58 = None
    add_62: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_59, primals_65);  mul_59 = primals_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_30: "f32[512, 768]" = torch.ops.aten.view.default(add_62, [512, 768])
    permute_15: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
    addmm_15: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_67, view_30, permute_15);  primals_67 = None
    view_31: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_15, [1, 512, 3072]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_60: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_31, 0.5)
    pow_8: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_31, 3.0)
    mul_61: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
    add_63: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_31, mul_61);  mul_61 = None
    mul_62: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_63, 0.7978845608028654);  add_63 = None
    tanh_7: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_62);  mul_62 = None
    alias_7: "f32[1, 512, 3072]" = torch.ops.aten.alias.default(tanh_7)
    add_64: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
    mul_63: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_60, add_64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_32: "f32[512, 3072]" = torch.ops.aten.view.default(mul_63, [512, 3072]);  mul_63 = None
    permute_16: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_68, [1, 0]);  primals_68 = None
    addmm_16: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_69, view_32, permute_16);  primals_69 = None
    view_33: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_16, [1, 512, 768]);  addmm_16 = None
    
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
    sub_16: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_65, getitem_51)
    mul_64: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    mul_65: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_64, primals_70);  mul_64 = None
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
    sub_17: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_68, getitem_53)
    mul_66: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    mul_67: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_66, primals_72);  mul_66 = None
    add_70: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_67, primals_73);  mul_67 = primals_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_34: "f32[512, 768]" = torch.ops.aten.view.default(add_70, [512, 768])
    permute_17: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_74, [1, 0]);  primals_74 = None
    addmm_17: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_75, view_34, permute_17);  primals_75 = None
    view_35: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_17, [1, 512, 3072]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_68: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_35, 0.5)
    pow_9: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_35, 3.0)
    mul_69: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
    add_71: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_35, mul_69);  mul_69 = None
    mul_70: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_71, 0.7978845608028654);  add_71 = None
    tanh_8: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_70);  mul_70 = None
    alias_8: "f32[1, 512, 3072]" = torch.ops.aten.alias.default(tanh_8)
    add_72: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
    mul_71: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_68, add_72)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_36: "f32[512, 3072]" = torch.ops.aten.view.default(mul_71, [512, 3072]);  mul_71 = None
    permute_18: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_76, [1, 0]);  primals_76 = None
    addmm_18: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_77, view_36, permute_18);  primals_77 = None
    view_37: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_18, [1, 512, 768]);  addmm_18 = None
    
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
    sub_18: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_73, getitem_57)
    mul_72: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    mul_73: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_72, primals_78);  mul_72 = None
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
    sub_19: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_76, getitem_59)
    mul_74: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    mul_75: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_74, primals_80);  mul_74 = None
    add_78: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_75, primals_81);  mul_75 = primals_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_38: "f32[512, 768]" = torch.ops.aten.view.default(add_78, [512, 768])
    permute_19: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_82, [1, 0]);  primals_82 = None
    addmm_19: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_83, view_38, permute_19);  primals_83 = None
    view_39: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_19, [1, 512, 3072]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_76: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_39, 0.5)
    pow_10: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_39, 3.0)
    mul_77: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
    add_79: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_39, mul_77);  mul_77 = None
    mul_78: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_79, 0.7978845608028654);  add_79 = None
    tanh_9: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
    alias_9: "f32[1, 512, 3072]" = torch.ops.aten.alias.default(tanh_9)
    add_80: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
    mul_79: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_76, add_80)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_40: "f32[512, 3072]" = torch.ops.aten.view.default(mul_79, [512, 3072]);  mul_79 = None
    permute_20: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_84, [1, 0]);  primals_84 = None
    addmm_20: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_85, view_40, permute_20);  primals_85 = None
    view_41: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_20, [1, 512, 768]);  addmm_20 = None
    
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
    sub_20: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_81, getitem_63)
    mul_80: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    mul_81: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_80, primals_86);  mul_80 = None
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
    sub_21: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_84, getitem_65)
    mul_82: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    mul_83: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_82, primals_88);  mul_82 = None
    add_86: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_83, primals_89);  mul_83 = primals_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_42: "f32[512, 768]" = torch.ops.aten.view.default(add_86, [512, 768])
    permute_21: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_90, [1, 0]);  primals_90 = None
    addmm_21: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_91, view_42, permute_21);  primals_91 = None
    view_43: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_21, [1, 512, 3072]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_84: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_43, 0.5)
    pow_11: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_43, 3.0)
    mul_85: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
    add_87: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_43, mul_85);  mul_85 = None
    mul_86: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_87, 0.7978845608028654);  add_87 = None
    tanh_10: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_86);  mul_86 = None
    alias_10: "f32[1, 512, 3072]" = torch.ops.aten.alias.default(tanh_10)
    add_88: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
    mul_87: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_84, add_88)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_44: "f32[512, 3072]" = torch.ops.aten.view.default(mul_87, [512, 3072]);  mul_87 = None
    permute_22: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
    addmm_22: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_93, view_44, permute_22);  primals_93 = None
    view_45: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_22, [1, 512, 768]);  addmm_22 = None
    
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
    sub_22: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_89, getitem_69)
    mul_88: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    mul_89: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_88, primals_94);  mul_88 = None
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
    sub_23: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_92, getitem_71)
    mul_90: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
    mul_91: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_90, primals_96);  mul_90 = None
    add_94: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_91, primals_97);  mul_91 = primals_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_46: "f32[512, 768]" = torch.ops.aten.view.default(add_94, [512, 768])
    permute_23: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_98, [1, 0]);  primals_98 = None
    addmm_23: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_99, view_46, permute_23);  primals_99 = None
    view_47: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_23, [1, 512, 3072]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_92: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_47, 0.5)
    pow_12: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_47, 3.0)
    mul_93: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
    add_95: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(view_47, mul_93);  mul_93 = None
    mul_94: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_95, 0.7978845608028654);  add_95 = None
    tanh_11: "f32[1, 512, 3072]" = torch.ops.aten.tanh.default(mul_94);  mul_94 = None
    alias_11: "f32[1, 512, 3072]" = torch.ops.aten.alias.default(tanh_11)
    add_96: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
    mul_95: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_92, add_96)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_48: "f32[512, 3072]" = torch.ops.aten.view.default(mul_95, [512, 3072]);  mul_95 = None
    permute_24: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_100, [1, 0]);  primals_100 = None
    addmm_24: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_101, view_48, permute_24);  primals_101 = None
    view_49: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_24, [1, 512, 768]);  addmm_24 = None
    
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
    sub_24: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_97, getitem_75)
    mul_96: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
    mul_97: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_96, primals_102);  mul_96 = None
    add_99: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_97, primals_103);  mul_97 = primals_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:345, code: hidden_states = self.dense(hidden_states)
    view_50: "f32[512, 768]" = torch.ops.aten.view.default(add_99, [512, 768]);  add_99 = None
    permute_26: "f32[768, 768]" = torch.ops.aten.permute.default(primals_106, [1, 0]);  primals_106 = None
    addmm_26: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_107, view_50, permute_26);  primals_107 = None
    view_51: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_26, [1, 512, 768]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_98: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_51, 0.5)
    pow_13: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(view_51, 3.0)
    mul_99: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(pow_13, 0.044715);  pow_13 = None
    add_100: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_51, mul_99);  mul_99 = None
    mul_100: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_100, 0.7978845608028654);  add_100 = None
    tanh_13: "f32[1, 512, 768]" = torch.ops.aten.tanh.default(mul_100);  mul_100 = None
    alias_13: "f32[1, 512, 768]" = torch.ops.aten.alias.default(tanh_13)
    add_101: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(tanh_13, 1.0);  tanh_13 = None
    mul_101: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_98, add_101)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:347, code: hidden_states = self.LayerNorm(hidden_states)
    var_mean_25 = torch.ops.aten.var_mean.correction(mul_101, [2], correction = 0, keepdim = True)
    getitem_76: "f32[1, 512, 1]" = var_mean_25[0]
    getitem_77: "f32[1, 512, 1]" = var_mean_25[1];  var_mean_25 = None
    add_102: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-12);  getitem_76 = None
    rsqrt_25: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
    sub_25: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_101, getitem_77)
    mul_102: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
    mul_103: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_102, primals_108);  mul_102 = None
    add_103: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_103, primals_109);  mul_103 = primals_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:365, code: hidden_states = self.decoder(hidden_states)
    view_52: "f32[512, 768]" = torch.ops.aten.view.default(add_103, [512, 768]);  add_103 = None
    permute_27: "f32[768, 32000]" = torch.ops.aten.permute.default(primals_110, [1, 0]);  primals_110 = None
    addmm_27: "f32[512, 32000]" = torch.ops.aten.addmm.default(primals_111, view_52, permute_27);  primals_111 = None
    view_53: "f32[1, 512, 32000]" = torch.ops.aten.view.default(addmm_27, [1, 512, 32000]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:775, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    view_54: "f32[512, 32000]" = torch.ops.aten.view.default(view_53, [-1, 32000])
    view_55: "i64[512]" = torch.ops.aten.view.default(primals_115, [-1]);  primals_115 = None
    amax: "f32[512, 1]" = torch.ops.aten.amax.default(view_54, [1], True)
    sub_26: "f32[512, 32000]" = torch.ops.aten.sub.Tensor(view_54, amax);  view_54 = amax = None
    exp: "f32[512, 32000]" = torch.ops.aten.exp.default(sub_26)
    sum_1: "f32[512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
    log: "f32[512, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
    sub_27: "f32[512, 32000]" = torch.ops.aten.sub.Tensor(sub_26, log);  sub_26 = log = None
    alias_14: "f32[512, 32000]" = torch.ops.aten.alias.default(sub_27)
    ne: "b8[512]" = torch.ops.aten.ne.Scalar(view_55, -100)
    scalar_tensor: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'))
    where: "i64[512]" = torch.ops.aten.where.self(ne, view_55, scalar_tensor);  ne = scalar_tensor = None
    unsqueeze: "i64[512, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
    gather: "f32[512, 1]" = torch.ops.aten.gather.default(sub_27, 1, unsqueeze);  sub_27 = unsqueeze = None
    squeeze: "f32[512]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[512]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
    ne_1: "b8[512]" = torch.ops.aten.ne.Scalar(view_55, -100)
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_1: "f32[512]" = torch.ops.aten.where.self(ne_1, neg, scalar_tensor_1);  ne_1 = neg = scalar_tensor_1 = None
    ne_2: "b8[512]" = torch.ops.aten.ne.Scalar(view_55, -100)
    sum_2: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
    convert_element_type_12: "f32[]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
    sum_3: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
    div: "f32[]" = torch.ops.aten.div.Tensor(sum_3, convert_element_type_12);  sum_3 = None
    div_1: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_12);  tangents_1 = convert_element_type_12 = None
    unsqueeze_1: "i64[512, 1]" = torch.ops.aten.unsqueeze.default(view_55, 1);  view_55 = None
    ne_3: "b8[512, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_1, -100)
    scalar_tensor_2: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'))
    where_2: "i64[512, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_1, scalar_tensor_2);  ne_3 = scalar_tensor_2 = None
    full: "f32[512, 32000]" = torch.ops.aten.full.default([512, 32000], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    scatter: "f32[512, 32000]" = torch.ops.aten.scatter.value(full, 1, where_2, -1.0);  full = where_2 = None
    ne_4: "b8[512, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_1, -100);  unsqueeze_1 = None
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_3: "f32[512, 1]" = torch.ops.aten.where.self(ne_4, div_1, scalar_tensor_3);  ne_4 = div_1 = scalar_tensor_3 = None
    mul_104: "f32[512, 32000]" = torch.ops.aten.mul.Tensor(scatter, where_3);  scatter = where_3 = None
    alias_15: "f32[512, 32000]" = torch.ops.aten.alias.default(alias_14);  alias_14 = None
    exp_1: "f32[512, 32000]" = torch.ops.aten.exp.default(alias_15);  alias_15 = None
    sum_4: "f32[512, 1]" = torch.ops.aten.sum.dim_IntList(mul_104, [1], True)
    mul_105: "f32[512, 32000]" = torch.ops.aten.mul.Tensor(exp_1, sum_4);  exp_1 = sum_4 = None
    sub_28: "f32[512, 32000]" = torch.ops.aten.sub.Tensor(mul_104, mul_105);  mul_104 = mul_105 = None
    view_56: "f32[1, 512, 32000]" = torch.ops.aten.view.default(sub_28, [1, 512, 32000]);  sub_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:775, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    add_104: "f32[1, 512, 32000]" = torch.ops.aten.add.Tensor(tangents_2, view_56);  tangents_2 = view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:365, code: hidden_states = self.decoder(hidden_states)
    view_57: "f32[512, 32000]" = torch.ops.aten.view.default(add_104, [512, 32000]);  add_104 = None
    permute_28: "f32[32000, 768]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
    mm: "f32[512, 768]" = torch.ops.aten.mm.default(view_57, permute_28);  permute_28 = None
    permute_29: "f32[32000, 512]" = torch.ops.aten.permute.default(view_57, [1, 0])
    mm_1: "f32[32000, 768]" = torch.ops.aten.mm.default(permute_29, view_52);  permute_29 = view_52 = None
    permute_30: "f32[768, 32000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_5: "f32[1, 32000]" = torch.ops.aten.sum.dim_IntList(view_57, [0], True);  view_57 = None
    view_58: "f32[32000]" = torch.ops.aten.view.default(sum_5, [32000]);  sum_5 = None
    permute_31: "f32[32000, 768]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
    view_59: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm, [1, 512, 768]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:347, code: hidden_states = self.LayerNorm(hidden_states)
    sub_29: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_101, getitem_77);  mul_101 = getitem_77 = None
    mul_106: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_25);  sub_29 = None
    mul_107: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_59, primals_108);  primals_108 = None
    mul_108: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_107, 768)
    sum_6: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_107, [2], True)
    mul_109: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_107, mul_106);  mul_107 = None
    sum_7: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_109, [2], True);  mul_109 = None
    mul_110: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_106, sum_7);  sum_7 = None
    sub_30: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_108, sum_6);  mul_108 = sum_6 = None
    sub_31: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_30, mul_110);  sub_30 = mul_110 = None
    div_2: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 768);  rsqrt_25 = None
    mul_111: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_2, sub_31);  div_2 = sub_31 = None
    mul_112: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_59, mul_106);  mul_106 = None
    sum_8: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_112, [0, 1]);  mul_112 = None
    sum_9: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_59, [0, 1]);  view_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_113: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_111, mul_98);  mul_98 = None
    mul_114: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_111, add_101);  mul_111 = add_101 = None
    alias_16: "f32[1, 512, 768]" = torch.ops.aten.alias.default(alias_13);  alias_13 = None
    mul_115: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(alias_16, alias_16);  alias_16 = None
    sub_32: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, mul_115);  mul_115 = None
    mul_116: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_113, sub_32);  mul_113 = sub_32 = None
    mul_117: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_116, 0.7978845608028654);  mul_116 = None
    mul_118: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_117, 0.044715)
    pow_14: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(view_51, 2.0);  view_51 = None
    mul_119: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_14, 3.0);  pow_14 = None
    mul_120: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_118, mul_119);  mul_118 = mul_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_105: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_117, mul_120);  mul_117 = mul_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_121: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_114, 0.5);  mul_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_106: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_105, mul_121);  add_105 = mul_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:345, code: hidden_states = self.dense(hidden_states)
    view_60: "f32[512, 768]" = torch.ops.aten.view.default(add_106, [512, 768]);  add_106 = None
    permute_32: "f32[768, 768]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
    mm_2: "f32[512, 768]" = torch.ops.aten.mm.default(view_60, permute_32);  permute_32 = None
    permute_33: "f32[768, 512]" = torch.ops.aten.permute.default(view_60, [1, 0])
    mm_3: "f32[768, 768]" = torch.ops.aten.mm.default(permute_33, view_50);  permute_33 = view_50 = None
    permute_34: "f32[768, 768]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_10: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_60, [0], True);  view_60 = None
    view_61: "f32[768]" = torch.ops.aten.view.default(sum_10, [768]);  sum_10 = None
    permute_35: "f32[768, 768]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    view_62: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_2, [1, 512, 768]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_33: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_97, getitem_75);  add_97 = getitem_75 = None
    mul_122: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_24);  sub_33 = None
    mul_123: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_62, primals_102);  primals_102 = None
    mul_124: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_123, 768)
    sum_11: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_123, [2], True)
    mul_125: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_123, mul_122);  mul_123 = None
    sum_12: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_125, [2], True);  mul_125 = None
    mul_126: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_122, sum_12);  sum_12 = None
    sub_34: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_124, sum_11);  mul_124 = sum_11 = None
    sub_35: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_34, mul_126);  sub_34 = mul_126 = None
    div_3: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None
    mul_127: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_3, sub_35);  div_3 = sub_35 = None
    mul_128: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_62, mul_122);  mul_122 = None
    sum_13: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_128, [0, 1]);  mul_128 = None
    sum_14: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_62, [0, 1]);  view_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_13: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_73, torch.float32);  getitem_73 = None
    mul_129: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
    mul_130: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_127, mul_129);  mul_129 = None
    clone: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_130, memory_format = torch.contiguous_format);  mul_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_63: "f32[512, 768]" = torch.ops.aten.view.default(clone, [512, 768]);  clone = None
    permute_36: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
    mm_4: "f32[512, 3072]" = torch.ops.aten.mm.default(view_63, permute_36);  permute_36 = None
    permute_37: "f32[768, 512]" = torch.ops.aten.permute.default(view_63, [1, 0])
    mm_5: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_37, view_48);  permute_37 = view_48 = None
    permute_38: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_15: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_63, [0], True);  view_63 = None
    view_64: "f32[768]" = torch.ops.aten.view.default(sum_15, [768]);  sum_15 = None
    permute_39: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
    view_65: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_4, [1, 512, 3072]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_131: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_65, mul_92);  mul_92 = None
    mul_132: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_65, add_96);  view_65 = add_96 = None
    alias_17: "f32[1, 512, 3072]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    mul_133: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(alias_17, alias_17);  alias_17 = None
    sub_36: "f32[1, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_133);  mul_133 = None
    mul_134: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_131, sub_36);  mul_131 = sub_36 = None
    mul_135: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_134, 0.7978845608028654);  mul_134 = None
    mul_136: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_135, 0.044715)
    pow_15: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_47, 2.0);  view_47 = None
    mul_137: "f32[1, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_15, 3.0);  pow_15 = None
    mul_138: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_136, mul_137);  mul_136 = mul_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_107: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_135, mul_138);  mul_135 = mul_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_139: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_132, 0.5);  mul_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_108: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(add_107, mul_139);  add_107 = mul_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_66: "f32[512, 3072]" = torch.ops.aten.view.default(add_108, [512, 3072]);  add_108 = None
    permute_40: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    mm_6: "f32[512, 768]" = torch.ops.aten.mm.default(view_66, permute_40);  permute_40 = None
    permute_41: "f32[3072, 512]" = torch.ops.aten.permute.default(view_66, [1, 0])
    mm_7: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_41, view_46);  permute_41 = view_46 = None
    permute_42: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_16: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_66, [0], True);  view_66 = None
    view_67: "f32[3072]" = torch.ops.aten.view.default(sum_16, [3072]);  sum_16 = None
    permute_43: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    view_68: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_6, [1, 512, 768]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    add_109: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_127, view_68);  mul_127 = view_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    sub_37: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_92, getitem_71);  add_92 = getitem_71 = None
    mul_140: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_23);  sub_37 = None
    mul_141: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_109, primals_96);  primals_96 = None
    mul_142: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_141, 768)
    sum_17: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_141, [2], True)
    mul_143: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_141, mul_140);  mul_141 = None
    sum_18: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_143, [2], True);  mul_143 = None
    mul_144: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_140, sum_18);  sum_18 = None
    sub_38: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_142, sum_17);  mul_142 = sum_17 = None
    sub_39: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_38, mul_144);  sub_38 = mul_144 = None
    div_4: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
    mul_145: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_4, sub_39);  div_4 = sub_39 = None
    mul_146: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_109, mul_140);  mul_140 = None
    sum_19: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_146, [0, 1]);  mul_146 = None
    sum_20: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_109, [0, 1]);  add_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    full_1: "f32[1, 512, 768, 2]" = torch.ops.aten.full.default([1, 512, 768, 2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    select_scatter: "f32[1, 512, 768, 2]" = torch.ops.aten.select_scatter.default(full_1, mul_145, 3, 0);  full_1 = None
    view_as_complex: "c64[1, 512, 768]" = torch.ops.aten.view_as_complex.default(select_scatter);  select_scatter = None
    _fft_c2c_12: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(view_as_complex, [1, 2], 0, False);  view_as_complex = None
    view_as_real_12: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_12);  _fft_c2c_12 = None
    select_13: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_12, 3, 0);  view_as_real_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    add_110: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_145, select_13);  mul_145 = select_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_40: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_89, getitem_69);  add_89 = getitem_69 = None
    mul_147: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_22);  sub_40 = None
    mul_148: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_110, primals_94);  primals_94 = None
    mul_149: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_148, 768)
    sum_21: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_148, [2], True)
    mul_150: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_148, mul_147);  mul_148 = None
    sum_22: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_150, [2], True);  mul_150 = None
    mul_151: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_147, sum_22);  sum_22 = None
    sub_41: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_149, sum_21);  mul_149 = sum_21 = None
    sub_42: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_41, mul_151);  sub_41 = mul_151 = None
    div_5: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
    mul_152: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_5, sub_42);  div_5 = sub_42 = None
    mul_153: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_110, mul_147);  mul_147 = None
    sum_23: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_153, [0, 1]);  mul_153 = None
    sum_24: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_110, [0, 1]);  add_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_14: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_67, torch.float32);  getitem_67 = None
    mul_154: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 1.1111111111111112);  convert_element_type_14 = None
    mul_155: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_152, mul_154);  mul_154 = None
    clone_1: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_155, memory_format = torch.contiguous_format);  mul_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_69: "f32[512, 768]" = torch.ops.aten.view.default(clone_1, [512, 768]);  clone_1 = None
    permute_44: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    mm_8: "f32[512, 3072]" = torch.ops.aten.mm.default(view_69, permute_44);  permute_44 = None
    permute_45: "f32[768, 512]" = torch.ops.aten.permute.default(view_69, [1, 0])
    mm_9: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_45, view_44);  permute_45 = view_44 = None
    permute_46: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_25: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_69, [0], True);  view_69 = None
    view_70: "f32[768]" = torch.ops.aten.view.default(sum_25, [768]);  sum_25 = None
    permute_47: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
    view_71: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_8, [1, 512, 3072]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_156: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_71, mul_84);  mul_84 = None
    mul_157: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_71, add_88);  view_71 = add_88 = None
    alias_18: "f32[1, 512, 3072]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    mul_158: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(alias_18, alias_18);  alias_18 = None
    sub_43: "f32[1, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_158);  mul_158 = None
    mul_159: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_156, sub_43);  mul_156 = sub_43 = None
    mul_160: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_159, 0.7978845608028654);  mul_159 = None
    mul_161: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_160, 0.044715)
    pow_16: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_43, 2.0);  view_43 = None
    mul_162: "f32[1, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_16, 3.0);  pow_16 = None
    mul_163: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_161, mul_162);  mul_161 = mul_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_111: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_160, mul_163);  mul_160 = mul_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_164: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_157, 0.5);  mul_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_112: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(add_111, mul_164);  add_111 = mul_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_72: "f32[512, 3072]" = torch.ops.aten.view.default(add_112, [512, 3072]);  add_112 = None
    permute_48: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    mm_10: "f32[512, 768]" = torch.ops.aten.mm.default(view_72, permute_48);  permute_48 = None
    permute_49: "f32[3072, 512]" = torch.ops.aten.permute.default(view_72, [1, 0])
    mm_11: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_49, view_42);  permute_49 = view_42 = None
    permute_50: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_26: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_72, [0], True);  view_72 = None
    view_73: "f32[3072]" = torch.ops.aten.view.default(sum_26, [3072]);  sum_26 = None
    permute_51: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_50, [1, 0]);  permute_50 = None
    view_74: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_10, [1, 512, 768]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    add_113: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_152, view_74);  mul_152 = view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    sub_44: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_84, getitem_65);  add_84 = getitem_65 = None
    mul_165: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_21);  sub_44 = None
    mul_166: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_113, primals_88);  primals_88 = None
    mul_167: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_166, 768)
    sum_27: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_166, [2], True)
    mul_168: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_166, mul_165);  mul_166 = None
    sum_28: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_168, [2], True);  mul_168 = None
    mul_169: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_165, sum_28);  sum_28 = None
    sub_45: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_167, sum_27);  mul_167 = sum_27 = None
    sub_46: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_45, mul_169);  sub_45 = mul_169 = None
    div_6: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
    mul_170: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_6, sub_46);  div_6 = sub_46 = None
    mul_171: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_113, mul_165);  mul_165 = None
    sum_29: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_171, [0, 1]);  mul_171 = None
    sum_30: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_113, [0, 1]);  add_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    full_2: "f32[1, 512, 768, 2]" = torch.ops.aten.full.default([1, 512, 768, 2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    select_scatter_1: "f32[1, 512, 768, 2]" = torch.ops.aten.select_scatter.default(full_2, mul_170, 3, 0);  full_2 = None
    view_as_complex_1: "c64[1, 512, 768]" = torch.ops.aten.view_as_complex.default(select_scatter_1);  select_scatter_1 = None
    _fft_c2c_13: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(view_as_complex_1, [1, 2], 0, False);  view_as_complex_1 = None
    view_as_real_13: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_13);  _fft_c2c_13 = None
    select_14: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_13, 3, 0);  view_as_real_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    add_114: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_170, select_14);  mul_170 = select_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_47: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_81, getitem_63);  add_81 = getitem_63 = None
    mul_172: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_20);  sub_47 = None
    mul_173: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_114, primals_86);  primals_86 = None
    mul_174: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_173, 768)
    sum_31: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_173, [2], True)
    mul_175: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_173, mul_172);  mul_173 = None
    sum_32: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_175, [2], True);  mul_175 = None
    mul_176: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_172, sum_32);  sum_32 = None
    sub_48: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_174, sum_31);  mul_174 = sum_31 = None
    sub_49: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_48, mul_176);  sub_48 = mul_176 = None
    div_7: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
    mul_177: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_7, sub_49);  div_7 = sub_49 = None
    mul_178: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_114, mul_172);  mul_172 = None
    sum_33: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_178, [0, 1]);  mul_178 = None
    sum_34: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_114, [0, 1]);  add_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_15: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_61, torch.float32);  getitem_61 = None
    mul_179: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_15, 1.1111111111111112);  convert_element_type_15 = None
    mul_180: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_177, mul_179);  mul_179 = None
    clone_2: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_180, memory_format = torch.contiguous_format);  mul_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_75: "f32[512, 768]" = torch.ops.aten.view.default(clone_2, [512, 768]);  clone_2 = None
    permute_52: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    mm_12: "f32[512, 3072]" = torch.ops.aten.mm.default(view_75, permute_52);  permute_52 = None
    permute_53: "f32[768, 512]" = torch.ops.aten.permute.default(view_75, [1, 0])
    mm_13: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_53, view_40);  permute_53 = view_40 = None
    permute_54: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_35: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_75, [0], True);  view_75 = None
    view_76: "f32[768]" = torch.ops.aten.view.default(sum_35, [768]);  sum_35 = None
    permute_55: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    view_77: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_12, [1, 512, 3072]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_181: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_77, mul_76);  mul_76 = None
    mul_182: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_77, add_80);  view_77 = add_80 = None
    alias_19: "f32[1, 512, 3072]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    mul_183: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(alias_19, alias_19);  alias_19 = None
    sub_50: "f32[1, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_183);  mul_183 = None
    mul_184: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_181, sub_50);  mul_181 = sub_50 = None
    mul_185: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_184, 0.7978845608028654);  mul_184 = None
    mul_186: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_185, 0.044715)
    pow_17: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_39, 2.0);  view_39 = None
    mul_187: "f32[1, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_17, 3.0);  pow_17 = None
    mul_188: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_186, mul_187);  mul_186 = mul_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_115: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_185, mul_188);  mul_185 = mul_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_189: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_182, 0.5);  mul_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_116: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(add_115, mul_189);  add_115 = mul_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_78: "f32[512, 3072]" = torch.ops.aten.view.default(add_116, [512, 3072]);  add_116 = None
    permute_56: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
    mm_14: "f32[512, 768]" = torch.ops.aten.mm.default(view_78, permute_56);  permute_56 = None
    permute_57: "f32[3072, 512]" = torch.ops.aten.permute.default(view_78, [1, 0])
    mm_15: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_57, view_38);  permute_57 = view_38 = None
    permute_58: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_36: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_78, [0], True);  view_78 = None
    view_79: "f32[3072]" = torch.ops.aten.view.default(sum_36, [3072]);  sum_36 = None
    permute_59: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
    view_80: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_14, [1, 512, 768]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    add_117: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_177, view_80);  mul_177 = view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    sub_51: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_76, getitem_59);  add_76 = getitem_59 = None
    mul_190: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_19);  sub_51 = None
    mul_191: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_117, primals_80);  primals_80 = None
    mul_192: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_191, 768)
    sum_37: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_191, [2], True)
    mul_193: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_191, mul_190);  mul_191 = None
    sum_38: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_193, [2], True);  mul_193 = None
    mul_194: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_190, sum_38);  sum_38 = None
    sub_52: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_192, sum_37);  mul_192 = sum_37 = None
    sub_53: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_52, mul_194);  sub_52 = mul_194 = None
    div_8: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
    mul_195: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_8, sub_53);  div_8 = sub_53 = None
    mul_196: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_117, mul_190);  mul_190 = None
    sum_39: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_196, [0, 1]);  mul_196 = None
    sum_40: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_117, [0, 1]);  add_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    full_3: "f32[1, 512, 768, 2]" = torch.ops.aten.full.default([1, 512, 768, 2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    select_scatter_2: "f32[1, 512, 768, 2]" = torch.ops.aten.select_scatter.default(full_3, mul_195, 3, 0);  full_3 = None
    view_as_complex_2: "c64[1, 512, 768]" = torch.ops.aten.view_as_complex.default(select_scatter_2);  select_scatter_2 = None
    _fft_c2c_14: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(view_as_complex_2, [1, 2], 0, False);  view_as_complex_2 = None
    view_as_real_14: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_14);  _fft_c2c_14 = None
    select_15: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_14, 3, 0);  view_as_real_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    add_118: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_195, select_15);  mul_195 = select_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_54: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_73, getitem_57);  add_73 = getitem_57 = None
    mul_197: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_18);  sub_54 = None
    mul_198: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_118, primals_78);  primals_78 = None
    mul_199: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_198, 768)
    sum_41: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_198, [2], True)
    mul_200: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_198, mul_197);  mul_198 = None
    sum_42: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_200, [2], True);  mul_200 = None
    mul_201: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_197, sum_42);  sum_42 = None
    sub_55: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_199, sum_41);  mul_199 = sum_41 = None
    sub_56: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_55, mul_201);  sub_55 = mul_201 = None
    div_9: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
    mul_202: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_9, sub_56);  div_9 = sub_56 = None
    mul_203: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_118, mul_197);  mul_197 = None
    sum_43: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_203, [0, 1]);  mul_203 = None
    sum_44: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_118, [0, 1]);  add_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_16: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_55, torch.float32);  getitem_55 = None
    mul_204: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_16, 1.1111111111111112);  convert_element_type_16 = None
    mul_205: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_202, mul_204);  mul_204 = None
    clone_3: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_205, memory_format = torch.contiguous_format);  mul_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_81: "f32[512, 768]" = torch.ops.aten.view.default(clone_3, [512, 768]);  clone_3 = None
    permute_60: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
    mm_16: "f32[512, 3072]" = torch.ops.aten.mm.default(view_81, permute_60);  permute_60 = None
    permute_61: "f32[768, 512]" = torch.ops.aten.permute.default(view_81, [1, 0])
    mm_17: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_61, view_36);  permute_61 = view_36 = None
    permute_62: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_45: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_81, [0], True);  view_81 = None
    view_82: "f32[768]" = torch.ops.aten.view.default(sum_45, [768]);  sum_45 = None
    permute_63: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_62, [1, 0]);  permute_62 = None
    view_83: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_16, [1, 512, 3072]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_206: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_83, mul_68);  mul_68 = None
    mul_207: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_83, add_72);  view_83 = add_72 = None
    alias_20: "f32[1, 512, 3072]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    mul_208: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(alias_20, alias_20);  alias_20 = None
    sub_57: "f32[1, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_208);  mul_208 = None
    mul_209: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_206, sub_57);  mul_206 = sub_57 = None
    mul_210: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_209, 0.7978845608028654);  mul_209 = None
    mul_211: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_210, 0.044715)
    pow_18: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_35, 2.0);  view_35 = None
    mul_212: "f32[1, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_18, 3.0);  pow_18 = None
    mul_213: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_211, mul_212);  mul_211 = mul_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_119: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_210, mul_213);  mul_210 = mul_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_214: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_207, 0.5);  mul_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_120: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(add_119, mul_214);  add_119 = mul_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_84: "f32[512, 3072]" = torch.ops.aten.view.default(add_120, [512, 3072]);  add_120 = None
    permute_64: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
    mm_18: "f32[512, 768]" = torch.ops.aten.mm.default(view_84, permute_64);  permute_64 = None
    permute_65: "f32[3072, 512]" = torch.ops.aten.permute.default(view_84, [1, 0])
    mm_19: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_65, view_34);  permute_65 = view_34 = None
    permute_66: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_46: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_84, [0], True);  view_84 = None
    view_85: "f32[3072]" = torch.ops.aten.view.default(sum_46, [3072]);  sum_46 = None
    permute_67: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    view_86: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_18, [1, 512, 768]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    add_121: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_202, view_86);  mul_202 = view_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    sub_58: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_68, getitem_53);  add_68 = getitem_53 = None
    mul_215: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_17);  sub_58 = None
    mul_216: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_121, primals_72);  primals_72 = None
    mul_217: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_216, 768)
    sum_47: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_216, [2], True)
    mul_218: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_216, mul_215);  mul_216 = None
    sum_48: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_218, [2], True);  mul_218 = None
    mul_219: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_215, sum_48);  sum_48 = None
    sub_59: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_217, sum_47);  mul_217 = sum_47 = None
    sub_60: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_59, mul_219);  sub_59 = mul_219 = None
    div_10: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
    mul_220: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_10, sub_60);  div_10 = sub_60 = None
    mul_221: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_121, mul_215);  mul_215 = None
    sum_49: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_221, [0, 1]);  mul_221 = None
    sum_50: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_121, [0, 1]);  add_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    full_4: "f32[1, 512, 768, 2]" = torch.ops.aten.full.default([1, 512, 768, 2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    select_scatter_3: "f32[1, 512, 768, 2]" = torch.ops.aten.select_scatter.default(full_4, mul_220, 3, 0);  full_4 = None
    view_as_complex_3: "c64[1, 512, 768]" = torch.ops.aten.view_as_complex.default(select_scatter_3);  select_scatter_3 = None
    _fft_c2c_15: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(view_as_complex_3, [1, 2], 0, False);  view_as_complex_3 = None
    view_as_real_15: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_15);  _fft_c2c_15 = None
    select_16: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_15, 3, 0);  view_as_real_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    add_122: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_220, select_16);  mul_220 = select_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_61: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_65, getitem_51);  add_65 = getitem_51 = None
    mul_222: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_16);  sub_61 = None
    mul_223: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_122, primals_70);  primals_70 = None
    mul_224: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_223, 768)
    sum_51: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_223, [2], True)
    mul_225: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_223, mul_222);  mul_223 = None
    sum_52: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_225, [2], True);  mul_225 = None
    mul_226: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_222, sum_52);  sum_52 = None
    sub_62: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_224, sum_51);  mul_224 = sum_51 = None
    sub_63: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_62, mul_226);  sub_62 = mul_226 = None
    div_11: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
    mul_227: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_11, sub_63);  div_11 = sub_63 = None
    mul_228: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_122, mul_222);  mul_222 = None
    sum_53: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_228, [0, 1]);  mul_228 = None
    sum_54: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_122, [0, 1]);  add_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_17: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_49, torch.float32);  getitem_49 = None
    mul_229: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_17, 1.1111111111111112);  convert_element_type_17 = None
    mul_230: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_227, mul_229);  mul_229 = None
    clone_4: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_230, memory_format = torch.contiguous_format);  mul_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_87: "f32[512, 768]" = torch.ops.aten.view.default(clone_4, [512, 768]);  clone_4 = None
    permute_68: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None
    mm_20: "f32[512, 3072]" = torch.ops.aten.mm.default(view_87, permute_68);  permute_68 = None
    permute_69: "f32[768, 512]" = torch.ops.aten.permute.default(view_87, [1, 0])
    mm_21: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_69, view_32);  permute_69 = view_32 = None
    permute_70: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_55: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_87, [0], True);  view_87 = None
    view_88: "f32[768]" = torch.ops.aten.view.default(sum_55, [768]);  sum_55 = None
    permute_71: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
    view_89: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_20, [1, 512, 3072]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_231: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_89, mul_60);  mul_60 = None
    mul_232: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_89, add_64);  view_89 = add_64 = None
    alias_21: "f32[1, 512, 3072]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    mul_233: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(alias_21, alias_21);  alias_21 = None
    sub_64: "f32[1, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_233);  mul_233 = None
    mul_234: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_231, sub_64);  mul_231 = sub_64 = None
    mul_235: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_234, 0.7978845608028654);  mul_234 = None
    mul_236: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_235, 0.044715)
    pow_19: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_31, 2.0);  view_31 = None
    mul_237: "f32[1, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_19, 3.0);  pow_19 = None
    mul_238: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_236, mul_237);  mul_236 = mul_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_123: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_235, mul_238);  mul_235 = mul_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_239: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_232, 0.5);  mul_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_124: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(add_123, mul_239);  add_123 = mul_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_90: "f32[512, 3072]" = torch.ops.aten.view.default(add_124, [512, 3072]);  add_124 = None
    permute_72: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
    mm_22: "f32[512, 768]" = torch.ops.aten.mm.default(view_90, permute_72);  permute_72 = None
    permute_73: "f32[3072, 512]" = torch.ops.aten.permute.default(view_90, [1, 0])
    mm_23: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_73, view_30);  permute_73 = view_30 = None
    permute_74: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_56: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_90, [0], True);  view_90 = None
    view_91: "f32[3072]" = torch.ops.aten.view.default(sum_56, [3072]);  sum_56 = None
    permute_75: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
    view_92: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_22, [1, 512, 768]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    add_125: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_227, view_92);  mul_227 = view_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    sub_65: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_60, getitem_47);  add_60 = getitem_47 = None
    mul_240: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_15);  sub_65 = None
    mul_241: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_125, primals_64);  primals_64 = None
    mul_242: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_241, 768)
    sum_57: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_241, [2], True)
    mul_243: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_241, mul_240);  mul_241 = None
    sum_58: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_243, [2], True);  mul_243 = None
    mul_244: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_240, sum_58);  sum_58 = None
    sub_66: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_242, sum_57);  mul_242 = sum_57 = None
    sub_67: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_66, mul_244);  sub_66 = mul_244 = None
    div_12: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
    mul_245: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_12, sub_67);  div_12 = sub_67 = None
    mul_246: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_125, mul_240);  mul_240 = None
    sum_59: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_246, [0, 1]);  mul_246 = None
    sum_60: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_125, [0, 1]);  add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    full_5: "f32[1, 512, 768, 2]" = torch.ops.aten.full.default([1, 512, 768, 2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    select_scatter_4: "f32[1, 512, 768, 2]" = torch.ops.aten.select_scatter.default(full_5, mul_245, 3, 0);  full_5 = None
    view_as_complex_4: "c64[1, 512, 768]" = torch.ops.aten.view_as_complex.default(select_scatter_4);  select_scatter_4 = None
    _fft_c2c_16: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(view_as_complex_4, [1, 2], 0, False);  view_as_complex_4 = None
    view_as_real_16: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_16);  _fft_c2c_16 = None
    select_17: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_16, 3, 0);  view_as_real_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    add_126: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_245, select_17);  mul_245 = select_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_68: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_57, getitem_45);  add_57 = getitem_45 = None
    mul_247: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_14);  sub_68 = None
    mul_248: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_126, primals_62);  primals_62 = None
    mul_249: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_248, 768)
    sum_61: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_248, [2], True)
    mul_250: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_248, mul_247);  mul_248 = None
    sum_62: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_250, [2], True);  mul_250 = None
    mul_251: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_247, sum_62);  sum_62 = None
    sub_69: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_249, sum_61);  mul_249 = sum_61 = None
    sub_70: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_69, mul_251);  sub_69 = mul_251 = None
    div_13: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
    mul_252: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_13, sub_70);  div_13 = sub_70 = None
    mul_253: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_126, mul_247);  mul_247 = None
    sum_63: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_253, [0, 1]);  mul_253 = None
    sum_64: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_126, [0, 1]);  add_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_18: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_43, torch.float32);  getitem_43 = None
    mul_254: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_18, 1.1111111111111112);  convert_element_type_18 = None
    mul_255: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_252, mul_254);  mul_254 = None
    clone_5: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_255, memory_format = torch.contiguous_format);  mul_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_93: "f32[512, 768]" = torch.ops.aten.view.default(clone_5, [512, 768]);  clone_5 = None
    permute_76: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
    mm_24: "f32[512, 3072]" = torch.ops.aten.mm.default(view_93, permute_76);  permute_76 = None
    permute_77: "f32[768, 512]" = torch.ops.aten.permute.default(view_93, [1, 0])
    mm_25: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_77, view_28);  permute_77 = view_28 = None
    permute_78: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_65: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_93, [0], True);  view_93 = None
    view_94: "f32[768]" = torch.ops.aten.view.default(sum_65, [768]);  sum_65 = None
    permute_79: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
    view_95: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_24, [1, 512, 3072]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_256: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_95, mul_52);  mul_52 = None
    mul_257: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_95, add_56);  view_95 = add_56 = None
    alias_22: "f32[1, 512, 3072]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    mul_258: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(alias_22, alias_22);  alias_22 = None
    sub_71: "f32[1, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_258);  mul_258 = None
    mul_259: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_256, sub_71);  mul_256 = sub_71 = None
    mul_260: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_259, 0.7978845608028654);  mul_259 = None
    mul_261: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_260, 0.044715)
    pow_20: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_27, 2.0);  view_27 = None
    mul_262: "f32[1, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_20, 3.0);  pow_20 = None
    mul_263: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_261, mul_262);  mul_261 = mul_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_127: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_260, mul_263);  mul_260 = mul_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_264: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_257, 0.5);  mul_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_128: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(add_127, mul_264);  add_127 = mul_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_96: "f32[512, 3072]" = torch.ops.aten.view.default(add_128, [512, 3072]);  add_128 = None
    permute_80: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    mm_26: "f32[512, 768]" = torch.ops.aten.mm.default(view_96, permute_80);  permute_80 = None
    permute_81: "f32[3072, 512]" = torch.ops.aten.permute.default(view_96, [1, 0])
    mm_27: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_81, view_26);  permute_81 = view_26 = None
    permute_82: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_66: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_96, [0], True);  view_96 = None
    view_97: "f32[3072]" = torch.ops.aten.view.default(sum_66, [3072]);  sum_66 = None
    permute_83: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_82, [1, 0]);  permute_82 = None
    view_98: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_26, [1, 512, 768]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    add_129: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_252, view_98);  mul_252 = view_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    sub_72: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_52, getitem_41);  add_52 = getitem_41 = None
    mul_265: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_13);  sub_72 = None
    mul_266: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_129, primals_56);  primals_56 = None
    mul_267: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_266, 768)
    sum_67: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_266, [2], True)
    mul_268: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_266, mul_265);  mul_266 = None
    sum_68: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_268, [2], True);  mul_268 = None
    mul_269: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_265, sum_68);  sum_68 = None
    sub_73: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_267, sum_67);  mul_267 = sum_67 = None
    sub_74: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_73, mul_269);  sub_73 = mul_269 = None
    div_14: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
    mul_270: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_14, sub_74);  div_14 = sub_74 = None
    mul_271: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_129, mul_265);  mul_265 = None
    sum_69: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_271, [0, 1]);  mul_271 = None
    sum_70: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_129, [0, 1]);  add_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    full_6: "f32[1, 512, 768, 2]" = torch.ops.aten.full.default([1, 512, 768, 2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    select_scatter_5: "f32[1, 512, 768, 2]" = torch.ops.aten.select_scatter.default(full_6, mul_270, 3, 0);  full_6 = None
    view_as_complex_5: "c64[1, 512, 768]" = torch.ops.aten.view_as_complex.default(select_scatter_5);  select_scatter_5 = None
    _fft_c2c_17: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(view_as_complex_5, [1, 2], 0, False);  view_as_complex_5 = None
    view_as_real_17: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_17);  _fft_c2c_17 = None
    select_18: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_17, 3, 0);  view_as_real_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    add_130: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_270, select_18);  mul_270 = select_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_75: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_49, getitem_39);  add_49 = getitem_39 = None
    mul_272: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_12);  sub_75 = None
    mul_273: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_130, primals_54);  primals_54 = None
    mul_274: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_273, 768)
    sum_71: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_273, [2], True)
    mul_275: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_273, mul_272);  mul_273 = None
    sum_72: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_275, [2], True);  mul_275 = None
    mul_276: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_272, sum_72);  sum_72 = None
    sub_76: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_274, sum_71);  mul_274 = sum_71 = None
    sub_77: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_76, mul_276);  sub_76 = mul_276 = None
    div_15: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
    mul_277: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_15, sub_77);  div_15 = sub_77 = None
    mul_278: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_130, mul_272);  mul_272 = None
    sum_73: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_278, [0, 1]);  mul_278 = None
    sum_74: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_130, [0, 1]);  add_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_19: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_37, torch.float32);  getitem_37 = None
    mul_279: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_19, 1.1111111111111112);  convert_element_type_19 = None
    mul_280: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_277, mul_279);  mul_279 = None
    clone_6: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_280, memory_format = torch.contiguous_format);  mul_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_99: "f32[512, 768]" = torch.ops.aten.view.default(clone_6, [512, 768]);  clone_6 = None
    permute_84: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    mm_28: "f32[512, 3072]" = torch.ops.aten.mm.default(view_99, permute_84);  permute_84 = None
    permute_85: "f32[768, 512]" = torch.ops.aten.permute.default(view_99, [1, 0])
    mm_29: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_85, view_24);  permute_85 = view_24 = None
    permute_86: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_75: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_99, [0], True);  view_99 = None
    view_100: "f32[768]" = torch.ops.aten.view.default(sum_75, [768]);  sum_75 = None
    permute_87: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
    view_101: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_28, [1, 512, 3072]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_281: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_101, mul_44);  mul_44 = None
    mul_282: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_101, add_48);  view_101 = add_48 = None
    alias_23: "f32[1, 512, 3072]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    mul_283: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(alias_23, alias_23);  alias_23 = None
    sub_78: "f32[1, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_283);  mul_283 = None
    mul_284: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_281, sub_78);  mul_281 = sub_78 = None
    mul_285: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_284, 0.7978845608028654);  mul_284 = None
    mul_286: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_285, 0.044715)
    pow_21: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_23, 2.0);  view_23 = None
    mul_287: "f32[1, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_21, 3.0);  pow_21 = None
    mul_288: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_286, mul_287);  mul_286 = mul_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_131: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_285, mul_288);  mul_285 = mul_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_289: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_282, 0.5);  mul_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_132: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(add_131, mul_289);  add_131 = mul_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_102: "f32[512, 3072]" = torch.ops.aten.view.default(add_132, [512, 3072]);  add_132 = None
    permute_88: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    mm_30: "f32[512, 768]" = torch.ops.aten.mm.default(view_102, permute_88);  permute_88 = None
    permute_89: "f32[3072, 512]" = torch.ops.aten.permute.default(view_102, [1, 0])
    mm_31: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_89, view_22);  permute_89 = view_22 = None
    permute_90: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_76: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_102, [0], True);  view_102 = None
    view_103: "f32[3072]" = torch.ops.aten.view.default(sum_76, [3072]);  sum_76 = None
    permute_91: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
    view_104: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_30, [1, 512, 768]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    add_133: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_277, view_104);  mul_277 = view_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    sub_79: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_44, getitem_35);  add_44 = getitem_35 = None
    mul_290: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_11);  sub_79 = None
    mul_291: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_133, primals_48);  primals_48 = None
    mul_292: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_291, 768)
    sum_77: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_291, [2], True)
    mul_293: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_291, mul_290);  mul_291 = None
    sum_78: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_293, [2], True);  mul_293 = None
    mul_294: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_290, sum_78);  sum_78 = None
    sub_80: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_292, sum_77);  mul_292 = sum_77 = None
    sub_81: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_80, mul_294);  sub_80 = mul_294 = None
    div_16: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
    mul_295: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_16, sub_81);  div_16 = sub_81 = None
    mul_296: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_133, mul_290);  mul_290 = None
    sum_79: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_296, [0, 1]);  mul_296 = None
    sum_80: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_133, [0, 1]);  add_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    full_7: "f32[1, 512, 768, 2]" = torch.ops.aten.full.default([1, 512, 768, 2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    select_scatter_6: "f32[1, 512, 768, 2]" = torch.ops.aten.select_scatter.default(full_7, mul_295, 3, 0);  full_7 = None
    view_as_complex_6: "c64[1, 512, 768]" = torch.ops.aten.view_as_complex.default(select_scatter_6);  select_scatter_6 = None
    _fft_c2c_18: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(view_as_complex_6, [1, 2], 0, False);  view_as_complex_6 = None
    view_as_real_18: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_18);  _fft_c2c_18 = None
    select_19: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_18, 3, 0);  view_as_real_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    add_134: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_295, select_19);  mul_295 = select_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_82: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_41, getitem_33);  add_41 = getitem_33 = None
    mul_297: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_10);  sub_82 = None
    mul_298: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_134, primals_46);  primals_46 = None
    mul_299: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_298, 768)
    sum_81: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_298, [2], True)
    mul_300: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_298, mul_297);  mul_298 = None
    sum_82: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_300, [2], True);  mul_300 = None
    mul_301: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_297, sum_82);  sum_82 = None
    sub_83: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_299, sum_81);  mul_299 = sum_81 = None
    sub_84: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_83, mul_301);  sub_83 = mul_301 = None
    div_17: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
    mul_302: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_17, sub_84);  div_17 = sub_84 = None
    mul_303: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_134, mul_297);  mul_297 = None
    sum_83: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_303, [0, 1]);  mul_303 = None
    sum_84: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_134, [0, 1]);  add_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_20: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_31, torch.float32);  getitem_31 = None
    mul_304: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.1111111111111112);  convert_element_type_20 = None
    mul_305: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_302, mul_304);  mul_304 = None
    clone_7: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_305, memory_format = torch.contiguous_format);  mul_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_105: "f32[512, 768]" = torch.ops.aten.view.default(clone_7, [512, 768]);  clone_7 = None
    permute_92: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    mm_32: "f32[512, 3072]" = torch.ops.aten.mm.default(view_105, permute_92);  permute_92 = None
    permute_93: "f32[768, 512]" = torch.ops.aten.permute.default(view_105, [1, 0])
    mm_33: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_93, view_20);  permute_93 = view_20 = None
    permute_94: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_85: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_105, [0], True);  view_105 = None
    view_106: "f32[768]" = torch.ops.aten.view.default(sum_85, [768]);  sum_85 = None
    permute_95: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_94, [1, 0]);  permute_94 = None
    view_107: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_32, [1, 512, 3072]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_306: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, mul_36);  mul_36 = None
    mul_307: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, add_40);  view_107 = add_40 = None
    alias_24: "f32[1, 512, 3072]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    mul_308: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(alias_24, alias_24);  alias_24 = None
    sub_85: "f32[1, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_308);  mul_308 = None
    mul_309: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_306, sub_85);  mul_306 = sub_85 = None
    mul_310: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_309, 0.7978845608028654);  mul_309 = None
    mul_311: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_310, 0.044715)
    pow_22: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_19, 2.0);  view_19 = None
    mul_312: "f32[1, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_22, 3.0);  pow_22 = None
    mul_313: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_311, mul_312);  mul_311 = mul_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_135: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_310, mul_313);  mul_310 = mul_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_314: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_307, 0.5);  mul_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_136: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(add_135, mul_314);  add_135 = mul_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_108: "f32[512, 3072]" = torch.ops.aten.view.default(add_136, [512, 3072]);  add_136 = None
    permute_96: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    mm_34: "f32[512, 768]" = torch.ops.aten.mm.default(view_108, permute_96);  permute_96 = None
    permute_97: "f32[3072, 512]" = torch.ops.aten.permute.default(view_108, [1, 0])
    mm_35: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_97, view_18);  permute_97 = view_18 = None
    permute_98: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_86: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_108, [0], True);  view_108 = None
    view_109: "f32[3072]" = torch.ops.aten.view.default(sum_86, [3072]);  sum_86 = None
    permute_99: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
    view_110: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_34, [1, 512, 768]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    add_137: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_302, view_110);  mul_302 = view_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    sub_86: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_36, getitem_29);  add_36 = getitem_29 = None
    mul_315: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_9);  sub_86 = None
    mul_316: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_137, primals_40);  primals_40 = None
    mul_317: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_316, 768)
    sum_87: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_316, [2], True)
    mul_318: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_316, mul_315);  mul_316 = None
    sum_88: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_318, [2], True);  mul_318 = None
    mul_319: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_315, sum_88);  sum_88 = None
    sub_87: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_317, sum_87);  mul_317 = sum_87 = None
    sub_88: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_87, mul_319);  sub_87 = mul_319 = None
    div_18: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
    mul_320: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_18, sub_88);  div_18 = sub_88 = None
    mul_321: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_137, mul_315);  mul_315 = None
    sum_89: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_321, [0, 1]);  mul_321 = None
    sum_90: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_137, [0, 1]);  add_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    full_8: "f32[1, 512, 768, 2]" = torch.ops.aten.full.default([1, 512, 768, 2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    select_scatter_7: "f32[1, 512, 768, 2]" = torch.ops.aten.select_scatter.default(full_8, mul_320, 3, 0);  full_8 = None
    view_as_complex_7: "c64[1, 512, 768]" = torch.ops.aten.view_as_complex.default(select_scatter_7);  select_scatter_7 = None
    _fft_c2c_19: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(view_as_complex_7, [1, 2], 0, False);  view_as_complex_7 = None
    view_as_real_19: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_19);  _fft_c2c_19 = None
    select_20: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_19, 3, 0);  view_as_real_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    add_138: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_320, select_20);  mul_320 = select_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_89: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_33, getitem_27);  add_33 = getitem_27 = None
    mul_322: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_89, rsqrt_8);  sub_89 = None
    mul_323: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_138, primals_38);  primals_38 = None
    mul_324: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_323, 768)
    sum_91: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_323, [2], True)
    mul_325: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_323, mul_322);  mul_323 = None
    sum_92: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_325, [2], True);  mul_325 = None
    mul_326: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_322, sum_92);  sum_92 = None
    sub_90: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_324, sum_91);  mul_324 = sum_91 = None
    sub_91: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_90, mul_326);  sub_90 = mul_326 = None
    div_19: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
    mul_327: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_19, sub_91);  div_19 = sub_91 = None
    mul_328: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_138, mul_322);  mul_322 = None
    sum_93: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_328, [0, 1]);  mul_328 = None
    sum_94: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_138, [0, 1]);  add_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_21: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_25, torch.float32);  getitem_25 = None
    mul_329: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 1.1111111111111112);  convert_element_type_21 = None
    mul_330: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_327, mul_329);  mul_329 = None
    clone_8: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_330, memory_format = torch.contiguous_format);  mul_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_111: "f32[512, 768]" = torch.ops.aten.view.default(clone_8, [512, 768]);  clone_8 = None
    permute_100: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    mm_36: "f32[512, 3072]" = torch.ops.aten.mm.default(view_111, permute_100);  permute_100 = None
    permute_101: "f32[768, 512]" = torch.ops.aten.permute.default(view_111, [1, 0])
    mm_37: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_101, view_16);  permute_101 = view_16 = None
    permute_102: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_95: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_111, [0], True);  view_111 = None
    view_112: "f32[768]" = torch.ops.aten.view.default(sum_95, [768]);  sum_95 = None
    permute_103: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_102, [1, 0]);  permute_102 = None
    view_113: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_36, [1, 512, 3072]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_331: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_113, mul_28);  mul_28 = None
    mul_332: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_113, add_32);  view_113 = add_32 = None
    alias_25: "f32[1, 512, 3072]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    mul_333: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(alias_25, alias_25);  alias_25 = None
    sub_92: "f32[1, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_333);  mul_333 = None
    mul_334: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_331, sub_92);  mul_331 = sub_92 = None
    mul_335: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_334, 0.7978845608028654);  mul_334 = None
    mul_336: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_335, 0.044715)
    pow_23: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_15, 2.0);  view_15 = None
    mul_337: "f32[1, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_23, 3.0);  pow_23 = None
    mul_338: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_336, mul_337);  mul_336 = mul_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_139: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_335, mul_338);  mul_335 = mul_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_339: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_332, 0.5);  mul_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_140: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(add_139, mul_339);  add_139 = mul_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_114: "f32[512, 3072]" = torch.ops.aten.view.default(add_140, [512, 3072]);  add_140 = None
    permute_104: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
    mm_38: "f32[512, 768]" = torch.ops.aten.mm.default(view_114, permute_104);  permute_104 = None
    permute_105: "f32[3072, 512]" = torch.ops.aten.permute.default(view_114, [1, 0])
    mm_39: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_105, view_14);  permute_105 = view_14 = None
    permute_106: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_96: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_114, [0], True);  view_114 = None
    view_115: "f32[3072]" = torch.ops.aten.view.default(sum_96, [3072]);  sum_96 = None
    permute_107: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_106, [1, 0]);  permute_106 = None
    view_116: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_38, [1, 512, 768]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    add_141: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_327, view_116);  mul_327 = view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    sub_93: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_28, getitem_23);  add_28 = getitem_23 = None
    mul_340: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_93, rsqrt_7);  sub_93 = None
    mul_341: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_141, primals_32);  primals_32 = None
    mul_342: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_341, 768)
    sum_97: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_341, [2], True)
    mul_343: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_341, mul_340);  mul_341 = None
    sum_98: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_343, [2], True);  mul_343 = None
    mul_344: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_340, sum_98);  sum_98 = None
    sub_94: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_342, sum_97);  mul_342 = sum_97 = None
    sub_95: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_94, mul_344);  sub_94 = mul_344 = None
    div_20: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
    mul_345: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_20, sub_95);  div_20 = sub_95 = None
    mul_346: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_141, mul_340);  mul_340 = None
    sum_99: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_346, [0, 1]);  mul_346 = None
    sum_100: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_141, [0, 1]);  add_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    full_9: "f32[1, 512, 768, 2]" = torch.ops.aten.full.default([1, 512, 768, 2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    select_scatter_8: "f32[1, 512, 768, 2]" = torch.ops.aten.select_scatter.default(full_9, mul_345, 3, 0);  full_9 = None
    view_as_complex_8: "c64[1, 512, 768]" = torch.ops.aten.view_as_complex.default(select_scatter_8);  select_scatter_8 = None
    _fft_c2c_20: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(view_as_complex_8, [1, 2], 0, False);  view_as_complex_8 = None
    view_as_real_20: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_20);  _fft_c2c_20 = None
    select_21: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_20, 3, 0);  view_as_real_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    add_142: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_345, select_21);  mul_345 = select_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_96: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_25, getitem_21);  add_25 = getitem_21 = None
    mul_347: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_96, rsqrt_6);  sub_96 = None
    mul_348: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_142, primals_30);  primals_30 = None
    mul_349: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_348, 768)
    sum_101: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_348, [2], True)
    mul_350: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_348, mul_347);  mul_348 = None
    sum_102: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_350, [2], True);  mul_350 = None
    mul_351: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_347, sum_102);  sum_102 = None
    sub_97: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_349, sum_101);  mul_349 = sum_101 = None
    sub_98: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_97, mul_351);  sub_97 = mul_351 = None
    div_21: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
    mul_352: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_21, sub_98);  div_21 = sub_98 = None
    mul_353: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_142, mul_347);  mul_347 = None
    sum_103: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_353, [0, 1]);  mul_353 = None
    sum_104: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_142, [0, 1]);  add_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_22: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_19, torch.float32);  getitem_19 = None
    mul_354: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 1.1111111111111112);  convert_element_type_22 = None
    mul_355: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_352, mul_354);  mul_354 = None
    clone_9: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_355, memory_format = torch.contiguous_format);  mul_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_117: "f32[512, 768]" = torch.ops.aten.view.default(clone_9, [512, 768]);  clone_9 = None
    permute_108: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
    mm_40: "f32[512, 3072]" = torch.ops.aten.mm.default(view_117, permute_108);  permute_108 = None
    permute_109: "f32[768, 512]" = torch.ops.aten.permute.default(view_117, [1, 0])
    mm_41: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_109, view_12);  permute_109 = view_12 = None
    permute_110: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_105: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_117, [0], True);  view_117 = None
    view_118: "f32[768]" = torch.ops.aten.view.default(sum_105, [768]);  sum_105 = None
    permute_111: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
    view_119: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_40, [1, 512, 3072]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_356: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_119, mul_20);  mul_20 = None
    mul_357: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_119, add_24);  view_119 = add_24 = None
    alias_26: "f32[1, 512, 3072]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    mul_358: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(alias_26, alias_26);  alias_26 = None
    sub_99: "f32[1, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_358);  mul_358 = None
    mul_359: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_356, sub_99);  mul_356 = sub_99 = None
    mul_360: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_359, 0.7978845608028654);  mul_359 = None
    mul_361: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_360, 0.044715)
    pow_24: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_11, 2.0);  view_11 = None
    mul_362: "f32[1, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_24, 3.0);  pow_24 = None
    mul_363: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_361, mul_362);  mul_361 = mul_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_143: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_360, mul_363);  mul_360 = mul_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_364: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_357, 0.5);  mul_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_144: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(add_143, mul_364);  add_143 = mul_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_120: "f32[512, 3072]" = torch.ops.aten.view.default(add_144, [512, 3072]);  add_144 = None
    permute_112: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
    mm_42: "f32[512, 768]" = torch.ops.aten.mm.default(view_120, permute_112);  permute_112 = None
    permute_113: "f32[3072, 512]" = torch.ops.aten.permute.default(view_120, [1, 0])
    mm_43: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_113, view_10);  permute_113 = view_10 = None
    permute_114: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_106: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_120, [0], True);  view_120 = None
    view_121: "f32[3072]" = torch.ops.aten.view.default(sum_106, [3072]);  sum_106 = None
    permute_115: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_114, [1, 0]);  permute_114 = None
    view_122: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_42, [1, 512, 768]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    add_145: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_352, view_122);  mul_352 = view_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    sub_100: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_20, getitem_17);  add_20 = getitem_17 = None
    mul_365: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_100, rsqrt_5);  sub_100 = None
    mul_366: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_145, primals_24);  primals_24 = None
    mul_367: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_366, 768)
    sum_107: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_366, [2], True)
    mul_368: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_366, mul_365);  mul_366 = None
    sum_108: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_368, [2], True);  mul_368 = None
    mul_369: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_365, sum_108);  sum_108 = None
    sub_101: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_367, sum_107);  mul_367 = sum_107 = None
    sub_102: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_101, mul_369);  sub_101 = mul_369 = None
    div_22: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
    mul_370: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_22, sub_102);  div_22 = sub_102 = None
    mul_371: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_145, mul_365);  mul_365 = None
    sum_109: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_371, [0, 1]);  mul_371 = None
    sum_110: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_145, [0, 1]);  add_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    full_10: "f32[1, 512, 768, 2]" = torch.ops.aten.full.default([1, 512, 768, 2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    select_scatter_9: "f32[1, 512, 768, 2]" = torch.ops.aten.select_scatter.default(full_10, mul_370, 3, 0);  full_10 = None
    view_as_complex_9: "c64[1, 512, 768]" = torch.ops.aten.view_as_complex.default(select_scatter_9);  select_scatter_9 = None
    _fft_c2c_21: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(view_as_complex_9, [1, 2], 0, False);  view_as_complex_9 = None
    view_as_real_21: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_21);  _fft_c2c_21 = None
    select_22: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_21, 3, 0);  view_as_real_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    add_146: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_370, select_22);  mul_370 = select_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_103: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_17, getitem_15);  add_17 = getitem_15 = None
    mul_372: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_103, rsqrt_4);  sub_103 = None
    mul_373: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_146, primals_22);  primals_22 = None
    mul_374: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_373, 768)
    sum_111: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_373, [2], True)
    mul_375: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_373, mul_372);  mul_373 = None
    sum_112: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_375, [2], True);  mul_375 = None
    mul_376: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_372, sum_112);  sum_112 = None
    sub_104: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_374, sum_111);  mul_374 = sum_111 = None
    sub_105: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_104, mul_376);  sub_104 = mul_376 = None
    div_23: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
    mul_377: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_23, sub_105);  div_23 = sub_105 = None
    mul_378: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_146, mul_372);  mul_372 = None
    sum_113: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_378, [0, 1]);  mul_378 = None
    sum_114: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_146, [0, 1]);  add_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_23: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_13, torch.float32);  getitem_13 = None
    mul_379: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_23, 1.1111111111111112);  convert_element_type_23 = None
    mul_380: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_377, mul_379);  mul_379 = None
    clone_10: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_380, memory_format = torch.contiguous_format);  mul_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_123: "f32[512, 768]" = torch.ops.aten.view.default(clone_10, [512, 768]);  clone_10 = None
    permute_116: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
    mm_44: "f32[512, 3072]" = torch.ops.aten.mm.default(view_123, permute_116);  permute_116 = None
    permute_117: "f32[768, 512]" = torch.ops.aten.permute.default(view_123, [1, 0])
    mm_45: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_117, view_8);  permute_117 = view_8 = None
    permute_118: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_115: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_123, [0], True);  view_123 = None
    view_124: "f32[768]" = torch.ops.aten.view.default(sum_115, [768]);  sum_115 = None
    permute_119: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
    view_125: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_44, [1, 512, 3072]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_381: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_125, mul_12);  mul_12 = None
    mul_382: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_125, add_16);  view_125 = add_16 = None
    alias_27: "f32[1, 512, 3072]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    mul_383: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(alias_27, alias_27);  alias_27 = None
    sub_106: "f32[1, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_383);  mul_383 = None
    mul_384: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_381, sub_106);  mul_381 = sub_106 = None
    mul_385: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_384, 0.7978845608028654);  mul_384 = None
    mul_386: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_385, 0.044715)
    pow_25: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_7, 2.0);  view_7 = None
    mul_387: "f32[1, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_25, 3.0);  pow_25 = None
    mul_388: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_386, mul_387);  mul_386 = mul_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_147: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_385, mul_388);  mul_385 = mul_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_389: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_382, 0.5);  mul_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_148: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(add_147, mul_389);  add_147 = mul_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_126: "f32[512, 3072]" = torch.ops.aten.view.default(add_148, [512, 3072]);  add_148 = None
    permute_120: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    mm_46: "f32[512, 768]" = torch.ops.aten.mm.default(view_126, permute_120);  permute_120 = None
    permute_121: "f32[3072, 512]" = torch.ops.aten.permute.default(view_126, [1, 0])
    mm_47: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_121, view_6);  permute_121 = view_6 = None
    permute_122: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_116: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_126, [0], True);  view_126 = None
    view_127: "f32[3072]" = torch.ops.aten.view.default(sum_116, [3072]);  sum_116 = None
    permute_123: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
    view_128: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_46, [1, 512, 768]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    add_149: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_377, view_128);  mul_377 = view_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    sub_107: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_12, getitem_11);  add_12 = getitem_11 = None
    mul_390: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_107, rsqrt_3);  sub_107 = None
    mul_391: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_149, primals_16);  primals_16 = None
    mul_392: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_391, 768)
    sum_117: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_391, [2], True)
    mul_393: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_391, mul_390);  mul_391 = None
    sum_118: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_393, [2], True);  mul_393 = None
    mul_394: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_390, sum_118);  sum_118 = None
    sub_108: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_392, sum_117);  mul_392 = sum_117 = None
    sub_109: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_108, mul_394);  sub_108 = mul_394 = None
    div_24: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
    mul_395: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_24, sub_109);  div_24 = sub_109 = None
    mul_396: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_149, mul_390);  mul_390 = None
    sum_119: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_396, [0, 1]);  mul_396 = None
    sum_120: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_149, [0, 1]);  add_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    full_11: "f32[1, 512, 768, 2]" = torch.ops.aten.full.default([1, 512, 768, 2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    select_scatter_10: "f32[1, 512, 768, 2]" = torch.ops.aten.select_scatter.default(full_11, mul_395, 3, 0);  full_11 = None
    view_as_complex_10: "c64[1, 512, 768]" = torch.ops.aten.view_as_complex.default(select_scatter_10);  select_scatter_10 = None
    _fft_c2c_22: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(view_as_complex_10, [1, 2], 0, False);  view_as_complex_10 = None
    view_as_real_22: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_22);  _fft_c2c_22 = None
    select_23: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_22, 3, 0);  view_as_real_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    add_150: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_395, select_23);  mul_395 = select_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:249, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_110: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_9, getitem_9);  add_9 = getitem_9 = None
    mul_397: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_110, rsqrt_2);  sub_110 = None
    mul_398: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_150, primals_14);  primals_14 = None
    mul_399: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_398, 768)
    sum_121: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_398, [2], True)
    mul_400: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_398, mul_397);  mul_398 = None
    sum_122: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_400, [2], True);  mul_400 = None
    mul_401: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_397, sum_122);  sum_122 = None
    sub_111: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_399, sum_121);  mul_399 = sum_121 = None
    sub_112: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_111, mul_401);  sub_111 = mul_401 = None
    div_25: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
    mul_402: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_25, sub_112);  div_25 = sub_112 = None
    mul_403: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_150, mul_397);  mul_397 = None
    sum_123: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_403, [0, 1]);  mul_403 = None
    sum_124: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_150, [0, 1]);  add_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:248, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_24: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_7, torch.float32);  getitem_7 = None
    mul_404: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 1.1111111111111112);  convert_element_type_24 = None
    mul_405: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_402, mul_404);  mul_404 = None
    clone_11: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_405, memory_format = torch.contiguous_format);  mul_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:247, code: hidden_states = self.dense(hidden_states)
    view_129: "f32[512, 768]" = torch.ops.aten.view.default(clone_11, [512, 768]);  clone_11 = None
    permute_124: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    mm_48: "f32[512, 3072]" = torch.ops.aten.mm.default(view_129, permute_124);  permute_124 = None
    permute_125: "f32[768, 512]" = torch.ops.aten.permute.default(view_129, [1, 0])
    mm_49: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_125, view_4);  permute_125 = view_4 = None
    permute_126: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_125: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_129, [0], True);  view_129 = None
    view_130: "f32[768]" = torch.ops.aten.view.default(sum_125, [768]);  sum_125 = None
    permute_127: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_126, [1, 0]);  permute_126 = None
    view_131: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_48, [1, 512, 3072]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_406: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_131, mul_4);  mul_4 = None
    mul_407: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_131, add_8);  view_131 = add_8 = None
    alias_28: "f32[1, 512, 3072]" = torch.ops.aten.alias.default(alias);  alias = None
    mul_408: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(alias_28, alias_28);  alias_28 = None
    sub_113: "f32[1, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_408);  mul_408 = None
    mul_409: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_406, sub_113);  mul_406 = sub_113 = None
    mul_410: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_409, 0.7978845608028654);  mul_409 = None
    mul_411: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_410, 0.044715)
    pow_26: "f32[1, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_3, 2.0);  view_3 = None
    mul_412: "f32[1, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_26, 3.0);  pow_26 = None
    mul_413: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_411, mul_412);  mul_411 = mul_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_151: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_410, mul_413);  mul_410 = mul_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_414: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_407, 0.5);  mul_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_152: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(add_151, mul_414);  add_151 = mul_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    view_132: "f32[512, 3072]" = torch.ops.aten.view.default(add_152, [512, 3072]);  add_152 = None
    permute_128: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    mm_50: "f32[512, 768]" = torch.ops.aten.mm.default(view_132, permute_128);  permute_128 = None
    permute_129: "f32[3072, 512]" = torch.ops.aten.permute.default(view_132, [1, 0])
    mm_51: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_129, view_2);  permute_129 = view_2 = None
    permute_130: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_126: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_132, [0], True);  view_132 = None
    view_133: "f32[3072]" = torch.ops.aten.view.default(sum_126, [3072]);  sum_126 = None
    permute_131: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
    view_134: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_50, [1, 512, 768]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:233, code: hidden_states = self.dense(hidden_states)
    add_153: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_402, view_134);  mul_402 = view_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:205, code: hidden_states = self.LayerNorm(input_tensor + hidden_states)
    sub_114: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_4, getitem_5);  add_4 = getitem_5 = None
    mul_415: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_114, rsqrt_1);  sub_114 = None
    mul_416: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_153, primals_8);  primals_8 = None
    mul_417: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_416, 768)
    sum_127: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_416, [2], True)
    mul_418: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_416, mul_415);  mul_416 = None
    sum_128: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_418, [2], True);  mul_418 = None
    mul_419: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_415, sum_128);  sum_128 = None
    sub_115: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_417, sum_127);  mul_417 = sum_127 = None
    sub_116: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_115, mul_419);  sub_115 = mul_419 = None
    div_26: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
    mul_420: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_26, sub_116);  div_26 = sub_116 = None
    mul_421: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_153, mul_415);  mul_415 = None
    sum_129: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_421, [0, 1]);  mul_421 = None
    sum_130: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_153, [0, 1]);  add_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    full_12: "f32[1, 512, 768, 2]" = torch.ops.aten.full.default([1, 512, 768, 2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    select_scatter_11: "f32[1, 512, 768, 2]" = torch.ops.aten.select_scatter.default(full_12, mul_420, 3, 0);  full_12 = None
    view_as_complex_11: "c64[1, 512, 768]" = torch.ops.aten.view_as_complex.default(select_scatter_11);  select_scatter_11 = None
    _fft_c2c_23: "c64[1, 512, 768]" = torch.ops.aten._fft_c2c.default(view_as_complex_11, [1, 2], 0, False);  view_as_complex_11 = None
    view_as_real_23: "f32[1, 512, 768, 2]" = torch.ops.aten.view_as_real.default(_fft_c2c_23);  _fft_c2c_23 = None
    select_24: "f32[1, 512, 768]" = torch.ops.aten.select.int(view_as_real_23, 3, 0);  view_as_real_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:195, code: outputs = self.fourier_transform(hidden_states).real
    add_154: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_420, select_24);  mul_420 = select_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:157, code: embeddings = self.dropout(embeddings)
    convert_element_type_25: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_3, torch.float32);  getitem_3 = None
    mul_422: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.1111111111111112);  convert_element_type_25 = None
    mul_423: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_154, mul_422);  add_154 = mul_422 = None
    clone_12: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_423, memory_format = torch.contiguous_format);  mul_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:156, code: embeddings = self.projection(embeddings)
    view_135: "f32[512, 768]" = torch.ops.aten.view.default(clone_12, [512, 768]);  clone_12 = None
    permute_132: "f32[768, 768]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    mm_52: "f32[512, 768]" = torch.ops.aten.mm.default(view_135, permute_132);  permute_132 = None
    permute_133: "f32[768, 512]" = torch.ops.aten.permute.default(view_135, [1, 0])
    mm_53: "f32[768, 768]" = torch.ops.aten.mm.default(permute_133, view);  permute_133 = view = None
    permute_134: "f32[768, 768]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_131: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_135, [0], True);  view_135 = None
    view_136: "f32[768]" = torch.ops.aten.view.default(sum_131, [768]);  sum_131 = None
    permute_135: "f32[768, 768]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
    view_137: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_52, [1, 512, 768]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:155, code: embeddings = self.LayerNorm(embeddings)
    sub_117: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
    mul_424: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_117, rsqrt);  sub_117 = None
    mul_425: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_137, primals_4);  primals_4 = None
    mul_426: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_425, 768)
    sum_132: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_425, [2], True)
    mul_427: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_425, mul_424);  mul_425 = None
    sum_133: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_427, [2], True);  mul_427 = None
    mul_428: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_424, sum_133);  sum_133 = None
    sub_118: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_426, sum_132);  mul_426 = sum_132 = None
    sub_119: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_118, mul_428);  sub_118 = mul_428 = None
    div_27: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
    mul_429: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_27, sub_119);  div_27 = sub_119 = None
    mul_430: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_137, mul_424);  mul_424 = None
    sum_134: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_430, [0, 1]);  mul_430 = None
    sum_135: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_137, [0, 1]);  view_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:153, code: position_embeddings = self.position_embeddings(position_ids)
    eq: "b8[1, 512]" = torch.ops.aten.eq.Scalar(slice_2, -1)
    unsqueeze_2: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
    scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_4: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_2, scalar_tensor_4, mul_429);  unsqueeze_2 = scalar_tensor_4 = None
    full_13: "f32[512, 768]" = torch.ops.aten.full.default([512, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put: "f32[512, 768]" = torch.ops.aten._unsafe_index_put.default(full_13, [slice_2], where_4, True);  full_13 = slice_2 = where_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:149, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    eq_1: "b8[1, 512]" = torch.ops.aten.eq.Scalar(expand, -1)
    unsqueeze_3: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_5: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_3, scalar_tensor_5, mul_429);  unsqueeze_3 = scalar_tensor_5 = None
    full_14: "f32[4, 768]" = torch.ops.aten.full.default([4, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_1: "f32[4, 768]" = torch.ops.aten._unsafe_index_put.default(full_14, [expand], where_5, True);  full_14 = expand = where_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/fnet/modeling_fnet.py:148, code: inputs_embeds = self.word_embeddings(input_ids)
    eq_2: "b8[1, 512]" = torch.ops.aten.eq.Scalar(primals_114, 3)
    unsqueeze_4: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_2, -1);  eq_2 = None
    scalar_tensor_6: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_6: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_4, scalar_tensor_6, mul_429);  unsqueeze_4 = scalar_tensor_6 = mul_429 = None
    full_15: "f32[32000, 768]" = torch.ops.aten.full.default([32000, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_2: "f32[32000, 768]" = torch.ops.aten._unsafe_index_put.default(full_15, [primals_114], where_6, True);  full_15 = primals_114 = where_6 = None
    return pytree.tree_unflatten([div, view_53, _unsafe_index_put_2, _unsafe_index_put_1, _unsafe_index_put, sum_134, sum_135, permute_135, view_136, sum_129, sum_130, permute_131, view_133, permute_127, view_130, sum_123, sum_124, sum_119, sum_120, permute_123, view_127, permute_119, view_124, sum_113, sum_114, sum_109, sum_110, permute_115, view_121, permute_111, view_118, sum_103, sum_104, sum_99, sum_100, permute_107, view_115, permute_103, view_112, sum_93, sum_94, sum_89, sum_90, permute_99, view_109, permute_95, view_106, sum_83, sum_84, sum_79, sum_80, permute_91, view_103, permute_87, view_100, sum_73, sum_74, sum_69, sum_70, permute_83, view_97, permute_79, view_94, sum_63, sum_64, sum_59, sum_60, permute_75, view_91, permute_71, view_88, sum_53, sum_54, sum_49, sum_50, permute_67, view_85, permute_63, view_82, sum_43, sum_44, sum_39, sum_40, permute_59, view_79, permute_55, view_76, sum_33, sum_34, sum_29, sum_30, permute_51, view_73, permute_47, view_70, sum_23, sum_24, sum_19, sum_20, permute_43, view_67, permute_39, view_64, sum_13, sum_14, None, None, permute_35, view_61, sum_8, sum_9, permute_31, view_58, None, None, None, None], self._out_spec)
    