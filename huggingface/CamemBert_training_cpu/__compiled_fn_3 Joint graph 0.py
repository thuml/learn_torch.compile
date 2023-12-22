from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[32005, 768]"; primals_2: "f32[1, 768]"; primals_3: "f32[514, 768]"; primals_4: "f32[768]"; primals_5: "f32[768]"; primals_6: "f32[768, 768]"; primals_7: "f32[768]"; primals_8: "f32[768, 768]"; primals_9: "f32[768]"; primals_10: "f32[768, 768]"; primals_11: "f32[768]"; primals_12: "f32[768, 768]"; primals_13: "f32[768]"; primals_14: "f32[768]"; primals_15: "f32[768]"; primals_16: "f32[3072, 768]"; primals_17: "f32[3072]"; primals_18: "f32[768, 3072]"; primals_19: "f32[768]"; primals_20: "f32[768]"; primals_21: "f32[768]"; primals_22: "f32[768, 768]"; primals_23: "f32[768]"; primals_24: "f32[768, 768]"; primals_25: "f32[768]"; primals_26: "f32[768, 768]"; primals_27: "f32[768]"; primals_28: "f32[768, 768]"; primals_29: "f32[768]"; primals_30: "f32[768]"; primals_31: "f32[768]"; primals_32: "f32[3072, 768]"; primals_33: "f32[3072]"; primals_34: "f32[768, 3072]"; primals_35: "f32[768]"; primals_36: "f32[768]"; primals_37: "f32[768]"; primals_38: "f32[768, 768]"; primals_39: "f32[768]"; primals_40: "f32[768, 768]"; primals_41: "f32[768]"; primals_42: "f32[768, 768]"; primals_43: "f32[768]"; primals_44: "f32[768, 768]"; primals_45: "f32[768]"; primals_46: "f32[768]"; primals_47: "f32[768]"; primals_48: "f32[3072, 768]"; primals_49: "f32[3072]"; primals_50: "f32[768, 3072]"; primals_51: "f32[768]"; primals_52: "f32[768]"; primals_53: "f32[768]"; primals_54: "f32[768, 768]"; primals_55: "f32[768]"; primals_56: "f32[768, 768]"; primals_57: "f32[768]"; primals_58: "f32[768, 768]"; primals_59: "f32[768]"; primals_60: "f32[768, 768]"; primals_61: "f32[768]"; primals_62: "f32[768]"; primals_63: "f32[768]"; primals_64: "f32[3072, 768]"; primals_65: "f32[3072]"; primals_66: "f32[768, 3072]"; primals_67: "f32[768]"; primals_68: "f32[768]"; primals_69: "f32[768]"; primals_70: "f32[768, 768]"; primals_71: "f32[768]"; primals_72: "f32[768, 768]"; primals_73: "f32[768]"; primals_74: "f32[768, 768]"; primals_75: "f32[768]"; primals_76: "f32[768, 768]"; primals_77: "f32[768]"; primals_78: "f32[768]"; primals_79: "f32[768]"; primals_80: "f32[3072, 768]"; primals_81: "f32[3072]"; primals_82: "f32[768, 3072]"; primals_83: "f32[768]"; primals_84: "f32[768]"; primals_85: "f32[768]"; primals_86: "f32[768, 768]"; primals_87: "f32[768]"; primals_88: "f32[768, 768]"; primals_89: "f32[768]"; primals_90: "f32[768, 768]"; primals_91: "f32[768]"; primals_92: "f32[768, 768]"; primals_93: "f32[768]"; primals_94: "f32[768]"; primals_95: "f32[768]"; primals_96: "f32[3072, 768]"; primals_97: "f32[3072]"; primals_98: "f32[768, 3072]"; primals_99: "f32[768]"; primals_100: "f32[768]"; primals_101: "f32[768]"; primals_102: "f32[768, 768]"; primals_103: "f32[768]"; primals_104: "f32[768, 768]"; primals_105: "f32[768]"; primals_106: "f32[768, 768]"; primals_107: "f32[768]"; primals_108: "f32[768, 768]"; primals_109: "f32[768]"; primals_110: "f32[768]"; primals_111: "f32[768]"; primals_112: "f32[3072, 768]"; primals_113: "f32[3072]"; primals_114: "f32[768, 3072]"; primals_115: "f32[768]"; primals_116: "f32[768]"; primals_117: "f32[768]"; primals_118: "f32[768, 768]"; primals_119: "f32[768]"; primals_120: "f32[768, 768]"; primals_121: "f32[768]"; primals_122: "f32[768, 768]"; primals_123: "f32[768]"; primals_124: "f32[768, 768]"; primals_125: "f32[768]"; primals_126: "f32[768]"; primals_127: "f32[768]"; primals_128: "f32[3072, 768]"; primals_129: "f32[3072]"; primals_130: "f32[768, 3072]"; primals_131: "f32[768]"; primals_132: "f32[768]"; primals_133: "f32[768]"; primals_134: "f32[768, 768]"; primals_135: "f32[768]"; primals_136: "f32[768, 768]"; primals_137: "f32[768]"; primals_138: "f32[768, 768]"; primals_139: "f32[768]"; primals_140: "f32[768, 768]"; primals_141: "f32[768]"; primals_142: "f32[768]"; primals_143: "f32[768]"; primals_144: "f32[3072, 768]"; primals_145: "f32[3072]"; primals_146: "f32[768, 3072]"; primals_147: "f32[768]"; primals_148: "f32[768]"; primals_149: "f32[768]"; primals_150: "f32[768, 768]"; primals_151: "f32[768]"; primals_152: "f32[768, 768]"; primals_153: "f32[768]"; primals_154: "f32[768, 768]"; primals_155: "f32[768]"; primals_156: "f32[768, 768]"; primals_157: "f32[768]"; primals_158: "f32[768]"; primals_159: "f32[768]"; primals_160: "f32[3072, 768]"; primals_161: "f32[3072]"; primals_162: "f32[768, 3072]"; primals_163: "f32[768]"; primals_164: "f32[768]"; primals_165: "f32[768]"; primals_166: "f32[768, 768]"; primals_167: "f32[768]"; primals_168: "f32[768, 768]"; primals_169: "f32[768]"; primals_170: "f32[768, 768]"; primals_171: "f32[768]"; primals_172: "f32[768, 768]"; primals_173: "f32[768]"; primals_174: "f32[768]"; primals_175: "f32[768]"; primals_176: "f32[3072, 768]"; primals_177: "f32[3072]"; primals_178: "f32[768, 3072]"; primals_179: "f32[768]"; primals_180: "f32[768]"; primals_181: "f32[768]"; primals_182: "f32[768, 768]"; primals_183: "f32[768]"; primals_184: "f32[768, 768]"; primals_185: "f32[768]"; primals_186: "f32[768, 768]"; primals_187: "f32[768]"; primals_188: "f32[768, 768]"; primals_189: "f32[768]"; primals_190: "f32[768]"; primals_191: "f32[768]"; primals_192: "f32[3072, 768]"; primals_193: "f32[3072]"; primals_194: "f32[768, 3072]"; primals_195: "f32[768]"; primals_196: "f32[768]"; primals_197: "f32[768]"; primals_198: "f32[768, 768]"; primals_199: "f32[768]"; primals_200: "f32[768]"; primals_201: "f32[768]"; primals_202: "f32[32005, 768]"; primals_203: "f32[32005]"; primals_204: "i64[1, 514]"; primals_205: "i64[1, 512]"; primals_206: "i64[1, 512]"; tangents_1: "f32[]"; tangents_2: "f32[1, 512, 32005]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, tangents_1, tangents_2, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:858, code: attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
    full: "f32[1, 512]" = torch.ops.aten.full.default([1, 512], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:862, code: buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
    slice_1: "i64[1, 514]" = torch.ops.aten.slice.Tensor(primals_204, 0, 0, 9223372036854775807);  primals_204 = None
    slice_2: "i64[1, 512]" = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 512);  slice_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:863, code: buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
    expand: "i64[1, 512]" = torch.ops.aten.expand.default(slice_2, [1, 512]);  slice_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:916, code: extended_attention_mask = attention_mask[:, None, None, :]
    slice_3: "f32[1, 512]" = torch.ops.aten.slice.Tensor(full, 0, 0, 9223372036854775807);  full = None
    unsqueeze: "f32[1, 1, 512]" = torch.ops.aten.unsqueeze.default(slice_3, 1);  slice_3 = None
    unsqueeze_1: "f32[1, 1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
    slice_4: "f32[1, 1, 1, 512]" = torch.ops.aten.slice.Tensor(unsqueeze_1, 3, 0, 9223372036854775807);  unsqueeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:928, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    sub: "f32[1, 1, 1, 512]" = torch.ops.aten.sub.Tensor(1.0, slice_4);  slice_4 = None
    mul: "f32[1, 1, 1, 512]" = torch.ops.aten.mul.Tensor(sub, -3.4028234663852886e+38);  sub = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:1570, code: mask = input_ids.ne(padding_idx).int()
    ne: "b8[1, 512]" = torch.ops.aten.ne.Scalar(primals_205, 1)
    convert_element_type: "i32[1, 512]" = torch.ops.prims.convert_element_type.default(ne, torch.int32);  ne = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:1571, code: incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    cumsum: "i64[1, 512]" = torch.ops.aten.cumsum.default(convert_element_type, 1)
    convert_element_type_1: "i32[1, 512]" = torch.ops.prims.convert_element_type.default(cumsum, torch.int32);  cumsum = None
    add: "i32[1, 512]" = torch.ops.aten.add.Tensor(convert_element_type_1, 0);  convert_element_type_1 = None
    mul_1: "i32[1, 512]" = torch.ops.aten.mul.Tensor(add, convert_element_type);  add = convert_element_type = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:1572, code: return incremental_indices.long() + padding_idx
    convert_element_type_2: "i64[1, 512]" = torch.ops.prims.convert_element_type.default(mul_1, torch.int64);  mul_1 = None
    add_1: "i64[1, 512]" = torch.ops.aten.add.Tensor(convert_element_type_2, 1);  convert_element_type_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:139, code: inputs_embeds = self.word_embeddings(input_ids)
    embedding: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(primals_1, primals_205, 1);  primals_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:140, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    embedding_1: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(primals_2, expand);  primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:142, code: embeddings = inputs_embeds + token_type_embeddings
    add_2: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:144, code: position_embeddings = self.position_embeddings(position_ids)
    embedding_2: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(primals_3, add_1, 1);  primals_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:145, code: embeddings += position_embeddings
    add_3: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_2, embedding_2);  add_2 = embedding_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:146, code: embeddings = self.LayerNorm(embeddings)
    var_mean = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 512, 1]" = var_mean[0]
    getitem_1: "f32[1, 512, 1]" = var_mean[1];  var_mean = None
    add_4: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    sub_1: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_3, getitem_1)
    mul_2: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = None
    mul_3: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_2, primals_4);  mul_2 = None
    add_5: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_3, primals_5);  mul_3 = primals_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:147, code: embeddings = self.dropout(embeddings)
    native_dropout = torch.ops.aten.native_dropout.default(add_5, 0.1, True);  add_5 = None
    getitem_2: "f32[1, 512, 768]" = native_dropout[0]
    getitem_3: "b8[1, 512, 768]" = native_dropout[1];  native_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:211, code: mixed_query_layer = self.query(hidden_states)
    view: "f32[512, 768]" = torch.ops.aten.view.default(getitem_2, [512, 768])
    permute: "f32[768, 768]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
    addmm: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_7, view, permute);  primals_7 = None
    view_1: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm, [1, 512, 768]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:233, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_2: "f32[512, 768]" = torch.ops.aten.view.default(getitem_2, [512, 768])
    permute_1: "f32[768, 768]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
    addmm_1: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_9, view_2, permute_1);  primals_9 = None
    view_3: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_1, [1, 512, 768]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_4: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_3, [1, 512, 12, 64]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_2: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:234, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_5: "f32[512, 768]" = torch.ops.aten.view.default(getitem_2, [512, 768])
    permute_3: "f32[768, 768]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
    addmm_2: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_11, view_5, permute_3);  primals_11 = None
    view_6: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_2, [1, 512, 768]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_7: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_6, [1, 512, 12, 64]);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_4: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_8: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_1, [1, 512, 12, 64]);  view_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_5: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:250, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_6: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_2, [0, 1, 3, 2]);  permute_2 = None
    expand_1: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_5, [1, 12, 512, 64]);  permute_5 = None
    view_9: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_1, [12, 512, 64]);  expand_1 = None
    expand_2: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_6, [1, 12, 64, 512]);  permute_6 = None
    view_10: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_2, [12, 64, 512]);  expand_2 = None
    bmm: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_9, view_10)
    view_11: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm, [1, 12, 512, 512]);  bmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:274, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_11, 8.0);  view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:277, code: attention_scores = attention_scores + attention_mask
    add_6: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(div, mul);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:280, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_6, [-1], True)
    sub_2: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_6, amax);  add_6 = amax = None
    exp: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
    sum_1: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div_1: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(div_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:284, code: attention_probs = self.dropout(attention_probs)
    native_dropout_1 = torch.ops.aten.native_dropout.default(div_1, 0.1, True);  div_1 = None
    getitem_4: "f32[1, 12, 512, 512]" = native_dropout_1[0]
    getitem_5: "b8[1, 12, 512, 512]" = native_dropout_1[1];  native_dropout_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:290, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_3: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(getitem_4, [1, 12, 512, 512]);  getitem_4 = None
    view_12: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_3, [12, 512, 512]);  expand_3 = None
    expand_4: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_4, [1, 12, 512, 64]);  permute_4 = None
    view_13: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_4, [12, 512, 64]);  expand_4 = None
    bmm_1: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_12, view_13)
    view_14: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_1, [1, 12, 512, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:292, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_7: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
    clone: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:294, code: context_layer = context_layer.view(new_context_layer_shape)
    view_15: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone, [1, 512, 768]);  clone = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:312, code: hidden_states = self.dense(hidden_states)
    view_16: "f32[512, 768]" = torch.ops.aten.view.default(view_15, [512, 768]);  view_15 = None
    permute_8: "f32[768, 768]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
    addmm_3: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_13, view_16, permute_8);  primals_13 = None
    view_17: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_3, [1, 512, 768]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:313, code: hidden_states = self.dropout(hidden_states)
    native_dropout_2 = torch.ops.aten.native_dropout.default(view_17, 0.1, True);  view_17 = None
    getitem_6: "f32[1, 512, 768]" = native_dropout_2[0]
    getitem_7: "b8[1, 512, 768]" = native_dropout_2[1];  native_dropout_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:314, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_7: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_6, getitem_2);  getitem_6 = getitem_2 = None
    var_mean_1 = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
    getitem_8: "f32[1, 512, 1]" = var_mean_1[0]
    getitem_9: "f32[1, 512, 1]" = var_mean_1[1];  var_mean_1 = None
    add_8: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
    rsqrt_1: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
    sub_3: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_7, getitem_9)
    mul_4: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = None
    mul_5: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_4, primals_14);  mul_4 = None
    add_9: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_5, primals_15);  mul_5 = primals_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:379, code: hidden_states = self.dense(hidden_states)
    view_18: "f32[512, 768]" = torch.ops.aten.view.default(add_9, [512, 768])
    permute_9: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
    addmm_4: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_17, view_18, permute_9);  primals_17 = None
    view_19: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_4, [1, 512, 3072]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_6: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.5)
    mul_7: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476)
    erf: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_7);  mul_7 = None
    add_10: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_8: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_6, add_10);  mul_6 = add_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:393, code: hidden_states = self.dense(hidden_states)
    view_20: "f32[512, 3072]" = torch.ops.aten.view.default(mul_8, [512, 3072]);  mul_8 = None
    permute_10: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
    addmm_5: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_19, view_20, permute_10);  primals_19 = None
    view_21: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_5, [1, 512, 768]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:394, code: hidden_states = self.dropout(hidden_states)
    native_dropout_3 = torch.ops.aten.native_dropout.default(view_21, 0.1, True);  view_21 = None
    getitem_10: "f32[1, 512, 768]" = native_dropout_3[0]
    getitem_11: "b8[1, 512, 768]" = native_dropout_3[1];  native_dropout_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:395, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_11: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_10, add_9);  getitem_10 = add_9 = None
    var_mean_2 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
    getitem_12: "f32[1, 512, 1]" = var_mean_2[0]
    getitem_13: "f32[1, 512, 1]" = var_mean_2[1];  var_mean_2 = None
    add_12: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
    rsqrt_2: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
    sub_4: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_11, getitem_13)
    mul_9: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = None
    mul_10: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_9, primals_20);  mul_9 = None
    add_13: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_10, primals_21);  mul_10 = primals_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:211, code: mixed_query_layer = self.query(hidden_states)
    view_22: "f32[512, 768]" = torch.ops.aten.view.default(add_13, [512, 768])
    permute_11: "f32[768, 768]" = torch.ops.aten.permute.default(primals_22, [1, 0]);  primals_22 = None
    addmm_6: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_23, view_22, permute_11);  primals_23 = None
    view_23: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_6, [1, 512, 768]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:233, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_24: "f32[512, 768]" = torch.ops.aten.view.default(add_13, [512, 768])
    permute_12: "f32[768, 768]" = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
    addmm_7: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_25, view_24, permute_12);  primals_25 = None
    view_25: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_7, [1, 512, 768]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_26: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_25, [1, 512, 12, 64]);  view_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_13: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:234, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_27: "f32[512, 768]" = torch.ops.aten.view.default(add_13, [512, 768])
    permute_14: "f32[768, 768]" = torch.ops.aten.permute.default(primals_26, [1, 0]);  primals_26 = None
    addmm_8: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_27, view_27, permute_14);  primals_27 = None
    view_28: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_8, [1, 512, 768]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_29: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_28, [1, 512, 12, 64]);  view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_15: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_30: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_23, [1, 512, 12, 64]);  view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_16: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:250, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_17: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_13, [0, 1, 3, 2]);  permute_13 = None
    expand_5: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_16, [1, 12, 512, 64]);  permute_16 = None
    view_31: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_5, [12, 512, 64]);  expand_5 = None
    expand_6: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_17, [1, 12, 64, 512]);  permute_17 = None
    view_32: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_6, [12, 64, 512]);  expand_6 = None
    bmm_2: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_31, view_32)
    view_33: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_2, [1, 12, 512, 512]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:274, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_2: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_33, 8.0);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:277, code: attention_scores = attention_scores + attention_mask
    add_14: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(div_2, mul);  div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:280, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_1: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_14, [-1], True)
    sub_5: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_14, amax_1);  add_14 = amax_1 = None
    exp_1: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
    sum_2: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_3: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    alias_1: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(div_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:284, code: attention_probs = self.dropout(attention_probs)
    native_dropout_4 = torch.ops.aten.native_dropout.default(div_3, 0.1, True);  div_3 = None
    getitem_14: "f32[1, 12, 512, 512]" = native_dropout_4[0]
    getitem_15: "b8[1, 12, 512, 512]" = native_dropout_4[1];  native_dropout_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:290, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_7: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(getitem_14, [1, 12, 512, 512]);  getitem_14 = None
    view_34: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_7, [12, 512, 512]);  expand_7 = None
    expand_8: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_15, [1, 12, 512, 64]);  permute_15 = None
    view_35: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_8, [12, 512, 64]);  expand_8 = None
    bmm_3: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_34, view_35)
    view_36: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_3, [1, 12, 512, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:292, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_18: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None
    clone_1: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:294, code: context_layer = context_layer.view(new_context_layer_shape)
    view_37: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_1, [1, 512, 768]);  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:312, code: hidden_states = self.dense(hidden_states)
    view_38: "f32[512, 768]" = torch.ops.aten.view.default(view_37, [512, 768]);  view_37 = None
    permute_19: "f32[768, 768]" = torch.ops.aten.permute.default(primals_28, [1, 0]);  primals_28 = None
    addmm_9: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_29, view_38, permute_19);  primals_29 = None
    view_39: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_9, [1, 512, 768]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:313, code: hidden_states = self.dropout(hidden_states)
    native_dropout_5 = torch.ops.aten.native_dropout.default(view_39, 0.1, True);  view_39 = None
    getitem_16: "f32[1, 512, 768]" = native_dropout_5[0]
    getitem_17: "b8[1, 512, 768]" = native_dropout_5[1];  native_dropout_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:314, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_15: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_16, add_13);  getitem_16 = add_13 = None
    var_mean_3 = torch.ops.aten.var_mean.correction(add_15, [2], correction = 0, keepdim = True)
    getitem_18: "f32[1, 512, 1]" = var_mean_3[0]
    getitem_19: "f32[1, 512, 1]" = var_mean_3[1];  var_mean_3 = None
    add_16: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
    rsqrt_3: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    sub_6: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_15, getitem_19)
    mul_11: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_3);  sub_6 = None
    mul_12: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_11, primals_30);  mul_11 = None
    add_17: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_12, primals_31);  mul_12 = primals_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:379, code: hidden_states = self.dense(hidden_states)
    view_40: "f32[512, 768]" = torch.ops.aten.view.default(add_17, [512, 768])
    permute_20: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_32, [1, 0]);  primals_32 = None
    addmm_10: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_33, view_40, permute_20);  primals_33 = None
    view_41: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_10, [1, 512, 3072]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_13: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, 0.5)
    mul_14: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476)
    erf_1: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_14);  mul_14 = None
    add_18: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_15: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_13, add_18);  mul_13 = add_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:393, code: hidden_states = self.dense(hidden_states)
    view_42: "f32[512, 3072]" = torch.ops.aten.view.default(mul_15, [512, 3072]);  mul_15 = None
    permute_21: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
    addmm_11: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_35, view_42, permute_21);  primals_35 = None
    view_43: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_11, [1, 512, 768]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:394, code: hidden_states = self.dropout(hidden_states)
    native_dropout_6 = torch.ops.aten.native_dropout.default(view_43, 0.1, True);  view_43 = None
    getitem_20: "f32[1, 512, 768]" = native_dropout_6[0]
    getitem_21: "b8[1, 512, 768]" = native_dropout_6[1];  native_dropout_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:395, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_19: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_20, add_17);  getitem_20 = add_17 = None
    var_mean_4 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
    getitem_22: "f32[1, 512, 1]" = var_mean_4[0]
    getitem_23: "f32[1, 512, 1]" = var_mean_4[1];  var_mean_4 = None
    add_20: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
    rsqrt_4: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
    sub_7: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_19, getitem_23)
    mul_16: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_4);  sub_7 = None
    mul_17: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_16, primals_36);  mul_16 = None
    add_21: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_17, primals_37);  mul_17 = primals_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:211, code: mixed_query_layer = self.query(hidden_states)
    view_44: "f32[512, 768]" = torch.ops.aten.view.default(add_21, [512, 768])
    permute_22: "f32[768, 768]" = torch.ops.aten.permute.default(primals_38, [1, 0]);  primals_38 = None
    addmm_12: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_39, view_44, permute_22);  primals_39 = None
    view_45: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_12, [1, 512, 768]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:233, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_46: "f32[512, 768]" = torch.ops.aten.view.default(add_21, [512, 768])
    permute_23: "f32[768, 768]" = torch.ops.aten.permute.default(primals_40, [1, 0]);  primals_40 = None
    addmm_13: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_41, view_46, permute_23);  primals_41 = None
    view_47: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_13, [1, 512, 768]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_48: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_47, [1, 512, 12, 64]);  view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_24: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:234, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_49: "f32[512, 768]" = torch.ops.aten.view.default(add_21, [512, 768])
    permute_25: "f32[768, 768]" = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
    addmm_14: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_43, view_49, permute_25);  primals_43 = None
    view_50: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_14, [1, 512, 768]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_51: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_50, [1, 512, 12, 64]);  view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_26: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_52: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_45, [1, 512, 12, 64]);  view_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_27: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:250, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_28: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_24, [0, 1, 3, 2]);  permute_24 = None
    expand_9: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_27, [1, 12, 512, 64]);  permute_27 = None
    view_53: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_9, [12, 512, 64]);  expand_9 = None
    expand_10: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_28, [1, 12, 64, 512]);  permute_28 = None
    view_54: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_10, [12, 64, 512]);  expand_10 = None
    bmm_4: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_53, view_54)
    view_55: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_4, [1, 12, 512, 512]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:274, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_4: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_55, 8.0);  view_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:277, code: attention_scores = attention_scores + attention_mask
    add_22: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(div_4, mul);  div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:280, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_2: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_22, [-1], True)
    sub_8: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_22, amax_2);  add_22 = amax_2 = None
    exp_2: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_8);  sub_8 = None
    sum_3: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_5: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    alias_2: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(div_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:284, code: attention_probs = self.dropout(attention_probs)
    native_dropout_7 = torch.ops.aten.native_dropout.default(div_5, 0.1, True);  div_5 = None
    getitem_24: "f32[1, 12, 512, 512]" = native_dropout_7[0]
    getitem_25: "b8[1, 12, 512, 512]" = native_dropout_7[1];  native_dropout_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:290, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_11: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(getitem_24, [1, 12, 512, 512]);  getitem_24 = None
    view_56: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_11, [12, 512, 512]);  expand_11 = None
    expand_12: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_26, [1, 12, 512, 64]);  permute_26 = None
    view_57: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_12, [12, 512, 64]);  expand_12 = None
    bmm_5: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_56, view_57)
    view_58: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_5, [1, 12, 512, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:292, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_29: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
    clone_2: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:294, code: context_layer = context_layer.view(new_context_layer_shape)
    view_59: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_2, [1, 512, 768]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:312, code: hidden_states = self.dense(hidden_states)
    view_60: "f32[512, 768]" = torch.ops.aten.view.default(view_59, [512, 768]);  view_59 = None
    permute_30: "f32[768, 768]" = torch.ops.aten.permute.default(primals_44, [1, 0]);  primals_44 = None
    addmm_15: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_45, view_60, permute_30);  primals_45 = None
    view_61: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_15, [1, 512, 768]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:313, code: hidden_states = self.dropout(hidden_states)
    native_dropout_8 = torch.ops.aten.native_dropout.default(view_61, 0.1, True);  view_61 = None
    getitem_26: "f32[1, 512, 768]" = native_dropout_8[0]
    getitem_27: "b8[1, 512, 768]" = native_dropout_8[1];  native_dropout_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:314, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_23: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_26, add_21);  getitem_26 = add_21 = None
    var_mean_5 = torch.ops.aten.var_mean.correction(add_23, [2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 512, 1]" = var_mean_5[0]
    getitem_29: "f32[1, 512, 1]" = var_mean_5[1];  var_mean_5 = None
    add_24: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
    rsqrt_5: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
    sub_9: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_23, getitem_29)
    mul_18: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_5);  sub_9 = None
    mul_19: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_18, primals_46);  mul_18 = None
    add_25: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_19, primals_47);  mul_19 = primals_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:379, code: hidden_states = self.dense(hidden_states)
    view_62: "f32[512, 768]" = torch.ops.aten.view.default(add_25, [512, 768])
    permute_31: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_48, [1, 0]);  primals_48 = None
    addmm_16: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_49, view_62, permute_31);  primals_49 = None
    view_63: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_16, [1, 512, 3072]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_20: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, 0.5)
    mul_21: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476)
    erf_2: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_21);  mul_21 = None
    add_26: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_22: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_20, add_26);  mul_20 = add_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:393, code: hidden_states = self.dense(hidden_states)
    view_64: "f32[512, 3072]" = torch.ops.aten.view.default(mul_22, [512, 3072]);  mul_22 = None
    permute_32: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_50, [1, 0]);  primals_50 = None
    addmm_17: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_51, view_64, permute_32);  primals_51 = None
    view_65: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_17, [1, 512, 768]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:394, code: hidden_states = self.dropout(hidden_states)
    native_dropout_9 = torch.ops.aten.native_dropout.default(view_65, 0.1, True);  view_65 = None
    getitem_30: "f32[1, 512, 768]" = native_dropout_9[0]
    getitem_31: "b8[1, 512, 768]" = native_dropout_9[1];  native_dropout_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:395, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_27: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_30, add_25);  getitem_30 = add_25 = None
    var_mean_6 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
    getitem_32: "f32[1, 512, 1]" = var_mean_6[0]
    getitem_33: "f32[1, 512, 1]" = var_mean_6[1];  var_mean_6 = None
    add_28: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
    rsqrt_6: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    sub_10: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_27, getitem_33)
    mul_23: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_6);  sub_10 = None
    mul_24: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_23, primals_52);  mul_23 = None
    add_29: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_24, primals_53);  mul_24 = primals_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:211, code: mixed_query_layer = self.query(hidden_states)
    view_66: "f32[512, 768]" = torch.ops.aten.view.default(add_29, [512, 768])
    permute_33: "f32[768, 768]" = torch.ops.aten.permute.default(primals_54, [1, 0]);  primals_54 = None
    addmm_18: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_55, view_66, permute_33);  primals_55 = None
    view_67: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_18, [1, 512, 768]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:233, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_68: "f32[512, 768]" = torch.ops.aten.view.default(add_29, [512, 768])
    permute_34: "f32[768, 768]" = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
    addmm_19: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_57, view_68, permute_34);  primals_57 = None
    view_69: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_19, [1, 512, 768]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_70: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_69, [1, 512, 12, 64]);  view_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_35: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:234, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_71: "f32[512, 768]" = torch.ops.aten.view.default(add_29, [512, 768])
    permute_36: "f32[768, 768]" = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
    addmm_20: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_59, view_71, permute_36);  primals_59 = None
    view_72: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_20, [1, 512, 768]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_73: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_72, [1, 512, 12, 64]);  view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_37: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_74: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_67, [1, 512, 12, 64]);  view_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_38: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:250, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_39: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_35, [0, 1, 3, 2]);  permute_35 = None
    expand_13: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_38, [1, 12, 512, 64]);  permute_38 = None
    view_75: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_13, [12, 512, 64]);  expand_13 = None
    expand_14: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_39, [1, 12, 64, 512]);  permute_39 = None
    view_76: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_14, [12, 64, 512]);  expand_14 = None
    bmm_6: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_75, view_76)
    view_77: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_6, [1, 12, 512, 512]);  bmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:274, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_6: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_77, 8.0);  view_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:277, code: attention_scores = attention_scores + attention_mask
    add_30: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(div_6, mul);  div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:280, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_3: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_30, [-1], True)
    sub_11: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_30, amax_3);  add_30 = amax_3 = None
    exp_3: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
    sum_4: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_7: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    alias_3: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(div_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:284, code: attention_probs = self.dropout(attention_probs)
    native_dropout_10 = torch.ops.aten.native_dropout.default(div_7, 0.1, True);  div_7 = None
    getitem_34: "f32[1, 12, 512, 512]" = native_dropout_10[0]
    getitem_35: "b8[1, 12, 512, 512]" = native_dropout_10[1];  native_dropout_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:290, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_15: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(getitem_34, [1, 12, 512, 512]);  getitem_34 = None
    view_78: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_15, [12, 512, 512]);  expand_15 = None
    expand_16: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_37, [1, 12, 512, 64]);  permute_37 = None
    view_79: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_16, [12, 512, 64]);  expand_16 = None
    bmm_7: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_78, view_79)
    view_80: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_7, [1, 12, 512, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:292, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_40: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_80, [0, 2, 1, 3]);  view_80 = None
    clone_3: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:294, code: context_layer = context_layer.view(new_context_layer_shape)
    view_81: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_3, [1, 512, 768]);  clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:312, code: hidden_states = self.dense(hidden_states)
    view_82: "f32[512, 768]" = torch.ops.aten.view.default(view_81, [512, 768]);  view_81 = None
    permute_41: "f32[768, 768]" = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
    addmm_21: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_61, view_82, permute_41);  primals_61 = None
    view_83: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_21, [1, 512, 768]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:313, code: hidden_states = self.dropout(hidden_states)
    native_dropout_11 = torch.ops.aten.native_dropout.default(view_83, 0.1, True);  view_83 = None
    getitem_36: "f32[1, 512, 768]" = native_dropout_11[0]
    getitem_37: "b8[1, 512, 768]" = native_dropout_11[1];  native_dropout_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:314, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_31: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_36, add_29);  getitem_36 = add_29 = None
    var_mean_7 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
    getitem_38: "f32[1, 512, 1]" = var_mean_7[0]
    getitem_39: "f32[1, 512, 1]" = var_mean_7[1];  var_mean_7 = None
    add_32: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
    rsqrt_7: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    sub_12: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_31, getitem_39)
    mul_25: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_7);  sub_12 = None
    mul_26: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_25, primals_62);  mul_25 = None
    add_33: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_26, primals_63);  mul_26 = primals_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:379, code: hidden_states = self.dense(hidden_states)
    view_84: "f32[512, 768]" = torch.ops.aten.view.default(add_33, [512, 768])
    permute_42: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
    addmm_22: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_65, view_84, permute_42);  primals_65 = None
    view_85: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_22, [1, 512, 3072]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_27: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, 0.5)
    mul_28: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476)
    erf_3: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_28);  mul_28 = None
    add_34: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_29: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_27, add_34);  mul_27 = add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:393, code: hidden_states = self.dense(hidden_states)
    view_86: "f32[512, 3072]" = torch.ops.aten.view.default(mul_29, [512, 3072]);  mul_29 = None
    permute_43: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
    addmm_23: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_67, view_86, permute_43);  primals_67 = None
    view_87: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_23, [1, 512, 768]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:394, code: hidden_states = self.dropout(hidden_states)
    native_dropout_12 = torch.ops.aten.native_dropout.default(view_87, 0.1, True);  view_87 = None
    getitem_40: "f32[1, 512, 768]" = native_dropout_12[0]
    getitem_41: "b8[1, 512, 768]" = native_dropout_12[1];  native_dropout_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:395, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_35: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_40, add_33);  getitem_40 = add_33 = None
    var_mean_8 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
    getitem_42: "f32[1, 512, 1]" = var_mean_8[0]
    getitem_43: "f32[1, 512, 1]" = var_mean_8[1];  var_mean_8 = None
    add_36: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
    rsqrt_8: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    sub_13: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_43)
    mul_30: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_8);  sub_13 = None
    mul_31: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_30, primals_68);  mul_30 = None
    add_37: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_31, primals_69);  mul_31 = primals_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:211, code: mixed_query_layer = self.query(hidden_states)
    view_88: "f32[512, 768]" = torch.ops.aten.view.default(add_37, [512, 768])
    permute_44: "f32[768, 768]" = torch.ops.aten.permute.default(primals_70, [1, 0]);  primals_70 = None
    addmm_24: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_71, view_88, permute_44);  primals_71 = None
    view_89: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_24, [1, 512, 768]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:233, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_90: "f32[512, 768]" = torch.ops.aten.view.default(add_37, [512, 768])
    permute_45: "f32[768, 768]" = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
    addmm_25: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_73, view_90, permute_45);  primals_73 = None
    view_91: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_25, [1, 512, 768]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_92: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_91, [1, 512, 12, 64]);  view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_46: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:234, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_93: "f32[512, 768]" = torch.ops.aten.view.default(add_37, [512, 768])
    permute_47: "f32[768, 768]" = torch.ops.aten.permute.default(primals_74, [1, 0]);  primals_74 = None
    addmm_26: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_75, view_93, permute_47);  primals_75 = None
    view_94: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_26, [1, 512, 768]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_95: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_94, [1, 512, 12, 64]);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_48: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_96: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_89, [1, 512, 12, 64]);  view_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_49: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:250, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_50: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_46, [0, 1, 3, 2]);  permute_46 = None
    expand_17: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_49, [1, 12, 512, 64]);  permute_49 = None
    view_97: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_17, [12, 512, 64]);  expand_17 = None
    expand_18: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_50, [1, 12, 64, 512]);  permute_50 = None
    view_98: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_18, [12, 64, 512]);  expand_18 = None
    bmm_8: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_97, view_98)
    view_99: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_8, [1, 12, 512, 512]);  bmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:274, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_8: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_99, 8.0);  view_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:277, code: attention_scores = attention_scores + attention_mask
    add_38: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(div_8, mul);  div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:280, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_4: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_38, [-1], True)
    sub_14: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_38, amax_4);  add_38 = amax_4 = None
    exp_4: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
    sum_5: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_9: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    alias_4: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(div_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:284, code: attention_probs = self.dropout(attention_probs)
    native_dropout_13 = torch.ops.aten.native_dropout.default(div_9, 0.1, True);  div_9 = None
    getitem_44: "f32[1, 12, 512, 512]" = native_dropout_13[0]
    getitem_45: "b8[1, 12, 512, 512]" = native_dropout_13[1];  native_dropout_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:290, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_19: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(getitem_44, [1, 12, 512, 512]);  getitem_44 = None
    view_100: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_19, [12, 512, 512]);  expand_19 = None
    expand_20: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_48, [1, 12, 512, 64]);  permute_48 = None
    view_101: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_20, [12, 512, 64]);  expand_20 = None
    bmm_9: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_100, view_101)
    view_102: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_9, [1, 12, 512, 64]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:292, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_51: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
    clone_4: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:294, code: context_layer = context_layer.view(new_context_layer_shape)
    view_103: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_4, [1, 512, 768]);  clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:312, code: hidden_states = self.dense(hidden_states)
    view_104: "f32[512, 768]" = torch.ops.aten.view.default(view_103, [512, 768]);  view_103 = None
    permute_52: "f32[768, 768]" = torch.ops.aten.permute.default(primals_76, [1, 0]);  primals_76 = None
    addmm_27: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_77, view_104, permute_52);  primals_77 = None
    view_105: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_27, [1, 512, 768]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:313, code: hidden_states = self.dropout(hidden_states)
    native_dropout_14 = torch.ops.aten.native_dropout.default(view_105, 0.1, True);  view_105 = None
    getitem_46: "f32[1, 512, 768]" = native_dropout_14[0]
    getitem_47: "b8[1, 512, 768]" = native_dropout_14[1];  native_dropout_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:314, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_39: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_46, add_37);  getitem_46 = add_37 = None
    var_mean_9 = torch.ops.aten.var_mean.correction(add_39, [2], correction = 0, keepdim = True)
    getitem_48: "f32[1, 512, 1]" = var_mean_9[0]
    getitem_49: "f32[1, 512, 1]" = var_mean_9[1];  var_mean_9 = None
    add_40: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
    rsqrt_9: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
    sub_15: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_39, getitem_49)
    mul_32: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_9);  sub_15 = None
    mul_33: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_32, primals_78);  mul_32 = None
    add_41: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_33, primals_79);  mul_33 = primals_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:379, code: hidden_states = self.dense(hidden_states)
    view_106: "f32[512, 768]" = torch.ops.aten.view.default(add_41, [512, 768])
    permute_53: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_80, [1, 0]);  primals_80 = None
    addmm_28: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_81, view_106, permute_53);  primals_81 = None
    view_107: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_28, [1, 512, 3072]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_34: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.5)
    mul_35: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476)
    erf_4: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_35);  mul_35 = None
    add_42: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_36: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_34, add_42);  mul_34 = add_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:393, code: hidden_states = self.dense(hidden_states)
    view_108: "f32[512, 3072]" = torch.ops.aten.view.default(mul_36, [512, 3072]);  mul_36 = None
    permute_54: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_82, [1, 0]);  primals_82 = None
    addmm_29: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_83, view_108, permute_54);  primals_83 = None
    view_109: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_29, [1, 512, 768]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:394, code: hidden_states = self.dropout(hidden_states)
    native_dropout_15 = torch.ops.aten.native_dropout.default(view_109, 0.1, True);  view_109 = None
    getitem_50: "f32[1, 512, 768]" = native_dropout_15[0]
    getitem_51: "b8[1, 512, 768]" = native_dropout_15[1];  native_dropout_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:395, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_43: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_50, add_41);  getitem_50 = add_41 = None
    var_mean_10 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
    getitem_52: "f32[1, 512, 1]" = var_mean_10[0]
    getitem_53: "f32[1, 512, 1]" = var_mean_10[1];  var_mean_10 = None
    add_44: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
    rsqrt_10: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
    sub_16: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_43, getitem_53)
    mul_37: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_10);  sub_16 = None
    mul_38: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_37, primals_84);  mul_37 = None
    add_45: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_38, primals_85);  mul_38 = primals_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:211, code: mixed_query_layer = self.query(hidden_states)
    view_110: "f32[512, 768]" = torch.ops.aten.view.default(add_45, [512, 768])
    permute_55: "f32[768, 768]" = torch.ops.aten.permute.default(primals_86, [1, 0]);  primals_86 = None
    addmm_30: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_87, view_110, permute_55);  primals_87 = None
    view_111: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_30, [1, 512, 768]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:233, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_112: "f32[512, 768]" = torch.ops.aten.view.default(add_45, [512, 768])
    permute_56: "f32[768, 768]" = torch.ops.aten.permute.default(primals_88, [1, 0]);  primals_88 = None
    addmm_31: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_89, view_112, permute_56);  primals_89 = None
    view_113: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_31, [1, 512, 768]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_114: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_113, [1, 512, 12, 64]);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_57: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:234, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_115: "f32[512, 768]" = torch.ops.aten.view.default(add_45, [512, 768])
    permute_58: "f32[768, 768]" = torch.ops.aten.permute.default(primals_90, [1, 0]);  primals_90 = None
    addmm_32: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_91, view_115, permute_58);  primals_91 = None
    view_116: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_32, [1, 512, 768]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_117: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_116, [1, 512, 12, 64]);  view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_59: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_118: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_111, [1, 512, 12, 64]);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_60: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:250, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_61: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_57, [0, 1, 3, 2]);  permute_57 = None
    expand_21: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_60, [1, 12, 512, 64]);  permute_60 = None
    view_119: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_21, [12, 512, 64]);  expand_21 = None
    expand_22: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_61, [1, 12, 64, 512]);  permute_61 = None
    view_120: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_22, [12, 64, 512]);  expand_22 = None
    bmm_10: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_119, view_120)
    view_121: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_10, [1, 12, 512, 512]);  bmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:274, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_10: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_121, 8.0);  view_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:277, code: attention_scores = attention_scores + attention_mask
    add_46: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(div_10, mul);  div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:280, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_5: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_46, [-1], True)
    sub_17: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_46, amax_5);  add_46 = amax_5 = None
    exp_5: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
    sum_6: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_11: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    alias_5: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(div_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:284, code: attention_probs = self.dropout(attention_probs)
    native_dropout_16 = torch.ops.aten.native_dropout.default(div_11, 0.1, True);  div_11 = None
    getitem_54: "f32[1, 12, 512, 512]" = native_dropout_16[0]
    getitem_55: "b8[1, 12, 512, 512]" = native_dropout_16[1];  native_dropout_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:290, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_23: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(getitem_54, [1, 12, 512, 512]);  getitem_54 = None
    view_122: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_23, [12, 512, 512]);  expand_23 = None
    expand_24: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_59, [1, 12, 512, 64]);  permute_59 = None
    view_123: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_24, [12, 512, 64]);  expand_24 = None
    bmm_11: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_122, view_123)
    view_124: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_11, [1, 12, 512, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:292, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_62: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None
    clone_5: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:294, code: context_layer = context_layer.view(new_context_layer_shape)
    view_125: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_5, [1, 512, 768]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:312, code: hidden_states = self.dense(hidden_states)
    view_126: "f32[512, 768]" = torch.ops.aten.view.default(view_125, [512, 768]);  view_125 = None
    permute_63: "f32[768, 768]" = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
    addmm_33: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_93, view_126, permute_63);  primals_93 = None
    view_127: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_33, [1, 512, 768]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:313, code: hidden_states = self.dropout(hidden_states)
    native_dropout_17 = torch.ops.aten.native_dropout.default(view_127, 0.1, True);  view_127 = None
    getitem_56: "f32[1, 512, 768]" = native_dropout_17[0]
    getitem_57: "b8[1, 512, 768]" = native_dropout_17[1];  native_dropout_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:314, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_47: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_56, add_45);  getitem_56 = add_45 = None
    var_mean_11 = torch.ops.aten.var_mean.correction(add_47, [2], correction = 0, keepdim = True)
    getitem_58: "f32[1, 512, 1]" = var_mean_11[0]
    getitem_59: "f32[1, 512, 1]" = var_mean_11[1];  var_mean_11 = None
    add_48: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
    rsqrt_11: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
    sub_18: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_47, getitem_59)
    mul_39: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_11);  sub_18 = None
    mul_40: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_39, primals_94);  mul_39 = None
    add_49: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_40, primals_95);  mul_40 = primals_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:379, code: hidden_states = self.dense(hidden_states)
    view_128: "f32[512, 768]" = torch.ops.aten.view.default(add_49, [512, 768])
    permute_64: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_96, [1, 0]);  primals_96 = None
    addmm_34: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_97, view_128, permute_64);  primals_97 = None
    view_129: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_34, [1, 512, 3072]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_41: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, 0.5)
    mul_42: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476)
    erf_5: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_42);  mul_42 = None
    add_50: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_43: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_41, add_50);  mul_41 = add_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:393, code: hidden_states = self.dense(hidden_states)
    view_130: "f32[512, 3072]" = torch.ops.aten.view.default(mul_43, [512, 3072]);  mul_43 = None
    permute_65: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_98, [1, 0]);  primals_98 = None
    addmm_35: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_99, view_130, permute_65);  primals_99 = None
    view_131: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_35, [1, 512, 768]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:394, code: hidden_states = self.dropout(hidden_states)
    native_dropout_18 = torch.ops.aten.native_dropout.default(view_131, 0.1, True);  view_131 = None
    getitem_60: "f32[1, 512, 768]" = native_dropout_18[0]
    getitem_61: "b8[1, 512, 768]" = native_dropout_18[1];  native_dropout_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:395, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_51: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_60, add_49);  getitem_60 = add_49 = None
    var_mean_12 = torch.ops.aten.var_mean.correction(add_51, [2], correction = 0, keepdim = True)
    getitem_62: "f32[1, 512, 1]" = var_mean_12[0]
    getitem_63: "f32[1, 512, 1]" = var_mean_12[1];  var_mean_12 = None
    add_52: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
    rsqrt_12: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    sub_19: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_51, getitem_63)
    mul_44: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_12);  sub_19 = None
    mul_45: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_44, primals_100);  mul_44 = None
    add_53: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_45, primals_101);  mul_45 = primals_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:211, code: mixed_query_layer = self.query(hidden_states)
    view_132: "f32[512, 768]" = torch.ops.aten.view.default(add_53, [512, 768])
    permute_66: "f32[768, 768]" = torch.ops.aten.permute.default(primals_102, [1, 0]);  primals_102 = None
    addmm_36: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_103, view_132, permute_66);  primals_103 = None
    view_133: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_36, [1, 512, 768]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:233, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_134: "f32[512, 768]" = torch.ops.aten.view.default(add_53, [512, 768])
    permute_67: "f32[768, 768]" = torch.ops.aten.permute.default(primals_104, [1, 0]);  primals_104 = None
    addmm_37: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_105, view_134, permute_67);  primals_105 = None
    view_135: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_37, [1, 512, 768]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_136: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_135, [1, 512, 12, 64]);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_68: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:234, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_137: "f32[512, 768]" = torch.ops.aten.view.default(add_53, [512, 768])
    permute_69: "f32[768, 768]" = torch.ops.aten.permute.default(primals_106, [1, 0]);  primals_106 = None
    addmm_38: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_107, view_137, permute_69);  primals_107 = None
    view_138: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_38, [1, 512, 768]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_139: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_138, [1, 512, 12, 64]);  view_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_70: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_139, [0, 2, 1, 3]);  view_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_140: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_133, [1, 512, 12, 64]);  view_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_71: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:250, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_72: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_68, [0, 1, 3, 2]);  permute_68 = None
    expand_25: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_71, [1, 12, 512, 64]);  permute_71 = None
    view_141: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_25, [12, 512, 64]);  expand_25 = None
    expand_26: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_72, [1, 12, 64, 512]);  permute_72 = None
    view_142: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_26, [12, 64, 512]);  expand_26 = None
    bmm_12: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_141, view_142)
    view_143: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_12, [1, 12, 512, 512]);  bmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:274, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_12: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_143, 8.0);  view_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:277, code: attention_scores = attention_scores + attention_mask
    add_54: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(div_12, mul);  div_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:280, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_6: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_54, [-1], True)
    sub_20: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_54, amax_6);  add_54 = amax_6 = None
    exp_6: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
    sum_7: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_13: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    alias_6: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(div_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:284, code: attention_probs = self.dropout(attention_probs)
    native_dropout_19 = torch.ops.aten.native_dropout.default(div_13, 0.1, True);  div_13 = None
    getitem_64: "f32[1, 12, 512, 512]" = native_dropout_19[0]
    getitem_65: "b8[1, 12, 512, 512]" = native_dropout_19[1];  native_dropout_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:290, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_27: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(getitem_64, [1, 12, 512, 512]);  getitem_64 = None
    view_144: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_27, [12, 512, 512]);  expand_27 = None
    expand_28: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_70, [1, 12, 512, 64]);  permute_70 = None
    view_145: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_28, [12, 512, 64]);  expand_28 = None
    bmm_13: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_144, view_145)
    view_146: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_13, [1, 12, 512, 64]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:292, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_73: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
    clone_6: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:294, code: context_layer = context_layer.view(new_context_layer_shape)
    view_147: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_6, [1, 512, 768]);  clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:312, code: hidden_states = self.dense(hidden_states)
    view_148: "f32[512, 768]" = torch.ops.aten.view.default(view_147, [512, 768]);  view_147 = None
    permute_74: "f32[768, 768]" = torch.ops.aten.permute.default(primals_108, [1, 0]);  primals_108 = None
    addmm_39: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_109, view_148, permute_74);  primals_109 = None
    view_149: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_39, [1, 512, 768]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:313, code: hidden_states = self.dropout(hidden_states)
    native_dropout_20 = torch.ops.aten.native_dropout.default(view_149, 0.1, True);  view_149 = None
    getitem_66: "f32[1, 512, 768]" = native_dropout_20[0]
    getitem_67: "b8[1, 512, 768]" = native_dropout_20[1];  native_dropout_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:314, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_55: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_66, add_53);  getitem_66 = add_53 = None
    var_mean_13 = torch.ops.aten.var_mean.correction(add_55, [2], correction = 0, keepdim = True)
    getitem_68: "f32[1, 512, 1]" = var_mean_13[0]
    getitem_69: "f32[1, 512, 1]" = var_mean_13[1];  var_mean_13 = None
    add_56: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05);  getitem_68 = None
    rsqrt_13: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    sub_21: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_55, getitem_69)
    mul_46: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_13);  sub_21 = None
    mul_47: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_46, primals_110);  mul_46 = None
    add_57: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_47, primals_111);  mul_47 = primals_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:379, code: hidden_states = self.dense(hidden_states)
    view_150: "f32[512, 768]" = torch.ops.aten.view.default(add_57, [512, 768])
    permute_75: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_112, [1, 0]);  primals_112 = None
    addmm_40: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_113, view_150, permute_75);  primals_113 = None
    view_151: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_40, [1, 512, 3072]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_48: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, 0.5)
    mul_49: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476)
    erf_6: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_49);  mul_49 = None
    add_58: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_50: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_48, add_58);  mul_48 = add_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:393, code: hidden_states = self.dense(hidden_states)
    view_152: "f32[512, 3072]" = torch.ops.aten.view.default(mul_50, [512, 3072]);  mul_50 = None
    permute_76: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_114, [1, 0]);  primals_114 = None
    addmm_41: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_115, view_152, permute_76);  primals_115 = None
    view_153: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_41, [1, 512, 768]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:394, code: hidden_states = self.dropout(hidden_states)
    native_dropout_21 = torch.ops.aten.native_dropout.default(view_153, 0.1, True);  view_153 = None
    getitem_70: "f32[1, 512, 768]" = native_dropout_21[0]
    getitem_71: "b8[1, 512, 768]" = native_dropout_21[1];  native_dropout_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:395, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_59: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_70, add_57);  getitem_70 = add_57 = None
    var_mean_14 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
    getitem_72: "f32[1, 512, 1]" = var_mean_14[0]
    getitem_73: "f32[1, 512, 1]" = var_mean_14[1];  var_mean_14 = None
    add_60: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05);  getitem_72 = None
    rsqrt_14: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    sub_22: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_59, getitem_73)
    mul_51: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_14);  sub_22 = None
    mul_52: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_51, primals_116);  mul_51 = None
    add_61: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_52, primals_117);  mul_52 = primals_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:211, code: mixed_query_layer = self.query(hidden_states)
    view_154: "f32[512, 768]" = torch.ops.aten.view.default(add_61, [512, 768])
    permute_77: "f32[768, 768]" = torch.ops.aten.permute.default(primals_118, [1, 0]);  primals_118 = None
    addmm_42: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_119, view_154, permute_77);  primals_119 = None
    view_155: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_42, [1, 512, 768]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:233, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_156: "f32[512, 768]" = torch.ops.aten.view.default(add_61, [512, 768])
    permute_78: "f32[768, 768]" = torch.ops.aten.permute.default(primals_120, [1, 0]);  primals_120 = None
    addmm_43: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_121, view_156, permute_78);  primals_121 = None
    view_157: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_43, [1, 512, 768]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_158: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_157, [1, 512, 12, 64]);  view_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_79: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_158, [0, 2, 1, 3]);  view_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:234, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_159: "f32[512, 768]" = torch.ops.aten.view.default(add_61, [512, 768])
    permute_80: "f32[768, 768]" = torch.ops.aten.permute.default(primals_122, [1, 0]);  primals_122 = None
    addmm_44: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_123, view_159, permute_80);  primals_123 = None
    view_160: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_44, [1, 512, 768]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_161: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_160, [1, 512, 12, 64]);  view_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_81: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_161, [0, 2, 1, 3]);  view_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_162: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_155, [1, 512, 12, 64]);  view_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_82: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:250, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_83: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_79, [0, 1, 3, 2]);  permute_79 = None
    expand_29: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_82, [1, 12, 512, 64]);  permute_82 = None
    view_163: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_29, [12, 512, 64]);  expand_29 = None
    expand_30: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_83, [1, 12, 64, 512]);  permute_83 = None
    view_164: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_30, [12, 64, 512]);  expand_30 = None
    bmm_14: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_163, view_164)
    view_165: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_14, [1, 12, 512, 512]);  bmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:274, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_14: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_165, 8.0);  view_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:277, code: attention_scores = attention_scores + attention_mask
    add_62: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(div_14, mul);  div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:280, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_7: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_62, [-1], True)
    sub_23: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_62, amax_7);  add_62 = amax_7 = None
    exp_7: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_23);  sub_23 = None
    sum_8: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_15: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    alias_7: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(div_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:284, code: attention_probs = self.dropout(attention_probs)
    native_dropout_22 = torch.ops.aten.native_dropout.default(div_15, 0.1, True);  div_15 = None
    getitem_74: "f32[1, 12, 512, 512]" = native_dropout_22[0]
    getitem_75: "b8[1, 12, 512, 512]" = native_dropout_22[1];  native_dropout_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:290, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_31: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(getitem_74, [1, 12, 512, 512]);  getitem_74 = None
    view_166: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_31, [12, 512, 512]);  expand_31 = None
    expand_32: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_81, [1, 12, 512, 64]);  permute_81 = None
    view_167: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_32, [12, 512, 64]);  expand_32 = None
    bmm_15: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_166, view_167)
    view_168: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_15, [1, 12, 512, 64]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:292, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_84: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
    clone_7: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:294, code: context_layer = context_layer.view(new_context_layer_shape)
    view_169: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_7, [1, 512, 768]);  clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:312, code: hidden_states = self.dense(hidden_states)
    view_170: "f32[512, 768]" = torch.ops.aten.view.default(view_169, [512, 768]);  view_169 = None
    permute_85: "f32[768, 768]" = torch.ops.aten.permute.default(primals_124, [1, 0]);  primals_124 = None
    addmm_45: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_125, view_170, permute_85);  primals_125 = None
    view_171: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_45, [1, 512, 768]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:313, code: hidden_states = self.dropout(hidden_states)
    native_dropout_23 = torch.ops.aten.native_dropout.default(view_171, 0.1, True);  view_171 = None
    getitem_76: "f32[1, 512, 768]" = native_dropout_23[0]
    getitem_77: "b8[1, 512, 768]" = native_dropout_23[1];  native_dropout_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:314, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_63: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_76, add_61);  getitem_76 = add_61 = None
    var_mean_15 = torch.ops.aten.var_mean.correction(add_63, [2], correction = 0, keepdim = True)
    getitem_78: "f32[1, 512, 1]" = var_mean_15[0]
    getitem_79: "f32[1, 512, 1]" = var_mean_15[1];  var_mean_15 = None
    add_64: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
    rsqrt_15: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
    sub_24: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_63, getitem_79)
    mul_53: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_15);  sub_24 = None
    mul_54: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_53, primals_126);  mul_53 = None
    add_65: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_54, primals_127);  mul_54 = primals_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:379, code: hidden_states = self.dense(hidden_states)
    view_172: "f32[512, 768]" = torch.ops.aten.view.default(add_65, [512, 768])
    permute_86: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_128, [1, 0]);  primals_128 = None
    addmm_46: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_129, view_172, permute_86);  primals_129 = None
    view_173: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_46, [1, 512, 3072]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_55: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.5)
    mul_56: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476)
    erf_7: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_56);  mul_56 = None
    add_66: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_57: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_55, add_66);  mul_55 = add_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:393, code: hidden_states = self.dense(hidden_states)
    view_174: "f32[512, 3072]" = torch.ops.aten.view.default(mul_57, [512, 3072]);  mul_57 = None
    permute_87: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_130, [1, 0]);  primals_130 = None
    addmm_47: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_131, view_174, permute_87);  primals_131 = None
    view_175: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_47, [1, 512, 768]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:394, code: hidden_states = self.dropout(hidden_states)
    native_dropout_24 = torch.ops.aten.native_dropout.default(view_175, 0.1, True);  view_175 = None
    getitem_80: "f32[1, 512, 768]" = native_dropout_24[0]
    getitem_81: "b8[1, 512, 768]" = native_dropout_24[1];  native_dropout_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:395, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_67: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_80, add_65);  getitem_80 = add_65 = None
    var_mean_16 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
    getitem_82: "f32[1, 512, 1]" = var_mean_16[0]
    getitem_83: "f32[1, 512, 1]" = var_mean_16[1];  var_mean_16 = None
    add_68: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
    rsqrt_16: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
    sub_25: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_67, getitem_83)
    mul_58: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_16);  sub_25 = None
    mul_59: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_58, primals_132);  mul_58 = None
    add_69: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_59, primals_133);  mul_59 = primals_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:211, code: mixed_query_layer = self.query(hidden_states)
    view_176: "f32[512, 768]" = torch.ops.aten.view.default(add_69, [512, 768])
    permute_88: "f32[768, 768]" = torch.ops.aten.permute.default(primals_134, [1, 0]);  primals_134 = None
    addmm_48: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_135, view_176, permute_88);  primals_135 = None
    view_177: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_48, [1, 512, 768]);  addmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:233, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_178: "f32[512, 768]" = torch.ops.aten.view.default(add_69, [512, 768])
    permute_89: "f32[768, 768]" = torch.ops.aten.permute.default(primals_136, [1, 0]);  primals_136 = None
    addmm_49: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_137, view_178, permute_89);  primals_137 = None
    view_179: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_49, [1, 512, 768]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_180: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_179, [1, 512, 12, 64]);  view_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_90: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_180, [0, 2, 1, 3]);  view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:234, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_181: "f32[512, 768]" = torch.ops.aten.view.default(add_69, [512, 768])
    permute_91: "f32[768, 768]" = torch.ops.aten.permute.default(primals_138, [1, 0]);  primals_138 = None
    addmm_50: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_139, view_181, permute_91);  primals_139 = None
    view_182: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_50, [1, 512, 768]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_183: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_182, [1, 512, 12, 64]);  view_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_92: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_184: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_177, [1, 512, 12, 64]);  view_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_93: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:250, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_94: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_90, [0, 1, 3, 2]);  permute_90 = None
    expand_33: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_93, [1, 12, 512, 64]);  permute_93 = None
    view_185: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_33, [12, 512, 64]);  expand_33 = None
    expand_34: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_94, [1, 12, 64, 512]);  permute_94 = None
    view_186: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_34, [12, 64, 512]);  expand_34 = None
    bmm_16: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_185, view_186)
    view_187: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_16, [1, 12, 512, 512]);  bmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:274, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_16: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_187, 8.0);  view_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:277, code: attention_scores = attention_scores + attention_mask
    add_70: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(div_16, mul);  div_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:280, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_8: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_70, [-1], True)
    sub_26: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_70, amax_8);  add_70 = amax_8 = None
    exp_8: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_26);  sub_26 = None
    sum_9: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_17: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    alias_8: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(div_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:284, code: attention_probs = self.dropout(attention_probs)
    native_dropout_25 = torch.ops.aten.native_dropout.default(div_17, 0.1, True);  div_17 = None
    getitem_84: "f32[1, 12, 512, 512]" = native_dropout_25[0]
    getitem_85: "b8[1, 12, 512, 512]" = native_dropout_25[1];  native_dropout_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:290, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_35: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(getitem_84, [1, 12, 512, 512]);  getitem_84 = None
    view_188: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_35, [12, 512, 512]);  expand_35 = None
    expand_36: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_92, [1, 12, 512, 64]);  permute_92 = None
    view_189: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_36, [12, 512, 64]);  expand_36 = None
    bmm_17: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_188, view_189)
    view_190: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_17, [1, 12, 512, 64]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:292, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_95: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_190, [0, 2, 1, 3]);  view_190 = None
    clone_8: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:294, code: context_layer = context_layer.view(new_context_layer_shape)
    view_191: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_8, [1, 512, 768]);  clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:312, code: hidden_states = self.dense(hidden_states)
    view_192: "f32[512, 768]" = torch.ops.aten.view.default(view_191, [512, 768]);  view_191 = None
    permute_96: "f32[768, 768]" = torch.ops.aten.permute.default(primals_140, [1, 0]);  primals_140 = None
    addmm_51: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_141, view_192, permute_96);  primals_141 = None
    view_193: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_51, [1, 512, 768]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:313, code: hidden_states = self.dropout(hidden_states)
    native_dropout_26 = torch.ops.aten.native_dropout.default(view_193, 0.1, True);  view_193 = None
    getitem_86: "f32[1, 512, 768]" = native_dropout_26[0]
    getitem_87: "b8[1, 512, 768]" = native_dropout_26[1];  native_dropout_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:314, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_71: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_86, add_69);  getitem_86 = add_69 = None
    var_mean_17 = torch.ops.aten.var_mean.correction(add_71, [2], correction = 0, keepdim = True)
    getitem_88: "f32[1, 512, 1]" = var_mean_17[0]
    getitem_89: "f32[1, 512, 1]" = var_mean_17[1];  var_mean_17 = None
    add_72: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
    rsqrt_17: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
    sub_27: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_71, getitem_89)
    mul_60: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_17);  sub_27 = None
    mul_61: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_60, primals_142);  mul_60 = None
    add_73: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_61, primals_143);  mul_61 = primals_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:379, code: hidden_states = self.dense(hidden_states)
    view_194: "f32[512, 768]" = torch.ops.aten.view.default(add_73, [512, 768])
    permute_97: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_144, [1, 0]);  primals_144 = None
    addmm_52: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_145, view_194, permute_97);  primals_145 = None
    view_195: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_52, [1, 512, 3072]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_62: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, 0.5)
    mul_63: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476)
    erf_8: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_63);  mul_63 = None
    add_74: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_64: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_62, add_74);  mul_62 = add_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:393, code: hidden_states = self.dense(hidden_states)
    view_196: "f32[512, 3072]" = torch.ops.aten.view.default(mul_64, [512, 3072]);  mul_64 = None
    permute_98: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_146, [1, 0]);  primals_146 = None
    addmm_53: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_147, view_196, permute_98);  primals_147 = None
    view_197: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_53, [1, 512, 768]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:394, code: hidden_states = self.dropout(hidden_states)
    native_dropout_27 = torch.ops.aten.native_dropout.default(view_197, 0.1, True);  view_197 = None
    getitem_90: "f32[1, 512, 768]" = native_dropout_27[0]
    getitem_91: "b8[1, 512, 768]" = native_dropout_27[1];  native_dropout_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:395, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_75: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_90, add_73);  getitem_90 = add_73 = None
    var_mean_18 = torch.ops.aten.var_mean.correction(add_75, [2], correction = 0, keepdim = True)
    getitem_92: "f32[1, 512, 1]" = var_mean_18[0]
    getitem_93: "f32[1, 512, 1]" = var_mean_18[1];  var_mean_18 = None
    add_76: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05);  getitem_92 = None
    rsqrt_18: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
    sub_28: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_75, getitem_93)
    mul_65: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_18);  sub_28 = None
    mul_66: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_65, primals_148);  mul_65 = None
    add_77: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_66, primals_149);  mul_66 = primals_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:211, code: mixed_query_layer = self.query(hidden_states)
    view_198: "f32[512, 768]" = torch.ops.aten.view.default(add_77, [512, 768])
    permute_99: "f32[768, 768]" = torch.ops.aten.permute.default(primals_150, [1, 0]);  primals_150 = None
    addmm_54: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_151, view_198, permute_99);  primals_151 = None
    view_199: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_54, [1, 512, 768]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:233, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_200: "f32[512, 768]" = torch.ops.aten.view.default(add_77, [512, 768])
    permute_100: "f32[768, 768]" = torch.ops.aten.permute.default(primals_152, [1, 0]);  primals_152 = None
    addmm_55: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_153, view_200, permute_100);  primals_153 = None
    view_201: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_55, [1, 512, 768]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_202: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_201, [1, 512, 12, 64]);  view_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_101: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_202, [0, 2, 1, 3]);  view_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:234, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_203: "f32[512, 768]" = torch.ops.aten.view.default(add_77, [512, 768])
    permute_102: "f32[768, 768]" = torch.ops.aten.permute.default(primals_154, [1, 0]);  primals_154 = None
    addmm_56: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_155, view_203, permute_102);  primals_155 = None
    view_204: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_56, [1, 512, 768]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_205: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_204, [1, 512, 12, 64]);  view_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_103: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_205, [0, 2, 1, 3]);  view_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_206: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_199, [1, 512, 12, 64]);  view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_104: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:250, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_105: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_101, [0, 1, 3, 2]);  permute_101 = None
    expand_37: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_104, [1, 12, 512, 64]);  permute_104 = None
    view_207: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_37, [12, 512, 64]);  expand_37 = None
    expand_38: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_105, [1, 12, 64, 512]);  permute_105 = None
    view_208: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_38, [12, 64, 512]);  expand_38 = None
    bmm_18: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_207, view_208)
    view_209: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_18, [1, 12, 512, 512]);  bmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:274, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_18: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_209, 8.0);  view_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:277, code: attention_scores = attention_scores + attention_mask
    add_78: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(div_18, mul);  div_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:280, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_9: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_78, [-1], True)
    sub_29: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_78, amax_9);  add_78 = amax_9 = None
    exp_9: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_29);  sub_29 = None
    sum_10: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_19: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    alias_9: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(div_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:284, code: attention_probs = self.dropout(attention_probs)
    native_dropout_28 = torch.ops.aten.native_dropout.default(div_19, 0.1, True);  div_19 = None
    getitem_94: "f32[1, 12, 512, 512]" = native_dropout_28[0]
    getitem_95: "b8[1, 12, 512, 512]" = native_dropout_28[1];  native_dropout_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:290, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_39: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(getitem_94, [1, 12, 512, 512]);  getitem_94 = None
    view_210: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_39, [12, 512, 512]);  expand_39 = None
    expand_40: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_103, [1, 12, 512, 64]);  permute_103 = None
    view_211: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_40, [12, 512, 64]);  expand_40 = None
    bmm_19: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_210, view_211)
    view_212: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_19, [1, 12, 512, 64]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:292, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_106: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None
    clone_9: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:294, code: context_layer = context_layer.view(new_context_layer_shape)
    view_213: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_9, [1, 512, 768]);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:312, code: hidden_states = self.dense(hidden_states)
    view_214: "f32[512, 768]" = torch.ops.aten.view.default(view_213, [512, 768]);  view_213 = None
    permute_107: "f32[768, 768]" = torch.ops.aten.permute.default(primals_156, [1, 0]);  primals_156 = None
    addmm_57: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_157, view_214, permute_107);  primals_157 = None
    view_215: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_57, [1, 512, 768]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:313, code: hidden_states = self.dropout(hidden_states)
    native_dropout_29 = torch.ops.aten.native_dropout.default(view_215, 0.1, True);  view_215 = None
    getitem_96: "f32[1, 512, 768]" = native_dropout_29[0]
    getitem_97: "b8[1, 512, 768]" = native_dropout_29[1];  native_dropout_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:314, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_79: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_96, add_77);  getitem_96 = add_77 = None
    var_mean_19 = torch.ops.aten.var_mean.correction(add_79, [2], correction = 0, keepdim = True)
    getitem_98: "f32[1, 512, 1]" = var_mean_19[0]
    getitem_99: "f32[1, 512, 1]" = var_mean_19[1];  var_mean_19 = None
    add_80: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-05);  getitem_98 = None
    rsqrt_19: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
    sub_30: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_79, getitem_99)
    mul_67: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_19);  sub_30 = None
    mul_68: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_67, primals_158);  mul_67 = None
    add_81: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_68, primals_159);  mul_68 = primals_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:379, code: hidden_states = self.dense(hidden_states)
    view_216: "f32[512, 768]" = torch.ops.aten.view.default(add_81, [512, 768])
    permute_108: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_160, [1, 0]);  primals_160 = None
    addmm_58: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_161, view_216, permute_108);  primals_161 = None
    view_217: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_58, [1, 512, 3072]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_69: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, 0.5)
    mul_70: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476)
    erf_9: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_70);  mul_70 = None
    add_82: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_71: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_69, add_82);  mul_69 = add_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:393, code: hidden_states = self.dense(hidden_states)
    view_218: "f32[512, 3072]" = torch.ops.aten.view.default(mul_71, [512, 3072]);  mul_71 = None
    permute_109: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_162, [1, 0]);  primals_162 = None
    addmm_59: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_163, view_218, permute_109);  primals_163 = None
    view_219: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_59, [1, 512, 768]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:394, code: hidden_states = self.dropout(hidden_states)
    native_dropout_30 = torch.ops.aten.native_dropout.default(view_219, 0.1, True);  view_219 = None
    getitem_100: "f32[1, 512, 768]" = native_dropout_30[0]
    getitem_101: "b8[1, 512, 768]" = native_dropout_30[1];  native_dropout_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:395, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_83: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_100, add_81);  getitem_100 = add_81 = None
    var_mean_20 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
    getitem_102: "f32[1, 512, 1]" = var_mean_20[0]
    getitem_103: "f32[1, 512, 1]" = var_mean_20[1];  var_mean_20 = None
    add_84: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-05);  getitem_102 = None
    rsqrt_20: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
    sub_31: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_83, getitem_103)
    mul_72: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_20);  sub_31 = None
    mul_73: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_72, primals_164);  mul_72 = None
    add_85: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_73, primals_165);  mul_73 = primals_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:211, code: mixed_query_layer = self.query(hidden_states)
    view_220: "f32[512, 768]" = torch.ops.aten.view.default(add_85, [512, 768])
    permute_110: "f32[768, 768]" = torch.ops.aten.permute.default(primals_166, [1, 0]);  primals_166 = None
    addmm_60: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_167, view_220, permute_110);  primals_167 = None
    view_221: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_60, [1, 512, 768]);  addmm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:233, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_222: "f32[512, 768]" = torch.ops.aten.view.default(add_85, [512, 768])
    permute_111: "f32[768, 768]" = torch.ops.aten.permute.default(primals_168, [1, 0]);  primals_168 = None
    addmm_61: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_169, view_222, permute_111);  primals_169 = None
    view_223: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_61, [1, 512, 768]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_224: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_223, [1, 512, 12, 64]);  view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_112: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:234, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_225: "f32[512, 768]" = torch.ops.aten.view.default(add_85, [512, 768])
    permute_113: "f32[768, 768]" = torch.ops.aten.permute.default(primals_170, [1, 0]);  primals_170 = None
    addmm_62: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_171, view_225, permute_113);  primals_171 = None
    view_226: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_62, [1, 512, 768]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_227: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_226, [1, 512, 12, 64]);  view_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_114: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_227, [0, 2, 1, 3]);  view_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_228: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_221, [1, 512, 12, 64]);  view_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_115: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:250, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_116: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_112, [0, 1, 3, 2]);  permute_112 = None
    expand_41: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_115, [1, 12, 512, 64]);  permute_115 = None
    view_229: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_41, [12, 512, 64]);  expand_41 = None
    expand_42: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_116, [1, 12, 64, 512]);  permute_116 = None
    view_230: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_42, [12, 64, 512]);  expand_42 = None
    bmm_20: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_229, view_230)
    view_231: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_20, [1, 12, 512, 512]);  bmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:274, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_20: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_231, 8.0);  view_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:277, code: attention_scores = attention_scores + attention_mask
    add_86: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(div_20, mul);  div_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:280, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_10: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_86, [-1], True)
    sub_32: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_86, amax_10);  add_86 = amax_10 = None
    exp_10: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_32);  sub_32 = None
    sum_11: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_21: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    alias_10: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(div_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:284, code: attention_probs = self.dropout(attention_probs)
    native_dropout_31 = torch.ops.aten.native_dropout.default(div_21, 0.1, True);  div_21 = None
    getitem_104: "f32[1, 12, 512, 512]" = native_dropout_31[0]
    getitem_105: "b8[1, 12, 512, 512]" = native_dropout_31[1];  native_dropout_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:290, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_43: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(getitem_104, [1, 12, 512, 512]);  getitem_104 = None
    view_232: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_43, [12, 512, 512]);  expand_43 = None
    expand_44: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_114, [1, 12, 512, 64]);  permute_114 = None
    view_233: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_44, [12, 512, 64]);  expand_44 = None
    bmm_21: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_232, view_233)
    view_234: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_21, [1, 12, 512, 64]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:292, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_117: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_234, [0, 2, 1, 3]);  view_234 = None
    clone_10: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:294, code: context_layer = context_layer.view(new_context_layer_shape)
    view_235: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_10, [1, 512, 768]);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:312, code: hidden_states = self.dense(hidden_states)
    view_236: "f32[512, 768]" = torch.ops.aten.view.default(view_235, [512, 768]);  view_235 = None
    permute_118: "f32[768, 768]" = torch.ops.aten.permute.default(primals_172, [1, 0]);  primals_172 = None
    addmm_63: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_173, view_236, permute_118);  primals_173 = None
    view_237: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_63, [1, 512, 768]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:313, code: hidden_states = self.dropout(hidden_states)
    native_dropout_32 = torch.ops.aten.native_dropout.default(view_237, 0.1, True);  view_237 = None
    getitem_106: "f32[1, 512, 768]" = native_dropout_32[0]
    getitem_107: "b8[1, 512, 768]" = native_dropout_32[1];  native_dropout_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:314, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_87: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_106, add_85);  getitem_106 = add_85 = None
    var_mean_21 = torch.ops.aten.var_mean.correction(add_87, [2], correction = 0, keepdim = True)
    getitem_108: "f32[1, 512, 1]" = var_mean_21[0]
    getitem_109: "f32[1, 512, 1]" = var_mean_21[1];  var_mean_21 = None
    add_88: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05);  getitem_108 = None
    rsqrt_21: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
    sub_33: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_87, getitem_109)
    mul_74: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_21);  sub_33 = None
    mul_75: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_74, primals_174);  mul_74 = None
    add_89: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_75, primals_175);  mul_75 = primals_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:379, code: hidden_states = self.dense(hidden_states)
    view_238: "f32[512, 768]" = torch.ops.aten.view.default(add_89, [512, 768])
    permute_119: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_176, [1, 0]);  primals_176 = None
    addmm_64: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_177, view_238, permute_119);  primals_177 = None
    view_239: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_64, [1, 512, 3072]);  addmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_76: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, 0.5)
    mul_77: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, 0.7071067811865476)
    erf_10: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_77);  mul_77 = None
    add_90: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_78: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_76, add_90);  mul_76 = add_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:393, code: hidden_states = self.dense(hidden_states)
    view_240: "f32[512, 3072]" = torch.ops.aten.view.default(mul_78, [512, 3072]);  mul_78 = None
    permute_120: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_178, [1, 0]);  primals_178 = None
    addmm_65: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_179, view_240, permute_120);  primals_179 = None
    view_241: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_65, [1, 512, 768]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:394, code: hidden_states = self.dropout(hidden_states)
    native_dropout_33 = torch.ops.aten.native_dropout.default(view_241, 0.1, True);  view_241 = None
    getitem_110: "f32[1, 512, 768]" = native_dropout_33[0]
    getitem_111: "b8[1, 512, 768]" = native_dropout_33[1];  native_dropout_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:395, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_91: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_110, add_89);  getitem_110 = add_89 = None
    var_mean_22 = torch.ops.aten.var_mean.correction(add_91, [2], correction = 0, keepdim = True)
    getitem_112: "f32[1, 512, 1]" = var_mean_22[0]
    getitem_113: "f32[1, 512, 1]" = var_mean_22[1];  var_mean_22 = None
    add_92: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05);  getitem_112 = None
    rsqrt_22: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
    sub_34: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_91, getitem_113)
    mul_79: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_22);  sub_34 = None
    mul_80: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_79, primals_180);  mul_79 = None
    add_93: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_80, primals_181);  mul_80 = primals_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:211, code: mixed_query_layer = self.query(hidden_states)
    view_242: "f32[512, 768]" = torch.ops.aten.view.default(add_93, [512, 768])
    permute_121: "f32[768, 768]" = torch.ops.aten.permute.default(primals_182, [1, 0]);  primals_182 = None
    addmm_66: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_183, view_242, permute_121);  primals_183 = None
    view_243: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_66, [1, 512, 768]);  addmm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:233, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_244: "f32[512, 768]" = torch.ops.aten.view.default(add_93, [512, 768])
    permute_122: "f32[768, 768]" = torch.ops.aten.permute.default(primals_184, [1, 0]);  primals_184 = None
    addmm_67: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_185, view_244, permute_122);  primals_185 = None
    view_245: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_67, [1, 512, 768]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_246: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_245, [1, 512, 12, 64]);  view_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_123: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:234, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_247: "f32[512, 768]" = torch.ops.aten.view.default(add_93, [512, 768])
    permute_124: "f32[768, 768]" = torch.ops.aten.permute.default(primals_186, [1, 0]);  primals_186 = None
    addmm_68: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_187, view_247, permute_124);  primals_187 = None
    view_248: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_68, [1, 512, 768]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_249: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_248, [1, 512, 12, 64]);  view_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_125: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_249, [0, 2, 1, 3]);  view_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_250: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_243, [1, 512, 12, 64]);  view_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_126: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:250, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_127: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_123, [0, 1, 3, 2]);  permute_123 = None
    expand_45: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_126, [1, 12, 512, 64]);  permute_126 = None
    view_251: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_45, [12, 512, 64]);  expand_45 = None
    expand_46: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_127, [1, 12, 64, 512]);  permute_127 = None
    view_252: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_46, [12, 64, 512]);  expand_46 = None
    bmm_22: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_251, view_252)
    view_253: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_22, [1, 12, 512, 512]);  bmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:274, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_22: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_253, 8.0);  view_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:277, code: attention_scores = attention_scores + attention_mask
    add_94: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(div_22, mul);  div_22 = mul = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:280, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_11: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_94, [-1], True)
    sub_35: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_94, amax_11);  add_94 = amax_11 = None
    exp_11: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_35);  sub_35 = None
    sum_12: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_23: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    alias_11: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(div_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:284, code: attention_probs = self.dropout(attention_probs)
    native_dropout_34 = torch.ops.aten.native_dropout.default(div_23, 0.1, True);  div_23 = None
    getitem_114: "f32[1, 12, 512, 512]" = native_dropout_34[0]
    getitem_115: "b8[1, 12, 512, 512]" = native_dropout_34[1];  native_dropout_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:290, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_47: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(getitem_114, [1, 12, 512, 512]);  getitem_114 = None
    view_254: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_47, [12, 512, 512]);  expand_47 = None
    expand_48: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_125, [1, 12, 512, 64]);  permute_125 = None
    view_255: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_48, [12, 512, 64]);  expand_48 = None
    bmm_23: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_254, view_255)
    view_256: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_23, [1, 12, 512, 64]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:292, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_128: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_256, [0, 2, 1, 3]);  view_256 = None
    clone_11: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:294, code: context_layer = context_layer.view(new_context_layer_shape)
    view_257: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_11, [1, 512, 768]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:312, code: hidden_states = self.dense(hidden_states)
    view_258: "f32[512, 768]" = torch.ops.aten.view.default(view_257, [512, 768]);  view_257 = None
    permute_129: "f32[768, 768]" = torch.ops.aten.permute.default(primals_188, [1, 0]);  primals_188 = None
    addmm_69: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_189, view_258, permute_129);  primals_189 = None
    view_259: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_69, [1, 512, 768]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:313, code: hidden_states = self.dropout(hidden_states)
    native_dropout_35 = torch.ops.aten.native_dropout.default(view_259, 0.1, True);  view_259 = None
    getitem_116: "f32[1, 512, 768]" = native_dropout_35[0]
    getitem_117: "b8[1, 512, 768]" = native_dropout_35[1];  native_dropout_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:314, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_95: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_116, add_93);  getitem_116 = add_93 = None
    var_mean_23 = torch.ops.aten.var_mean.correction(add_95, [2], correction = 0, keepdim = True)
    getitem_118: "f32[1, 512, 1]" = var_mean_23[0]
    getitem_119: "f32[1, 512, 1]" = var_mean_23[1];  var_mean_23 = None
    add_96: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-05);  getitem_118 = None
    rsqrt_23: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
    sub_36: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_95, getitem_119)
    mul_81: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_23);  sub_36 = None
    mul_82: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_81, primals_190);  mul_81 = None
    add_97: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_82, primals_191);  mul_82 = primals_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:379, code: hidden_states = self.dense(hidden_states)
    view_260: "f32[512, 768]" = torch.ops.aten.view.default(add_97, [512, 768])
    permute_130: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_192, [1, 0]);  primals_192 = None
    addmm_70: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_193, view_260, permute_130);  primals_193 = None
    view_261: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_70, [1, 512, 3072]);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_83: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, 0.5)
    mul_84: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476)
    erf_11: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_84);  mul_84 = None
    add_98: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_85: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_83, add_98);  mul_83 = add_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:393, code: hidden_states = self.dense(hidden_states)
    view_262: "f32[512, 3072]" = torch.ops.aten.view.default(mul_85, [512, 3072]);  mul_85 = None
    permute_131: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_194, [1, 0]);  primals_194 = None
    addmm_71: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_195, view_262, permute_131);  primals_195 = None
    view_263: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_71, [1, 512, 768]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:394, code: hidden_states = self.dropout(hidden_states)
    native_dropout_36 = torch.ops.aten.native_dropout.default(view_263, 0.1, True);  view_263 = None
    getitem_120: "f32[1, 512, 768]" = native_dropout_36[0]
    getitem_121: "b8[1, 512, 768]" = native_dropout_36[1];  native_dropout_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:395, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_99: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_120, add_97);  getitem_120 = add_97 = None
    var_mean_24 = torch.ops.aten.var_mean.correction(add_99, [2], correction = 0, keepdim = True)
    getitem_122: "f32[1, 512, 1]" = var_mean_24[0]
    getitem_123: "f32[1, 512, 1]" = var_mean_24[1];  var_mean_24 = None
    add_100: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_122, 1e-05);  getitem_122 = None
    rsqrt_24: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
    sub_37: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_99, getitem_123)
    mul_86: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_24);  sub_37 = None
    mul_87: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_86, primals_196);  mul_86 = None
    add_101: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_87, primals_197);  mul_87 = primals_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:720, code: x = self.dense(features)
    view_264: "f32[512, 768]" = torch.ops.aten.view.default(add_101, [512, 768]);  add_101 = None
    permute_132: "f32[768, 768]" = torch.ops.aten.permute.default(primals_198, [1, 0]);  primals_198 = None
    addmm_72: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_199, view_264, permute_132);  primals_199 = None
    view_265: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_72, [1, 512, 768]);  addmm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_88: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_265, 0.5)
    mul_89: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_265, 0.7071067811865476)
    erf_12: "f32[1, 512, 768]" = torch.ops.aten.erf.default(mul_89);  mul_89 = None
    add_102: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_90: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_88, add_102);  mul_88 = add_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:722, code: x = self.layer_norm(x)
    var_mean_25 = torch.ops.aten.var_mean.correction(mul_90, [2], correction = 0, keepdim = True)
    getitem_124: "f32[1, 512, 1]" = var_mean_25[0]
    getitem_125: "f32[1, 512, 1]" = var_mean_25[1];  var_mean_25 = None
    add_103: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_124, 1e-05);  getitem_124 = None
    rsqrt_25: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_103);  add_103 = None
    sub_38: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_90, getitem_125)
    mul_91: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_25);  sub_38 = None
    mul_92: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_91, primals_200);  mul_91 = None
    add_104: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_92, primals_201);  mul_92 = primals_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:725, code: x = self.decoder(x)
    view_266: "f32[512, 768]" = torch.ops.aten.view.default(add_104, [512, 768]);  add_104 = None
    permute_133: "f32[768, 32005]" = torch.ops.aten.permute.default(primals_202, [1, 0]);  primals_202 = None
    addmm_73: "f32[512, 32005]" = torch.ops.aten.addmm.default(primals_203, view_266, permute_133);  primals_203 = None
    view_267: "f32[1, 512, 32005]" = torch.ops.aten.view.default(addmm_73, [1, 512, 32005]);  addmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:1009, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    view_268: "f32[512, 32005]" = torch.ops.aten.view.default(view_267, [-1, 32005])
    view_269: "i64[512]" = torch.ops.aten.view.default(primals_206, [-1]);  primals_206 = None
    amax_12: "f32[512, 1]" = torch.ops.aten.amax.default(view_268, [1], True)
    sub_39: "f32[512, 32005]" = torch.ops.aten.sub.Tensor(view_268, amax_12);  view_268 = amax_12 = None
    exp_12: "f32[512, 32005]" = torch.ops.aten.exp.default(sub_39)
    sum_13: "f32[512, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [1], True);  exp_12 = None
    log: "f32[512, 1]" = torch.ops.aten.log.default(sum_13);  sum_13 = None
    sub_40: "f32[512, 32005]" = torch.ops.aten.sub.Tensor(sub_39, log);  sub_39 = log = None
    alias_12: "f32[512, 32005]" = torch.ops.aten.alias.default(sub_40)
    ne_1: "b8[512]" = torch.ops.aten.ne.Scalar(view_269, -100)
    scalar_tensor: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'))
    where: "i64[512]" = torch.ops.aten.where.self(ne_1, view_269, scalar_tensor);  ne_1 = scalar_tensor = None
    unsqueeze_2: "i64[512, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
    gather: "f32[512, 1]" = torch.ops.aten.gather.default(sub_40, 1, unsqueeze_2);  sub_40 = unsqueeze_2 = None
    squeeze: "f32[512]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[512]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
    ne_2: "b8[512]" = torch.ops.aten.ne.Scalar(view_269, -100)
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_1: "f32[512]" = torch.ops.aten.where.self(ne_2, neg, scalar_tensor_1);  ne_2 = neg = scalar_tensor_1 = None
    ne_3: "b8[512]" = torch.ops.aten.ne.Scalar(view_269, -100)
    sum_14: "i64[]" = torch.ops.aten.sum.default(ne_3);  ne_3 = None
    convert_element_type_3: "f32[]" = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None
    sum_15: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
    div_24: "f32[]" = torch.ops.aten.div.Tensor(sum_15, convert_element_type_3);  sum_15 = None
    div_25: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_3);  tangents_1 = convert_element_type_3 = None
    unsqueeze_3: "i64[512, 1]" = torch.ops.aten.unsqueeze.default(view_269, 1);  view_269 = None
    ne_4: "b8[512, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_3, -100)
    scalar_tensor_2: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'))
    where_2: "i64[512, 1]" = torch.ops.aten.where.self(ne_4, unsqueeze_3, scalar_tensor_2);  ne_4 = scalar_tensor_2 = None
    full_1: "f32[512, 32005]" = torch.ops.aten.full.default([512, 32005], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    scatter: "f32[512, 32005]" = torch.ops.aten.scatter.value(full_1, 1, where_2, -1.0);  full_1 = where_2 = None
    ne_5: "b8[512, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_3, -100);  unsqueeze_3 = None
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_3: "f32[512, 1]" = torch.ops.aten.where.self(ne_5, div_25, scalar_tensor_3);  ne_5 = div_25 = scalar_tensor_3 = None
    mul_93: "f32[512, 32005]" = torch.ops.aten.mul.Tensor(scatter, where_3);  scatter = where_3 = None
    alias_13: "f32[512, 32005]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    exp_13: "f32[512, 32005]" = torch.ops.aten.exp.default(alias_13);  alias_13 = None
    sum_16: "f32[512, 1]" = torch.ops.aten.sum.dim_IntList(mul_93, [1], True)
    mul_94: "f32[512, 32005]" = torch.ops.aten.mul.Tensor(exp_13, sum_16);  exp_13 = sum_16 = None
    sub_41: "f32[512, 32005]" = torch.ops.aten.sub.Tensor(mul_93, mul_94);  mul_93 = mul_94 = None
    view_270: "f32[1, 512, 32005]" = torch.ops.aten.view.default(sub_41, [1, 512, 32005]);  sub_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:1009, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    add_105: "f32[1, 512, 32005]" = torch.ops.aten.add.Tensor(tangents_2, view_270);  tangents_2 = view_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:725, code: x = self.decoder(x)
    view_271: "f32[512, 32005]" = torch.ops.aten.view.default(add_105, [512, 32005]);  add_105 = None
    permute_134: "f32[32005, 768]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
    mm: "f32[512, 768]" = torch.ops.aten.mm.default(view_271, permute_134);  permute_134 = None
    permute_135: "f32[32005, 512]" = torch.ops.aten.permute.default(view_271, [1, 0])
    mm_1: "f32[32005, 768]" = torch.ops.aten.mm.default(permute_135, view_266);  permute_135 = view_266 = None
    permute_136: "f32[768, 32005]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_17: "f32[1, 32005]" = torch.ops.aten.sum.dim_IntList(view_271, [0], True);  view_271 = None
    view_272: "f32[32005]" = torch.ops.aten.view.default(sum_17, [32005]);  sum_17 = None
    permute_137: "f32[32005, 768]" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
    view_273: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm, [1, 512, 768]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:722, code: x = self.layer_norm(x)
    sub_42: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_90, getitem_125);  mul_90 = getitem_125 = None
    mul_95: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_25);  sub_42 = None
    mul_96: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_273, primals_200);  primals_200 = None
    mul_97: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_96, 768)
    sum_18: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_96, [2], True)
    mul_98: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_96, mul_95);  mul_96 = None
    sum_19: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_98, [2], True);  mul_98 = None
    mul_99: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_95, sum_19);  sum_19 = None
    sub_43: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_97, sum_18);  mul_97 = sum_18 = None
    sub_44: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_43, mul_99);  sub_43 = mul_99 = None
    div_26: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 768);  rsqrt_25 = None
    mul_100: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_26, sub_44);  div_26 = sub_44 = None
    mul_101: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_273, mul_95);  mul_95 = None
    sum_20: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_101, [0, 1]);  mul_101 = None
    sum_21: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_273, [0, 1]);  view_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_102: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_265, 0.7071067811865476)
    erf_13: "f32[1, 512, 768]" = torch.ops.aten.erf.default(mul_102);  mul_102 = None
    add_106: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_103: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_106, 0.5);  add_106 = None
    mul_104: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_265, view_265)
    mul_105: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_104, -0.5);  mul_104 = None
    exp_14: "f32[1, 512, 768]" = torch.ops.aten.exp.default(mul_105);  mul_105 = None
    mul_106: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
    mul_107: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_265, mul_106);  view_265 = mul_106 = None
    add_107: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_103, mul_107);  mul_103 = mul_107 = None
    mul_108: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_100, add_107);  mul_100 = add_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:720, code: x = self.dense(features)
    view_274: "f32[512, 768]" = torch.ops.aten.view.default(mul_108, [512, 768]);  mul_108 = None
    permute_138: "f32[768, 768]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
    mm_2: "f32[512, 768]" = torch.ops.aten.mm.default(view_274, permute_138);  permute_138 = None
    permute_139: "f32[768, 512]" = torch.ops.aten.permute.default(view_274, [1, 0])
    mm_3: "f32[768, 768]" = torch.ops.aten.mm.default(permute_139, view_264);  permute_139 = view_264 = None
    permute_140: "f32[768, 768]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_22: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_274, [0], True);  view_274 = None
    view_275: "f32[768]" = torch.ops.aten.view.default(sum_22, [768]);  sum_22 = None
    permute_141: "f32[768, 768]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
    view_276: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_2, [1, 512, 768]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:395, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_45: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_99, getitem_123);  add_99 = getitem_123 = None
    mul_109: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_24);  sub_45 = None
    mul_110: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_276, primals_196);  primals_196 = None
    mul_111: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_110, 768)
    sum_23: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_110, [2], True)
    mul_112: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_110, mul_109);  mul_110 = None
    sum_24: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_112, [2], True);  mul_112 = None
    mul_113: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_109, sum_24);  sum_24 = None
    sub_46: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_111, sum_23);  mul_111 = sum_23 = None
    sub_47: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_46, mul_113);  sub_46 = mul_113 = None
    div_27: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None
    mul_114: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_27, sub_47);  div_27 = sub_47 = None
    mul_115: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_276, mul_109);  mul_109 = None
    sum_25: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_115, [0, 1]);  mul_115 = None
    sum_26: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_276, [0, 1]);  view_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:394, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_4: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_121, torch.float32);  getitem_121 = None
    mul_116: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 1.1111111111111112);  convert_element_type_4 = None
    mul_117: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_114, mul_116);  mul_116 = None
    clone_12: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_117, memory_format = torch.contiguous_format);  mul_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:393, code: hidden_states = self.dense(hidden_states)
    view_277: "f32[512, 768]" = torch.ops.aten.view.default(clone_12, [512, 768]);  clone_12 = None
    permute_142: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
    mm_4: "f32[512, 3072]" = torch.ops.aten.mm.default(view_277, permute_142);  permute_142 = None
    permute_143: "f32[768, 512]" = torch.ops.aten.permute.default(view_277, [1, 0])
    mm_5: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_143, view_262);  permute_143 = view_262 = None
    permute_144: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_27: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_277, [0], True);  view_277 = None
    view_278: "f32[768]" = torch.ops.aten.view.default(sum_27, [768]);  sum_27 = None
    permute_145: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
    view_279: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_4, [1, 512, 3072]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_118: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476)
    erf_14: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_118);  mul_118 = None
    add_108: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_119: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_108, 0.5);  add_108 = None
    mul_120: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, view_261)
    mul_121: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_120, -0.5);  mul_120 = None
    exp_15: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_121);  mul_121 = None
    mul_122: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_15, 0.3989422804014327);  exp_15 = None
    mul_123: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, mul_122);  view_261 = mul_122 = None
    add_109: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_119, mul_123);  mul_119 = mul_123 = None
    mul_124: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_279, add_109);  view_279 = add_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:379, code: hidden_states = self.dense(hidden_states)
    view_280: "f32[512, 3072]" = torch.ops.aten.view.default(mul_124, [512, 3072]);  mul_124 = None
    permute_146: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
    mm_6: "f32[512, 768]" = torch.ops.aten.mm.default(view_280, permute_146);  permute_146 = None
    permute_147: "f32[3072, 512]" = torch.ops.aten.permute.default(view_280, [1, 0])
    mm_7: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_147, view_260);  permute_147 = view_260 = None
    permute_148: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_28: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_280, [0], True);  view_280 = None
    view_281: "f32[3072]" = torch.ops.aten.view.default(sum_28, [3072]);  sum_28 = None
    permute_149: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_148, [1, 0]);  permute_148 = None
    view_282: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_6, [1, 512, 768]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:379, code: hidden_states = self.dense(hidden_states)
    add_110: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_114, view_282);  mul_114 = view_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:314, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_48: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_95, getitem_119);  add_95 = getitem_119 = None
    mul_125: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_23);  sub_48 = None
    mul_126: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_110, primals_190);  primals_190 = None
    mul_127: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_126, 768)
    sum_29: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_126, [2], True)
    mul_128: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_126, mul_125);  mul_126 = None
    sum_30: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_128, [2], True);  mul_128 = None
    mul_129: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_125, sum_30);  sum_30 = None
    sub_49: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_127, sum_29);  mul_127 = sum_29 = None
    sub_50: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_49, mul_129);  sub_49 = mul_129 = None
    div_28: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
    mul_130: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_28, sub_50);  div_28 = sub_50 = None
    mul_131: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_110, mul_125);  mul_125 = None
    sum_31: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_131, [0, 1]);  mul_131 = None
    sum_32: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_110, [0, 1]);  add_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:313, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_5: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_117, torch.float32);  getitem_117 = None
    mul_132: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 1.1111111111111112);  convert_element_type_5 = None
    mul_133: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_130, mul_132);  mul_132 = None
    clone_13: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_133, memory_format = torch.contiguous_format);  mul_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:312, code: hidden_states = self.dense(hidden_states)
    view_283: "f32[512, 768]" = torch.ops.aten.view.default(clone_13, [512, 768]);  clone_13 = None
    permute_150: "f32[768, 768]" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
    mm_8: "f32[512, 768]" = torch.ops.aten.mm.default(view_283, permute_150);  permute_150 = None
    permute_151: "f32[768, 512]" = torch.ops.aten.permute.default(view_283, [1, 0])
    mm_9: "f32[768, 768]" = torch.ops.aten.mm.default(permute_151, view_258);  permute_151 = view_258 = None
    permute_152: "f32[768, 768]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_33: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_283, [0], True);  view_283 = None
    view_284: "f32[768]" = torch.ops.aten.view.default(sum_33, [768]);  sum_33 = None
    permute_153: "f32[768, 768]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
    view_285: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_8, [1, 512, 768]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:294, code: context_layer = context_layer.view(new_context_layer_shape)
    view_286: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_285, [1, 512, 12, 64]);  view_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:292, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_154: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_286, [0, 2, 1, 3]);  view_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:290, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_287: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_154, [12, 512, 64]);  permute_154 = None
    permute_155: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None
    bmm_24: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_155, view_287);  permute_155 = None
    permute_156: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_255, [0, 2, 1]);  view_255 = None
    bmm_25: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_287, permute_156);  view_287 = permute_156 = None
    view_288: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_24, [1, 12, 512, 64]);  bmm_24 = None
    view_289: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_25, [1, 12, 512, 512]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:284, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_6: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_115, torch.float32);  getitem_115 = None
    mul_134: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 1.1111111111111112);  convert_element_type_6 = None
    mul_135: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_289, mul_134);  view_289 = mul_134 = None
    clone_14: "f32[1, 12, 512, 512]" = torch.ops.aten.clone.default(mul_135, memory_format = torch.contiguous_format);  mul_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:280, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_14: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    mul_136: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(clone_14, alias_14);  clone_14 = None
    sum_34: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_136, [-1], True)
    mul_137: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_14, sum_34);  alias_14 = sum_34 = None
    sub_51: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_136, mul_137);  mul_136 = mul_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:274, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_29: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_51, 8.0);  sub_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:250, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_290: "f32[12, 512, 512]" = torch.ops.aten.view.default(div_29, [12, 512, 512]);  div_29 = None
    permute_157: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_251, [0, 2, 1]);  view_251 = None
    bmm_26: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_157, view_290);  permute_157 = None
    permute_158: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_252, [0, 2, 1]);  view_252 = None
    bmm_27: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_290, permute_158);  view_290 = permute_158 = None
    view_291: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_26, [1, 12, 64, 512]);  bmm_26 = None
    view_292: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_27, [1, 12, 512, 64]);  bmm_27 = None
    permute_159: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_291, [0, 1, 3, 2]);  view_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_160: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_292, [0, 2, 1, 3]);  view_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    clone_15: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_160, memory_format = torch.contiguous_format);  permute_160 = None
    view_293: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_15, [1, 512, 768]);  clone_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_161: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_288, [0, 2, 1, 3]);  view_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    clone_16: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
    view_294: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_16, [1, 512, 768]);  clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:234, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_295: "f32[512, 768]" = torch.ops.aten.view.default(view_294, [512, 768]);  view_294 = None
    permute_162: "f32[768, 768]" = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
    mm_10: "f32[512, 768]" = torch.ops.aten.mm.default(view_295, permute_162);  permute_162 = None
    permute_163: "f32[768, 512]" = torch.ops.aten.permute.default(view_295, [1, 0])
    mm_11: "f32[768, 768]" = torch.ops.aten.mm.default(permute_163, view_247);  permute_163 = view_247 = None
    permute_164: "f32[768, 768]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_35: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_295, [0], True);  view_295 = None
    view_296: "f32[768]" = torch.ops.aten.view.default(sum_35, [768]);  sum_35 = None
    permute_165: "f32[768, 768]" = torch.ops.aten.permute.default(permute_164, [1, 0]);  permute_164 = None
    view_297: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_10, [1, 512, 768]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:234, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_111: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_130, view_297);  mul_130 = view_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_166: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(permute_159, [0, 2, 1, 3]);  permute_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_298: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_166, [1, 512, 768]);  permute_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:233, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_299: "f32[512, 768]" = torch.ops.aten.view.default(view_298, [512, 768]);  view_298 = None
    permute_167: "f32[768, 768]" = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
    mm_12: "f32[512, 768]" = torch.ops.aten.mm.default(view_299, permute_167);  permute_167 = None
    permute_168: "f32[768, 512]" = torch.ops.aten.permute.default(view_299, [1, 0])
    mm_13: "f32[768, 768]" = torch.ops.aten.mm.default(permute_168, view_244);  permute_168 = view_244 = None
    permute_169: "f32[768, 768]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_36: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_299, [0], True);  view_299 = None
    view_300: "f32[768]" = torch.ops.aten.view.default(sum_36, [768]);  sum_36 = None
    permute_170: "f32[768, 768]" = torch.ops.aten.permute.default(permute_169, [1, 0]);  permute_169 = None
    view_301: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_12, [1, 512, 768]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:233, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_112: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_111, view_301);  add_111 = view_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:211, code: mixed_query_layer = self.query(hidden_states)
    view_302: "f32[512, 768]" = torch.ops.aten.view.default(view_293, [512, 768]);  view_293 = None
    permute_171: "f32[768, 768]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
    mm_14: "f32[512, 768]" = torch.ops.aten.mm.default(view_302, permute_171);  permute_171 = None
    permute_172: "f32[768, 512]" = torch.ops.aten.permute.default(view_302, [1, 0])
    mm_15: "f32[768, 768]" = torch.ops.aten.mm.default(permute_172, view_242);  permute_172 = view_242 = None
    permute_173: "f32[768, 768]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_37: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_302, [0], True);  view_302 = None
    view_303: "f32[768]" = torch.ops.aten.view.default(sum_37, [768]);  sum_37 = None
    permute_174: "f32[768, 768]" = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
    view_304: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_14, [1, 512, 768]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:211, code: mixed_query_layer = self.query(hidden_states)
    add_113: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_112, view_304);  add_112 = view_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:395, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_52: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_91, getitem_113);  add_91 = getitem_113 = None
    mul_138: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_22);  sub_52 = None
    mul_139: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_113, primals_180);  primals_180 = None
    mul_140: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_139, 768)
    sum_38: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_139, [2], True)
    mul_141: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_139, mul_138);  mul_139 = None
    sum_39: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_141, [2], True);  mul_141 = None
    mul_142: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_138, sum_39);  sum_39 = None
    sub_53: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_140, sum_38);  mul_140 = sum_38 = None
    sub_54: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_53, mul_142);  sub_53 = mul_142 = None
    div_30: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
    mul_143: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_30, sub_54);  div_30 = sub_54 = None
    mul_144: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_113, mul_138);  mul_138 = None
    sum_40: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_144, [0, 1]);  mul_144 = None
    sum_41: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_113, [0, 1]);  add_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:394, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_7: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_111, torch.float32);  getitem_111 = None
    mul_145: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_7, 1.1111111111111112);  convert_element_type_7 = None
    mul_146: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_143, mul_145);  mul_145 = None
    clone_17: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_146, memory_format = torch.contiguous_format);  mul_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:393, code: hidden_states = self.dense(hidden_states)
    view_305: "f32[512, 768]" = torch.ops.aten.view.default(clone_17, [512, 768]);  clone_17 = None
    permute_175: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
    mm_16: "f32[512, 3072]" = torch.ops.aten.mm.default(view_305, permute_175);  permute_175 = None
    permute_176: "f32[768, 512]" = torch.ops.aten.permute.default(view_305, [1, 0])
    mm_17: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_176, view_240);  permute_176 = view_240 = None
    permute_177: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_42: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_305, [0], True);  view_305 = None
    view_306: "f32[768]" = torch.ops.aten.view.default(sum_42, [768]);  sum_42 = None
    permute_178: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_177, [1, 0]);  permute_177 = None
    view_307: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_16, [1, 512, 3072]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_147: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, 0.7071067811865476)
    erf_15: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_147);  mul_147 = None
    add_114: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_148: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_114, 0.5);  add_114 = None
    mul_149: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, view_239)
    mul_150: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_149, -0.5);  mul_149 = None
    exp_16: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_150);  mul_150 = None
    mul_151: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
    mul_152: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, mul_151);  view_239 = mul_151 = None
    add_115: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_148, mul_152);  mul_148 = mul_152 = None
    mul_153: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_307, add_115);  view_307 = add_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:379, code: hidden_states = self.dense(hidden_states)
    view_308: "f32[512, 3072]" = torch.ops.aten.view.default(mul_153, [512, 3072]);  mul_153 = None
    permute_179: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
    mm_18: "f32[512, 768]" = torch.ops.aten.mm.default(view_308, permute_179);  permute_179 = None
    permute_180: "f32[3072, 512]" = torch.ops.aten.permute.default(view_308, [1, 0])
    mm_19: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_180, view_238);  permute_180 = view_238 = None
    permute_181: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_43: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_308, [0], True);  view_308 = None
    view_309: "f32[3072]" = torch.ops.aten.view.default(sum_43, [3072]);  sum_43 = None
    permute_182: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None
    view_310: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_18, [1, 512, 768]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:379, code: hidden_states = self.dense(hidden_states)
    add_116: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_143, view_310);  mul_143 = view_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:314, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_55: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_87, getitem_109);  add_87 = getitem_109 = None
    mul_154: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_21);  sub_55 = None
    mul_155: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_116, primals_174);  primals_174 = None
    mul_156: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_155, 768)
    sum_44: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_155, [2], True)
    mul_157: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_155, mul_154);  mul_155 = None
    sum_45: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_157, [2], True);  mul_157 = None
    mul_158: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_154, sum_45);  sum_45 = None
    sub_56: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_156, sum_44);  mul_156 = sum_44 = None
    sub_57: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_56, mul_158);  sub_56 = mul_158 = None
    div_31: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
    mul_159: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_31, sub_57);  div_31 = sub_57 = None
    mul_160: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_116, mul_154);  mul_154 = None
    sum_46: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_160, [0, 1]);  mul_160 = None
    sum_47: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_116, [0, 1]);  add_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:313, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_8: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_107, torch.float32);  getitem_107 = None
    mul_161: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.1111111111111112);  convert_element_type_8 = None
    mul_162: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_159, mul_161);  mul_161 = None
    clone_18: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_162, memory_format = torch.contiguous_format);  mul_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:312, code: hidden_states = self.dense(hidden_states)
    view_311: "f32[512, 768]" = torch.ops.aten.view.default(clone_18, [512, 768]);  clone_18 = None
    permute_183: "f32[768, 768]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
    mm_20: "f32[512, 768]" = torch.ops.aten.mm.default(view_311, permute_183);  permute_183 = None
    permute_184: "f32[768, 512]" = torch.ops.aten.permute.default(view_311, [1, 0])
    mm_21: "f32[768, 768]" = torch.ops.aten.mm.default(permute_184, view_236);  permute_184 = view_236 = None
    permute_185: "f32[768, 768]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_48: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_311, [0], True);  view_311 = None
    view_312: "f32[768]" = torch.ops.aten.view.default(sum_48, [768]);  sum_48 = None
    permute_186: "f32[768, 768]" = torch.ops.aten.permute.default(permute_185, [1, 0]);  permute_185 = None
    view_313: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_20, [1, 512, 768]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:294, code: context_layer = context_layer.view(new_context_layer_shape)
    view_314: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_313, [1, 512, 12, 64]);  view_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:292, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_187: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_314, [0, 2, 1, 3]);  view_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:290, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_315: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_187, [12, 512, 64]);  permute_187 = None
    permute_188: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_232, [0, 2, 1]);  view_232 = None
    bmm_28: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_188, view_315);  permute_188 = None
    permute_189: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_233, [0, 2, 1]);  view_233 = None
    bmm_29: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_315, permute_189);  view_315 = permute_189 = None
    view_316: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_28, [1, 12, 512, 64]);  bmm_28 = None
    view_317: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_29, [1, 12, 512, 512]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:284, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_9: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_105, torch.float32);  getitem_105 = None
    mul_163: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_9, 1.1111111111111112);  convert_element_type_9 = None
    mul_164: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_317, mul_163);  view_317 = mul_163 = None
    clone_19: "f32[1, 12, 512, 512]" = torch.ops.aten.clone.default(mul_164, memory_format = torch.contiguous_format);  mul_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:280, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_15: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    mul_165: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(clone_19, alias_15);  clone_19 = None
    sum_49: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_165, [-1], True)
    mul_166: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_15, sum_49);  alias_15 = sum_49 = None
    sub_58: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_165, mul_166);  mul_165 = mul_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:274, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_32: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_58, 8.0);  sub_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:250, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_318: "f32[12, 512, 512]" = torch.ops.aten.view.default(div_32, [12, 512, 512]);  div_32 = None
    permute_190: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_229, [0, 2, 1]);  view_229 = None
    bmm_30: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_190, view_318);  permute_190 = None
    permute_191: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_230, [0, 2, 1]);  view_230 = None
    bmm_31: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_318, permute_191);  view_318 = permute_191 = None
    view_319: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_30, [1, 12, 64, 512]);  bmm_30 = None
    view_320: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_31, [1, 12, 512, 64]);  bmm_31 = None
    permute_192: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_319, [0, 1, 3, 2]);  view_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_193: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_320, [0, 2, 1, 3]);  view_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    clone_20: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_193, memory_format = torch.contiguous_format);  permute_193 = None
    view_321: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_20, [1, 512, 768]);  clone_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_194: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_316, [0, 2, 1, 3]);  view_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    clone_21: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
    view_322: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_21, [1, 512, 768]);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:234, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_323: "f32[512, 768]" = torch.ops.aten.view.default(view_322, [512, 768]);  view_322 = None
    permute_195: "f32[768, 768]" = torch.ops.aten.permute.default(permute_113, [1, 0]);  permute_113 = None
    mm_22: "f32[512, 768]" = torch.ops.aten.mm.default(view_323, permute_195);  permute_195 = None
    permute_196: "f32[768, 512]" = torch.ops.aten.permute.default(view_323, [1, 0])
    mm_23: "f32[768, 768]" = torch.ops.aten.mm.default(permute_196, view_225);  permute_196 = view_225 = None
    permute_197: "f32[768, 768]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_50: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_323, [0], True);  view_323 = None
    view_324: "f32[768]" = torch.ops.aten.view.default(sum_50, [768]);  sum_50 = None
    permute_198: "f32[768, 768]" = torch.ops.aten.permute.default(permute_197, [1, 0]);  permute_197 = None
    view_325: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_22, [1, 512, 768]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:234, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_117: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_159, view_325);  mul_159 = view_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_199: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(permute_192, [0, 2, 1, 3]);  permute_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_326: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_199, [1, 512, 768]);  permute_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:233, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_327: "f32[512, 768]" = torch.ops.aten.view.default(view_326, [512, 768]);  view_326 = None
    permute_200: "f32[768, 768]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
    mm_24: "f32[512, 768]" = torch.ops.aten.mm.default(view_327, permute_200);  permute_200 = None
    permute_201: "f32[768, 512]" = torch.ops.aten.permute.default(view_327, [1, 0])
    mm_25: "f32[768, 768]" = torch.ops.aten.mm.default(permute_201, view_222);  permute_201 = view_222 = None
    permute_202: "f32[768, 768]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_51: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_327, [0], True);  view_327 = None
    view_328: "f32[768]" = torch.ops.aten.view.default(sum_51, [768]);  sum_51 = None
    permute_203: "f32[768, 768]" = torch.ops.aten.permute.default(permute_202, [1, 0]);  permute_202 = None
    view_329: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_24, [1, 512, 768]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:233, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_118: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_117, view_329);  add_117 = view_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:211, code: mixed_query_layer = self.query(hidden_states)
    view_330: "f32[512, 768]" = torch.ops.aten.view.default(view_321, [512, 768]);  view_321 = None
    permute_204: "f32[768, 768]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
    mm_26: "f32[512, 768]" = torch.ops.aten.mm.default(view_330, permute_204);  permute_204 = None
    permute_205: "f32[768, 512]" = torch.ops.aten.permute.default(view_330, [1, 0])
    mm_27: "f32[768, 768]" = torch.ops.aten.mm.default(permute_205, view_220);  permute_205 = view_220 = None
    permute_206: "f32[768, 768]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_52: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_330, [0], True);  view_330 = None
    view_331: "f32[768]" = torch.ops.aten.view.default(sum_52, [768]);  sum_52 = None
    permute_207: "f32[768, 768]" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None
    view_332: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_26, [1, 512, 768]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:211, code: mixed_query_layer = self.query(hidden_states)
    add_119: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_118, view_332);  add_118 = view_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:395, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_59: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_83, getitem_103);  add_83 = getitem_103 = None
    mul_167: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_20);  sub_59 = None
    mul_168: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_119, primals_164);  primals_164 = None
    mul_169: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_168, 768)
    sum_53: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_168, [2], True)
    mul_170: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_168, mul_167);  mul_168 = None
    sum_54: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_170, [2], True);  mul_170 = None
    mul_171: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_167, sum_54);  sum_54 = None
    sub_60: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_169, sum_53);  mul_169 = sum_53 = None
    sub_61: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_60, mul_171);  sub_60 = mul_171 = None
    div_33: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
    mul_172: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_33, sub_61);  div_33 = sub_61 = None
    mul_173: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_119, mul_167);  mul_167 = None
    sum_55: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_173, [0, 1]);  mul_173 = None
    sum_56: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_119, [0, 1]);  add_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:394, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_10: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_101, torch.float32);  getitem_101 = None
    mul_174: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.1111111111111112);  convert_element_type_10 = None
    mul_175: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_172, mul_174);  mul_174 = None
    clone_22: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_175, memory_format = torch.contiguous_format);  mul_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:393, code: hidden_states = self.dense(hidden_states)
    view_333: "f32[512, 768]" = torch.ops.aten.view.default(clone_22, [512, 768]);  clone_22 = None
    permute_208: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
    mm_28: "f32[512, 3072]" = torch.ops.aten.mm.default(view_333, permute_208);  permute_208 = None
    permute_209: "f32[768, 512]" = torch.ops.aten.permute.default(view_333, [1, 0])
    mm_29: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_209, view_218);  permute_209 = view_218 = None
    permute_210: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_57: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_333, [0], True);  view_333 = None
    view_334: "f32[768]" = torch.ops.aten.view.default(sum_57, [768]);  sum_57 = None
    permute_211: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_210, [1, 0]);  permute_210 = None
    view_335: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_28, [1, 512, 3072]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_176: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476)
    erf_16: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_176);  mul_176 = None
    add_120: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_177: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_120, 0.5);  add_120 = None
    mul_178: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, view_217)
    mul_179: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_178, -0.5);  mul_178 = None
    exp_17: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_179);  mul_179 = None
    mul_180: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_17, 0.3989422804014327);  exp_17 = None
    mul_181: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, mul_180);  view_217 = mul_180 = None
    add_121: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_177, mul_181);  mul_177 = mul_181 = None
    mul_182: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_335, add_121);  view_335 = add_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:379, code: hidden_states = self.dense(hidden_states)
    view_336: "f32[512, 3072]" = torch.ops.aten.view.default(mul_182, [512, 3072]);  mul_182 = None
    permute_212: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
    mm_30: "f32[512, 768]" = torch.ops.aten.mm.default(view_336, permute_212);  permute_212 = None
    permute_213: "f32[3072, 512]" = torch.ops.aten.permute.default(view_336, [1, 0])
    mm_31: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_213, view_216);  permute_213 = view_216 = None
    permute_214: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_58: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_336, [0], True);  view_336 = None
    view_337: "f32[3072]" = torch.ops.aten.view.default(sum_58, [3072]);  sum_58 = None
    permute_215: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_214, [1, 0]);  permute_214 = None
    view_338: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_30, [1, 512, 768]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:379, code: hidden_states = self.dense(hidden_states)
    add_122: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_172, view_338);  mul_172 = view_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:314, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_62: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_79, getitem_99);  add_79 = getitem_99 = None
    mul_183: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_19);  sub_62 = None
    mul_184: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_122, primals_158);  primals_158 = None
    mul_185: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_184, 768)
    sum_59: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_184, [2], True)
    mul_186: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_184, mul_183);  mul_184 = None
    sum_60: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_186, [2], True);  mul_186 = None
    mul_187: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_183, sum_60);  sum_60 = None
    sub_63: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_185, sum_59);  mul_185 = sum_59 = None
    sub_64: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_63, mul_187);  sub_63 = mul_187 = None
    div_34: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
    mul_188: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_34, sub_64);  div_34 = sub_64 = None
    mul_189: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_122, mul_183);  mul_183 = None
    sum_61: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_189, [0, 1]);  mul_189 = None
    sum_62: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_122, [0, 1]);  add_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:313, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_11: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_97, torch.float32);  getitem_97 = None
    mul_190: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 1.1111111111111112);  convert_element_type_11 = None
    mul_191: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_188, mul_190);  mul_190 = None
    clone_23: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_191, memory_format = torch.contiguous_format);  mul_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:312, code: hidden_states = self.dense(hidden_states)
    view_339: "f32[512, 768]" = torch.ops.aten.view.default(clone_23, [512, 768]);  clone_23 = None
    permute_216: "f32[768, 768]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
    mm_32: "f32[512, 768]" = torch.ops.aten.mm.default(view_339, permute_216);  permute_216 = None
    permute_217: "f32[768, 512]" = torch.ops.aten.permute.default(view_339, [1, 0])
    mm_33: "f32[768, 768]" = torch.ops.aten.mm.default(permute_217, view_214);  permute_217 = view_214 = None
    permute_218: "f32[768, 768]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_63: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_339, [0], True);  view_339 = None
    view_340: "f32[768]" = torch.ops.aten.view.default(sum_63, [768]);  sum_63 = None
    permute_219: "f32[768, 768]" = torch.ops.aten.permute.default(permute_218, [1, 0]);  permute_218 = None
    view_341: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_32, [1, 512, 768]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:294, code: context_layer = context_layer.view(new_context_layer_shape)
    view_342: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_341, [1, 512, 12, 64]);  view_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:292, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_220: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_342, [0, 2, 1, 3]);  view_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:290, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_343: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_220, [12, 512, 64]);  permute_220 = None
    permute_221: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None
    bmm_32: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_221, view_343);  permute_221 = None
    permute_222: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_211, [0, 2, 1]);  view_211 = None
    bmm_33: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_343, permute_222);  view_343 = permute_222 = None
    view_344: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_32, [1, 12, 512, 64]);  bmm_32 = None
    view_345: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_33, [1, 12, 512, 512]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:284, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_12: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_95, torch.float32);  getitem_95 = None
    mul_192: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_12, 1.1111111111111112);  convert_element_type_12 = None
    mul_193: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_345, mul_192);  view_345 = mul_192 = None
    clone_24: "f32[1, 12, 512, 512]" = torch.ops.aten.clone.default(mul_193, memory_format = torch.contiguous_format);  mul_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:280, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_16: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    mul_194: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(clone_24, alias_16);  clone_24 = None
    sum_64: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_194, [-1], True)
    mul_195: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_16, sum_64);  alias_16 = sum_64 = None
    sub_65: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_194, mul_195);  mul_194 = mul_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:274, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_35: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_65, 8.0);  sub_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:250, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_346: "f32[12, 512, 512]" = torch.ops.aten.view.default(div_35, [12, 512, 512]);  div_35 = None
    permute_223: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_207, [0, 2, 1]);  view_207 = None
    bmm_34: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_223, view_346);  permute_223 = None
    permute_224: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_208, [0, 2, 1]);  view_208 = None
    bmm_35: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_346, permute_224);  view_346 = permute_224 = None
    view_347: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_34, [1, 12, 64, 512]);  bmm_34 = None
    view_348: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_35, [1, 12, 512, 64]);  bmm_35 = None
    permute_225: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_347, [0, 1, 3, 2]);  view_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_226: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_348, [0, 2, 1, 3]);  view_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    clone_25: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_226, memory_format = torch.contiguous_format);  permute_226 = None
    view_349: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_25, [1, 512, 768]);  clone_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_227: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_344, [0, 2, 1, 3]);  view_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    clone_26: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
    view_350: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_26, [1, 512, 768]);  clone_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:234, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_351: "f32[512, 768]" = torch.ops.aten.view.default(view_350, [512, 768]);  view_350 = None
    permute_228: "f32[768, 768]" = torch.ops.aten.permute.default(permute_102, [1, 0]);  permute_102 = None
    mm_34: "f32[512, 768]" = torch.ops.aten.mm.default(view_351, permute_228);  permute_228 = None
    permute_229: "f32[768, 512]" = torch.ops.aten.permute.default(view_351, [1, 0])
    mm_35: "f32[768, 768]" = torch.ops.aten.mm.default(permute_229, view_203);  permute_229 = view_203 = None
    permute_230: "f32[768, 768]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_65: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_351, [0], True);  view_351 = None
    view_352: "f32[768]" = torch.ops.aten.view.default(sum_65, [768]);  sum_65 = None
    permute_231: "f32[768, 768]" = torch.ops.aten.permute.default(permute_230, [1, 0]);  permute_230 = None
    view_353: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_34, [1, 512, 768]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:234, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_123: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_188, view_353);  mul_188 = view_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_232: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(permute_225, [0, 2, 1, 3]);  permute_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_354: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_232, [1, 512, 768]);  permute_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:233, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_355: "f32[512, 768]" = torch.ops.aten.view.default(view_354, [512, 768]);  view_354 = None
    permute_233: "f32[768, 768]" = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
    mm_36: "f32[512, 768]" = torch.ops.aten.mm.default(view_355, permute_233);  permute_233 = None
    permute_234: "f32[768, 512]" = torch.ops.aten.permute.default(view_355, [1, 0])
    mm_37: "f32[768, 768]" = torch.ops.aten.mm.default(permute_234, view_200);  permute_234 = view_200 = None
    permute_235: "f32[768, 768]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_66: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_355, [0], True);  view_355 = None
    view_356: "f32[768]" = torch.ops.aten.view.default(sum_66, [768]);  sum_66 = None
    permute_236: "f32[768, 768]" = torch.ops.aten.permute.default(permute_235, [1, 0]);  permute_235 = None
    view_357: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_36, [1, 512, 768]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:233, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_124: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_123, view_357);  add_123 = view_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:211, code: mixed_query_layer = self.query(hidden_states)
    view_358: "f32[512, 768]" = torch.ops.aten.view.default(view_349, [512, 768]);  view_349 = None
    permute_237: "f32[768, 768]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
    mm_38: "f32[512, 768]" = torch.ops.aten.mm.default(view_358, permute_237);  permute_237 = None
    permute_238: "f32[768, 512]" = torch.ops.aten.permute.default(view_358, [1, 0])
    mm_39: "f32[768, 768]" = torch.ops.aten.mm.default(permute_238, view_198);  permute_238 = view_198 = None
    permute_239: "f32[768, 768]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_67: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_358, [0], True);  view_358 = None
    view_359: "f32[768]" = torch.ops.aten.view.default(sum_67, [768]);  sum_67 = None
    permute_240: "f32[768, 768]" = torch.ops.aten.permute.default(permute_239, [1, 0]);  permute_239 = None
    view_360: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_38, [1, 512, 768]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:211, code: mixed_query_layer = self.query(hidden_states)
    add_125: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_124, view_360);  add_124 = view_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:395, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_66: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_75, getitem_93);  add_75 = getitem_93 = None
    mul_196: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_18);  sub_66 = None
    mul_197: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_125, primals_148);  primals_148 = None
    mul_198: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_197, 768)
    sum_68: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_197, [2], True)
    mul_199: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_197, mul_196);  mul_197 = None
    sum_69: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_199, [2], True);  mul_199 = None
    mul_200: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_196, sum_69);  sum_69 = None
    sub_67: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_198, sum_68);  mul_198 = sum_68 = None
    sub_68: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_67, mul_200);  sub_67 = mul_200 = None
    div_36: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
    mul_201: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_36, sub_68);  div_36 = sub_68 = None
    mul_202: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_125, mul_196);  mul_196 = None
    sum_70: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_202, [0, 1]);  mul_202 = None
    sum_71: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_125, [0, 1]);  add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:394, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_13: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_91, torch.float32);  getitem_91 = None
    mul_203: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
    mul_204: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_201, mul_203);  mul_203 = None
    clone_27: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_204, memory_format = torch.contiguous_format);  mul_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:393, code: hidden_states = self.dense(hidden_states)
    view_361: "f32[512, 768]" = torch.ops.aten.view.default(clone_27, [512, 768]);  clone_27 = None
    permute_241: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
    mm_40: "f32[512, 3072]" = torch.ops.aten.mm.default(view_361, permute_241);  permute_241 = None
    permute_242: "f32[768, 512]" = torch.ops.aten.permute.default(view_361, [1, 0])
    mm_41: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_242, view_196);  permute_242 = view_196 = None
    permute_243: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_72: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_361, [0], True);  view_361 = None
    view_362: "f32[768]" = torch.ops.aten.view.default(sum_72, [768]);  sum_72 = None
    permute_244: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
    view_363: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_40, [1, 512, 3072]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_205: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476)
    erf_17: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_205);  mul_205 = None
    add_126: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_206: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_126, 0.5);  add_126 = None
    mul_207: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, view_195)
    mul_208: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_207, -0.5);  mul_207 = None
    exp_18: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_208);  mul_208 = None
    mul_209: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
    mul_210: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, mul_209);  view_195 = mul_209 = None
    add_127: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_206, mul_210);  mul_206 = mul_210 = None
    mul_211: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_363, add_127);  view_363 = add_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:379, code: hidden_states = self.dense(hidden_states)
    view_364: "f32[512, 3072]" = torch.ops.aten.view.default(mul_211, [512, 3072]);  mul_211 = None
    permute_245: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
    mm_42: "f32[512, 768]" = torch.ops.aten.mm.default(view_364, permute_245);  permute_245 = None
    permute_246: "f32[3072, 512]" = torch.ops.aten.permute.default(view_364, [1, 0])
    mm_43: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_246, view_194);  permute_246 = view_194 = None
    permute_247: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_73: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_364, [0], True);  view_364 = None
    view_365: "f32[3072]" = torch.ops.aten.view.default(sum_73, [3072]);  sum_73 = None
    permute_248: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_247, [1, 0]);  permute_247 = None
    view_366: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_42, [1, 512, 768]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:379, code: hidden_states = self.dense(hidden_states)
    add_128: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_201, view_366);  mul_201 = view_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:314, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_69: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_71, getitem_89);  add_71 = getitem_89 = None
    mul_212: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_17);  sub_69 = None
    mul_213: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_128, primals_142);  primals_142 = None
    mul_214: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_213, 768)
    sum_74: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_213, [2], True)
    mul_215: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_213, mul_212);  mul_213 = None
    sum_75: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_215, [2], True);  mul_215 = None
    mul_216: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_212, sum_75);  sum_75 = None
    sub_70: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_214, sum_74);  mul_214 = sum_74 = None
    sub_71: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_70, mul_216);  sub_70 = mul_216 = None
    div_37: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
    mul_217: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_37, sub_71);  div_37 = sub_71 = None
    mul_218: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_128, mul_212);  mul_212 = None
    sum_76: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_218, [0, 1]);  mul_218 = None
    sum_77: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_128, [0, 1]);  add_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:313, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_14: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_87, torch.float32);  getitem_87 = None
    mul_219: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 1.1111111111111112);  convert_element_type_14 = None
    mul_220: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_217, mul_219);  mul_219 = None
    clone_28: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_220, memory_format = torch.contiguous_format);  mul_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:312, code: hidden_states = self.dense(hidden_states)
    view_367: "f32[512, 768]" = torch.ops.aten.view.default(clone_28, [512, 768]);  clone_28 = None
    permute_249: "f32[768, 768]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
    mm_44: "f32[512, 768]" = torch.ops.aten.mm.default(view_367, permute_249);  permute_249 = None
    permute_250: "f32[768, 512]" = torch.ops.aten.permute.default(view_367, [1, 0])
    mm_45: "f32[768, 768]" = torch.ops.aten.mm.default(permute_250, view_192);  permute_250 = view_192 = None
    permute_251: "f32[768, 768]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_78: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_367, [0], True);  view_367 = None
    view_368: "f32[768]" = torch.ops.aten.view.default(sum_78, [768]);  sum_78 = None
    permute_252: "f32[768, 768]" = torch.ops.aten.permute.default(permute_251, [1, 0]);  permute_251 = None
    view_369: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_44, [1, 512, 768]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:294, code: context_layer = context_layer.view(new_context_layer_shape)
    view_370: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_369, [1, 512, 12, 64]);  view_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:292, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_253: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_370, [0, 2, 1, 3]);  view_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:290, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_371: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_253, [12, 512, 64]);  permute_253 = None
    permute_254: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None
    bmm_36: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_254, view_371);  permute_254 = None
    permute_255: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_189, [0, 2, 1]);  view_189 = None
    bmm_37: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_371, permute_255);  view_371 = permute_255 = None
    view_372: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_36, [1, 12, 512, 64]);  bmm_36 = None
    view_373: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_37, [1, 12, 512, 512]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:284, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_15: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_85, torch.float32);  getitem_85 = None
    mul_221: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_15, 1.1111111111111112);  convert_element_type_15 = None
    mul_222: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_373, mul_221);  view_373 = mul_221 = None
    clone_29: "f32[1, 12, 512, 512]" = torch.ops.aten.clone.default(mul_222, memory_format = torch.contiguous_format);  mul_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:280, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_17: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    mul_223: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(clone_29, alias_17);  clone_29 = None
    sum_79: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_223, [-1], True)
    mul_224: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_17, sum_79);  alias_17 = sum_79 = None
    sub_72: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_223, mul_224);  mul_223 = mul_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:274, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_38: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_72, 8.0);  sub_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:250, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_374: "f32[12, 512, 512]" = torch.ops.aten.view.default(div_38, [12, 512, 512]);  div_38 = None
    permute_256: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_185, [0, 2, 1]);  view_185 = None
    bmm_38: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_256, view_374);  permute_256 = None
    permute_257: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1]);  view_186 = None
    bmm_39: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_374, permute_257);  view_374 = permute_257 = None
    view_375: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_38, [1, 12, 64, 512]);  bmm_38 = None
    view_376: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_39, [1, 12, 512, 64]);  bmm_39 = None
    permute_258: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_375, [0, 1, 3, 2]);  view_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_259: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_376, [0, 2, 1, 3]);  view_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    clone_30: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_259, memory_format = torch.contiguous_format);  permute_259 = None
    view_377: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_30, [1, 512, 768]);  clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_260: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_372, [0, 2, 1, 3]);  view_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    clone_31: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None
    view_378: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_31, [1, 512, 768]);  clone_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:234, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_379: "f32[512, 768]" = torch.ops.aten.view.default(view_378, [512, 768]);  view_378 = None
    permute_261: "f32[768, 768]" = torch.ops.aten.permute.default(permute_91, [1, 0]);  permute_91 = None
    mm_46: "f32[512, 768]" = torch.ops.aten.mm.default(view_379, permute_261);  permute_261 = None
    permute_262: "f32[768, 512]" = torch.ops.aten.permute.default(view_379, [1, 0])
    mm_47: "f32[768, 768]" = torch.ops.aten.mm.default(permute_262, view_181);  permute_262 = view_181 = None
    permute_263: "f32[768, 768]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_80: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_379, [0], True);  view_379 = None
    view_380: "f32[768]" = torch.ops.aten.view.default(sum_80, [768]);  sum_80 = None
    permute_264: "f32[768, 768]" = torch.ops.aten.permute.default(permute_263, [1, 0]);  permute_263 = None
    view_381: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_46, [1, 512, 768]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:234, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_129: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_217, view_381);  mul_217 = view_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_265: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(permute_258, [0, 2, 1, 3]);  permute_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_382: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_265, [1, 512, 768]);  permute_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:233, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_383: "f32[512, 768]" = torch.ops.aten.view.default(view_382, [512, 768]);  view_382 = None
    permute_266: "f32[768, 768]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
    mm_48: "f32[512, 768]" = torch.ops.aten.mm.default(view_383, permute_266);  permute_266 = None
    permute_267: "f32[768, 512]" = torch.ops.aten.permute.default(view_383, [1, 0])
    mm_49: "f32[768, 768]" = torch.ops.aten.mm.default(permute_267, view_178);  permute_267 = view_178 = None
    permute_268: "f32[768, 768]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_81: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_383, [0], True);  view_383 = None
    view_384: "f32[768]" = torch.ops.aten.view.default(sum_81, [768]);  sum_81 = None
    permute_269: "f32[768, 768]" = torch.ops.aten.permute.default(permute_268, [1, 0]);  permute_268 = None
    view_385: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_48, [1, 512, 768]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:233, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_130: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_129, view_385);  add_129 = view_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:211, code: mixed_query_layer = self.query(hidden_states)
    view_386: "f32[512, 768]" = torch.ops.aten.view.default(view_377, [512, 768]);  view_377 = None
    permute_270: "f32[768, 768]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
    mm_50: "f32[512, 768]" = torch.ops.aten.mm.default(view_386, permute_270);  permute_270 = None
    permute_271: "f32[768, 512]" = torch.ops.aten.permute.default(view_386, [1, 0])
    mm_51: "f32[768, 768]" = torch.ops.aten.mm.default(permute_271, view_176);  permute_271 = view_176 = None
    permute_272: "f32[768, 768]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_82: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_386, [0], True);  view_386 = None
    view_387: "f32[768]" = torch.ops.aten.view.default(sum_82, [768]);  sum_82 = None
    permute_273: "f32[768, 768]" = torch.ops.aten.permute.default(permute_272, [1, 0]);  permute_272 = None
    view_388: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_50, [1, 512, 768]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:211, code: mixed_query_layer = self.query(hidden_states)
    add_131: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_130, view_388);  add_130 = view_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:395, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_73: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_67, getitem_83);  add_67 = getitem_83 = None
    mul_225: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_16);  sub_73 = None
    mul_226: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_131, primals_132);  primals_132 = None
    mul_227: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_226, 768)
    sum_83: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_226, [2], True)
    mul_228: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_226, mul_225);  mul_226 = None
    sum_84: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_228, [2], True);  mul_228 = None
    mul_229: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_225, sum_84);  sum_84 = None
    sub_74: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_227, sum_83);  mul_227 = sum_83 = None
    sub_75: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_74, mul_229);  sub_74 = mul_229 = None
    div_39: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
    mul_230: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_39, sub_75);  div_39 = sub_75 = None
    mul_231: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_131, mul_225);  mul_225 = None
    sum_85: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_231, [0, 1]);  mul_231 = None
    sum_86: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_131, [0, 1]);  add_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:394, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_16: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_81, torch.float32);  getitem_81 = None
    mul_232: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_16, 1.1111111111111112);  convert_element_type_16 = None
    mul_233: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_230, mul_232);  mul_232 = None
    clone_32: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_233, memory_format = torch.contiguous_format);  mul_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:393, code: hidden_states = self.dense(hidden_states)
    view_389: "f32[512, 768]" = torch.ops.aten.view.default(clone_32, [512, 768]);  clone_32 = None
    permute_274: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
    mm_52: "f32[512, 3072]" = torch.ops.aten.mm.default(view_389, permute_274);  permute_274 = None
    permute_275: "f32[768, 512]" = torch.ops.aten.permute.default(view_389, [1, 0])
    mm_53: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_275, view_174);  permute_275 = view_174 = None
    permute_276: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_87: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_389, [0], True);  view_389 = None
    view_390: "f32[768]" = torch.ops.aten.view.default(sum_87, [768]);  sum_87 = None
    permute_277: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_276, [1, 0]);  permute_276 = None
    view_391: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_52, [1, 512, 3072]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_234: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476)
    erf_18: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_234);  mul_234 = None
    add_132: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_235: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_132, 0.5);  add_132 = None
    mul_236: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, view_173)
    mul_237: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_236, -0.5);  mul_236 = None
    exp_19: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_237);  mul_237 = None
    mul_238: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
    mul_239: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, mul_238);  view_173 = mul_238 = None
    add_133: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_235, mul_239);  mul_235 = mul_239 = None
    mul_240: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_391, add_133);  view_391 = add_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:379, code: hidden_states = self.dense(hidden_states)
    view_392: "f32[512, 3072]" = torch.ops.aten.view.default(mul_240, [512, 3072]);  mul_240 = None
    permute_278: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
    mm_54: "f32[512, 768]" = torch.ops.aten.mm.default(view_392, permute_278);  permute_278 = None
    permute_279: "f32[3072, 512]" = torch.ops.aten.permute.default(view_392, [1, 0])
    mm_55: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_279, view_172);  permute_279 = view_172 = None
    permute_280: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_88: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_392, [0], True);  view_392 = None
    view_393: "f32[3072]" = torch.ops.aten.view.default(sum_88, [3072]);  sum_88 = None
    permute_281: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_280, [1, 0]);  permute_280 = None
    view_394: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_54, [1, 512, 768]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:379, code: hidden_states = self.dense(hidden_states)
    add_134: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_230, view_394);  mul_230 = view_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:314, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_76: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_63, getitem_79);  add_63 = getitem_79 = None
    mul_241: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_15);  sub_76 = None
    mul_242: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_134, primals_126);  primals_126 = None
    mul_243: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_242, 768)
    sum_89: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_242, [2], True)
    mul_244: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_242, mul_241);  mul_242 = None
    sum_90: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_244, [2], True);  mul_244 = None
    mul_245: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_241, sum_90);  sum_90 = None
    sub_77: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_243, sum_89);  mul_243 = sum_89 = None
    sub_78: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_77, mul_245);  sub_77 = mul_245 = None
    div_40: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
    mul_246: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_40, sub_78);  div_40 = sub_78 = None
    mul_247: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_134, mul_241);  mul_241 = None
    sum_91: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_247, [0, 1]);  mul_247 = None
    sum_92: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_134, [0, 1]);  add_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:313, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_17: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_77, torch.float32);  getitem_77 = None
    mul_248: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_17, 1.1111111111111112);  convert_element_type_17 = None
    mul_249: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_246, mul_248);  mul_248 = None
    clone_33: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_249, memory_format = torch.contiguous_format);  mul_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:312, code: hidden_states = self.dense(hidden_states)
    view_395: "f32[512, 768]" = torch.ops.aten.view.default(clone_33, [512, 768]);  clone_33 = None
    permute_282: "f32[768, 768]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
    mm_56: "f32[512, 768]" = torch.ops.aten.mm.default(view_395, permute_282);  permute_282 = None
    permute_283: "f32[768, 512]" = torch.ops.aten.permute.default(view_395, [1, 0])
    mm_57: "f32[768, 768]" = torch.ops.aten.mm.default(permute_283, view_170);  permute_283 = view_170 = None
    permute_284: "f32[768, 768]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_93: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_395, [0], True);  view_395 = None
    view_396: "f32[768]" = torch.ops.aten.view.default(sum_93, [768]);  sum_93 = None
    permute_285: "f32[768, 768]" = torch.ops.aten.permute.default(permute_284, [1, 0]);  permute_284 = None
    view_397: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_56, [1, 512, 768]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:294, code: context_layer = context_layer.view(new_context_layer_shape)
    view_398: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_397, [1, 512, 12, 64]);  view_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:292, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_286: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_398, [0, 2, 1, 3]);  view_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:290, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_399: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_286, [12, 512, 64]);  permute_286 = None
    permute_287: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_166, [0, 2, 1]);  view_166 = None
    bmm_40: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_287, view_399);  permute_287 = None
    permute_288: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_167, [0, 2, 1]);  view_167 = None
    bmm_41: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_399, permute_288);  view_399 = permute_288 = None
    view_400: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_40, [1, 12, 512, 64]);  bmm_40 = None
    view_401: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_41, [1, 12, 512, 512]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:284, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_18: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_75, torch.float32);  getitem_75 = None
    mul_250: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_18, 1.1111111111111112);  convert_element_type_18 = None
    mul_251: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_401, mul_250);  view_401 = mul_250 = None
    clone_34: "f32[1, 12, 512, 512]" = torch.ops.aten.clone.default(mul_251, memory_format = torch.contiguous_format);  mul_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:280, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_18: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    mul_252: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(clone_34, alias_18);  clone_34 = None
    sum_94: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_252, [-1], True)
    mul_253: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_18, sum_94);  alias_18 = sum_94 = None
    sub_79: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_252, mul_253);  mul_252 = mul_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:274, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_41: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_79, 8.0);  sub_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:250, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_402: "f32[12, 512, 512]" = torch.ops.aten.view.default(div_41, [12, 512, 512]);  div_41 = None
    permute_289: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_163, [0, 2, 1]);  view_163 = None
    bmm_42: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_289, view_402);  permute_289 = None
    permute_290: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_164, [0, 2, 1]);  view_164 = None
    bmm_43: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_402, permute_290);  view_402 = permute_290 = None
    view_403: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_42, [1, 12, 64, 512]);  bmm_42 = None
    view_404: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_43, [1, 12, 512, 64]);  bmm_43 = None
    permute_291: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_403, [0, 1, 3, 2]);  view_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_292: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_404, [0, 2, 1, 3]);  view_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    clone_35: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_292, memory_format = torch.contiguous_format);  permute_292 = None
    view_405: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_35, [1, 512, 768]);  clone_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_293: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_400, [0, 2, 1, 3]);  view_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    clone_36: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_293, memory_format = torch.contiguous_format);  permute_293 = None
    view_406: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_36, [1, 512, 768]);  clone_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:234, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_407: "f32[512, 768]" = torch.ops.aten.view.default(view_406, [512, 768]);  view_406 = None
    permute_294: "f32[768, 768]" = torch.ops.aten.permute.default(permute_80, [1, 0]);  permute_80 = None
    mm_58: "f32[512, 768]" = torch.ops.aten.mm.default(view_407, permute_294);  permute_294 = None
    permute_295: "f32[768, 512]" = torch.ops.aten.permute.default(view_407, [1, 0])
    mm_59: "f32[768, 768]" = torch.ops.aten.mm.default(permute_295, view_159);  permute_295 = view_159 = None
    permute_296: "f32[768, 768]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_95: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_407, [0], True);  view_407 = None
    view_408: "f32[768]" = torch.ops.aten.view.default(sum_95, [768]);  sum_95 = None
    permute_297: "f32[768, 768]" = torch.ops.aten.permute.default(permute_296, [1, 0]);  permute_296 = None
    view_409: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_58, [1, 512, 768]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:234, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_135: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_246, view_409);  mul_246 = view_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_298: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(permute_291, [0, 2, 1, 3]);  permute_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_410: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_298, [1, 512, 768]);  permute_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:233, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_411: "f32[512, 768]" = torch.ops.aten.view.default(view_410, [512, 768]);  view_410 = None
    permute_299: "f32[768, 768]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
    mm_60: "f32[512, 768]" = torch.ops.aten.mm.default(view_411, permute_299);  permute_299 = None
    permute_300: "f32[768, 512]" = torch.ops.aten.permute.default(view_411, [1, 0])
    mm_61: "f32[768, 768]" = torch.ops.aten.mm.default(permute_300, view_156);  permute_300 = view_156 = None
    permute_301: "f32[768, 768]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_96: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_411, [0], True);  view_411 = None
    view_412: "f32[768]" = torch.ops.aten.view.default(sum_96, [768]);  sum_96 = None
    permute_302: "f32[768, 768]" = torch.ops.aten.permute.default(permute_301, [1, 0]);  permute_301 = None
    view_413: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_60, [1, 512, 768]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:233, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_136: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_135, view_413);  add_135 = view_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:211, code: mixed_query_layer = self.query(hidden_states)
    view_414: "f32[512, 768]" = torch.ops.aten.view.default(view_405, [512, 768]);  view_405 = None
    permute_303: "f32[768, 768]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
    mm_62: "f32[512, 768]" = torch.ops.aten.mm.default(view_414, permute_303);  permute_303 = None
    permute_304: "f32[768, 512]" = torch.ops.aten.permute.default(view_414, [1, 0])
    mm_63: "f32[768, 768]" = torch.ops.aten.mm.default(permute_304, view_154);  permute_304 = view_154 = None
    permute_305: "f32[768, 768]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_97: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_414, [0], True);  view_414 = None
    view_415: "f32[768]" = torch.ops.aten.view.default(sum_97, [768]);  sum_97 = None
    permute_306: "f32[768, 768]" = torch.ops.aten.permute.default(permute_305, [1, 0]);  permute_305 = None
    view_416: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_62, [1, 512, 768]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:211, code: mixed_query_layer = self.query(hidden_states)
    add_137: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_136, view_416);  add_136 = view_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:395, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_80: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_59, getitem_73);  add_59 = getitem_73 = None
    mul_254: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_80, rsqrt_14);  sub_80 = None
    mul_255: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_137, primals_116);  primals_116 = None
    mul_256: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_255, 768)
    sum_98: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_255, [2], True)
    mul_257: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_255, mul_254);  mul_255 = None
    sum_99: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_257, [2], True);  mul_257 = None
    mul_258: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_254, sum_99);  sum_99 = None
    sub_81: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_256, sum_98);  mul_256 = sum_98 = None
    sub_82: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_81, mul_258);  sub_81 = mul_258 = None
    div_42: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
    mul_259: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_42, sub_82);  div_42 = sub_82 = None
    mul_260: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_137, mul_254);  mul_254 = None
    sum_100: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_260, [0, 1]);  mul_260 = None
    sum_101: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_137, [0, 1]);  add_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:394, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_19: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_71, torch.float32);  getitem_71 = None
    mul_261: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_19, 1.1111111111111112);  convert_element_type_19 = None
    mul_262: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_259, mul_261);  mul_261 = None
    clone_37: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_262, memory_format = torch.contiguous_format);  mul_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:393, code: hidden_states = self.dense(hidden_states)
    view_417: "f32[512, 768]" = torch.ops.aten.view.default(clone_37, [512, 768]);  clone_37 = None
    permute_307: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
    mm_64: "f32[512, 3072]" = torch.ops.aten.mm.default(view_417, permute_307);  permute_307 = None
    permute_308: "f32[768, 512]" = torch.ops.aten.permute.default(view_417, [1, 0])
    mm_65: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_308, view_152);  permute_308 = view_152 = None
    permute_309: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_102: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_417, [0], True);  view_417 = None
    view_418: "f32[768]" = torch.ops.aten.view.default(sum_102, [768]);  sum_102 = None
    permute_310: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_309, [1, 0]);  permute_309 = None
    view_419: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_64, [1, 512, 3072]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_263: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476)
    erf_19: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_263);  mul_263 = None
    add_138: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_264: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_138, 0.5);  add_138 = None
    mul_265: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, view_151)
    mul_266: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_265, -0.5);  mul_265 = None
    exp_20: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_266);  mul_266 = None
    mul_267: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
    mul_268: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, mul_267);  view_151 = mul_267 = None
    add_139: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_264, mul_268);  mul_264 = mul_268 = None
    mul_269: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_419, add_139);  view_419 = add_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:379, code: hidden_states = self.dense(hidden_states)
    view_420: "f32[512, 3072]" = torch.ops.aten.view.default(mul_269, [512, 3072]);  mul_269 = None
    permute_311: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
    mm_66: "f32[512, 768]" = torch.ops.aten.mm.default(view_420, permute_311);  permute_311 = None
    permute_312: "f32[3072, 512]" = torch.ops.aten.permute.default(view_420, [1, 0])
    mm_67: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_312, view_150);  permute_312 = view_150 = None
    permute_313: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_103: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_420, [0], True);  view_420 = None
    view_421: "f32[3072]" = torch.ops.aten.view.default(sum_103, [3072]);  sum_103 = None
    permute_314: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_313, [1, 0]);  permute_313 = None
    view_422: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_66, [1, 512, 768]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:379, code: hidden_states = self.dense(hidden_states)
    add_140: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_259, view_422);  mul_259 = view_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:314, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_83: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_55, getitem_69);  add_55 = getitem_69 = None
    mul_270: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_13);  sub_83 = None
    mul_271: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_140, primals_110);  primals_110 = None
    mul_272: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_271, 768)
    sum_104: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_271, [2], True)
    mul_273: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_271, mul_270);  mul_271 = None
    sum_105: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_273, [2], True);  mul_273 = None
    mul_274: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_270, sum_105);  sum_105 = None
    sub_84: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_272, sum_104);  mul_272 = sum_104 = None
    sub_85: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_84, mul_274);  sub_84 = mul_274 = None
    div_43: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
    mul_275: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_43, sub_85);  div_43 = sub_85 = None
    mul_276: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_140, mul_270);  mul_270 = None
    sum_106: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_276, [0, 1]);  mul_276 = None
    sum_107: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_140, [0, 1]);  add_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:313, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_20: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_67, torch.float32);  getitem_67 = None
    mul_277: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.1111111111111112);  convert_element_type_20 = None
    mul_278: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_275, mul_277);  mul_277 = None
    clone_38: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_278, memory_format = torch.contiguous_format);  mul_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:312, code: hidden_states = self.dense(hidden_states)
    view_423: "f32[512, 768]" = torch.ops.aten.view.default(clone_38, [512, 768]);  clone_38 = None
    permute_315: "f32[768, 768]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
    mm_68: "f32[512, 768]" = torch.ops.aten.mm.default(view_423, permute_315);  permute_315 = None
    permute_316: "f32[768, 512]" = torch.ops.aten.permute.default(view_423, [1, 0])
    mm_69: "f32[768, 768]" = torch.ops.aten.mm.default(permute_316, view_148);  permute_316 = view_148 = None
    permute_317: "f32[768, 768]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_108: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_423, [0], True);  view_423 = None
    view_424: "f32[768]" = torch.ops.aten.view.default(sum_108, [768]);  sum_108 = None
    permute_318: "f32[768, 768]" = torch.ops.aten.permute.default(permute_317, [1, 0]);  permute_317 = None
    view_425: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_68, [1, 512, 768]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:294, code: context_layer = context_layer.view(new_context_layer_shape)
    view_426: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_425, [1, 512, 12, 64]);  view_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:292, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_319: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_426, [0, 2, 1, 3]);  view_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:290, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_427: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_319, [12, 512, 64]);  permute_319 = None
    permute_320: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_144, [0, 2, 1]);  view_144 = None
    bmm_44: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_320, view_427);  permute_320 = None
    permute_321: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_145, [0, 2, 1]);  view_145 = None
    bmm_45: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_427, permute_321);  view_427 = permute_321 = None
    view_428: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_44, [1, 12, 512, 64]);  bmm_44 = None
    view_429: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_45, [1, 12, 512, 512]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:284, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_21: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_65, torch.float32);  getitem_65 = None
    mul_279: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 1.1111111111111112);  convert_element_type_21 = None
    mul_280: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_429, mul_279);  view_429 = mul_279 = None
    clone_39: "f32[1, 12, 512, 512]" = torch.ops.aten.clone.default(mul_280, memory_format = torch.contiguous_format);  mul_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:280, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_19: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    mul_281: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(clone_39, alias_19);  clone_39 = None
    sum_109: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_281, [-1], True)
    mul_282: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_19, sum_109);  alias_19 = sum_109 = None
    sub_86: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_281, mul_282);  mul_281 = mul_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:274, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_44: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_86, 8.0);  sub_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:250, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_430: "f32[12, 512, 512]" = torch.ops.aten.view.default(div_44, [12, 512, 512]);  div_44 = None
    permute_322: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_141, [0, 2, 1]);  view_141 = None
    bmm_46: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_322, view_430);  permute_322 = None
    permute_323: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_142, [0, 2, 1]);  view_142 = None
    bmm_47: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_430, permute_323);  view_430 = permute_323 = None
    view_431: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_46, [1, 12, 64, 512]);  bmm_46 = None
    view_432: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_47, [1, 12, 512, 64]);  bmm_47 = None
    permute_324: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_431, [0, 1, 3, 2]);  view_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_325: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_432, [0, 2, 1, 3]);  view_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    clone_40: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_325, memory_format = torch.contiguous_format);  permute_325 = None
    view_433: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_40, [1, 512, 768]);  clone_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_326: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_428, [0, 2, 1, 3]);  view_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    clone_41: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_326, memory_format = torch.contiguous_format);  permute_326 = None
    view_434: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_41, [1, 512, 768]);  clone_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:234, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_435: "f32[512, 768]" = torch.ops.aten.view.default(view_434, [512, 768]);  view_434 = None
    permute_327: "f32[768, 768]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
    mm_70: "f32[512, 768]" = torch.ops.aten.mm.default(view_435, permute_327);  permute_327 = None
    permute_328: "f32[768, 512]" = torch.ops.aten.permute.default(view_435, [1, 0])
    mm_71: "f32[768, 768]" = torch.ops.aten.mm.default(permute_328, view_137);  permute_328 = view_137 = None
    permute_329: "f32[768, 768]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_110: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_435, [0], True);  view_435 = None
    view_436: "f32[768]" = torch.ops.aten.view.default(sum_110, [768]);  sum_110 = None
    permute_330: "f32[768, 768]" = torch.ops.aten.permute.default(permute_329, [1, 0]);  permute_329 = None
    view_437: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_70, [1, 512, 768]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:234, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_141: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_275, view_437);  mul_275 = view_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_331: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(permute_324, [0, 2, 1, 3]);  permute_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_438: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_331, [1, 512, 768]);  permute_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:233, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_439: "f32[512, 768]" = torch.ops.aten.view.default(view_438, [512, 768]);  view_438 = None
    permute_332: "f32[768, 768]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
    mm_72: "f32[512, 768]" = torch.ops.aten.mm.default(view_439, permute_332);  permute_332 = None
    permute_333: "f32[768, 512]" = torch.ops.aten.permute.default(view_439, [1, 0])
    mm_73: "f32[768, 768]" = torch.ops.aten.mm.default(permute_333, view_134);  permute_333 = view_134 = None
    permute_334: "f32[768, 768]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_111: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_439, [0], True);  view_439 = None
    view_440: "f32[768]" = torch.ops.aten.view.default(sum_111, [768]);  sum_111 = None
    permute_335: "f32[768, 768]" = torch.ops.aten.permute.default(permute_334, [1, 0]);  permute_334 = None
    view_441: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_72, [1, 512, 768]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:233, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_142: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_141, view_441);  add_141 = view_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:211, code: mixed_query_layer = self.query(hidden_states)
    view_442: "f32[512, 768]" = torch.ops.aten.view.default(view_433, [512, 768]);  view_433 = None
    permute_336: "f32[768, 768]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    mm_74: "f32[512, 768]" = torch.ops.aten.mm.default(view_442, permute_336);  permute_336 = None
    permute_337: "f32[768, 512]" = torch.ops.aten.permute.default(view_442, [1, 0])
    mm_75: "f32[768, 768]" = torch.ops.aten.mm.default(permute_337, view_132);  permute_337 = view_132 = None
    permute_338: "f32[768, 768]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_112: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_442, [0], True);  view_442 = None
    view_443: "f32[768]" = torch.ops.aten.view.default(sum_112, [768]);  sum_112 = None
    permute_339: "f32[768, 768]" = torch.ops.aten.permute.default(permute_338, [1, 0]);  permute_338 = None
    view_444: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_74, [1, 512, 768]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:211, code: mixed_query_layer = self.query(hidden_states)
    add_143: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_142, view_444);  add_142 = view_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:395, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_87: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_51, getitem_63);  add_51 = getitem_63 = None
    mul_283: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_12);  sub_87 = None
    mul_284: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_143, primals_100);  primals_100 = None
    mul_285: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_284, 768)
    sum_113: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_284, [2], True)
    mul_286: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_284, mul_283);  mul_284 = None
    sum_114: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_286, [2], True);  mul_286 = None
    mul_287: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_283, sum_114);  sum_114 = None
    sub_88: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_285, sum_113);  mul_285 = sum_113 = None
    sub_89: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_88, mul_287);  sub_88 = mul_287 = None
    div_45: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
    mul_288: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_45, sub_89);  div_45 = sub_89 = None
    mul_289: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_143, mul_283);  mul_283 = None
    sum_115: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_289, [0, 1]);  mul_289 = None
    sum_116: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_143, [0, 1]);  add_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:394, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_22: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_61, torch.float32);  getitem_61 = None
    mul_290: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 1.1111111111111112);  convert_element_type_22 = None
    mul_291: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_288, mul_290);  mul_290 = None
    clone_42: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_291, memory_format = torch.contiguous_format);  mul_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:393, code: hidden_states = self.dense(hidden_states)
    view_445: "f32[512, 768]" = torch.ops.aten.view.default(clone_42, [512, 768]);  clone_42 = None
    permute_340: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    mm_76: "f32[512, 3072]" = torch.ops.aten.mm.default(view_445, permute_340);  permute_340 = None
    permute_341: "f32[768, 512]" = torch.ops.aten.permute.default(view_445, [1, 0])
    mm_77: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_341, view_130);  permute_341 = view_130 = None
    permute_342: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_117: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_445, [0], True);  view_445 = None
    view_446: "f32[768]" = torch.ops.aten.view.default(sum_117, [768]);  sum_117 = None
    permute_343: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_342, [1, 0]);  permute_342 = None
    view_447: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_76, [1, 512, 3072]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_292: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476)
    erf_20: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_292);  mul_292 = None
    add_144: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_293: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_144, 0.5);  add_144 = None
    mul_294: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, view_129)
    mul_295: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_294, -0.5);  mul_294 = None
    exp_21: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_295);  mul_295 = None
    mul_296: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
    mul_297: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, mul_296);  view_129 = mul_296 = None
    add_145: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_293, mul_297);  mul_293 = mul_297 = None
    mul_298: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_447, add_145);  view_447 = add_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:379, code: hidden_states = self.dense(hidden_states)
    view_448: "f32[512, 3072]" = torch.ops.aten.view.default(mul_298, [512, 3072]);  mul_298 = None
    permute_344: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
    mm_78: "f32[512, 768]" = torch.ops.aten.mm.default(view_448, permute_344);  permute_344 = None
    permute_345: "f32[3072, 512]" = torch.ops.aten.permute.default(view_448, [1, 0])
    mm_79: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_345, view_128);  permute_345 = view_128 = None
    permute_346: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_118: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_448, [0], True);  view_448 = None
    view_449: "f32[3072]" = torch.ops.aten.view.default(sum_118, [3072]);  sum_118 = None
    permute_347: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_346, [1, 0]);  permute_346 = None
    view_450: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_78, [1, 512, 768]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:379, code: hidden_states = self.dense(hidden_states)
    add_146: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_288, view_450);  mul_288 = view_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:314, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_90: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_47, getitem_59);  add_47 = getitem_59 = None
    mul_299: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_90, rsqrt_11);  sub_90 = None
    mul_300: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_146, primals_94);  primals_94 = None
    mul_301: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_300, 768)
    sum_119: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_300, [2], True)
    mul_302: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_300, mul_299);  mul_300 = None
    sum_120: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_302, [2], True);  mul_302 = None
    mul_303: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_299, sum_120);  sum_120 = None
    sub_91: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_301, sum_119);  mul_301 = sum_119 = None
    sub_92: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_91, mul_303);  sub_91 = mul_303 = None
    div_46: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
    mul_304: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_46, sub_92);  div_46 = sub_92 = None
    mul_305: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_146, mul_299);  mul_299 = None
    sum_121: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_305, [0, 1]);  mul_305 = None
    sum_122: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_146, [0, 1]);  add_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:313, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_23: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_57, torch.float32);  getitem_57 = None
    mul_306: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_23, 1.1111111111111112);  convert_element_type_23 = None
    mul_307: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_304, mul_306);  mul_306 = None
    clone_43: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_307, memory_format = torch.contiguous_format);  mul_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:312, code: hidden_states = self.dense(hidden_states)
    view_451: "f32[512, 768]" = torch.ops.aten.view.default(clone_43, [512, 768]);  clone_43 = None
    permute_348: "f32[768, 768]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
    mm_80: "f32[512, 768]" = torch.ops.aten.mm.default(view_451, permute_348);  permute_348 = None
    permute_349: "f32[768, 512]" = torch.ops.aten.permute.default(view_451, [1, 0])
    mm_81: "f32[768, 768]" = torch.ops.aten.mm.default(permute_349, view_126);  permute_349 = view_126 = None
    permute_350: "f32[768, 768]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_123: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_451, [0], True);  view_451 = None
    view_452: "f32[768]" = torch.ops.aten.view.default(sum_123, [768]);  sum_123 = None
    permute_351: "f32[768, 768]" = torch.ops.aten.permute.default(permute_350, [1, 0]);  permute_350 = None
    view_453: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_80, [1, 512, 768]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:294, code: context_layer = context_layer.view(new_context_layer_shape)
    view_454: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_453, [1, 512, 12, 64]);  view_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:292, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_352: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_454, [0, 2, 1, 3]);  view_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:290, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_455: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_352, [12, 512, 64]);  permute_352 = None
    permute_353: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None
    bmm_48: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_353, view_455);  permute_353 = None
    permute_354: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None
    bmm_49: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_455, permute_354);  view_455 = permute_354 = None
    view_456: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_48, [1, 12, 512, 64]);  bmm_48 = None
    view_457: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_49, [1, 12, 512, 512]);  bmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:284, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_24: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_55, torch.float32);  getitem_55 = None
    mul_308: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 1.1111111111111112);  convert_element_type_24 = None
    mul_309: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_457, mul_308);  view_457 = mul_308 = None
    clone_44: "f32[1, 12, 512, 512]" = torch.ops.aten.clone.default(mul_309, memory_format = torch.contiguous_format);  mul_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:280, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_20: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    mul_310: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(clone_44, alias_20);  clone_44 = None
    sum_124: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_310, [-1], True)
    mul_311: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_20, sum_124);  alias_20 = sum_124 = None
    sub_93: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_310, mul_311);  mul_310 = mul_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:274, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_47: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_93, 8.0);  sub_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:250, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_458: "f32[12, 512, 512]" = torch.ops.aten.view.default(div_47, [12, 512, 512]);  div_47 = None
    permute_355: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_119, [0, 2, 1]);  view_119 = None
    bmm_50: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_355, view_458);  permute_355 = None
    permute_356: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1]);  view_120 = None
    bmm_51: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_458, permute_356);  view_458 = permute_356 = None
    view_459: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_50, [1, 12, 64, 512]);  bmm_50 = None
    view_460: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_51, [1, 12, 512, 64]);  bmm_51 = None
    permute_357: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_459, [0, 1, 3, 2]);  view_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_358: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_460, [0, 2, 1, 3]);  view_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    clone_45: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_358, memory_format = torch.contiguous_format);  permute_358 = None
    view_461: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_45, [1, 512, 768]);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_359: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_456, [0, 2, 1, 3]);  view_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    clone_46: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_359, memory_format = torch.contiguous_format);  permute_359 = None
    view_462: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_46, [1, 512, 768]);  clone_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:234, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_463: "f32[512, 768]" = torch.ops.aten.view.default(view_462, [512, 768]);  view_462 = None
    permute_360: "f32[768, 768]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
    mm_82: "f32[512, 768]" = torch.ops.aten.mm.default(view_463, permute_360);  permute_360 = None
    permute_361: "f32[768, 512]" = torch.ops.aten.permute.default(view_463, [1, 0])
    mm_83: "f32[768, 768]" = torch.ops.aten.mm.default(permute_361, view_115);  permute_361 = view_115 = None
    permute_362: "f32[768, 768]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_125: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_463, [0], True);  view_463 = None
    view_464: "f32[768]" = torch.ops.aten.view.default(sum_125, [768]);  sum_125 = None
    permute_363: "f32[768, 768]" = torch.ops.aten.permute.default(permute_362, [1, 0]);  permute_362 = None
    view_465: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_82, [1, 512, 768]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:234, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_147: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_304, view_465);  mul_304 = view_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_364: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(permute_357, [0, 2, 1, 3]);  permute_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_466: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_364, [1, 512, 768]);  permute_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:233, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_467: "f32[512, 768]" = torch.ops.aten.view.default(view_466, [512, 768]);  view_466 = None
    permute_365: "f32[768, 768]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    mm_84: "f32[512, 768]" = torch.ops.aten.mm.default(view_467, permute_365);  permute_365 = None
    permute_366: "f32[768, 512]" = torch.ops.aten.permute.default(view_467, [1, 0])
    mm_85: "f32[768, 768]" = torch.ops.aten.mm.default(permute_366, view_112);  permute_366 = view_112 = None
    permute_367: "f32[768, 768]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_126: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_467, [0], True);  view_467 = None
    view_468: "f32[768]" = torch.ops.aten.view.default(sum_126, [768]);  sum_126 = None
    permute_368: "f32[768, 768]" = torch.ops.aten.permute.default(permute_367, [1, 0]);  permute_367 = None
    view_469: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_84, [1, 512, 768]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:233, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_148: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_147, view_469);  add_147 = view_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:211, code: mixed_query_layer = self.query(hidden_states)
    view_470: "f32[512, 768]" = torch.ops.aten.view.default(view_461, [512, 768]);  view_461 = None
    permute_369: "f32[768, 768]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    mm_86: "f32[512, 768]" = torch.ops.aten.mm.default(view_470, permute_369);  permute_369 = None
    permute_370: "f32[768, 512]" = torch.ops.aten.permute.default(view_470, [1, 0])
    mm_87: "f32[768, 768]" = torch.ops.aten.mm.default(permute_370, view_110);  permute_370 = view_110 = None
    permute_371: "f32[768, 768]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
    sum_127: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_470, [0], True);  view_470 = None
    view_471: "f32[768]" = torch.ops.aten.view.default(sum_127, [768]);  sum_127 = None
    permute_372: "f32[768, 768]" = torch.ops.aten.permute.default(permute_371, [1, 0]);  permute_371 = None
    view_472: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_86, [1, 512, 768]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:211, code: mixed_query_layer = self.query(hidden_states)
    add_149: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_148, view_472);  add_148 = view_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:395, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_94: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_43, getitem_53);  add_43 = getitem_53 = None
    mul_312: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_94, rsqrt_10);  sub_94 = None
    mul_313: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_149, primals_84);  primals_84 = None
    mul_314: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_313, 768)
    sum_128: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_313, [2], True)
    mul_315: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_313, mul_312);  mul_313 = None
    sum_129: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_315, [2], True);  mul_315 = None
    mul_316: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_312, sum_129);  sum_129 = None
    sub_95: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_314, sum_128);  mul_314 = sum_128 = None
    sub_96: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_95, mul_316);  sub_95 = mul_316 = None
    div_48: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
    mul_317: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_48, sub_96);  div_48 = sub_96 = None
    mul_318: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_149, mul_312);  mul_312 = None
    sum_130: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_318, [0, 1]);  mul_318 = None
    sum_131: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_149, [0, 1]);  add_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:394, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_25: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_51, torch.float32);  getitem_51 = None
    mul_319: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.1111111111111112);  convert_element_type_25 = None
    mul_320: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_317, mul_319);  mul_319 = None
    clone_47: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_320, memory_format = torch.contiguous_format);  mul_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:393, code: hidden_states = self.dense(hidden_states)
    view_473: "f32[512, 768]" = torch.ops.aten.view.default(clone_47, [512, 768]);  clone_47 = None
    permute_373: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    mm_88: "f32[512, 3072]" = torch.ops.aten.mm.default(view_473, permute_373);  permute_373 = None
    permute_374: "f32[768, 512]" = torch.ops.aten.permute.default(view_473, [1, 0])
    mm_89: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_374, view_108);  permute_374 = view_108 = None
    permute_375: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    sum_132: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_473, [0], True);  view_473 = None
    view_474: "f32[768]" = torch.ops.aten.view.default(sum_132, [768]);  sum_132 = None
    permute_376: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_375, [1, 0]);  permute_375 = None
    view_475: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_88, [1, 512, 3072]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_321: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476)
    erf_21: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_321);  mul_321 = None
    add_150: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_322: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_150, 0.5);  add_150 = None
    mul_323: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, view_107)
    mul_324: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_323, -0.5);  mul_323 = None
    exp_22: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_324);  mul_324 = None
    mul_325: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
    mul_326: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, mul_325);  view_107 = mul_325 = None
    add_151: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_322, mul_326);  mul_322 = mul_326 = None
    mul_327: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_475, add_151);  view_475 = add_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:379, code: hidden_states = self.dense(hidden_states)
    view_476: "f32[512, 3072]" = torch.ops.aten.view.default(mul_327, [512, 3072]);  mul_327 = None
    permute_377: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
    mm_90: "f32[512, 768]" = torch.ops.aten.mm.default(view_476, permute_377);  permute_377 = None
    permute_378: "f32[3072, 512]" = torch.ops.aten.permute.default(view_476, [1, 0])
    mm_91: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_378, view_106);  permute_378 = view_106 = None
    permute_379: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_133: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_476, [0], True);  view_476 = None
    view_477: "f32[3072]" = torch.ops.aten.view.default(sum_133, [3072]);  sum_133 = None
    permute_380: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_379, [1, 0]);  permute_379 = None
    view_478: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_90, [1, 512, 768]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:379, code: hidden_states = self.dense(hidden_states)
    add_152: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_317, view_478);  mul_317 = view_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:314, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_97: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_39, getitem_49);  add_39 = getitem_49 = None
    mul_328: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_97, rsqrt_9);  sub_97 = None
    mul_329: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_152, primals_78);  primals_78 = None
    mul_330: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_329, 768)
    sum_134: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_329, [2], True)
    mul_331: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_329, mul_328);  mul_329 = None
    sum_135: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_331, [2], True);  mul_331 = None
    mul_332: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_328, sum_135);  sum_135 = None
    sub_98: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_330, sum_134);  mul_330 = sum_134 = None
    sub_99: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_98, mul_332);  sub_98 = mul_332 = None
    div_49: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
    mul_333: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_49, sub_99);  div_49 = sub_99 = None
    mul_334: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_152, mul_328);  mul_328 = None
    sum_136: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_334, [0, 1]);  mul_334 = None
    sum_137: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_152, [0, 1]);  add_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:313, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_26: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_47, torch.float32);  getitem_47 = None
    mul_335: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_26, 1.1111111111111112);  convert_element_type_26 = None
    mul_336: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_333, mul_335);  mul_335 = None
    clone_48: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_336, memory_format = torch.contiguous_format);  mul_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:312, code: hidden_states = self.dense(hidden_states)
    view_479: "f32[512, 768]" = torch.ops.aten.view.default(clone_48, [512, 768]);  clone_48 = None
    permute_381: "f32[768, 768]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
    mm_92: "f32[512, 768]" = torch.ops.aten.mm.default(view_479, permute_381);  permute_381 = None
    permute_382: "f32[768, 512]" = torch.ops.aten.permute.default(view_479, [1, 0])
    mm_93: "f32[768, 768]" = torch.ops.aten.mm.default(permute_382, view_104);  permute_382 = view_104 = None
    permute_383: "f32[768, 768]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_138: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_479, [0], True);  view_479 = None
    view_480: "f32[768]" = torch.ops.aten.view.default(sum_138, [768]);  sum_138 = None
    permute_384: "f32[768, 768]" = torch.ops.aten.permute.default(permute_383, [1, 0]);  permute_383 = None
    view_481: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_92, [1, 512, 768]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:294, code: context_layer = context_layer.view(new_context_layer_shape)
    view_482: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_481, [1, 512, 12, 64]);  view_481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:292, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_385: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_482, [0, 2, 1, 3]);  view_482 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:290, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_483: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_385, [12, 512, 64]);  permute_385 = None
    permute_386: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_100, [0, 2, 1]);  view_100 = None
    bmm_52: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_386, view_483);  permute_386 = None
    permute_387: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_101, [0, 2, 1]);  view_101 = None
    bmm_53: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_483, permute_387);  view_483 = permute_387 = None
    view_484: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_52, [1, 12, 512, 64]);  bmm_52 = None
    view_485: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_53, [1, 12, 512, 512]);  bmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:284, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_27: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_45, torch.float32);  getitem_45 = None
    mul_337: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_27, 1.1111111111111112);  convert_element_type_27 = None
    mul_338: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_485, mul_337);  view_485 = mul_337 = None
    clone_49: "f32[1, 12, 512, 512]" = torch.ops.aten.clone.default(mul_338, memory_format = torch.contiguous_format);  mul_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:280, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_21: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    mul_339: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(clone_49, alias_21);  clone_49 = None
    sum_139: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_339, [-1], True)
    mul_340: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_21, sum_139);  alias_21 = sum_139 = None
    sub_100: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_339, mul_340);  mul_339 = mul_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:274, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_50: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_100, 8.0);  sub_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:250, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_486: "f32[12, 512, 512]" = torch.ops.aten.view.default(div_50, [12, 512, 512]);  div_50 = None
    permute_388: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_97, [0, 2, 1]);  view_97 = None
    bmm_54: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_388, view_486);  permute_388 = None
    permute_389: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_98, [0, 2, 1]);  view_98 = None
    bmm_55: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_486, permute_389);  view_486 = permute_389 = None
    view_487: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_54, [1, 12, 64, 512]);  bmm_54 = None
    view_488: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_55, [1, 12, 512, 64]);  bmm_55 = None
    permute_390: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_487, [0, 1, 3, 2]);  view_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_391: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_488, [0, 2, 1, 3]);  view_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    clone_50: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_391, memory_format = torch.contiguous_format);  permute_391 = None
    view_489: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_50, [1, 512, 768]);  clone_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_392: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_484, [0, 2, 1, 3]);  view_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    clone_51: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_392, memory_format = torch.contiguous_format);  permute_392 = None
    view_490: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_51, [1, 512, 768]);  clone_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:234, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_491: "f32[512, 768]" = torch.ops.aten.view.default(view_490, [512, 768]);  view_490 = None
    permute_393: "f32[768, 768]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
    mm_94: "f32[512, 768]" = torch.ops.aten.mm.default(view_491, permute_393);  permute_393 = None
    permute_394: "f32[768, 512]" = torch.ops.aten.permute.default(view_491, [1, 0])
    mm_95: "f32[768, 768]" = torch.ops.aten.mm.default(permute_394, view_93);  permute_394 = view_93 = None
    permute_395: "f32[768, 768]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_140: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_491, [0], True);  view_491 = None
    view_492: "f32[768]" = torch.ops.aten.view.default(sum_140, [768]);  sum_140 = None
    permute_396: "f32[768, 768]" = torch.ops.aten.permute.default(permute_395, [1, 0]);  permute_395 = None
    view_493: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_94, [1, 512, 768]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:234, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_153: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_333, view_493);  mul_333 = view_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_397: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(permute_390, [0, 2, 1, 3]);  permute_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_494: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_397, [1, 512, 768]);  permute_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:233, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_495: "f32[512, 768]" = torch.ops.aten.view.default(view_494, [512, 768]);  view_494 = None
    permute_398: "f32[768, 768]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
    mm_96: "f32[512, 768]" = torch.ops.aten.mm.default(view_495, permute_398);  permute_398 = None
    permute_399: "f32[768, 512]" = torch.ops.aten.permute.default(view_495, [1, 0])
    mm_97: "f32[768, 768]" = torch.ops.aten.mm.default(permute_399, view_90);  permute_399 = view_90 = None
    permute_400: "f32[768, 768]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_141: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_495, [0], True);  view_495 = None
    view_496: "f32[768]" = torch.ops.aten.view.default(sum_141, [768]);  sum_141 = None
    permute_401: "f32[768, 768]" = torch.ops.aten.permute.default(permute_400, [1, 0]);  permute_400 = None
    view_497: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_96, [1, 512, 768]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:233, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_154: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_153, view_497);  add_153 = view_497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:211, code: mixed_query_layer = self.query(hidden_states)
    view_498: "f32[512, 768]" = torch.ops.aten.view.default(view_489, [512, 768]);  view_489 = None
    permute_402: "f32[768, 768]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
    mm_98: "f32[512, 768]" = torch.ops.aten.mm.default(view_498, permute_402);  permute_402 = None
    permute_403: "f32[768, 512]" = torch.ops.aten.permute.default(view_498, [1, 0])
    mm_99: "f32[768, 768]" = torch.ops.aten.mm.default(permute_403, view_88);  permute_403 = view_88 = None
    permute_404: "f32[768, 768]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    sum_142: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_498, [0], True);  view_498 = None
    view_499: "f32[768]" = torch.ops.aten.view.default(sum_142, [768]);  sum_142 = None
    permute_405: "f32[768, 768]" = torch.ops.aten.permute.default(permute_404, [1, 0]);  permute_404 = None
    view_500: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_98, [1, 512, 768]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:211, code: mixed_query_layer = self.query(hidden_states)
    add_155: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_154, view_500);  add_154 = view_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:395, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_101: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_43);  add_35 = getitem_43 = None
    mul_341: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_101, rsqrt_8);  sub_101 = None
    mul_342: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_155, primals_68);  primals_68 = None
    mul_343: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_342, 768)
    sum_143: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_342, [2], True)
    mul_344: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_342, mul_341);  mul_342 = None
    sum_144: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_344, [2], True);  mul_344 = None
    mul_345: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_341, sum_144);  sum_144 = None
    sub_102: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_343, sum_143);  mul_343 = sum_143 = None
    sub_103: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_102, mul_345);  sub_102 = mul_345 = None
    div_51: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
    mul_346: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_51, sub_103);  div_51 = sub_103 = None
    mul_347: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_155, mul_341);  mul_341 = None
    sum_145: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_347, [0, 1]);  mul_347 = None
    sum_146: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_155, [0, 1]);  add_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:394, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_28: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_41, torch.float32);  getitem_41 = None
    mul_348: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_28, 1.1111111111111112);  convert_element_type_28 = None
    mul_349: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_346, mul_348);  mul_348 = None
    clone_52: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_349, memory_format = torch.contiguous_format);  mul_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:393, code: hidden_states = self.dense(hidden_states)
    view_501: "f32[512, 768]" = torch.ops.aten.view.default(clone_52, [512, 768]);  clone_52 = None
    permute_406: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
    mm_100: "f32[512, 3072]" = torch.ops.aten.mm.default(view_501, permute_406);  permute_406 = None
    permute_407: "f32[768, 512]" = torch.ops.aten.permute.default(view_501, [1, 0])
    mm_101: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_407, view_86);  permute_407 = view_86 = None
    permute_408: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_101, [1, 0]);  mm_101 = None
    sum_147: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_501, [0], True);  view_501 = None
    view_502: "f32[768]" = torch.ops.aten.view.default(sum_147, [768]);  sum_147 = None
    permute_409: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_408, [1, 0]);  permute_408 = None
    view_503: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_100, [1, 512, 3072]);  mm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_350: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476)
    erf_22: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_350);  mul_350 = None
    add_156: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_351: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_156, 0.5);  add_156 = None
    mul_352: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, view_85)
    mul_353: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_352, -0.5);  mul_352 = None
    exp_23: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_353);  mul_353 = None
    mul_354: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
    mul_355: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, mul_354);  view_85 = mul_354 = None
    add_157: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_351, mul_355);  mul_351 = mul_355 = None
    mul_356: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_503, add_157);  view_503 = add_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:379, code: hidden_states = self.dense(hidden_states)
    view_504: "f32[512, 3072]" = torch.ops.aten.view.default(mul_356, [512, 3072]);  mul_356 = None
    permute_410: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    mm_102: "f32[512, 768]" = torch.ops.aten.mm.default(view_504, permute_410);  permute_410 = None
    permute_411: "f32[3072, 512]" = torch.ops.aten.permute.default(view_504, [1, 0])
    mm_103: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_411, view_84);  permute_411 = view_84 = None
    permute_412: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
    sum_148: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_504, [0], True);  view_504 = None
    view_505: "f32[3072]" = torch.ops.aten.view.default(sum_148, [3072]);  sum_148 = None
    permute_413: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_412, [1, 0]);  permute_412 = None
    view_506: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_102, [1, 512, 768]);  mm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:379, code: hidden_states = self.dense(hidden_states)
    add_158: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_346, view_506);  mul_346 = view_506 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:314, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_104: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_31, getitem_39);  add_31 = getitem_39 = None
    mul_357: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_104, rsqrt_7);  sub_104 = None
    mul_358: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_158, primals_62);  primals_62 = None
    mul_359: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_358, 768)
    sum_149: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_358, [2], True)
    mul_360: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_358, mul_357);  mul_358 = None
    sum_150: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_360, [2], True);  mul_360 = None
    mul_361: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_357, sum_150);  sum_150 = None
    sub_105: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_359, sum_149);  mul_359 = sum_149 = None
    sub_106: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_105, mul_361);  sub_105 = mul_361 = None
    div_52: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
    mul_362: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_52, sub_106);  div_52 = sub_106 = None
    mul_363: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_158, mul_357);  mul_357 = None
    sum_151: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_363, [0, 1]);  mul_363 = None
    sum_152: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_158, [0, 1]);  add_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:313, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_29: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_37, torch.float32);  getitem_37 = None
    mul_364: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_29, 1.1111111111111112);  convert_element_type_29 = None
    mul_365: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_362, mul_364);  mul_364 = None
    clone_53: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_365, memory_format = torch.contiguous_format);  mul_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:312, code: hidden_states = self.dense(hidden_states)
    view_507: "f32[512, 768]" = torch.ops.aten.view.default(clone_53, [512, 768]);  clone_53 = None
    permute_414: "f32[768, 768]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
    mm_104: "f32[512, 768]" = torch.ops.aten.mm.default(view_507, permute_414);  permute_414 = None
    permute_415: "f32[768, 512]" = torch.ops.aten.permute.default(view_507, [1, 0])
    mm_105: "f32[768, 768]" = torch.ops.aten.mm.default(permute_415, view_82);  permute_415 = view_82 = None
    permute_416: "f32[768, 768]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
    sum_153: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_507, [0], True);  view_507 = None
    view_508: "f32[768]" = torch.ops.aten.view.default(sum_153, [768]);  sum_153 = None
    permute_417: "f32[768, 768]" = torch.ops.aten.permute.default(permute_416, [1, 0]);  permute_416 = None
    view_509: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_104, [1, 512, 768]);  mm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:294, code: context_layer = context_layer.view(new_context_layer_shape)
    view_510: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_509, [1, 512, 12, 64]);  view_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:292, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_418: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_510, [0, 2, 1, 3]);  view_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:290, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_511: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_418, [12, 512, 64]);  permute_418 = None
    permute_419: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
    bmm_56: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_419, view_511);  permute_419 = None
    permute_420: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_79, [0, 2, 1]);  view_79 = None
    bmm_57: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_511, permute_420);  view_511 = permute_420 = None
    view_512: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_56, [1, 12, 512, 64]);  bmm_56 = None
    view_513: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_57, [1, 12, 512, 512]);  bmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:284, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_30: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_35, torch.float32);  getitem_35 = None
    mul_366: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_30, 1.1111111111111112);  convert_element_type_30 = None
    mul_367: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_513, mul_366);  view_513 = mul_366 = None
    clone_54: "f32[1, 12, 512, 512]" = torch.ops.aten.clone.default(mul_367, memory_format = torch.contiguous_format);  mul_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:280, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_22: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    mul_368: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(clone_54, alias_22);  clone_54 = None
    sum_154: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_368, [-1], True)
    mul_369: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_22, sum_154);  alias_22 = sum_154 = None
    sub_107: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_368, mul_369);  mul_368 = mul_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:274, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_53: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_107, 8.0);  sub_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:250, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_514: "f32[12, 512, 512]" = torch.ops.aten.view.default(div_53, [12, 512, 512]);  div_53 = None
    permute_421: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_75, [0, 2, 1]);  view_75 = None
    bmm_58: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_421, view_514);  permute_421 = None
    permute_422: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1]);  view_76 = None
    bmm_59: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_514, permute_422);  view_514 = permute_422 = None
    view_515: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_58, [1, 12, 64, 512]);  bmm_58 = None
    view_516: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_59, [1, 12, 512, 64]);  bmm_59 = None
    permute_423: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_515, [0, 1, 3, 2]);  view_515 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_424: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_516, [0, 2, 1, 3]);  view_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    clone_55: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_424, memory_format = torch.contiguous_format);  permute_424 = None
    view_517: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_55, [1, 512, 768]);  clone_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_425: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_512, [0, 2, 1, 3]);  view_512 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    clone_56: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_425, memory_format = torch.contiguous_format);  permute_425 = None
    view_518: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_56, [1, 512, 768]);  clone_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:234, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_519: "f32[512, 768]" = torch.ops.aten.view.default(view_518, [512, 768]);  view_518 = None
    permute_426: "f32[768, 768]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
    mm_106: "f32[512, 768]" = torch.ops.aten.mm.default(view_519, permute_426);  permute_426 = None
    permute_427: "f32[768, 512]" = torch.ops.aten.permute.default(view_519, [1, 0])
    mm_107: "f32[768, 768]" = torch.ops.aten.mm.default(permute_427, view_71);  permute_427 = view_71 = None
    permute_428: "f32[768, 768]" = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
    sum_155: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_519, [0], True);  view_519 = None
    view_520: "f32[768]" = torch.ops.aten.view.default(sum_155, [768]);  sum_155 = None
    permute_429: "f32[768, 768]" = torch.ops.aten.permute.default(permute_428, [1, 0]);  permute_428 = None
    view_521: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_106, [1, 512, 768]);  mm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:234, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_159: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_362, view_521);  mul_362 = view_521 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_430: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(permute_423, [0, 2, 1, 3]);  permute_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_522: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_430, [1, 512, 768]);  permute_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:233, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_523: "f32[512, 768]" = torch.ops.aten.view.default(view_522, [512, 768]);  view_522 = None
    permute_431: "f32[768, 768]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    mm_108: "f32[512, 768]" = torch.ops.aten.mm.default(view_523, permute_431);  permute_431 = None
    permute_432: "f32[768, 512]" = torch.ops.aten.permute.default(view_523, [1, 0])
    mm_109: "f32[768, 768]" = torch.ops.aten.mm.default(permute_432, view_68);  permute_432 = view_68 = None
    permute_433: "f32[768, 768]" = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
    sum_156: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_523, [0], True);  view_523 = None
    view_524: "f32[768]" = torch.ops.aten.view.default(sum_156, [768]);  sum_156 = None
    permute_434: "f32[768, 768]" = torch.ops.aten.permute.default(permute_433, [1, 0]);  permute_433 = None
    view_525: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_108, [1, 512, 768]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:233, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_160: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_159, view_525);  add_159 = view_525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:211, code: mixed_query_layer = self.query(hidden_states)
    view_526: "f32[512, 768]" = torch.ops.aten.view.default(view_517, [512, 768]);  view_517 = None
    permute_435: "f32[768, 768]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    mm_110: "f32[512, 768]" = torch.ops.aten.mm.default(view_526, permute_435);  permute_435 = None
    permute_436: "f32[768, 512]" = torch.ops.aten.permute.default(view_526, [1, 0])
    mm_111: "f32[768, 768]" = torch.ops.aten.mm.default(permute_436, view_66);  permute_436 = view_66 = None
    permute_437: "f32[768, 768]" = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
    sum_157: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_526, [0], True);  view_526 = None
    view_527: "f32[768]" = torch.ops.aten.view.default(sum_157, [768]);  sum_157 = None
    permute_438: "f32[768, 768]" = torch.ops.aten.permute.default(permute_437, [1, 0]);  permute_437 = None
    view_528: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_110, [1, 512, 768]);  mm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:211, code: mixed_query_layer = self.query(hidden_states)
    add_161: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_160, view_528);  add_160 = view_528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:395, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_108: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_27, getitem_33);  add_27 = getitem_33 = None
    mul_370: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_108, rsqrt_6);  sub_108 = None
    mul_371: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_161, primals_52);  primals_52 = None
    mul_372: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_371, 768)
    sum_158: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_371, [2], True)
    mul_373: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_371, mul_370);  mul_371 = None
    sum_159: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_373, [2], True);  mul_373 = None
    mul_374: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_370, sum_159);  sum_159 = None
    sub_109: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_372, sum_158);  mul_372 = sum_158 = None
    sub_110: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_109, mul_374);  sub_109 = mul_374 = None
    div_54: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
    mul_375: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_54, sub_110);  div_54 = sub_110 = None
    mul_376: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_161, mul_370);  mul_370 = None
    sum_160: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_376, [0, 1]);  mul_376 = None
    sum_161: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_161, [0, 1]);  add_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:394, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_31: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_31, torch.float32);  getitem_31 = None
    mul_377: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_31, 1.1111111111111112);  convert_element_type_31 = None
    mul_378: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_375, mul_377);  mul_377 = None
    clone_57: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_378, memory_format = torch.contiguous_format);  mul_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:393, code: hidden_states = self.dense(hidden_states)
    view_529: "f32[512, 768]" = torch.ops.aten.view.default(clone_57, [512, 768]);  clone_57 = None
    permute_439: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    mm_112: "f32[512, 3072]" = torch.ops.aten.mm.default(view_529, permute_439);  permute_439 = None
    permute_440: "f32[768, 512]" = torch.ops.aten.permute.default(view_529, [1, 0])
    mm_113: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_440, view_64);  permute_440 = view_64 = None
    permute_441: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_113, [1, 0]);  mm_113 = None
    sum_162: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_529, [0], True);  view_529 = None
    view_530: "f32[768]" = torch.ops.aten.view.default(sum_162, [768]);  sum_162 = None
    permute_442: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_441, [1, 0]);  permute_441 = None
    view_531: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_112, [1, 512, 3072]);  mm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_379: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476)
    erf_23: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_379);  mul_379 = None
    add_162: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_380: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_162, 0.5);  add_162 = None
    mul_381: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, view_63)
    mul_382: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_381, -0.5);  mul_381 = None
    exp_24: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_382);  mul_382 = None
    mul_383: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
    mul_384: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, mul_383);  view_63 = mul_383 = None
    add_163: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_380, mul_384);  mul_380 = mul_384 = None
    mul_385: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_531, add_163);  view_531 = add_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:379, code: hidden_states = self.dense(hidden_states)
    view_532: "f32[512, 3072]" = torch.ops.aten.view.default(mul_385, [512, 3072]);  mul_385 = None
    permute_443: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
    mm_114: "f32[512, 768]" = torch.ops.aten.mm.default(view_532, permute_443);  permute_443 = None
    permute_444: "f32[3072, 512]" = torch.ops.aten.permute.default(view_532, [1, 0])
    mm_115: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_444, view_62);  permute_444 = view_62 = None
    permute_445: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
    sum_163: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_532, [0], True);  view_532 = None
    view_533: "f32[3072]" = torch.ops.aten.view.default(sum_163, [3072]);  sum_163 = None
    permute_446: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_445, [1, 0]);  permute_445 = None
    view_534: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_114, [1, 512, 768]);  mm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:379, code: hidden_states = self.dense(hidden_states)
    add_164: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_375, view_534);  mul_375 = view_534 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:314, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_111: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_23, getitem_29);  add_23 = getitem_29 = None
    mul_386: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_111, rsqrt_5);  sub_111 = None
    mul_387: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_164, primals_46);  primals_46 = None
    mul_388: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_387, 768)
    sum_164: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_387, [2], True)
    mul_389: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_387, mul_386);  mul_387 = None
    sum_165: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_389, [2], True);  mul_389 = None
    mul_390: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_386, sum_165);  sum_165 = None
    sub_112: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_388, sum_164);  mul_388 = sum_164 = None
    sub_113: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_112, mul_390);  sub_112 = mul_390 = None
    div_55: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
    mul_391: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_55, sub_113);  div_55 = sub_113 = None
    mul_392: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_164, mul_386);  mul_386 = None
    sum_166: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_392, [0, 1]);  mul_392 = None
    sum_167: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_164, [0, 1]);  add_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:313, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_32: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_27, torch.float32);  getitem_27 = None
    mul_393: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_32, 1.1111111111111112);  convert_element_type_32 = None
    mul_394: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_391, mul_393);  mul_393 = None
    clone_58: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_394, memory_format = torch.contiguous_format);  mul_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:312, code: hidden_states = self.dense(hidden_states)
    view_535: "f32[512, 768]" = torch.ops.aten.view.default(clone_58, [512, 768]);  clone_58 = None
    permute_447: "f32[768, 768]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
    mm_116: "f32[512, 768]" = torch.ops.aten.mm.default(view_535, permute_447);  permute_447 = None
    permute_448: "f32[768, 512]" = torch.ops.aten.permute.default(view_535, [1, 0])
    mm_117: "f32[768, 768]" = torch.ops.aten.mm.default(permute_448, view_60);  permute_448 = view_60 = None
    permute_449: "f32[768, 768]" = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
    sum_168: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_535, [0], True);  view_535 = None
    view_536: "f32[768]" = torch.ops.aten.view.default(sum_168, [768]);  sum_168 = None
    permute_450: "f32[768, 768]" = torch.ops.aten.permute.default(permute_449, [1, 0]);  permute_449 = None
    view_537: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_116, [1, 512, 768]);  mm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:294, code: context_layer = context_layer.view(new_context_layer_shape)
    view_538: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_537, [1, 512, 12, 64]);  view_537 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:292, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_451: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_538, [0, 2, 1, 3]);  view_538 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:290, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_539: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_451, [12, 512, 64]);  permute_451 = None
    permute_452: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None
    bmm_60: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_452, view_539);  permute_452 = None
    permute_453: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_57, [0, 2, 1]);  view_57 = None
    bmm_61: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_539, permute_453);  view_539 = permute_453 = None
    view_540: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_60, [1, 12, 512, 64]);  bmm_60 = None
    view_541: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_61, [1, 12, 512, 512]);  bmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:284, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_33: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_25, torch.float32);  getitem_25 = None
    mul_395: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_33, 1.1111111111111112);  convert_element_type_33 = None
    mul_396: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_541, mul_395);  view_541 = mul_395 = None
    clone_59: "f32[1, 12, 512, 512]" = torch.ops.aten.clone.default(mul_396, memory_format = torch.contiguous_format);  mul_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:280, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_23: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    mul_397: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(clone_59, alias_23);  clone_59 = None
    sum_169: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_397, [-1], True)
    mul_398: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_23, sum_169);  alias_23 = sum_169 = None
    sub_114: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_397, mul_398);  mul_397 = mul_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:274, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_56: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_114, 8.0);  sub_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:250, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_542: "f32[12, 512, 512]" = torch.ops.aten.view.default(div_56, [12, 512, 512]);  div_56 = None
    permute_454: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_53, [0, 2, 1]);  view_53 = None
    bmm_62: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_454, view_542);  permute_454 = None
    permute_455: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1]);  view_54 = None
    bmm_63: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_542, permute_455);  view_542 = permute_455 = None
    view_543: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_62, [1, 12, 64, 512]);  bmm_62 = None
    view_544: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_63, [1, 12, 512, 64]);  bmm_63 = None
    permute_456: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_543, [0, 1, 3, 2]);  view_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_457: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_544, [0, 2, 1, 3]);  view_544 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    clone_60: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_457, memory_format = torch.contiguous_format);  permute_457 = None
    view_545: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_60, [1, 512, 768]);  clone_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_458: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_540, [0, 2, 1, 3]);  view_540 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    clone_61: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_458, memory_format = torch.contiguous_format);  permute_458 = None
    view_546: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_61, [1, 512, 768]);  clone_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:234, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_547: "f32[512, 768]" = torch.ops.aten.view.default(view_546, [512, 768]);  view_546 = None
    permute_459: "f32[768, 768]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
    mm_118: "f32[512, 768]" = torch.ops.aten.mm.default(view_547, permute_459);  permute_459 = None
    permute_460: "f32[768, 512]" = torch.ops.aten.permute.default(view_547, [1, 0])
    mm_119: "f32[768, 768]" = torch.ops.aten.mm.default(permute_460, view_49);  permute_460 = view_49 = None
    permute_461: "f32[768, 768]" = torch.ops.aten.permute.default(mm_119, [1, 0]);  mm_119 = None
    sum_170: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_547, [0], True);  view_547 = None
    view_548: "f32[768]" = torch.ops.aten.view.default(sum_170, [768]);  sum_170 = None
    permute_462: "f32[768, 768]" = torch.ops.aten.permute.default(permute_461, [1, 0]);  permute_461 = None
    view_549: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_118, [1, 512, 768]);  mm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:234, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_165: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_391, view_549);  mul_391 = view_549 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_463: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(permute_456, [0, 2, 1, 3]);  permute_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_550: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_463, [1, 512, 768]);  permute_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:233, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_551: "f32[512, 768]" = torch.ops.aten.view.default(view_550, [512, 768]);  view_550 = None
    permute_464: "f32[768, 768]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    mm_120: "f32[512, 768]" = torch.ops.aten.mm.default(view_551, permute_464);  permute_464 = None
    permute_465: "f32[768, 512]" = torch.ops.aten.permute.default(view_551, [1, 0])
    mm_121: "f32[768, 768]" = torch.ops.aten.mm.default(permute_465, view_46);  permute_465 = view_46 = None
    permute_466: "f32[768, 768]" = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
    sum_171: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_551, [0], True);  view_551 = None
    view_552: "f32[768]" = torch.ops.aten.view.default(sum_171, [768]);  sum_171 = None
    permute_467: "f32[768, 768]" = torch.ops.aten.permute.default(permute_466, [1, 0]);  permute_466 = None
    view_553: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_120, [1, 512, 768]);  mm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:233, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_166: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_165, view_553);  add_165 = view_553 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:211, code: mixed_query_layer = self.query(hidden_states)
    view_554: "f32[512, 768]" = torch.ops.aten.view.default(view_545, [512, 768]);  view_545 = None
    permute_468: "f32[768, 768]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    mm_122: "f32[512, 768]" = torch.ops.aten.mm.default(view_554, permute_468);  permute_468 = None
    permute_469: "f32[768, 512]" = torch.ops.aten.permute.default(view_554, [1, 0])
    mm_123: "f32[768, 768]" = torch.ops.aten.mm.default(permute_469, view_44);  permute_469 = view_44 = None
    permute_470: "f32[768, 768]" = torch.ops.aten.permute.default(mm_123, [1, 0]);  mm_123 = None
    sum_172: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_554, [0], True);  view_554 = None
    view_555: "f32[768]" = torch.ops.aten.view.default(sum_172, [768]);  sum_172 = None
    permute_471: "f32[768, 768]" = torch.ops.aten.permute.default(permute_470, [1, 0]);  permute_470 = None
    view_556: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_122, [1, 512, 768]);  mm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:211, code: mixed_query_layer = self.query(hidden_states)
    add_167: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_166, view_556);  add_166 = view_556 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:395, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_115: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_19, getitem_23);  add_19 = getitem_23 = None
    mul_399: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_115, rsqrt_4);  sub_115 = None
    mul_400: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_167, primals_36);  primals_36 = None
    mul_401: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_400, 768)
    sum_173: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_400, [2], True)
    mul_402: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_400, mul_399);  mul_400 = None
    sum_174: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_402, [2], True);  mul_402 = None
    mul_403: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_399, sum_174);  sum_174 = None
    sub_116: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_401, sum_173);  mul_401 = sum_173 = None
    sub_117: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_116, mul_403);  sub_116 = mul_403 = None
    div_57: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
    mul_404: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_57, sub_117);  div_57 = sub_117 = None
    mul_405: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_167, mul_399);  mul_399 = None
    sum_175: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_405, [0, 1]);  mul_405 = None
    sum_176: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_167, [0, 1]);  add_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:394, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_34: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_21, torch.float32);  getitem_21 = None
    mul_406: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_34, 1.1111111111111112);  convert_element_type_34 = None
    mul_407: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_404, mul_406);  mul_406 = None
    clone_62: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_407, memory_format = torch.contiguous_format);  mul_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:393, code: hidden_states = self.dense(hidden_states)
    view_557: "f32[512, 768]" = torch.ops.aten.view.default(clone_62, [512, 768]);  clone_62 = None
    permute_472: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    mm_124: "f32[512, 3072]" = torch.ops.aten.mm.default(view_557, permute_472);  permute_472 = None
    permute_473: "f32[768, 512]" = torch.ops.aten.permute.default(view_557, [1, 0])
    mm_125: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_473, view_42);  permute_473 = view_42 = None
    permute_474: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_125, [1, 0]);  mm_125 = None
    sum_177: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_557, [0], True);  view_557 = None
    view_558: "f32[768]" = torch.ops.aten.view.default(sum_177, [768]);  sum_177 = None
    permute_475: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_474, [1, 0]);  permute_474 = None
    view_559: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_124, [1, 512, 3072]);  mm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_408: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476)
    erf_24: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_408);  mul_408 = None
    add_168: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
    mul_409: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_168, 0.5);  add_168 = None
    mul_410: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, view_41)
    mul_411: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_410, -0.5);  mul_410 = None
    exp_25: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_411);  mul_411 = None
    mul_412: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
    mul_413: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, mul_412);  view_41 = mul_412 = None
    add_169: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_409, mul_413);  mul_409 = mul_413 = None
    mul_414: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_559, add_169);  view_559 = add_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:379, code: hidden_states = self.dense(hidden_states)
    view_560: "f32[512, 3072]" = torch.ops.aten.view.default(mul_414, [512, 3072]);  mul_414 = None
    permute_476: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    mm_126: "f32[512, 768]" = torch.ops.aten.mm.default(view_560, permute_476);  permute_476 = None
    permute_477: "f32[3072, 512]" = torch.ops.aten.permute.default(view_560, [1, 0])
    mm_127: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_477, view_40);  permute_477 = view_40 = None
    permute_478: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_127, [1, 0]);  mm_127 = None
    sum_178: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_560, [0], True);  view_560 = None
    view_561: "f32[3072]" = torch.ops.aten.view.default(sum_178, [3072]);  sum_178 = None
    permute_479: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_478, [1, 0]);  permute_478 = None
    view_562: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_126, [1, 512, 768]);  mm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:379, code: hidden_states = self.dense(hidden_states)
    add_170: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_404, view_562);  mul_404 = view_562 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:314, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_118: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_15, getitem_19);  add_15 = getitem_19 = None
    mul_415: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_118, rsqrt_3);  sub_118 = None
    mul_416: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_170, primals_30);  primals_30 = None
    mul_417: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_416, 768)
    sum_179: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_416, [2], True)
    mul_418: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_416, mul_415);  mul_416 = None
    sum_180: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_418, [2], True);  mul_418 = None
    mul_419: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_415, sum_180);  sum_180 = None
    sub_119: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_417, sum_179);  mul_417 = sum_179 = None
    sub_120: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_119, mul_419);  sub_119 = mul_419 = None
    div_58: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
    mul_420: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_58, sub_120);  div_58 = sub_120 = None
    mul_421: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_170, mul_415);  mul_415 = None
    sum_181: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_421, [0, 1]);  mul_421 = None
    sum_182: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_170, [0, 1]);  add_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:313, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_35: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_17, torch.float32);  getitem_17 = None
    mul_422: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_35, 1.1111111111111112);  convert_element_type_35 = None
    mul_423: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_420, mul_422);  mul_422 = None
    clone_63: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_423, memory_format = torch.contiguous_format);  mul_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:312, code: hidden_states = self.dense(hidden_states)
    view_563: "f32[512, 768]" = torch.ops.aten.view.default(clone_63, [512, 768]);  clone_63 = None
    permute_480: "f32[768, 768]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
    mm_128: "f32[512, 768]" = torch.ops.aten.mm.default(view_563, permute_480);  permute_480 = None
    permute_481: "f32[768, 512]" = torch.ops.aten.permute.default(view_563, [1, 0])
    mm_129: "f32[768, 768]" = torch.ops.aten.mm.default(permute_481, view_38);  permute_481 = view_38 = None
    permute_482: "f32[768, 768]" = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
    sum_183: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_563, [0], True);  view_563 = None
    view_564: "f32[768]" = torch.ops.aten.view.default(sum_183, [768]);  sum_183 = None
    permute_483: "f32[768, 768]" = torch.ops.aten.permute.default(permute_482, [1, 0]);  permute_482 = None
    view_565: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_128, [1, 512, 768]);  mm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:294, code: context_layer = context_layer.view(new_context_layer_shape)
    view_566: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_565, [1, 512, 12, 64]);  view_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:292, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_484: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_566, [0, 2, 1, 3]);  view_566 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:290, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_567: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_484, [12, 512, 64]);  permute_484 = None
    permute_485: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
    bmm_64: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_485, view_567);  permute_485 = None
    permute_486: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_35, [0, 2, 1]);  view_35 = None
    bmm_65: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_567, permute_486);  view_567 = permute_486 = None
    view_568: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_64, [1, 12, 512, 64]);  bmm_64 = None
    view_569: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_65, [1, 12, 512, 512]);  bmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:284, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_36: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_15, torch.float32);  getitem_15 = None
    mul_424: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_36, 1.1111111111111112);  convert_element_type_36 = None
    mul_425: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_569, mul_424);  view_569 = mul_424 = None
    clone_64: "f32[1, 12, 512, 512]" = torch.ops.aten.clone.default(mul_425, memory_format = torch.contiguous_format);  mul_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:280, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_24: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    mul_426: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(clone_64, alias_24);  clone_64 = None
    sum_184: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_426, [-1], True)
    mul_427: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_24, sum_184);  alias_24 = sum_184 = None
    sub_121: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_426, mul_427);  mul_426 = mul_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:274, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_59: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_121, 8.0);  sub_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:250, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_570: "f32[12, 512, 512]" = torch.ops.aten.view.default(div_59, [12, 512, 512]);  div_59 = None
    permute_487: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_31, [0, 2, 1]);  view_31 = None
    bmm_66: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_487, view_570);  permute_487 = None
    permute_488: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_32, [0, 2, 1]);  view_32 = None
    bmm_67: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_570, permute_488);  view_570 = permute_488 = None
    view_571: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_66, [1, 12, 64, 512]);  bmm_66 = None
    view_572: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_67, [1, 12, 512, 64]);  bmm_67 = None
    permute_489: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_571, [0, 1, 3, 2]);  view_571 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_490: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_572, [0, 2, 1, 3]);  view_572 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    clone_65: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_490, memory_format = torch.contiguous_format);  permute_490 = None
    view_573: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_65, [1, 512, 768]);  clone_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_491: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_568, [0, 2, 1, 3]);  view_568 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    clone_66: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_491, memory_format = torch.contiguous_format);  permute_491 = None
    view_574: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_66, [1, 512, 768]);  clone_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:234, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_575: "f32[512, 768]" = torch.ops.aten.view.default(view_574, [512, 768]);  view_574 = None
    permute_492: "f32[768, 768]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
    mm_130: "f32[512, 768]" = torch.ops.aten.mm.default(view_575, permute_492);  permute_492 = None
    permute_493: "f32[768, 512]" = torch.ops.aten.permute.default(view_575, [1, 0])
    mm_131: "f32[768, 768]" = torch.ops.aten.mm.default(permute_493, view_27);  permute_493 = view_27 = None
    permute_494: "f32[768, 768]" = torch.ops.aten.permute.default(mm_131, [1, 0]);  mm_131 = None
    sum_185: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_575, [0], True);  view_575 = None
    view_576: "f32[768]" = torch.ops.aten.view.default(sum_185, [768]);  sum_185 = None
    permute_495: "f32[768, 768]" = torch.ops.aten.permute.default(permute_494, [1, 0]);  permute_494 = None
    view_577: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_130, [1, 512, 768]);  mm_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:234, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_171: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_420, view_577);  mul_420 = view_577 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_496: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(permute_489, [0, 2, 1, 3]);  permute_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_578: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_496, [1, 512, 768]);  permute_496 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:233, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_579: "f32[512, 768]" = torch.ops.aten.view.default(view_578, [512, 768]);  view_578 = None
    permute_497: "f32[768, 768]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    mm_132: "f32[512, 768]" = torch.ops.aten.mm.default(view_579, permute_497);  permute_497 = None
    permute_498: "f32[768, 512]" = torch.ops.aten.permute.default(view_579, [1, 0])
    mm_133: "f32[768, 768]" = torch.ops.aten.mm.default(permute_498, view_24);  permute_498 = view_24 = None
    permute_499: "f32[768, 768]" = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
    sum_186: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_579, [0], True);  view_579 = None
    view_580: "f32[768]" = torch.ops.aten.view.default(sum_186, [768]);  sum_186 = None
    permute_500: "f32[768, 768]" = torch.ops.aten.permute.default(permute_499, [1, 0]);  permute_499 = None
    view_581: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_132, [1, 512, 768]);  mm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:233, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_172: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_171, view_581);  add_171 = view_581 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:211, code: mixed_query_layer = self.query(hidden_states)
    view_582: "f32[512, 768]" = torch.ops.aten.view.default(view_573, [512, 768]);  view_573 = None
    permute_501: "f32[768, 768]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    mm_134: "f32[512, 768]" = torch.ops.aten.mm.default(view_582, permute_501);  permute_501 = None
    permute_502: "f32[768, 512]" = torch.ops.aten.permute.default(view_582, [1, 0])
    mm_135: "f32[768, 768]" = torch.ops.aten.mm.default(permute_502, view_22);  permute_502 = view_22 = None
    permute_503: "f32[768, 768]" = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
    sum_187: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_582, [0], True);  view_582 = None
    view_583: "f32[768]" = torch.ops.aten.view.default(sum_187, [768]);  sum_187 = None
    permute_504: "f32[768, 768]" = torch.ops.aten.permute.default(permute_503, [1, 0]);  permute_503 = None
    view_584: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_134, [1, 512, 768]);  mm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:211, code: mixed_query_layer = self.query(hidden_states)
    add_173: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_172, view_584);  add_172 = view_584 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:395, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_122: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_11, getitem_13);  add_11 = getitem_13 = None
    mul_428: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_122, rsqrt_2);  sub_122 = None
    mul_429: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_173, primals_20);  primals_20 = None
    mul_430: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_429, 768)
    sum_188: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_429, [2], True)
    mul_431: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_429, mul_428);  mul_429 = None
    sum_189: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_431, [2], True);  mul_431 = None
    mul_432: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_428, sum_189);  sum_189 = None
    sub_123: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_430, sum_188);  mul_430 = sum_188 = None
    sub_124: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_123, mul_432);  sub_123 = mul_432 = None
    div_60: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
    mul_433: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_60, sub_124);  div_60 = sub_124 = None
    mul_434: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_173, mul_428);  mul_428 = None
    sum_190: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_434, [0, 1]);  mul_434 = None
    sum_191: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_173, [0, 1]);  add_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:394, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_37: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_11, torch.float32);  getitem_11 = None
    mul_435: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_37, 1.1111111111111112);  convert_element_type_37 = None
    mul_436: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_433, mul_435);  mul_435 = None
    clone_67: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_436, memory_format = torch.contiguous_format);  mul_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:393, code: hidden_states = self.dense(hidden_states)
    view_585: "f32[512, 768]" = torch.ops.aten.view.default(clone_67, [512, 768]);  clone_67 = None
    permute_505: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    mm_136: "f32[512, 3072]" = torch.ops.aten.mm.default(view_585, permute_505);  permute_505 = None
    permute_506: "f32[768, 512]" = torch.ops.aten.permute.default(view_585, [1, 0])
    mm_137: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_506, view_20);  permute_506 = view_20 = None
    permute_507: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_137, [1, 0]);  mm_137 = None
    sum_192: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_585, [0], True);  view_585 = None
    view_586: "f32[768]" = torch.ops.aten.view.default(sum_192, [768]);  sum_192 = None
    permute_508: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_507, [1, 0]);  permute_507 = None
    view_587: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_136, [1, 512, 3072]);  mm_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_437: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476)
    erf_25: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_437);  mul_437 = None
    add_174: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
    mul_438: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_174, 0.5);  add_174 = None
    mul_439: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, view_19)
    mul_440: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_439, -0.5);  mul_439 = None
    exp_26: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_440);  mul_440 = None
    mul_441: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
    mul_442: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, mul_441);  view_19 = mul_441 = None
    add_175: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_438, mul_442);  mul_438 = mul_442 = None
    mul_443: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_587, add_175);  view_587 = add_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:379, code: hidden_states = self.dense(hidden_states)
    view_588: "f32[512, 3072]" = torch.ops.aten.view.default(mul_443, [512, 3072]);  mul_443 = None
    permute_509: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    mm_138: "f32[512, 768]" = torch.ops.aten.mm.default(view_588, permute_509);  permute_509 = None
    permute_510: "f32[3072, 512]" = torch.ops.aten.permute.default(view_588, [1, 0])
    mm_139: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_510, view_18);  permute_510 = view_18 = None
    permute_511: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
    sum_193: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_588, [0], True);  view_588 = None
    view_589: "f32[3072]" = torch.ops.aten.view.default(sum_193, [3072]);  sum_193 = None
    permute_512: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_511, [1, 0]);  permute_511 = None
    view_590: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_138, [1, 512, 768]);  mm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:379, code: hidden_states = self.dense(hidden_states)
    add_176: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_433, view_590);  mul_433 = view_590 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:314, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_125: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_7, getitem_9);  add_7 = getitem_9 = None
    mul_444: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_125, rsqrt_1);  sub_125 = None
    mul_445: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_176, primals_14);  primals_14 = None
    mul_446: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_445, 768)
    sum_194: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_445, [2], True)
    mul_447: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_445, mul_444);  mul_445 = None
    sum_195: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_447, [2], True);  mul_447 = None
    mul_448: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_444, sum_195);  sum_195 = None
    sub_126: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_446, sum_194);  mul_446 = sum_194 = None
    sub_127: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_126, mul_448);  sub_126 = mul_448 = None
    div_61: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
    mul_449: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_61, sub_127);  div_61 = sub_127 = None
    mul_450: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_176, mul_444);  mul_444 = None
    sum_196: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_450, [0, 1]);  mul_450 = None
    sum_197: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_176, [0, 1]);  add_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:313, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_38: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_7, torch.float32);  getitem_7 = None
    mul_451: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_38, 1.1111111111111112);  convert_element_type_38 = None
    mul_452: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_449, mul_451);  mul_451 = None
    clone_68: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_452, memory_format = torch.contiguous_format);  mul_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:312, code: hidden_states = self.dense(hidden_states)
    view_591: "f32[512, 768]" = torch.ops.aten.view.default(clone_68, [512, 768]);  clone_68 = None
    permute_513: "f32[768, 768]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    mm_140: "f32[512, 768]" = torch.ops.aten.mm.default(view_591, permute_513);  permute_513 = None
    permute_514: "f32[768, 512]" = torch.ops.aten.permute.default(view_591, [1, 0])
    mm_141: "f32[768, 768]" = torch.ops.aten.mm.default(permute_514, view_16);  permute_514 = view_16 = None
    permute_515: "f32[768, 768]" = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
    sum_198: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_591, [0], True);  view_591 = None
    view_592: "f32[768]" = torch.ops.aten.view.default(sum_198, [768]);  sum_198 = None
    permute_516: "f32[768, 768]" = torch.ops.aten.permute.default(permute_515, [1, 0]);  permute_515 = None
    view_593: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_140, [1, 512, 768]);  mm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:294, code: context_layer = context_layer.view(new_context_layer_shape)
    view_594: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_593, [1, 512, 12, 64]);  view_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:292, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_517: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_594, [0, 2, 1, 3]);  view_594 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:290, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_595: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_517, [12, 512, 64]);  permute_517 = None
    permute_518: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
    bmm_68: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_518, view_595);  permute_518 = None
    permute_519: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None
    bmm_69: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_595, permute_519);  view_595 = permute_519 = None
    view_596: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_68, [1, 12, 512, 64]);  bmm_68 = None
    view_597: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_69, [1, 12, 512, 512]);  bmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:284, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_39: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_5, torch.float32);  getitem_5 = None
    mul_453: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_39, 1.1111111111111112);  convert_element_type_39 = None
    mul_454: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_597, mul_453);  view_597 = mul_453 = None
    clone_69: "f32[1, 12, 512, 512]" = torch.ops.aten.clone.default(mul_454, memory_format = torch.contiguous_format);  mul_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:280, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_25: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias);  alias = None
    mul_455: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(clone_69, alias_25);  clone_69 = None
    sum_199: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_455, [-1], True)
    mul_456: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_25, sum_199);  alias_25 = sum_199 = None
    sub_128: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_455, mul_456);  mul_455 = mul_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:274, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_62: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_128, 8.0);  sub_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:250, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_598: "f32[12, 512, 512]" = torch.ops.aten.view.default(div_62, [12, 512, 512]);  div_62 = None
    permute_520: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
    bmm_70: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_520, view_598);  permute_520 = None
    permute_521: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
    bmm_71: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_598, permute_521);  view_598 = permute_521 = None
    view_599: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_70, [1, 12, 64, 512]);  bmm_70 = None
    view_600: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_71, [1, 12, 512, 64]);  bmm_71 = None
    permute_522: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_599, [0, 1, 3, 2]);  view_599 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_523: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_600, [0, 2, 1, 3]);  view_600 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    clone_70: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_523, memory_format = torch.contiguous_format);  permute_523 = None
    view_601: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_70, [1, 512, 768]);  clone_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_524: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_596, [0, 2, 1, 3]);  view_596 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    clone_71: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_524, memory_format = torch.contiguous_format);  permute_524 = None
    view_602: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_71, [1, 512, 768]);  clone_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:234, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_603: "f32[512, 768]" = torch.ops.aten.view.default(view_602, [512, 768]);  view_602 = None
    permute_525: "f32[768, 768]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    mm_142: "f32[512, 768]" = torch.ops.aten.mm.default(view_603, permute_525);  permute_525 = None
    permute_526: "f32[768, 512]" = torch.ops.aten.permute.default(view_603, [1, 0])
    mm_143: "f32[768, 768]" = torch.ops.aten.mm.default(permute_526, view_5);  permute_526 = view_5 = None
    permute_527: "f32[768, 768]" = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
    sum_200: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_603, [0], True);  view_603 = None
    view_604: "f32[768]" = torch.ops.aten.view.default(sum_200, [768]);  sum_200 = None
    permute_528: "f32[768, 768]" = torch.ops.aten.permute.default(permute_527, [1, 0]);  permute_527 = None
    view_605: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_142, [1, 512, 768]);  mm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:234, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_177: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_449, view_605);  mul_449 = view_605 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:199, code: return x.permute(0, 2, 1, 3)
    permute_529: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(permute_522, [0, 2, 1, 3]);  permute_522 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:198, code: x = x.view(new_x_shape)
    view_606: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_529, [1, 512, 768]);  permute_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:233, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_607: "f32[512, 768]" = torch.ops.aten.view.default(view_606, [512, 768]);  view_606 = None
    permute_530: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    mm_144: "f32[512, 768]" = torch.ops.aten.mm.default(view_607, permute_530);  permute_530 = None
    permute_531: "f32[768, 512]" = torch.ops.aten.permute.default(view_607, [1, 0])
    mm_145: "f32[768, 768]" = torch.ops.aten.mm.default(permute_531, view_2);  permute_531 = view_2 = None
    permute_532: "f32[768, 768]" = torch.ops.aten.permute.default(mm_145, [1, 0]);  mm_145 = None
    sum_201: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_607, [0], True);  view_607 = None
    view_608: "f32[768]" = torch.ops.aten.view.default(sum_201, [768]);  sum_201 = None
    permute_533: "f32[768, 768]" = torch.ops.aten.permute.default(permute_532, [1, 0]);  permute_532 = None
    view_609: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_144, [1, 512, 768]);  mm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:233, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_178: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_177, view_609);  add_177 = view_609 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:211, code: mixed_query_layer = self.query(hidden_states)
    view_610: "f32[512, 768]" = torch.ops.aten.view.default(view_601, [512, 768]);  view_601 = None
    permute_534: "f32[768, 768]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    mm_146: "f32[512, 768]" = torch.ops.aten.mm.default(view_610, permute_534);  permute_534 = None
    permute_535: "f32[768, 512]" = torch.ops.aten.permute.default(view_610, [1, 0])
    mm_147: "f32[768, 768]" = torch.ops.aten.mm.default(permute_535, view);  permute_535 = view = None
    permute_536: "f32[768, 768]" = torch.ops.aten.permute.default(mm_147, [1, 0]);  mm_147 = None
    sum_202: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_610, [0], True);  view_610 = None
    view_611: "f32[768]" = torch.ops.aten.view.default(sum_202, [768]);  sum_202 = None
    permute_537: "f32[768, 768]" = torch.ops.aten.permute.default(permute_536, [1, 0]);  permute_536 = None
    view_612: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_146, [1, 512, 768]);  mm_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:211, code: mixed_query_layer = self.query(hidden_states)
    add_179: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_178, view_612);  add_178 = view_612 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:147, code: embeddings = self.dropout(embeddings)
    convert_element_type_40: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_3, torch.float32);  getitem_3 = None
    mul_457: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_40, 1.1111111111111112);  convert_element_type_40 = None
    mul_458: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_179, mul_457);  add_179 = mul_457 = None
    clone_72: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_458, memory_format = torch.contiguous_format);  mul_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:146, code: embeddings = self.LayerNorm(embeddings)
    sub_129: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_3, getitem_1);  add_3 = getitem_1 = None
    mul_459: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_129, rsqrt);  sub_129 = None
    mul_460: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(clone_72, primals_4);  primals_4 = None
    mul_461: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_460, 768)
    sum_203: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_460, [2], True)
    mul_462: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_460, mul_459);  mul_460 = None
    sum_204: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_462, [2], True);  mul_462 = None
    mul_463: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_459, sum_204);  sum_204 = None
    sub_130: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_461, sum_203);  mul_461 = sum_203 = None
    sub_131: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_130, mul_463);  sub_130 = mul_463 = None
    div_63: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
    mul_464: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_63, sub_131);  div_63 = sub_131 = None
    mul_465: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(clone_72, mul_459);  mul_459 = None
    sum_205: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_465, [0, 1]);  mul_465 = None
    sum_206: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_72, [0, 1]);  clone_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:144, code: position_embeddings = self.position_embeddings(position_ids)
    eq: "b8[1, 512]" = torch.ops.aten.eq.Scalar(add_1, 1)
    unsqueeze_4: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
    scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_4: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_4, scalar_tensor_4, mul_464);  unsqueeze_4 = scalar_tensor_4 = None
    full_2: "f32[514, 768]" = torch.ops.aten.full.default([514, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put: "f32[514, 768]" = torch.ops.aten._unsafe_index_put.default(full_2, [add_1], where_4, True);  full_2 = add_1 = where_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:140, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    eq_1: "b8[1, 512]" = torch.ops.aten.eq.Scalar(expand, -1)
    unsqueeze_5: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_5: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_5, scalar_tensor_5, mul_464);  unsqueeze_5 = scalar_tensor_5 = None
    full_3: "f32[1, 768]" = torch.ops.aten.full.default([1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_1: "f32[1, 768]" = torch.ops.aten._unsafe_index_put.default(full_3, [expand], where_5, True);  full_3 = expand = where_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/camembert/modeling_camembert.py:139, code: inputs_embeds = self.word_embeddings(input_ids)
    eq_2: "b8[1, 512]" = torch.ops.aten.eq.Scalar(primals_205, 1)
    unsqueeze_6: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_2, -1);  eq_2 = None
    scalar_tensor_6: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_6: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_6, scalar_tensor_6, mul_464);  unsqueeze_6 = scalar_tensor_6 = mul_464 = None
    full_4: "f32[32005, 768]" = torch.ops.aten.full.default([32005, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_2: "f32[32005, 768]" = torch.ops.aten._unsafe_index_put.default(full_4, [primals_205], where_6, True);  full_4 = primals_205 = where_6 = None
    return pytree.tree_unflatten([div_24, view_267, _unsafe_index_put_2, _unsafe_index_put_1, _unsafe_index_put, sum_205, sum_206, permute_537, view_611, permute_533, view_608, permute_528, view_604, permute_516, view_592, sum_196, sum_197, permute_512, view_589, permute_508, view_586, sum_190, sum_191, permute_504, view_583, permute_500, view_580, permute_495, view_576, permute_483, view_564, sum_181, sum_182, permute_479, view_561, permute_475, view_558, sum_175, sum_176, permute_471, view_555, permute_467, view_552, permute_462, view_548, permute_450, view_536, sum_166, sum_167, permute_446, view_533, permute_442, view_530, sum_160, sum_161, permute_438, view_527, permute_434, view_524, permute_429, view_520, permute_417, view_508, sum_151, sum_152, permute_413, view_505, permute_409, view_502, sum_145, sum_146, permute_405, view_499, permute_401, view_496, permute_396, view_492, permute_384, view_480, sum_136, sum_137, permute_380, view_477, permute_376, view_474, sum_130, sum_131, permute_372, view_471, permute_368, view_468, permute_363, view_464, permute_351, view_452, sum_121, sum_122, permute_347, view_449, permute_343, view_446, sum_115, sum_116, permute_339, view_443, permute_335, view_440, permute_330, view_436, permute_318, view_424, sum_106, sum_107, permute_314, view_421, permute_310, view_418, sum_100, sum_101, permute_306, view_415, permute_302, view_412, permute_297, view_408, permute_285, view_396, sum_91, sum_92, permute_281, view_393, permute_277, view_390, sum_85, sum_86, permute_273, view_387, permute_269, view_384, permute_264, view_380, permute_252, view_368, sum_76, sum_77, permute_248, view_365, permute_244, view_362, sum_70, sum_71, permute_240, view_359, permute_236, view_356, permute_231, view_352, permute_219, view_340, sum_61, sum_62, permute_215, view_337, permute_211, view_334, sum_55, sum_56, permute_207, view_331, permute_203, view_328, permute_198, view_324, permute_186, view_312, sum_46, sum_47, permute_182, view_309, permute_178, view_306, sum_40, sum_41, permute_174, view_303, permute_170, view_300, permute_165, view_296, permute_153, view_284, sum_31, sum_32, permute_149, view_281, permute_145, view_278, sum_25, sum_26, permute_141, view_275, sum_20, sum_21, permute_137, view_272, None, None, None], self._out_spec)
    