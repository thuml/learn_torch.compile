from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[50265, 768]"; primals_2: "f32[2, 768]"; primals_3: "f32[512, 768]"; primals_4: "f32[768]"; primals_5: "f32[768]"; primals_6: "f32[768, 768]"; primals_7: "f32[768]"; primals_8: "f32[768, 768]"; primals_9: "f32[768]"; primals_10: "f32[768, 768]"; primals_11: "f32[768]"; primals_12: "f32[768, 768]"; primals_13: "f32[768]"; primals_14: "f32[768]"; primals_15: "f32[768]"; primals_16: "f32[3072, 768]"; primals_17: "f32[3072]"; primals_18: "f32[768, 3072]"; primals_19: "f32[768]"; primals_20: "f32[768]"; primals_21: "f32[768]"; primals_22: "f32[768, 768]"; primals_23: "f32[768]"; primals_24: "f32[768, 768]"; primals_25: "f32[768]"; primals_26: "f32[768, 768]"; primals_27: "f32[768]"; primals_28: "f32[768, 768]"; primals_29: "f32[768]"; primals_30: "f32[768]"; primals_31: "f32[768]"; primals_32: "f32[3072, 768]"; primals_33: "f32[3072]"; primals_34: "f32[768, 3072]"; primals_35: "f32[768]"; primals_36: "f32[768]"; primals_37: "f32[768]"; primals_38: "f32[768, 768]"; primals_39: "f32[768]"; primals_40: "f32[768, 768]"; primals_41: "f32[768]"; primals_42: "f32[768, 768]"; primals_43: "f32[768]"; primals_44: "f32[768, 768]"; primals_45: "f32[768]"; primals_46: "f32[768]"; primals_47: "f32[768]"; primals_48: "f32[3072, 768]"; primals_49: "f32[3072]"; primals_50: "f32[768, 3072]"; primals_51: "f32[768]"; primals_52: "f32[768]"; primals_53: "f32[768]"; primals_54: "f32[768, 768]"; primals_55: "f32[768]"; primals_56: "f32[768, 768]"; primals_57: "f32[768]"; primals_58: "f32[768, 768]"; primals_59: "f32[768]"; primals_60: "f32[768, 768]"; primals_61: "f32[768]"; primals_62: "f32[768]"; primals_63: "f32[768]"; primals_64: "f32[3072, 768]"; primals_65: "f32[3072]"; primals_66: "f32[768, 3072]"; primals_67: "f32[768]"; primals_68: "f32[768]"; primals_69: "f32[768]"; primals_70: "f32[768, 768]"; primals_71: "f32[768]"; primals_72: "f32[768, 768]"; primals_73: "f32[768]"; primals_74: "f32[768, 768]"; primals_75: "f32[768]"; primals_76: "f32[768, 768]"; primals_77: "f32[768]"; primals_78: "f32[768]"; primals_79: "f32[768]"; primals_80: "f32[3072, 768]"; primals_81: "f32[3072]"; primals_82: "f32[768, 3072]"; primals_83: "f32[768]"; primals_84: "f32[768]"; primals_85: "f32[768]"; primals_86: "f32[768, 768]"; primals_87: "f32[768]"; primals_88: "f32[768, 768]"; primals_89: "f32[768]"; primals_90: "f32[768, 768]"; primals_91: "f32[768]"; primals_92: "f32[768, 768]"; primals_93: "f32[768]"; primals_94: "f32[768]"; primals_95: "f32[768]"; primals_96: "f32[3072, 768]"; primals_97: "f32[3072]"; primals_98: "f32[768, 3072]"; primals_99: "f32[768]"; primals_100: "f32[768]"; primals_101: "f32[768]"; primals_102: "f32[768, 768]"; primals_103: "f32[768]"; primals_104: "f32[768, 768]"; primals_105: "f32[768]"; primals_106: "f32[768, 768]"; primals_107: "f32[768]"; primals_108: "f32[768, 768]"; primals_109: "f32[768]"; primals_110: "f32[768]"; primals_111: "f32[768]"; primals_112: "f32[3072, 768]"; primals_113: "f32[3072]"; primals_114: "f32[768, 3072]"; primals_115: "f32[768]"; primals_116: "f32[768]"; primals_117: "f32[768]"; primals_118: "f32[768, 768]"; primals_119: "f32[768]"; primals_120: "f32[768, 768]"; primals_121: "f32[768]"; primals_122: "f32[768, 768]"; primals_123: "f32[768]"; primals_124: "f32[768, 768]"; primals_125: "f32[768]"; primals_126: "f32[768]"; primals_127: "f32[768]"; primals_128: "f32[3072, 768]"; primals_129: "f32[3072]"; primals_130: "f32[768, 3072]"; primals_131: "f32[768]"; primals_132: "f32[768]"; primals_133: "f32[768]"; primals_134: "f32[768, 768]"; primals_135: "f32[768]"; primals_136: "f32[768, 768]"; primals_137: "f32[768]"; primals_138: "f32[768, 768]"; primals_139: "f32[768]"; primals_140: "f32[768, 768]"; primals_141: "f32[768]"; primals_142: "f32[768]"; primals_143: "f32[768]"; primals_144: "f32[3072, 768]"; primals_145: "f32[3072]"; primals_146: "f32[768, 3072]"; primals_147: "f32[768]"; primals_148: "f32[768]"; primals_149: "f32[768]"; primals_150: "f32[768, 768]"; primals_151: "f32[768]"; primals_152: "f32[768, 768]"; primals_153: "f32[768]"; primals_154: "f32[768, 768]"; primals_155: "f32[768]"; primals_156: "f32[768, 768]"; primals_157: "f32[768]"; primals_158: "f32[768]"; primals_159: "f32[768]"; primals_160: "f32[3072, 768]"; primals_161: "f32[3072]"; primals_162: "f32[768, 3072]"; primals_163: "f32[768]"; primals_164: "f32[768]"; primals_165: "f32[768]"; primals_166: "f32[768, 768]"; primals_167: "f32[768]"; primals_168: "f32[768, 768]"; primals_169: "f32[768]"; primals_170: "f32[768, 768]"; primals_171: "f32[768]"; primals_172: "f32[768, 768]"; primals_173: "f32[768]"; primals_174: "f32[768]"; primals_175: "f32[768]"; primals_176: "f32[3072, 768]"; primals_177: "f32[3072]"; primals_178: "f32[768, 3072]"; primals_179: "f32[768]"; primals_180: "f32[768]"; primals_181: "f32[768]"; primals_182: "f32[768, 768]"; primals_183: "f32[768]"; primals_184: "f32[768, 768]"; primals_185: "f32[768]"; primals_186: "f32[768, 768]"; primals_187: "f32[768]"; primals_188: "f32[768, 768]"; primals_189: "f32[768]"; primals_190: "f32[768]"; primals_191: "f32[768]"; primals_192: "f32[3072, 768]"; primals_193: "f32[3072]"; primals_194: "f32[768, 3072]"; primals_195: "f32[768]"; primals_196: "f32[768]"; primals_197: "f32[768]"; primals_198: "f32[2, 768]"; primals_199: "f32[2]"; primals_200: "i64[1, 512]"; primals_201: "i64[1, 512]"; primals_202: "i64[1]"; primals_203: "i64[1]"; tangents_1: "f32[]"; tangents_2: "f32[1, 512]"; tangents_3: "f32[1, 512]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, tangents_1, tangents_2, tangents_3, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:805, code: attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
    full: "f32[1, 512]" = torch.ops.aten.full.default([1, 512], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:809, code: buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
    slice_1: "i64[1, 512]" = torch.ops.aten.slice.Tensor(primals_200, 0, 0, 9223372036854775807);  primals_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:810, code: buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
    expand: "i64[1, 512]" = torch.ops.aten.expand.default(slice_1, [1, 512]);  slice_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:916, code: extended_attention_mask = attention_mask[:, None, None, :]
    slice_2: "f32[1, 512]" = torch.ops.aten.slice.Tensor(full, 0, 0, 9223372036854775807);  full = None
    unsqueeze: "f32[1, 1, 512]" = torch.ops.aten.unsqueeze.default(slice_2, 1);  slice_2 = None
    unsqueeze_1: "f32[1, 1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
    slice_3: "f32[1, 1, 1, 512]" = torch.ops.aten.slice.Tensor(unsqueeze_1, 3, 0, 9223372036854775807);  unsqueeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:928, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    sub: "f32[1, 1, 1, 512]" = torch.ops.aten.sub.Tensor(1.0, slice_3);  slice_3 = None
    mul: "f32[1, 1, 1, 512]" = torch.ops.aten.mul.Tensor(sub, -3.4028234663852886e+38);  sub = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:1558, code: mask = input_ids.ne(padding_idx).int()
    ne: "b8[1, 512]" = torch.ops.aten.ne.Scalar(primals_201, 0)
    convert_element_type: "i32[1, 512]" = torch.ops.prims.convert_element_type.default(ne, torch.int32);  ne = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:1559, code: incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    cumsum: "i64[1, 512]" = torch.ops.aten.cumsum.default(convert_element_type, 1)
    convert_element_type_1: "i32[1, 512]" = torch.ops.prims.convert_element_type.default(cumsum, torch.int32);  cumsum = None
    add: "i32[1, 512]" = torch.ops.aten.add.Tensor(convert_element_type_1, 0);  convert_element_type_1 = None
    mul_1: "i32[1, 512]" = torch.ops.aten.mul.Tensor(add, convert_element_type);  add = convert_element_type = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:1560, code: return incremental_indices.long() + padding_idx
    convert_element_type_2: "i64[1, 512]" = torch.ops.prims.convert_element_type.default(mul_1, torch.int64);  mul_1 = None
    add_1: "i64[1, 512]" = torch.ops.aten.add.Tensor(convert_element_type_2, 0);  convert_element_type_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:125, code: inputs_embeds = self.word_embeddings(input_ids)
    embedding: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(primals_1, primals_201, 0);  primals_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:126, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    embedding_1: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(primals_2, expand);  primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:128, code: embeddings = inputs_embeds + token_type_embeddings
    add_2: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:130, code: position_embeddings = self.position_embeddings(position_ids)
    embedding_2: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(primals_3, add_1, 0);  primals_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:131, code: embeddings += position_embeddings
    add_3: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_2, embedding_2);  add_2 = embedding_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:132, code: embeddings = self.LayerNorm(embeddings)
    var_mean = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 512, 1]" = var_mean[0]
    getitem_1: "f32[1, 512, 1]" = var_mean[1];  var_mean = None
    add_4: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
    rsqrt: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    sub_1: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_3, getitem_1)
    mul_2: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = None
    mul_3: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_2, primals_4);  mul_2 = None
    add_5: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_3, primals_5);  mul_3 = primals_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:133, code: embeddings = self.dropout(embeddings)
    native_dropout = torch.ops.aten.native_dropout.default(add_5, 0.1, True);  add_5 = None
    getitem_2: "f32[1, 512, 768]" = native_dropout[0]
    getitem_3: "b8[1, 512, 768]" = native_dropout[1];  native_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    view: "f32[512, 768]" = torch.ops.aten.view.default(getitem_2, [512, 768])
    permute: "f32[768, 768]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
    addmm: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_7, view, permute);  primals_7 = None
    view_1: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm, [1, 512, 768]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_2: "f32[512, 768]" = torch.ops.aten.view.default(getitem_2, [512, 768])
    permute_1: "f32[768, 768]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
    addmm_1: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_9, view_2, permute_1);  primals_9 = None
    view_3: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_1, [1, 512, 768]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_4: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_3, [1, 512, 12, 64]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_2: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_5: "f32[512, 768]" = torch.ops.aten.view.default(getitem_2, [512, 768])
    permute_3: "f32[768, 768]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
    addmm_2: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_11, view_5, permute_3);  primals_11 = None
    view_6: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_2, [1, 512, 768]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_7: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_6, [1, 512, 12, 64]);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_4: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_8: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_1, [1, 512, 12, 64]);  view_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_5: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:236, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_6: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_2, [0, 1, 3, 2]);  permute_2 = None
    expand_1: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_5, [1, 12, 512, 64]);  permute_5 = None
    view_9: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_1, [12, 512, 64]);  expand_1 = None
    expand_2: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_6, [1, 12, 64, 512]);  permute_6 = None
    view_10: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_2, [12, 64, 512]);  expand_2 = None
    bmm: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_9, view_10)
    view_11: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm, [1, 12, 512, 512]);  bmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:260, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_11, 8.0);  view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:263, code: attention_scores = attention_scores + attention_mask
    add_6: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(div, mul);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:266, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_6, [-1], True)
    sub_2: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_6, amax);  add_6 = amax = None
    exp: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
    sum_1: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div_1: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(div_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:270, code: attention_probs = self.dropout(attention_probs)
    native_dropout_1 = torch.ops.aten.native_dropout.default(div_1, 0.1, True);  div_1 = None
    getitem_4: "f32[1, 12, 512, 512]" = native_dropout_1[0]
    getitem_5: "b8[1, 12, 512, 512]" = native_dropout_1[1];  native_dropout_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:276, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_3: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(getitem_4, [1, 12, 512, 512]);  getitem_4 = None
    view_12: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_3, [12, 512, 512]);  expand_3 = None
    expand_4: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_4, [1, 12, 512, 64]);  permute_4 = None
    view_13: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_4, [12, 512, 64]);  expand_4 = None
    bmm_1: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_12, view_13)
    view_14: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_1, [1, 12, 512, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:278, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_7: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
    clone: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:280, code: context_layer = context_layer.view(new_context_layer_shape)
    view_15: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone, [1, 512, 768]);  clone = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:298, code: hidden_states = self.dense(hidden_states)
    view_16: "f32[512, 768]" = torch.ops.aten.view.default(view_15, [512, 768]);  view_15 = None
    permute_8: "f32[768, 768]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
    addmm_3: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_13, view_16, permute_8);  primals_13 = None
    view_17: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_3, [1, 512, 768]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:299, code: hidden_states = self.dropout(hidden_states)
    native_dropout_2 = torch.ops.aten.native_dropout.default(view_17, 0.1, True);  view_17 = None
    getitem_6: "f32[1, 512, 768]" = native_dropout_2[0]
    getitem_7: "b8[1, 512, 768]" = native_dropout_2[1];  native_dropout_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:300, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_7: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_6, getitem_2);  getitem_6 = getitem_2 = None
    var_mean_1 = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
    getitem_8: "f32[1, 512, 1]" = var_mean_1[0]
    getitem_9: "f32[1, 512, 1]" = var_mean_1[1];  var_mean_1 = None
    add_8: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
    rsqrt_1: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
    sub_3: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_7, getitem_9)
    mul_4: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = None
    mul_5: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_4, primals_14);  mul_4 = None
    add_9: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_5, primals_15);  mul_5 = primals_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379, code: hidden_states = self.dense(hidden_states)
    view_20: "f32[512, 3072]" = torch.ops.aten.view.default(mul_8, [512, 3072]);  mul_8 = None
    permute_10: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
    addmm_5: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_19, view_20, permute_10);  primals_19 = None
    view_21: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_5, [1, 512, 768]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380, code: hidden_states = self.dropout(hidden_states)
    native_dropout_3 = torch.ops.aten.native_dropout.default(view_21, 0.1, True);  view_21 = None
    getitem_10: "f32[1, 512, 768]" = native_dropout_3[0]
    getitem_11: "b8[1, 512, 768]" = native_dropout_3[1];  native_dropout_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:381, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_11: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_10, add_9);  getitem_10 = add_9 = None
    var_mean_2 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
    getitem_12: "f32[1, 512, 1]" = var_mean_2[0]
    getitem_13: "f32[1, 512, 1]" = var_mean_2[1];  var_mean_2 = None
    add_12: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
    rsqrt_2: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
    sub_4: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_11, getitem_13)
    mul_9: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = None
    mul_10: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_9, primals_20);  mul_9 = None
    add_13: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_10, primals_21);  mul_10 = primals_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    view_22: "f32[512, 768]" = torch.ops.aten.view.default(add_13, [512, 768])
    permute_11: "f32[768, 768]" = torch.ops.aten.permute.default(primals_22, [1, 0]);  primals_22 = None
    addmm_6: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_23, view_22, permute_11);  primals_23 = None
    view_23: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_6, [1, 512, 768]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_24: "f32[512, 768]" = torch.ops.aten.view.default(add_13, [512, 768])
    permute_12: "f32[768, 768]" = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
    addmm_7: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_25, view_24, permute_12);  primals_25 = None
    view_25: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_7, [1, 512, 768]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_26: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_25, [1, 512, 12, 64]);  view_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_13: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_27: "f32[512, 768]" = torch.ops.aten.view.default(add_13, [512, 768])
    permute_14: "f32[768, 768]" = torch.ops.aten.permute.default(primals_26, [1, 0]);  primals_26 = None
    addmm_8: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_27, view_27, permute_14);  primals_27 = None
    view_28: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_8, [1, 512, 768]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_29: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_28, [1, 512, 12, 64]);  view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_15: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_30: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_23, [1, 512, 12, 64]);  view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_16: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:236, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_17: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_13, [0, 1, 3, 2]);  permute_13 = None
    expand_5: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_16, [1, 12, 512, 64]);  permute_16 = None
    view_31: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_5, [12, 512, 64]);  expand_5 = None
    expand_6: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_17, [1, 12, 64, 512]);  permute_17 = None
    view_32: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_6, [12, 64, 512]);  expand_6 = None
    bmm_2: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_31, view_32)
    view_33: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_2, [1, 12, 512, 512]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:260, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_2: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_33, 8.0);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:263, code: attention_scores = attention_scores + attention_mask
    add_14: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(div_2, mul);  div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:266, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_1: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_14, [-1], True)
    sub_5: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_14, amax_1);  add_14 = amax_1 = None
    exp_1: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
    sum_2: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_3: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    alias_1: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(div_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:270, code: attention_probs = self.dropout(attention_probs)
    native_dropout_4 = torch.ops.aten.native_dropout.default(div_3, 0.1, True);  div_3 = None
    getitem_14: "f32[1, 12, 512, 512]" = native_dropout_4[0]
    getitem_15: "b8[1, 12, 512, 512]" = native_dropout_4[1];  native_dropout_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:276, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_7: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(getitem_14, [1, 12, 512, 512]);  getitem_14 = None
    view_34: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_7, [12, 512, 512]);  expand_7 = None
    expand_8: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_15, [1, 12, 512, 64]);  permute_15 = None
    view_35: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_8, [12, 512, 64]);  expand_8 = None
    bmm_3: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_34, view_35)
    view_36: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_3, [1, 12, 512, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:278, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_18: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None
    clone_1: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:280, code: context_layer = context_layer.view(new_context_layer_shape)
    view_37: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_1, [1, 512, 768]);  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:298, code: hidden_states = self.dense(hidden_states)
    view_38: "f32[512, 768]" = torch.ops.aten.view.default(view_37, [512, 768]);  view_37 = None
    permute_19: "f32[768, 768]" = torch.ops.aten.permute.default(primals_28, [1, 0]);  primals_28 = None
    addmm_9: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_29, view_38, permute_19);  primals_29 = None
    view_39: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_9, [1, 512, 768]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:299, code: hidden_states = self.dropout(hidden_states)
    native_dropout_5 = torch.ops.aten.native_dropout.default(view_39, 0.1, True);  view_39 = None
    getitem_16: "f32[1, 512, 768]" = native_dropout_5[0]
    getitem_17: "b8[1, 512, 768]" = native_dropout_5[1];  native_dropout_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:300, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_15: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_16, add_13);  getitem_16 = add_13 = None
    var_mean_3 = torch.ops.aten.var_mean.correction(add_15, [2], correction = 0, keepdim = True)
    getitem_18: "f32[1, 512, 1]" = var_mean_3[0]
    getitem_19: "f32[1, 512, 1]" = var_mean_3[1];  var_mean_3 = None
    add_16: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
    rsqrt_3: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    sub_6: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_15, getitem_19)
    mul_11: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_3);  sub_6 = None
    mul_12: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_11, primals_30);  mul_11 = None
    add_17: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_12, primals_31);  mul_12 = primals_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379, code: hidden_states = self.dense(hidden_states)
    view_42: "f32[512, 3072]" = torch.ops.aten.view.default(mul_15, [512, 3072]);  mul_15 = None
    permute_21: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
    addmm_11: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_35, view_42, permute_21);  primals_35 = None
    view_43: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_11, [1, 512, 768]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380, code: hidden_states = self.dropout(hidden_states)
    native_dropout_6 = torch.ops.aten.native_dropout.default(view_43, 0.1, True);  view_43 = None
    getitem_20: "f32[1, 512, 768]" = native_dropout_6[0]
    getitem_21: "b8[1, 512, 768]" = native_dropout_6[1];  native_dropout_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:381, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_19: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_20, add_17);  getitem_20 = add_17 = None
    var_mean_4 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
    getitem_22: "f32[1, 512, 1]" = var_mean_4[0]
    getitem_23: "f32[1, 512, 1]" = var_mean_4[1];  var_mean_4 = None
    add_20: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
    rsqrt_4: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
    sub_7: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_19, getitem_23)
    mul_16: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_4);  sub_7 = None
    mul_17: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_16, primals_36);  mul_16 = None
    add_21: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_17, primals_37);  mul_17 = primals_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    view_44: "f32[512, 768]" = torch.ops.aten.view.default(add_21, [512, 768])
    permute_22: "f32[768, 768]" = torch.ops.aten.permute.default(primals_38, [1, 0]);  primals_38 = None
    addmm_12: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_39, view_44, permute_22);  primals_39 = None
    view_45: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_12, [1, 512, 768]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_46: "f32[512, 768]" = torch.ops.aten.view.default(add_21, [512, 768])
    permute_23: "f32[768, 768]" = torch.ops.aten.permute.default(primals_40, [1, 0]);  primals_40 = None
    addmm_13: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_41, view_46, permute_23);  primals_41 = None
    view_47: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_13, [1, 512, 768]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_48: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_47, [1, 512, 12, 64]);  view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_24: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_49: "f32[512, 768]" = torch.ops.aten.view.default(add_21, [512, 768])
    permute_25: "f32[768, 768]" = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
    addmm_14: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_43, view_49, permute_25);  primals_43 = None
    view_50: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_14, [1, 512, 768]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_51: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_50, [1, 512, 12, 64]);  view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_26: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_52: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_45, [1, 512, 12, 64]);  view_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_27: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:236, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_28: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_24, [0, 1, 3, 2]);  permute_24 = None
    expand_9: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_27, [1, 12, 512, 64]);  permute_27 = None
    view_53: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_9, [12, 512, 64]);  expand_9 = None
    expand_10: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_28, [1, 12, 64, 512]);  permute_28 = None
    view_54: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_10, [12, 64, 512]);  expand_10 = None
    bmm_4: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_53, view_54)
    view_55: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_4, [1, 12, 512, 512]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:260, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_4: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_55, 8.0);  view_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:263, code: attention_scores = attention_scores + attention_mask
    add_22: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(div_4, mul);  div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:266, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_2: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_22, [-1], True)
    sub_8: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_22, amax_2);  add_22 = amax_2 = None
    exp_2: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_8);  sub_8 = None
    sum_3: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_5: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    alias_2: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(div_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:270, code: attention_probs = self.dropout(attention_probs)
    native_dropout_7 = torch.ops.aten.native_dropout.default(div_5, 0.1, True);  div_5 = None
    getitem_24: "f32[1, 12, 512, 512]" = native_dropout_7[0]
    getitem_25: "b8[1, 12, 512, 512]" = native_dropout_7[1];  native_dropout_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:276, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_11: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(getitem_24, [1, 12, 512, 512]);  getitem_24 = None
    view_56: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_11, [12, 512, 512]);  expand_11 = None
    expand_12: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_26, [1, 12, 512, 64]);  permute_26 = None
    view_57: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_12, [12, 512, 64]);  expand_12 = None
    bmm_5: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_56, view_57)
    view_58: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_5, [1, 12, 512, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:278, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_29: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
    clone_2: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:280, code: context_layer = context_layer.view(new_context_layer_shape)
    view_59: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_2, [1, 512, 768]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:298, code: hidden_states = self.dense(hidden_states)
    view_60: "f32[512, 768]" = torch.ops.aten.view.default(view_59, [512, 768]);  view_59 = None
    permute_30: "f32[768, 768]" = torch.ops.aten.permute.default(primals_44, [1, 0]);  primals_44 = None
    addmm_15: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_45, view_60, permute_30);  primals_45 = None
    view_61: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_15, [1, 512, 768]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:299, code: hidden_states = self.dropout(hidden_states)
    native_dropout_8 = torch.ops.aten.native_dropout.default(view_61, 0.1, True);  view_61 = None
    getitem_26: "f32[1, 512, 768]" = native_dropout_8[0]
    getitem_27: "b8[1, 512, 768]" = native_dropout_8[1];  native_dropout_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:300, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_23: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_26, add_21);  getitem_26 = add_21 = None
    var_mean_5 = torch.ops.aten.var_mean.correction(add_23, [2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 512, 1]" = var_mean_5[0]
    getitem_29: "f32[1, 512, 1]" = var_mean_5[1];  var_mean_5 = None
    add_24: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
    rsqrt_5: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
    sub_9: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_23, getitem_29)
    mul_18: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_5);  sub_9 = None
    mul_19: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_18, primals_46);  mul_18 = None
    add_25: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_19, primals_47);  mul_19 = primals_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379, code: hidden_states = self.dense(hidden_states)
    view_64: "f32[512, 3072]" = torch.ops.aten.view.default(mul_22, [512, 3072]);  mul_22 = None
    permute_32: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_50, [1, 0]);  primals_50 = None
    addmm_17: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_51, view_64, permute_32);  primals_51 = None
    view_65: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_17, [1, 512, 768]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380, code: hidden_states = self.dropout(hidden_states)
    native_dropout_9 = torch.ops.aten.native_dropout.default(view_65, 0.1, True);  view_65 = None
    getitem_30: "f32[1, 512, 768]" = native_dropout_9[0]
    getitem_31: "b8[1, 512, 768]" = native_dropout_9[1];  native_dropout_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:381, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_27: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_30, add_25);  getitem_30 = add_25 = None
    var_mean_6 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
    getitem_32: "f32[1, 512, 1]" = var_mean_6[0]
    getitem_33: "f32[1, 512, 1]" = var_mean_6[1];  var_mean_6 = None
    add_28: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
    rsqrt_6: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    sub_10: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_27, getitem_33)
    mul_23: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_6);  sub_10 = None
    mul_24: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_23, primals_52);  mul_23 = None
    add_29: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_24, primals_53);  mul_24 = primals_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    view_66: "f32[512, 768]" = torch.ops.aten.view.default(add_29, [512, 768])
    permute_33: "f32[768, 768]" = torch.ops.aten.permute.default(primals_54, [1, 0]);  primals_54 = None
    addmm_18: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_55, view_66, permute_33);  primals_55 = None
    view_67: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_18, [1, 512, 768]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_68: "f32[512, 768]" = torch.ops.aten.view.default(add_29, [512, 768])
    permute_34: "f32[768, 768]" = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
    addmm_19: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_57, view_68, permute_34);  primals_57 = None
    view_69: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_19, [1, 512, 768]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_70: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_69, [1, 512, 12, 64]);  view_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_35: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_71: "f32[512, 768]" = torch.ops.aten.view.default(add_29, [512, 768])
    permute_36: "f32[768, 768]" = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
    addmm_20: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_59, view_71, permute_36);  primals_59 = None
    view_72: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_20, [1, 512, 768]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_73: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_72, [1, 512, 12, 64]);  view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_37: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_74: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_67, [1, 512, 12, 64]);  view_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_38: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:236, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_39: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_35, [0, 1, 3, 2]);  permute_35 = None
    expand_13: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_38, [1, 12, 512, 64]);  permute_38 = None
    view_75: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_13, [12, 512, 64]);  expand_13 = None
    expand_14: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_39, [1, 12, 64, 512]);  permute_39 = None
    view_76: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_14, [12, 64, 512]);  expand_14 = None
    bmm_6: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_75, view_76)
    view_77: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_6, [1, 12, 512, 512]);  bmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:260, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_6: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_77, 8.0);  view_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:263, code: attention_scores = attention_scores + attention_mask
    add_30: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(div_6, mul);  div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:266, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_3: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_30, [-1], True)
    sub_11: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_30, amax_3);  add_30 = amax_3 = None
    exp_3: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
    sum_4: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_7: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    alias_3: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(div_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:270, code: attention_probs = self.dropout(attention_probs)
    native_dropout_10 = torch.ops.aten.native_dropout.default(div_7, 0.1, True);  div_7 = None
    getitem_34: "f32[1, 12, 512, 512]" = native_dropout_10[0]
    getitem_35: "b8[1, 12, 512, 512]" = native_dropout_10[1];  native_dropout_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:276, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_15: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(getitem_34, [1, 12, 512, 512]);  getitem_34 = None
    view_78: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_15, [12, 512, 512]);  expand_15 = None
    expand_16: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_37, [1, 12, 512, 64]);  permute_37 = None
    view_79: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_16, [12, 512, 64]);  expand_16 = None
    bmm_7: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_78, view_79)
    view_80: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_7, [1, 12, 512, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:278, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_40: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_80, [0, 2, 1, 3]);  view_80 = None
    clone_3: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:280, code: context_layer = context_layer.view(new_context_layer_shape)
    view_81: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_3, [1, 512, 768]);  clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:298, code: hidden_states = self.dense(hidden_states)
    view_82: "f32[512, 768]" = torch.ops.aten.view.default(view_81, [512, 768]);  view_81 = None
    permute_41: "f32[768, 768]" = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
    addmm_21: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_61, view_82, permute_41);  primals_61 = None
    view_83: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_21, [1, 512, 768]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:299, code: hidden_states = self.dropout(hidden_states)
    native_dropout_11 = torch.ops.aten.native_dropout.default(view_83, 0.1, True);  view_83 = None
    getitem_36: "f32[1, 512, 768]" = native_dropout_11[0]
    getitem_37: "b8[1, 512, 768]" = native_dropout_11[1];  native_dropout_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:300, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_31: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_36, add_29);  getitem_36 = add_29 = None
    var_mean_7 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
    getitem_38: "f32[1, 512, 1]" = var_mean_7[0]
    getitem_39: "f32[1, 512, 1]" = var_mean_7[1];  var_mean_7 = None
    add_32: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
    rsqrt_7: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    sub_12: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_31, getitem_39)
    mul_25: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_7);  sub_12 = None
    mul_26: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_25, primals_62);  mul_25 = None
    add_33: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_26, primals_63);  mul_26 = primals_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379, code: hidden_states = self.dense(hidden_states)
    view_86: "f32[512, 3072]" = torch.ops.aten.view.default(mul_29, [512, 3072]);  mul_29 = None
    permute_43: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
    addmm_23: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_67, view_86, permute_43);  primals_67 = None
    view_87: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_23, [1, 512, 768]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380, code: hidden_states = self.dropout(hidden_states)
    native_dropout_12 = torch.ops.aten.native_dropout.default(view_87, 0.1, True);  view_87 = None
    getitem_40: "f32[1, 512, 768]" = native_dropout_12[0]
    getitem_41: "b8[1, 512, 768]" = native_dropout_12[1];  native_dropout_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:381, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_35: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_40, add_33);  getitem_40 = add_33 = None
    var_mean_8 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
    getitem_42: "f32[1, 512, 1]" = var_mean_8[0]
    getitem_43: "f32[1, 512, 1]" = var_mean_8[1];  var_mean_8 = None
    add_36: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
    rsqrt_8: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    sub_13: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_43)
    mul_30: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_8);  sub_13 = None
    mul_31: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_30, primals_68);  mul_30 = None
    add_37: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_31, primals_69);  mul_31 = primals_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    view_88: "f32[512, 768]" = torch.ops.aten.view.default(add_37, [512, 768])
    permute_44: "f32[768, 768]" = torch.ops.aten.permute.default(primals_70, [1, 0]);  primals_70 = None
    addmm_24: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_71, view_88, permute_44);  primals_71 = None
    view_89: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_24, [1, 512, 768]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_90: "f32[512, 768]" = torch.ops.aten.view.default(add_37, [512, 768])
    permute_45: "f32[768, 768]" = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
    addmm_25: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_73, view_90, permute_45);  primals_73 = None
    view_91: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_25, [1, 512, 768]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_92: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_91, [1, 512, 12, 64]);  view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_46: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_93: "f32[512, 768]" = torch.ops.aten.view.default(add_37, [512, 768])
    permute_47: "f32[768, 768]" = torch.ops.aten.permute.default(primals_74, [1, 0]);  primals_74 = None
    addmm_26: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_75, view_93, permute_47);  primals_75 = None
    view_94: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_26, [1, 512, 768]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_95: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_94, [1, 512, 12, 64]);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_48: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_96: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_89, [1, 512, 12, 64]);  view_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_49: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:236, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_50: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_46, [0, 1, 3, 2]);  permute_46 = None
    expand_17: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_49, [1, 12, 512, 64]);  permute_49 = None
    view_97: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_17, [12, 512, 64]);  expand_17 = None
    expand_18: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_50, [1, 12, 64, 512]);  permute_50 = None
    view_98: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_18, [12, 64, 512]);  expand_18 = None
    bmm_8: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_97, view_98)
    view_99: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_8, [1, 12, 512, 512]);  bmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:260, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_8: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_99, 8.0);  view_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:263, code: attention_scores = attention_scores + attention_mask
    add_38: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(div_8, mul);  div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:266, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_4: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_38, [-1], True)
    sub_14: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_38, amax_4);  add_38 = amax_4 = None
    exp_4: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
    sum_5: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_9: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    alias_4: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(div_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:270, code: attention_probs = self.dropout(attention_probs)
    native_dropout_13 = torch.ops.aten.native_dropout.default(div_9, 0.1, True);  div_9 = None
    getitem_44: "f32[1, 12, 512, 512]" = native_dropout_13[0]
    getitem_45: "b8[1, 12, 512, 512]" = native_dropout_13[1];  native_dropout_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:276, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_19: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(getitem_44, [1, 12, 512, 512]);  getitem_44 = None
    view_100: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_19, [12, 512, 512]);  expand_19 = None
    expand_20: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_48, [1, 12, 512, 64]);  permute_48 = None
    view_101: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_20, [12, 512, 64]);  expand_20 = None
    bmm_9: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_100, view_101)
    view_102: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_9, [1, 12, 512, 64]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:278, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_51: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
    clone_4: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:280, code: context_layer = context_layer.view(new_context_layer_shape)
    view_103: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_4, [1, 512, 768]);  clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:298, code: hidden_states = self.dense(hidden_states)
    view_104: "f32[512, 768]" = torch.ops.aten.view.default(view_103, [512, 768]);  view_103 = None
    permute_52: "f32[768, 768]" = torch.ops.aten.permute.default(primals_76, [1, 0]);  primals_76 = None
    addmm_27: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_77, view_104, permute_52);  primals_77 = None
    view_105: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_27, [1, 512, 768]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:299, code: hidden_states = self.dropout(hidden_states)
    native_dropout_14 = torch.ops.aten.native_dropout.default(view_105, 0.1, True);  view_105 = None
    getitem_46: "f32[1, 512, 768]" = native_dropout_14[0]
    getitem_47: "b8[1, 512, 768]" = native_dropout_14[1];  native_dropout_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:300, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_39: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_46, add_37);  getitem_46 = add_37 = None
    var_mean_9 = torch.ops.aten.var_mean.correction(add_39, [2], correction = 0, keepdim = True)
    getitem_48: "f32[1, 512, 1]" = var_mean_9[0]
    getitem_49: "f32[1, 512, 1]" = var_mean_9[1];  var_mean_9 = None
    add_40: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
    rsqrt_9: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
    sub_15: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_39, getitem_49)
    mul_32: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_9);  sub_15 = None
    mul_33: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_32, primals_78);  mul_32 = None
    add_41: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_33, primals_79);  mul_33 = primals_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379, code: hidden_states = self.dense(hidden_states)
    view_108: "f32[512, 3072]" = torch.ops.aten.view.default(mul_36, [512, 3072]);  mul_36 = None
    permute_54: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_82, [1, 0]);  primals_82 = None
    addmm_29: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_83, view_108, permute_54);  primals_83 = None
    view_109: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_29, [1, 512, 768]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380, code: hidden_states = self.dropout(hidden_states)
    native_dropout_15 = torch.ops.aten.native_dropout.default(view_109, 0.1, True);  view_109 = None
    getitem_50: "f32[1, 512, 768]" = native_dropout_15[0]
    getitem_51: "b8[1, 512, 768]" = native_dropout_15[1];  native_dropout_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:381, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_43: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_50, add_41);  getitem_50 = add_41 = None
    var_mean_10 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
    getitem_52: "f32[1, 512, 1]" = var_mean_10[0]
    getitem_53: "f32[1, 512, 1]" = var_mean_10[1];  var_mean_10 = None
    add_44: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-12);  getitem_52 = None
    rsqrt_10: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
    sub_16: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_43, getitem_53)
    mul_37: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_10);  sub_16 = None
    mul_38: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_37, primals_84);  mul_37 = None
    add_45: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_38, primals_85);  mul_38 = primals_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    view_110: "f32[512, 768]" = torch.ops.aten.view.default(add_45, [512, 768])
    permute_55: "f32[768, 768]" = torch.ops.aten.permute.default(primals_86, [1, 0]);  primals_86 = None
    addmm_30: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_87, view_110, permute_55);  primals_87 = None
    view_111: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_30, [1, 512, 768]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_112: "f32[512, 768]" = torch.ops.aten.view.default(add_45, [512, 768])
    permute_56: "f32[768, 768]" = torch.ops.aten.permute.default(primals_88, [1, 0]);  primals_88 = None
    addmm_31: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_89, view_112, permute_56);  primals_89 = None
    view_113: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_31, [1, 512, 768]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_114: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_113, [1, 512, 12, 64]);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_57: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_115: "f32[512, 768]" = torch.ops.aten.view.default(add_45, [512, 768])
    permute_58: "f32[768, 768]" = torch.ops.aten.permute.default(primals_90, [1, 0]);  primals_90 = None
    addmm_32: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_91, view_115, permute_58);  primals_91 = None
    view_116: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_32, [1, 512, 768]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_117: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_116, [1, 512, 12, 64]);  view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_59: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_118: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_111, [1, 512, 12, 64]);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_60: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:236, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_61: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_57, [0, 1, 3, 2]);  permute_57 = None
    expand_21: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_60, [1, 12, 512, 64]);  permute_60 = None
    view_119: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_21, [12, 512, 64]);  expand_21 = None
    expand_22: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_61, [1, 12, 64, 512]);  permute_61 = None
    view_120: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_22, [12, 64, 512]);  expand_22 = None
    bmm_10: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_119, view_120)
    view_121: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_10, [1, 12, 512, 512]);  bmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:260, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_10: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_121, 8.0);  view_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:263, code: attention_scores = attention_scores + attention_mask
    add_46: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(div_10, mul);  div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:266, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_5: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_46, [-1], True)
    sub_17: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_46, amax_5);  add_46 = amax_5 = None
    exp_5: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
    sum_6: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_11: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    alias_5: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(div_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:270, code: attention_probs = self.dropout(attention_probs)
    native_dropout_16 = torch.ops.aten.native_dropout.default(div_11, 0.1, True);  div_11 = None
    getitem_54: "f32[1, 12, 512, 512]" = native_dropout_16[0]
    getitem_55: "b8[1, 12, 512, 512]" = native_dropout_16[1];  native_dropout_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:276, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_23: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(getitem_54, [1, 12, 512, 512]);  getitem_54 = None
    view_122: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_23, [12, 512, 512]);  expand_23 = None
    expand_24: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_59, [1, 12, 512, 64]);  permute_59 = None
    view_123: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_24, [12, 512, 64]);  expand_24 = None
    bmm_11: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_122, view_123)
    view_124: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_11, [1, 12, 512, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:278, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_62: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None
    clone_5: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:280, code: context_layer = context_layer.view(new_context_layer_shape)
    view_125: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_5, [1, 512, 768]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:298, code: hidden_states = self.dense(hidden_states)
    view_126: "f32[512, 768]" = torch.ops.aten.view.default(view_125, [512, 768]);  view_125 = None
    permute_63: "f32[768, 768]" = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
    addmm_33: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_93, view_126, permute_63);  primals_93 = None
    view_127: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_33, [1, 512, 768]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:299, code: hidden_states = self.dropout(hidden_states)
    native_dropout_17 = torch.ops.aten.native_dropout.default(view_127, 0.1, True);  view_127 = None
    getitem_56: "f32[1, 512, 768]" = native_dropout_17[0]
    getitem_57: "b8[1, 512, 768]" = native_dropout_17[1];  native_dropout_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:300, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_47: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_56, add_45);  getitem_56 = add_45 = None
    var_mean_11 = torch.ops.aten.var_mean.correction(add_47, [2], correction = 0, keepdim = True)
    getitem_58: "f32[1, 512, 1]" = var_mean_11[0]
    getitem_59: "f32[1, 512, 1]" = var_mean_11[1];  var_mean_11 = None
    add_48: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-12);  getitem_58 = None
    rsqrt_11: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
    sub_18: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_47, getitem_59)
    mul_39: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_11);  sub_18 = None
    mul_40: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_39, primals_94);  mul_39 = None
    add_49: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_40, primals_95);  mul_40 = primals_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379, code: hidden_states = self.dense(hidden_states)
    view_130: "f32[512, 3072]" = torch.ops.aten.view.default(mul_43, [512, 3072]);  mul_43 = None
    permute_65: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_98, [1, 0]);  primals_98 = None
    addmm_35: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_99, view_130, permute_65);  primals_99 = None
    view_131: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_35, [1, 512, 768]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380, code: hidden_states = self.dropout(hidden_states)
    native_dropout_18 = torch.ops.aten.native_dropout.default(view_131, 0.1, True);  view_131 = None
    getitem_60: "f32[1, 512, 768]" = native_dropout_18[0]
    getitem_61: "b8[1, 512, 768]" = native_dropout_18[1];  native_dropout_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:381, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_51: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_60, add_49);  getitem_60 = add_49 = None
    var_mean_12 = torch.ops.aten.var_mean.correction(add_51, [2], correction = 0, keepdim = True)
    getitem_62: "f32[1, 512, 1]" = var_mean_12[0]
    getitem_63: "f32[1, 512, 1]" = var_mean_12[1];  var_mean_12 = None
    add_52: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-12);  getitem_62 = None
    rsqrt_12: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    sub_19: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_51, getitem_63)
    mul_44: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_12);  sub_19 = None
    mul_45: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_44, primals_100);  mul_44 = None
    add_53: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_45, primals_101);  mul_45 = primals_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    view_132: "f32[512, 768]" = torch.ops.aten.view.default(add_53, [512, 768])
    permute_66: "f32[768, 768]" = torch.ops.aten.permute.default(primals_102, [1, 0]);  primals_102 = None
    addmm_36: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_103, view_132, permute_66);  primals_103 = None
    view_133: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_36, [1, 512, 768]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_134: "f32[512, 768]" = torch.ops.aten.view.default(add_53, [512, 768])
    permute_67: "f32[768, 768]" = torch.ops.aten.permute.default(primals_104, [1, 0]);  primals_104 = None
    addmm_37: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_105, view_134, permute_67);  primals_105 = None
    view_135: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_37, [1, 512, 768]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_136: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_135, [1, 512, 12, 64]);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_68: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_137: "f32[512, 768]" = torch.ops.aten.view.default(add_53, [512, 768])
    permute_69: "f32[768, 768]" = torch.ops.aten.permute.default(primals_106, [1, 0]);  primals_106 = None
    addmm_38: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_107, view_137, permute_69);  primals_107 = None
    view_138: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_38, [1, 512, 768]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_139: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_138, [1, 512, 12, 64]);  view_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_70: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_139, [0, 2, 1, 3]);  view_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_140: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_133, [1, 512, 12, 64]);  view_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_71: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:236, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_72: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_68, [0, 1, 3, 2]);  permute_68 = None
    expand_25: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_71, [1, 12, 512, 64]);  permute_71 = None
    view_141: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_25, [12, 512, 64]);  expand_25 = None
    expand_26: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_72, [1, 12, 64, 512]);  permute_72 = None
    view_142: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_26, [12, 64, 512]);  expand_26 = None
    bmm_12: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_141, view_142)
    view_143: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_12, [1, 12, 512, 512]);  bmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:260, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_12: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_143, 8.0);  view_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:263, code: attention_scores = attention_scores + attention_mask
    add_54: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(div_12, mul);  div_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:266, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_6: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_54, [-1], True)
    sub_20: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_54, amax_6);  add_54 = amax_6 = None
    exp_6: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
    sum_7: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_13: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    alias_6: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(div_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:270, code: attention_probs = self.dropout(attention_probs)
    native_dropout_19 = torch.ops.aten.native_dropout.default(div_13, 0.1, True);  div_13 = None
    getitem_64: "f32[1, 12, 512, 512]" = native_dropout_19[0]
    getitem_65: "b8[1, 12, 512, 512]" = native_dropout_19[1];  native_dropout_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:276, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_27: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(getitem_64, [1, 12, 512, 512]);  getitem_64 = None
    view_144: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_27, [12, 512, 512]);  expand_27 = None
    expand_28: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_70, [1, 12, 512, 64]);  permute_70 = None
    view_145: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_28, [12, 512, 64]);  expand_28 = None
    bmm_13: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_144, view_145)
    view_146: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_13, [1, 12, 512, 64]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:278, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_73: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
    clone_6: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:280, code: context_layer = context_layer.view(new_context_layer_shape)
    view_147: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_6, [1, 512, 768]);  clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:298, code: hidden_states = self.dense(hidden_states)
    view_148: "f32[512, 768]" = torch.ops.aten.view.default(view_147, [512, 768]);  view_147 = None
    permute_74: "f32[768, 768]" = torch.ops.aten.permute.default(primals_108, [1, 0]);  primals_108 = None
    addmm_39: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_109, view_148, permute_74);  primals_109 = None
    view_149: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_39, [1, 512, 768]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:299, code: hidden_states = self.dropout(hidden_states)
    native_dropout_20 = torch.ops.aten.native_dropout.default(view_149, 0.1, True);  view_149 = None
    getitem_66: "f32[1, 512, 768]" = native_dropout_20[0]
    getitem_67: "b8[1, 512, 768]" = native_dropout_20[1];  native_dropout_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:300, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_55: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_66, add_53);  getitem_66 = add_53 = None
    var_mean_13 = torch.ops.aten.var_mean.correction(add_55, [2], correction = 0, keepdim = True)
    getitem_68: "f32[1, 512, 1]" = var_mean_13[0]
    getitem_69: "f32[1, 512, 1]" = var_mean_13[1];  var_mean_13 = None
    add_56: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-12);  getitem_68 = None
    rsqrt_13: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    sub_21: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_55, getitem_69)
    mul_46: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_13);  sub_21 = None
    mul_47: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_46, primals_110);  mul_46 = None
    add_57: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_47, primals_111);  mul_47 = primals_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379, code: hidden_states = self.dense(hidden_states)
    view_152: "f32[512, 3072]" = torch.ops.aten.view.default(mul_50, [512, 3072]);  mul_50 = None
    permute_76: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_114, [1, 0]);  primals_114 = None
    addmm_41: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_115, view_152, permute_76);  primals_115 = None
    view_153: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_41, [1, 512, 768]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380, code: hidden_states = self.dropout(hidden_states)
    native_dropout_21 = torch.ops.aten.native_dropout.default(view_153, 0.1, True);  view_153 = None
    getitem_70: "f32[1, 512, 768]" = native_dropout_21[0]
    getitem_71: "b8[1, 512, 768]" = native_dropout_21[1];  native_dropout_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:381, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_59: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_70, add_57);  getitem_70 = add_57 = None
    var_mean_14 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
    getitem_72: "f32[1, 512, 1]" = var_mean_14[0]
    getitem_73: "f32[1, 512, 1]" = var_mean_14[1];  var_mean_14 = None
    add_60: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-12);  getitem_72 = None
    rsqrt_14: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    sub_22: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_59, getitem_73)
    mul_51: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_14);  sub_22 = None
    mul_52: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_51, primals_116);  mul_51 = None
    add_61: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_52, primals_117);  mul_52 = primals_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    view_154: "f32[512, 768]" = torch.ops.aten.view.default(add_61, [512, 768])
    permute_77: "f32[768, 768]" = torch.ops.aten.permute.default(primals_118, [1, 0]);  primals_118 = None
    addmm_42: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_119, view_154, permute_77);  primals_119 = None
    view_155: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_42, [1, 512, 768]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_156: "f32[512, 768]" = torch.ops.aten.view.default(add_61, [512, 768])
    permute_78: "f32[768, 768]" = torch.ops.aten.permute.default(primals_120, [1, 0]);  primals_120 = None
    addmm_43: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_121, view_156, permute_78);  primals_121 = None
    view_157: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_43, [1, 512, 768]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_158: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_157, [1, 512, 12, 64]);  view_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_79: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_158, [0, 2, 1, 3]);  view_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_159: "f32[512, 768]" = torch.ops.aten.view.default(add_61, [512, 768])
    permute_80: "f32[768, 768]" = torch.ops.aten.permute.default(primals_122, [1, 0]);  primals_122 = None
    addmm_44: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_123, view_159, permute_80);  primals_123 = None
    view_160: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_44, [1, 512, 768]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_161: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_160, [1, 512, 12, 64]);  view_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_81: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_161, [0, 2, 1, 3]);  view_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_162: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_155, [1, 512, 12, 64]);  view_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_82: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:236, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_83: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_79, [0, 1, 3, 2]);  permute_79 = None
    expand_29: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_82, [1, 12, 512, 64]);  permute_82 = None
    view_163: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_29, [12, 512, 64]);  expand_29 = None
    expand_30: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_83, [1, 12, 64, 512]);  permute_83 = None
    view_164: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_30, [12, 64, 512]);  expand_30 = None
    bmm_14: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_163, view_164)
    view_165: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_14, [1, 12, 512, 512]);  bmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:260, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_14: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_165, 8.0);  view_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:263, code: attention_scores = attention_scores + attention_mask
    add_62: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(div_14, mul);  div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:266, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_7: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_62, [-1], True)
    sub_23: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_62, amax_7);  add_62 = amax_7 = None
    exp_7: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_23);  sub_23 = None
    sum_8: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_15: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    alias_7: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(div_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:270, code: attention_probs = self.dropout(attention_probs)
    native_dropout_22 = torch.ops.aten.native_dropout.default(div_15, 0.1, True);  div_15 = None
    getitem_74: "f32[1, 12, 512, 512]" = native_dropout_22[0]
    getitem_75: "b8[1, 12, 512, 512]" = native_dropout_22[1];  native_dropout_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:276, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_31: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(getitem_74, [1, 12, 512, 512]);  getitem_74 = None
    view_166: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_31, [12, 512, 512]);  expand_31 = None
    expand_32: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_81, [1, 12, 512, 64]);  permute_81 = None
    view_167: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_32, [12, 512, 64]);  expand_32 = None
    bmm_15: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_166, view_167)
    view_168: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_15, [1, 12, 512, 64]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:278, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_84: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
    clone_7: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:280, code: context_layer = context_layer.view(new_context_layer_shape)
    view_169: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_7, [1, 512, 768]);  clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:298, code: hidden_states = self.dense(hidden_states)
    view_170: "f32[512, 768]" = torch.ops.aten.view.default(view_169, [512, 768]);  view_169 = None
    permute_85: "f32[768, 768]" = torch.ops.aten.permute.default(primals_124, [1, 0]);  primals_124 = None
    addmm_45: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_125, view_170, permute_85);  primals_125 = None
    view_171: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_45, [1, 512, 768]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:299, code: hidden_states = self.dropout(hidden_states)
    native_dropout_23 = torch.ops.aten.native_dropout.default(view_171, 0.1, True);  view_171 = None
    getitem_76: "f32[1, 512, 768]" = native_dropout_23[0]
    getitem_77: "b8[1, 512, 768]" = native_dropout_23[1];  native_dropout_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:300, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_63: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_76, add_61);  getitem_76 = add_61 = None
    var_mean_15 = torch.ops.aten.var_mean.correction(add_63, [2], correction = 0, keepdim = True)
    getitem_78: "f32[1, 512, 1]" = var_mean_15[0]
    getitem_79: "f32[1, 512, 1]" = var_mean_15[1];  var_mean_15 = None
    add_64: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-12);  getitem_78 = None
    rsqrt_15: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
    sub_24: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_63, getitem_79)
    mul_53: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_15);  sub_24 = None
    mul_54: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_53, primals_126);  mul_53 = None
    add_65: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_54, primals_127);  mul_54 = primals_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379, code: hidden_states = self.dense(hidden_states)
    view_174: "f32[512, 3072]" = torch.ops.aten.view.default(mul_57, [512, 3072]);  mul_57 = None
    permute_87: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_130, [1, 0]);  primals_130 = None
    addmm_47: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_131, view_174, permute_87);  primals_131 = None
    view_175: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_47, [1, 512, 768]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380, code: hidden_states = self.dropout(hidden_states)
    native_dropout_24 = torch.ops.aten.native_dropout.default(view_175, 0.1, True);  view_175 = None
    getitem_80: "f32[1, 512, 768]" = native_dropout_24[0]
    getitem_81: "b8[1, 512, 768]" = native_dropout_24[1];  native_dropout_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:381, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_67: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_80, add_65);  getitem_80 = add_65 = None
    var_mean_16 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
    getitem_82: "f32[1, 512, 1]" = var_mean_16[0]
    getitem_83: "f32[1, 512, 1]" = var_mean_16[1];  var_mean_16 = None
    add_68: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-12);  getitem_82 = None
    rsqrt_16: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
    sub_25: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_67, getitem_83)
    mul_58: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_16);  sub_25 = None
    mul_59: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_58, primals_132);  mul_58 = None
    add_69: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_59, primals_133);  mul_59 = primals_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    view_176: "f32[512, 768]" = torch.ops.aten.view.default(add_69, [512, 768])
    permute_88: "f32[768, 768]" = torch.ops.aten.permute.default(primals_134, [1, 0]);  primals_134 = None
    addmm_48: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_135, view_176, permute_88);  primals_135 = None
    view_177: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_48, [1, 512, 768]);  addmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_178: "f32[512, 768]" = torch.ops.aten.view.default(add_69, [512, 768])
    permute_89: "f32[768, 768]" = torch.ops.aten.permute.default(primals_136, [1, 0]);  primals_136 = None
    addmm_49: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_137, view_178, permute_89);  primals_137 = None
    view_179: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_49, [1, 512, 768]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_180: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_179, [1, 512, 12, 64]);  view_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_90: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_180, [0, 2, 1, 3]);  view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_181: "f32[512, 768]" = torch.ops.aten.view.default(add_69, [512, 768])
    permute_91: "f32[768, 768]" = torch.ops.aten.permute.default(primals_138, [1, 0]);  primals_138 = None
    addmm_50: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_139, view_181, permute_91);  primals_139 = None
    view_182: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_50, [1, 512, 768]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_183: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_182, [1, 512, 12, 64]);  view_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_92: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_184: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_177, [1, 512, 12, 64]);  view_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_93: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:236, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_94: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_90, [0, 1, 3, 2]);  permute_90 = None
    expand_33: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_93, [1, 12, 512, 64]);  permute_93 = None
    view_185: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_33, [12, 512, 64]);  expand_33 = None
    expand_34: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_94, [1, 12, 64, 512]);  permute_94 = None
    view_186: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_34, [12, 64, 512]);  expand_34 = None
    bmm_16: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_185, view_186)
    view_187: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_16, [1, 12, 512, 512]);  bmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:260, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_16: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_187, 8.0);  view_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:263, code: attention_scores = attention_scores + attention_mask
    add_70: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(div_16, mul);  div_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:266, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_8: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_70, [-1], True)
    sub_26: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_70, amax_8);  add_70 = amax_8 = None
    exp_8: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_26);  sub_26 = None
    sum_9: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_17: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    alias_8: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(div_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:270, code: attention_probs = self.dropout(attention_probs)
    native_dropout_25 = torch.ops.aten.native_dropout.default(div_17, 0.1, True);  div_17 = None
    getitem_84: "f32[1, 12, 512, 512]" = native_dropout_25[0]
    getitem_85: "b8[1, 12, 512, 512]" = native_dropout_25[1];  native_dropout_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:276, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_35: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(getitem_84, [1, 12, 512, 512]);  getitem_84 = None
    view_188: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_35, [12, 512, 512]);  expand_35 = None
    expand_36: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_92, [1, 12, 512, 64]);  permute_92 = None
    view_189: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_36, [12, 512, 64]);  expand_36 = None
    bmm_17: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_188, view_189)
    view_190: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_17, [1, 12, 512, 64]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:278, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_95: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_190, [0, 2, 1, 3]);  view_190 = None
    clone_8: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:280, code: context_layer = context_layer.view(new_context_layer_shape)
    view_191: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_8, [1, 512, 768]);  clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:298, code: hidden_states = self.dense(hidden_states)
    view_192: "f32[512, 768]" = torch.ops.aten.view.default(view_191, [512, 768]);  view_191 = None
    permute_96: "f32[768, 768]" = torch.ops.aten.permute.default(primals_140, [1, 0]);  primals_140 = None
    addmm_51: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_141, view_192, permute_96);  primals_141 = None
    view_193: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_51, [1, 512, 768]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:299, code: hidden_states = self.dropout(hidden_states)
    native_dropout_26 = torch.ops.aten.native_dropout.default(view_193, 0.1, True);  view_193 = None
    getitem_86: "f32[1, 512, 768]" = native_dropout_26[0]
    getitem_87: "b8[1, 512, 768]" = native_dropout_26[1];  native_dropout_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:300, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_71: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_86, add_69);  getitem_86 = add_69 = None
    var_mean_17 = torch.ops.aten.var_mean.correction(add_71, [2], correction = 0, keepdim = True)
    getitem_88: "f32[1, 512, 1]" = var_mean_17[0]
    getitem_89: "f32[1, 512, 1]" = var_mean_17[1];  var_mean_17 = None
    add_72: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-12);  getitem_88 = None
    rsqrt_17: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
    sub_27: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_71, getitem_89)
    mul_60: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_17);  sub_27 = None
    mul_61: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_60, primals_142);  mul_60 = None
    add_73: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_61, primals_143);  mul_61 = primals_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379, code: hidden_states = self.dense(hidden_states)
    view_196: "f32[512, 3072]" = torch.ops.aten.view.default(mul_64, [512, 3072]);  mul_64 = None
    permute_98: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_146, [1, 0]);  primals_146 = None
    addmm_53: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_147, view_196, permute_98);  primals_147 = None
    view_197: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_53, [1, 512, 768]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380, code: hidden_states = self.dropout(hidden_states)
    native_dropout_27 = torch.ops.aten.native_dropout.default(view_197, 0.1, True);  view_197 = None
    getitem_90: "f32[1, 512, 768]" = native_dropout_27[0]
    getitem_91: "b8[1, 512, 768]" = native_dropout_27[1];  native_dropout_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:381, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_75: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_90, add_73);  getitem_90 = add_73 = None
    var_mean_18 = torch.ops.aten.var_mean.correction(add_75, [2], correction = 0, keepdim = True)
    getitem_92: "f32[1, 512, 1]" = var_mean_18[0]
    getitem_93: "f32[1, 512, 1]" = var_mean_18[1];  var_mean_18 = None
    add_76: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-12);  getitem_92 = None
    rsqrt_18: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
    sub_28: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_75, getitem_93)
    mul_65: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_18);  sub_28 = None
    mul_66: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_65, primals_148);  mul_65 = None
    add_77: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_66, primals_149);  mul_66 = primals_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    view_198: "f32[512, 768]" = torch.ops.aten.view.default(add_77, [512, 768])
    permute_99: "f32[768, 768]" = torch.ops.aten.permute.default(primals_150, [1, 0]);  primals_150 = None
    addmm_54: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_151, view_198, permute_99);  primals_151 = None
    view_199: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_54, [1, 512, 768]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_200: "f32[512, 768]" = torch.ops.aten.view.default(add_77, [512, 768])
    permute_100: "f32[768, 768]" = torch.ops.aten.permute.default(primals_152, [1, 0]);  primals_152 = None
    addmm_55: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_153, view_200, permute_100);  primals_153 = None
    view_201: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_55, [1, 512, 768]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_202: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_201, [1, 512, 12, 64]);  view_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_101: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_202, [0, 2, 1, 3]);  view_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_203: "f32[512, 768]" = torch.ops.aten.view.default(add_77, [512, 768])
    permute_102: "f32[768, 768]" = torch.ops.aten.permute.default(primals_154, [1, 0]);  primals_154 = None
    addmm_56: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_155, view_203, permute_102);  primals_155 = None
    view_204: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_56, [1, 512, 768]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_205: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_204, [1, 512, 12, 64]);  view_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_103: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_205, [0, 2, 1, 3]);  view_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_206: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_199, [1, 512, 12, 64]);  view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_104: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:236, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_105: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_101, [0, 1, 3, 2]);  permute_101 = None
    expand_37: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_104, [1, 12, 512, 64]);  permute_104 = None
    view_207: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_37, [12, 512, 64]);  expand_37 = None
    expand_38: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_105, [1, 12, 64, 512]);  permute_105 = None
    view_208: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_38, [12, 64, 512]);  expand_38 = None
    bmm_18: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_207, view_208)
    view_209: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_18, [1, 12, 512, 512]);  bmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:260, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_18: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_209, 8.0);  view_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:263, code: attention_scores = attention_scores + attention_mask
    add_78: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(div_18, mul);  div_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:266, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_9: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_78, [-1], True)
    sub_29: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_78, amax_9);  add_78 = amax_9 = None
    exp_9: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_29);  sub_29 = None
    sum_10: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_19: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    alias_9: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(div_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:270, code: attention_probs = self.dropout(attention_probs)
    native_dropout_28 = torch.ops.aten.native_dropout.default(div_19, 0.1, True);  div_19 = None
    getitem_94: "f32[1, 12, 512, 512]" = native_dropout_28[0]
    getitem_95: "b8[1, 12, 512, 512]" = native_dropout_28[1];  native_dropout_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:276, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_39: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(getitem_94, [1, 12, 512, 512]);  getitem_94 = None
    view_210: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_39, [12, 512, 512]);  expand_39 = None
    expand_40: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_103, [1, 12, 512, 64]);  permute_103 = None
    view_211: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_40, [12, 512, 64]);  expand_40 = None
    bmm_19: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_210, view_211)
    view_212: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_19, [1, 12, 512, 64]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:278, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_106: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None
    clone_9: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:280, code: context_layer = context_layer.view(new_context_layer_shape)
    view_213: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_9, [1, 512, 768]);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:298, code: hidden_states = self.dense(hidden_states)
    view_214: "f32[512, 768]" = torch.ops.aten.view.default(view_213, [512, 768]);  view_213 = None
    permute_107: "f32[768, 768]" = torch.ops.aten.permute.default(primals_156, [1, 0]);  primals_156 = None
    addmm_57: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_157, view_214, permute_107);  primals_157 = None
    view_215: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_57, [1, 512, 768]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:299, code: hidden_states = self.dropout(hidden_states)
    native_dropout_29 = torch.ops.aten.native_dropout.default(view_215, 0.1, True);  view_215 = None
    getitem_96: "f32[1, 512, 768]" = native_dropout_29[0]
    getitem_97: "b8[1, 512, 768]" = native_dropout_29[1];  native_dropout_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:300, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_79: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_96, add_77);  getitem_96 = add_77 = None
    var_mean_19 = torch.ops.aten.var_mean.correction(add_79, [2], correction = 0, keepdim = True)
    getitem_98: "f32[1, 512, 1]" = var_mean_19[0]
    getitem_99: "f32[1, 512, 1]" = var_mean_19[1];  var_mean_19 = None
    add_80: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-12);  getitem_98 = None
    rsqrt_19: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
    sub_30: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_79, getitem_99)
    mul_67: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_19);  sub_30 = None
    mul_68: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_67, primals_158);  mul_67 = None
    add_81: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_68, primals_159);  mul_68 = primals_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379, code: hidden_states = self.dense(hidden_states)
    view_218: "f32[512, 3072]" = torch.ops.aten.view.default(mul_71, [512, 3072]);  mul_71 = None
    permute_109: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_162, [1, 0]);  primals_162 = None
    addmm_59: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_163, view_218, permute_109);  primals_163 = None
    view_219: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_59, [1, 512, 768]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380, code: hidden_states = self.dropout(hidden_states)
    native_dropout_30 = torch.ops.aten.native_dropout.default(view_219, 0.1, True);  view_219 = None
    getitem_100: "f32[1, 512, 768]" = native_dropout_30[0]
    getitem_101: "b8[1, 512, 768]" = native_dropout_30[1];  native_dropout_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:381, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_83: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_100, add_81);  getitem_100 = add_81 = None
    var_mean_20 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
    getitem_102: "f32[1, 512, 1]" = var_mean_20[0]
    getitem_103: "f32[1, 512, 1]" = var_mean_20[1];  var_mean_20 = None
    add_84: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-12);  getitem_102 = None
    rsqrt_20: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
    sub_31: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_83, getitem_103)
    mul_72: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_20);  sub_31 = None
    mul_73: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_72, primals_164);  mul_72 = None
    add_85: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_73, primals_165);  mul_73 = primals_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    view_220: "f32[512, 768]" = torch.ops.aten.view.default(add_85, [512, 768])
    permute_110: "f32[768, 768]" = torch.ops.aten.permute.default(primals_166, [1, 0]);  primals_166 = None
    addmm_60: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_167, view_220, permute_110);  primals_167 = None
    view_221: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_60, [1, 512, 768]);  addmm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_222: "f32[512, 768]" = torch.ops.aten.view.default(add_85, [512, 768])
    permute_111: "f32[768, 768]" = torch.ops.aten.permute.default(primals_168, [1, 0]);  primals_168 = None
    addmm_61: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_169, view_222, permute_111);  primals_169 = None
    view_223: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_61, [1, 512, 768]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_224: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_223, [1, 512, 12, 64]);  view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_112: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_225: "f32[512, 768]" = torch.ops.aten.view.default(add_85, [512, 768])
    permute_113: "f32[768, 768]" = torch.ops.aten.permute.default(primals_170, [1, 0]);  primals_170 = None
    addmm_62: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_171, view_225, permute_113);  primals_171 = None
    view_226: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_62, [1, 512, 768]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_227: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_226, [1, 512, 12, 64]);  view_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_114: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_227, [0, 2, 1, 3]);  view_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_228: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_221, [1, 512, 12, 64]);  view_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_115: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:236, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_116: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_112, [0, 1, 3, 2]);  permute_112 = None
    expand_41: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_115, [1, 12, 512, 64]);  permute_115 = None
    view_229: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_41, [12, 512, 64]);  expand_41 = None
    expand_42: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_116, [1, 12, 64, 512]);  permute_116 = None
    view_230: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_42, [12, 64, 512]);  expand_42 = None
    bmm_20: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_229, view_230)
    view_231: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_20, [1, 12, 512, 512]);  bmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:260, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_20: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_231, 8.0);  view_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:263, code: attention_scores = attention_scores + attention_mask
    add_86: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(div_20, mul);  div_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:266, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_10: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_86, [-1], True)
    sub_32: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_86, amax_10);  add_86 = amax_10 = None
    exp_10: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_32);  sub_32 = None
    sum_11: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_21: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    alias_10: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(div_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:270, code: attention_probs = self.dropout(attention_probs)
    native_dropout_31 = torch.ops.aten.native_dropout.default(div_21, 0.1, True);  div_21 = None
    getitem_104: "f32[1, 12, 512, 512]" = native_dropout_31[0]
    getitem_105: "b8[1, 12, 512, 512]" = native_dropout_31[1];  native_dropout_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:276, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_43: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(getitem_104, [1, 12, 512, 512]);  getitem_104 = None
    view_232: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_43, [12, 512, 512]);  expand_43 = None
    expand_44: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_114, [1, 12, 512, 64]);  permute_114 = None
    view_233: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_44, [12, 512, 64]);  expand_44 = None
    bmm_21: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_232, view_233)
    view_234: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_21, [1, 12, 512, 64]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:278, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_117: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_234, [0, 2, 1, 3]);  view_234 = None
    clone_10: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:280, code: context_layer = context_layer.view(new_context_layer_shape)
    view_235: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_10, [1, 512, 768]);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:298, code: hidden_states = self.dense(hidden_states)
    view_236: "f32[512, 768]" = torch.ops.aten.view.default(view_235, [512, 768]);  view_235 = None
    permute_118: "f32[768, 768]" = torch.ops.aten.permute.default(primals_172, [1, 0]);  primals_172 = None
    addmm_63: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_173, view_236, permute_118);  primals_173 = None
    view_237: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_63, [1, 512, 768]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:299, code: hidden_states = self.dropout(hidden_states)
    native_dropout_32 = torch.ops.aten.native_dropout.default(view_237, 0.1, True);  view_237 = None
    getitem_106: "f32[1, 512, 768]" = native_dropout_32[0]
    getitem_107: "b8[1, 512, 768]" = native_dropout_32[1];  native_dropout_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:300, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_87: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_106, add_85);  getitem_106 = add_85 = None
    var_mean_21 = torch.ops.aten.var_mean.correction(add_87, [2], correction = 0, keepdim = True)
    getitem_108: "f32[1, 512, 1]" = var_mean_21[0]
    getitem_109: "f32[1, 512, 1]" = var_mean_21[1];  var_mean_21 = None
    add_88: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-12);  getitem_108 = None
    rsqrt_21: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
    sub_33: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_87, getitem_109)
    mul_74: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_21);  sub_33 = None
    mul_75: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_74, primals_174);  mul_74 = None
    add_89: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_75, primals_175);  mul_75 = primals_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379, code: hidden_states = self.dense(hidden_states)
    view_240: "f32[512, 3072]" = torch.ops.aten.view.default(mul_78, [512, 3072]);  mul_78 = None
    permute_120: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_178, [1, 0]);  primals_178 = None
    addmm_65: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_179, view_240, permute_120);  primals_179 = None
    view_241: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_65, [1, 512, 768]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380, code: hidden_states = self.dropout(hidden_states)
    native_dropout_33 = torch.ops.aten.native_dropout.default(view_241, 0.1, True);  view_241 = None
    getitem_110: "f32[1, 512, 768]" = native_dropout_33[0]
    getitem_111: "b8[1, 512, 768]" = native_dropout_33[1];  native_dropout_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:381, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_91: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_110, add_89);  getitem_110 = add_89 = None
    var_mean_22 = torch.ops.aten.var_mean.correction(add_91, [2], correction = 0, keepdim = True)
    getitem_112: "f32[1, 512, 1]" = var_mean_22[0]
    getitem_113: "f32[1, 512, 1]" = var_mean_22[1];  var_mean_22 = None
    add_92: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-12);  getitem_112 = None
    rsqrt_22: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
    sub_34: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_91, getitem_113)
    mul_79: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_22);  sub_34 = None
    mul_80: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_79, primals_180);  mul_79 = None
    add_93: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_80, primals_181);  mul_80 = primals_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    view_242: "f32[512, 768]" = torch.ops.aten.view.default(add_93, [512, 768])
    permute_121: "f32[768, 768]" = torch.ops.aten.permute.default(primals_182, [1, 0]);  primals_182 = None
    addmm_66: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_183, view_242, permute_121);  primals_183 = None
    view_243: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_66, [1, 512, 768]);  addmm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_244: "f32[512, 768]" = torch.ops.aten.view.default(add_93, [512, 768])
    permute_122: "f32[768, 768]" = torch.ops.aten.permute.default(primals_184, [1, 0]);  primals_184 = None
    addmm_67: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_185, view_244, permute_122);  primals_185 = None
    view_245: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_67, [1, 512, 768]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_246: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_245, [1, 512, 12, 64]);  view_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_123: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_247: "f32[512, 768]" = torch.ops.aten.view.default(add_93, [512, 768])
    permute_124: "f32[768, 768]" = torch.ops.aten.permute.default(primals_186, [1, 0]);  primals_186 = None
    addmm_68: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_187, view_247, permute_124);  primals_187 = None
    view_248: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_68, [1, 512, 768]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_249: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_248, [1, 512, 12, 64]);  view_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_125: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_249, [0, 2, 1, 3]);  view_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_250: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_243, [1, 512, 12, 64]);  view_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_126: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:236, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_127: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(permute_123, [0, 1, 3, 2]);  permute_123 = None
    expand_45: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_126, [1, 12, 512, 64]);  permute_126 = None
    view_251: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_45, [12, 512, 64]);  expand_45 = None
    expand_46: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_127, [1, 12, 64, 512]);  permute_127 = None
    view_252: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_46, [12, 64, 512]);  expand_46 = None
    bmm_22: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_251, view_252)
    view_253: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_22, [1, 12, 512, 512]);  bmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:260, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_22: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_253, 8.0);  view_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:263, code: attention_scores = attention_scores + attention_mask
    add_94: "f32[1, 12, 512, 512]" = torch.ops.aten.add.Tensor(div_22, mul);  div_22 = mul = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:266, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_11: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(add_94, [-1], True)
    sub_35: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add_94, amax_11);  add_94 = amax_11 = None
    exp_11: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_35);  sub_35 = None
    sum_12: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_23: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    alias_11: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(div_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:270, code: attention_probs = self.dropout(attention_probs)
    native_dropout_34 = torch.ops.aten.native_dropout.default(div_23, 0.1, True);  div_23 = None
    getitem_114: "f32[1, 12, 512, 512]" = native_dropout_34[0]
    getitem_115: "b8[1, 12, 512, 512]" = native_dropout_34[1];  native_dropout_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:276, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_47: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(getitem_114, [1, 12, 512, 512]);  getitem_114 = None
    view_254: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_47, [12, 512, 512]);  expand_47 = None
    expand_48: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_125, [1, 12, 512, 64]);  permute_125 = None
    view_255: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_48, [12, 512, 64]);  expand_48 = None
    bmm_23: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_254, view_255)
    view_256: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_23, [1, 12, 512, 64]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:278, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_128: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_256, [0, 2, 1, 3]);  view_256 = None
    clone_11: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:280, code: context_layer = context_layer.view(new_context_layer_shape)
    view_257: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_11, [1, 512, 768]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:298, code: hidden_states = self.dense(hidden_states)
    view_258: "f32[512, 768]" = torch.ops.aten.view.default(view_257, [512, 768]);  view_257 = None
    permute_129: "f32[768, 768]" = torch.ops.aten.permute.default(primals_188, [1, 0]);  primals_188 = None
    addmm_69: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_189, view_258, permute_129);  primals_189 = None
    view_259: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_69, [1, 512, 768]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:299, code: hidden_states = self.dropout(hidden_states)
    native_dropout_35 = torch.ops.aten.native_dropout.default(view_259, 0.1, True);  view_259 = None
    getitem_116: "f32[1, 512, 768]" = native_dropout_35[0]
    getitem_117: "b8[1, 512, 768]" = native_dropout_35[1];  native_dropout_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:300, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_95: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_116, add_93);  getitem_116 = add_93 = None
    var_mean_23 = torch.ops.aten.var_mean.correction(add_95, [2], correction = 0, keepdim = True)
    getitem_118: "f32[1, 512, 1]" = var_mean_23[0]
    getitem_119: "f32[1, 512, 1]" = var_mean_23[1];  var_mean_23 = None
    add_96: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-12);  getitem_118 = None
    rsqrt_23: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
    sub_36: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_95, getitem_119)
    mul_81: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_23);  sub_36 = None
    mul_82: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_81, primals_190);  mul_81 = None
    add_97: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_82, primals_191);  mul_82 = primals_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379, code: hidden_states = self.dense(hidden_states)
    view_262: "f32[512, 3072]" = torch.ops.aten.view.default(mul_85, [512, 3072]);  mul_85 = None
    permute_131: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_194, [1, 0]);  primals_194 = None
    addmm_71: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_195, view_262, permute_131);  primals_195 = None
    view_263: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_71, [1, 512, 768]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380, code: hidden_states = self.dropout(hidden_states)
    native_dropout_36 = torch.ops.aten.native_dropout.default(view_263, 0.1, True);  view_263 = None
    getitem_120: "f32[1, 512, 768]" = native_dropout_36[0]
    getitem_121: "b8[1, 512, 768]" = native_dropout_36[1];  native_dropout_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:381, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_99: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_120, add_97);  getitem_120 = add_97 = None
    var_mean_24 = torch.ops.aten.var_mean.correction(add_99, [2], correction = 0, keepdim = True)
    getitem_122: "f32[1, 512, 1]" = var_mean_24[0]
    getitem_123: "f32[1, 512, 1]" = var_mean_24[1];  var_mean_24 = None
    add_100: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_122, 1e-12);  getitem_122 = None
    rsqrt_24: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
    sub_37: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_99, getitem_123)
    mul_86: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_24);  sub_37 = None
    mul_87: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_86, primals_196);  mul_86 = None
    add_101: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_87, primals_197);  mul_87 = primals_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:1512, code: logits = self.qa_outputs(sequence_output)
    view_264: "f32[512, 768]" = torch.ops.aten.view.default(add_101, [512, 768]);  add_101 = None
    permute_132: "f32[768, 2]" = torch.ops.aten.permute.default(primals_198, [1, 0]);  primals_198 = None
    addmm_72: "f32[512, 2]" = torch.ops.aten.addmm.default(primals_199, view_264, permute_132);  primals_199 = None
    view_265: "f32[1, 512, 2]" = torch.ops.aten.view.default(addmm_72, [1, 512, 2]);  addmm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:1513, code: start_logits, end_logits = logits.split(1, dim=-1)
    split_with_sizes = torch.ops.aten.split_with_sizes.default(view_265, [1, 1], 2);  view_265 = None
    getitem_124: "f32[1, 512, 1]" = split_with_sizes[0]
    getitem_125: "f32[1, 512, 1]" = split_with_sizes[1];  split_with_sizes = None
    
    # No stacktrace found for following nodes
    squeeze: "f32[1, 512]" = torch.ops.aten.squeeze.dim(getitem_124, -1);  getitem_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:1514, code: start_logits = start_logits.squeeze(-1).contiguous()
    clone_12: "f32[1, 512]" = torch.ops.aten.clone.default(squeeze, memory_format = torch.contiguous_format);  squeeze = None
    
    # No stacktrace found for following nodes
    squeeze_1: "f32[1, 512]" = torch.ops.aten.squeeze.dim(getitem_125, -1);  getitem_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:1515, code: end_logits = end_logits.squeeze(-1).contiguous()
    clone_13: "f32[1, 512]" = torch.ops.aten.clone.default(squeeze_1, memory_format = torch.contiguous_format);  squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:1526, code: start_positions = start_positions.clamp(0, ignored_index)
    clamp_min: "i64[1]" = torch.ops.aten.clamp_min.default(primals_202, 0);  primals_202 = None
    clamp_max: "i64[1]" = torch.ops.aten.clamp_max.default(clamp_min, 512);  clamp_min = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:1527, code: end_positions = end_positions.clamp(0, ignored_index)
    clamp_min_1: "i64[1]" = torch.ops.aten.clamp_min.default(primals_203, 0);  primals_203 = None
    clamp_max_1: "i64[1]" = torch.ops.aten.clamp_max.default(clamp_min_1, 512);  clamp_min_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:1530, code: start_loss = loss_fct(start_logits, start_positions)
    amax_12: "f32[1, 1]" = torch.ops.aten.amax.default(clone_12, [1], True)
    sub_38: "f32[1, 512]" = torch.ops.aten.sub.Tensor(clone_12, amax_12);  amax_12 = None
    exp_12: "f32[1, 512]" = torch.ops.aten.exp.default(sub_38)
    sum_13: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [1], True);  exp_12 = None
    log: "f32[1, 1]" = torch.ops.aten.log.default(sum_13);  sum_13 = None
    sub_39: "f32[1, 512]" = torch.ops.aten.sub.Tensor(sub_38, log);  sub_38 = log = None
    alias_12: "f32[1, 512]" = torch.ops.aten.alias.default(sub_39)
    ne_1: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max, 512)
    scalar_tensor: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0))
    where: "i64[1]" = torch.ops.aten.where.self(ne_1, clamp_max, scalar_tensor);  ne_1 = scalar_tensor = None
    unsqueeze_2: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
    gather: "f32[1, 1]" = torch.ops.aten.gather.default(sub_39, 1, unsqueeze_2);  sub_39 = unsqueeze_2 = None
    squeeze_2: "f32[1]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[1]" = torch.ops.aten.neg.default(squeeze_2);  squeeze_2 = None
    ne_2: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max, 512)
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_1: "f32[1]" = torch.ops.aten.where.self(ne_2, neg, scalar_tensor_1);  ne_2 = neg = scalar_tensor_1 = None
    ne_3: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max, 512)
    sum_14: "i64[]" = torch.ops.aten.sum.default(ne_3);  ne_3 = None
    convert_element_type_3: "f32[]" = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None
    sum_15: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
    div_24: "f32[]" = torch.ops.aten.div.Tensor(sum_15, convert_element_type_3);  sum_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:1531, code: end_loss = loss_fct(end_logits, end_positions)
    amax_13: "f32[1, 1]" = torch.ops.aten.amax.default(clone_13, [1], True)
    sub_40: "f32[1, 512]" = torch.ops.aten.sub.Tensor(clone_13, amax_13);  amax_13 = None
    exp_13: "f32[1, 512]" = torch.ops.aten.exp.default(sub_40)
    sum_16: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [1], True);  exp_13 = None
    log_1: "f32[1, 1]" = torch.ops.aten.log.default(sum_16);  sum_16 = None
    sub_41: "f32[1, 512]" = torch.ops.aten.sub.Tensor(sub_40, log_1);  sub_40 = log_1 = None
    alias_13: "f32[1, 512]" = torch.ops.aten.alias.default(sub_41)
    ne_4: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max_1, 512)
    scalar_tensor_2: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0))
    where_2: "i64[1]" = torch.ops.aten.where.self(ne_4, clamp_max_1, scalar_tensor_2);  ne_4 = scalar_tensor_2 = None
    unsqueeze_3: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(where_2, 1);  where_2 = None
    gather_1: "f32[1, 1]" = torch.ops.aten.gather.default(sub_41, 1, unsqueeze_3);  sub_41 = unsqueeze_3 = None
    squeeze_3: "f32[1]" = torch.ops.aten.squeeze.dim(gather_1, 1);  gather_1 = None
    neg_1: "f32[1]" = torch.ops.aten.neg.default(squeeze_3);  squeeze_3 = None
    ne_5: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max_1, 512)
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_3: "f32[1]" = torch.ops.aten.where.self(ne_5, neg_1, scalar_tensor_3);  ne_5 = neg_1 = scalar_tensor_3 = None
    ne_6: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max_1, 512)
    sum_17: "i64[]" = torch.ops.aten.sum.default(ne_6);  ne_6 = None
    convert_element_type_4: "f32[]" = torch.ops.prims.convert_element_type.default(sum_17, torch.float32);  sum_17 = None
    sum_18: "f32[]" = torch.ops.aten.sum.default(where_3);  where_3 = None
    div_25: "f32[]" = torch.ops.aten.div.Tensor(sum_18, convert_element_type_4);  sum_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:1532, code: total_loss = (start_loss + end_loss) / 2
    add_102: "f32[]" = torch.ops.aten.add.Tensor(div_24, div_25);  div_24 = div_25 = None
    div_26: "f32[]" = torch.ops.aten.div.Tensor(add_102, 2);  add_102 = None
    div_27: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, 2);  tangents_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:1531, code: end_loss = loss_fct(end_logits, end_positions)
    div_28: "f32[]" = torch.ops.aten.div.Tensor(div_27, convert_element_type_4);  convert_element_type_4 = None
    unsqueeze_4: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(clamp_max_1, 1);  clamp_max_1 = None
    ne_7: "b8[1, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_4, 512)
    scalar_tensor_4: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0))
    where_4: "i64[1, 1]" = torch.ops.aten.where.self(ne_7, unsqueeze_4, scalar_tensor_4);  ne_7 = scalar_tensor_4 = None
    full_1: "f32[1, 512]" = torch.ops.aten.full.default([1, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    scatter: "f32[1, 512]" = torch.ops.aten.scatter.value(full_1, 1, where_4, -1.0);  full_1 = where_4 = None
    ne_8: "b8[1, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_4, 512);  unsqueeze_4 = None
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_5: "f32[1, 1]" = torch.ops.aten.where.self(ne_8, div_28, scalar_tensor_5);  ne_8 = div_28 = scalar_tensor_5 = None
    mul_88: "f32[1, 512]" = torch.ops.aten.mul.Tensor(scatter, where_5);  scatter = where_5 = None
    alias_14: "f32[1, 512]" = torch.ops.aten.alias.default(alias_13);  alias_13 = None
    exp_14: "f32[1, 512]" = torch.ops.aten.exp.default(alias_14);  alias_14 = None
    sum_19: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(mul_88, [1], True)
    mul_89: "f32[1, 512]" = torch.ops.aten.mul.Tensor(exp_14, sum_19);  exp_14 = sum_19 = None
    sub_42: "f32[1, 512]" = torch.ops.aten.sub.Tensor(mul_88, mul_89);  mul_88 = mul_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:1531, code: end_loss = loss_fct(end_logits, end_positions)
    add_103: "f32[1, 512]" = torch.ops.aten.add.Tensor(tangents_3, sub_42);  tangents_3 = sub_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:1530, code: start_loss = loss_fct(start_logits, start_positions)
    div_29: "f32[]" = torch.ops.aten.div.Tensor(div_27, convert_element_type_3);  div_27 = convert_element_type_3 = None
    unsqueeze_5: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(clamp_max, 1);  clamp_max = None
    ne_9: "b8[1, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_5, 512)
    scalar_tensor_6: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0))
    where_6: "i64[1, 1]" = torch.ops.aten.where.self(ne_9, unsqueeze_5, scalar_tensor_6);  ne_9 = scalar_tensor_6 = None
    full_2: "f32[1, 512]" = torch.ops.aten.full.default([1, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    scatter_1: "f32[1, 512]" = torch.ops.aten.scatter.value(full_2, 1, where_6, -1.0);  full_2 = where_6 = None
    ne_10: "b8[1, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_5, 512);  unsqueeze_5 = None
    scalar_tensor_7: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_7: "f32[1, 1]" = torch.ops.aten.where.self(ne_10, div_29, scalar_tensor_7);  ne_10 = div_29 = scalar_tensor_7 = None
    mul_90: "f32[1, 512]" = torch.ops.aten.mul.Tensor(scatter_1, where_7);  scatter_1 = where_7 = None
    alias_15: "f32[1, 512]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    exp_15: "f32[1, 512]" = torch.ops.aten.exp.default(alias_15);  alias_15 = None
    sum_20: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(mul_90, [1], True)
    mul_91: "f32[1, 512]" = torch.ops.aten.mul.Tensor(exp_15, sum_20);  exp_15 = sum_20 = None
    sub_43: "f32[1, 512]" = torch.ops.aten.sub.Tensor(mul_90, mul_91);  mul_90 = mul_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:1530, code: start_loss = loss_fct(start_logits, start_positions)
    add_104: "f32[1, 512]" = torch.ops.aten.add.Tensor(tangents_2, sub_43);  tangents_2 = sub_43 = None
    
    # No stacktrace found for following nodes
    unsqueeze_6: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(add_103, 2);  add_103 = None
    unsqueeze_7: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(add_104, 2);  add_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:1513, code: start_logits, end_logits = logits.split(1, dim=-1)
    cat: "f32[1, 512, 2]" = torch.ops.aten.cat.default([unsqueeze_7, unsqueeze_6], 2);  unsqueeze_7 = unsqueeze_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:1512, code: logits = self.qa_outputs(sequence_output)
    view_266: "f32[512, 2]" = torch.ops.aten.view.default(cat, [512, 2]);  cat = None
    permute_133: "f32[2, 768]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
    mm: "f32[512, 768]" = torch.ops.aten.mm.default(view_266, permute_133);  permute_133 = None
    permute_134: "f32[2, 512]" = torch.ops.aten.permute.default(view_266, [1, 0])
    mm_1: "f32[2, 768]" = torch.ops.aten.mm.default(permute_134, view_264);  permute_134 = view_264 = None
    permute_135: "f32[768, 2]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_21: "f32[1, 2]" = torch.ops.aten.sum.dim_IntList(view_266, [0], True);  view_266 = None
    view_267: "f32[2]" = torch.ops.aten.view.default(sum_21, [2]);  sum_21 = None
    permute_136: "f32[2, 768]" = torch.ops.aten.permute.default(permute_135, [1, 0]);  permute_135 = None
    view_268: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm, [1, 512, 768]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:381, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_44: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_99, getitem_123);  add_99 = getitem_123 = None
    mul_92: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_24);  sub_44 = None
    mul_93: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_268, primals_196);  primals_196 = None
    mul_94: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_93, 768)
    sum_22: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_93, [2], True)
    mul_95: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_93, mul_92);  mul_93 = None
    sum_23: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_95, [2], True);  mul_95 = None
    mul_96: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_92, sum_23);  sum_23 = None
    sub_45: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_94, sum_22);  mul_94 = sum_22 = None
    sub_46: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_45, mul_96);  sub_45 = mul_96 = None
    div_30: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None
    mul_97: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_30, sub_46);  div_30 = sub_46 = None
    mul_98: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_268, mul_92);  mul_92 = None
    sum_24: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_98, [0, 1]);  mul_98 = None
    sum_25: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_268, [0, 1]);  view_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_5: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_121, torch.float32);  getitem_121 = None
    mul_99: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 1.1111111111111112);  convert_element_type_5 = None
    mul_100: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_97, mul_99);  mul_99 = None
    clone_14: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_100, memory_format = torch.contiguous_format);  mul_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379, code: hidden_states = self.dense(hidden_states)
    view_269: "f32[512, 768]" = torch.ops.aten.view.default(clone_14, [512, 768]);  clone_14 = None
    permute_137: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
    mm_2: "f32[512, 3072]" = torch.ops.aten.mm.default(view_269, permute_137);  permute_137 = None
    permute_138: "f32[768, 512]" = torch.ops.aten.permute.default(view_269, [1, 0])
    mm_3: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_138, view_262);  permute_138 = view_262 = None
    permute_139: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_26: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_269, [0], True);  view_269 = None
    view_270: "f32[768]" = torch.ops.aten.view.default(sum_26, [768]);  sum_26 = None
    permute_140: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_139, [1, 0]);  permute_139 = None
    view_271: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_2, [1, 512, 3072]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_101: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476)
    erf_12: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_101);  mul_101 = None
    add_105: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_102: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_105, 0.5);  add_105 = None
    mul_103: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, view_261)
    mul_104: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_103, -0.5);  mul_103 = None
    exp_16: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_104);  mul_104 = None
    mul_105: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
    mul_106: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, mul_105);  view_261 = mul_105 = None
    add_106: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_102, mul_106);  mul_102 = mul_106 = None
    mul_107: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_271, add_106);  view_271 = add_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    view_272: "f32[512, 3072]" = torch.ops.aten.view.default(mul_107, [512, 3072]);  mul_107 = None
    permute_141: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
    mm_4: "f32[512, 768]" = torch.ops.aten.mm.default(view_272, permute_141);  permute_141 = None
    permute_142: "f32[3072, 512]" = torch.ops.aten.permute.default(view_272, [1, 0])
    mm_5: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_142, view_260);  permute_142 = view_260 = None
    permute_143: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_27: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_272, [0], True);  view_272 = None
    view_273: "f32[3072]" = torch.ops.aten.view.default(sum_27, [3072]);  sum_27 = None
    permute_144: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
    view_274: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_4, [1, 512, 768]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    add_107: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_97, view_274);  mul_97 = view_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:300, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_47: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_95, getitem_119);  add_95 = getitem_119 = None
    mul_108: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_23);  sub_47 = None
    mul_109: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_107, primals_190);  primals_190 = None
    mul_110: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_109, 768)
    sum_28: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_109, [2], True)
    mul_111: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_109, mul_108);  mul_109 = None
    sum_29: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_111, [2], True);  mul_111 = None
    mul_112: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_108, sum_29);  sum_29 = None
    sub_48: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_110, sum_28);  mul_110 = sum_28 = None
    sub_49: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_48, mul_112);  sub_48 = mul_112 = None
    div_31: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
    mul_113: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_31, sub_49);  div_31 = sub_49 = None
    mul_114: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_107, mul_108);  mul_108 = None
    sum_30: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_114, [0, 1]);  mul_114 = None
    sum_31: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_107, [0, 1]);  add_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:299, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_6: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_117, torch.float32);  getitem_117 = None
    mul_115: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 1.1111111111111112);  convert_element_type_6 = None
    mul_116: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_113, mul_115);  mul_115 = None
    clone_15: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_116, memory_format = torch.contiguous_format);  mul_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:298, code: hidden_states = self.dense(hidden_states)
    view_275: "f32[512, 768]" = torch.ops.aten.view.default(clone_15, [512, 768]);  clone_15 = None
    permute_145: "f32[768, 768]" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
    mm_6: "f32[512, 768]" = torch.ops.aten.mm.default(view_275, permute_145);  permute_145 = None
    permute_146: "f32[768, 512]" = torch.ops.aten.permute.default(view_275, [1, 0])
    mm_7: "f32[768, 768]" = torch.ops.aten.mm.default(permute_146, view_258);  permute_146 = view_258 = None
    permute_147: "f32[768, 768]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_32: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_275, [0], True);  view_275 = None
    view_276: "f32[768]" = torch.ops.aten.view.default(sum_32, [768]);  sum_32 = None
    permute_148: "f32[768, 768]" = torch.ops.aten.permute.default(permute_147, [1, 0]);  permute_147 = None
    view_277: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_6, [1, 512, 768]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:280, code: context_layer = context_layer.view(new_context_layer_shape)
    view_278: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_277, [1, 512, 12, 64]);  view_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:278, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_149: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_278, [0, 2, 1, 3]);  view_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:276, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_279: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_149, [12, 512, 64]);  permute_149 = None
    permute_150: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None
    bmm_24: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_150, view_279);  permute_150 = None
    permute_151: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_255, [0, 2, 1]);  view_255 = None
    bmm_25: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_279, permute_151);  view_279 = permute_151 = None
    view_280: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_24, [1, 12, 512, 64]);  bmm_24 = None
    view_281: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_25, [1, 12, 512, 512]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:270, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_7: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_115, torch.float32);  getitem_115 = None
    mul_117: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_7, 1.1111111111111112);  convert_element_type_7 = None
    mul_118: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_281, mul_117);  view_281 = mul_117 = None
    clone_16: "f32[1, 12, 512, 512]" = torch.ops.aten.clone.default(mul_118, memory_format = torch.contiguous_format);  mul_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:266, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_16: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    mul_119: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(clone_16, alias_16);  clone_16 = None
    sum_33: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_119, [-1], True)
    mul_120: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_16, sum_33);  alias_16 = sum_33 = None
    sub_50: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_119, mul_120);  mul_119 = mul_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:260, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_32: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_50, 8.0);  sub_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:236, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_282: "f32[12, 512, 512]" = torch.ops.aten.view.default(div_32, [12, 512, 512]);  div_32 = None
    permute_152: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_251, [0, 2, 1]);  view_251 = None
    bmm_26: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_152, view_282);  permute_152 = None
    permute_153: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_252, [0, 2, 1]);  view_252 = None
    bmm_27: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_282, permute_153);  view_282 = permute_153 = None
    view_283: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_26, [1, 12, 64, 512]);  bmm_26 = None
    view_284: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_27, [1, 12, 512, 64]);  bmm_27 = None
    permute_154: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_283, [0, 1, 3, 2]);  view_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_155: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_284, [0, 2, 1, 3]);  view_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_17: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_155, memory_format = torch.contiguous_format);  permute_155 = None
    view_285: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_17, [1, 512, 768]);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_156: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_280, [0, 2, 1, 3]);  view_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_18: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_156, memory_format = torch.contiguous_format);  permute_156 = None
    view_286: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_18, [1, 512, 768]);  clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_287: "f32[512, 768]" = torch.ops.aten.view.default(view_286, [512, 768]);  view_286 = None
    permute_157: "f32[768, 768]" = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
    mm_8: "f32[512, 768]" = torch.ops.aten.mm.default(view_287, permute_157);  permute_157 = None
    permute_158: "f32[768, 512]" = torch.ops.aten.permute.default(view_287, [1, 0])
    mm_9: "f32[768, 768]" = torch.ops.aten.mm.default(permute_158, view_247);  permute_158 = view_247 = None
    permute_159: "f32[768, 768]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_34: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_287, [0], True);  view_287 = None
    view_288: "f32[768]" = torch.ops.aten.view.default(sum_34, [768]);  sum_34 = None
    permute_160: "f32[768, 768]" = torch.ops.aten.permute.default(permute_159, [1, 0]);  permute_159 = None
    view_289: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_8, [1, 512, 768]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_108: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_113, view_289);  mul_113 = view_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_161: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(permute_154, [0, 2, 1, 3]);  permute_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_290: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_161, [1, 512, 768]);  permute_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_291: "f32[512, 768]" = torch.ops.aten.view.default(view_290, [512, 768]);  view_290 = None
    permute_162: "f32[768, 768]" = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
    mm_10: "f32[512, 768]" = torch.ops.aten.mm.default(view_291, permute_162);  permute_162 = None
    permute_163: "f32[768, 512]" = torch.ops.aten.permute.default(view_291, [1, 0])
    mm_11: "f32[768, 768]" = torch.ops.aten.mm.default(permute_163, view_244);  permute_163 = view_244 = None
    permute_164: "f32[768, 768]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_35: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_291, [0], True);  view_291 = None
    view_292: "f32[768]" = torch.ops.aten.view.default(sum_35, [768]);  sum_35 = None
    permute_165: "f32[768, 768]" = torch.ops.aten.permute.default(permute_164, [1, 0]);  permute_164 = None
    view_293: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_10, [1, 512, 768]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_109: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_108, view_293);  add_108 = view_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    view_294: "f32[512, 768]" = torch.ops.aten.view.default(view_285, [512, 768]);  view_285 = None
    permute_166: "f32[768, 768]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
    mm_12: "f32[512, 768]" = torch.ops.aten.mm.default(view_294, permute_166);  permute_166 = None
    permute_167: "f32[768, 512]" = torch.ops.aten.permute.default(view_294, [1, 0])
    mm_13: "f32[768, 768]" = torch.ops.aten.mm.default(permute_167, view_242);  permute_167 = view_242 = None
    permute_168: "f32[768, 768]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_36: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_294, [0], True);  view_294 = None
    view_295: "f32[768]" = torch.ops.aten.view.default(sum_36, [768]);  sum_36 = None
    permute_169: "f32[768, 768]" = torch.ops.aten.permute.default(permute_168, [1, 0]);  permute_168 = None
    view_296: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_12, [1, 512, 768]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    add_110: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_109, view_296);  add_109 = view_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:381, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_51: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_91, getitem_113);  add_91 = getitem_113 = None
    mul_121: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_22);  sub_51 = None
    mul_122: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_110, primals_180);  primals_180 = None
    mul_123: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_122, 768)
    sum_37: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_122, [2], True)
    mul_124: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_122, mul_121);  mul_122 = None
    sum_38: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_124, [2], True);  mul_124 = None
    mul_125: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_121, sum_38);  sum_38 = None
    sub_52: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_123, sum_37);  mul_123 = sum_37 = None
    sub_53: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_52, mul_125);  sub_52 = mul_125 = None
    div_33: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
    mul_126: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_33, sub_53);  div_33 = sub_53 = None
    mul_127: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_110, mul_121);  mul_121 = None
    sum_39: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_127, [0, 1]);  mul_127 = None
    sum_40: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_110, [0, 1]);  add_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_8: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_111, torch.float32);  getitem_111 = None
    mul_128: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.1111111111111112);  convert_element_type_8 = None
    mul_129: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_126, mul_128);  mul_128 = None
    clone_19: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_129, memory_format = torch.contiguous_format);  mul_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379, code: hidden_states = self.dense(hidden_states)
    view_297: "f32[512, 768]" = torch.ops.aten.view.default(clone_19, [512, 768]);  clone_19 = None
    permute_170: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
    mm_14: "f32[512, 3072]" = torch.ops.aten.mm.default(view_297, permute_170);  permute_170 = None
    permute_171: "f32[768, 512]" = torch.ops.aten.permute.default(view_297, [1, 0])
    mm_15: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_171, view_240);  permute_171 = view_240 = None
    permute_172: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_41: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_297, [0], True);  view_297 = None
    view_298: "f32[768]" = torch.ops.aten.view.default(sum_41, [768]);  sum_41 = None
    permute_173: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None
    view_299: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_14, [1, 512, 3072]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_130: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, 0.7071067811865476)
    erf_13: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_130);  mul_130 = None
    add_111: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_131: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_111, 0.5);  add_111 = None
    mul_132: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, view_239)
    mul_133: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_132, -0.5);  mul_132 = None
    exp_17: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_133);  mul_133 = None
    mul_134: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_17, 0.3989422804014327);  exp_17 = None
    mul_135: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, mul_134);  view_239 = mul_134 = None
    add_112: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_131, mul_135);  mul_131 = mul_135 = None
    mul_136: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_299, add_112);  view_299 = add_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    view_300: "f32[512, 3072]" = torch.ops.aten.view.default(mul_136, [512, 3072]);  mul_136 = None
    permute_174: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
    mm_16: "f32[512, 768]" = torch.ops.aten.mm.default(view_300, permute_174);  permute_174 = None
    permute_175: "f32[3072, 512]" = torch.ops.aten.permute.default(view_300, [1, 0])
    mm_17: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_175, view_238);  permute_175 = view_238 = None
    permute_176: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_42: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_300, [0], True);  view_300 = None
    view_301: "f32[3072]" = torch.ops.aten.view.default(sum_42, [3072]);  sum_42 = None
    permute_177: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
    view_302: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_16, [1, 512, 768]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    add_113: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_126, view_302);  mul_126 = view_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:300, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_54: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_87, getitem_109);  add_87 = getitem_109 = None
    mul_137: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_21);  sub_54 = None
    mul_138: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_113, primals_174);  primals_174 = None
    mul_139: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_138, 768)
    sum_43: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_138, [2], True)
    mul_140: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_138, mul_137);  mul_138 = None
    sum_44: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_140, [2], True);  mul_140 = None
    mul_141: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_137, sum_44);  sum_44 = None
    sub_55: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_139, sum_43);  mul_139 = sum_43 = None
    sub_56: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_55, mul_141);  sub_55 = mul_141 = None
    div_34: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
    mul_142: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_34, sub_56);  div_34 = sub_56 = None
    mul_143: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_113, mul_137);  mul_137 = None
    sum_45: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_143, [0, 1]);  mul_143 = None
    sum_46: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_113, [0, 1]);  add_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:299, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_9: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_107, torch.float32);  getitem_107 = None
    mul_144: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_9, 1.1111111111111112);  convert_element_type_9 = None
    mul_145: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_142, mul_144);  mul_144 = None
    clone_20: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_145, memory_format = torch.contiguous_format);  mul_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:298, code: hidden_states = self.dense(hidden_states)
    view_303: "f32[512, 768]" = torch.ops.aten.view.default(clone_20, [512, 768]);  clone_20 = None
    permute_178: "f32[768, 768]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
    mm_18: "f32[512, 768]" = torch.ops.aten.mm.default(view_303, permute_178);  permute_178 = None
    permute_179: "f32[768, 512]" = torch.ops.aten.permute.default(view_303, [1, 0])
    mm_19: "f32[768, 768]" = torch.ops.aten.mm.default(permute_179, view_236);  permute_179 = view_236 = None
    permute_180: "f32[768, 768]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_47: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_303, [0], True);  view_303 = None
    view_304: "f32[768]" = torch.ops.aten.view.default(sum_47, [768]);  sum_47 = None
    permute_181: "f32[768, 768]" = torch.ops.aten.permute.default(permute_180, [1, 0]);  permute_180 = None
    view_305: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_18, [1, 512, 768]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:280, code: context_layer = context_layer.view(new_context_layer_shape)
    view_306: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_305, [1, 512, 12, 64]);  view_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:278, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_182: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_306, [0, 2, 1, 3]);  view_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:276, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_307: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_182, [12, 512, 64]);  permute_182 = None
    permute_183: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_232, [0, 2, 1]);  view_232 = None
    bmm_28: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_183, view_307);  permute_183 = None
    permute_184: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_233, [0, 2, 1]);  view_233 = None
    bmm_29: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_307, permute_184);  view_307 = permute_184 = None
    view_308: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_28, [1, 12, 512, 64]);  bmm_28 = None
    view_309: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_29, [1, 12, 512, 512]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:270, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_10: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_105, torch.float32);  getitem_105 = None
    mul_146: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.1111111111111112);  convert_element_type_10 = None
    mul_147: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_309, mul_146);  view_309 = mul_146 = None
    clone_21: "f32[1, 12, 512, 512]" = torch.ops.aten.clone.default(mul_147, memory_format = torch.contiguous_format);  mul_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:266, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_17: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    mul_148: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(clone_21, alias_17);  clone_21 = None
    sum_48: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_148, [-1], True)
    mul_149: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_17, sum_48);  alias_17 = sum_48 = None
    sub_57: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_148, mul_149);  mul_148 = mul_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:260, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_35: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_57, 8.0);  sub_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:236, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_310: "f32[12, 512, 512]" = torch.ops.aten.view.default(div_35, [12, 512, 512]);  div_35 = None
    permute_185: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_229, [0, 2, 1]);  view_229 = None
    bmm_30: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_185, view_310);  permute_185 = None
    permute_186: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_230, [0, 2, 1]);  view_230 = None
    bmm_31: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_310, permute_186);  view_310 = permute_186 = None
    view_311: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_30, [1, 12, 64, 512]);  bmm_30 = None
    view_312: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_31, [1, 12, 512, 64]);  bmm_31 = None
    permute_187: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_311, [0, 1, 3, 2]);  view_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_188: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_312, [0, 2, 1, 3]);  view_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_22: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_188, memory_format = torch.contiguous_format);  permute_188 = None
    view_313: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_22, [1, 512, 768]);  clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_189: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_308, [0, 2, 1, 3]);  view_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_23: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None
    view_314: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_23, [1, 512, 768]);  clone_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_315: "f32[512, 768]" = torch.ops.aten.view.default(view_314, [512, 768]);  view_314 = None
    permute_190: "f32[768, 768]" = torch.ops.aten.permute.default(permute_113, [1, 0]);  permute_113 = None
    mm_20: "f32[512, 768]" = torch.ops.aten.mm.default(view_315, permute_190);  permute_190 = None
    permute_191: "f32[768, 512]" = torch.ops.aten.permute.default(view_315, [1, 0])
    mm_21: "f32[768, 768]" = torch.ops.aten.mm.default(permute_191, view_225);  permute_191 = view_225 = None
    permute_192: "f32[768, 768]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_49: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_315, [0], True);  view_315 = None
    view_316: "f32[768]" = torch.ops.aten.view.default(sum_49, [768]);  sum_49 = None
    permute_193: "f32[768, 768]" = torch.ops.aten.permute.default(permute_192, [1, 0]);  permute_192 = None
    view_317: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_20, [1, 512, 768]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_114: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_142, view_317);  mul_142 = view_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_194: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(permute_187, [0, 2, 1, 3]);  permute_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_318: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_194, [1, 512, 768]);  permute_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_319: "f32[512, 768]" = torch.ops.aten.view.default(view_318, [512, 768]);  view_318 = None
    permute_195: "f32[768, 768]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
    mm_22: "f32[512, 768]" = torch.ops.aten.mm.default(view_319, permute_195);  permute_195 = None
    permute_196: "f32[768, 512]" = torch.ops.aten.permute.default(view_319, [1, 0])
    mm_23: "f32[768, 768]" = torch.ops.aten.mm.default(permute_196, view_222);  permute_196 = view_222 = None
    permute_197: "f32[768, 768]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_50: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_319, [0], True);  view_319 = None
    view_320: "f32[768]" = torch.ops.aten.view.default(sum_50, [768]);  sum_50 = None
    permute_198: "f32[768, 768]" = torch.ops.aten.permute.default(permute_197, [1, 0]);  permute_197 = None
    view_321: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_22, [1, 512, 768]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_115: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_114, view_321);  add_114 = view_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    view_322: "f32[512, 768]" = torch.ops.aten.view.default(view_313, [512, 768]);  view_313 = None
    permute_199: "f32[768, 768]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
    mm_24: "f32[512, 768]" = torch.ops.aten.mm.default(view_322, permute_199);  permute_199 = None
    permute_200: "f32[768, 512]" = torch.ops.aten.permute.default(view_322, [1, 0])
    mm_25: "f32[768, 768]" = torch.ops.aten.mm.default(permute_200, view_220);  permute_200 = view_220 = None
    permute_201: "f32[768, 768]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_51: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_322, [0], True);  view_322 = None
    view_323: "f32[768]" = torch.ops.aten.view.default(sum_51, [768]);  sum_51 = None
    permute_202: "f32[768, 768]" = torch.ops.aten.permute.default(permute_201, [1, 0]);  permute_201 = None
    view_324: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_24, [1, 512, 768]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    add_116: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_115, view_324);  add_115 = view_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:381, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_58: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_83, getitem_103);  add_83 = getitem_103 = None
    mul_150: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_20);  sub_58 = None
    mul_151: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_116, primals_164);  primals_164 = None
    mul_152: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_151, 768)
    sum_52: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_151, [2], True)
    mul_153: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_151, mul_150);  mul_151 = None
    sum_53: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_153, [2], True);  mul_153 = None
    mul_154: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_150, sum_53);  sum_53 = None
    sub_59: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_152, sum_52);  mul_152 = sum_52 = None
    sub_60: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_59, mul_154);  sub_59 = mul_154 = None
    div_36: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
    mul_155: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_36, sub_60);  div_36 = sub_60 = None
    mul_156: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_116, mul_150);  mul_150 = None
    sum_54: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_156, [0, 1]);  mul_156 = None
    sum_55: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_116, [0, 1]);  add_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_11: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_101, torch.float32);  getitem_101 = None
    mul_157: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 1.1111111111111112);  convert_element_type_11 = None
    mul_158: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_155, mul_157);  mul_157 = None
    clone_24: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_158, memory_format = torch.contiguous_format);  mul_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379, code: hidden_states = self.dense(hidden_states)
    view_325: "f32[512, 768]" = torch.ops.aten.view.default(clone_24, [512, 768]);  clone_24 = None
    permute_203: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
    mm_26: "f32[512, 3072]" = torch.ops.aten.mm.default(view_325, permute_203);  permute_203 = None
    permute_204: "f32[768, 512]" = torch.ops.aten.permute.default(view_325, [1, 0])
    mm_27: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_204, view_218);  permute_204 = view_218 = None
    permute_205: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_56: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_325, [0], True);  view_325 = None
    view_326: "f32[768]" = torch.ops.aten.view.default(sum_56, [768]);  sum_56 = None
    permute_206: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_205, [1, 0]);  permute_205 = None
    view_327: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_26, [1, 512, 3072]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_159: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476)
    erf_14: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_159);  mul_159 = None
    add_117: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_160: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_117, 0.5);  add_117 = None
    mul_161: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, view_217)
    mul_162: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_161, -0.5);  mul_161 = None
    exp_18: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_162);  mul_162 = None
    mul_163: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
    mul_164: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, mul_163);  view_217 = mul_163 = None
    add_118: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_160, mul_164);  mul_160 = mul_164 = None
    mul_165: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_327, add_118);  view_327 = add_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    view_328: "f32[512, 3072]" = torch.ops.aten.view.default(mul_165, [512, 3072]);  mul_165 = None
    permute_207: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
    mm_28: "f32[512, 768]" = torch.ops.aten.mm.default(view_328, permute_207);  permute_207 = None
    permute_208: "f32[3072, 512]" = torch.ops.aten.permute.default(view_328, [1, 0])
    mm_29: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_208, view_216);  permute_208 = view_216 = None
    permute_209: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_57: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_328, [0], True);  view_328 = None
    view_329: "f32[3072]" = torch.ops.aten.view.default(sum_57, [3072]);  sum_57 = None
    permute_210: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_209, [1, 0]);  permute_209 = None
    view_330: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_28, [1, 512, 768]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    add_119: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_155, view_330);  mul_155 = view_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:300, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_61: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_79, getitem_99);  add_79 = getitem_99 = None
    mul_166: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_19);  sub_61 = None
    mul_167: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_119, primals_158);  primals_158 = None
    mul_168: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_167, 768)
    sum_58: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_167, [2], True)
    mul_169: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_167, mul_166);  mul_167 = None
    sum_59: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_169, [2], True);  mul_169 = None
    mul_170: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_166, sum_59);  sum_59 = None
    sub_62: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_168, sum_58);  mul_168 = sum_58 = None
    sub_63: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_62, mul_170);  sub_62 = mul_170 = None
    div_37: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
    mul_171: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_37, sub_63);  div_37 = sub_63 = None
    mul_172: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_119, mul_166);  mul_166 = None
    sum_60: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_172, [0, 1]);  mul_172 = None
    sum_61: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_119, [0, 1]);  add_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:299, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_12: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_97, torch.float32);  getitem_97 = None
    mul_173: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_12, 1.1111111111111112);  convert_element_type_12 = None
    mul_174: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_171, mul_173);  mul_173 = None
    clone_25: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_174, memory_format = torch.contiguous_format);  mul_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:298, code: hidden_states = self.dense(hidden_states)
    view_331: "f32[512, 768]" = torch.ops.aten.view.default(clone_25, [512, 768]);  clone_25 = None
    permute_211: "f32[768, 768]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
    mm_30: "f32[512, 768]" = torch.ops.aten.mm.default(view_331, permute_211);  permute_211 = None
    permute_212: "f32[768, 512]" = torch.ops.aten.permute.default(view_331, [1, 0])
    mm_31: "f32[768, 768]" = torch.ops.aten.mm.default(permute_212, view_214);  permute_212 = view_214 = None
    permute_213: "f32[768, 768]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_62: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_331, [0], True);  view_331 = None
    view_332: "f32[768]" = torch.ops.aten.view.default(sum_62, [768]);  sum_62 = None
    permute_214: "f32[768, 768]" = torch.ops.aten.permute.default(permute_213, [1, 0]);  permute_213 = None
    view_333: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_30, [1, 512, 768]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:280, code: context_layer = context_layer.view(new_context_layer_shape)
    view_334: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_333, [1, 512, 12, 64]);  view_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:278, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_215: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_334, [0, 2, 1, 3]);  view_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:276, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_335: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_215, [12, 512, 64]);  permute_215 = None
    permute_216: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None
    bmm_32: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_216, view_335);  permute_216 = None
    permute_217: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_211, [0, 2, 1]);  view_211 = None
    bmm_33: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_335, permute_217);  view_335 = permute_217 = None
    view_336: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_32, [1, 12, 512, 64]);  bmm_32 = None
    view_337: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_33, [1, 12, 512, 512]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:270, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_13: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_95, torch.float32);  getitem_95 = None
    mul_175: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
    mul_176: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_337, mul_175);  view_337 = mul_175 = None
    clone_26: "f32[1, 12, 512, 512]" = torch.ops.aten.clone.default(mul_176, memory_format = torch.contiguous_format);  mul_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:266, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_18: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    mul_177: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(clone_26, alias_18);  clone_26 = None
    sum_63: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_177, [-1], True)
    mul_178: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_18, sum_63);  alias_18 = sum_63 = None
    sub_64: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_177, mul_178);  mul_177 = mul_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:260, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_38: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_64, 8.0);  sub_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:236, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_338: "f32[12, 512, 512]" = torch.ops.aten.view.default(div_38, [12, 512, 512]);  div_38 = None
    permute_218: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_207, [0, 2, 1]);  view_207 = None
    bmm_34: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_218, view_338);  permute_218 = None
    permute_219: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_208, [0, 2, 1]);  view_208 = None
    bmm_35: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_338, permute_219);  view_338 = permute_219 = None
    view_339: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_34, [1, 12, 64, 512]);  bmm_34 = None
    view_340: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_35, [1, 12, 512, 64]);  bmm_35 = None
    permute_220: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_339, [0, 1, 3, 2]);  view_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_221: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_340, [0, 2, 1, 3]);  view_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_27: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_221, memory_format = torch.contiguous_format);  permute_221 = None
    view_341: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_27, [1, 512, 768]);  clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_222: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_336, [0, 2, 1, 3]);  view_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_28: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_222, memory_format = torch.contiguous_format);  permute_222 = None
    view_342: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_28, [1, 512, 768]);  clone_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_343: "f32[512, 768]" = torch.ops.aten.view.default(view_342, [512, 768]);  view_342 = None
    permute_223: "f32[768, 768]" = torch.ops.aten.permute.default(permute_102, [1, 0]);  permute_102 = None
    mm_32: "f32[512, 768]" = torch.ops.aten.mm.default(view_343, permute_223);  permute_223 = None
    permute_224: "f32[768, 512]" = torch.ops.aten.permute.default(view_343, [1, 0])
    mm_33: "f32[768, 768]" = torch.ops.aten.mm.default(permute_224, view_203);  permute_224 = view_203 = None
    permute_225: "f32[768, 768]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_64: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_343, [0], True);  view_343 = None
    view_344: "f32[768]" = torch.ops.aten.view.default(sum_64, [768]);  sum_64 = None
    permute_226: "f32[768, 768]" = torch.ops.aten.permute.default(permute_225, [1, 0]);  permute_225 = None
    view_345: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_32, [1, 512, 768]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_120: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_171, view_345);  mul_171 = view_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_227: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(permute_220, [0, 2, 1, 3]);  permute_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_346: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_227, [1, 512, 768]);  permute_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_347: "f32[512, 768]" = torch.ops.aten.view.default(view_346, [512, 768]);  view_346 = None
    permute_228: "f32[768, 768]" = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
    mm_34: "f32[512, 768]" = torch.ops.aten.mm.default(view_347, permute_228);  permute_228 = None
    permute_229: "f32[768, 512]" = torch.ops.aten.permute.default(view_347, [1, 0])
    mm_35: "f32[768, 768]" = torch.ops.aten.mm.default(permute_229, view_200);  permute_229 = view_200 = None
    permute_230: "f32[768, 768]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_65: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_347, [0], True);  view_347 = None
    view_348: "f32[768]" = torch.ops.aten.view.default(sum_65, [768]);  sum_65 = None
    permute_231: "f32[768, 768]" = torch.ops.aten.permute.default(permute_230, [1, 0]);  permute_230 = None
    view_349: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_34, [1, 512, 768]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_121: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_120, view_349);  add_120 = view_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    view_350: "f32[512, 768]" = torch.ops.aten.view.default(view_341, [512, 768]);  view_341 = None
    permute_232: "f32[768, 768]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
    mm_36: "f32[512, 768]" = torch.ops.aten.mm.default(view_350, permute_232);  permute_232 = None
    permute_233: "f32[768, 512]" = torch.ops.aten.permute.default(view_350, [1, 0])
    mm_37: "f32[768, 768]" = torch.ops.aten.mm.default(permute_233, view_198);  permute_233 = view_198 = None
    permute_234: "f32[768, 768]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_66: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_350, [0], True);  view_350 = None
    view_351: "f32[768]" = torch.ops.aten.view.default(sum_66, [768]);  sum_66 = None
    permute_235: "f32[768, 768]" = torch.ops.aten.permute.default(permute_234, [1, 0]);  permute_234 = None
    view_352: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_36, [1, 512, 768]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    add_122: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_121, view_352);  add_121 = view_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:381, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_65: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_75, getitem_93);  add_75 = getitem_93 = None
    mul_179: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_18);  sub_65 = None
    mul_180: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_122, primals_148);  primals_148 = None
    mul_181: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_180, 768)
    sum_67: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_180, [2], True)
    mul_182: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_180, mul_179);  mul_180 = None
    sum_68: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_182, [2], True);  mul_182 = None
    mul_183: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_179, sum_68);  sum_68 = None
    sub_66: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_181, sum_67);  mul_181 = sum_67 = None
    sub_67: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_66, mul_183);  sub_66 = mul_183 = None
    div_39: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
    mul_184: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_39, sub_67);  div_39 = sub_67 = None
    mul_185: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_122, mul_179);  mul_179 = None
    sum_69: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_185, [0, 1]);  mul_185 = None
    sum_70: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_122, [0, 1]);  add_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_14: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_91, torch.float32);  getitem_91 = None
    mul_186: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 1.1111111111111112);  convert_element_type_14 = None
    mul_187: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_184, mul_186);  mul_186 = None
    clone_29: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_187, memory_format = torch.contiguous_format);  mul_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379, code: hidden_states = self.dense(hidden_states)
    view_353: "f32[512, 768]" = torch.ops.aten.view.default(clone_29, [512, 768]);  clone_29 = None
    permute_236: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
    mm_38: "f32[512, 3072]" = torch.ops.aten.mm.default(view_353, permute_236);  permute_236 = None
    permute_237: "f32[768, 512]" = torch.ops.aten.permute.default(view_353, [1, 0])
    mm_39: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_237, view_196);  permute_237 = view_196 = None
    permute_238: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_71: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_353, [0], True);  view_353 = None
    view_354: "f32[768]" = torch.ops.aten.view.default(sum_71, [768]);  sum_71 = None
    permute_239: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_238, [1, 0]);  permute_238 = None
    view_355: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_38, [1, 512, 3072]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_188: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476)
    erf_15: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_188);  mul_188 = None
    add_123: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_189: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_123, 0.5);  add_123 = None
    mul_190: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, view_195)
    mul_191: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_190, -0.5);  mul_190 = None
    exp_19: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_191);  mul_191 = None
    mul_192: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
    mul_193: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, mul_192);  view_195 = mul_192 = None
    add_124: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_189, mul_193);  mul_189 = mul_193 = None
    mul_194: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_355, add_124);  view_355 = add_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    view_356: "f32[512, 3072]" = torch.ops.aten.view.default(mul_194, [512, 3072]);  mul_194 = None
    permute_240: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
    mm_40: "f32[512, 768]" = torch.ops.aten.mm.default(view_356, permute_240);  permute_240 = None
    permute_241: "f32[3072, 512]" = torch.ops.aten.permute.default(view_356, [1, 0])
    mm_41: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_241, view_194);  permute_241 = view_194 = None
    permute_242: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_72: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_356, [0], True);  view_356 = None
    view_357: "f32[3072]" = torch.ops.aten.view.default(sum_72, [3072]);  sum_72 = None
    permute_243: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None
    view_358: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_40, [1, 512, 768]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    add_125: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_184, view_358);  mul_184 = view_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:300, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_68: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_71, getitem_89);  add_71 = getitem_89 = None
    mul_195: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_17);  sub_68 = None
    mul_196: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_125, primals_142);  primals_142 = None
    mul_197: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_196, 768)
    sum_73: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_196, [2], True)
    mul_198: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_196, mul_195);  mul_196 = None
    sum_74: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_198, [2], True);  mul_198 = None
    mul_199: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_195, sum_74);  sum_74 = None
    sub_69: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_197, sum_73);  mul_197 = sum_73 = None
    sub_70: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_69, mul_199);  sub_69 = mul_199 = None
    div_40: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
    mul_200: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_40, sub_70);  div_40 = sub_70 = None
    mul_201: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_125, mul_195);  mul_195 = None
    sum_75: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_201, [0, 1]);  mul_201 = None
    sum_76: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_125, [0, 1]);  add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:299, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_15: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_87, torch.float32);  getitem_87 = None
    mul_202: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_15, 1.1111111111111112);  convert_element_type_15 = None
    mul_203: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_200, mul_202);  mul_202 = None
    clone_30: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_203, memory_format = torch.contiguous_format);  mul_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:298, code: hidden_states = self.dense(hidden_states)
    view_359: "f32[512, 768]" = torch.ops.aten.view.default(clone_30, [512, 768]);  clone_30 = None
    permute_244: "f32[768, 768]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
    mm_42: "f32[512, 768]" = torch.ops.aten.mm.default(view_359, permute_244);  permute_244 = None
    permute_245: "f32[768, 512]" = torch.ops.aten.permute.default(view_359, [1, 0])
    mm_43: "f32[768, 768]" = torch.ops.aten.mm.default(permute_245, view_192);  permute_245 = view_192 = None
    permute_246: "f32[768, 768]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_77: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_359, [0], True);  view_359 = None
    view_360: "f32[768]" = torch.ops.aten.view.default(sum_77, [768]);  sum_77 = None
    permute_247: "f32[768, 768]" = torch.ops.aten.permute.default(permute_246, [1, 0]);  permute_246 = None
    view_361: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_42, [1, 512, 768]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:280, code: context_layer = context_layer.view(new_context_layer_shape)
    view_362: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_361, [1, 512, 12, 64]);  view_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:278, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_248: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_362, [0, 2, 1, 3]);  view_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:276, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_363: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_248, [12, 512, 64]);  permute_248 = None
    permute_249: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None
    bmm_36: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_249, view_363);  permute_249 = None
    permute_250: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_189, [0, 2, 1]);  view_189 = None
    bmm_37: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_363, permute_250);  view_363 = permute_250 = None
    view_364: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_36, [1, 12, 512, 64]);  bmm_36 = None
    view_365: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_37, [1, 12, 512, 512]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:270, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_16: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_85, torch.float32);  getitem_85 = None
    mul_204: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_16, 1.1111111111111112);  convert_element_type_16 = None
    mul_205: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_365, mul_204);  view_365 = mul_204 = None
    clone_31: "f32[1, 12, 512, 512]" = torch.ops.aten.clone.default(mul_205, memory_format = torch.contiguous_format);  mul_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:266, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_19: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    mul_206: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(clone_31, alias_19);  clone_31 = None
    sum_78: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_206, [-1], True)
    mul_207: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_19, sum_78);  alias_19 = sum_78 = None
    sub_71: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_206, mul_207);  mul_206 = mul_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:260, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_41: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_71, 8.0);  sub_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:236, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_366: "f32[12, 512, 512]" = torch.ops.aten.view.default(div_41, [12, 512, 512]);  div_41 = None
    permute_251: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_185, [0, 2, 1]);  view_185 = None
    bmm_38: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_251, view_366);  permute_251 = None
    permute_252: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1]);  view_186 = None
    bmm_39: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_366, permute_252);  view_366 = permute_252 = None
    view_367: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_38, [1, 12, 64, 512]);  bmm_38 = None
    view_368: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_39, [1, 12, 512, 64]);  bmm_39 = None
    permute_253: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_367, [0, 1, 3, 2]);  view_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_254: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_368, [0, 2, 1, 3]);  view_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_32: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_254, memory_format = torch.contiguous_format);  permute_254 = None
    view_369: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_32, [1, 512, 768]);  clone_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_255: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_364, [0, 2, 1, 3]);  view_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_33: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_255, memory_format = torch.contiguous_format);  permute_255 = None
    view_370: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_33, [1, 512, 768]);  clone_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_371: "f32[512, 768]" = torch.ops.aten.view.default(view_370, [512, 768]);  view_370 = None
    permute_256: "f32[768, 768]" = torch.ops.aten.permute.default(permute_91, [1, 0]);  permute_91 = None
    mm_44: "f32[512, 768]" = torch.ops.aten.mm.default(view_371, permute_256);  permute_256 = None
    permute_257: "f32[768, 512]" = torch.ops.aten.permute.default(view_371, [1, 0])
    mm_45: "f32[768, 768]" = torch.ops.aten.mm.default(permute_257, view_181);  permute_257 = view_181 = None
    permute_258: "f32[768, 768]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_79: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_371, [0], True);  view_371 = None
    view_372: "f32[768]" = torch.ops.aten.view.default(sum_79, [768]);  sum_79 = None
    permute_259: "f32[768, 768]" = torch.ops.aten.permute.default(permute_258, [1, 0]);  permute_258 = None
    view_373: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_44, [1, 512, 768]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_126: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_200, view_373);  mul_200 = view_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_260: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(permute_253, [0, 2, 1, 3]);  permute_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_374: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_260, [1, 512, 768]);  permute_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_375: "f32[512, 768]" = torch.ops.aten.view.default(view_374, [512, 768]);  view_374 = None
    permute_261: "f32[768, 768]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
    mm_46: "f32[512, 768]" = torch.ops.aten.mm.default(view_375, permute_261);  permute_261 = None
    permute_262: "f32[768, 512]" = torch.ops.aten.permute.default(view_375, [1, 0])
    mm_47: "f32[768, 768]" = torch.ops.aten.mm.default(permute_262, view_178);  permute_262 = view_178 = None
    permute_263: "f32[768, 768]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_80: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_375, [0], True);  view_375 = None
    view_376: "f32[768]" = torch.ops.aten.view.default(sum_80, [768]);  sum_80 = None
    permute_264: "f32[768, 768]" = torch.ops.aten.permute.default(permute_263, [1, 0]);  permute_263 = None
    view_377: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_46, [1, 512, 768]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_127: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_126, view_377);  add_126 = view_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    view_378: "f32[512, 768]" = torch.ops.aten.view.default(view_369, [512, 768]);  view_369 = None
    permute_265: "f32[768, 768]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
    mm_48: "f32[512, 768]" = torch.ops.aten.mm.default(view_378, permute_265);  permute_265 = None
    permute_266: "f32[768, 512]" = torch.ops.aten.permute.default(view_378, [1, 0])
    mm_49: "f32[768, 768]" = torch.ops.aten.mm.default(permute_266, view_176);  permute_266 = view_176 = None
    permute_267: "f32[768, 768]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_81: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_378, [0], True);  view_378 = None
    view_379: "f32[768]" = torch.ops.aten.view.default(sum_81, [768]);  sum_81 = None
    permute_268: "f32[768, 768]" = torch.ops.aten.permute.default(permute_267, [1, 0]);  permute_267 = None
    view_380: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_48, [1, 512, 768]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    add_128: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_127, view_380);  add_127 = view_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:381, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_72: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_67, getitem_83);  add_67 = getitem_83 = None
    mul_208: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_16);  sub_72 = None
    mul_209: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_128, primals_132);  primals_132 = None
    mul_210: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_209, 768)
    sum_82: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_209, [2], True)
    mul_211: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_209, mul_208);  mul_209 = None
    sum_83: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_211, [2], True);  mul_211 = None
    mul_212: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_208, sum_83);  sum_83 = None
    sub_73: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_210, sum_82);  mul_210 = sum_82 = None
    sub_74: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_73, mul_212);  sub_73 = mul_212 = None
    div_42: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
    mul_213: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_42, sub_74);  div_42 = sub_74 = None
    mul_214: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_128, mul_208);  mul_208 = None
    sum_84: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_214, [0, 1]);  mul_214 = None
    sum_85: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_128, [0, 1]);  add_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_17: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_81, torch.float32);  getitem_81 = None
    mul_215: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_17, 1.1111111111111112);  convert_element_type_17 = None
    mul_216: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_213, mul_215);  mul_215 = None
    clone_34: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_216, memory_format = torch.contiguous_format);  mul_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379, code: hidden_states = self.dense(hidden_states)
    view_381: "f32[512, 768]" = torch.ops.aten.view.default(clone_34, [512, 768]);  clone_34 = None
    permute_269: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
    mm_50: "f32[512, 3072]" = torch.ops.aten.mm.default(view_381, permute_269);  permute_269 = None
    permute_270: "f32[768, 512]" = torch.ops.aten.permute.default(view_381, [1, 0])
    mm_51: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_270, view_174);  permute_270 = view_174 = None
    permute_271: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_86: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_381, [0], True);  view_381 = None
    view_382: "f32[768]" = torch.ops.aten.view.default(sum_86, [768]);  sum_86 = None
    permute_272: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_271, [1, 0]);  permute_271 = None
    view_383: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_50, [1, 512, 3072]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_217: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476)
    erf_16: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_217);  mul_217 = None
    add_129: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_218: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_129, 0.5);  add_129 = None
    mul_219: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, view_173)
    mul_220: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_219, -0.5);  mul_219 = None
    exp_20: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_220);  mul_220 = None
    mul_221: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
    mul_222: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, mul_221);  view_173 = mul_221 = None
    add_130: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_218, mul_222);  mul_218 = mul_222 = None
    mul_223: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_383, add_130);  view_383 = add_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    view_384: "f32[512, 3072]" = torch.ops.aten.view.default(mul_223, [512, 3072]);  mul_223 = None
    permute_273: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
    mm_52: "f32[512, 768]" = torch.ops.aten.mm.default(view_384, permute_273);  permute_273 = None
    permute_274: "f32[3072, 512]" = torch.ops.aten.permute.default(view_384, [1, 0])
    mm_53: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_274, view_172);  permute_274 = view_172 = None
    permute_275: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_87: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_384, [0], True);  view_384 = None
    view_385: "f32[3072]" = torch.ops.aten.view.default(sum_87, [3072]);  sum_87 = None
    permute_276: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_275, [1, 0]);  permute_275 = None
    view_386: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_52, [1, 512, 768]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    add_131: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_213, view_386);  mul_213 = view_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:300, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_75: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_63, getitem_79);  add_63 = getitem_79 = None
    mul_224: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_15);  sub_75 = None
    mul_225: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_131, primals_126);  primals_126 = None
    mul_226: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_225, 768)
    sum_88: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_225, [2], True)
    mul_227: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_225, mul_224);  mul_225 = None
    sum_89: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_227, [2], True);  mul_227 = None
    mul_228: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_224, sum_89);  sum_89 = None
    sub_76: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_226, sum_88);  mul_226 = sum_88 = None
    sub_77: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_76, mul_228);  sub_76 = mul_228 = None
    div_43: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
    mul_229: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_43, sub_77);  div_43 = sub_77 = None
    mul_230: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_131, mul_224);  mul_224 = None
    sum_90: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_230, [0, 1]);  mul_230 = None
    sum_91: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_131, [0, 1]);  add_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:299, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_18: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_77, torch.float32);  getitem_77 = None
    mul_231: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_18, 1.1111111111111112);  convert_element_type_18 = None
    mul_232: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_229, mul_231);  mul_231 = None
    clone_35: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_232, memory_format = torch.contiguous_format);  mul_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:298, code: hidden_states = self.dense(hidden_states)
    view_387: "f32[512, 768]" = torch.ops.aten.view.default(clone_35, [512, 768]);  clone_35 = None
    permute_277: "f32[768, 768]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
    mm_54: "f32[512, 768]" = torch.ops.aten.mm.default(view_387, permute_277);  permute_277 = None
    permute_278: "f32[768, 512]" = torch.ops.aten.permute.default(view_387, [1, 0])
    mm_55: "f32[768, 768]" = torch.ops.aten.mm.default(permute_278, view_170);  permute_278 = view_170 = None
    permute_279: "f32[768, 768]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_92: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_387, [0], True);  view_387 = None
    view_388: "f32[768]" = torch.ops.aten.view.default(sum_92, [768]);  sum_92 = None
    permute_280: "f32[768, 768]" = torch.ops.aten.permute.default(permute_279, [1, 0]);  permute_279 = None
    view_389: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_54, [1, 512, 768]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:280, code: context_layer = context_layer.view(new_context_layer_shape)
    view_390: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_389, [1, 512, 12, 64]);  view_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:278, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_281: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_390, [0, 2, 1, 3]);  view_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:276, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_391: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_281, [12, 512, 64]);  permute_281 = None
    permute_282: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_166, [0, 2, 1]);  view_166 = None
    bmm_40: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_282, view_391);  permute_282 = None
    permute_283: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_167, [0, 2, 1]);  view_167 = None
    bmm_41: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_391, permute_283);  view_391 = permute_283 = None
    view_392: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_40, [1, 12, 512, 64]);  bmm_40 = None
    view_393: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_41, [1, 12, 512, 512]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:270, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_19: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_75, torch.float32);  getitem_75 = None
    mul_233: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_19, 1.1111111111111112);  convert_element_type_19 = None
    mul_234: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_393, mul_233);  view_393 = mul_233 = None
    clone_36: "f32[1, 12, 512, 512]" = torch.ops.aten.clone.default(mul_234, memory_format = torch.contiguous_format);  mul_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:266, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_20: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    mul_235: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(clone_36, alias_20);  clone_36 = None
    sum_93: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_235, [-1], True)
    mul_236: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_20, sum_93);  alias_20 = sum_93 = None
    sub_78: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_235, mul_236);  mul_235 = mul_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:260, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_44: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_78, 8.0);  sub_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:236, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_394: "f32[12, 512, 512]" = torch.ops.aten.view.default(div_44, [12, 512, 512]);  div_44 = None
    permute_284: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_163, [0, 2, 1]);  view_163 = None
    bmm_42: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_284, view_394);  permute_284 = None
    permute_285: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_164, [0, 2, 1]);  view_164 = None
    bmm_43: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_394, permute_285);  view_394 = permute_285 = None
    view_395: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_42, [1, 12, 64, 512]);  bmm_42 = None
    view_396: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_43, [1, 12, 512, 64]);  bmm_43 = None
    permute_286: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_395, [0, 1, 3, 2]);  view_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_287: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_396, [0, 2, 1, 3]);  view_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_37: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_287, memory_format = torch.contiguous_format);  permute_287 = None
    view_397: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_37, [1, 512, 768]);  clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_288: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_392, [0, 2, 1, 3]);  view_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_38: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_288, memory_format = torch.contiguous_format);  permute_288 = None
    view_398: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_38, [1, 512, 768]);  clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_399: "f32[512, 768]" = torch.ops.aten.view.default(view_398, [512, 768]);  view_398 = None
    permute_289: "f32[768, 768]" = torch.ops.aten.permute.default(permute_80, [1, 0]);  permute_80 = None
    mm_56: "f32[512, 768]" = torch.ops.aten.mm.default(view_399, permute_289);  permute_289 = None
    permute_290: "f32[768, 512]" = torch.ops.aten.permute.default(view_399, [1, 0])
    mm_57: "f32[768, 768]" = torch.ops.aten.mm.default(permute_290, view_159);  permute_290 = view_159 = None
    permute_291: "f32[768, 768]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_94: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_399, [0], True);  view_399 = None
    view_400: "f32[768]" = torch.ops.aten.view.default(sum_94, [768]);  sum_94 = None
    permute_292: "f32[768, 768]" = torch.ops.aten.permute.default(permute_291, [1, 0]);  permute_291 = None
    view_401: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_56, [1, 512, 768]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_132: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_229, view_401);  mul_229 = view_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_293: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(permute_286, [0, 2, 1, 3]);  permute_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_402: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_293, [1, 512, 768]);  permute_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_403: "f32[512, 768]" = torch.ops.aten.view.default(view_402, [512, 768]);  view_402 = None
    permute_294: "f32[768, 768]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
    mm_58: "f32[512, 768]" = torch.ops.aten.mm.default(view_403, permute_294);  permute_294 = None
    permute_295: "f32[768, 512]" = torch.ops.aten.permute.default(view_403, [1, 0])
    mm_59: "f32[768, 768]" = torch.ops.aten.mm.default(permute_295, view_156);  permute_295 = view_156 = None
    permute_296: "f32[768, 768]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_95: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_403, [0], True);  view_403 = None
    view_404: "f32[768]" = torch.ops.aten.view.default(sum_95, [768]);  sum_95 = None
    permute_297: "f32[768, 768]" = torch.ops.aten.permute.default(permute_296, [1, 0]);  permute_296 = None
    view_405: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_58, [1, 512, 768]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_133: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_132, view_405);  add_132 = view_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    view_406: "f32[512, 768]" = torch.ops.aten.view.default(view_397, [512, 768]);  view_397 = None
    permute_298: "f32[768, 768]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
    mm_60: "f32[512, 768]" = torch.ops.aten.mm.default(view_406, permute_298);  permute_298 = None
    permute_299: "f32[768, 512]" = torch.ops.aten.permute.default(view_406, [1, 0])
    mm_61: "f32[768, 768]" = torch.ops.aten.mm.default(permute_299, view_154);  permute_299 = view_154 = None
    permute_300: "f32[768, 768]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_96: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_406, [0], True);  view_406 = None
    view_407: "f32[768]" = torch.ops.aten.view.default(sum_96, [768]);  sum_96 = None
    permute_301: "f32[768, 768]" = torch.ops.aten.permute.default(permute_300, [1, 0]);  permute_300 = None
    view_408: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_60, [1, 512, 768]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    add_134: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_133, view_408);  add_133 = view_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:381, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_79: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_59, getitem_73);  add_59 = getitem_73 = None
    mul_237: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_14);  sub_79 = None
    mul_238: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_134, primals_116);  primals_116 = None
    mul_239: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_238, 768)
    sum_97: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_238, [2], True)
    mul_240: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_238, mul_237);  mul_238 = None
    sum_98: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_240, [2], True);  mul_240 = None
    mul_241: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_237, sum_98);  sum_98 = None
    sub_80: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_239, sum_97);  mul_239 = sum_97 = None
    sub_81: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_80, mul_241);  sub_80 = mul_241 = None
    div_45: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
    mul_242: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_45, sub_81);  div_45 = sub_81 = None
    mul_243: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_134, mul_237);  mul_237 = None
    sum_99: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_243, [0, 1]);  mul_243 = None
    sum_100: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_134, [0, 1]);  add_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_20: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_71, torch.float32);  getitem_71 = None
    mul_244: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.1111111111111112);  convert_element_type_20 = None
    mul_245: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_242, mul_244);  mul_244 = None
    clone_39: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_245, memory_format = torch.contiguous_format);  mul_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379, code: hidden_states = self.dense(hidden_states)
    view_409: "f32[512, 768]" = torch.ops.aten.view.default(clone_39, [512, 768]);  clone_39 = None
    permute_302: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
    mm_62: "f32[512, 3072]" = torch.ops.aten.mm.default(view_409, permute_302);  permute_302 = None
    permute_303: "f32[768, 512]" = torch.ops.aten.permute.default(view_409, [1, 0])
    mm_63: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_303, view_152);  permute_303 = view_152 = None
    permute_304: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_101: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_409, [0], True);  view_409 = None
    view_410: "f32[768]" = torch.ops.aten.view.default(sum_101, [768]);  sum_101 = None
    permute_305: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_304, [1, 0]);  permute_304 = None
    view_411: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_62, [1, 512, 3072]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_246: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476)
    erf_17: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_246);  mul_246 = None
    add_135: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_247: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_135, 0.5);  add_135 = None
    mul_248: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, view_151)
    mul_249: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_248, -0.5);  mul_248 = None
    exp_21: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_249);  mul_249 = None
    mul_250: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
    mul_251: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, mul_250);  view_151 = mul_250 = None
    add_136: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_247, mul_251);  mul_247 = mul_251 = None
    mul_252: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_411, add_136);  view_411 = add_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    view_412: "f32[512, 3072]" = torch.ops.aten.view.default(mul_252, [512, 3072]);  mul_252 = None
    permute_306: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
    mm_64: "f32[512, 768]" = torch.ops.aten.mm.default(view_412, permute_306);  permute_306 = None
    permute_307: "f32[3072, 512]" = torch.ops.aten.permute.default(view_412, [1, 0])
    mm_65: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_307, view_150);  permute_307 = view_150 = None
    permute_308: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_102: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_412, [0], True);  view_412 = None
    view_413: "f32[3072]" = torch.ops.aten.view.default(sum_102, [3072]);  sum_102 = None
    permute_309: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_308, [1, 0]);  permute_308 = None
    view_414: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_64, [1, 512, 768]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    add_137: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_242, view_414);  mul_242 = view_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:300, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_82: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_55, getitem_69);  add_55 = getitem_69 = None
    mul_253: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_13);  sub_82 = None
    mul_254: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_137, primals_110);  primals_110 = None
    mul_255: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_254, 768)
    sum_103: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_254, [2], True)
    mul_256: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_254, mul_253);  mul_254 = None
    sum_104: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_256, [2], True);  mul_256 = None
    mul_257: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_253, sum_104);  sum_104 = None
    sub_83: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_255, sum_103);  mul_255 = sum_103 = None
    sub_84: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_83, mul_257);  sub_83 = mul_257 = None
    div_46: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
    mul_258: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_46, sub_84);  div_46 = sub_84 = None
    mul_259: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_137, mul_253);  mul_253 = None
    sum_105: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_259, [0, 1]);  mul_259 = None
    sum_106: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_137, [0, 1]);  add_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:299, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_21: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_67, torch.float32);  getitem_67 = None
    mul_260: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 1.1111111111111112);  convert_element_type_21 = None
    mul_261: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_258, mul_260);  mul_260 = None
    clone_40: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_261, memory_format = torch.contiguous_format);  mul_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:298, code: hidden_states = self.dense(hidden_states)
    view_415: "f32[512, 768]" = torch.ops.aten.view.default(clone_40, [512, 768]);  clone_40 = None
    permute_310: "f32[768, 768]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
    mm_66: "f32[512, 768]" = torch.ops.aten.mm.default(view_415, permute_310);  permute_310 = None
    permute_311: "f32[768, 512]" = torch.ops.aten.permute.default(view_415, [1, 0])
    mm_67: "f32[768, 768]" = torch.ops.aten.mm.default(permute_311, view_148);  permute_311 = view_148 = None
    permute_312: "f32[768, 768]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_107: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_415, [0], True);  view_415 = None
    view_416: "f32[768]" = torch.ops.aten.view.default(sum_107, [768]);  sum_107 = None
    permute_313: "f32[768, 768]" = torch.ops.aten.permute.default(permute_312, [1, 0]);  permute_312 = None
    view_417: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_66, [1, 512, 768]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:280, code: context_layer = context_layer.view(new_context_layer_shape)
    view_418: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_417, [1, 512, 12, 64]);  view_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:278, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_314: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_418, [0, 2, 1, 3]);  view_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:276, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_419: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_314, [12, 512, 64]);  permute_314 = None
    permute_315: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_144, [0, 2, 1]);  view_144 = None
    bmm_44: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_315, view_419);  permute_315 = None
    permute_316: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_145, [0, 2, 1]);  view_145 = None
    bmm_45: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_419, permute_316);  view_419 = permute_316 = None
    view_420: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_44, [1, 12, 512, 64]);  bmm_44 = None
    view_421: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_45, [1, 12, 512, 512]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:270, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_22: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_65, torch.float32);  getitem_65 = None
    mul_262: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 1.1111111111111112);  convert_element_type_22 = None
    mul_263: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_421, mul_262);  view_421 = mul_262 = None
    clone_41: "f32[1, 12, 512, 512]" = torch.ops.aten.clone.default(mul_263, memory_format = torch.contiguous_format);  mul_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:266, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_21: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    mul_264: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(clone_41, alias_21);  clone_41 = None
    sum_108: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_264, [-1], True)
    mul_265: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_21, sum_108);  alias_21 = sum_108 = None
    sub_85: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_264, mul_265);  mul_264 = mul_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:260, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_47: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_85, 8.0);  sub_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:236, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_422: "f32[12, 512, 512]" = torch.ops.aten.view.default(div_47, [12, 512, 512]);  div_47 = None
    permute_317: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_141, [0, 2, 1]);  view_141 = None
    bmm_46: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_317, view_422);  permute_317 = None
    permute_318: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_142, [0, 2, 1]);  view_142 = None
    bmm_47: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_422, permute_318);  view_422 = permute_318 = None
    view_423: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_46, [1, 12, 64, 512]);  bmm_46 = None
    view_424: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_47, [1, 12, 512, 64]);  bmm_47 = None
    permute_319: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_423, [0, 1, 3, 2]);  view_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_320: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_424, [0, 2, 1, 3]);  view_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_42: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_320, memory_format = torch.contiguous_format);  permute_320 = None
    view_425: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_42, [1, 512, 768]);  clone_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_321: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_420, [0, 2, 1, 3]);  view_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_43: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_321, memory_format = torch.contiguous_format);  permute_321 = None
    view_426: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_43, [1, 512, 768]);  clone_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_427: "f32[512, 768]" = torch.ops.aten.view.default(view_426, [512, 768]);  view_426 = None
    permute_322: "f32[768, 768]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
    mm_68: "f32[512, 768]" = torch.ops.aten.mm.default(view_427, permute_322);  permute_322 = None
    permute_323: "f32[768, 512]" = torch.ops.aten.permute.default(view_427, [1, 0])
    mm_69: "f32[768, 768]" = torch.ops.aten.mm.default(permute_323, view_137);  permute_323 = view_137 = None
    permute_324: "f32[768, 768]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_109: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_427, [0], True);  view_427 = None
    view_428: "f32[768]" = torch.ops.aten.view.default(sum_109, [768]);  sum_109 = None
    permute_325: "f32[768, 768]" = torch.ops.aten.permute.default(permute_324, [1, 0]);  permute_324 = None
    view_429: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_68, [1, 512, 768]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_138: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_258, view_429);  mul_258 = view_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_326: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(permute_319, [0, 2, 1, 3]);  permute_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_430: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_326, [1, 512, 768]);  permute_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_431: "f32[512, 768]" = torch.ops.aten.view.default(view_430, [512, 768]);  view_430 = None
    permute_327: "f32[768, 768]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
    mm_70: "f32[512, 768]" = torch.ops.aten.mm.default(view_431, permute_327);  permute_327 = None
    permute_328: "f32[768, 512]" = torch.ops.aten.permute.default(view_431, [1, 0])
    mm_71: "f32[768, 768]" = torch.ops.aten.mm.default(permute_328, view_134);  permute_328 = view_134 = None
    permute_329: "f32[768, 768]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_110: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_431, [0], True);  view_431 = None
    view_432: "f32[768]" = torch.ops.aten.view.default(sum_110, [768]);  sum_110 = None
    permute_330: "f32[768, 768]" = torch.ops.aten.permute.default(permute_329, [1, 0]);  permute_329 = None
    view_433: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_70, [1, 512, 768]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_139: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_138, view_433);  add_138 = view_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    view_434: "f32[512, 768]" = torch.ops.aten.view.default(view_425, [512, 768]);  view_425 = None
    permute_331: "f32[768, 768]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    mm_72: "f32[512, 768]" = torch.ops.aten.mm.default(view_434, permute_331);  permute_331 = None
    permute_332: "f32[768, 512]" = torch.ops.aten.permute.default(view_434, [1, 0])
    mm_73: "f32[768, 768]" = torch.ops.aten.mm.default(permute_332, view_132);  permute_332 = view_132 = None
    permute_333: "f32[768, 768]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_111: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_434, [0], True);  view_434 = None
    view_435: "f32[768]" = torch.ops.aten.view.default(sum_111, [768]);  sum_111 = None
    permute_334: "f32[768, 768]" = torch.ops.aten.permute.default(permute_333, [1, 0]);  permute_333 = None
    view_436: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_72, [1, 512, 768]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    add_140: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_139, view_436);  add_139 = view_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:381, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_86: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_51, getitem_63);  add_51 = getitem_63 = None
    mul_266: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_12);  sub_86 = None
    mul_267: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_140, primals_100);  primals_100 = None
    mul_268: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_267, 768)
    sum_112: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_267, [2], True)
    mul_269: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_267, mul_266);  mul_267 = None
    sum_113: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_269, [2], True);  mul_269 = None
    mul_270: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_266, sum_113);  sum_113 = None
    sub_87: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_268, sum_112);  mul_268 = sum_112 = None
    sub_88: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_87, mul_270);  sub_87 = mul_270 = None
    div_48: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
    mul_271: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_48, sub_88);  div_48 = sub_88 = None
    mul_272: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_140, mul_266);  mul_266 = None
    sum_114: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_272, [0, 1]);  mul_272 = None
    sum_115: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_140, [0, 1]);  add_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_23: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_61, torch.float32);  getitem_61 = None
    mul_273: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_23, 1.1111111111111112);  convert_element_type_23 = None
    mul_274: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_271, mul_273);  mul_273 = None
    clone_44: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_274, memory_format = torch.contiguous_format);  mul_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379, code: hidden_states = self.dense(hidden_states)
    view_437: "f32[512, 768]" = torch.ops.aten.view.default(clone_44, [512, 768]);  clone_44 = None
    permute_335: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    mm_74: "f32[512, 3072]" = torch.ops.aten.mm.default(view_437, permute_335);  permute_335 = None
    permute_336: "f32[768, 512]" = torch.ops.aten.permute.default(view_437, [1, 0])
    mm_75: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_336, view_130);  permute_336 = view_130 = None
    permute_337: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_116: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_437, [0], True);  view_437 = None
    view_438: "f32[768]" = torch.ops.aten.view.default(sum_116, [768]);  sum_116 = None
    permute_338: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_337, [1, 0]);  permute_337 = None
    view_439: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_74, [1, 512, 3072]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_275: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476)
    erf_18: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_275);  mul_275 = None
    add_141: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_276: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_141, 0.5);  add_141 = None
    mul_277: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, view_129)
    mul_278: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_277, -0.5);  mul_277 = None
    exp_22: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_278);  mul_278 = None
    mul_279: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
    mul_280: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, mul_279);  view_129 = mul_279 = None
    add_142: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_276, mul_280);  mul_276 = mul_280 = None
    mul_281: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_439, add_142);  view_439 = add_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    view_440: "f32[512, 3072]" = torch.ops.aten.view.default(mul_281, [512, 3072]);  mul_281 = None
    permute_339: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
    mm_76: "f32[512, 768]" = torch.ops.aten.mm.default(view_440, permute_339);  permute_339 = None
    permute_340: "f32[3072, 512]" = torch.ops.aten.permute.default(view_440, [1, 0])
    mm_77: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_340, view_128);  permute_340 = view_128 = None
    permute_341: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_117: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_440, [0], True);  view_440 = None
    view_441: "f32[3072]" = torch.ops.aten.view.default(sum_117, [3072]);  sum_117 = None
    permute_342: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_341, [1, 0]);  permute_341 = None
    view_442: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_76, [1, 512, 768]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    add_143: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_271, view_442);  mul_271 = view_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:300, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_89: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_47, getitem_59);  add_47 = getitem_59 = None
    mul_282: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_89, rsqrt_11);  sub_89 = None
    mul_283: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_143, primals_94);  primals_94 = None
    mul_284: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_283, 768)
    sum_118: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_283, [2], True)
    mul_285: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_283, mul_282);  mul_283 = None
    sum_119: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_285, [2], True);  mul_285 = None
    mul_286: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_282, sum_119);  sum_119 = None
    sub_90: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_284, sum_118);  mul_284 = sum_118 = None
    sub_91: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_90, mul_286);  sub_90 = mul_286 = None
    div_49: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
    mul_287: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_49, sub_91);  div_49 = sub_91 = None
    mul_288: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_143, mul_282);  mul_282 = None
    sum_120: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_288, [0, 1]);  mul_288 = None
    sum_121: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_143, [0, 1]);  add_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:299, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_24: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_57, torch.float32);  getitem_57 = None
    mul_289: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 1.1111111111111112);  convert_element_type_24 = None
    mul_290: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_287, mul_289);  mul_289 = None
    clone_45: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_290, memory_format = torch.contiguous_format);  mul_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:298, code: hidden_states = self.dense(hidden_states)
    view_443: "f32[512, 768]" = torch.ops.aten.view.default(clone_45, [512, 768]);  clone_45 = None
    permute_343: "f32[768, 768]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
    mm_78: "f32[512, 768]" = torch.ops.aten.mm.default(view_443, permute_343);  permute_343 = None
    permute_344: "f32[768, 512]" = torch.ops.aten.permute.default(view_443, [1, 0])
    mm_79: "f32[768, 768]" = torch.ops.aten.mm.default(permute_344, view_126);  permute_344 = view_126 = None
    permute_345: "f32[768, 768]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_122: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_443, [0], True);  view_443 = None
    view_444: "f32[768]" = torch.ops.aten.view.default(sum_122, [768]);  sum_122 = None
    permute_346: "f32[768, 768]" = torch.ops.aten.permute.default(permute_345, [1, 0]);  permute_345 = None
    view_445: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_78, [1, 512, 768]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:280, code: context_layer = context_layer.view(new_context_layer_shape)
    view_446: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_445, [1, 512, 12, 64]);  view_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:278, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_347: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_446, [0, 2, 1, 3]);  view_446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:276, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_447: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_347, [12, 512, 64]);  permute_347 = None
    permute_348: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None
    bmm_48: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_348, view_447);  permute_348 = None
    permute_349: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None
    bmm_49: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_447, permute_349);  view_447 = permute_349 = None
    view_448: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_48, [1, 12, 512, 64]);  bmm_48 = None
    view_449: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_49, [1, 12, 512, 512]);  bmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:270, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_25: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_55, torch.float32);  getitem_55 = None
    mul_291: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.1111111111111112);  convert_element_type_25 = None
    mul_292: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_449, mul_291);  view_449 = mul_291 = None
    clone_46: "f32[1, 12, 512, 512]" = torch.ops.aten.clone.default(mul_292, memory_format = torch.contiguous_format);  mul_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:266, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_22: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    mul_293: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(clone_46, alias_22);  clone_46 = None
    sum_123: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_293, [-1], True)
    mul_294: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_22, sum_123);  alias_22 = sum_123 = None
    sub_92: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_293, mul_294);  mul_293 = mul_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:260, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_50: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_92, 8.0);  sub_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:236, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_450: "f32[12, 512, 512]" = torch.ops.aten.view.default(div_50, [12, 512, 512]);  div_50 = None
    permute_350: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_119, [0, 2, 1]);  view_119 = None
    bmm_50: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_350, view_450);  permute_350 = None
    permute_351: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1]);  view_120 = None
    bmm_51: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_450, permute_351);  view_450 = permute_351 = None
    view_451: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_50, [1, 12, 64, 512]);  bmm_50 = None
    view_452: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_51, [1, 12, 512, 64]);  bmm_51 = None
    permute_352: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_451, [0, 1, 3, 2]);  view_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_353: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_452, [0, 2, 1, 3]);  view_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_47: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_353, memory_format = torch.contiguous_format);  permute_353 = None
    view_453: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_47, [1, 512, 768]);  clone_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_354: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_448, [0, 2, 1, 3]);  view_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_48: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_354, memory_format = torch.contiguous_format);  permute_354 = None
    view_454: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_48, [1, 512, 768]);  clone_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_455: "f32[512, 768]" = torch.ops.aten.view.default(view_454, [512, 768]);  view_454 = None
    permute_355: "f32[768, 768]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
    mm_80: "f32[512, 768]" = torch.ops.aten.mm.default(view_455, permute_355);  permute_355 = None
    permute_356: "f32[768, 512]" = torch.ops.aten.permute.default(view_455, [1, 0])
    mm_81: "f32[768, 768]" = torch.ops.aten.mm.default(permute_356, view_115);  permute_356 = view_115 = None
    permute_357: "f32[768, 768]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_124: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_455, [0], True);  view_455 = None
    view_456: "f32[768]" = torch.ops.aten.view.default(sum_124, [768]);  sum_124 = None
    permute_358: "f32[768, 768]" = torch.ops.aten.permute.default(permute_357, [1, 0]);  permute_357 = None
    view_457: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_80, [1, 512, 768]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_144: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_287, view_457);  mul_287 = view_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_359: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(permute_352, [0, 2, 1, 3]);  permute_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_458: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_359, [1, 512, 768]);  permute_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_459: "f32[512, 768]" = torch.ops.aten.view.default(view_458, [512, 768]);  view_458 = None
    permute_360: "f32[768, 768]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    mm_82: "f32[512, 768]" = torch.ops.aten.mm.default(view_459, permute_360);  permute_360 = None
    permute_361: "f32[768, 512]" = torch.ops.aten.permute.default(view_459, [1, 0])
    mm_83: "f32[768, 768]" = torch.ops.aten.mm.default(permute_361, view_112);  permute_361 = view_112 = None
    permute_362: "f32[768, 768]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_125: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_459, [0], True);  view_459 = None
    view_460: "f32[768]" = torch.ops.aten.view.default(sum_125, [768]);  sum_125 = None
    permute_363: "f32[768, 768]" = torch.ops.aten.permute.default(permute_362, [1, 0]);  permute_362 = None
    view_461: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_82, [1, 512, 768]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_145: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_144, view_461);  add_144 = view_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    view_462: "f32[512, 768]" = torch.ops.aten.view.default(view_453, [512, 768]);  view_453 = None
    permute_364: "f32[768, 768]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    mm_84: "f32[512, 768]" = torch.ops.aten.mm.default(view_462, permute_364);  permute_364 = None
    permute_365: "f32[768, 512]" = torch.ops.aten.permute.default(view_462, [1, 0])
    mm_85: "f32[768, 768]" = torch.ops.aten.mm.default(permute_365, view_110);  permute_365 = view_110 = None
    permute_366: "f32[768, 768]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_126: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_462, [0], True);  view_462 = None
    view_463: "f32[768]" = torch.ops.aten.view.default(sum_126, [768]);  sum_126 = None
    permute_367: "f32[768, 768]" = torch.ops.aten.permute.default(permute_366, [1, 0]);  permute_366 = None
    view_464: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_84, [1, 512, 768]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    add_146: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_145, view_464);  add_145 = view_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:381, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_93: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_43, getitem_53);  add_43 = getitem_53 = None
    mul_295: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_93, rsqrt_10);  sub_93 = None
    mul_296: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_146, primals_84);  primals_84 = None
    mul_297: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_296, 768)
    sum_127: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_296, [2], True)
    mul_298: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_296, mul_295);  mul_296 = None
    sum_128: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_298, [2], True);  mul_298 = None
    mul_299: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_295, sum_128);  sum_128 = None
    sub_94: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_297, sum_127);  mul_297 = sum_127 = None
    sub_95: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_94, mul_299);  sub_94 = mul_299 = None
    div_51: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
    mul_300: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_51, sub_95);  div_51 = sub_95 = None
    mul_301: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_146, mul_295);  mul_295 = None
    sum_129: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_301, [0, 1]);  mul_301 = None
    sum_130: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_146, [0, 1]);  add_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_26: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_51, torch.float32);  getitem_51 = None
    mul_302: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_26, 1.1111111111111112);  convert_element_type_26 = None
    mul_303: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_300, mul_302);  mul_302 = None
    clone_49: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_303, memory_format = torch.contiguous_format);  mul_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379, code: hidden_states = self.dense(hidden_states)
    view_465: "f32[512, 768]" = torch.ops.aten.view.default(clone_49, [512, 768]);  clone_49 = None
    permute_368: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    mm_86: "f32[512, 3072]" = torch.ops.aten.mm.default(view_465, permute_368);  permute_368 = None
    permute_369: "f32[768, 512]" = torch.ops.aten.permute.default(view_465, [1, 0])
    mm_87: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_369, view_108);  permute_369 = view_108 = None
    permute_370: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
    sum_131: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_465, [0], True);  view_465 = None
    view_466: "f32[768]" = torch.ops.aten.view.default(sum_131, [768]);  sum_131 = None
    permute_371: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_370, [1, 0]);  permute_370 = None
    view_467: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_86, [1, 512, 3072]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_304: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476)
    erf_19: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_304);  mul_304 = None
    add_147: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_305: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_147, 0.5);  add_147 = None
    mul_306: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, view_107)
    mul_307: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_306, -0.5);  mul_306 = None
    exp_23: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_307);  mul_307 = None
    mul_308: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
    mul_309: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, mul_308);  view_107 = mul_308 = None
    add_148: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_305, mul_309);  mul_305 = mul_309 = None
    mul_310: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_467, add_148);  view_467 = add_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    view_468: "f32[512, 3072]" = torch.ops.aten.view.default(mul_310, [512, 3072]);  mul_310 = None
    permute_372: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
    mm_88: "f32[512, 768]" = torch.ops.aten.mm.default(view_468, permute_372);  permute_372 = None
    permute_373: "f32[3072, 512]" = torch.ops.aten.permute.default(view_468, [1, 0])
    mm_89: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_373, view_106);  permute_373 = view_106 = None
    permute_374: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    sum_132: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_468, [0], True);  view_468 = None
    view_469: "f32[3072]" = torch.ops.aten.view.default(sum_132, [3072]);  sum_132 = None
    permute_375: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_374, [1, 0]);  permute_374 = None
    view_470: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_88, [1, 512, 768]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    add_149: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_300, view_470);  mul_300 = view_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:300, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_96: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_39, getitem_49);  add_39 = getitem_49 = None
    mul_311: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_96, rsqrt_9);  sub_96 = None
    mul_312: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_149, primals_78);  primals_78 = None
    mul_313: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_312, 768)
    sum_133: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_312, [2], True)
    mul_314: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_312, mul_311);  mul_312 = None
    sum_134: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_314, [2], True);  mul_314 = None
    mul_315: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_311, sum_134);  sum_134 = None
    sub_97: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_313, sum_133);  mul_313 = sum_133 = None
    sub_98: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_97, mul_315);  sub_97 = mul_315 = None
    div_52: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
    mul_316: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_52, sub_98);  div_52 = sub_98 = None
    mul_317: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_149, mul_311);  mul_311 = None
    sum_135: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_317, [0, 1]);  mul_317 = None
    sum_136: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_149, [0, 1]);  add_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:299, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_27: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_47, torch.float32);  getitem_47 = None
    mul_318: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_27, 1.1111111111111112);  convert_element_type_27 = None
    mul_319: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_316, mul_318);  mul_318 = None
    clone_50: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_319, memory_format = torch.contiguous_format);  mul_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:298, code: hidden_states = self.dense(hidden_states)
    view_471: "f32[512, 768]" = torch.ops.aten.view.default(clone_50, [512, 768]);  clone_50 = None
    permute_376: "f32[768, 768]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
    mm_90: "f32[512, 768]" = torch.ops.aten.mm.default(view_471, permute_376);  permute_376 = None
    permute_377: "f32[768, 512]" = torch.ops.aten.permute.default(view_471, [1, 0])
    mm_91: "f32[768, 768]" = torch.ops.aten.mm.default(permute_377, view_104);  permute_377 = view_104 = None
    permute_378: "f32[768, 768]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_137: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_471, [0], True);  view_471 = None
    view_472: "f32[768]" = torch.ops.aten.view.default(sum_137, [768]);  sum_137 = None
    permute_379: "f32[768, 768]" = torch.ops.aten.permute.default(permute_378, [1, 0]);  permute_378 = None
    view_473: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_90, [1, 512, 768]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:280, code: context_layer = context_layer.view(new_context_layer_shape)
    view_474: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_473, [1, 512, 12, 64]);  view_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:278, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_380: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_474, [0, 2, 1, 3]);  view_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:276, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_475: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_380, [12, 512, 64]);  permute_380 = None
    permute_381: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_100, [0, 2, 1]);  view_100 = None
    bmm_52: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_381, view_475);  permute_381 = None
    permute_382: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_101, [0, 2, 1]);  view_101 = None
    bmm_53: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_475, permute_382);  view_475 = permute_382 = None
    view_476: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_52, [1, 12, 512, 64]);  bmm_52 = None
    view_477: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_53, [1, 12, 512, 512]);  bmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:270, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_28: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_45, torch.float32);  getitem_45 = None
    mul_320: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_28, 1.1111111111111112);  convert_element_type_28 = None
    mul_321: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_477, mul_320);  view_477 = mul_320 = None
    clone_51: "f32[1, 12, 512, 512]" = torch.ops.aten.clone.default(mul_321, memory_format = torch.contiguous_format);  mul_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:266, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_23: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    mul_322: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(clone_51, alias_23);  clone_51 = None
    sum_138: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_322, [-1], True)
    mul_323: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_23, sum_138);  alias_23 = sum_138 = None
    sub_99: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_322, mul_323);  mul_322 = mul_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:260, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_53: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_99, 8.0);  sub_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:236, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_478: "f32[12, 512, 512]" = torch.ops.aten.view.default(div_53, [12, 512, 512]);  div_53 = None
    permute_383: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_97, [0, 2, 1]);  view_97 = None
    bmm_54: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_383, view_478);  permute_383 = None
    permute_384: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_98, [0, 2, 1]);  view_98 = None
    bmm_55: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_478, permute_384);  view_478 = permute_384 = None
    view_479: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_54, [1, 12, 64, 512]);  bmm_54 = None
    view_480: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_55, [1, 12, 512, 64]);  bmm_55 = None
    permute_385: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_479, [0, 1, 3, 2]);  view_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_386: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_480, [0, 2, 1, 3]);  view_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_52: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_386, memory_format = torch.contiguous_format);  permute_386 = None
    view_481: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_52, [1, 512, 768]);  clone_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_387: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_476, [0, 2, 1, 3]);  view_476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_53: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_387, memory_format = torch.contiguous_format);  permute_387 = None
    view_482: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_53, [1, 512, 768]);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_483: "f32[512, 768]" = torch.ops.aten.view.default(view_482, [512, 768]);  view_482 = None
    permute_388: "f32[768, 768]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
    mm_92: "f32[512, 768]" = torch.ops.aten.mm.default(view_483, permute_388);  permute_388 = None
    permute_389: "f32[768, 512]" = torch.ops.aten.permute.default(view_483, [1, 0])
    mm_93: "f32[768, 768]" = torch.ops.aten.mm.default(permute_389, view_93);  permute_389 = view_93 = None
    permute_390: "f32[768, 768]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_139: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_483, [0], True);  view_483 = None
    view_484: "f32[768]" = torch.ops.aten.view.default(sum_139, [768]);  sum_139 = None
    permute_391: "f32[768, 768]" = torch.ops.aten.permute.default(permute_390, [1, 0]);  permute_390 = None
    view_485: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_92, [1, 512, 768]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_150: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_316, view_485);  mul_316 = view_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_392: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(permute_385, [0, 2, 1, 3]);  permute_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_486: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_392, [1, 512, 768]);  permute_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_487: "f32[512, 768]" = torch.ops.aten.view.default(view_486, [512, 768]);  view_486 = None
    permute_393: "f32[768, 768]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
    mm_94: "f32[512, 768]" = torch.ops.aten.mm.default(view_487, permute_393);  permute_393 = None
    permute_394: "f32[768, 512]" = torch.ops.aten.permute.default(view_487, [1, 0])
    mm_95: "f32[768, 768]" = torch.ops.aten.mm.default(permute_394, view_90);  permute_394 = view_90 = None
    permute_395: "f32[768, 768]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_140: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_487, [0], True);  view_487 = None
    view_488: "f32[768]" = torch.ops.aten.view.default(sum_140, [768]);  sum_140 = None
    permute_396: "f32[768, 768]" = torch.ops.aten.permute.default(permute_395, [1, 0]);  permute_395 = None
    view_489: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_94, [1, 512, 768]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_151: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_150, view_489);  add_150 = view_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    view_490: "f32[512, 768]" = torch.ops.aten.view.default(view_481, [512, 768]);  view_481 = None
    permute_397: "f32[768, 768]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
    mm_96: "f32[512, 768]" = torch.ops.aten.mm.default(view_490, permute_397);  permute_397 = None
    permute_398: "f32[768, 512]" = torch.ops.aten.permute.default(view_490, [1, 0])
    mm_97: "f32[768, 768]" = torch.ops.aten.mm.default(permute_398, view_88);  permute_398 = view_88 = None
    permute_399: "f32[768, 768]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_141: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_490, [0], True);  view_490 = None
    view_491: "f32[768]" = torch.ops.aten.view.default(sum_141, [768]);  sum_141 = None
    permute_400: "f32[768, 768]" = torch.ops.aten.permute.default(permute_399, [1, 0]);  permute_399 = None
    view_492: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_96, [1, 512, 768]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    add_152: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_151, view_492);  add_151 = view_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:381, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_100: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_43);  add_35 = getitem_43 = None
    mul_324: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_100, rsqrt_8);  sub_100 = None
    mul_325: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_152, primals_68);  primals_68 = None
    mul_326: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_325, 768)
    sum_142: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_325, [2], True)
    mul_327: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_325, mul_324);  mul_325 = None
    sum_143: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_327, [2], True);  mul_327 = None
    mul_328: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_324, sum_143);  sum_143 = None
    sub_101: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_326, sum_142);  mul_326 = sum_142 = None
    sub_102: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_101, mul_328);  sub_101 = mul_328 = None
    div_54: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
    mul_329: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_54, sub_102);  div_54 = sub_102 = None
    mul_330: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_152, mul_324);  mul_324 = None
    sum_144: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_330, [0, 1]);  mul_330 = None
    sum_145: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_152, [0, 1]);  add_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_29: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_41, torch.float32);  getitem_41 = None
    mul_331: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_29, 1.1111111111111112);  convert_element_type_29 = None
    mul_332: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_329, mul_331);  mul_331 = None
    clone_54: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_332, memory_format = torch.contiguous_format);  mul_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379, code: hidden_states = self.dense(hidden_states)
    view_493: "f32[512, 768]" = torch.ops.aten.view.default(clone_54, [512, 768]);  clone_54 = None
    permute_401: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
    mm_98: "f32[512, 3072]" = torch.ops.aten.mm.default(view_493, permute_401);  permute_401 = None
    permute_402: "f32[768, 512]" = torch.ops.aten.permute.default(view_493, [1, 0])
    mm_99: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_402, view_86);  permute_402 = view_86 = None
    permute_403: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    sum_146: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_493, [0], True);  view_493 = None
    view_494: "f32[768]" = torch.ops.aten.view.default(sum_146, [768]);  sum_146 = None
    permute_404: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_403, [1, 0]);  permute_403 = None
    view_495: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_98, [1, 512, 3072]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_333: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476)
    erf_20: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_333);  mul_333 = None
    add_153: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_334: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_153, 0.5);  add_153 = None
    mul_335: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, view_85)
    mul_336: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_335, -0.5);  mul_335 = None
    exp_24: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_336);  mul_336 = None
    mul_337: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
    mul_338: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, mul_337);  view_85 = mul_337 = None
    add_154: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_334, mul_338);  mul_334 = mul_338 = None
    mul_339: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_495, add_154);  view_495 = add_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    view_496: "f32[512, 3072]" = torch.ops.aten.view.default(mul_339, [512, 3072]);  mul_339 = None
    permute_405: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    mm_100: "f32[512, 768]" = torch.ops.aten.mm.default(view_496, permute_405);  permute_405 = None
    permute_406: "f32[3072, 512]" = torch.ops.aten.permute.default(view_496, [1, 0])
    mm_101: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_406, view_84);  permute_406 = view_84 = None
    permute_407: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_101, [1, 0]);  mm_101 = None
    sum_147: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_496, [0], True);  view_496 = None
    view_497: "f32[3072]" = torch.ops.aten.view.default(sum_147, [3072]);  sum_147 = None
    permute_408: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_407, [1, 0]);  permute_407 = None
    view_498: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_100, [1, 512, 768]);  mm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    add_155: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_329, view_498);  mul_329 = view_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:300, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_103: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_31, getitem_39);  add_31 = getitem_39 = None
    mul_340: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_103, rsqrt_7);  sub_103 = None
    mul_341: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_155, primals_62);  primals_62 = None
    mul_342: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_341, 768)
    sum_148: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_341, [2], True)
    mul_343: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_341, mul_340);  mul_341 = None
    sum_149: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_343, [2], True);  mul_343 = None
    mul_344: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_340, sum_149);  sum_149 = None
    sub_104: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_342, sum_148);  mul_342 = sum_148 = None
    sub_105: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_104, mul_344);  sub_104 = mul_344 = None
    div_55: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
    mul_345: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_55, sub_105);  div_55 = sub_105 = None
    mul_346: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_155, mul_340);  mul_340 = None
    sum_150: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_346, [0, 1]);  mul_346 = None
    sum_151: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_155, [0, 1]);  add_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:299, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_30: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_37, torch.float32);  getitem_37 = None
    mul_347: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_30, 1.1111111111111112);  convert_element_type_30 = None
    mul_348: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_345, mul_347);  mul_347 = None
    clone_55: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_348, memory_format = torch.contiguous_format);  mul_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:298, code: hidden_states = self.dense(hidden_states)
    view_499: "f32[512, 768]" = torch.ops.aten.view.default(clone_55, [512, 768]);  clone_55 = None
    permute_409: "f32[768, 768]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
    mm_102: "f32[512, 768]" = torch.ops.aten.mm.default(view_499, permute_409);  permute_409 = None
    permute_410: "f32[768, 512]" = torch.ops.aten.permute.default(view_499, [1, 0])
    mm_103: "f32[768, 768]" = torch.ops.aten.mm.default(permute_410, view_82);  permute_410 = view_82 = None
    permute_411: "f32[768, 768]" = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
    sum_152: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_499, [0], True);  view_499 = None
    view_500: "f32[768]" = torch.ops.aten.view.default(sum_152, [768]);  sum_152 = None
    permute_412: "f32[768, 768]" = torch.ops.aten.permute.default(permute_411, [1, 0]);  permute_411 = None
    view_501: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_102, [1, 512, 768]);  mm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:280, code: context_layer = context_layer.view(new_context_layer_shape)
    view_502: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_501, [1, 512, 12, 64]);  view_501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:278, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_413: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_502, [0, 2, 1, 3]);  view_502 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:276, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_503: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_413, [12, 512, 64]);  permute_413 = None
    permute_414: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
    bmm_56: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_414, view_503);  permute_414 = None
    permute_415: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_79, [0, 2, 1]);  view_79 = None
    bmm_57: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_503, permute_415);  view_503 = permute_415 = None
    view_504: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_56, [1, 12, 512, 64]);  bmm_56 = None
    view_505: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_57, [1, 12, 512, 512]);  bmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:270, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_31: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_35, torch.float32);  getitem_35 = None
    mul_349: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_31, 1.1111111111111112);  convert_element_type_31 = None
    mul_350: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_505, mul_349);  view_505 = mul_349 = None
    clone_56: "f32[1, 12, 512, 512]" = torch.ops.aten.clone.default(mul_350, memory_format = torch.contiguous_format);  mul_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:266, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_24: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    mul_351: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(clone_56, alias_24);  clone_56 = None
    sum_153: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_351, [-1], True)
    mul_352: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_24, sum_153);  alias_24 = sum_153 = None
    sub_106: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_351, mul_352);  mul_351 = mul_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:260, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_56: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_106, 8.0);  sub_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:236, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_506: "f32[12, 512, 512]" = torch.ops.aten.view.default(div_56, [12, 512, 512]);  div_56 = None
    permute_416: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_75, [0, 2, 1]);  view_75 = None
    bmm_58: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_416, view_506);  permute_416 = None
    permute_417: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1]);  view_76 = None
    bmm_59: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_506, permute_417);  view_506 = permute_417 = None
    view_507: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_58, [1, 12, 64, 512]);  bmm_58 = None
    view_508: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_59, [1, 12, 512, 64]);  bmm_59 = None
    permute_418: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_507, [0, 1, 3, 2]);  view_507 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_419: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_508, [0, 2, 1, 3]);  view_508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_57: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_419, memory_format = torch.contiguous_format);  permute_419 = None
    view_509: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_57, [1, 512, 768]);  clone_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_420: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_504, [0, 2, 1, 3]);  view_504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_58: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_420, memory_format = torch.contiguous_format);  permute_420 = None
    view_510: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_58, [1, 512, 768]);  clone_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_511: "f32[512, 768]" = torch.ops.aten.view.default(view_510, [512, 768]);  view_510 = None
    permute_421: "f32[768, 768]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
    mm_104: "f32[512, 768]" = torch.ops.aten.mm.default(view_511, permute_421);  permute_421 = None
    permute_422: "f32[768, 512]" = torch.ops.aten.permute.default(view_511, [1, 0])
    mm_105: "f32[768, 768]" = torch.ops.aten.mm.default(permute_422, view_71);  permute_422 = view_71 = None
    permute_423: "f32[768, 768]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
    sum_154: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_511, [0], True);  view_511 = None
    view_512: "f32[768]" = torch.ops.aten.view.default(sum_154, [768]);  sum_154 = None
    permute_424: "f32[768, 768]" = torch.ops.aten.permute.default(permute_423, [1, 0]);  permute_423 = None
    view_513: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_104, [1, 512, 768]);  mm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_156: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_345, view_513);  mul_345 = view_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_425: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(permute_418, [0, 2, 1, 3]);  permute_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_514: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_425, [1, 512, 768]);  permute_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_515: "f32[512, 768]" = torch.ops.aten.view.default(view_514, [512, 768]);  view_514 = None
    permute_426: "f32[768, 768]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    mm_106: "f32[512, 768]" = torch.ops.aten.mm.default(view_515, permute_426);  permute_426 = None
    permute_427: "f32[768, 512]" = torch.ops.aten.permute.default(view_515, [1, 0])
    mm_107: "f32[768, 768]" = torch.ops.aten.mm.default(permute_427, view_68);  permute_427 = view_68 = None
    permute_428: "f32[768, 768]" = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
    sum_155: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_515, [0], True);  view_515 = None
    view_516: "f32[768]" = torch.ops.aten.view.default(sum_155, [768]);  sum_155 = None
    permute_429: "f32[768, 768]" = torch.ops.aten.permute.default(permute_428, [1, 0]);  permute_428 = None
    view_517: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_106, [1, 512, 768]);  mm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_157: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_156, view_517);  add_156 = view_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    view_518: "f32[512, 768]" = torch.ops.aten.view.default(view_509, [512, 768]);  view_509 = None
    permute_430: "f32[768, 768]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    mm_108: "f32[512, 768]" = torch.ops.aten.mm.default(view_518, permute_430);  permute_430 = None
    permute_431: "f32[768, 512]" = torch.ops.aten.permute.default(view_518, [1, 0])
    mm_109: "f32[768, 768]" = torch.ops.aten.mm.default(permute_431, view_66);  permute_431 = view_66 = None
    permute_432: "f32[768, 768]" = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
    sum_156: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_518, [0], True);  view_518 = None
    view_519: "f32[768]" = torch.ops.aten.view.default(sum_156, [768]);  sum_156 = None
    permute_433: "f32[768, 768]" = torch.ops.aten.permute.default(permute_432, [1, 0]);  permute_432 = None
    view_520: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_108, [1, 512, 768]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    add_158: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_157, view_520);  add_157 = view_520 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:381, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_107: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_27, getitem_33);  add_27 = getitem_33 = None
    mul_353: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_107, rsqrt_6);  sub_107 = None
    mul_354: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_158, primals_52);  primals_52 = None
    mul_355: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_354, 768)
    sum_157: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_354, [2], True)
    mul_356: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_354, mul_353);  mul_354 = None
    sum_158: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_356, [2], True);  mul_356 = None
    mul_357: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_353, sum_158);  sum_158 = None
    sub_108: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_355, sum_157);  mul_355 = sum_157 = None
    sub_109: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_108, mul_357);  sub_108 = mul_357 = None
    div_57: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
    mul_358: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_57, sub_109);  div_57 = sub_109 = None
    mul_359: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_158, mul_353);  mul_353 = None
    sum_159: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_359, [0, 1]);  mul_359 = None
    sum_160: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_158, [0, 1]);  add_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_32: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_31, torch.float32);  getitem_31 = None
    mul_360: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_32, 1.1111111111111112);  convert_element_type_32 = None
    mul_361: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_358, mul_360);  mul_360 = None
    clone_59: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_361, memory_format = torch.contiguous_format);  mul_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379, code: hidden_states = self.dense(hidden_states)
    view_521: "f32[512, 768]" = torch.ops.aten.view.default(clone_59, [512, 768]);  clone_59 = None
    permute_434: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    mm_110: "f32[512, 3072]" = torch.ops.aten.mm.default(view_521, permute_434);  permute_434 = None
    permute_435: "f32[768, 512]" = torch.ops.aten.permute.default(view_521, [1, 0])
    mm_111: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_435, view_64);  permute_435 = view_64 = None
    permute_436: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
    sum_161: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_521, [0], True);  view_521 = None
    view_522: "f32[768]" = torch.ops.aten.view.default(sum_161, [768]);  sum_161 = None
    permute_437: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_436, [1, 0]);  permute_436 = None
    view_523: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_110, [1, 512, 3072]);  mm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_362: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476)
    erf_21: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_362);  mul_362 = None
    add_159: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_363: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_159, 0.5);  add_159 = None
    mul_364: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, view_63)
    mul_365: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_364, -0.5);  mul_364 = None
    exp_25: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_365);  mul_365 = None
    mul_366: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
    mul_367: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, mul_366);  view_63 = mul_366 = None
    add_160: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_363, mul_367);  mul_363 = mul_367 = None
    mul_368: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_523, add_160);  view_523 = add_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    view_524: "f32[512, 3072]" = torch.ops.aten.view.default(mul_368, [512, 3072]);  mul_368 = None
    permute_438: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
    mm_112: "f32[512, 768]" = torch.ops.aten.mm.default(view_524, permute_438);  permute_438 = None
    permute_439: "f32[3072, 512]" = torch.ops.aten.permute.default(view_524, [1, 0])
    mm_113: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_439, view_62);  permute_439 = view_62 = None
    permute_440: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_113, [1, 0]);  mm_113 = None
    sum_162: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_524, [0], True);  view_524 = None
    view_525: "f32[3072]" = torch.ops.aten.view.default(sum_162, [3072]);  sum_162 = None
    permute_441: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_440, [1, 0]);  permute_440 = None
    view_526: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_112, [1, 512, 768]);  mm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    add_161: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_358, view_526);  mul_358 = view_526 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:300, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_110: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_23, getitem_29);  add_23 = getitem_29 = None
    mul_369: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_110, rsqrt_5);  sub_110 = None
    mul_370: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_161, primals_46);  primals_46 = None
    mul_371: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_370, 768)
    sum_163: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_370, [2], True)
    mul_372: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_370, mul_369);  mul_370 = None
    sum_164: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_372, [2], True);  mul_372 = None
    mul_373: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_369, sum_164);  sum_164 = None
    sub_111: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_371, sum_163);  mul_371 = sum_163 = None
    sub_112: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_111, mul_373);  sub_111 = mul_373 = None
    div_58: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
    mul_374: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_58, sub_112);  div_58 = sub_112 = None
    mul_375: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_161, mul_369);  mul_369 = None
    sum_165: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_375, [0, 1]);  mul_375 = None
    sum_166: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_161, [0, 1]);  add_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:299, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_33: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_27, torch.float32);  getitem_27 = None
    mul_376: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_33, 1.1111111111111112);  convert_element_type_33 = None
    mul_377: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_374, mul_376);  mul_376 = None
    clone_60: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_377, memory_format = torch.contiguous_format);  mul_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:298, code: hidden_states = self.dense(hidden_states)
    view_527: "f32[512, 768]" = torch.ops.aten.view.default(clone_60, [512, 768]);  clone_60 = None
    permute_442: "f32[768, 768]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
    mm_114: "f32[512, 768]" = torch.ops.aten.mm.default(view_527, permute_442);  permute_442 = None
    permute_443: "f32[768, 512]" = torch.ops.aten.permute.default(view_527, [1, 0])
    mm_115: "f32[768, 768]" = torch.ops.aten.mm.default(permute_443, view_60);  permute_443 = view_60 = None
    permute_444: "f32[768, 768]" = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
    sum_167: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_527, [0], True);  view_527 = None
    view_528: "f32[768]" = torch.ops.aten.view.default(sum_167, [768]);  sum_167 = None
    permute_445: "f32[768, 768]" = torch.ops.aten.permute.default(permute_444, [1, 0]);  permute_444 = None
    view_529: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_114, [1, 512, 768]);  mm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:280, code: context_layer = context_layer.view(new_context_layer_shape)
    view_530: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_529, [1, 512, 12, 64]);  view_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:278, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_446: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_530, [0, 2, 1, 3]);  view_530 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:276, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_531: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_446, [12, 512, 64]);  permute_446 = None
    permute_447: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None
    bmm_60: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_447, view_531);  permute_447 = None
    permute_448: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_57, [0, 2, 1]);  view_57 = None
    bmm_61: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_531, permute_448);  view_531 = permute_448 = None
    view_532: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_60, [1, 12, 512, 64]);  bmm_60 = None
    view_533: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_61, [1, 12, 512, 512]);  bmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:270, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_34: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_25, torch.float32);  getitem_25 = None
    mul_378: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_34, 1.1111111111111112);  convert_element_type_34 = None
    mul_379: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_533, mul_378);  view_533 = mul_378 = None
    clone_61: "f32[1, 12, 512, 512]" = torch.ops.aten.clone.default(mul_379, memory_format = torch.contiguous_format);  mul_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:266, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_25: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    mul_380: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(clone_61, alias_25);  clone_61 = None
    sum_168: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_380, [-1], True)
    mul_381: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_25, sum_168);  alias_25 = sum_168 = None
    sub_113: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_380, mul_381);  mul_380 = mul_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:260, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_59: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_113, 8.0);  sub_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:236, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_534: "f32[12, 512, 512]" = torch.ops.aten.view.default(div_59, [12, 512, 512]);  div_59 = None
    permute_449: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_53, [0, 2, 1]);  view_53 = None
    bmm_62: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_449, view_534);  permute_449 = None
    permute_450: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1]);  view_54 = None
    bmm_63: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_534, permute_450);  view_534 = permute_450 = None
    view_535: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_62, [1, 12, 64, 512]);  bmm_62 = None
    view_536: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_63, [1, 12, 512, 64]);  bmm_63 = None
    permute_451: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_535, [0, 1, 3, 2]);  view_535 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_452: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_536, [0, 2, 1, 3]);  view_536 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_62: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_452, memory_format = torch.contiguous_format);  permute_452 = None
    view_537: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_62, [1, 512, 768]);  clone_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_453: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_532, [0, 2, 1, 3]);  view_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_63: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_453, memory_format = torch.contiguous_format);  permute_453 = None
    view_538: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_63, [1, 512, 768]);  clone_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_539: "f32[512, 768]" = torch.ops.aten.view.default(view_538, [512, 768]);  view_538 = None
    permute_454: "f32[768, 768]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
    mm_116: "f32[512, 768]" = torch.ops.aten.mm.default(view_539, permute_454);  permute_454 = None
    permute_455: "f32[768, 512]" = torch.ops.aten.permute.default(view_539, [1, 0])
    mm_117: "f32[768, 768]" = torch.ops.aten.mm.default(permute_455, view_49);  permute_455 = view_49 = None
    permute_456: "f32[768, 768]" = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
    sum_169: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_539, [0], True);  view_539 = None
    view_540: "f32[768]" = torch.ops.aten.view.default(sum_169, [768]);  sum_169 = None
    permute_457: "f32[768, 768]" = torch.ops.aten.permute.default(permute_456, [1, 0]);  permute_456 = None
    view_541: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_116, [1, 512, 768]);  mm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_162: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_374, view_541);  mul_374 = view_541 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_458: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(permute_451, [0, 2, 1, 3]);  permute_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_542: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_458, [1, 512, 768]);  permute_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_543: "f32[512, 768]" = torch.ops.aten.view.default(view_542, [512, 768]);  view_542 = None
    permute_459: "f32[768, 768]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    mm_118: "f32[512, 768]" = torch.ops.aten.mm.default(view_543, permute_459);  permute_459 = None
    permute_460: "f32[768, 512]" = torch.ops.aten.permute.default(view_543, [1, 0])
    mm_119: "f32[768, 768]" = torch.ops.aten.mm.default(permute_460, view_46);  permute_460 = view_46 = None
    permute_461: "f32[768, 768]" = torch.ops.aten.permute.default(mm_119, [1, 0]);  mm_119 = None
    sum_170: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_543, [0], True);  view_543 = None
    view_544: "f32[768]" = torch.ops.aten.view.default(sum_170, [768]);  sum_170 = None
    permute_462: "f32[768, 768]" = torch.ops.aten.permute.default(permute_461, [1, 0]);  permute_461 = None
    view_545: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_118, [1, 512, 768]);  mm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_163: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_162, view_545);  add_162 = view_545 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    view_546: "f32[512, 768]" = torch.ops.aten.view.default(view_537, [512, 768]);  view_537 = None
    permute_463: "f32[768, 768]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    mm_120: "f32[512, 768]" = torch.ops.aten.mm.default(view_546, permute_463);  permute_463 = None
    permute_464: "f32[768, 512]" = torch.ops.aten.permute.default(view_546, [1, 0])
    mm_121: "f32[768, 768]" = torch.ops.aten.mm.default(permute_464, view_44);  permute_464 = view_44 = None
    permute_465: "f32[768, 768]" = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
    sum_171: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_546, [0], True);  view_546 = None
    view_547: "f32[768]" = torch.ops.aten.view.default(sum_171, [768]);  sum_171 = None
    permute_466: "f32[768, 768]" = torch.ops.aten.permute.default(permute_465, [1, 0]);  permute_465 = None
    view_548: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_120, [1, 512, 768]);  mm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    add_164: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_163, view_548);  add_163 = view_548 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:381, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_114: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_19, getitem_23);  add_19 = getitem_23 = None
    mul_382: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_114, rsqrt_4);  sub_114 = None
    mul_383: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_164, primals_36);  primals_36 = None
    mul_384: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_383, 768)
    sum_172: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_383, [2], True)
    mul_385: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_383, mul_382);  mul_383 = None
    sum_173: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_385, [2], True);  mul_385 = None
    mul_386: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_382, sum_173);  sum_173 = None
    sub_115: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_384, sum_172);  mul_384 = sum_172 = None
    sub_116: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_115, mul_386);  sub_115 = mul_386 = None
    div_60: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
    mul_387: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_60, sub_116);  div_60 = sub_116 = None
    mul_388: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_164, mul_382);  mul_382 = None
    sum_174: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_388, [0, 1]);  mul_388 = None
    sum_175: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_164, [0, 1]);  add_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_35: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_21, torch.float32);  getitem_21 = None
    mul_389: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_35, 1.1111111111111112);  convert_element_type_35 = None
    mul_390: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_387, mul_389);  mul_389 = None
    clone_64: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_390, memory_format = torch.contiguous_format);  mul_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379, code: hidden_states = self.dense(hidden_states)
    view_549: "f32[512, 768]" = torch.ops.aten.view.default(clone_64, [512, 768]);  clone_64 = None
    permute_467: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    mm_122: "f32[512, 3072]" = torch.ops.aten.mm.default(view_549, permute_467);  permute_467 = None
    permute_468: "f32[768, 512]" = torch.ops.aten.permute.default(view_549, [1, 0])
    mm_123: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_468, view_42);  permute_468 = view_42 = None
    permute_469: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_123, [1, 0]);  mm_123 = None
    sum_176: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_549, [0], True);  view_549 = None
    view_550: "f32[768]" = torch.ops.aten.view.default(sum_176, [768]);  sum_176 = None
    permute_470: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_469, [1, 0]);  permute_469 = None
    view_551: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_122, [1, 512, 3072]);  mm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_391: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476)
    erf_22: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_391);  mul_391 = None
    add_165: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_392: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_165, 0.5);  add_165 = None
    mul_393: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, view_41)
    mul_394: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_393, -0.5);  mul_393 = None
    exp_26: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_394);  mul_394 = None
    mul_395: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
    mul_396: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, mul_395);  view_41 = mul_395 = None
    add_166: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_392, mul_396);  mul_392 = mul_396 = None
    mul_397: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_551, add_166);  view_551 = add_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    view_552: "f32[512, 3072]" = torch.ops.aten.view.default(mul_397, [512, 3072]);  mul_397 = None
    permute_471: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    mm_124: "f32[512, 768]" = torch.ops.aten.mm.default(view_552, permute_471);  permute_471 = None
    permute_472: "f32[3072, 512]" = torch.ops.aten.permute.default(view_552, [1, 0])
    mm_125: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_472, view_40);  permute_472 = view_40 = None
    permute_473: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_125, [1, 0]);  mm_125 = None
    sum_177: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_552, [0], True);  view_552 = None
    view_553: "f32[3072]" = torch.ops.aten.view.default(sum_177, [3072]);  sum_177 = None
    permute_474: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_473, [1, 0]);  permute_473 = None
    view_554: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_124, [1, 512, 768]);  mm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    add_167: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_387, view_554);  mul_387 = view_554 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:300, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_117: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_15, getitem_19);  add_15 = getitem_19 = None
    mul_398: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_117, rsqrt_3);  sub_117 = None
    mul_399: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_167, primals_30);  primals_30 = None
    mul_400: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_399, 768)
    sum_178: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_399, [2], True)
    mul_401: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_399, mul_398);  mul_399 = None
    sum_179: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_401, [2], True);  mul_401 = None
    mul_402: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_398, sum_179);  sum_179 = None
    sub_118: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_400, sum_178);  mul_400 = sum_178 = None
    sub_119: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_118, mul_402);  sub_118 = mul_402 = None
    div_61: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
    mul_403: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_61, sub_119);  div_61 = sub_119 = None
    mul_404: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_167, mul_398);  mul_398 = None
    sum_180: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_404, [0, 1]);  mul_404 = None
    sum_181: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_167, [0, 1]);  add_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:299, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_36: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_17, torch.float32);  getitem_17 = None
    mul_405: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_36, 1.1111111111111112);  convert_element_type_36 = None
    mul_406: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_403, mul_405);  mul_405 = None
    clone_65: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_406, memory_format = torch.contiguous_format);  mul_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:298, code: hidden_states = self.dense(hidden_states)
    view_555: "f32[512, 768]" = torch.ops.aten.view.default(clone_65, [512, 768]);  clone_65 = None
    permute_475: "f32[768, 768]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
    mm_126: "f32[512, 768]" = torch.ops.aten.mm.default(view_555, permute_475);  permute_475 = None
    permute_476: "f32[768, 512]" = torch.ops.aten.permute.default(view_555, [1, 0])
    mm_127: "f32[768, 768]" = torch.ops.aten.mm.default(permute_476, view_38);  permute_476 = view_38 = None
    permute_477: "f32[768, 768]" = torch.ops.aten.permute.default(mm_127, [1, 0]);  mm_127 = None
    sum_182: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_555, [0], True);  view_555 = None
    view_556: "f32[768]" = torch.ops.aten.view.default(sum_182, [768]);  sum_182 = None
    permute_478: "f32[768, 768]" = torch.ops.aten.permute.default(permute_477, [1, 0]);  permute_477 = None
    view_557: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_126, [1, 512, 768]);  mm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:280, code: context_layer = context_layer.view(new_context_layer_shape)
    view_558: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_557, [1, 512, 12, 64]);  view_557 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:278, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_479: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_558, [0, 2, 1, 3]);  view_558 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:276, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_559: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_479, [12, 512, 64]);  permute_479 = None
    permute_480: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
    bmm_64: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_480, view_559);  permute_480 = None
    permute_481: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_35, [0, 2, 1]);  view_35 = None
    bmm_65: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_559, permute_481);  view_559 = permute_481 = None
    view_560: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_64, [1, 12, 512, 64]);  bmm_64 = None
    view_561: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_65, [1, 12, 512, 512]);  bmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:270, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_37: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_15, torch.float32);  getitem_15 = None
    mul_407: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_37, 1.1111111111111112);  convert_element_type_37 = None
    mul_408: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_561, mul_407);  view_561 = mul_407 = None
    clone_66: "f32[1, 12, 512, 512]" = torch.ops.aten.clone.default(mul_408, memory_format = torch.contiguous_format);  mul_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:266, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_26: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    mul_409: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(clone_66, alias_26);  clone_66 = None
    sum_183: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_409, [-1], True)
    mul_410: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_26, sum_183);  alias_26 = sum_183 = None
    sub_120: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_409, mul_410);  mul_409 = mul_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:260, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_62: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_120, 8.0);  sub_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:236, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_562: "f32[12, 512, 512]" = torch.ops.aten.view.default(div_62, [12, 512, 512]);  div_62 = None
    permute_482: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_31, [0, 2, 1]);  view_31 = None
    bmm_66: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_482, view_562);  permute_482 = None
    permute_483: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_32, [0, 2, 1]);  view_32 = None
    bmm_67: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_562, permute_483);  view_562 = permute_483 = None
    view_563: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_66, [1, 12, 64, 512]);  bmm_66 = None
    view_564: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_67, [1, 12, 512, 64]);  bmm_67 = None
    permute_484: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_563, [0, 1, 3, 2]);  view_563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_485: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_564, [0, 2, 1, 3]);  view_564 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_67: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_485, memory_format = torch.contiguous_format);  permute_485 = None
    view_565: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_67, [1, 512, 768]);  clone_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_486: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_560, [0, 2, 1, 3]);  view_560 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_68: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_486, memory_format = torch.contiguous_format);  permute_486 = None
    view_566: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_68, [1, 512, 768]);  clone_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_567: "f32[512, 768]" = torch.ops.aten.view.default(view_566, [512, 768]);  view_566 = None
    permute_487: "f32[768, 768]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
    mm_128: "f32[512, 768]" = torch.ops.aten.mm.default(view_567, permute_487);  permute_487 = None
    permute_488: "f32[768, 512]" = torch.ops.aten.permute.default(view_567, [1, 0])
    mm_129: "f32[768, 768]" = torch.ops.aten.mm.default(permute_488, view_27);  permute_488 = view_27 = None
    permute_489: "f32[768, 768]" = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
    sum_184: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_567, [0], True);  view_567 = None
    view_568: "f32[768]" = torch.ops.aten.view.default(sum_184, [768]);  sum_184 = None
    permute_490: "f32[768, 768]" = torch.ops.aten.permute.default(permute_489, [1, 0]);  permute_489 = None
    view_569: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_128, [1, 512, 768]);  mm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_168: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_403, view_569);  mul_403 = view_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_491: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(permute_484, [0, 2, 1, 3]);  permute_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_570: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_491, [1, 512, 768]);  permute_491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_571: "f32[512, 768]" = torch.ops.aten.view.default(view_570, [512, 768]);  view_570 = None
    permute_492: "f32[768, 768]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    mm_130: "f32[512, 768]" = torch.ops.aten.mm.default(view_571, permute_492);  permute_492 = None
    permute_493: "f32[768, 512]" = torch.ops.aten.permute.default(view_571, [1, 0])
    mm_131: "f32[768, 768]" = torch.ops.aten.mm.default(permute_493, view_24);  permute_493 = view_24 = None
    permute_494: "f32[768, 768]" = torch.ops.aten.permute.default(mm_131, [1, 0]);  mm_131 = None
    sum_185: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_571, [0], True);  view_571 = None
    view_572: "f32[768]" = torch.ops.aten.view.default(sum_185, [768]);  sum_185 = None
    permute_495: "f32[768, 768]" = torch.ops.aten.permute.default(permute_494, [1, 0]);  permute_494 = None
    view_573: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_130, [1, 512, 768]);  mm_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_169: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_168, view_573);  add_168 = view_573 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    view_574: "f32[512, 768]" = torch.ops.aten.view.default(view_565, [512, 768]);  view_565 = None
    permute_496: "f32[768, 768]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    mm_132: "f32[512, 768]" = torch.ops.aten.mm.default(view_574, permute_496);  permute_496 = None
    permute_497: "f32[768, 512]" = torch.ops.aten.permute.default(view_574, [1, 0])
    mm_133: "f32[768, 768]" = torch.ops.aten.mm.default(permute_497, view_22);  permute_497 = view_22 = None
    permute_498: "f32[768, 768]" = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
    sum_186: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_574, [0], True);  view_574 = None
    view_575: "f32[768]" = torch.ops.aten.view.default(sum_186, [768]);  sum_186 = None
    permute_499: "f32[768, 768]" = torch.ops.aten.permute.default(permute_498, [1, 0]);  permute_498 = None
    view_576: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_132, [1, 512, 768]);  mm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    add_170: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_169, view_576);  add_169 = view_576 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:381, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_121: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_11, getitem_13);  add_11 = getitem_13 = None
    mul_411: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_121, rsqrt_2);  sub_121 = None
    mul_412: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_170, primals_20);  primals_20 = None
    mul_413: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_412, 768)
    sum_187: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_412, [2], True)
    mul_414: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_412, mul_411);  mul_412 = None
    sum_188: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_414, [2], True);  mul_414 = None
    mul_415: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_411, sum_188);  sum_188 = None
    sub_122: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_413, sum_187);  mul_413 = sum_187 = None
    sub_123: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_122, mul_415);  sub_122 = mul_415 = None
    div_63: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
    mul_416: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_63, sub_123);  div_63 = sub_123 = None
    mul_417: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_170, mul_411);  mul_411 = None
    sum_189: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_417, [0, 1]);  mul_417 = None
    sum_190: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_170, [0, 1]);  add_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_38: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_11, torch.float32);  getitem_11 = None
    mul_418: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_38, 1.1111111111111112);  convert_element_type_38 = None
    mul_419: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_416, mul_418);  mul_418 = None
    clone_69: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_419, memory_format = torch.contiguous_format);  mul_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379, code: hidden_states = self.dense(hidden_states)
    view_577: "f32[512, 768]" = torch.ops.aten.view.default(clone_69, [512, 768]);  clone_69 = None
    permute_500: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    mm_134: "f32[512, 3072]" = torch.ops.aten.mm.default(view_577, permute_500);  permute_500 = None
    permute_501: "f32[768, 512]" = torch.ops.aten.permute.default(view_577, [1, 0])
    mm_135: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_501, view_20);  permute_501 = view_20 = None
    permute_502: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
    sum_191: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_577, [0], True);  view_577 = None
    view_578: "f32[768]" = torch.ops.aten.view.default(sum_191, [768]);  sum_191 = None
    permute_503: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_502, [1, 0]);  permute_502 = None
    view_579: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_134, [1, 512, 3072]);  mm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_420: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476)
    erf_23: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_420);  mul_420 = None
    add_171: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_421: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_171, 0.5);  add_171 = None
    mul_422: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, view_19)
    mul_423: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_422, -0.5);  mul_422 = None
    exp_27: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_423);  mul_423 = None
    mul_424: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_27, 0.3989422804014327);  exp_27 = None
    mul_425: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, mul_424);  view_19 = mul_424 = None
    add_172: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_421, mul_425);  mul_421 = mul_425 = None
    mul_426: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_579, add_172);  view_579 = add_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    view_580: "f32[512, 3072]" = torch.ops.aten.view.default(mul_426, [512, 3072]);  mul_426 = None
    permute_504: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    mm_136: "f32[512, 768]" = torch.ops.aten.mm.default(view_580, permute_504);  permute_504 = None
    permute_505: "f32[3072, 512]" = torch.ops.aten.permute.default(view_580, [1, 0])
    mm_137: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_505, view_18);  permute_505 = view_18 = None
    permute_506: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_137, [1, 0]);  mm_137 = None
    sum_192: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_580, [0], True);  view_580 = None
    view_581: "f32[3072]" = torch.ops.aten.view.default(sum_192, [3072]);  sum_192 = None
    permute_507: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_506, [1, 0]);  permute_506 = None
    view_582: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_136, [1, 512, 768]);  mm_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    add_173: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_416, view_582);  mul_416 = view_582 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:300, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_124: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_7, getitem_9);  add_7 = getitem_9 = None
    mul_427: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_124, rsqrt_1);  sub_124 = None
    mul_428: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_173, primals_14);  primals_14 = None
    mul_429: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_428, 768)
    sum_193: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_428, [2], True)
    mul_430: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_428, mul_427);  mul_428 = None
    sum_194: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_430, [2], True);  mul_430 = None
    mul_431: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_427, sum_194);  sum_194 = None
    sub_125: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_429, sum_193);  mul_429 = sum_193 = None
    sub_126: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_125, mul_431);  sub_125 = mul_431 = None
    div_64: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
    mul_432: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_64, sub_126);  div_64 = sub_126 = None
    mul_433: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_173, mul_427);  mul_427 = None
    sum_195: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_433, [0, 1]);  mul_433 = None
    sum_196: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_173, [0, 1]);  add_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:299, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_39: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_7, torch.float32);  getitem_7 = None
    mul_434: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_39, 1.1111111111111112);  convert_element_type_39 = None
    mul_435: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_432, mul_434);  mul_434 = None
    clone_70: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_435, memory_format = torch.contiguous_format);  mul_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:298, code: hidden_states = self.dense(hidden_states)
    view_583: "f32[512, 768]" = torch.ops.aten.view.default(clone_70, [512, 768]);  clone_70 = None
    permute_508: "f32[768, 768]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    mm_138: "f32[512, 768]" = torch.ops.aten.mm.default(view_583, permute_508);  permute_508 = None
    permute_509: "f32[768, 512]" = torch.ops.aten.permute.default(view_583, [1, 0])
    mm_139: "f32[768, 768]" = torch.ops.aten.mm.default(permute_509, view_16);  permute_509 = view_16 = None
    permute_510: "f32[768, 768]" = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
    sum_197: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_583, [0], True);  view_583 = None
    view_584: "f32[768]" = torch.ops.aten.view.default(sum_197, [768]);  sum_197 = None
    permute_511: "f32[768, 768]" = torch.ops.aten.permute.default(permute_510, [1, 0]);  permute_510 = None
    view_585: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_138, [1, 512, 768]);  mm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:280, code: context_layer = context_layer.view(new_context_layer_shape)
    view_586: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_585, [1, 512, 12, 64]);  view_585 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:278, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_512: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_586, [0, 2, 1, 3]);  view_586 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:276, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_587: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_512, [12, 512, 64]);  permute_512 = None
    permute_513: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
    bmm_68: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_513, view_587);  permute_513 = None
    permute_514: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None
    bmm_69: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_587, permute_514);  view_587 = permute_514 = None
    view_588: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_68, [1, 12, 512, 64]);  bmm_68 = None
    view_589: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_69, [1, 12, 512, 512]);  bmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:270, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_40: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_5, torch.float32);  getitem_5 = None
    mul_436: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_40, 1.1111111111111112);  convert_element_type_40 = None
    mul_437: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_589, mul_436);  view_589 = mul_436 = None
    clone_71: "f32[1, 12, 512, 512]" = torch.ops.aten.clone.default(mul_437, memory_format = torch.contiguous_format);  mul_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:266, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_27: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias);  alias = None
    mul_438: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(clone_71, alias_27);  clone_71 = None
    sum_198: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_438, [-1], True)
    mul_439: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_27, sum_198);  alias_27 = sum_198 = None
    sub_127: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_438, mul_439);  mul_438 = mul_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:260, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_65: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_127, 8.0);  sub_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:236, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_590: "f32[12, 512, 512]" = torch.ops.aten.view.default(div_65, [12, 512, 512]);  div_65 = None
    permute_515: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
    bmm_70: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_515, view_590);  permute_515 = None
    permute_516: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
    bmm_71: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_590, permute_516);  view_590 = permute_516 = None
    view_591: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_70, [1, 12, 64, 512]);  bmm_70 = None
    view_592: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_71, [1, 12, 512, 64]);  bmm_71 = None
    permute_517: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_591, [0, 1, 3, 2]);  view_591 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_518: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_592, [0, 2, 1, 3]);  view_592 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_72: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_518, memory_format = torch.contiguous_format);  permute_518 = None
    view_593: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_72, [1, 512, 768]);  clone_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_519: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_588, [0, 2, 1, 3]);  view_588 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_73: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_519, memory_format = torch.contiguous_format);  permute_519 = None
    view_594: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_73, [1, 512, 768]);  clone_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_595: "f32[512, 768]" = torch.ops.aten.view.default(view_594, [512, 768]);  view_594 = None
    permute_520: "f32[768, 768]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    mm_140: "f32[512, 768]" = torch.ops.aten.mm.default(view_595, permute_520);  permute_520 = None
    permute_521: "f32[768, 512]" = torch.ops.aten.permute.default(view_595, [1, 0])
    mm_141: "f32[768, 768]" = torch.ops.aten.mm.default(permute_521, view_5);  permute_521 = view_5 = None
    permute_522: "f32[768, 768]" = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
    sum_199: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_595, [0], True);  view_595 = None
    view_596: "f32[768]" = torch.ops.aten.view.default(sum_199, [768]);  sum_199 = None
    permute_523: "f32[768, 768]" = torch.ops.aten.permute.default(permute_522, [1, 0]);  permute_522 = None
    view_597: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_140, [1, 512, 768]);  mm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_174: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_432, view_597);  mul_432 = view_597 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_524: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(permute_517, [0, 2, 1, 3]);  permute_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_598: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_524, [1, 512, 768]);  permute_524 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_599: "f32[512, 768]" = torch.ops.aten.view.default(view_598, [512, 768]);  view_598 = None
    permute_525: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    mm_142: "f32[512, 768]" = torch.ops.aten.mm.default(view_599, permute_525);  permute_525 = None
    permute_526: "f32[768, 512]" = torch.ops.aten.permute.default(view_599, [1, 0])
    mm_143: "f32[768, 768]" = torch.ops.aten.mm.default(permute_526, view_2);  permute_526 = view_2 = None
    permute_527: "f32[768, 768]" = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
    sum_200: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_599, [0], True);  view_599 = None
    view_600: "f32[768]" = torch.ops.aten.view.default(sum_200, [768]);  sum_200 = None
    permute_528: "f32[768, 768]" = torch.ops.aten.permute.default(permute_527, [1, 0]);  permute_527 = None
    view_601: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_142, [1, 512, 768]);  mm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_175: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_174, view_601);  add_174 = view_601 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    view_602: "f32[512, 768]" = torch.ops.aten.view.default(view_593, [512, 768]);  view_593 = None
    permute_529: "f32[768, 768]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    mm_144: "f32[512, 768]" = torch.ops.aten.mm.default(view_602, permute_529);  permute_529 = None
    permute_530: "f32[768, 512]" = torch.ops.aten.permute.default(view_602, [1, 0])
    mm_145: "f32[768, 768]" = torch.ops.aten.mm.default(permute_530, view);  permute_530 = view = None
    permute_531: "f32[768, 768]" = torch.ops.aten.permute.default(mm_145, [1, 0]);  mm_145 = None
    sum_201: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_602, [0], True);  view_602 = None
    view_603: "f32[768]" = torch.ops.aten.view.default(sum_201, [768]);  sum_201 = None
    permute_532: "f32[768, 768]" = torch.ops.aten.permute.default(permute_531, [1, 0]);  permute_531 = None
    view_604: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_144, [1, 512, 768]);  mm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    add_176: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_175, view_604);  add_175 = view_604 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:133, code: embeddings = self.dropout(embeddings)
    convert_element_type_41: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_3, torch.float32);  getitem_3 = None
    mul_440: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_41, 1.1111111111111112);  convert_element_type_41 = None
    mul_441: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_176, mul_440);  add_176 = mul_440 = None
    clone_74: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_441, memory_format = torch.contiguous_format);  mul_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:132, code: embeddings = self.LayerNorm(embeddings)
    sub_128: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_3, getitem_1);  add_3 = getitem_1 = None
    mul_442: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_128, rsqrt);  sub_128 = None
    mul_443: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(clone_74, primals_4);  primals_4 = None
    mul_444: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_443, 768)
    sum_202: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_443, [2], True)
    mul_445: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_443, mul_442);  mul_443 = None
    sum_203: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_445, [2], True);  mul_445 = None
    mul_446: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_442, sum_203);  sum_203 = None
    sub_129: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_444, sum_202);  mul_444 = sum_202 = None
    sub_130: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_129, mul_446);  sub_129 = mul_446 = None
    div_66: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
    mul_447: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_66, sub_130);  div_66 = sub_130 = None
    mul_448: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(clone_74, mul_442);  mul_442 = None
    sum_204: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_448, [0, 1]);  mul_448 = None
    sum_205: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_74, [0, 1]);  clone_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:130, code: position_embeddings = self.position_embeddings(position_ids)
    eq: "b8[1, 512]" = torch.ops.aten.eq.Scalar(add_1, 0)
    unsqueeze_8: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
    scalar_tensor_8: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_8: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_8, scalar_tensor_8, mul_447);  unsqueeze_8 = scalar_tensor_8 = None
    full_3: "f32[512, 768]" = torch.ops.aten.full.default([512, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put: "f32[512, 768]" = torch.ops.aten._unsafe_index_put.default(full_3, [add_1], where_8, True);  full_3 = add_1 = where_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:126, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    eq_1: "b8[1, 512]" = torch.ops.aten.eq.Scalar(expand, -1)
    unsqueeze_9: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
    scalar_tensor_9: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_9: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_9, scalar_tensor_9, mul_447);  unsqueeze_9 = scalar_tensor_9 = None
    full_4: "f32[2, 768]" = torch.ops.aten.full.default([2, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put_1: "f32[2, 768]" = torch.ops.aten._unsafe_index_put.default(full_4, [expand], where_9, True);  full_4 = expand = where_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:125, code: inputs_embeds = self.word_embeddings(input_ids)
    eq_2: "b8[1, 512]" = torch.ops.aten.eq.Scalar(primals_201, 0)
    unsqueeze_10: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_2, -1);  eq_2 = None
    scalar_tensor_10: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_10: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_10, scalar_tensor_10, mul_447);  unsqueeze_10 = scalar_tensor_10 = mul_447 = None
    full_5: "f32[50265, 768]" = torch.ops.aten.full.default([50265, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put_2: "f32[50265, 768]" = torch.ops.aten._unsafe_index_put.default(full_5, [primals_201], where_10, True);  full_5 = primals_201 = where_10 = None
    return pytree.tree_unflatten([div_26, clone_12, clone_13, _unsafe_index_put_2, _unsafe_index_put_1, _unsafe_index_put, sum_204, sum_205, permute_532, view_603, permute_528, view_600, permute_523, view_596, permute_511, view_584, sum_195, sum_196, permute_507, view_581, permute_503, view_578, sum_189, sum_190, permute_499, view_575, permute_495, view_572, permute_490, view_568, permute_478, view_556, sum_180, sum_181, permute_474, view_553, permute_470, view_550, sum_174, sum_175, permute_466, view_547, permute_462, view_544, permute_457, view_540, permute_445, view_528, sum_165, sum_166, permute_441, view_525, permute_437, view_522, sum_159, sum_160, permute_433, view_519, permute_429, view_516, permute_424, view_512, permute_412, view_500, sum_150, sum_151, permute_408, view_497, permute_404, view_494, sum_144, sum_145, permute_400, view_491, permute_396, view_488, permute_391, view_484, permute_379, view_472, sum_135, sum_136, permute_375, view_469, permute_371, view_466, sum_129, sum_130, permute_367, view_463, permute_363, view_460, permute_358, view_456, permute_346, view_444, sum_120, sum_121, permute_342, view_441, permute_338, view_438, sum_114, sum_115, permute_334, view_435, permute_330, view_432, permute_325, view_428, permute_313, view_416, sum_105, sum_106, permute_309, view_413, permute_305, view_410, sum_99, sum_100, permute_301, view_407, permute_297, view_404, permute_292, view_400, permute_280, view_388, sum_90, sum_91, permute_276, view_385, permute_272, view_382, sum_84, sum_85, permute_268, view_379, permute_264, view_376, permute_259, view_372, permute_247, view_360, sum_75, sum_76, permute_243, view_357, permute_239, view_354, sum_69, sum_70, permute_235, view_351, permute_231, view_348, permute_226, view_344, permute_214, view_332, sum_60, sum_61, permute_210, view_329, permute_206, view_326, sum_54, sum_55, permute_202, view_323, permute_198, view_320, permute_193, view_316, permute_181, view_304, sum_45, sum_46, permute_177, view_301, permute_173, view_298, sum_39, sum_40, permute_169, view_295, permute_165, view_292, permute_160, view_288, permute_148, view_276, sum_30, sum_31, permute_144, view_273, permute_140, view_270, sum_24, sum_25, permute_136, view_267, None, None, None, None], self._out_spec)
    