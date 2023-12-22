from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[30522, 128]"; primals_2: "f32[2, 128]"; primals_3: "f32[512, 128]"; primals_4: "f32[128]"; primals_5: "f32[128]"; primals_6: "f32[256, 128]"; primals_7: "f32[256]"; primals_8: "f32[256, 256]"; primals_9: "f32[256]"; primals_10: "f32[256, 256]"; primals_11: "f32[256]"; primals_12: "f32[256, 256]"; primals_13: "f32[256]"; primals_14: "f32[256, 256]"; primals_15: "f32[256]"; primals_16: "f32[256]"; primals_17: "f32[256]"; primals_18: "f32[1024, 256]"; primals_19: "f32[1024]"; primals_20: "f32[256, 1024]"; primals_21: "f32[256]"; primals_22: "f32[256]"; primals_23: "f32[256]"; primals_24: "f32[256, 256]"; primals_25: "f32[256]"; primals_26: "f32[256, 256]"; primals_27: "f32[256]"; primals_28: "f32[256, 256]"; primals_29: "f32[256]"; primals_30: "f32[256, 256]"; primals_31: "f32[256]"; primals_32: "f32[256]"; primals_33: "f32[256]"; primals_34: "f32[1024, 256]"; primals_35: "f32[1024]"; primals_36: "f32[256, 1024]"; primals_37: "f32[256]"; primals_38: "f32[256]"; primals_39: "f32[256]"; primals_40: "f32[256, 256]"; primals_41: "f32[256]"; primals_42: "f32[256, 256]"; primals_43: "f32[256]"; primals_44: "f32[256, 256]"; primals_45: "f32[256]"; primals_46: "f32[256, 256]"; primals_47: "f32[256]"; primals_48: "f32[256]"; primals_49: "f32[256]"; primals_50: "f32[1024, 256]"; primals_51: "f32[1024]"; primals_52: "f32[256, 1024]"; primals_53: "f32[256]"; primals_54: "f32[256]"; primals_55: "f32[256]"; primals_56: "f32[256, 256]"; primals_57: "f32[256]"; primals_58: "f32[256, 256]"; primals_59: "f32[256]"; primals_60: "f32[256, 256]"; primals_61: "f32[256]"; primals_62: "f32[256, 256]"; primals_63: "f32[256]"; primals_64: "f32[256]"; primals_65: "f32[256]"; primals_66: "f32[1024, 256]"; primals_67: "f32[1024]"; primals_68: "f32[256, 1024]"; primals_69: "f32[256]"; primals_70: "f32[256]"; primals_71: "f32[256]"; primals_72: "f32[256, 256]"; primals_73: "f32[256]"; primals_74: "f32[256, 256]"; primals_75: "f32[256]"; primals_76: "f32[256, 256]"; primals_77: "f32[256]"; primals_78: "f32[256, 256]"; primals_79: "f32[256]"; primals_80: "f32[256]"; primals_81: "f32[256]"; primals_82: "f32[1024, 256]"; primals_83: "f32[1024]"; primals_84: "f32[256, 1024]"; primals_85: "f32[256]"; primals_86: "f32[256]"; primals_87: "f32[256]"; primals_88: "f32[256, 256]"; primals_89: "f32[256]"; primals_90: "f32[256, 256]"; primals_91: "f32[256]"; primals_92: "f32[256, 256]"; primals_93: "f32[256]"; primals_94: "f32[256, 256]"; primals_95: "f32[256]"; primals_96: "f32[256]"; primals_97: "f32[256]"; primals_98: "f32[1024, 256]"; primals_99: "f32[1024]"; primals_100: "f32[256, 1024]"; primals_101: "f32[256]"; primals_102: "f32[256]"; primals_103: "f32[256]"; primals_104: "f32[256, 256]"; primals_105: "f32[256]"; primals_106: "f32[256, 256]"; primals_107: "f32[256]"; primals_108: "f32[256, 256]"; primals_109: "f32[256]"; primals_110: "f32[256, 256]"; primals_111: "f32[256]"; primals_112: "f32[256]"; primals_113: "f32[256]"; primals_114: "f32[1024, 256]"; primals_115: "f32[1024]"; primals_116: "f32[256, 1024]"; primals_117: "f32[256]"; primals_118: "f32[256]"; primals_119: "f32[256]"; primals_120: "f32[256, 256]"; primals_121: "f32[256]"; primals_122: "f32[256, 256]"; primals_123: "f32[256]"; primals_124: "f32[256, 256]"; primals_125: "f32[256]"; primals_126: "f32[256, 256]"; primals_127: "f32[256]"; primals_128: "f32[256]"; primals_129: "f32[256]"; primals_130: "f32[1024, 256]"; primals_131: "f32[1024]"; primals_132: "f32[256, 1024]"; primals_133: "f32[256]"; primals_134: "f32[256]"; primals_135: "f32[256]"; primals_136: "f32[256, 256]"; primals_137: "f32[256]"; primals_138: "f32[256, 256]"; primals_139: "f32[256]"; primals_140: "f32[256, 256]"; primals_141: "f32[256]"; primals_142: "f32[256, 256]"; primals_143: "f32[256]"; primals_144: "f32[256]"; primals_145: "f32[256]"; primals_146: "f32[1024, 256]"; primals_147: "f32[1024]"; primals_148: "f32[256, 1024]"; primals_149: "f32[256]"; primals_150: "f32[256]"; primals_151: "f32[256]"; primals_152: "f32[256, 256]"; primals_153: "f32[256]"; primals_154: "f32[256, 256]"; primals_155: "f32[256]"; primals_156: "f32[256, 256]"; primals_157: "f32[256]"; primals_158: "f32[256, 256]"; primals_159: "f32[256]"; primals_160: "f32[256]"; primals_161: "f32[256]"; primals_162: "f32[1024, 256]"; primals_163: "f32[1024]"; primals_164: "f32[256, 1024]"; primals_165: "f32[256]"; primals_166: "f32[256]"; primals_167: "f32[256]"; primals_168: "f32[256, 256]"; primals_169: "f32[256]"; primals_170: "f32[256, 256]"; primals_171: "f32[256]"; primals_172: "f32[256, 256]"; primals_173: "f32[256]"; primals_174: "f32[256, 256]"; primals_175: "f32[256]"; primals_176: "f32[256]"; primals_177: "f32[256]"; primals_178: "f32[1024, 256]"; primals_179: "f32[1024]"; primals_180: "f32[256, 1024]"; primals_181: "f32[256]"; primals_182: "f32[256]"; primals_183: "f32[256]"; primals_184: "f32[256, 256]"; primals_185: "f32[256]"; primals_186: "f32[256, 256]"; primals_187: "f32[256]"; primals_188: "f32[256, 256]"; primals_189: "f32[256]"; primals_190: "f32[256, 256]"; primals_191: "f32[256]"; primals_192: "f32[256]"; primals_193: "f32[256]"; primals_194: "f32[1024, 256]"; primals_195: "f32[1024]"; primals_196: "f32[256, 1024]"; primals_197: "f32[256]"; primals_198: "f32[256]"; primals_199: "f32[256]"; primals_200: "f32[128, 256]"; primals_201: "f32[128]"; primals_202: "f32[128]"; primals_203: "f32[128]"; primals_204: "f32[30522, 128]"; primals_205: "f32[30522]"; primals_206: "i64[1, 512]"; primals_207: "i64[1, 512]"; primals_208: "i64[1, 512]"; primals_209: "i64[1, 512]"; tangents_1: "f32[]"; tangents_2: "f32[1, 512, 30522]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, tangents_1, tangents_2, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:885, code: attention_mask = torch.ones(input_shape, device=device)
    full: "f32[1, 512]" = torch.ops.aten.full.default([1, 512], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:888, code: buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
    slice_1: "i64[1, 512]" = torch.ops.aten.slice.Tensor(primals_206, 0, 0, 9223372036854775807);  primals_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:889, code: buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
    expand: "i64[1, 512]" = torch.ops.aten.expand.default(slice_1, [1, 512]);  slice_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:916, code: extended_attention_mask = attention_mask[:, None, None, :]
    slice_2: "f32[1, 512]" = torch.ops.aten.slice.Tensor(full, 0, 0, 9223372036854775807);  full = None
    unsqueeze: "f32[1, 1, 512]" = torch.ops.aten.unsqueeze.default(slice_2, 1);  slice_2 = None
    unsqueeze_1: "f32[1, 1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
    slice_3: "f32[1, 1, 1, 512]" = torch.ops.aten.slice.Tensor(unsqueeze_1, 3, 0, 9223372036854775807);  unsqueeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:928, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    sub: "f32[1, 1, 1, 512]" = torch.ops.aten.sub.Tensor(1.0, slice_3);  slice_3 = None
    mul: "f32[1, 1, 1, 512]" = torch.ops.aten.mul.Tensor(sub, -3.4028234663852886e+38);  sub = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:189, code: position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
    slice_4: "i64[1, 512]" = torch.ops.aten.slice.Tensor(primals_207, 0, 0, 9223372036854775807);  primals_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:203, code: inputs_embeds = self.word_embeddings(input_ids)
    embedding: "f32[1, 512, 128]" = torch.ops.aten.embedding.default(primals_1, primals_209, 0);  primals_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:204, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    embedding_1: "f32[1, 512, 128]" = torch.ops.aten.embedding.default(primals_2, expand);  primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:206, code: embeddings = inputs_embeds + token_type_embeddings
    add: "f32[1, 512, 128]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:208, code: position_embeddings = self.position_embeddings(position_ids)
    embedding_2: "f32[1, 512, 128]" = torch.ops.aten.embedding.default(primals_3, slice_4);  primals_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:209, code: embeddings += position_embeddings
    add_1: "f32[1, 512, 128]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:210, code: embeddings = self.LayerNorm(embeddings)
    var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 512, 1]" = var_mean[0]
    getitem_1: "f32[1, 512, 1]" = var_mean[1];  var_mean = None
    add_2: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
    rsqrt: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
    sub_1: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(add_1, getitem_1)
    mul_1: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = None
    mul_2: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_1, primals_4);  mul_1 = None
    add_3: "f32[1, 512, 128]" = torch.ops.aten.add.Tensor(mul_2, primals_5);  mul_2 = primals_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:211, code: embeddings = self.dropout(embeddings)
    native_dropout = torch.ops.aten.native_dropout.default(add_3, 0.1, True);  add_3 = None
    getitem_2: "f32[1, 512, 128]" = native_dropout[0]
    getitem_3: "b8[1, 512, 128]" = native_dropout[1];  native_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:918, code: hidden_states = self.embeddings_project(hidden_states)
    view: "f32[512, 128]" = torch.ops.aten.view.default(getitem_2, [512, 128]);  getitem_2 = None
    permute: "f32[128, 256]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
    addmm: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_7, view, permute);  primals_7 = None
    view_1: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm, [1, 512, 256]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_2: "f32[512, 256]" = torch.ops.aten.view.default(view_1, [512, 256])
    permute_1: "f32[256, 256]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
    addmm_1: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_9, view_2, permute_1);  primals_9 = None
    view_3: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_1, [1, 512, 256]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_4: "f32[512, 256]" = torch.ops.aten.view.default(view_1, [512, 256])
    permute_2: "f32[256, 256]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
    addmm_2: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_11, view_4, permute_2);  primals_11 = None
    view_5: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_2, [1, 512, 256]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_6: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_5, [1, 512, 4, 64]);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_3: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_7: "f32[512, 256]" = torch.ops.aten.view.default(view_1, [512, 256])
    permute_4: "f32[256, 256]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
    addmm_3: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_13, view_7, permute_4);  primals_13 = None
    view_8: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_3, [1, 512, 256]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_9: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_8, [1, 512, 4, 64]);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_5: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_10: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_3, [1, 512, 4, 64]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_6: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_7: "f32[1, 4, 64, 512]" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
    expand_1: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_6, [1, 4, 512, 64]);  permute_6 = None
    view_11: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_1, [4, 512, 64]);  expand_1 = None
    expand_2: "f32[1, 4, 64, 512]" = torch.ops.aten.expand.default(permute_7, [1, 4, 64, 512]);  permute_7 = None
    view_12: "f32[4, 64, 512]" = torch.ops.aten.view.default(expand_2, [4, 64, 512]);  expand_2 = None
    bmm: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_11, view_12)
    view_13: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm, [1, 4, 512, 512]);  bmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(view_13, 8.0);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:324, code: attention_scores = attention_scores + attention_mask
    add_4: "f32[1, 4, 512, 512]" = torch.ops.aten.add.Tensor(div, mul);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax: "f32[1, 4, 512, 1]" = torch.ops.aten.amax.default(add_4, [-1], True)
    sub_2: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_4, amax);  add_4 = amax = None
    exp: "f32[1, 4, 512, 512]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
    sum_1: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div_1: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(div_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    native_dropout_1 = torch.ops.aten.native_dropout.default(div_1, 0.1, True);  div_1 = None
    getitem_4: "f32[1, 4, 512, 512]" = native_dropout_1[0]
    getitem_5: "b8[1, 4, 512, 512]" = native_dropout_1[1];  native_dropout_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_3: "f32[1, 4, 512, 512]" = torch.ops.aten.expand.default(getitem_4, [1, 4, 512, 512]);  getitem_4 = None
    view_14: "f32[4, 512, 512]" = torch.ops.aten.view.default(expand_3, [4, 512, 512]);  expand_3 = None
    expand_4: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_5, [1, 4, 512, 64]);  permute_5 = None
    view_15: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_4, [4, 512, 64]);  expand_4 = None
    bmm_1: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_14, view_15)
    view_16: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_1, [1, 4, 512, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_8: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
    clone: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_17: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone, [1, 512, 256]);  clone = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_18: "f32[512, 256]" = torch.ops.aten.view.default(view_17, [512, 256]);  view_17 = None
    permute_9: "f32[256, 256]" = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None
    addmm_4: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_15, view_18, permute_9);  primals_15 = None
    view_19: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_4, [1, 512, 256]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    native_dropout_2 = torch.ops.aten.native_dropout.default(view_19, 0.1, True);  view_19 = None
    getitem_6: "f32[1, 512, 256]" = native_dropout_2[0]
    getitem_7: "b8[1, 512, 256]" = native_dropout_2[1];  native_dropout_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_5: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(getitem_6, view_1);  getitem_6 = view_1 = None
    var_mean_1 = torch.ops.aten.var_mean.correction(add_5, [2], correction = 0, keepdim = True)
    getitem_8: "f32[1, 512, 1]" = var_mean_1[0]
    getitem_9: "f32[1, 512, 1]" = var_mean_1[1];  var_mean_1 = None
    add_6: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
    rsqrt_1: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_3: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_5, getitem_9)
    mul_3: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = None
    mul_4: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_3, primals_16);  mul_3 = None
    add_7: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_4, primals_17);  mul_4 = primals_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_20: "f32[512, 256]" = torch.ops.aten.view.default(add_7, [512, 256])
    permute_10: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
    addmm_5: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_19, view_20, permute_10);  primals_19 = None
    view_21: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_5, [1, 512, 1024]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_5: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_21, 0.5)
    mul_6: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_21, 0.7071067811865476)
    erf: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
    add_8: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_7: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_5, add_8);  mul_5 = add_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_22: "f32[512, 1024]" = torch.ops.aten.view.default(mul_7, [512, 1024]);  mul_7 = None
    permute_11: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_20, [1, 0]);  primals_20 = None
    addmm_6: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_21, view_22, permute_11);  primals_21 = None
    view_23: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_6, [1, 512, 256]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    native_dropout_3 = torch.ops.aten.native_dropout.default(view_23, 0.1, True);  view_23 = None
    getitem_10: "f32[1, 512, 256]" = native_dropout_3[0]
    getitem_11: "b8[1, 512, 256]" = native_dropout_3[1];  native_dropout_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_9: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(getitem_10, add_7);  getitem_10 = add_7 = None
    var_mean_2 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
    getitem_12: "f32[1, 512, 1]" = var_mean_2[0]
    getitem_13: "f32[1, 512, 1]" = var_mean_2[1];  var_mean_2 = None
    add_10: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
    rsqrt_2: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
    sub_4: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_9, getitem_13)
    mul_8: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = None
    mul_9: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_8, primals_22);  mul_8 = None
    add_11: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_9, primals_23);  mul_9 = primals_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_24: "f32[512, 256]" = torch.ops.aten.view.default(add_11, [512, 256])
    permute_12: "f32[256, 256]" = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
    addmm_7: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_25, view_24, permute_12);  primals_25 = None
    view_25: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_7, [1, 512, 256]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_26: "f32[512, 256]" = torch.ops.aten.view.default(add_11, [512, 256])
    permute_13: "f32[256, 256]" = torch.ops.aten.permute.default(primals_26, [1, 0]);  primals_26 = None
    addmm_8: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_27, view_26, permute_13);  primals_27 = None
    view_27: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_8, [1, 512, 256]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_28: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_27, [1, 512, 4, 64]);  view_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_14: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_29: "f32[512, 256]" = torch.ops.aten.view.default(add_11, [512, 256])
    permute_15: "f32[256, 256]" = torch.ops.aten.permute.default(primals_28, [1, 0]);  primals_28 = None
    addmm_9: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_29, view_29, permute_15);  primals_29 = None
    view_30: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_9, [1, 512, 256]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_31: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_30, [1, 512, 4, 64]);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_16: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_32: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_25, [1, 512, 4, 64]);  view_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_17: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_18: "f32[1, 4, 64, 512]" = torch.ops.aten.permute.default(permute_14, [0, 1, 3, 2]);  permute_14 = None
    expand_5: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_17, [1, 4, 512, 64]);  permute_17 = None
    view_33: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_5, [4, 512, 64]);  expand_5 = None
    expand_6: "f32[1, 4, 64, 512]" = torch.ops.aten.expand.default(permute_18, [1, 4, 64, 512]);  permute_18 = None
    view_34: "f32[4, 64, 512]" = torch.ops.aten.view.default(expand_6, [4, 64, 512]);  expand_6 = None
    bmm_2: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_33, view_34)
    view_35: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_2, [1, 4, 512, 512]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_2: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(view_35, 8.0);  view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:324, code: attention_scores = attention_scores + attention_mask
    add_12: "f32[1, 4, 512, 512]" = torch.ops.aten.add.Tensor(div_2, mul);  div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_1: "f32[1, 4, 512, 1]" = torch.ops.aten.amax.default(add_12, [-1], True)
    sub_5: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_12, amax_1);  add_12 = amax_1 = None
    exp_1: "f32[1, 4, 512, 512]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
    sum_2: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_3: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    alias_1: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(div_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    native_dropout_4 = torch.ops.aten.native_dropout.default(div_3, 0.1, True);  div_3 = None
    getitem_14: "f32[1, 4, 512, 512]" = native_dropout_4[0]
    getitem_15: "b8[1, 4, 512, 512]" = native_dropout_4[1];  native_dropout_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_7: "f32[1, 4, 512, 512]" = torch.ops.aten.expand.default(getitem_14, [1, 4, 512, 512]);  getitem_14 = None
    view_36: "f32[4, 512, 512]" = torch.ops.aten.view.default(expand_7, [4, 512, 512]);  expand_7 = None
    expand_8: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_16, [1, 4, 512, 64]);  permute_16 = None
    view_37: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_8, [4, 512, 64]);  expand_8 = None
    bmm_3: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_36, view_37)
    view_38: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_3, [1, 4, 512, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_19: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
    clone_1: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_39: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_1, [1, 512, 256]);  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_40: "f32[512, 256]" = torch.ops.aten.view.default(view_39, [512, 256]);  view_39 = None
    permute_20: "f32[256, 256]" = torch.ops.aten.permute.default(primals_30, [1, 0]);  primals_30 = None
    addmm_10: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_31, view_40, permute_20);  primals_31 = None
    view_41: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_10, [1, 512, 256]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    native_dropout_5 = torch.ops.aten.native_dropout.default(view_41, 0.1, True);  view_41 = None
    getitem_16: "f32[1, 512, 256]" = native_dropout_5[0]
    getitem_17: "b8[1, 512, 256]" = native_dropout_5[1];  native_dropout_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_13: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(getitem_16, add_11);  getitem_16 = add_11 = None
    var_mean_3 = torch.ops.aten.var_mean.correction(add_13, [2], correction = 0, keepdim = True)
    getitem_18: "f32[1, 512, 1]" = var_mean_3[0]
    getitem_19: "f32[1, 512, 1]" = var_mean_3[1];  var_mean_3 = None
    add_14: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
    rsqrt_3: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
    sub_6: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_13, getitem_19)
    mul_10: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_3);  sub_6 = None
    mul_11: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_10, primals_32);  mul_10 = None
    add_15: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_11, primals_33);  mul_11 = primals_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_42: "f32[512, 256]" = torch.ops.aten.view.default(add_15, [512, 256])
    permute_21: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
    addmm_11: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_35, view_42, permute_21);  primals_35 = None
    view_43: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_11, [1, 512, 1024]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_12: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_43, 0.5)
    mul_13: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_43, 0.7071067811865476)
    erf_1: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
    add_16: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_14: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_12, add_16);  mul_12 = add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_44: "f32[512, 1024]" = torch.ops.aten.view.default(mul_14, [512, 1024]);  mul_14 = None
    permute_22: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_36, [1, 0]);  primals_36 = None
    addmm_12: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_37, view_44, permute_22);  primals_37 = None
    view_45: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_12, [1, 512, 256]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    native_dropout_6 = torch.ops.aten.native_dropout.default(view_45, 0.1, True);  view_45 = None
    getitem_20: "f32[1, 512, 256]" = native_dropout_6[0]
    getitem_21: "b8[1, 512, 256]" = native_dropout_6[1];  native_dropout_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_17: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(getitem_20, add_15);  getitem_20 = add_15 = None
    var_mean_4 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
    getitem_22: "f32[1, 512, 1]" = var_mean_4[0]
    getitem_23: "f32[1, 512, 1]" = var_mean_4[1];  var_mean_4 = None
    add_18: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
    rsqrt_4: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    sub_7: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_17, getitem_23)
    mul_15: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_4);  sub_7 = None
    mul_16: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_15, primals_38);  mul_15 = None
    add_19: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_16, primals_39);  mul_16 = primals_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_46: "f32[512, 256]" = torch.ops.aten.view.default(add_19, [512, 256])
    permute_23: "f32[256, 256]" = torch.ops.aten.permute.default(primals_40, [1, 0]);  primals_40 = None
    addmm_13: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_41, view_46, permute_23);  primals_41 = None
    view_47: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_13, [1, 512, 256]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_48: "f32[512, 256]" = torch.ops.aten.view.default(add_19, [512, 256])
    permute_24: "f32[256, 256]" = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
    addmm_14: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_43, view_48, permute_24);  primals_43 = None
    view_49: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_14, [1, 512, 256]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_50: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_49, [1, 512, 4, 64]);  view_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_25: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_51: "f32[512, 256]" = torch.ops.aten.view.default(add_19, [512, 256])
    permute_26: "f32[256, 256]" = torch.ops.aten.permute.default(primals_44, [1, 0]);  primals_44 = None
    addmm_15: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_45, view_51, permute_26);  primals_45 = None
    view_52: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_15, [1, 512, 256]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_53: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_52, [1, 512, 4, 64]);  view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_27: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_54: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_47, [1, 512, 4, 64]);  view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_28: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_29: "f32[1, 4, 64, 512]" = torch.ops.aten.permute.default(permute_25, [0, 1, 3, 2]);  permute_25 = None
    expand_9: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_28, [1, 4, 512, 64]);  permute_28 = None
    view_55: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_9, [4, 512, 64]);  expand_9 = None
    expand_10: "f32[1, 4, 64, 512]" = torch.ops.aten.expand.default(permute_29, [1, 4, 64, 512]);  permute_29 = None
    view_56: "f32[4, 64, 512]" = torch.ops.aten.view.default(expand_10, [4, 64, 512]);  expand_10 = None
    bmm_4: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_55, view_56)
    view_57: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_4, [1, 4, 512, 512]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_4: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(view_57, 8.0);  view_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:324, code: attention_scores = attention_scores + attention_mask
    add_20: "f32[1, 4, 512, 512]" = torch.ops.aten.add.Tensor(div_4, mul);  div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_2: "f32[1, 4, 512, 1]" = torch.ops.aten.amax.default(add_20, [-1], True)
    sub_8: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_20, amax_2);  add_20 = amax_2 = None
    exp_2: "f32[1, 4, 512, 512]" = torch.ops.aten.exp.default(sub_8);  sub_8 = None
    sum_3: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_5: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    alias_2: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(div_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    native_dropout_7 = torch.ops.aten.native_dropout.default(div_5, 0.1, True);  div_5 = None
    getitem_24: "f32[1, 4, 512, 512]" = native_dropout_7[0]
    getitem_25: "b8[1, 4, 512, 512]" = native_dropout_7[1];  native_dropout_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_11: "f32[1, 4, 512, 512]" = torch.ops.aten.expand.default(getitem_24, [1, 4, 512, 512]);  getitem_24 = None
    view_58: "f32[4, 512, 512]" = torch.ops.aten.view.default(expand_11, [4, 512, 512]);  expand_11 = None
    expand_12: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_27, [1, 4, 512, 64]);  permute_27 = None
    view_59: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_12, [4, 512, 64]);  expand_12 = None
    bmm_5: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_58, view_59)
    view_60: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_5, [1, 4, 512, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_30: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_60, [0, 2, 1, 3]);  view_60 = None
    clone_2: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_61: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_2, [1, 512, 256]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_62: "f32[512, 256]" = torch.ops.aten.view.default(view_61, [512, 256]);  view_61 = None
    permute_31: "f32[256, 256]" = torch.ops.aten.permute.default(primals_46, [1, 0]);  primals_46 = None
    addmm_16: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_47, view_62, permute_31);  primals_47 = None
    view_63: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_16, [1, 512, 256]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    native_dropout_8 = torch.ops.aten.native_dropout.default(view_63, 0.1, True);  view_63 = None
    getitem_26: "f32[1, 512, 256]" = native_dropout_8[0]
    getitem_27: "b8[1, 512, 256]" = native_dropout_8[1];  native_dropout_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_21: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(getitem_26, add_19);  getitem_26 = add_19 = None
    var_mean_5 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 512, 1]" = var_mean_5[0]
    getitem_29: "f32[1, 512, 1]" = var_mean_5[1];  var_mean_5 = None
    add_22: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
    rsqrt_5: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
    sub_9: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_21, getitem_29)
    mul_17: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_5);  sub_9 = None
    mul_18: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_17, primals_48);  mul_17 = None
    add_23: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_18, primals_49);  mul_18 = primals_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_64: "f32[512, 256]" = torch.ops.aten.view.default(add_23, [512, 256])
    permute_32: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_50, [1, 0]);  primals_50 = None
    addmm_17: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_51, view_64, permute_32);  primals_51 = None
    view_65: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_17, [1, 512, 1024]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_19: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_65, 0.5)
    mul_20: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_65, 0.7071067811865476)
    erf_2: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_20);  mul_20 = None
    add_24: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_21: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_19, add_24);  mul_19 = add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_66: "f32[512, 1024]" = torch.ops.aten.view.default(mul_21, [512, 1024]);  mul_21 = None
    permute_33: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_52, [1, 0]);  primals_52 = None
    addmm_18: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_53, view_66, permute_33);  primals_53 = None
    view_67: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_18, [1, 512, 256]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    native_dropout_9 = torch.ops.aten.native_dropout.default(view_67, 0.1, True);  view_67 = None
    getitem_30: "f32[1, 512, 256]" = native_dropout_9[0]
    getitem_31: "b8[1, 512, 256]" = native_dropout_9[1];  native_dropout_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_25: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(getitem_30, add_23);  getitem_30 = add_23 = None
    var_mean_6 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
    getitem_32: "f32[1, 512, 1]" = var_mean_6[0]
    getitem_33: "f32[1, 512, 1]" = var_mean_6[1];  var_mean_6 = None
    add_26: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
    rsqrt_6: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
    sub_10: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_25, getitem_33)
    mul_22: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_6);  sub_10 = None
    mul_23: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_22, primals_54);  mul_22 = None
    add_27: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_23, primals_55);  mul_23 = primals_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_68: "f32[512, 256]" = torch.ops.aten.view.default(add_27, [512, 256])
    permute_34: "f32[256, 256]" = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
    addmm_19: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_57, view_68, permute_34);  primals_57 = None
    view_69: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_19, [1, 512, 256]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_70: "f32[512, 256]" = torch.ops.aten.view.default(add_27, [512, 256])
    permute_35: "f32[256, 256]" = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
    addmm_20: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_59, view_70, permute_35);  primals_59 = None
    view_71: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_20, [1, 512, 256]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_72: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_71, [1, 512, 4, 64]);  view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_36: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_73: "f32[512, 256]" = torch.ops.aten.view.default(add_27, [512, 256])
    permute_37: "f32[256, 256]" = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
    addmm_21: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_61, view_73, permute_37);  primals_61 = None
    view_74: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_21, [1, 512, 256]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_75: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_74, [1, 512, 4, 64]);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_38: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_76: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_69, [1, 512, 4, 64]);  view_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_39: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_40: "f32[1, 4, 64, 512]" = torch.ops.aten.permute.default(permute_36, [0, 1, 3, 2]);  permute_36 = None
    expand_13: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_39, [1, 4, 512, 64]);  permute_39 = None
    view_77: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_13, [4, 512, 64]);  expand_13 = None
    expand_14: "f32[1, 4, 64, 512]" = torch.ops.aten.expand.default(permute_40, [1, 4, 64, 512]);  permute_40 = None
    view_78: "f32[4, 64, 512]" = torch.ops.aten.view.default(expand_14, [4, 64, 512]);  expand_14 = None
    bmm_6: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_77, view_78)
    view_79: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_6, [1, 4, 512, 512]);  bmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_6: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(view_79, 8.0);  view_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:324, code: attention_scores = attention_scores + attention_mask
    add_28: "f32[1, 4, 512, 512]" = torch.ops.aten.add.Tensor(div_6, mul);  div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_3: "f32[1, 4, 512, 1]" = torch.ops.aten.amax.default(add_28, [-1], True)
    sub_11: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_28, amax_3);  add_28 = amax_3 = None
    exp_3: "f32[1, 4, 512, 512]" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
    sum_4: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_7: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    alias_3: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(div_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    native_dropout_10 = torch.ops.aten.native_dropout.default(div_7, 0.1, True);  div_7 = None
    getitem_34: "f32[1, 4, 512, 512]" = native_dropout_10[0]
    getitem_35: "b8[1, 4, 512, 512]" = native_dropout_10[1];  native_dropout_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_15: "f32[1, 4, 512, 512]" = torch.ops.aten.expand.default(getitem_34, [1, 4, 512, 512]);  getitem_34 = None
    view_80: "f32[4, 512, 512]" = torch.ops.aten.view.default(expand_15, [4, 512, 512]);  expand_15 = None
    expand_16: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_38, [1, 4, 512, 64]);  permute_38 = None
    view_81: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_16, [4, 512, 64]);  expand_16 = None
    bmm_7: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_80, view_81)
    view_82: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_7, [1, 4, 512, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_41: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
    clone_3: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_41, memory_format = torch.contiguous_format);  permute_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_83: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_3, [1, 512, 256]);  clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_84: "f32[512, 256]" = torch.ops.aten.view.default(view_83, [512, 256]);  view_83 = None
    permute_42: "f32[256, 256]" = torch.ops.aten.permute.default(primals_62, [1, 0]);  primals_62 = None
    addmm_22: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_63, view_84, permute_42);  primals_63 = None
    view_85: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_22, [1, 512, 256]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    native_dropout_11 = torch.ops.aten.native_dropout.default(view_85, 0.1, True);  view_85 = None
    getitem_36: "f32[1, 512, 256]" = native_dropout_11[0]
    getitem_37: "b8[1, 512, 256]" = native_dropout_11[1];  native_dropout_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_29: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(getitem_36, add_27);  getitem_36 = add_27 = None
    var_mean_7 = torch.ops.aten.var_mean.correction(add_29, [2], correction = 0, keepdim = True)
    getitem_38: "f32[1, 512, 1]" = var_mean_7[0]
    getitem_39: "f32[1, 512, 1]" = var_mean_7[1];  var_mean_7 = None
    add_30: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
    rsqrt_7: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
    sub_12: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_29, getitem_39)
    mul_24: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_7);  sub_12 = None
    mul_25: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_24, primals_64);  mul_24 = None
    add_31: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_25, primals_65);  mul_25 = primals_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_86: "f32[512, 256]" = torch.ops.aten.view.default(add_31, [512, 256])
    permute_43: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
    addmm_23: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_67, view_86, permute_43);  primals_67 = None
    view_87: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_23, [1, 512, 1024]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_26: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_87, 0.5)
    mul_27: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_87, 0.7071067811865476)
    erf_3: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_27);  mul_27 = None
    add_32: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_28: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_26, add_32);  mul_26 = add_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_88: "f32[512, 1024]" = torch.ops.aten.view.default(mul_28, [512, 1024]);  mul_28 = None
    permute_44: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_68, [1, 0]);  primals_68 = None
    addmm_24: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_69, view_88, permute_44);  primals_69 = None
    view_89: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_24, [1, 512, 256]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    native_dropout_12 = torch.ops.aten.native_dropout.default(view_89, 0.1, True);  view_89 = None
    getitem_40: "f32[1, 512, 256]" = native_dropout_12[0]
    getitem_41: "b8[1, 512, 256]" = native_dropout_12[1];  native_dropout_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_33: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(getitem_40, add_31);  getitem_40 = add_31 = None
    var_mean_8 = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
    getitem_42: "f32[1, 512, 1]" = var_mean_8[0]
    getitem_43: "f32[1, 512, 1]" = var_mean_8[1];  var_mean_8 = None
    add_34: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
    rsqrt_8: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
    sub_13: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_33, getitem_43)
    mul_29: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_8);  sub_13 = None
    mul_30: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_29, primals_70);  mul_29 = None
    add_35: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_30, primals_71);  mul_30 = primals_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_90: "f32[512, 256]" = torch.ops.aten.view.default(add_35, [512, 256])
    permute_45: "f32[256, 256]" = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
    addmm_25: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_73, view_90, permute_45);  primals_73 = None
    view_91: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_25, [1, 512, 256]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_92: "f32[512, 256]" = torch.ops.aten.view.default(add_35, [512, 256])
    permute_46: "f32[256, 256]" = torch.ops.aten.permute.default(primals_74, [1, 0]);  primals_74 = None
    addmm_26: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_75, view_92, permute_46);  primals_75 = None
    view_93: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_26, [1, 512, 256]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_94: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_93, [1, 512, 4, 64]);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_47: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_95: "f32[512, 256]" = torch.ops.aten.view.default(add_35, [512, 256])
    permute_48: "f32[256, 256]" = torch.ops.aten.permute.default(primals_76, [1, 0]);  primals_76 = None
    addmm_27: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_77, view_95, permute_48);  primals_77 = None
    view_96: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_27, [1, 512, 256]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_97: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_96, [1, 512, 4, 64]);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_49: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_97, [0, 2, 1, 3]);  view_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_98: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_91, [1, 512, 4, 64]);  view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_50: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_51: "f32[1, 4, 64, 512]" = torch.ops.aten.permute.default(permute_47, [0, 1, 3, 2]);  permute_47 = None
    expand_17: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_50, [1, 4, 512, 64]);  permute_50 = None
    view_99: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_17, [4, 512, 64]);  expand_17 = None
    expand_18: "f32[1, 4, 64, 512]" = torch.ops.aten.expand.default(permute_51, [1, 4, 64, 512]);  permute_51 = None
    view_100: "f32[4, 64, 512]" = torch.ops.aten.view.default(expand_18, [4, 64, 512]);  expand_18 = None
    bmm_8: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_99, view_100)
    view_101: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_8, [1, 4, 512, 512]);  bmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_8: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(view_101, 8.0);  view_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:324, code: attention_scores = attention_scores + attention_mask
    add_36: "f32[1, 4, 512, 512]" = torch.ops.aten.add.Tensor(div_8, mul);  div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_4: "f32[1, 4, 512, 1]" = torch.ops.aten.amax.default(add_36, [-1], True)
    sub_14: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_36, amax_4);  add_36 = amax_4 = None
    exp_4: "f32[1, 4, 512, 512]" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
    sum_5: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_9: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    alias_4: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(div_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    native_dropout_13 = torch.ops.aten.native_dropout.default(div_9, 0.1, True);  div_9 = None
    getitem_44: "f32[1, 4, 512, 512]" = native_dropout_13[0]
    getitem_45: "b8[1, 4, 512, 512]" = native_dropout_13[1];  native_dropout_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_19: "f32[1, 4, 512, 512]" = torch.ops.aten.expand.default(getitem_44, [1, 4, 512, 512]);  getitem_44 = None
    view_102: "f32[4, 512, 512]" = torch.ops.aten.view.default(expand_19, [4, 512, 512]);  expand_19 = None
    expand_20: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_49, [1, 4, 512, 64]);  permute_49 = None
    view_103: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_20, [4, 512, 64]);  expand_20 = None
    bmm_9: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_102, view_103)
    view_104: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_9, [1, 4, 512, 64]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_52: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None
    clone_4: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_105: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_4, [1, 512, 256]);  clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_106: "f32[512, 256]" = torch.ops.aten.view.default(view_105, [512, 256]);  view_105 = None
    permute_53: "f32[256, 256]" = torch.ops.aten.permute.default(primals_78, [1, 0]);  primals_78 = None
    addmm_28: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_79, view_106, permute_53);  primals_79 = None
    view_107: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_28, [1, 512, 256]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    native_dropout_14 = torch.ops.aten.native_dropout.default(view_107, 0.1, True);  view_107 = None
    getitem_46: "f32[1, 512, 256]" = native_dropout_14[0]
    getitem_47: "b8[1, 512, 256]" = native_dropout_14[1];  native_dropout_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_37: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(getitem_46, add_35);  getitem_46 = add_35 = None
    var_mean_9 = torch.ops.aten.var_mean.correction(add_37, [2], correction = 0, keepdim = True)
    getitem_48: "f32[1, 512, 1]" = var_mean_9[0]
    getitem_49: "f32[1, 512, 1]" = var_mean_9[1];  var_mean_9 = None
    add_38: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
    rsqrt_9: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
    sub_15: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_37, getitem_49)
    mul_31: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_9);  sub_15 = None
    mul_32: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_31, primals_80);  mul_31 = None
    add_39: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_32, primals_81);  mul_32 = primals_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_108: "f32[512, 256]" = torch.ops.aten.view.default(add_39, [512, 256])
    permute_54: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_82, [1, 0]);  primals_82 = None
    addmm_29: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_83, view_108, permute_54);  primals_83 = None
    view_109: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_29, [1, 512, 1024]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_33: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_109, 0.5)
    mul_34: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_109, 0.7071067811865476)
    erf_4: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_34);  mul_34 = None
    add_40: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_35: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_33, add_40);  mul_33 = add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_110: "f32[512, 1024]" = torch.ops.aten.view.default(mul_35, [512, 1024]);  mul_35 = None
    permute_55: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_84, [1, 0]);  primals_84 = None
    addmm_30: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_85, view_110, permute_55);  primals_85 = None
    view_111: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_30, [1, 512, 256]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    native_dropout_15 = torch.ops.aten.native_dropout.default(view_111, 0.1, True);  view_111 = None
    getitem_50: "f32[1, 512, 256]" = native_dropout_15[0]
    getitem_51: "b8[1, 512, 256]" = native_dropout_15[1];  native_dropout_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_41: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(getitem_50, add_39);  getitem_50 = add_39 = None
    var_mean_10 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
    getitem_52: "f32[1, 512, 1]" = var_mean_10[0]
    getitem_53: "f32[1, 512, 1]" = var_mean_10[1];  var_mean_10 = None
    add_42: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-12);  getitem_52 = None
    rsqrt_10: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_16: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_41, getitem_53)
    mul_36: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_10);  sub_16 = None
    mul_37: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_36, primals_86);  mul_36 = None
    add_43: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_37, primals_87);  mul_37 = primals_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_112: "f32[512, 256]" = torch.ops.aten.view.default(add_43, [512, 256])
    permute_56: "f32[256, 256]" = torch.ops.aten.permute.default(primals_88, [1, 0]);  primals_88 = None
    addmm_31: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_89, view_112, permute_56);  primals_89 = None
    view_113: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_31, [1, 512, 256]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_114: "f32[512, 256]" = torch.ops.aten.view.default(add_43, [512, 256])
    permute_57: "f32[256, 256]" = torch.ops.aten.permute.default(primals_90, [1, 0]);  primals_90 = None
    addmm_32: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_91, view_114, permute_57);  primals_91 = None
    view_115: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_32, [1, 512, 256]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_116: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_115, [1, 512, 4, 64]);  view_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_58: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_116, [0, 2, 1, 3]);  view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_117: "f32[512, 256]" = torch.ops.aten.view.default(add_43, [512, 256])
    permute_59: "f32[256, 256]" = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
    addmm_33: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_93, view_117, permute_59);  primals_93 = None
    view_118: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_33, [1, 512, 256]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_119: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_118, [1, 512, 4, 64]);  view_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_60: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_119, [0, 2, 1, 3]);  view_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_120: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_113, [1, 512, 4, 64]);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_61: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_62: "f32[1, 4, 64, 512]" = torch.ops.aten.permute.default(permute_58, [0, 1, 3, 2]);  permute_58 = None
    expand_21: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_61, [1, 4, 512, 64]);  permute_61 = None
    view_121: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_21, [4, 512, 64]);  expand_21 = None
    expand_22: "f32[1, 4, 64, 512]" = torch.ops.aten.expand.default(permute_62, [1, 4, 64, 512]);  permute_62 = None
    view_122: "f32[4, 64, 512]" = torch.ops.aten.view.default(expand_22, [4, 64, 512]);  expand_22 = None
    bmm_10: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_121, view_122)
    view_123: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_10, [1, 4, 512, 512]);  bmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_10: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(view_123, 8.0);  view_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:324, code: attention_scores = attention_scores + attention_mask
    add_44: "f32[1, 4, 512, 512]" = torch.ops.aten.add.Tensor(div_10, mul);  div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_5: "f32[1, 4, 512, 1]" = torch.ops.aten.amax.default(add_44, [-1], True)
    sub_17: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_44, amax_5);  add_44 = amax_5 = None
    exp_5: "f32[1, 4, 512, 512]" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
    sum_6: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_11: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    alias_5: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(div_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    native_dropout_16 = torch.ops.aten.native_dropout.default(div_11, 0.1, True);  div_11 = None
    getitem_54: "f32[1, 4, 512, 512]" = native_dropout_16[0]
    getitem_55: "b8[1, 4, 512, 512]" = native_dropout_16[1];  native_dropout_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_23: "f32[1, 4, 512, 512]" = torch.ops.aten.expand.default(getitem_54, [1, 4, 512, 512]);  getitem_54 = None
    view_124: "f32[4, 512, 512]" = torch.ops.aten.view.default(expand_23, [4, 512, 512]);  expand_23 = None
    expand_24: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_60, [1, 4, 512, 64]);  permute_60 = None
    view_125: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_24, [4, 512, 64]);  expand_24 = None
    bmm_11: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_124, view_125)
    view_126: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_11, [1, 4, 512, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_63: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
    clone_5: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format);  permute_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_127: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_5, [1, 512, 256]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_128: "f32[512, 256]" = torch.ops.aten.view.default(view_127, [512, 256]);  view_127 = None
    permute_64: "f32[256, 256]" = torch.ops.aten.permute.default(primals_94, [1, 0]);  primals_94 = None
    addmm_34: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_95, view_128, permute_64);  primals_95 = None
    view_129: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_34, [1, 512, 256]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    native_dropout_17 = torch.ops.aten.native_dropout.default(view_129, 0.1, True);  view_129 = None
    getitem_56: "f32[1, 512, 256]" = native_dropout_17[0]
    getitem_57: "b8[1, 512, 256]" = native_dropout_17[1];  native_dropout_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_45: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(getitem_56, add_43);  getitem_56 = add_43 = None
    var_mean_11 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
    getitem_58: "f32[1, 512, 1]" = var_mean_11[0]
    getitem_59: "f32[1, 512, 1]" = var_mean_11[1];  var_mean_11 = None
    add_46: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-12);  getitem_58 = None
    rsqrt_11: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
    sub_18: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_45, getitem_59)
    mul_38: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_11);  sub_18 = None
    mul_39: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_38, primals_96);  mul_38 = None
    add_47: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_39, primals_97);  mul_39 = primals_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_130: "f32[512, 256]" = torch.ops.aten.view.default(add_47, [512, 256])
    permute_65: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_98, [1, 0]);  primals_98 = None
    addmm_35: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_99, view_130, permute_65);  primals_99 = None
    view_131: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_35, [1, 512, 1024]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_40: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_131, 0.5)
    mul_41: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_131, 0.7071067811865476)
    erf_5: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_41);  mul_41 = None
    add_48: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_42: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_40, add_48);  mul_40 = add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_132: "f32[512, 1024]" = torch.ops.aten.view.default(mul_42, [512, 1024]);  mul_42 = None
    permute_66: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_100, [1, 0]);  primals_100 = None
    addmm_36: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_101, view_132, permute_66);  primals_101 = None
    view_133: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_36, [1, 512, 256]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    native_dropout_18 = torch.ops.aten.native_dropout.default(view_133, 0.1, True);  view_133 = None
    getitem_60: "f32[1, 512, 256]" = native_dropout_18[0]
    getitem_61: "b8[1, 512, 256]" = native_dropout_18[1];  native_dropout_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_49: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(getitem_60, add_47);  getitem_60 = add_47 = None
    var_mean_12 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
    getitem_62: "f32[1, 512, 1]" = var_mean_12[0]
    getitem_63: "f32[1, 512, 1]" = var_mean_12[1];  var_mean_12 = None
    add_50: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-12);  getitem_62 = None
    rsqrt_12: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
    sub_19: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_49, getitem_63)
    mul_43: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_12);  sub_19 = None
    mul_44: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_43, primals_102);  mul_43 = None
    add_51: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_44, primals_103);  mul_44 = primals_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_134: "f32[512, 256]" = torch.ops.aten.view.default(add_51, [512, 256])
    permute_67: "f32[256, 256]" = torch.ops.aten.permute.default(primals_104, [1, 0]);  primals_104 = None
    addmm_37: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_105, view_134, permute_67);  primals_105 = None
    view_135: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_37, [1, 512, 256]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_136: "f32[512, 256]" = torch.ops.aten.view.default(add_51, [512, 256])
    permute_68: "f32[256, 256]" = torch.ops.aten.permute.default(primals_106, [1, 0]);  primals_106 = None
    addmm_38: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_107, view_136, permute_68);  primals_107 = None
    view_137: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_38, [1, 512, 256]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_138: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_137, [1, 512, 4, 64]);  view_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_69: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_139: "f32[512, 256]" = torch.ops.aten.view.default(add_51, [512, 256])
    permute_70: "f32[256, 256]" = torch.ops.aten.permute.default(primals_108, [1, 0]);  primals_108 = None
    addmm_39: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_109, view_139, permute_70);  primals_109 = None
    view_140: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_39, [1, 512, 256]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_141: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_140, [1, 512, 4, 64]);  view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_71: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_141, [0, 2, 1, 3]);  view_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_142: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_135, [1, 512, 4, 64]);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_72: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_142, [0, 2, 1, 3]);  view_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_73: "f32[1, 4, 64, 512]" = torch.ops.aten.permute.default(permute_69, [0, 1, 3, 2]);  permute_69 = None
    expand_25: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_72, [1, 4, 512, 64]);  permute_72 = None
    view_143: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_25, [4, 512, 64]);  expand_25 = None
    expand_26: "f32[1, 4, 64, 512]" = torch.ops.aten.expand.default(permute_73, [1, 4, 64, 512]);  permute_73 = None
    view_144: "f32[4, 64, 512]" = torch.ops.aten.view.default(expand_26, [4, 64, 512]);  expand_26 = None
    bmm_12: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_143, view_144)
    view_145: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_12, [1, 4, 512, 512]);  bmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_12: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(view_145, 8.0);  view_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:324, code: attention_scores = attention_scores + attention_mask
    add_52: "f32[1, 4, 512, 512]" = torch.ops.aten.add.Tensor(div_12, mul);  div_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_6: "f32[1, 4, 512, 1]" = torch.ops.aten.amax.default(add_52, [-1], True)
    sub_20: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_52, amax_6);  add_52 = amax_6 = None
    exp_6: "f32[1, 4, 512, 512]" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
    sum_7: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_13: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    alias_6: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(div_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    native_dropout_19 = torch.ops.aten.native_dropout.default(div_13, 0.1, True);  div_13 = None
    getitem_64: "f32[1, 4, 512, 512]" = native_dropout_19[0]
    getitem_65: "b8[1, 4, 512, 512]" = native_dropout_19[1];  native_dropout_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_27: "f32[1, 4, 512, 512]" = torch.ops.aten.expand.default(getitem_64, [1, 4, 512, 512]);  getitem_64 = None
    view_146: "f32[4, 512, 512]" = torch.ops.aten.view.default(expand_27, [4, 512, 512]);  expand_27 = None
    expand_28: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_71, [1, 4, 512, 64]);  permute_71 = None
    view_147: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_28, [4, 512, 64]);  expand_28 = None
    bmm_13: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_146, view_147)
    view_148: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_13, [1, 4, 512, 64]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_74: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
    clone_6: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_74, memory_format = torch.contiguous_format);  permute_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_149: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_6, [1, 512, 256]);  clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_150: "f32[512, 256]" = torch.ops.aten.view.default(view_149, [512, 256]);  view_149 = None
    permute_75: "f32[256, 256]" = torch.ops.aten.permute.default(primals_110, [1, 0]);  primals_110 = None
    addmm_40: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_111, view_150, permute_75);  primals_111 = None
    view_151: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_40, [1, 512, 256]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    native_dropout_20 = torch.ops.aten.native_dropout.default(view_151, 0.1, True);  view_151 = None
    getitem_66: "f32[1, 512, 256]" = native_dropout_20[0]
    getitem_67: "b8[1, 512, 256]" = native_dropout_20[1];  native_dropout_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_53: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(getitem_66, add_51);  getitem_66 = add_51 = None
    var_mean_13 = torch.ops.aten.var_mean.correction(add_53, [2], correction = 0, keepdim = True)
    getitem_68: "f32[1, 512, 1]" = var_mean_13[0]
    getitem_69: "f32[1, 512, 1]" = var_mean_13[1];  var_mean_13 = None
    add_54: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-12);  getitem_68 = None
    rsqrt_13: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
    sub_21: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_53, getitem_69)
    mul_45: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_13);  sub_21 = None
    mul_46: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_45, primals_112);  mul_45 = None
    add_55: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_46, primals_113);  mul_46 = primals_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_152: "f32[512, 256]" = torch.ops.aten.view.default(add_55, [512, 256])
    permute_76: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_114, [1, 0]);  primals_114 = None
    addmm_41: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_115, view_152, permute_76);  primals_115 = None
    view_153: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_41, [1, 512, 1024]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_47: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_153, 0.5)
    mul_48: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_153, 0.7071067811865476)
    erf_6: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_48);  mul_48 = None
    add_56: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_49: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_47, add_56);  mul_47 = add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_154: "f32[512, 1024]" = torch.ops.aten.view.default(mul_49, [512, 1024]);  mul_49 = None
    permute_77: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_116, [1, 0]);  primals_116 = None
    addmm_42: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_117, view_154, permute_77);  primals_117 = None
    view_155: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_42, [1, 512, 256]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    native_dropout_21 = torch.ops.aten.native_dropout.default(view_155, 0.1, True);  view_155 = None
    getitem_70: "f32[1, 512, 256]" = native_dropout_21[0]
    getitem_71: "b8[1, 512, 256]" = native_dropout_21[1];  native_dropout_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_57: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(getitem_70, add_55);  getitem_70 = add_55 = None
    var_mean_14 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
    getitem_72: "f32[1, 512, 1]" = var_mean_14[0]
    getitem_73: "f32[1, 512, 1]" = var_mean_14[1];  var_mean_14 = None
    add_58: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-12);  getitem_72 = None
    rsqrt_14: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
    sub_22: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_57, getitem_73)
    mul_50: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_14);  sub_22 = None
    mul_51: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_50, primals_118);  mul_50 = None
    add_59: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_51, primals_119);  mul_51 = primals_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_156: "f32[512, 256]" = torch.ops.aten.view.default(add_59, [512, 256])
    permute_78: "f32[256, 256]" = torch.ops.aten.permute.default(primals_120, [1, 0]);  primals_120 = None
    addmm_43: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_121, view_156, permute_78);  primals_121 = None
    view_157: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_43, [1, 512, 256]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_158: "f32[512, 256]" = torch.ops.aten.view.default(add_59, [512, 256])
    permute_79: "f32[256, 256]" = torch.ops.aten.permute.default(primals_122, [1, 0]);  primals_122 = None
    addmm_44: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_123, view_158, permute_79);  primals_123 = None
    view_159: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_44, [1, 512, 256]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_160: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_159, [1, 512, 4, 64]);  view_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_80: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_160, [0, 2, 1, 3]);  view_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_161: "f32[512, 256]" = torch.ops.aten.view.default(add_59, [512, 256])
    permute_81: "f32[256, 256]" = torch.ops.aten.permute.default(primals_124, [1, 0]);  primals_124 = None
    addmm_45: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_125, view_161, permute_81);  primals_125 = None
    view_162: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_45, [1, 512, 256]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_163: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_162, [1, 512, 4, 64]);  view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_82: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_163, [0, 2, 1, 3]);  view_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_164: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_157, [1, 512, 4, 64]);  view_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_83: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_164, [0, 2, 1, 3]);  view_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_84: "f32[1, 4, 64, 512]" = torch.ops.aten.permute.default(permute_80, [0, 1, 3, 2]);  permute_80 = None
    expand_29: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_83, [1, 4, 512, 64]);  permute_83 = None
    view_165: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_29, [4, 512, 64]);  expand_29 = None
    expand_30: "f32[1, 4, 64, 512]" = torch.ops.aten.expand.default(permute_84, [1, 4, 64, 512]);  permute_84 = None
    view_166: "f32[4, 64, 512]" = torch.ops.aten.view.default(expand_30, [4, 64, 512]);  expand_30 = None
    bmm_14: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_165, view_166)
    view_167: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_14, [1, 4, 512, 512]);  bmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_14: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(view_167, 8.0);  view_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:324, code: attention_scores = attention_scores + attention_mask
    add_60: "f32[1, 4, 512, 512]" = torch.ops.aten.add.Tensor(div_14, mul);  div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_7: "f32[1, 4, 512, 1]" = torch.ops.aten.amax.default(add_60, [-1], True)
    sub_23: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_60, amax_7);  add_60 = amax_7 = None
    exp_7: "f32[1, 4, 512, 512]" = torch.ops.aten.exp.default(sub_23);  sub_23 = None
    sum_8: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_15: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    alias_7: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(div_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    native_dropout_22 = torch.ops.aten.native_dropout.default(div_15, 0.1, True);  div_15 = None
    getitem_74: "f32[1, 4, 512, 512]" = native_dropout_22[0]
    getitem_75: "b8[1, 4, 512, 512]" = native_dropout_22[1];  native_dropout_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_31: "f32[1, 4, 512, 512]" = torch.ops.aten.expand.default(getitem_74, [1, 4, 512, 512]);  getitem_74 = None
    view_168: "f32[4, 512, 512]" = torch.ops.aten.view.default(expand_31, [4, 512, 512]);  expand_31 = None
    expand_32: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_82, [1, 4, 512, 64]);  permute_82 = None
    view_169: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_32, [4, 512, 64]);  expand_32 = None
    bmm_15: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_168, view_169)
    view_170: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_15, [1, 4, 512, 64]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_85: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_170, [0, 2, 1, 3]);  view_170 = None
    clone_7: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_85, memory_format = torch.contiguous_format);  permute_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_171: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_7, [1, 512, 256]);  clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_172: "f32[512, 256]" = torch.ops.aten.view.default(view_171, [512, 256]);  view_171 = None
    permute_86: "f32[256, 256]" = torch.ops.aten.permute.default(primals_126, [1, 0]);  primals_126 = None
    addmm_46: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_127, view_172, permute_86);  primals_127 = None
    view_173: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_46, [1, 512, 256]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    native_dropout_23 = torch.ops.aten.native_dropout.default(view_173, 0.1, True);  view_173 = None
    getitem_76: "f32[1, 512, 256]" = native_dropout_23[0]
    getitem_77: "b8[1, 512, 256]" = native_dropout_23[1];  native_dropout_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_61: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(getitem_76, add_59);  getitem_76 = add_59 = None
    var_mean_15 = torch.ops.aten.var_mean.correction(add_61, [2], correction = 0, keepdim = True)
    getitem_78: "f32[1, 512, 1]" = var_mean_15[0]
    getitem_79: "f32[1, 512, 1]" = var_mean_15[1];  var_mean_15 = None
    add_62: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-12);  getitem_78 = None
    rsqrt_15: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
    sub_24: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_61, getitem_79)
    mul_52: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_15);  sub_24 = None
    mul_53: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_52, primals_128);  mul_52 = None
    add_63: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_53, primals_129);  mul_53 = primals_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_174: "f32[512, 256]" = torch.ops.aten.view.default(add_63, [512, 256])
    permute_87: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_130, [1, 0]);  primals_130 = None
    addmm_47: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_131, view_174, permute_87);  primals_131 = None
    view_175: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_47, [1, 512, 1024]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_54: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_175, 0.5)
    mul_55: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_175, 0.7071067811865476)
    erf_7: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
    add_64: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_56: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_54, add_64);  mul_54 = add_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_176: "f32[512, 1024]" = torch.ops.aten.view.default(mul_56, [512, 1024]);  mul_56 = None
    permute_88: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_132, [1, 0]);  primals_132 = None
    addmm_48: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_133, view_176, permute_88);  primals_133 = None
    view_177: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_48, [1, 512, 256]);  addmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    native_dropout_24 = torch.ops.aten.native_dropout.default(view_177, 0.1, True);  view_177 = None
    getitem_80: "f32[1, 512, 256]" = native_dropout_24[0]
    getitem_81: "b8[1, 512, 256]" = native_dropout_24[1];  native_dropout_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_65: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(getitem_80, add_63);  getitem_80 = add_63 = None
    var_mean_16 = torch.ops.aten.var_mean.correction(add_65, [2], correction = 0, keepdim = True)
    getitem_82: "f32[1, 512, 1]" = var_mean_16[0]
    getitem_83: "f32[1, 512, 1]" = var_mean_16[1];  var_mean_16 = None
    add_66: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-12);  getitem_82 = None
    rsqrt_16: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    sub_25: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_65, getitem_83)
    mul_57: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_16);  sub_25 = None
    mul_58: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_57, primals_134);  mul_57 = None
    add_67: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_58, primals_135);  mul_58 = primals_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_178: "f32[512, 256]" = torch.ops.aten.view.default(add_67, [512, 256])
    permute_89: "f32[256, 256]" = torch.ops.aten.permute.default(primals_136, [1, 0]);  primals_136 = None
    addmm_49: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_137, view_178, permute_89);  primals_137 = None
    view_179: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_49, [1, 512, 256]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_180: "f32[512, 256]" = torch.ops.aten.view.default(add_67, [512, 256])
    permute_90: "f32[256, 256]" = torch.ops.aten.permute.default(primals_138, [1, 0]);  primals_138 = None
    addmm_50: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_139, view_180, permute_90);  primals_139 = None
    view_181: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_50, [1, 512, 256]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_182: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_181, [1, 512, 4, 64]);  view_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_91: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_183: "f32[512, 256]" = torch.ops.aten.view.default(add_67, [512, 256])
    permute_92: "f32[256, 256]" = torch.ops.aten.permute.default(primals_140, [1, 0]);  primals_140 = None
    addmm_51: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_141, view_183, permute_92);  primals_141 = None
    view_184: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_51, [1, 512, 256]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_185: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_184, [1, 512, 4, 64]);  view_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_93: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_186: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_179, [1, 512, 4, 64]);  view_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_94: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_95: "f32[1, 4, 64, 512]" = torch.ops.aten.permute.default(permute_91, [0, 1, 3, 2]);  permute_91 = None
    expand_33: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_94, [1, 4, 512, 64]);  permute_94 = None
    view_187: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_33, [4, 512, 64]);  expand_33 = None
    expand_34: "f32[1, 4, 64, 512]" = torch.ops.aten.expand.default(permute_95, [1, 4, 64, 512]);  permute_95 = None
    view_188: "f32[4, 64, 512]" = torch.ops.aten.view.default(expand_34, [4, 64, 512]);  expand_34 = None
    bmm_16: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_187, view_188)
    view_189: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_16, [1, 4, 512, 512]);  bmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_16: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(view_189, 8.0);  view_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:324, code: attention_scores = attention_scores + attention_mask
    add_68: "f32[1, 4, 512, 512]" = torch.ops.aten.add.Tensor(div_16, mul);  div_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_8: "f32[1, 4, 512, 1]" = torch.ops.aten.amax.default(add_68, [-1], True)
    sub_26: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_68, amax_8);  add_68 = amax_8 = None
    exp_8: "f32[1, 4, 512, 512]" = torch.ops.aten.exp.default(sub_26);  sub_26 = None
    sum_9: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_17: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    alias_8: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(div_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    native_dropout_25 = torch.ops.aten.native_dropout.default(div_17, 0.1, True);  div_17 = None
    getitem_84: "f32[1, 4, 512, 512]" = native_dropout_25[0]
    getitem_85: "b8[1, 4, 512, 512]" = native_dropout_25[1];  native_dropout_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_35: "f32[1, 4, 512, 512]" = torch.ops.aten.expand.default(getitem_84, [1, 4, 512, 512]);  getitem_84 = None
    view_190: "f32[4, 512, 512]" = torch.ops.aten.view.default(expand_35, [4, 512, 512]);  expand_35 = None
    expand_36: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_93, [1, 4, 512, 64]);  permute_93 = None
    view_191: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_36, [4, 512, 64]);  expand_36 = None
    bmm_17: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_190, view_191)
    view_192: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_17, [1, 4, 512, 64]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_96: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None
    clone_8: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_96, memory_format = torch.contiguous_format);  permute_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_193: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_8, [1, 512, 256]);  clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_194: "f32[512, 256]" = torch.ops.aten.view.default(view_193, [512, 256]);  view_193 = None
    permute_97: "f32[256, 256]" = torch.ops.aten.permute.default(primals_142, [1, 0]);  primals_142 = None
    addmm_52: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_143, view_194, permute_97);  primals_143 = None
    view_195: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_52, [1, 512, 256]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    native_dropout_26 = torch.ops.aten.native_dropout.default(view_195, 0.1, True);  view_195 = None
    getitem_86: "f32[1, 512, 256]" = native_dropout_26[0]
    getitem_87: "b8[1, 512, 256]" = native_dropout_26[1];  native_dropout_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_69: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(getitem_86, add_67);  getitem_86 = add_67 = None
    var_mean_17 = torch.ops.aten.var_mean.correction(add_69, [2], correction = 0, keepdim = True)
    getitem_88: "f32[1, 512, 1]" = var_mean_17[0]
    getitem_89: "f32[1, 512, 1]" = var_mean_17[1];  var_mean_17 = None
    add_70: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-12);  getitem_88 = None
    rsqrt_17: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
    sub_27: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_69, getitem_89)
    mul_59: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_17);  sub_27 = None
    mul_60: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_59, primals_144);  mul_59 = None
    add_71: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_60, primals_145);  mul_60 = primals_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_196: "f32[512, 256]" = torch.ops.aten.view.default(add_71, [512, 256])
    permute_98: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_146, [1, 0]);  primals_146 = None
    addmm_53: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_147, view_196, permute_98);  primals_147 = None
    view_197: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_53, [1, 512, 1024]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_61: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_197, 0.5)
    mul_62: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_197, 0.7071067811865476)
    erf_8: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_62);  mul_62 = None
    add_72: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_63: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_61, add_72);  mul_61 = add_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_198: "f32[512, 1024]" = torch.ops.aten.view.default(mul_63, [512, 1024]);  mul_63 = None
    permute_99: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_148, [1, 0]);  primals_148 = None
    addmm_54: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_149, view_198, permute_99);  primals_149 = None
    view_199: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_54, [1, 512, 256]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    native_dropout_27 = torch.ops.aten.native_dropout.default(view_199, 0.1, True);  view_199 = None
    getitem_90: "f32[1, 512, 256]" = native_dropout_27[0]
    getitem_91: "b8[1, 512, 256]" = native_dropout_27[1];  native_dropout_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_73: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(getitem_90, add_71);  getitem_90 = add_71 = None
    var_mean_18 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
    getitem_92: "f32[1, 512, 1]" = var_mean_18[0]
    getitem_93: "f32[1, 512, 1]" = var_mean_18[1];  var_mean_18 = None
    add_74: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-12);  getitem_92 = None
    rsqrt_18: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    sub_28: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_73, getitem_93)
    mul_64: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_18);  sub_28 = None
    mul_65: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_64, primals_150);  mul_64 = None
    add_75: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_65, primals_151);  mul_65 = primals_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_200: "f32[512, 256]" = torch.ops.aten.view.default(add_75, [512, 256])
    permute_100: "f32[256, 256]" = torch.ops.aten.permute.default(primals_152, [1, 0]);  primals_152 = None
    addmm_55: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_153, view_200, permute_100);  primals_153 = None
    view_201: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_55, [1, 512, 256]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_202: "f32[512, 256]" = torch.ops.aten.view.default(add_75, [512, 256])
    permute_101: "f32[256, 256]" = torch.ops.aten.permute.default(primals_154, [1, 0]);  primals_154 = None
    addmm_56: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_155, view_202, permute_101);  primals_155 = None
    view_203: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_56, [1, 512, 256]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_204: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_203, [1, 512, 4, 64]);  view_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_102: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_204, [0, 2, 1, 3]);  view_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_205: "f32[512, 256]" = torch.ops.aten.view.default(add_75, [512, 256])
    permute_103: "f32[256, 256]" = torch.ops.aten.permute.default(primals_156, [1, 0]);  primals_156 = None
    addmm_57: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_157, view_205, permute_103);  primals_157 = None
    view_206: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_57, [1, 512, 256]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_207: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_206, [1, 512, 4, 64]);  view_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_104: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_207, [0, 2, 1, 3]);  view_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_208: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_201, [1, 512, 4, 64]);  view_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_105: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_106: "f32[1, 4, 64, 512]" = torch.ops.aten.permute.default(permute_102, [0, 1, 3, 2]);  permute_102 = None
    expand_37: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_105, [1, 4, 512, 64]);  permute_105 = None
    view_209: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_37, [4, 512, 64]);  expand_37 = None
    expand_38: "f32[1, 4, 64, 512]" = torch.ops.aten.expand.default(permute_106, [1, 4, 64, 512]);  permute_106 = None
    view_210: "f32[4, 64, 512]" = torch.ops.aten.view.default(expand_38, [4, 64, 512]);  expand_38 = None
    bmm_18: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_209, view_210)
    view_211: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_18, [1, 4, 512, 512]);  bmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_18: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(view_211, 8.0);  view_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:324, code: attention_scores = attention_scores + attention_mask
    add_76: "f32[1, 4, 512, 512]" = torch.ops.aten.add.Tensor(div_18, mul);  div_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_9: "f32[1, 4, 512, 1]" = torch.ops.aten.amax.default(add_76, [-1], True)
    sub_29: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_76, amax_9);  add_76 = amax_9 = None
    exp_9: "f32[1, 4, 512, 512]" = torch.ops.aten.exp.default(sub_29);  sub_29 = None
    sum_10: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_19: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    alias_9: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(div_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    native_dropout_28 = torch.ops.aten.native_dropout.default(div_19, 0.1, True);  div_19 = None
    getitem_94: "f32[1, 4, 512, 512]" = native_dropout_28[0]
    getitem_95: "b8[1, 4, 512, 512]" = native_dropout_28[1];  native_dropout_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_39: "f32[1, 4, 512, 512]" = torch.ops.aten.expand.default(getitem_94, [1, 4, 512, 512]);  getitem_94 = None
    view_212: "f32[4, 512, 512]" = torch.ops.aten.view.default(expand_39, [4, 512, 512]);  expand_39 = None
    expand_40: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_104, [1, 4, 512, 64]);  permute_104 = None
    view_213: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_40, [4, 512, 64]);  expand_40 = None
    bmm_19: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_212, view_213)
    view_214: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_19, [1, 4, 512, 64]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_107: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_214, [0, 2, 1, 3]);  view_214 = None
    clone_9: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_107, memory_format = torch.contiguous_format);  permute_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_215: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_9, [1, 512, 256]);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_216: "f32[512, 256]" = torch.ops.aten.view.default(view_215, [512, 256]);  view_215 = None
    permute_108: "f32[256, 256]" = torch.ops.aten.permute.default(primals_158, [1, 0]);  primals_158 = None
    addmm_58: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_159, view_216, permute_108);  primals_159 = None
    view_217: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_58, [1, 512, 256]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    native_dropout_29 = torch.ops.aten.native_dropout.default(view_217, 0.1, True);  view_217 = None
    getitem_96: "f32[1, 512, 256]" = native_dropout_29[0]
    getitem_97: "b8[1, 512, 256]" = native_dropout_29[1];  native_dropout_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_77: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(getitem_96, add_75);  getitem_96 = add_75 = None
    var_mean_19 = torch.ops.aten.var_mean.correction(add_77, [2], correction = 0, keepdim = True)
    getitem_98: "f32[1, 512, 1]" = var_mean_19[0]
    getitem_99: "f32[1, 512, 1]" = var_mean_19[1];  var_mean_19 = None
    add_78: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-12);  getitem_98 = None
    rsqrt_19: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    sub_30: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_77, getitem_99)
    mul_66: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_19);  sub_30 = None
    mul_67: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_66, primals_160);  mul_66 = None
    add_79: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_67, primals_161);  mul_67 = primals_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_218: "f32[512, 256]" = torch.ops.aten.view.default(add_79, [512, 256])
    permute_109: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_162, [1, 0]);  primals_162 = None
    addmm_59: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_163, view_218, permute_109);  primals_163 = None
    view_219: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_59, [1, 512, 1024]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_68: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_219, 0.5)
    mul_69: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_219, 0.7071067811865476)
    erf_9: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
    add_80: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_70: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_68, add_80);  mul_68 = add_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_220: "f32[512, 1024]" = torch.ops.aten.view.default(mul_70, [512, 1024]);  mul_70 = None
    permute_110: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_164, [1, 0]);  primals_164 = None
    addmm_60: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_165, view_220, permute_110);  primals_165 = None
    view_221: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_60, [1, 512, 256]);  addmm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    native_dropout_30 = torch.ops.aten.native_dropout.default(view_221, 0.1, True);  view_221 = None
    getitem_100: "f32[1, 512, 256]" = native_dropout_30[0]
    getitem_101: "b8[1, 512, 256]" = native_dropout_30[1];  native_dropout_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_81: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(getitem_100, add_79);  getitem_100 = add_79 = None
    var_mean_20 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
    getitem_102: "f32[1, 512, 1]" = var_mean_20[0]
    getitem_103: "f32[1, 512, 1]" = var_mean_20[1];  var_mean_20 = None
    add_82: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-12);  getitem_102 = None
    rsqrt_20: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
    sub_31: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_81, getitem_103)
    mul_71: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_20);  sub_31 = None
    mul_72: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_71, primals_166);  mul_71 = None
    add_83: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_72, primals_167);  mul_72 = primals_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_222: "f32[512, 256]" = torch.ops.aten.view.default(add_83, [512, 256])
    permute_111: "f32[256, 256]" = torch.ops.aten.permute.default(primals_168, [1, 0]);  primals_168 = None
    addmm_61: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_169, view_222, permute_111);  primals_169 = None
    view_223: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_61, [1, 512, 256]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_224: "f32[512, 256]" = torch.ops.aten.view.default(add_83, [512, 256])
    permute_112: "f32[256, 256]" = torch.ops.aten.permute.default(primals_170, [1, 0]);  primals_170 = None
    addmm_62: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_171, view_224, permute_112);  primals_171 = None
    view_225: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_62, [1, 512, 256]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_226: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_225, [1, 512, 4, 64]);  view_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_113: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_227: "f32[512, 256]" = torch.ops.aten.view.default(add_83, [512, 256])
    permute_114: "f32[256, 256]" = torch.ops.aten.permute.default(primals_172, [1, 0]);  primals_172 = None
    addmm_63: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_173, view_227, permute_114);  primals_173 = None
    view_228: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_63, [1, 512, 256]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_229: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_228, [1, 512, 4, 64]);  view_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_115: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_229, [0, 2, 1, 3]);  view_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_230: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_223, [1, 512, 4, 64]);  view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_116: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_230, [0, 2, 1, 3]);  view_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_117: "f32[1, 4, 64, 512]" = torch.ops.aten.permute.default(permute_113, [0, 1, 3, 2]);  permute_113 = None
    expand_41: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_116, [1, 4, 512, 64]);  permute_116 = None
    view_231: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_41, [4, 512, 64]);  expand_41 = None
    expand_42: "f32[1, 4, 64, 512]" = torch.ops.aten.expand.default(permute_117, [1, 4, 64, 512]);  permute_117 = None
    view_232: "f32[4, 64, 512]" = torch.ops.aten.view.default(expand_42, [4, 64, 512]);  expand_42 = None
    bmm_20: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_231, view_232)
    view_233: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_20, [1, 4, 512, 512]);  bmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_20: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(view_233, 8.0);  view_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:324, code: attention_scores = attention_scores + attention_mask
    add_84: "f32[1, 4, 512, 512]" = torch.ops.aten.add.Tensor(div_20, mul);  div_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_10: "f32[1, 4, 512, 1]" = torch.ops.aten.amax.default(add_84, [-1], True)
    sub_32: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_84, amax_10);  add_84 = amax_10 = None
    exp_10: "f32[1, 4, 512, 512]" = torch.ops.aten.exp.default(sub_32);  sub_32 = None
    sum_11: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_21: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    alias_10: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(div_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    native_dropout_31 = torch.ops.aten.native_dropout.default(div_21, 0.1, True);  div_21 = None
    getitem_104: "f32[1, 4, 512, 512]" = native_dropout_31[0]
    getitem_105: "b8[1, 4, 512, 512]" = native_dropout_31[1];  native_dropout_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_43: "f32[1, 4, 512, 512]" = torch.ops.aten.expand.default(getitem_104, [1, 4, 512, 512]);  getitem_104 = None
    view_234: "f32[4, 512, 512]" = torch.ops.aten.view.default(expand_43, [4, 512, 512]);  expand_43 = None
    expand_44: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_115, [1, 4, 512, 64]);  permute_115 = None
    view_235: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_44, [4, 512, 64]);  expand_44 = None
    bmm_21: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_234, view_235)
    view_236: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_21, [1, 4, 512, 64]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_118: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_236, [0, 2, 1, 3]);  view_236 = None
    clone_10: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_118, memory_format = torch.contiguous_format);  permute_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_237: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_10, [1, 512, 256]);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_238: "f32[512, 256]" = torch.ops.aten.view.default(view_237, [512, 256]);  view_237 = None
    permute_119: "f32[256, 256]" = torch.ops.aten.permute.default(primals_174, [1, 0]);  primals_174 = None
    addmm_64: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_175, view_238, permute_119);  primals_175 = None
    view_239: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_64, [1, 512, 256]);  addmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    native_dropout_32 = torch.ops.aten.native_dropout.default(view_239, 0.1, True);  view_239 = None
    getitem_106: "f32[1, 512, 256]" = native_dropout_32[0]
    getitem_107: "b8[1, 512, 256]" = native_dropout_32[1];  native_dropout_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_85: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(getitem_106, add_83);  getitem_106 = add_83 = None
    var_mean_21 = torch.ops.aten.var_mean.correction(add_85, [2], correction = 0, keepdim = True)
    getitem_108: "f32[1, 512, 1]" = var_mean_21[0]
    getitem_109: "f32[1, 512, 1]" = var_mean_21[1];  var_mean_21 = None
    add_86: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-12);  getitem_108 = None
    rsqrt_21: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
    sub_33: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_85, getitem_109)
    mul_73: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_21);  sub_33 = None
    mul_74: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_73, primals_176);  mul_73 = None
    add_87: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_74, primals_177);  mul_74 = primals_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_240: "f32[512, 256]" = torch.ops.aten.view.default(add_87, [512, 256])
    permute_120: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_178, [1, 0]);  primals_178 = None
    addmm_65: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_179, view_240, permute_120);  primals_179 = None
    view_241: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_65, [1, 512, 1024]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_75: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_241, 0.5)
    mul_76: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_241, 0.7071067811865476)
    erf_10: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_76);  mul_76 = None
    add_88: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_77: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_75, add_88);  mul_75 = add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_242: "f32[512, 1024]" = torch.ops.aten.view.default(mul_77, [512, 1024]);  mul_77 = None
    permute_121: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_180, [1, 0]);  primals_180 = None
    addmm_66: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_181, view_242, permute_121);  primals_181 = None
    view_243: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_66, [1, 512, 256]);  addmm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    native_dropout_33 = torch.ops.aten.native_dropout.default(view_243, 0.1, True);  view_243 = None
    getitem_110: "f32[1, 512, 256]" = native_dropout_33[0]
    getitem_111: "b8[1, 512, 256]" = native_dropout_33[1];  native_dropout_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_89: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(getitem_110, add_87);  getitem_110 = add_87 = None
    var_mean_22 = torch.ops.aten.var_mean.correction(add_89, [2], correction = 0, keepdim = True)
    getitem_112: "f32[1, 512, 1]" = var_mean_22[0]
    getitem_113: "f32[1, 512, 1]" = var_mean_22[1];  var_mean_22 = None
    add_90: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-12);  getitem_112 = None
    rsqrt_22: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
    sub_34: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_89, getitem_113)
    mul_78: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_22);  sub_34 = None
    mul_79: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_78, primals_182);  mul_78 = None
    add_91: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_79, primals_183);  mul_79 = primals_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_244: "f32[512, 256]" = torch.ops.aten.view.default(add_91, [512, 256])
    permute_122: "f32[256, 256]" = torch.ops.aten.permute.default(primals_184, [1, 0]);  primals_184 = None
    addmm_67: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_185, view_244, permute_122);  primals_185 = None
    view_245: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_67, [1, 512, 256]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_246: "f32[512, 256]" = torch.ops.aten.view.default(add_91, [512, 256])
    permute_123: "f32[256, 256]" = torch.ops.aten.permute.default(primals_186, [1, 0]);  primals_186 = None
    addmm_68: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_187, view_246, permute_123);  primals_187 = None
    view_247: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_68, [1, 512, 256]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_248: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_247, [1, 512, 4, 64]);  view_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_124: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_248, [0, 2, 1, 3]);  view_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_249: "f32[512, 256]" = torch.ops.aten.view.default(add_91, [512, 256])
    permute_125: "f32[256, 256]" = torch.ops.aten.permute.default(primals_188, [1, 0]);  primals_188 = None
    addmm_69: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_189, view_249, permute_125);  primals_189 = None
    view_250: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_69, [1, 512, 256]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_251: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_250, [1, 512, 4, 64]);  view_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_126: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_251, [0, 2, 1, 3]);  view_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_252: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_245, [1, 512, 4, 64]);  view_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_127: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_128: "f32[1, 4, 64, 512]" = torch.ops.aten.permute.default(permute_124, [0, 1, 3, 2]);  permute_124 = None
    expand_45: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_127, [1, 4, 512, 64]);  permute_127 = None
    view_253: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_45, [4, 512, 64]);  expand_45 = None
    expand_46: "f32[1, 4, 64, 512]" = torch.ops.aten.expand.default(permute_128, [1, 4, 64, 512]);  permute_128 = None
    view_254: "f32[4, 64, 512]" = torch.ops.aten.view.default(expand_46, [4, 64, 512]);  expand_46 = None
    bmm_22: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_253, view_254)
    view_255: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_22, [1, 4, 512, 512]);  bmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_22: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(view_255, 8.0);  view_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:324, code: attention_scores = attention_scores + attention_mask
    add_92: "f32[1, 4, 512, 512]" = torch.ops.aten.add.Tensor(div_22, mul);  div_22 = mul = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_11: "f32[1, 4, 512, 1]" = torch.ops.aten.amax.default(add_92, [-1], True)
    sub_35: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(add_92, amax_11);  add_92 = amax_11 = None
    exp_11: "f32[1, 4, 512, 512]" = torch.ops.aten.exp.default(sub_35);  sub_35 = None
    sum_12: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_23: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    alias_11: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(div_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    native_dropout_34 = torch.ops.aten.native_dropout.default(div_23, 0.1, True);  div_23 = None
    getitem_114: "f32[1, 4, 512, 512]" = native_dropout_34[0]
    getitem_115: "b8[1, 4, 512, 512]" = native_dropout_34[1];  native_dropout_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_47: "f32[1, 4, 512, 512]" = torch.ops.aten.expand.default(getitem_114, [1, 4, 512, 512]);  getitem_114 = None
    view_256: "f32[4, 512, 512]" = torch.ops.aten.view.default(expand_47, [4, 512, 512]);  expand_47 = None
    expand_48: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(permute_126, [1, 4, 512, 64]);  permute_126 = None
    view_257: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_48, [4, 512, 64]);  expand_48 = None
    bmm_23: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_256, view_257)
    view_258: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_23, [1, 4, 512, 64]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_129: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_258, [0, 2, 1, 3]);  view_258 = None
    clone_11: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_259: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_11, [1, 512, 256]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_260: "f32[512, 256]" = torch.ops.aten.view.default(view_259, [512, 256]);  view_259 = None
    permute_130: "f32[256, 256]" = torch.ops.aten.permute.default(primals_190, [1, 0]);  primals_190 = None
    addmm_70: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_191, view_260, permute_130);  primals_191 = None
    view_261: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_70, [1, 512, 256]);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    native_dropout_35 = torch.ops.aten.native_dropout.default(view_261, 0.1, True);  view_261 = None
    getitem_116: "f32[1, 512, 256]" = native_dropout_35[0]
    getitem_117: "b8[1, 512, 256]" = native_dropout_35[1];  native_dropout_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_93: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(getitem_116, add_91);  getitem_116 = add_91 = None
    var_mean_23 = torch.ops.aten.var_mean.correction(add_93, [2], correction = 0, keepdim = True)
    getitem_118: "f32[1, 512, 1]" = var_mean_23[0]
    getitem_119: "f32[1, 512, 1]" = var_mean_23[1];  var_mean_23 = None
    add_94: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-12);  getitem_118 = None
    rsqrt_23: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
    sub_36: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_93, getitem_119)
    mul_80: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_23);  sub_36 = None
    mul_81: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_80, primals_192);  mul_80 = None
    add_95: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_81, primals_193);  mul_81 = primals_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_262: "f32[512, 256]" = torch.ops.aten.view.default(add_95, [512, 256])
    permute_131: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_194, [1, 0]);  primals_194 = None
    addmm_71: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_195, view_262, permute_131);  primals_195 = None
    view_263: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_71, [1, 512, 1024]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_82: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_263, 0.5)
    mul_83: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_263, 0.7071067811865476)
    erf_11: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
    add_96: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_84: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_82, add_96);  mul_82 = add_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_264: "f32[512, 1024]" = torch.ops.aten.view.default(mul_84, [512, 1024]);  mul_84 = None
    permute_132: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_196, [1, 0]);  primals_196 = None
    addmm_72: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_197, view_264, permute_132);  primals_197 = None
    view_265: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_72, [1, 512, 256]);  addmm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    native_dropout_36 = torch.ops.aten.native_dropout.default(view_265, 0.1, True);  view_265 = None
    getitem_120: "f32[1, 512, 256]" = native_dropout_36[0]
    getitem_121: "b8[1, 512, 256]" = native_dropout_36[1];  native_dropout_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_97: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(getitem_120, add_95);  getitem_120 = add_95 = None
    var_mean_24 = torch.ops.aten.var_mean.correction(add_97, [2], correction = 0, keepdim = True)
    getitem_122: "f32[1, 512, 1]" = var_mean_24[0]
    getitem_123: "f32[1, 512, 1]" = var_mean_24[1];  var_mean_24 = None
    add_98: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_122, 1e-12);  getitem_122 = None
    rsqrt_24: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
    sub_37: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_97, getitem_123)
    mul_85: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_24);  sub_37 = None
    mul_86: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_85, primals_198);  mul_85 = None
    add_99: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_86, primals_199);  mul_86 = primals_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:660, code: hidden_states = self.dense(generator_hidden_states)
    view_266: "f32[512, 256]" = torch.ops.aten.view.default(add_99, [512, 256]);  add_99 = None
    permute_133: "f32[256, 128]" = torch.ops.aten.permute.default(primals_200, [1, 0]);  primals_200 = None
    addmm_73: "f32[512, 128]" = torch.ops.aten.addmm.default(primals_201, view_266, permute_133);  primals_201 = None
    view_267: "f32[1, 512, 128]" = torch.ops.aten.view.default(addmm_73, [1, 512, 128]);  addmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_87: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(view_267, 0.5)
    mul_88: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(view_267, 0.7071067811865476)
    erf_12: "f32[1, 512, 128]" = torch.ops.aten.erf.default(mul_88);  mul_88 = None
    add_100: "f32[1, 512, 128]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_89: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_87, add_100);  mul_87 = add_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:662, code: hidden_states = self.LayerNorm(hidden_states)
    var_mean_25 = torch.ops.aten.var_mean.correction(mul_89, [2], correction = 0, keepdim = True)
    getitem_124: "f32[1, 512, 1]" = var_mean_25[0]
    getitem_125: "f32[1, 512, 1]" = var_mean_25[1];  var_mean_25 = None
    add_101: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_124, 1e-12);  getitem_124 = None
    rsqrt_25: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
    sub_38: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(mul_89, getitem_125)
    mul_90: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_25);  sub_38 = None
    mul_91: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_90, primals_202);  mul_90 = None
    add_102: "f32[1, 512, 128]" = torch.ops.aten.add.Tensor(mul_91, primals_203);  mul_91 = primals_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1640, code: prediction_scores = self.generator_lm_head(self.generator_predictions(sequence_output))
    view_268: "f32[512, 128]" = torch.ops.aten.view.default(add_102, [512, 128]);  add_102 = None
    permute_134: "f32[128, 30522]" = torch.ops.aten.permute.default(primals_204, [1, 0]);  primals_204 = None
    addmm_74: "f32[512, 30522]" = torch.ops.aten.addmm.default(primals_205, view_268, permute_134);  primals_205 = None
    view_269: "f32[1, 512, 30522]" = torch.ops.aten.view.default(addmm_74, [1, 512, 30522]);  addmm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1645, code: shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
    slice_5: "f32[1, 512, 30522]" = torch.ops.aten.slice.Tensor(view_269, 0, 0, 9223372036854775807)
    slice_6: "f32[1, 511, 30522]" = torch.ops.aten.slice.Tensor(slice_5, 1, 0, -1);  slice_5 = None
    slice_7: "f32[1, 511, 30522]" = torch.ops.aten.slice.Tensor(slice_6, 2, 0, 9223372036854775807);  slice_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1646, code: labels = labels[:, 1:].contiguous()
    slice_8: "i64[1, 512]" = torch.ops.aten.slice.Tensor(primals_208, 0, 0, 9223372036854775807);  primals_208 = None
    slice_9: "i64[1, 511]" = torch.ops.aten.slice.Tensor(slice_8, 1, 1, 9223372036854775807);  slice_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1648, code: lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    view_270: "f32[511, 30522]" = torch.ops.aten.view.default(slice_7, [-1, 30522]);  slice_7 = None
    view_271: "i64[511]" = torch.ops.aten.view.default(slice_9, [-1]);  slice_9 = None
    amax_12: "f32[511, 1]" = torch.ops.aten.amax.default(view_270, [1], True)
    sub_39: "f32[511, 30522]" = torch.ops.aten.sub.Tensor(view_270, amax_12);  view_270 = amax_12 = None
    exp_12: "f32[511, 30522]" = torch.ops.aten.exp.default(sub_39)
    sum_13: "f32[511, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [1], True);  exp_12 = None
    log: "f32[511, 1]" = torch.ops.aten.log.default(sum_13);  sum_13 = None
    sub_40: "f32[511, 30522]" = torch.ops.aten.sub.Tensor(sub_39, log);  sub_39 = log = None
    alias_12: "f32[511, 30522]" = torch.ops.aten.alias.default(sub_40)
    ne: "b8[511]" = torch.ops.aten.ne.Scalar(view_271, -100)
    scalar_tensor: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0))
    where: "i64[511]" = torch.ops.aten.where.self(ne, view_271, scalar_tensor);  ne = scalar_tensor = None
    unsqueeze_2: "i64[511, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
    gather: "f32[511, 1]" = torch.ops.aten.gather.default(sub_40, 1, unsqueeze_2);  sub_40 = unsqueeze_2 = None
    squeeze: "f32[511]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[511]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
    ne_1: "b8[511]" = torch.ops.aten.ne.Scalar(view_271, -100)
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_1: "f32[511]" = torch.ops.aten.where.self(ne_1, neg, scalar_tensor_1);  ne_1 = neg = scalar_tensor_1 = None
    ne_2: "b8[511]" = torch.ops.aten.ne.Scalar(view_271, -100)
    sum_14: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None
    sum_15: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
    div_24: "f32[]" = torch.ops.aten.div.Tensor(sum_15, convert_element_type);  sum_15 = None
    div_25: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type);  tangents_1 = convert_element_type = None
    unsqueeze_3: "i64[511, 1]" = torch.ops.aten.unsqueeze.default(view_271, 1);  view_271 = None
    ne_3: "b8[511, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_3, -100)
    scalar_tensor_2: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0))
    where_2: "i64[511, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_3, scalar_tensor_2);  ne_3 = scalar_tensor_2 = None
    full_1: "f32[511, 30522]" = torch.ops.aten.full.default([511, 30522], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    scatter: "f32[511, 30522]" = torch.ops.aten.scatter.value(full_1, 1, where_2, -1.0);  full_1 = where_2 = None
    ne_4: "b8[511, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_3, -100);  unsqueeze_3 = None
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_3: "f32[511, 1]" = torch.ops.aten.where.self(ne_4, div_25, scalar_tensor_3);  ne_4 = div_25 = scalar_tensor_3 = None
    mul_92: "f32[511, 30522]" = torch.ops.aten.mul.Tensor(scatter, where_3);  scatter = where_3 = None
    alias_13: "f32[511, 30522]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    exp_13: "f32[511, 30522]" = torch.ops.aten.exp.default(alias_13);  alias_13 = None
    sum_16: "f32[511, 1]" = torch.ops.aten.sum.dim_IntList(mul_92, [1], True)
    mul_93: "f32[511, 30522]" = torch.ops.aten.mul.Tensor(exp_13, sum_16);  exp_13 = sum_16 = None
    sub_41: "f32[511, 30522]" = torch.ops.aten.sub.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
    view_272: "f32[1, 511, 30522]" = torch.ops.aten.view.default(sub_41, [1, 511, 30522]);  sub_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1645, code: shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
    full_2: "f32[1, 511, 30522]" = torch.ops.aten.full.default([1, 511, 30522], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter: "f32[1, 511, 30522]" = torch.ops.aten.slice_scatter.default(full_2, view_272, 2, 0, 9223372036854775807);  full_2 = view_272 = None
    full_3: "f32[1, 512, 30522]" = torch.ops.aten.full.default([1, 512, 30522], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_1: "f32[1, 512, 30522]" = torch.ops.aten.slice_scatter.default(full_3, slice_scatter, 1, 0, -1);  full_3 = slice_scatter = None
    full_4: "f32[1, 512, 30522]" = torch.ops.aten.full.default([1, 512, 30522], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_2: "f32[1, 512, 30522]" = torch.ops.aten.slice_scatter.default(full_4, slice_scatter_1, 0, 0, 9223372036854775807);  full_4 = slice_scatter_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1645, code: shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
    add_103: "f32[1, 512, 30522]" = torch.ops.aten.add.Tensor(tangents_2, slice_scatter_2);  tangents_2 = slice_scatter_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1640, code: prediction_scores = self.generator_lm_head(self.generator_predictions(sequence_output))
    view_273: "f32[512, 30522]" = torch.ops.aten.view.default(add_103, [512, 30522]);  add_103 = None
    permute_135: "f32[30522, 128]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
    mm: "f32[512, 128]" = torch.ops.aten.mm.default(view_273, permute_135);  permute_135 = None
    permute_136: "f32[30522, 512]" = torch.ops.aten.permute.default(view_273, [1, 0])
    mm_1: "f32[30522, 128]" = torch.ops.aten.mm.default(permute_136, view_268);  permute_136 = view_268 = None
    permute_137: "f32[128, 30522]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_17: "f32[1, 30522]" = torch.ops.aten.sum.dim_IntList(view_273, [0], True);  view_273 = None
    view_274: "f32[30522]" = torch.ops.aten.view.default(sum_17, [30522]);  sum_17 = None
    permute_138: "f32[30522, 128]" = torch.ops.aten.permute.default(permute_137, [1, 0]);  permute_137 = None
    view_275: "f32[1, 512, 128]" = torch.ops.aten.view.default(mm, [1, 512, 128]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:662, code: hidden_states = self.LayerNorm(hidden_states)
    sub_42: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(mul_89, getitem_125);  mul_89 = getitem_125 = None
    mul_94: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_25);  sub_42 = None
    mul_95: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(view_275, primals_202);  primals_202 = None
    mul_96: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_95, 128)
    sum_18: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_95, [2], True)
    mul_97: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_95, mul_94);  mul_95 = None
    sum_19: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_97, [2], True);  mul_97 = None
    mul_98: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_94, sum_19);  sum_19 = None
    sub_43: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(mul_96, sum_18);  mul_96 = sum_18 = None
    sub_44: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(sub_43, mul_98);  sub_43 = mul_98 = None
    div_26: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 128);  rsqrt_25 = None
    mul_99: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(div_26, sub_44);  div_26 = sub_44 = None
    mul_100: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(view_275, mul_94);  mul_94 = None
    sum_20: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_100, [0, 1]);  mul_100 = None
    sum_21: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_275, [0, 1]);  view_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_101: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(view_267, 0.7071067811865476)
    erf_13: "f32[1, 512, 128]" = torch.ops.aten.erf.default(mul_101);  mul_101 = None
    add_104: "f32[1, 512, 128]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_102: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(add_104, 0.5);  add_104 = None
    mul_103: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(view_267, view_267)
    mul_104: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_103, -0.5);  mul_103 = None
    exp_14: "f32[1, 512, 128]" = torch.ops.aten.exp.default(mul_104);  mul_104 = None
    mul_105: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
    mul_106: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(view_267, mul_105);  view_267 = mul_105 = None
    add_105: "f32[1, 512, 128]" = torch.ops.aten.add.Tensor(mul_102, mul_106);  mul_102 = mul_106 = None
    mul_107: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_99, add_105);  mul_99 = add_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:660, code: hidden_states = self.dense(generator_hidden_states)
    view_276: "f32[512, 128]" = torch.ops.aten.view.default(mul_107, [512, 128]);  mul_107 = None
    permute_139: "f32[128, 256]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
    mm_2: "f32[512, 256]" = torch.ops.aten.mm.default(view_276, permute_139);  permute_139 = None
    permute_140: "f32[128, 512]" = torch.ops.aten.permute.default(view_276, [1, 0])
    mm_3: "f32[128, 256]" = torch.ops.aten.mm.default(permute_140, view_266);  permute_140 = view_266 = None
    permute_141: "f32[256, 128]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_22: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_276, [0], True);  view_276 = None
    view_277: "f32[128]" = torch.ops.aten.view.default(sum_22, [128]);  sum_22 = None
    permute_142: "f32[128, 256]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
    view_278: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_2, [1, 512, 256]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_45: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_97, getitem_123);  add_97 = getitem_123 = None
    mul_108: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_24);  sub_45 = None
    mul_109: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(view_278, primals_198);  primals_198 = None
    mul_110: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_109, 256)
    sum_23: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_109, [2], True)
    mul_111: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_109, mul_108);  mul_109 = None
    sum_24: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_111, [2], True);  mul_111 = None
    mul_112: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_108, sum_24);  sum_24 = None
    sub_46: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_110, sum_23);  mul_110 = sum_23 = None
    sub_47: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_46, mul_112);  sub_46 = mul_112 = None
    div_27: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 256);  rsqrt_24 = None
    mul_113: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_27, sub_47);  div_27 = sub_47 = None
    mul_114: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(view_278, mul_108);  mul_108 = None
    sum_25: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_114, [0, 1]);  mul_114 = None
    sum_26: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_278, [0, 1]);  view_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_1: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_121, torch.float32);  getitem_121 = None
    mul_115: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
    mul_116: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_113, mul_115);  mul_115 = None
    clone_12: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_116, memory_format = torch.contiguous_format);  mul_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_279: "f32[512, 256]" = torch.ops.aten.view.default(clone_12, [512, 256]);  clone_12 = None
    permute_143: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
    mm_4: "f32[512, 1024]" = torch.ops.aten.mm.default(view_279, permute_143);  permute_143 = None
    permute_144: "f32[256, 512]" = torch.ops.aten.permute.default(view_279, [1, 0])
    mm_5: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_144, view_264);  permute_144 = view_264 = None
    permute_145: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_27: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_279, [0], True);  view_279 = None
    view_280: "f32[256]" = torch.ops.aten.view.default(sum_27, [256]);  sum_27 = None
    permute_146: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
    view_281: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_4, [1, 512, 1024]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_117: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_263, 0.7071067811865476)
    erf_14: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_117);  mul_117 = None
    add_106: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_118: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_106, 0.5);  add_106 = None
    mul_119: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_263, view_263)
    mul_120: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_119, -0.5);  mul_119 = None
    exp_15: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_120);  mul_120 = None
    mul_121: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_15, 0.3989422804014327);  exp_15 = None
    mul_122: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_263, mul_121);  view_263 = mul_121 = None
    add_107: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_118, mul_122);  mul_118 = mul_122 = None
    mul_123: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_281, add_107);  view_281 = add_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_282: "f32[512, 1024]" = torch.ops.aten.view.default(mul_123, [512, 1024]);  mul_123 = None
    permute_147: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
    mm_6: "f32[512, 256]" = torch.ops.aten.mm.default(view_282, permute_147);  permute_147 = None
    permute_148: "f32[1024, 512]" = torch.ops.aten.permute.default(view_282, [1, 0])
    mm_7: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_148, view_262);  permute_148 = view_262 = None
    permute_149: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_28: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_282, [0], True);  view_282 = None
    view_283: "f32[1024]" = torch.ops.aten.view.default(sum_28, [1024]);  sum_28 = None
    permute_150: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_149, [1, 0]);  permute_149 = None
    view_284: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_6, [1, 512, 256]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_108: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_113, view_284);  mul_113 = view_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_48: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_93, getitem_119);  add_93 = getitem_119 = None
    mul_124: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_23);  sub_48 = None
    mul_125: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_108, primals_192);  primals_192 = None
    mul_126: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_125, 256)
    sum_29: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_125, [2], True)
    mul_127: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_125, mul_124);  mul_125 = None
    sum_30: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_127, [2], True);  mul_127 = None
    mul_128: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_124, sum_30);  sum_30 = None
    sub_49: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_126, sum_29);  mul_126 = sum_29 = None
    sub_50: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_49, mul_128);  sub_49 = mul_128 = None
    div_28: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 256);  rsqrt_23 = None
    mul_129: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_28, sub_50);  div_28 = sub_50 = None
    mul_130: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_108, mul_124);  mul_124 = None
    sum_31: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_130, [0, 1]);  mul_130 = None
    sum_32: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_108, [0, 1]);  add_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_2: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_117, torch.float32);  getitem_117 = None
    mul_131: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
    mul_132: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_129, mul_131);  mul_131 = None
    clone_13: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_132, memory_format = torch.contiguous_format);  mul_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_285: "f32[512, 256]" = torch.ops.aten.view.default(clone_13, [512, 256]);  clone_13 = None
    permute_151: "f32[256, 256]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
    mm_8: "f32[512, 256]" = torch.ops.aten.mm.default(view_285, permute_151);  permute_151 = None
    permute_152: "f32[256, 512]" = torch.ops.aten.permute.default(view_285, [1, 0])
    mm_9: "f32[256, 256]" = torch.ops.aten.mm.default(permute_152, view_260);  permute_152 = view_260 = None
    permute_153: "f32[256, 256]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_33: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_285, [0], True);  view_285 = None
    view_286: "f32[256]" = torch.ops.aten.view.default(sum_33, [256]);  sum_33 = None
    permute_154: "f32[256, 256]" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
    view_287: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_8, [1, 512, 256]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_288: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_287, [1, 512, 4, 64]);  view_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_155: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_288, [0, 2, 1, 3]);  view_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_289: "f32[4, 512, 64]" = torch.ops.aten.view.default(permute_155, [4, 512, 64]);  permute_155 = None
    permute_156: "f32[4, 512, 512]" = torch.ops.aten.permute.default(view_256, [0, 2, 1]);  view_256 = None
    bmm_24: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(permute_156, view_289);  permute_156 = None
    permute_157: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_257, [0, 2, 1]);  view_257 = None
    bmm_25: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_289, permute_157);  view_289 = permute_157 = None
    view_290: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_24, [1, 4, 512, 64]);  bmm_24 = None
    view_291: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_25, [1, 4, 512, 512]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_3: "f32[1, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_115, torch.float32);  getitem_115 = None
    mul_133: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.1111111111111112);  convert_element_type_3 = None
    mul_134: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_291, mul_133);  view_291 = mul_133 = None
    clone_14: "f32[1, 4, 512, 512]" = torch.ops.aten.clone.default(mul_134, memory_format = torch.contiguous_format);  mul_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_14: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    mul_135: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(clone_14, alias_14);  clone_14 = None
    sum_34: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_135, [-1], True)
    mul_136: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(alias_14, sum_34);  alias_14 = sum_34 = None
    sub_51: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(mul_135, mul_136);  mul_135 = mul_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_29: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(sub_51, 8.0);  sub_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_292: "f32[4, 512, 512]" = torch.ops.aten.view.default(div_29, [4, 512, 512]);  div_29 = None
    permute_158: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_253, [0, 2, 1]);  view_253 = None
    bmm_26: "f32[4, 64, 512]" = torch.ops.aten.bmm.default(permute_158, view_292);  permute_158 = None
    permute_159: "f32[4, 512, 64]" = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None
    bmm_27: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_292, permute_159);  view_292 = permute_159 = None
    view_293: "f32[1, 4, 64, 512]" = torch.ops.aten.view.default(bmm_26, [1, 4, 64, 512]);  bmm_26 = None
    view_294: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_27, [1, 4, 512, 64]);  bmm_27 = None
    permute_160: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_293, [0, 1, 3, 2]);  view_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_161: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_294, [0, 2, 1, 3]);  view_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_15: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
    view_295: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_15, [1, 512, 256]);  clone_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_162: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_290, [0, 2, 1, 3]);  view_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_16: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_162, memory_format = torch.contiguous_format);  permute_162 = None
    view_296: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_16, [1, 512, 256]);  clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_297: "f32[512, 256]" = torch.ops.aten.view.default(view_296, [512, 256]);  view_296 = None
    permute_163: "f32[256, 256]" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
    mm_10: "f32[512, 256]" = torch.ops.aten.mm.default(view_297, permute_163);  permute_163 = None
    permute_164: "f32[256, 512]" = torch.ops.aten.permute.default(view_297, [1, 0])
    mm_11: "f32[256, 256]" = torch.ops.aten.mm.default(permute_164, view_249);  permute_164 = view_249 = None
    permute_165: "f32[256, 256]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_35: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_297, [0], True);  view_297 = None
    view_298: "f32[256]" = torch.ops.aten.view.default(sum_35, [256]);  sum_35 = None
    permute_166: "f32[256, 256]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
    view_299: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_10, [1, 512, 256]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_109: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_129, view_299);  mul_129 = view_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_167: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(permute_160, [0, 2, 1, 3]);  permute_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_300: "f32[1, 512, 256]" = torch.ops.aten.view.default(permute_167, [1, 512, 256]);  permute_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_301: "f32[512, 256]" = torch.ops.aten.view.default(view_300, [512, 256]);  view_300 = None
    permute_168: "f32[256, 256]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
    mm_12: "f32[512, 256]" = torch.ops.aten.mm.default(view_301, permute_168);  permute_168 = None
    permute_169: "f32[256, 512]" = torch.ops.aten.permute.default(view_301, [1, 0])
    mm_13: "f32[256, 256]" = torch.ops.aten.mm.default(permute_169, view_246);  permute_169 = view_246 = None
    permute_170: "f32[256, 256]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_36: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_301, [0], True);  view_301 = None
    view_302: "f32[256]" = torch.ops.aten.view.default(sum_36, [256]);  sum_36 = None
    permute_171: "f32[256, 256]" = torch.ops.aten.permute.default(permute_170, [1, 0]);  permute_170 = None
    view_303: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_12, [1, 512, 256]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_110: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_109, view_303);  add_109 = view_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_304: "f32[512, 256]" = torch.ops.aten.view.default(view_295, [512, 256]);  view_295 = None
    permute_172: "f32[256, 256]" = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
    mm_14: "f32[512, 256]" = torch.ops.aten.mm.default(view_304, permute_172);  permute_172 = None
    permute_173: "f32[256, 512]" = torch.ops.aten.permute.default(view_304, [1, 0])
    mm_15: "f32[256, 256]" = torch.ops.aten.mm.default(permute_173, view_244);  permute_173 = view_244 = None
    permute_174: "f32[256, 256]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_37: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_304, [0], True);  view_304 = None
    view_305: "f32[256]" = torch.ops.aten.view.default(sum_37, [256]);  sum_37 = None
    permute_175: "f32[256, 256]" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
    view_306: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_14, [1, 512, 256]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_111: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_110, view_306);  add_110 = view_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_52: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_89, getitem_113);  add_89 = getitem_113 = None
    mul_137: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_22);  sub_52 = None
    mul_138: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_111, primals_182);  primals_182 = None
    mul_139: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_138, 256)
    sum_38: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_138, [2], True)
    mul_140: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_138, mul_137);  mul_138 = None
    sum_39: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_140, [2], True);  mul_140 = None
    mul_141: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_137, sum_39);  sum_39 = None
    sub_53: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_139, sum_38);  mul_139 = sum_38 = None
    sub_54: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_53, mul_141);  sub_53 = mul_141 = None
    div_30: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 256);  rsqrt_22 = None
    mul_142: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_30, sub_54);  div_30 = sub_54 = None
    mul_143: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_111, mul_137);  mul_137 = None
    sum_40: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_143, [0, 1]);  mul_143 = None
    sum_41: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_111, [0, 1]);  add_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_4: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_111, torch.float32);  getitem_111 = None
    mul_144: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 1.1111111111111112);  convert_element_type_4 = None
    mul_145: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_142, mul_144);  mul_144 = None
    clone_17: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_145, memory_format = torch.contiguous_format);  mul_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_307: "f32[512, 256]" = torch.ops.aten.view.default(clone_17, [512, 256]);  clone_17 = None
    permute_176: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
    mm_16: "f32[512, 1024]" = torch.ops.aten.mm.default(view_307, permute_176);  permute_176 = None
    permute_177: "f32[256, 512]" = torch.ops.aten.permute.default(view_307, [1, 0])
    mm_17: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_177, view_242);  permute_177 = view_242 = None
    permute_178: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_42: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_307, [0], True);  view_307 = None
    view_308: "f32[256]" = torch.ops.aten.view.default(sum_42, [256]);  sum_42 = None
    permute_179: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None
    view_309: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_16, [1, 512, 1024]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_146: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_241, 0.7071067811865476)
    erf_15: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_146);  mul_146 = None
    add_112: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_147: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_112, 0.5);  add_112 = None
    mul_148: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_241, view_241)
    mul_149: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_148, -0.5);  mul_148 = None
    exp_16: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_149);  mul_149 = None
    mul_150: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
    mul_151: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_241, mul_150);  view_241 = mul_150 = None
    add_113: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_147, mul_151);  mul_147 = mul_151 = None
    mul_152: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_309, add_113);  view_309 = add_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_310: "f32[512, 1024]" = torch.ops.aten.view.default(mul_152, [512, 1024]);  mul_152 = None
    permute_180: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
    mm_18: "f32[512, 256]" = torch.ops.aten.mm.default(view_310, permute_180);  permute_180 = None
    permute_181: "f32[1024, 512]" = torch.ops.aten.permute.default(view_310, [1, 0])
    mm_19: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_181, view_240);  permute_181 = view_240 = None
    permute_182: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_43: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_310, [0], True);  view_310 = None
    view_311: "f32[1024]" = torch.ops.aten.view.default(sum_43, [1024]);  sum_43 = None
    permute_183: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_182, [1, 0]);  permute_182 = None
    view_312: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_18, [1, 512, 256]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_114: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_142, view_312);  mul_142 = view_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_55: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_85, getitem_109);  add_85 = getitem_109 = None
    mul_153: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_21);  sub_55 = None
    mul_154: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_114, primals_176);  primals_176 = None
    mul_155: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_154, 256)
    sum_44: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_154, [2], True)
    mul_156: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_154, mul_153);  mul_154 = None
    sum_45: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_156, [2], True);  mul_156 = None
    mul_157: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_153, sum_45);  sum_45 = None
    sub_56: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_155, sum_44);  mul_155 = sum_44 = None
    sub_57: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_56, mul_157);  sub_56 = mul_157 = None
    div_31: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 256);  rsqrt_21 = None
    mul_158: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_31, sub_57);  div_31 = sub_57 = None
    mul_159: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_114, mul_153);  mul_153 = None
    sum_46: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_159, [0, 1]);  mul_159 = None
    sum_47: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_114, [0, 1]);  add_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_5: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_107, torch.float32);  getitem_107 = None
    mul_160: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 1.1111111111111112);  convert_element_type_5 = None
    mul_161: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_158, mul_160);  mul_160 = None
    clone_18: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_161, memory_format = torch.contiguous_format);  mul_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_313: "f32[512, 256]" = torch.ops.aten.view.default(clone_18, [512, 256]);  clone_18 = None
    permute_184: "f32[256, 256]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
    mm_20: "f32[512, 256]" = torch.ops.aten.mm.default(view_313, permute_184);  permute_184 = None
    permute_185: "f32[256, 512]" = torch.ops.aten.permute.default(view_313, [1, 0])
    mm_21: "f32[256, 256]" = torch.ops.aten.mm.default(permute_185, view_238);  permute_185 = view_238 = None
    permute_186: "f32[256, 256]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_48: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_313, [0], True);  view_313 = None
    view_314: "f32[256]" = torch.ops.aten.view.default(sum_48, [256]);  sum_48 = None
    permute_187: "f32[256, 256]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
    view_315: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_20, [1, 512, 256]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_316: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_315, [1, 512, 4, 64]);  view_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_188: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_316, [0, 2, 1, 3]);  view_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_317: "f32[4, 512, 64]" = torch.ops.aten.view.default(permute_188, [4, 512, 64]);  permute_188 = None
    permute_189: "f32[4, 512, 512]" = torch.ops.aten.permute.default(view_234, [0, 2, 1]);  view_234 = None
    bmm_28: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(permute_189, view_317);  permute_189 = None
    permute_190: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_235, [0, 2, 1]);  view_235 = None
    bmm_29: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_317, permute_190);  view_317 = permute_190 = None
    view_318: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_28, [1, 4, 512, 64]);  bmm_28 = None
    view_319: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_29, [1, 4, 512, 512]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_6: "f32[1, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_105, torch.float32);  getitem_105 = None
    mul_162: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 1.1111111111111112);  convert_element_type_6 = None
    mul_163: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_319, mul_162);  view_319 = mul_162 = None
    clone_19: "f32[1, 4, 512, 512]" = torch.ops.aten.clone.default(mul_163, memory_format = torch.contiguous_format);  mul_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_15: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    mul_164: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(clone_19, alias_15);  clone_19 = None
    sum_49: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_164, [-1], True)
    mul_165: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(alias_15, sum_49);  alias_15 = sum_49 = None
    sub_58: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(mul_164, mul_165);  mul_164 = mul_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_32: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(sub_58, 8.0);  sub_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_320: "f32[4, 512, 512]" = torch.ops.aten.view.default(div_32, [4, 512, 512]);  div_32 = None
    permute_191: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_231, [0, 2, 1]);  view_231 = None
    bmm_30: "f32[4, 64, 512]" = torch.ops.aten.bmm.default(permute_191, view_320);  permute_191 = None
    permute_192: "f32[4, 512, 64]" = torch.ops.aten.permute.default(view_232, [0, 2, 1]);  view_232 = None
    bmm_31: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_320, permute_192);  view_320 = permute_192 = None
    view_321: "f32[1, 4, 64, 512]" = torch.ops.aten.view.default(bmm_30, [1, 4, 64, 512]);  bmm_30 = None
    view_322: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_31, [1, 4, 512, 64]);  bmm_31 = None
    permute_193: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_321, [0, 1, 3, 2]);  view_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_194: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_322, [0, 2, 1, 3]);  view_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_20: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
    view_323: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_20, [1, 512, 256]);  clone_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_195: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_318, [0, 2, 1, 3]);  view_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_21: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_195, memory_format = torch.contiguous_format);  permute_195 = None
    view_324: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_21, [1, 512, 256]);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_325: "f32[512, 256]" = torch.ops.aten.view.default(view_324, [512, 256]);  view_324 = None
    permute_196: "f32[256, 256]" = torch.ops.aten.permute.default(permute_114, [1, 0]);  permute_114 = None
    mm_22: "f32[512, 256]" = torch.ops.aten.mm.default(view_325, permute_196);  permute_196 = None
    permute_197: "f32[256, 512]" = torch.ops.aten.permute.default(view_325, [1, 0])
    mm_23: "f32[256, 256]" = torch.ops.aten.mm.default(permute_197, view_227);  permute_197 = view_227 = None
    permute_198: "f32[256, 256]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_50: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_325, [0], True);  view_325 = None
    view_326: "f32[256]" = torch.ops.aten.view.default(sum_50, [256]);  sum_50 = None
    permute_199: "f32[256, 256]" = torch.ops.aten.permute.default(permute_198, [1, 0]);  permute_198 = None
    view_327: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_22, [1, 512, 256]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_115: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_158, view_327);  mul_158 = view_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_200: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(permute_193, [0, 2, 1, 3]);  permute_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_328: "f32[1, 512, 256]" = torch.ops.aten.view.default(permute_200, [1, 512, 256]);  permute_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_329: "f32[512, 256]" = torch.ops.aten.view.default(view_328, [512, 256]);  view_328 = None
    permute_201: "f32[256, 256]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
    mm_24: "f32[512, 256]" = torch.ops.aten.mm.default(view_329, permute_201);  permute_201 = None
    permute_202: "f32[256, 512]" = torch.ops.aten.permute.default(view_329, [1, 0])
    mm_25: "f32[256, 256]" = torch.ops.aten.mm.default(permute_202, view_224);  permute_202 = view_224 = None
    permute_203: "f32[256, 256]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_51: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_329, [0], True);  view_329 = None
    view_330: "f32[256]" = torch.ops.aten.view.default(sum_51, [256]);  sum_51 = None
    permute_204: "f32[256, 256]" = torch.ops.aten.permute.default(permute_203, [1, 0]);  permute_203 = None
    view_331: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_24, [1, 512, 256]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_116: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_115, view_331);  add_115 = view_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_332: "f32[512, 256]" = torch.ops.aten.view.default(view_323, [512, 256]);  view_323 = None
    permute_205: "f32[256, 256]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
    mm_26: "f32[512, 256]" = torch.ops.aten.mm.default(view_332, permute_205);  permute_205 = None
    permute_206: "f32[256, 512]" = torch.ops.aten.permute.default(view_332, [1, 0])
    mm_27: "f32[256, 256]" = torch.ops.aten.mm.default(permute_206, view_222);  permute_206 = view_222 = None
    permute_207: "f32[256, 256]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_52: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_332, [0], True);  view_332 = None
    view_333: "f32[256]" = torch.ops.aten.view.default(sum_52, [256]);  sum_52 = None
    permute_208: "f32[256, 256]" = torch.ops.aten.permute.default(permute_207, [1, 0]);  permute_207 = None
    view_334: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_26, [1, 512, 256]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_117: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_116, view_334);  add_116 = view_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_59: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_81, getitem_103);  add_81 = getitem_103 = None
    mul_166: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_20);  sub_59 = None
    mul_167: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_117, primals_166);  primals_166 = None
    mul_168: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_167, 256)
    sum_53: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_167, [2], True)
    mul_169: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_167, mul_166);  mul_167 = None
    sum_54: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_169, [2], True);  mul_169 = None
    mul_170: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_166, sum_54);  sum_54 = None
    sub_60: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_168, sum_53);  mul_168 = sum_53 = None
    sub_61: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_60, mul_170);  sub_60 = mul_170 = None
    div_33: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 256);  rsqrt_20 = None
    mul_171: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_33, sub_61);  div_33 = sub_61 = None
    mul_172: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_117, mul_166);  mul_166 = None
    sum_55: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_172, [0, 1]);  mul_172 = None
    sum_56: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_117, [0, 1]);  add_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_7: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_101, torch.float32);  getitem_101 = None
    mul_173: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_7, 1.1111111111111112);  convert_element_type_7 = None
    mul_174: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_171, mul_173);  mul_173 = None
    clone_22: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_174, memory_format = torch.contiguous_format);  mul_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_335: "f32[512, 256]" = torch.ops.aten.view.default(clone_22, [512, 256]);  clone_22 = None
    permute_209: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
    mm_28: "f32[512, 1024]" = torch.ops.aten.mm.default(view_335, permute_209);  permute_209 = None
    permute_210: "f32[256, 512]" = torch.ops.aten.permute.default(view_335, [1, 0])
    mm_29: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_210, view_220);  permute_210 = view_220 = None
    permute_211: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_57: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_335, [0], True);  view_335 = None
    view_336: "f32[256]" = torch.ops.aten.view.default(sum_57, [256]);  sum_57 = None
    permute_212: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None
    view_337: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_28, [1, 512, 1024]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_175: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_219, 0.7071067811865476)
    erf_16: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_175);  mul_175 = None
    add_118: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_176: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_118, 0.5);  add_118 = None
    mul_177: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_219, view_219)
    mul_178: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_177, -0.5);  mul_177 = None
    exp_17: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_178);  mul_178 = None
    mul_179: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_17, 0.3989422804014327);  exp_17 = None
    mul_180: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_219, mul_179);  view_219 = mul_179 = None
    add_119: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_176, mul_180);  mul_176 = mul_180 = None
    mul_181: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_337, add_119);  view_337 = add_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_338: "f32[512, 1024]" = torch.ops.aten.view.default(mul_181, [512, 1024]);  mul_181 = None
    permute_213: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
    mm_30: "f32[512, 256]" = torch.ops.aten.mm.default(view_338, permute_213);  permute_213 = None
    permute_214: "f32[1024, 512]" = torch.ops.aten.permute.default(view_338, [1, 0])
    mm_31: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_214, view_218);  permute_214 = view_218 = None
    permute_215: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_58: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_338, [0], True);  view_338 = None
    view_339: "f32[1024]" = torch.ops.aten.view.default(sum_58, [1024]);  sum_58 = None
    permute_216: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_215, [1, 0]);  permute_215 = None
    view_340: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_30, [1, 512, 256]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_120: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_171, view_340);  mul_171 = view_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_62: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_77, getitem_99);  add_77 = getitem_99 = None
    mul_182: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_19);  sub_62 = None
    mul_183: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_120, primals_160);  primals_160 = None
    mul_184: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_183, 256)
    sum_59: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_183, [2], True)
    mul_185: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_183, mul_182);  mul_183 = None
    sum_60: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_185, [2], True);  mul_185 = None
    mul_186: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_182, sum_60);  sum_60 = None
    sub_63: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_184, sum_59);  mul_184 = sum_59 = None
    sub_64: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_63, mul_186);  sub_63 = mul_186 = None
    div_34: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 256);  rsqrt_19 = None
    mul_187: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_34, sub_64);  div_34 = sub_64 = None
    mul_188: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_120, mul_182);  mul_182 = None
    sum_61: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_188, [0, 1]);  mul_188 = None
    sum_62: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_120, [0, 1]);  add_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_8: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_97, torch.float32);  getitem_97 = None
    mul_189: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.1111111111111112);  convert_element_type_8 = None
    mul_190: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_187, mul_189);  mul_189 = None
    clone_23: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_190, memory_format = torch.contiguous_format);  mul_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_341: "f32[512, 256]" = torch.ops.aten.view.default(clone_23, [512, 256]);  clone_23 = None
    permute_217: "f32[256, 256]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
    mm_32: "f32[512, 256]" = torch.ops.aten.mm.default(view_341, permute_217);  permute_217 = None
    permute_218: "f32[256, 512]" = torch.ops.aten.permute.default(view_341, [1, 0])
    mm_33: "f32[256, 256]" = torch.ops.aten.mm.default(permute_218, view_216);  permute_218 = view_216 = None
    permute_219: "f32[256, 256]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_63: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_341, [0], True);  view_341 = None
    view_342: "f32[256]" = torch.ops.aten.view.default(sum_63, [256]);  sum_63 = None
    permute_220: "f32[256, 256]" = torch.ops.aten.permute.default(permute_219, [1, 0]);  permute_219 = None
    view_343: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_32, [1, 512, 256]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_344: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_343, [1, 512, 4, 64]);  view_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_221: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_344, [0, 2, 1, 3]);  view_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_345: "f32[4, 512, 64]" = torch.ops.aten.view.default(permute_221, [4, 512, 64]);  permute_221 = None
    permute_222: "f32[4, 512, 512]" = torch.ops.aten.permute.default(view_212, [0, 2, 1]);  view_212 = None
    bmm_32: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(permute_222, view_345);  permute_222 = None
    permute_223: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_213, [0, 2, 1]);  view_213 = None
    bmm_33: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_345, permute_223);  view_345 = permute_223 = None
    view_346: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_32, [1, 4, 512, 64]);  bmm_32 = None
    view_347: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_33, [1, 4, 512, 512]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_9: "f32[1, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_95, torch.float32);  getitem_95 = None
    mul_191: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_9, 1.1111111111111112);  convert_element_type_9 = None
    mul_192: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_347, mul_191);  view_347 = mul_191 = None
    clone_24: "f32[1, 4, 512, 512]" = torch.ops.aten.clone.default(mul_192, memory_format = torch.contiguous_format);  mul_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_16: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    mul_193: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(clone_24, alias_16);  clone_24 = None
    sum_64: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_193, [-1], True)
    mul_194: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(alias_16, sum_64);  alias_16 = sum_64 = None
    sub_65: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(mul_193, mul_194);  mul_193 = mul_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_35: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(sub_65, 8.0);  sub_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_348: "f32[4, 512, 512]" = torch.ops.aten.view.default(div_35, [4, 512, 512]);  div_35 = None
    permute_224: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_209, [0, 2, 1]);  view_209 = None
    bmm_34: "f32[4, 64, 512]" = torch.ops.aten.bmm.default(permute_224, view_348);  permute_224 = None
    permute_225: "f32[4, 512, 64]" = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None
    bmm_35: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_348, permute_225);  view_348 = permute_225 = None
    view_349: "f32[1, 4, 64, 512]" = torch.ops.aten.view.default(bmm_34, [1, 4, 64, 512]);  bmm_34 = None
    view_350: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_35, [1, 4, 512, 64]);  bmm_35 = None
    permute_226: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_349, [0, 1, 3, 2]);  view_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_227: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_350, [0, 2, 1, 3]);  view_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_25: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
    view_351: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_25, [1, 512, 256]);  clone_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_228: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_346, [0, 2, 1, 3]);  view_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_26: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_228, memory_format = torch.contiguous_format);  permute_228 = None
    view_352: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_26, [1, 512, 256]);  clone_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_353: "f32[512, 256]" = torch.ops.aten.view.default(view_352, [512, 256]);  view_352 = None
    permute_229: "f32[256, 256]" = torch.ops.aten.permute.default(permute_103, [1, 0]);  permute_103 = None
    mm_34: "f32[512, 256]" = torch.ops.aten.mm.default(view_353, permute_229);  permute_229 = None
    permute_230: "f32[256, 512]" = torch.ops.aten.permute.default(view_353, [1, 0])
    mm_35: "f32[256, 256]" = torch.ops.aten.mm.default(permute_230, view_205);  permute_230 = view_205 = None
    permute_231: "f32[256, 256]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_65: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_353, [0], True);  view_353 = None
    view_354: "f32[256]" = torch.ops.aten.view.default(sum_65, [256]);  sum_65 = None
    permute_232: "f32[256, 256]" = torch.ops.aten.permute.default(permute_231, [1, 0]);  permute_231 = None
    view_355: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_34, [1, 512, 256]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_121: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_187, view_355);  mul_187 = view_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_233: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(permute_226, [0, 2, 1, 3]);  permute_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_356: "f32[1, 512, 256]" = torch.ops.aten.view.default(permute_233, [1, 512, 256]);  permute_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_357: "f32[512, 256]" = torch.ops.aten.view.default(view_356, [512, 256]);  view_356 = None
    permute_234: "f32[256, 256]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
    mm_36: "f32[512, 256]" = torch.ops.aten.mm.default(view_357, permute_234);  permute_234 = None
    permute_235: "f32[256, 512]" = torch.ops.aten.permute.default(view_357, [1, 0])
    mm_37: "f32[256, 256]" = torch.ops.aten.mm.default(permute_235, view_202);  permute_235 = view_202 = None
    permute_236: "f32[256, 256]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_66: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_357, [0], True);  view_357 = None
    view_358: "f32[256]" = torch.ops.aten.view.default(sum_66, [256]);  sum_66 = None
    permute_237: "f32[256, 256]" = torch.ops.aten.permute.default(permute_236, [1, 0]);  permute_236 = None
    view_359: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_36, [1, 512, 256]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_122: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_121, view_359);  add_121 = view_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_360: "f32[512, 256]" = torch.ops.aten.view.default(view_351, [512, 256]);  view_351 = None
    permute_238: "f32[256, 256]" = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
    mm_38: "f32[512, 256]" = torch.ops.aten.mm.default(view_360, permute_238);  permute_238 = None
    permute_239: "f32[256, 512]" = torch.ops.aten.permute.default(view_360, [1, 0])
    mm_39: "f32[256, 256]" = torch.ops.aten.mm.default(permute_239, view_200);  permute_239 = view_200 = None
    permute_240: "f32[256, 256]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_67: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_360, [0], True);  view_360 = None
    view_361: "f32[256]" = torch.ops.aten.view.default(sum_67, [256]);  sum_67 = None
    permute_241: "f32[256, 256]" = torch.ops.aten.permute.default(permute_240, [1, 0]);  permute_240 = None
    view_362: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_38, [1, 512, 256]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_123: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_122, view_362);  add_122 = view_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_66: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_73, getitem_93);  add_73 = getitem_93 = None
    mul_195: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_18);  sub_66 = None
    mul_196: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_123, primals_150);  primals_150 = None
    mul_197: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_196, 256)
    sum_68: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_196, [2], True)
    mul_198: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_196, mul_195);  mul_196 = None
    sum_69: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_198, [2], True);  mul_198 = None
    mul_199: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_195, sum_69);  sum_69 = None
    sub_67: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_197, sum_68);  mul_197 = sum_68 = None
    sub_68: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_67, mul_199);  sub_67 = mul_199 = None
    div_36: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 256);  rsqrt_18 = None
    mul_200: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_36, sub_68);  div_36 = sub_68 = None
    mul_201: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_123, mul_195);  mul_195 = None
    sum_70: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_201, [0, 1]);  mul_201 = None
    sum_71: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_123, [0, 1]);  add_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_10: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_91, torch.float32);  getitem_91 = None
    mul_202: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.1111111111111112);  convert_element_type_10 = None
    mul_203: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_200, mul_202);  mul_202 = None
    clone_27: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_203, memory_format = torch.contiguous_format);  mul_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_363: "f32[512, 256]" = torch.ops.aten.view.default(clone_27, [512, 256]);  clone_27 = None
    permute_242: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
    mm_40: "f32[512, 1024]" = torch.ops.aten.mm.default(view_363, permute_242);  permute_242 = None
    permute_243: "f32[256, 512]" = torch.ops.aten.permute.default(view_363, [1, 0])
    mm_41: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_243, view_198);  permute_243 = view_198 = None
    permute_244: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_72: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_363, [0], True);  view_363 = None
    view_364: "f32[256]" = torch.ops.aten.view.default(sum_72, [256]);  sum_72 = None
    permute_245: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_244, [1, 0]);  permute_244 = None
    view_365: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_40, [1, 512, 1024]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_204: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_197, 0.7071067811865476)
    erf_17: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_204);  mul_204 = None
    add_124: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_205: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_124, 0.5);  add_124 = None
    mul_206: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_197, view_197)
    mul_207: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_206, -0.5);  mul_206 = None
    exp_18: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_207);  mul_207 = None
    mul_208: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
    mul_209: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_197, mul_208);  view_197 = mul_208 = None
    add_125: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_205, mul_209);  mul_205 = mul_209 = None
    mul_210: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_365, add_125);  view_365 = add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_366: "f32[512, 1024]" = torch.ops.aten.view.default(mul_210, [512, 1024]);  mul_210 = None
    permute_246: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
    mm_42: "f32[512, 256]" = torch.ops.aten.mm.default(view_366, permute_246);  permute_246 = None
    permute_247: "f32[1024, 512]" = torch.ops.aten.permute.default(view_366, [1, 0])
    mm_43: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_247, view_196);  permute_247 = view_196 = None
    permute_248: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_73: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_366, [0], True);  view_366 = None
    view_367: "f32[1024]" = torch.ops.aten.view.default(sum_73, [1024]);  sum_73 = None
    permute_249: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_248, [1, 0]);  permute_248 = None
    view_368: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_42, [1, 512, 256]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_126: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_200, view_368);  mul_200 = view_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_69: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_69, getitem_89);  add_69 = getitem_89 = None
    mul_211: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_17);  sub_69 = None
    mul_212: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_126, primals_144);  primals_144 = None
    mul_213: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_212, 256)
    sum_74: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_212, [2], True)
    mul_214: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_212, mul_211);  mul_212 = None
    sum_75: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_214, [2], True);  mul_214 = None
    mul_215: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_211, sum_75);  sum_75 = None
    sub_70: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_213, sum_74);  mul_213 = sum_74 = None
    sub_71: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_70, mul_215);  sub_70 = mul_215 = None
    div_37: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 256);  rsqrt_17 = None
    mul_216: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_37, sub_71);  div_37 = sub_71 = None
    mul_217: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_126, mul_211);  mul_211 = None
    sum_76: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_217, [0, 1]);  mul_217 = None
    sum_77: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_126, [0, 1]);  add_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_11: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_87, torch.float32);  getitem_87 = None
    mul_218: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 1.1111111111111112);  convert_element_type_11 = None
    mul_219: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_216, mul_218);  mul_218 = None
    clone_28: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_219, memory_format = torch.contiguous_format);  mul_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_369: "f32[512, 256]" = torch.ops.aten.view.default(clone_28, [512, 256]);  clone_28 = None
    permute_250: "f32[256, 256]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
    mm_44: "f32[512, 256]" = torch.ops.aten.mm.default(view_369, permute_250);  permute_250 = None
    permute_251: "f32[256, 512]" = torch.ops.aten.permute.default(view_369, [1, 0])
    mm_45: "f32[256, 256]" = torch.ops.aten.mm.default(permute_251, view_194);  permute_251 = view_194 = None
    permute_252: "f32[256, 256]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_78: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_369, [0], True);  view_369 = None
    view_370: "f32[256]" = torch.ops.aten.view.default(sum_78, [256]);  sum_78 = None
    permute_253: "f32[256, 256]" = torch.ops.aten.permute.default(permute_252, [1, 0]);  permute_252 = None
    view_371: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_44, [1, 512, 256]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_372: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_371, [1, 512, 4, 64]);  view_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_254: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_372, [0, 2, 1, 3]);  view_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_373: "f32[4, 512, 64]" = torch.ops.aten.view.default(permute_254, [4, 512, 64]);  permute_254 = None
    permute_255: "f32[4, 512, 512]" = torch.ops.aten.permute.default(view_190, [0, 2, 1]);  view_190 = None
    bmm_36: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(permute_255, view_373);  permute_255 = None
    permute_256: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_191, [0, 2, 1]);  view_191 = None
    bmm_37: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_373, permute_256);  view_373 = permute_256 = None
    view_374: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_36, [1, 4, 512, 64]);  bmm_36 = None
    view_375: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_37, [1, 4, 512, 512]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_12: "f32[1, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_85, torch.float32);  getitem_85 = None
    mul_220: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_12, 1.1111111111111112);  convert_element_type_12 = None
    mul_221: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_375, mul_220);  view_375 = mul_220 = None
    clone_29: "f32[1, 4, 512, 512]" = torch.ops.aten.clone.default(mul_221, memory_format = torch.contiguous_format);  mul_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_17: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    mul_222: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(clone_29, alias_17);  clone_29 = None
    sum_79: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_222, [-1], True)
    mul_223: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(alias_17, sum_79);  alias_17 = sum_79 = None
    sub_72: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(mul_222, mul_223);  mul_222 = mul_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_38: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(sub_72, 8.0);  sub_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_376: "f32[4, 512, 512]" = torch.ops.aten.view.default(div_38, [4, 512, 512]);  div_38 = None
    permute_257: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_187, [0, 2, 1]);  view_187 = None
    bmm_38: "f32[4, 64, 512]" = torch.ops.aten.bmm.default(permute_257, view_376);  permute_257 = None
    permute_258: "f32[4, 512, 64]" = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None
    bmm_39: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_376, permute_258);  view_376 = permute_258 = None
    view_377: "f32[1, 4, 64, 512]" = torch.ops.aten.view.default(bmm_38, [1, 4, 64, 512]);  bmm_38 = None
    view_378: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_39, [1, 4, 512, 64]);  bmm_39 = None
    permute_259: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_377, [0, 1, 3, 2]);  view_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_260: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_378, [0, 2, 1, 3]);  view_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_30: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None
    view_379: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_30, [1, 512, 256]);  clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_261: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_374, [0, 2, 1, 3]);  view_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_31: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_261, memory_format = torch.contiguous_format);  permute_261 = None
    view_380: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_31, [1, 512, 256]);  clone_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_381: "f32[512, 256]" = torch.ops.aten.view.default(view_380, [512, 256]);  view_380 = None
    permute_262: "f32[256, 256]" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
    mm_46: "f32[512, 256]" = torch.ops.aten.mm.default(view_381, permute_262);  permute_262 = None
    permute_263: "f32[256, 512]" = torch.ops.aten.permute.default(view_381, [1, 0])
    mm_47: "f32[256, 256]" = torch.ops.aten.mm.default(permute_263, view_183);  permute_263 = view_183 = None
    permute_264: "f32[256, 256]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_80: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_381, [0], True);  view_381 = None
    view_382: "f32[256]" = torch.ops.aten.view.default(sum_80, [256]);  sum_80 = None
    permute_265: "f32[256, 256]" = torch.ops.aten.permute.default(permute_264, [1, 0]);  permute_264 = None
    view_383: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_46, [1, 512, 256]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_127: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_216, view_383);  mul_216 = view_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_266: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(permute_259, [0, 2, 1, 3]);  permute_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_384: "f32[1, 512, 256]" = torch.ops.aten.view.default(permute_266, [1, 512, 256]);  permute_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_385: "f32[512, 256]" = torch.ops.aten.view.default(view_384, [512, 256]);  view_384 = None
    permute_267: "f32[256, 256]" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
    mm_48: "f32[512, 256]" = torch.ops.aten.mm.default(view_385, permute_267);  permute_267 = None
    permute_268: "f32[256, 512]" = torch.ops.aten.permute.default(view_385, [1, 0])
    mm_49: "f32[256, 256]" = torch.ops.aten.mm.default(permute_268, view_180);  permute_268 = view_180 = None
    permute_269: "f32[256, 256]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_81: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_385, [0], True);  view_385 = None
    view_386: "f32[256]" = torch.ops.aten.view.default(sum_81, [256]);  sum_81 = None
    permute_270: "f32[256, 256]" = torch.ops.aten.permute.default(permute_269, [1, 0]);  permute_269 = None
    view_387: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_48, [1, 512, 256]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_128: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_127, view_387);  add_127 = view_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_388: "f32[512, 256]" = torch.ops.aten.view.default(view_379, [512, 256]);  view_379 = None
    permute_271: "f32[256, 256]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
    mm_50: "f32[512, 256]" = torch.ops.aten.mm.default(view_388, permute_271);  permute_271 = None
    permute_272: "f32[256, 512]" = torch.ops.aten.permute.default(view_388, [1, 0])
    mm_51: "f32[256, 256]" = torch.ops.aten.mm.default(permute_272, view_178);  permute_272 = view_178 = None
    permute_273: "f32[256, 256]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_82: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_388, [0], True);  view_388 = None
    view_389: "f32[256]" = torch.ops.aten.view.default(sum_82, [256]);  sum_82 = None
    permute_274: "f32[256, 256]" = torch.ops.aten.permute.default(permute_273, [1, 0]);  permute_273 = None
    view_390: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_50, [1, 512, 256]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_129: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_128, view_390);  add_128 = view_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_73: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_65, getitem_83);  add_65 = getitem_83 = None
    mul_224: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_16);  sub_73 = None
    mul_225: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_129, primals_134);  primals_134 = None
    mul_226: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_225, 256)
    sum_83: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_225, [2], True)
    mul_227: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_225, mul_224);  mul_225 = None
    sum_84: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_227, [2], True);  mul_227 = None
    mul_228: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_224, sum_84);  sum_84 = None
    sub_74: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_226, sum_83);  mul_226 = sum_83 = None
    sub_75: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_74, mul_228);  sub_74 = mul_228 = None
    div_39: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 256);  rsqrt_16 = None
    mul_229: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_39, sub_75);  div_39 = sub_75 = None
    mul_230: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_129, mul_224);  mul_224 = None
    sum_85: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_230, [0, 1]);  mul_230 = None
    sum_86: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_129, [0, 1]);  add_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_13: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_81, torch.float32);  getitem_81 = None
    mul_231: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
    mul_232: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_229, mul_231);  mul_231 = None
    clone_32: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_232, memory_format = torch.contiguous_format);  mul_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_391: "f32[512, 256]" = torch.ops.aten.view.default(clone_32, [512, 256]);  clone_32 = None
    permute_275: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
    mm_52: "f32[512, 1024]" = torch.ops.aten.mm.default(view_391, permute_275);  permute_275 = None
    permute_276: "f32[256, 512]" = torch.ops.aten.permute.default(view_391, [1, 0])
    mm_53: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_276, view_176);  permute_276 = view_176 = None
    permute_277: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_87: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_391, [0], True);  view_391 = None
    view_392: "f32[256]" = torch.ops.aten.view.default(sum_87, [256]);  sum_87 = None
    permute_278: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_277, [1, 0]);  permute_277 = None
    view_393: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_52, [1, 512, 1024]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_233: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_175, 0.7071067811865476)
    erf_18: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_233);  mul_233 = None
    add_130: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_234: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_130, 0.5);  add_130 = None
    mul_235: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_175, view_175)
    mul_236: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_235, -0.5);  mul_235 = None
    exp_19: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_236);  mul_236 = None
    mul_237: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
    mul_238: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_175, mul_237);  view_175 = mul_237 = None
    add_131: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_234, mul_238);  mul_234 = mul_238 = None
    mul_239: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_393, add_131);  view_393 = add_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_394: "f32[512, 1024]" = torch.ops.aten.view.default(mul_239, [512, 1024]);  mul_239 = None
    permute_279: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
    mm_54: "f32[512, 256]" = torch.ops.aten.mm.default(view_394, permute_279);  permute_279 = None
    permute_280: "f32[1024, 512]" = torch.ops.aten.permute.default(view_394, [1, 0])
    mm_55: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_280, view_174);  permute_280 = view_174 = None
    permute_281: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_88: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_394, [0], True);  view_394 = None
    view_395: "f32[1024]" = torch.ops.aten.view.default(sum_88, [1024]);  sum_88 = None
    permute_282: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_281, [1, 0]);  permute_281 = None
    view_396: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_54, [1, 512, 256]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_132: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_229, view_396);  mul_229 = view_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_76: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_61, getitem_79);  add_61 = getitem_79 = None
    mul_240: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_15);  sub_76 = None
    mul_241: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_132, primals_128);  primals_128 = None
    mul_242: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_241, 256)
    sum_89: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_241, [2], True)
    mul_243: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_241, mul_240);  mul_241 = None
    sum_90: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_243, [2], True);  mul_243 = None
    mul_244: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_240, sum_90);  sum_90 = None
    sub_77: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_242, sum_89);  mul_242 = sum_89 = None
    sub_78: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_77, mul_244);  sub_77 = mul_244 = None
    div_40: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 256);  rsqrt_15 = None
    mul_245: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_40, sub_78);  div_40 = sub_78 = None
    mul_246: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_132, mul_240);  mul_240 = None
    sum_91: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_246, [0, 1]);  mul_246 = None
    sum_92: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_132, [0, 1]);  add_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_14: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_77, torch.float32);  getitem_77 = None
    mul_247: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 1.1111111111111112);  convert_element_type_14 = None
    mul_248: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_245, mul_247);  mul_247 = None
    clone_33: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_248, memory_format = torch.contiguous_format);  mul_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_397: "f32[512, 256]" = torch.ops.aten.view.default(clone_33, [512, 256]);  clone_33 = None
    permute_283: "f32[256, 256]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
    mm_56: "f32[512, 256]" = torch.ops.aten.mm.default(view_397, permute_283);  permute_283 = None
    permute_284: "f32[256, 512]" = torch.ops.aten.permute.default(view_397, [1, 0])
    mm_57: "f32[256, 256]" = torch.ops.aten.mm.default(permute_284, view_172);  permute_284 = view_172 = None
    permute_285: "f32[256, 256]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_93: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_397, [0], True);  view_397 = None
    view_398: "f32[256]" = torch.ops.aten.view.default(sum_93, [256]);  sum_93 = None
    permute_286: "f32[256, 256]" = torch.ops.aten.permute.default(permute_285, [1, 0]);  permute_285 = None
    view_399: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_56, [1, 512, 256]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_400: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_399, [1, 512, 4, 64]);  view_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_287: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_400, [0, 2, 1, 3]);  view_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_401: "f32[4, 512, 64]" = torch.ops.aten.view.default(permute_287, [4, 512, 64]);  permute_287 = None
    permute_288: "f32[4, 512, 512]" = torch.ops.aten.permute.default(view_168, [0, 2, 1]);  view_168 = None
    bmm_40: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(permute_288, view_401);  permute_288 = None
    permute_289: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_169, [0, 2, 1]);  view_169 = None
    bmm_41: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_401, permute_289);  view_401 = permute_289 = None
    view_402: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_40, [1, 4, 512, 64]);  bmm_40 = None
    view_403: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_41, [1, 4, 512, 512]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_15: "f32[1, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_75, torch.float32);  getitem_75 = None
    mul_249: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_15, 1.1111111111111112);  convert_element_type_15 = None
    mul_250: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_403, mul_249);  view_403 = mul_249 = None
    clone_34: "f32[1, 4, 512, 512]" = torch.ops.aten.clone.default(mul_250, memory_format = torch.contiguous_format);  mul_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_18: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    mul_251: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(clone_34, alias_18);  clone_34 = None
    sum_94: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_251, [-1], True)
    mul_252: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(alias_18, sum_94);  alias_18 = sum_94 = None
    sub_79: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(mul_251, mul_252);  mul_251 = mul_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_41: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(sub_79, 8.0);  sub_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_404: "f32[4, 512, 512]" = torch.ops.aten.view.default(div_41, [4, 512, 512]);  div_41 = None
    permute_290: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_165, [0, 2, 1]);  view_165 = None
    bmm_42: "f32[4, 64, 512]" = torch.ops.aten.bmm.default(permute_290, view_404);  permute_290 = None
    permute_291: "f32[4, 512, 64]" = torch.ops.aten.permute.default(view_166, [0, 2, 1]);  view_166 = None
    bmm_43: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_404, permute_291);  view_404 = permute_291 = None
    view_405: "f32[1, 4, 64, 512]" = torch.ops.aten.view.default(bmm_42, [1, 4, 64, 512]);  bmm_42 = None
    view_406: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_43, [1, 4, 512, 64]);  bmm_43 = None
    permute_292: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_405, [0, 1, 3, 2]);  view_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_293: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_406, [0, 2, 1, 3]);  view_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_35: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_293, memory_format = torch.contiguous_format);  permute_293 = None
    view_407: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_35, [1, 512, 256]);  clone_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_294: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_402, [0, 2, 1, 3]);  view_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_36: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_294, memory_format = torch.contiguous_format);  permute_294 = None
    view_408: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_36, [1, 512, 256]);  clone_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_409: "f32[512, 256]" = torch.ops.aten.view.default(view_408, [512, 256]);  view_408 = None
    permute_295: "f32[256, 256]" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
    mm_58: "f32[512, 256]" = torch.ops.aten.mm.default(view_409, permute_295);  permute_295 = None
    permute_296: "f32[256, 512]" = torch.ops.aten.permute.default(view_409, [1, 0])
    mm_59: "f32[256, 256]" = torch.ops.aten.mm.default(permute_296, view_161);  permute_296 = view_161 = None
    permute_297: "f32[256, 256]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_95: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_409, [0], True);  view_409 = None
    view_410: "f32[256]" = torch.ops.aten.view.default(sum_95, [256]);  sum_95 = None
    permute_298: "f32[256, 256]" = torch.ops.aten.permute.default(permute_297, [1, 0]);  permute_297 = None
    view_411: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_58, [1, 512, 256]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_133: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_245, view_411);  mul_245 = view_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_299: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(permute_292, [0, 2, 1, 3]);  permute_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_412: "f32[1, 512, 256]" = torch.ops.aten.view.default(permute_299, [1, 512, 256]);  permute_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_413: "f32[512, 256]" = torch.ops.aten.view.default(view_412, [512, 256]);  view_412 = None
    permute_300: "f32[256, 256]" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
    mm_60: "f32[512, 256]" = torch.ops.aten.mm.default(view_413, permute_300);  permute_300 = None
    permute_301: "f32[256, 512]" = torch.ops.aten.permute.default(view_413, [1, 0])
    mm_61: "f32[256, 256]" = torch.ops.aten.mm.default(permute_301, view_158);  permute_301 = view_158 = None
    permute_302: "f32[256, 256]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_96: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_413, [0], True);  view_413 = None
    view_414: "f32[256]" = torch.ops.aten.view.default(sum_96, [256]);  sum_96 = None
    permute_303: "f32[256, 256]" = torch.ops.aten.permute.default(permute_302, [1, 0]);  permute_302 = None
    view_415: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_60, [1, 512, 256]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_134: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_133, view_415);  add_133 = view_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_416: "f32[512, 256]" = torch.ops.aten.view.default(view_407, [512, 256]);  view_407 = None
    permute_304: "f32[256, 256]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
    mm_62: "f32[512, 256]" = torch.ops.aten.mm.default(view_416, permute_304);  permute_304 = None
    permute_305: "f32[256, 512]" = torch.ops.aten.permute.default(view_416, [1, 0])
    mm_63: "f32[256, 256]" = torch.ops.aten.mm.default(permute_305, view_156);  permute_305 = view_156 = None
    permute_306: "f32[256, 256]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_97: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_416, [0], True);  view_416 = None
    view_417: "f32[256]" = torch.ops.aten.view.default(sum_97, [256]);  sum_97 = None
    permute_307: "f32[256, 256]" = torch.ops.aten.permute.default(permute_306, [1, 0]);  permute_306 = None
    view_418: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_62, [1, 512, 256]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_135: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_134, view_418);  add_134 = view_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_80: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_57, getitem_73);  add_57 = getitem_73 = None
    mul_253: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_80, rsqrt_14);  sub_80 = None
    mul_254: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_135, primals_118);  primals_118 = None
    mul_255: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_254, 256)
    sum_98: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_254, [2], True)
    mul_256: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_254, mul_253);  mul_254 = None
    sum_99: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_256, [2], True);  mul_256 = None
    mul_257: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_253, sum_99);  sum_99 = None
    sub_81: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_255, sum_98);  mul_255 = sum_98 = None
    sub_82: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_81, mul_257);  sub_81 = mul_257 = None
    div_42: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 256);  rsqrt_14 = None
    mul_258: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_42, sub_82);  div_42 = sub_82 = None
    mul_259: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_135, mul_253);  mul_253 = None
    sum_100: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_259, [0, 1]);  mul_259 = None
    sum_101: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_135, [0, 1]);  add_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_16: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_71, torch.float32);  getitem_71 = None
    mul_260: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_16, 1.1111111111111112);  convert_element_type_16 = None
    mul_261: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_258, mul_260);  mul_260 = None
    clone_37: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_261, memory_format = torch.contiguous_format);  mul_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_419: "f32[512, 256]" = torch.ops.aten.view.default(clone_37, [512, 256]);  clone_37 = None
    permute_308: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
    mm_64: "f32[512, 1024]" = torch.ops.aten.mm.default(view_419, permute_308);  permute_308 = None
    permute_309: "f32[256, 512]" = torch.ops.aten.permute.default(view_419, [1, 0])
    mm_65: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_309, view_154);  permute_309 = view_154 = None
    permute_310: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_102: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_419, [0], True);  view_419 = None
    view_420: "f32[256]" = torch.ops.aten.view.default(sum_102, [256]);  sum_102 = None
    permute_311: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_310, [1, 0]);  permute_310 = None
    view_421: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_64, [1, 512, 1024]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_262: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_153, 0.7071067811865476)
    erf_19: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_262);  mul_262 = None
    add_136: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_263: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_136, 0.5);  add_136 = None
    mul_264: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_153, view_153)
    mul_265: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_264, -0.5);  mul_264 = None
    exp_20: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_265);  mul_265 = None
    mul_266: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
    mul_267: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_153, mul_266);  view_153 = mul_266 = None
    add_137: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_263, mul_267);  mul_263 = mul_267 = None
    mul_268: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_421, add_137);  view_421 = add_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_422: "f32[512, 1024]" = torch.ops.aten.view.default(mul_268, [512, 1024]);  mul_268 = None
    permute_312: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
    mm_66: "f32[512, 256]" = torch.ops.aten.mm.default(view_422, permute_312);  permute_312 = None
    permute_313: "f32[1024, 512]" = torch.ops.aten.permute.default(view_422, [1, 0])
    mm_67: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_313, view_152);  permute_313 = view_152 = None
    permute_314: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_103: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_422, [0], True);  view_422 = None
    view_423: "f32[1024]" = torch.ops.aten.view.default(sum_103, [1024]);  sum_103 = None
    permute_315: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_314, [1, 0]);  permute_314 = None
    view_424: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_66, [1, 512, 256]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_138: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_258, view_424);  mul_258 = view_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_83: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_53, getitem_69);  add_53 = getitem_69 = None
    mul_269: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_13);  sub_83 = None
    mul_270: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_138, primals_112);  primals_112 = None
    mul_271: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_270, 256)
    sum_104: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_270, [2], True)
    mul_272: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_270, mul_269);  mul_270 = None
    sum_105: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_272, [2], True);  mul_272 = None
    mul_273: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_269, sum_105);  sum_105 = None
    sub_84: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_271, sum_104);  mul_271 = sum_104 = None
    sub_85: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_84, mul_273);  sub_84 = mul_273 = None
    div_43: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 256);  rsqrt_13 = None
    mul_274: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_43, sub_85);  div_43 = sub_85 = None
    mul_275: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_138, mul_269);  mul_269 = None
    sum_106: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_275, [0, 1]);  mul_275 = None
    sum_107: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_138, [0, 1]);  add_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_17: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_67, torch.float32);  getitem_67 = None
    mul_276: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_17, 1.1111111111111112);  convert_element_type_17 = None
    mul_277: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_274, mul_276);  mul_276 = None
    clone_38: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_277, memory_format = torch.contiguous_format);  mul_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_425: "f32[512, 256]" = torch.ops.aten.view.default(clone_38, [512, 256]);  clone_38 = None
    permute_316: "f32[256, 256]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
    mm_68: "f32[512, 256]" = torch.ops.aten.mm.default(view_425, permute_316);  permute_316 = None
    permute_317: "f32[256, 512]" = torch.ops.aten.permute.default(view_425, [1, 0])
    mm_69: "f32[256, 256]" = torch.ops.aten.mm.default(permute_317, view_150);  permute_317 = view_150 = None
    permute_318: "f32[256, 256]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_108: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_425, [0], True);  view_425 = None
    view_426: "f32[256]" = torch.ops.aten.view.default(sum_108, [256]);  sum_108 = None
    permute_319: "f32[256, 256]" = torch.ops.aten.permute.default(permute_318, [1, 0]);  permute_318 = None
    view_427: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_68, [1, 512, 256]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_428: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_427, [1, 512, 4, 64]);  view_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_320: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_428, [0, 2, 1, 3]);  view_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_429: "f32[4, 512, 64]" = torch.ops.aten.view.default(permute_320, [4, 512, 64]);  permute_320 = None
    permute_321: "f32[4, 512, 512]" = torch.ops.aten.permute.default(view_146, [0, 2, 1]);  view_146 = None
    bmm_44: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(permute_321, view_429);  permute_321 = None
    permute_322: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_147, [0, 2, 1]);  view_147 = None
    bmm_45: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_429, permute_322);  view_429 = permute_322 = None
    view_430: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_44, [1, 4, 512, 64]);  bmm_44 = None
    view_431: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_45, [1, 4, 512, 512]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_18: "f32[1, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_65, torch.float32);  getitem_65 = None
    mul_278: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_18, 1.1111111111111112);  convert_element_type_18 = None
    mul_279: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_431, mul_278);  view_431 = mul_278 = None
    clone_39: "f32[1, 4, 512, 512]" = torch.ops.aten.clone.default(mul_279, memory_format = torch.contiguous_format);  mul_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_19: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    mul_280: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(clone_39, alias_19);  clone_39 = None
    sum_109: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_280, [-1], True)
    mul_281: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(alias_19, sum_109);  alias_19 = sum_109 = None
    sub_86: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(mul_280, mul_281);  mul_280 = mul_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_44: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(sub_86, 8.0);  sub_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_432: "f32[4, 512, 512]" = torch.ops.aten.view.default(div_44, [4, 512, 512]);  div_44 = None
    permute_323: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_143, [0, 2, 1]);  view_143 = None
    bmm_46: "f32[4, 64, 512]" = torch.ops.aten.bmm.default(permute_323, view_432);  permute_323 = None
    permute_324: "f32[4, 512, 64]" = torch.ops.aten.permute.default(view_144, [0, 2, 1]);  view_144 = None
    bmm_47: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_432, permute_324);  view_432 = permute_324 = None
    view_433: "f32[1, 4, 64, 512]" = torch.ops.aten.view.default(bmm_46, [1, 4, 64, 512]);  bmm_46 = None
    view_434: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_47, [1, 4, 512, 64]);  bmm_47 = None
    permute_325: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_433, [0, 1, 3, 2]);  view_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_326: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_434, [0, 2, 1, 3]);  view_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_40: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_326, memory_format = torch.contiguous_format);  permute_326 = None
    view_435: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_40, [1, 512, 256]);  clone_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_327: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_430, [0, 2, 1, 3]);  view_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_41: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_327, memory_format = torch.contiguous_format);  permute_327 = None
    view_436: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_41, [1, 512, 256]);  clone_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_437: "f32[512, 256]" = torch.ops.aten.view.default(view_436, [512, 256]);  view_436 = None
    permute_328: "f32[256, 256]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
    mm_70: "f32[512, 256]" = torch.ops.aten.mm.default(view_437, permute_328);  permute_328 = None
    permute_329: "f32[256, 512]" = torch.ops.aten.permute.default(view_437, [1, 0])
    mm_71: "f32[256, 256]" = torch.ops.aten.mm.default(permute_329, view_139);  permute_329 = view_139 = None
    permute_330: "f32[256, 256]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_110: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_437, [0], True);  view_437 = None
    view_438: "f32[256]" = torch.ops.aten.view.default(sum_110, [256]);  sum_110 = None
    permute_331: "f32[256, 256]" = torch.ops.aten.permute.default(permute_330, [1, 0]);  permute_330 = None
    view_439: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_70, [1, 512, 256]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_139: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_274, view_439);  mul_274 = view_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_332: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(permute_325, [0, 2, 1, 3]);  permute_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_440: "f32[1, 512, 256]" = torch.ops.aten.view.default(permute_332, [1, 512, 256]);  permute_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_441: "f32[512, 256]" = torch.ops.aten.view.default(view_440, [512, 256]);  view_440 = None
    permute_333: "f32[256, 256]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
    mm_72: "f32[512, 256]" = torch.ops.aten.mm.default(view_441, permute_333);  permute_333 = None
    permute_334: "f32[256, 512]" = torch.ops.aten.permute.default(view_441, [1, 0])
    mm_73: "f32[256, 256]" = torch.ops.aten.mm.default(permute_334, view_136);  permute_334 = view_136 = None
    permute_335: "f32[256, 256]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_111: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_441, [0], True);  view_441 = None
    view_442: "f32[256]" = torch.ops.aten.view.default(sum_111, [256]);  sum_111 = None
    permute_336: "f32[256, 256]" = torch.ops.aten.permute.default(permute_335, [1, 0]);  permute_335 = None
    view_443: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_72, [1, 512, 256]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_140: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_139, view_443);  add_139 = view_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_444: "f32[512, 256]" = torch.ops.aten.view.default(view_435, [512, 256]);  view_435 = None
    permute_337: "f32[256, 256]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
    mm_74: "f32[512, 256]" = torch.ops.aten.mm.default(view_444, permute_337);  permute_337 = None
    permute_338: "f32[256, 512]" = torch.ops.aten.permute.default(view_444, [1, 0])
    mm_75: "f32[256, 256]" = torch.ops.aten.mm.default(permute_338, view_134);  permute_338 = view_134 = None
    permute_339: "f32[256, 256]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_112: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_444, [0], True);  view_444 = None
    view_445: "f32[256]" = torch.ops.aten.view.default(sum_112, [256]);  sum_112 = None
    permute_340: "f32[256, 256]" = torch.ops.aten.permute.default(permute_339, [1, 0]);  permute_339 = None
    view_446: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_74, [1, 512, 256]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_141: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_140, view_446);  add_140 = view_446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_87: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_49, getitem_63);  add_49 = getitem_63 = None
    mul_282: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_12);  sub_87 = None
    mul_283: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_141, primals_102);  primals_102 = None
    mul_284: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_283, 256)
    sum_113: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_283, [2], True)
    mul_285: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_283, mul_282);  mul_283 = None
    sum_114: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_285, [2], True);  mul_285 = None
    mul_286: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_282, sum_114);  sum_114 = None
    sub_88: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_284, sum_113);  mul_284 = sum_113 = None
    sub_89: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_88, mul_286);  sub_88 = mul_286 = None
    div_45: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 256);  rsqrt_12 = None
    mul_287: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_45, sub_89);  div_45 = sub_89 = None
    mul_288: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_141, mul_282);  mul_282 = None
    sum_115: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_288, [0, 1]);  mul_288 = None
    sum_116: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_141, [0, 1]);  add_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_19: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_61, torch.float32);  getitem_61 = None
    mul_289: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_19, 1.1111111111111112);  convert_element_type_19 = None
    mul_290: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_287, mul_289);  mul_289 = None
    clone_42: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_290, memory_format = torch.contiguous_format);  mul_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_447: "f32[512, 256]" = torch.ops.aten.view.default(clone_42, [512, 256]);  clone_42 = None
    permute_341: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    mm_76: "f32[512, 1024]" = torch.ops.aten.mm.default(view_447, permute_341);  permute_341 = None
    permute_342: "f32[256, 512]" = torch.ops.aten.permute.default(view_447, [1, 0])
    mm_77: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_342, view_132);  permute_342 = view_132 = None
    permute_343: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_117: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_447, [0], True);  view_447 = None
    view_448: "f32[256]" = torch.ops.aten.view.default(sum_117, [256]);  sum_117 = None
    permute_344: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_343, [1, 0]);  permute_343 = None
    view_449: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_76, [1, 512, 1024]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_291: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_131, 0.7071067811865476)
    erf_20: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_291);  mul_291 = None
    add_142: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_292: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_142, 0.5);  add_142 = None
    mul_293: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_131, view_131)
    mul_294: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_293, -0.5);  mul_293 = None
    exp_21: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_294);  mul_294 = None
    mul_295: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
    mul_296: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_131, mul_295);  view_131 = mul_295 = None
    add_143: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_292, mul_296);  mul_292 = mul_296 = None
    mul_297: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_449, add_143);  view_449 = add_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_450: "f32[512, 1024]" = torch.ops.aten.view.default(mul_297, [512, 1024]);  mul_297 = None
    permute_345: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    mm_78: "f32[512, 256]" = torch.ops.aten.mm.default(view_450, permute_345);  permute_345 = None
    permute_346: "f32[1024, 512]" = torch.ops.aten.permute.default(view_450, [1, 0])
    mm_79: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_346, view_130);  permute_346 = view_130 = None
    permute_347: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_118: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_450, [0], True);  view_450 = None
    view_451: "f32[1024]" = torch.ops.aten.view.default(sum_118, [1024]);  sum_118 = None
    permute_348: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_347, [1, 0]);  permute_347 = None
    view_452: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_78, [1, 512, 256]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_144: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_287, view_452);  mul_287 = view_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_90: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_45, getitem_59);  add_45 = getitem_59 = None
    mul_298: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_90, rsqrt_11);  sub_90 = None
    mul_299: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_144, primals_96);  primals_96 = None
    mul_300: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_299, 256)
    sum_119: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_299, [2], True)
    mul_301: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_299, mul_298);  mul_299 = None
    sum_120: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_301, [2], True);  mul_301 = None
    mul_302: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_298, sum_120);  sum_120 = None
    sub_91: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_300, sum_119);  mul_300 = sum_119 = None
    sub_92: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_91, mul_302);  sub_91 = mul_302 = None
    div_46: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 256);  rsqrt_11 = None
    mul_303: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_46, sub_92);  div_46 = sub_92 = None
    mul_304: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_144, mul_298);  mul_298 = None
    sum_121: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_304, [0, 1]);  mul_304 = None
    sum_122: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_144, [0, 1]);  add_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_20: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_57, torch.float32);  getitem_57 = None
    mul_305: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.1111111111111112);  convert_element_type_20 = None
    mul_306: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_303, mul_305);  mul_305 = None
    clone_43: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_306, memory_format = torch.contiguous_format);  mul_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_453: "f32[512, 256]" = torch.ops.aten.view.default(clone_43, [512, 256]);  clone_43 = None
    permute_349: "f32[256, 256]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
    mm_80: "f32[512, 256]" = torch.ops.aten.mm.default(view_453, permute_349);  permute_349 = None
    permute_350: "f32[256, 512]" = torch.ops.aten.permute.default(view_453, [1, 0])
    mm_81: "f32[256, 256]" = torch.ops.aten.mm.default(permute_350, view_128);  permute_350 = view_128 = None
    permute_351: "f32[256, 256]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_123: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_453, [0], True);  view_453 = None
    view_454: "f32[256]" = torch.ops.aten.view.default(sum_123, [256]);  sum_123 = None
    permute_352: "f32[256, 256]" = torch.ops.aten.permute.default(permute_351, [1, 0]);  permute_351 = None
    view_455: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_80, [1, 512, 256]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_456: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_455, [1, 512, 4, 64]);  view_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_353: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_456, [0, 2, 1, 3]);  view_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_457: "f32[4, 512, 64]" = torch.ops.aten.view.default(permute_353, [4, 512, 64]);  permute_353 = None
    permute_354: "f32[4, 512, 512]" = torch.ops.aten.permute.default(view_124, [0, 2, 1]);  view_124 = None
    bmm_48: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(permute_354, view_457);  permute_354 = None
    permute_355: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_125, [0, 2, 1]);  view_125 = None
    bmm_49: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_457, permute_355);  view_457 = permute_355 = None
    view_458: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_48, [1, 4, 512, 64]);  bmm_48 = None
    view_459: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_49, [1, 4, 512, 512]);  bmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_21: "f32[1, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_55, torch.float32);  getitem_55 = None
    mul_307: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 1.1111111111111112);  convert_element_type_21 = None
    mul_308: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_459, mul_307);  view_459 = mul_307 = None
    clone_44: "f32[1, 4, 512, 512]" = torch.ops.aten.clone.default(mul_308, memory_format = torch.contiguous_format);  mul_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_20: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    mul_309: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(clone_44, alias_20);  clone_44 = None
    sum_124: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_309, [-1], True)
    mul_310: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(alias_20, sum_124);  alias_20 = sum_124 = None
    sub_93: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(mul_309, mul_310);  mul_309 = mul_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_47: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(sub_93, 8.0);  sub_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_460: "f32[4, 512, 512]" = torch.ops.aten.view.default(div_47, [4, 512, 512]);  div_47 = None
    permute_356: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_121, [0, 2, 1]);  view_121 = None
    bmm_50: "f32[4, 64, 512]" = torch.ops.aten.bmm.default(permute_356, view_460);  permute_356 = None
    permute_357: "f32[4, 512, 64]" = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None
    bmm_51: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_460, permute_357);  view_460 = permute_357 = None
    view_461: "f32[1, 4, 64, 512]" = torch.ops.aten.view.default(bmm_50, [1, 4, 64, 512]);  bmm_50 = None
    view_462: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_51, [1, 4, 512, 64]);  bmm_51 = None
    permute_358: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_461, [0, 1, 3, 2]);  view_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_359: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_462, [0, 2, 1, 3]);  view_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_45: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_359, memory_format = torch.contiguous_format);  permute_359 = None
    view_463: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_45, [1, 512, 256]);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_360: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_458, [0, 2, 1, 3]);  view_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_46: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_360, memory_format = torch.contiguous_format);  permute_360 = None
    view_464: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_46, [1, 512, 256]);  clone_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_465: "f32[512, 256]" = torch.ops.aten.view.default(view_464, [512, 256]);  view_464 = None
    permute_361: "f32[256, 256]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
    mm_82: "f32[512, 256]" = torch.ops.aten.mm.default(view_465, permute_361);  permute_361 = None
    permute_362: "f32[256, 512]" = torch.ops.aten.permute.default(view_465, [1, 0])
    mm_83: "f32[256, 256]" = torch.ops.aten.mm.default(permute_362, view_117);  permute_362 = view_117 = None
    permute_363: "f32[256, 256]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_125: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_465, [0], True);  view_465 = None
    view_466: "f32[256]" = torch.ops.aten.view.default(sum_125, [256]);  sum_125 = None
    permute_364: "f32[256, 256]" = torch.ops.aten.permute.default(permute_363, [1, 0]);  permute_363 = None
    view_467: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_82, [1, 512, 256]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_145: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_303, view_467);  mul_303 = view_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_365: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(permute_358, [0, 2, 1, 3]);  permute_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_468: "f32[1, 512, 256]" = torch.ops.aten.view.default(permute_365, [1, 512, 256]);  permute_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_469: "f32[512, 256]" = torch.ops.aten.view.default(view_468, [512, 256]);  view_468 = None
    permute_366: "f32[256, 256]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
    mm_84: "f32[512, 256]" = torch.ops.aten.mm.default(view_469, permute_366);  permute_366 = None
    permute_367: "f32[256, 512]" = torch.ops.aten.permute.default(view_469, [1, 0])
    mm_85: "f32[256, 256]" = torch.ops.aten.mm.default(permute_367, view_114);  permute_367 = view_114 = None
    permute_368: "f32[256, 256]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_126: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_469, [0], True);  view_469 = None
    view_470: "f32[256]" = torch.ops.aten.view.default(sum_126, [256]);  sum_126 = None
    permute_369: "f32[256, 256]" = torch.ops.aten.permute.default(permute_368, [1, 0]);  permute_368 = None
    view_471: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_84, [1, 512, 256]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_146: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_145, view_471);  add_145 = view_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_472: "f32[512, 256]" = torch.ops.aten.view.default(view_463, [512, 256]);  view_463 = None
    permute_370: "f32[256, 256]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    mm_86: "f32[512, 256]" = torch.ops.aten.mm.default(view_472, permute_370);  permute_370 = None
    permute_371: "f32[256, 512]" = torch.ops.aten.permute.default(view_472, [1, 0])
    mm_87: "f32[256, 256]" = torch.ops.aten.mm.default(permute_371, view_112);  permute_371 = view_112 = None
    permute_372: "f32[256, 256]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
    sum_127: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_472, [0], True);  view_472 = None
    view_473: "f32[256]" = torch.ops.aten.view.default(sum_127, [256]);  sum_127 = None
    permute_373: "f32[256, 256]" = torch.ops.aten.permute.default(permute_372, [1, 0]);  permute_372 = None
    view_474: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_86, [1, 512, 256]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_147: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_146, view_474);  add_146 = view_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_94: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_41, getitem_53);  add_41 = getitem_53 = None
    mul_311: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_94, rsqrt_10);  sub_94 = None
    mul_312: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_147, primals_86);  primals_86 = None
    mul_313: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_312, 256)
    sum_128: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_312, [2], True)
    mul_314: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_312, mul_311);  mul_312 = None
    sum_129: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_314, [2], True);  mul_314 = None
    mul_315: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_311, sum_129);  sum_129 = None
    sub_95: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_313, sum_128);  mul_313 = sum_128 = None
    sub_96: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_95, mul_315);  sub_95 = mul_315 = None
    div_48: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 256);  rsqrt_10 = None
    mul_316: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_48, sub_96);  div_48 = sub_96 = None
    mul_317: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_147, mul_311);  mul_311 = None
    sum_130: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_317, [0, 1]);  mul_317 = None
    sum_131: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_147, [0, 1]);  add_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_22: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_51, torch.float32);  getitem_51 = None
    mul_318: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 1.1111111111111112);  convert_element_type_22 = None
    mul_319: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_316, mul_318);  mul_318 = None
    clone_47: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_319, memory_format = torch.contiguous_format);  mul_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_475: "f32[512, 256]" = torch.ops.aten.view.default(clone_47, [512, 256]);  clone_47 = None
    permute_374: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    mm_88: "f32[512, 1024]" = torch.ops.aten.mm.default(view_475, permute_374);  permute_374 = None
    permute_375: "f32[256, 512]" = torch.ops.aten.permute.default(view_475, [1, 0])
    mm_89: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_375, view_110);  permute_375 = view_110 = None
    permute_376: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    sum_132: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_475, [0], True);  view_475 = None
    view_476: "f32[256]" = torch.ops.aten.view.default(sum_132, [256]);  sum_132 = None
    permute_377: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_376, [1, 0]);  permute_376 = None
    view_477: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_88, [1, 512, 1024]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_320: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_109, 0.7071067811865476)
    erf_21: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_320);  mul_320 = None
    add_148: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_321: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_148, 0.5);  add_148 = None
    mul_322: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_109, view_109)
    mul_323: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_322, -0.5);  mul_322 = None
    exp_22: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_323);  mul_323 = None
    mul_324: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
    mul_325: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_109, mul_324);  view_109 = mul_324 = None
    add_149: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_321, mul_325);  mul_321 = mul_325 = None
    mul_326: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_477, add_149);  view_477 = add_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_478: "f32[512, 1024]" = torch.ops.aten.view.default(mul_326, [512, 1024]);  mul_326 = None
    permute_378: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    mm_90: "f32[512, 256]" = torch.ops.aten.mm.default(view_478, permute_378);  permute_378 = None
    permute_379: "f32[1024, 512]" = torch.ops.aten.permute.default(view_478, [1, 0])
    mm_91: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_379, view_108);  permute_379 = view_108 = None
    permute_380: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_133: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_478, [0], True);  view_478 = None
    view_479: "f32[1024]" = torch.ops.aten.view.default(sum_133, [1024]);  sum_133 = None
    permute_381: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_380, [1, 0]);  permute_380 = None
    view_480: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_90, [1, 512, 256]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_150: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_316, view_480);  mul_316 = view_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_97: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_37, getitem_49);  add_37 = getitem_49 = None
    mul_327: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_97, rsqrt_9);  sub_97 = None
    mul_328: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_150, primals_80);  primals_80 = None
    mul_329: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_328, 256)
    sum_134: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_328, [2], True)
    mul_330: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_328, mul_327);  mul_328 = None
    sum_135: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_330, [2], True);  mul_330 = None
    mul_331: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_327, sum_135);  sum_135 = None
    sub_98: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_329, sum_134);  mul_329 = sum_134 = None
    sub_99: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_98, mul_331);  sub_98 = mul_331 = None
    div_49: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 256);  rsqrt_9 = None
    mul_332: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_49, sub_99);  div_49 = sub_99 = None
    mul_333: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_150, mul_327);  mul_327 = None
    sum_136: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_333, [0, 1]);  mul_333 = None
    sum_137: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_150, [0, 1]);  add_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_23: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_47, torch.float32);  getitem_47 = None
    mul_334: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_23, 1.1111111111111112);  convert_element_type_23 = None
    mul_335: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_332, mul_334);  mul_334 = None
    clone_48: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_335, memory_format = torch.contiguous_format);  mul_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_481: "f32[512, 256]" = torch.ops.aten.view.default(clone_48, [512, 256]);  clone_48 = None
    permute_382: "f32[256, 256]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
    mm_92: "f32[512, 256]" = torch.ops.aten.mm.default(view_481, permute_382);  permute_382 = None
    permute_383: "f32[256, 512]" = torch.ops.aten.permute.default(view_481, [1, 0])
    mm_93: "f32[256, 256]" = torch.ops.aten.mm.default(permute_383, view_106);  permute_383 = view_106 = None
    permute_384: "f32[256, 256]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_138: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_481, [0], True);  view_481 = None
    view_482: "f32[256]" = torch.ops.aten.view.default(sum_138, [256]);  sum_138 = None
    permute_385: "f32[256, 256]" = torch.ops.aten.permute.default(permute_384, [1, 0]);  permute_384 = None
    view_483: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_92, [1, 512, 256]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_484: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_483, [1, 512, 4, 64]);  view_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_386: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_484, [0, 2, 1, 3]);  view_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_485: "f32[4, 512, 64]" = torch.ops.aten.view.default(permute_386, [4, 512, 64]);  permute_386 = None
    permute_387: "f32[4, 512, 512]" = torch.ops.aten.permute.default(view_102, [0, 2, 1]);  view_102 = None
    bmm_52: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(permute_387, view_485);  permute_387 = None
    permute_388: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_103, [0, 2, 1]);  view_103 = None
    bmm_53: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_485, permute_388);  view_485 = permute_388 = None
    view_486: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_52, [1, 4, 512, 64]);  bmm_52 = None
    view_487: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_53, [1, 4, 512, 512]);  bmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_24: "f32[1, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_45, torch.float32);  getitem_45 = None
    mul_336: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 1.1111111111111112);  convert_element_type_24 = None
    mul_337: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_487, mul_336);  view_487 = mul_336 = None
    clone_49: "f32[1, 4, 512, 512]" = torch.ops.aten.clone.default(mul_337, memory_format = torch.contiguous_format);  mul_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_21: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    mul_338: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(clone_49, alias_21);  clone_49 = None
    sum_139: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_338, [-1], True)
    mul_339: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(alias_21, sum_139);  alias_21 = sum_139 = None
    sub_100: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(mul_338, mul_339);  mul_338 = mul_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_50: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(sub_100, 8.0);  sub_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_488: "f32[4, 512, 512]" = torch.ops.aten.view.default(div_50, [4, 512, 512]);  div_50 = None
    permute_389: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_99, [0, 2, 1]);  view_99 = None
    bmm_54: "f32[4, 64, 512]" = torch.ops.aten.bmm.default(permute_389, view_488);  permute_389 = None
    permute_390: "f32[4, 512, 64]" = torch.ops.aten.permute.default(view_100, [0, 2, 1]);  view_100 = None
    bmm_55: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_488, permute_390);  view_488 = permute_390 = None
    view_489: "f32[1, 4, 64, 512]" = torch.ops.aten.view.default(bmm_54, [1, 4, 64, 512]);  bmm_54 = None
    view_490: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_55, [1, 4, 512, 64]);  bmm_55 = None
    permute_391: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_489, [0, 1, 3, 2]);  view_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_392: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_490, [0, 2, 1, 3]);  view_490 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_50: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_392, memory_format = torch.contiguous_format);  permute_392 = None
    view_491: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_50, [1, 512, 256]);  clone_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_393: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_486, [0, 2, 1, 3]);  view_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_51: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_393, memory_format = torch.contiguous_format);  permute_393 = None
    view_492: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_51, [1, 512, 256]);  clone_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_493: "f32[512, 256]" = torch.ops.aten.view.default(view_492, [512, 256]);  view_492 = None
    permute_394: "f32[256, 256]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
    mm_94: "f32[512, 256]" = torch.ops.aten.mm.default(view_493, permute_394);  permute_394 = None
    permute_395: "f32[256, 512]" = torch.ops.aten.permute.default(view_493, [1, 0])
    mm_95: "f32[256, 256]" = torch.ops.aten.mm.default(permute_395, view_95);  permute_395 = view_95 = None
    permute_396: "f32[256, 256]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_140: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_493, [0], True);  view_493 = None
    view_494: "f32[256]" = torch.ops.aten.view.default(sum_140, [256]);  sum_140 = None
    permute_397: "f32[256, 256]" = torch.ops.aten.permute.default(permute_396, [1, 0]);  permute_396 = None
    view_495: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_94, [1, 512, 256]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_151: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_332, view_495);  mul_332 = view_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_398: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(permute_391, [0, 2, 1, 3]);  permute_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_496: "f32[1, 512, 256]" = torch.ops.aten.view.default(permute_398, [1, 512, 256]);  permute_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_497: "f32[512, 256]" = torch.ops.aten.view.default(view_496, [512, 256]);  view_496 = None
    permute_399: "f32[256, 256]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
    mm_96: "f32[512, 256]" = torch.ops.aten.mm.default(view_497, permute_399);  permute_399 = None
    permute_400: "f32[256, 512]" = torch.ops.aten.permute.default(view_497, [1, 0])
    mm_97: "f32[256, 256]" = torch.ops.aten.mm.default(permute_400, view_92);  permute_400 = view_92 = None
    permute_401: "f32[256, 256]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_141: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_497, [0], True);  view_497 = None
    view_498: "f32[256]" = torch.ops.aten.view.default(sum_141, [256]);  sum_141 = None
    permute_402: "f32[256, 256]" = torch.ops.aten.permute.default(permute_401, [1, 0]);  permute_401 = None
    view_499: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_96, [1, 512, 256]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_152: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_151, view_499);  add_151 = view_499 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_500: "f32[512, 256]" = torch.ops.aten.view.default(view_491, [512, 256]);  view_491 = None
    permute_403: "f32[256, 256]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
    mm_98: "f32[512, 256]" = torch.ops.aten.mm.default(view_500, permute_403);  permute_403 = None
    permute_404: "f32[256, 512]" = torch.ops.aten.permute.default(view_500, [1, 0])
    mm_99: "f32[256, 256]" = torch.ops.aten.mm.default(permute_404, view_90);  permute_404 = view_90 = None
    permute_405: "f32[256, 256]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    sum_142: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_500, [0], True);  view_500 = None
    view_501: "f32[256]" = torch.ops.aten.view.default(sum_142, [256]);  sum_142 = None
    permute_406: "f32[256, 256]" = torch.ops.aten.permute.default(permute_405, [1, 0]);  permute_405 = None
    view_502: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_98, [1, 512, 256]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_153: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_152, view_502);  add_152 = view_502 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_101: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_33, getitem_43);  add_33 = getitem_43 = None
    mul_340: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_101, rsqrt_8);  sub_101 = None
    mul_341: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_153, primals_70);  primals_70 = None
    mul_342: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_341, 256)
    sum_143: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_341, [2], True)
    mul_343: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_341, mul_340);  mul_341 = None
    sum_144: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_343, [2], True);  mul_343 = None
    mul_344: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_340, sum_144);  sum_144 = None
    sub_102: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_342, sum_143);  mul_342 = sum_143 = None
    sub_103: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_102, mul_344);  sub_102 = mul_344 = None
    div_51: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 256);  rsqrt_8 = None
    mul_345: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_51, sub_103);  div_51 = sub_103 = None
    mul_346: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_153, mul_340);  mul_340 = None
    sum_145: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_346, [0, 1]);  mul_346 = None
    sum_146: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_153, [0, 1]);  add_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_25: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_41, torch.float32);  getitem_41 = None
    mul_347: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.1111111111111112);  convert_element_type_25 = None
    mul_348: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_345, mul_347);  mul_347 = None
    clone_52: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_348, memory_format = torch.contiguous_format);  mul_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_503: "f32[512, 256]" = torch.ops.aten.view.default(clone_52, [512, 256]);  clone_52 = None
    permute_407: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
    mm_100: "f32[512, 1024]" = torch.ops.aten.mm.default(view_503, permute_407);  permute_407 = None
    permute_408: "f32[256, 512]" = torch.ops.aten.permute.default(view_503, [1, 0])
    mm_101: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_408, view_88);  permute_408 = view_88 = None
    permute_409: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_101, [1, 0]);  mm_101 = None
    sum_147: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_503, [0], True);  view_503 = None
    view_504: "f32[256]" = torch.ops.aten.view.default(sum_147, [256]);  sum_147 = None
    permute_410: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_409, [1, 0]);  permute_409 = None
    view_505: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_100, [1, 512, 1024]);  mm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_349: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_87, 0.7071067811865476)
    erf_22: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_349);  mul_349 = None
    add_154: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_350: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_154, 0.5);  add_154 = None
    mul_351: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_87, view_87)
    mul_352: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_351, -0.5);  mul_351 = None
    exp_23: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_352);  mul_352 = None
    mul_353: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
    mul_354: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_87, mul_353);  view_87 = mul_353 = None
    add_155: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_350, mul_354);  mul_350 = mul_354 = None
    mul_355: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_505, add_155);  view_505 = add_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_506: "f32[512, 1024]" = torch.ops.aten.view.default(mul_355, [512, 1024]);  mul_355 = None
    permute_411: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
    mm_102: "f32[512, 256]" = torch.ops.aten.mm.default(view_506, permute_411);  permute_411 = None
    permute_412: "f32[1024, 512]" = torch.ops.aten.permute.default(view_506, [1, 0])
    mm_103: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_412, view_86);  permute_412 = view_86 = None
    permute_413: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
    sum_148: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_506, [0], True);  view_506 = None
    view_507: "f32[1024]" = torch.ops.aten.view.default(sum_148, [1024]);  sum_148 = None
    permute_414: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_413, [1, 0]);  permute_413 = None
    view_508: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_102, [1, 512, 256]);  mm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_156: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_345, view_508);  mul_345 = view_508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_104: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_29, getitem_39);  add_29 = getitem_39 = None
    mul_356: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_104, rsqrt_7);  sub_104 = None
    mul_357: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_156, primals_64);  primals_64 = None
    mul_358: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_357, 256)
    sum_149: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_357, [2], True)
    mul_359: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_357, mul_356);  mul_357 = None
    sum_150: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_359, [2], True);  mul_359 = None
    mul_360: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_356, sum_150);  sum_150 = None
    sub_105: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_358, sum_149);  mul_358 = sum_149 = None
    sub_106: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_105, mul_360);  sub_105 = mul_360 = None
    div_52: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 256);  rsqrt_7 = None
    mul_361: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_52, sub_106);  div_52 = sub_106 = None
    mul_362: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_156, mul_356);  mul_356 = None
    sum_151: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_362, [0, 1]);  mul_362 = None
    sum_152: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_156, [0, 1]);  add_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_26: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_37, torch.float32);  getitem_37 = None
    mul_363: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_26, 1.1111111111111112);  convert_element_type_26 = None
    mul_364: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_361, mul_363);  mul_363 = None
    clone_53: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_364, memory_format = torch.contiguous_format);  mul_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_509: "f32[512, 256]" = torch.ops.aten.view.default(clone_53, [512, 256]);  clone_53 = None
    permute_415: "f32[256, 256]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    mm_104: "f32[512, 256]" = torch.ops.aten.mm.default(view_509, permute_415);  permute_415 = None
    permute_416: "f32[256, 512]" = torch.ops.aten.permute.default(view_509, [1, 0])
    mm_105: "f32[256, 256]" = torch.ops.aten.mm.default(permute_416, view_84);  permute_416 = view_84 = None
    permute_417: "f32[256, 256]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
    sum_153: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_509, [0], True);  view_509 = None
    view_510: "f32[256]" = torch.ops.aten.view.default(sum_153, [256]);  sum_153 = None
    permute_418: "f32[256, 256]" = torch.ops.aten.permute.default(permute_417, [1, 0]);  permute_417 = None
    view_511: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_104, [1, 512, 256]);  mm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_512: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_511, [1, 512, 4, 64]);  view_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_419: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_512, [0, 2, 1, 3]);  view_512 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_513: "f32[4, 512, 64]" = torch.ops.aten.view.default(permute_419, [4, 512, 64]);  permute_419 = None
    permute_420: "f32[4, 512, 512]" = torch.ops.aten.permute.default(view_80, [0, 2, 1]);  view_80 = None
    bmm_56: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(permute_420, view_513);  permute_420 = None
    permute_421: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_81, [0, 2, 1]);  view_81 = None
    bmm_57: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_513, permute_421);  view_513 = permute_421 = None
    view_514: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_56, [1, 4, 512, 64]);  bmm_56 = None
    view_515: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_57, [1, 4, 512, 512]);  bmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_27: "f32[1, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_35, torch.float32);  getitem_35 = None
    mul_365: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_27, 1.1111111111111112);  convert_element_type_27 = None
    mul_366: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_515, mul_365);  view_515 = mul_365 = None
    clone_54: "f32[1, 4, 512, 512]" = torch.ops.aten.clone.default(mul_366, memory_format = torch.contiguous_format);  mul_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_22: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    mul_367: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(clone_54, alias_22);  clone_54 = None
    sum_154: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_367, [-1], True)
    mul_368: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(alias_22, sum_154);  alias_22 = sum_154 = None
    sub_107: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(mul_367, mul_368);  mul_367 = mul_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_53: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(sub_107, 8.0);  sub_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_516: "f32[4, 512, 512]" = torch.ops.aten.view.default(div_53, [4, 512, 512]);  div_53 = None
    permute_422: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_77, [0, 2, 1]);  view_77 = None
    bmm_58: "f32[4, 64, 512]" = torch.ops.aten.bmm.default(permute_422, view_516);  permute_422 = None
    permute_423: "f32[4, 512, 64]" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
    bmm_59: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_516, permute_423);  view_516 = permute_423 = None
    view_517: "f32[1, 4, 64, 512]" = torch.ops.aten.view.default(bmm_58, [1, 4, 64, 512]);  bmm_58 = None
    view_518: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_59, [1, 4, 512, 64]);  bmm_59 = None
    permute_424: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_517, [0, 1, 3, 2]);  view_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_425: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_518, [0, 2, 1, 3]);  view_518 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_55: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_425, memory_format = torch.contiguous_format);  permute_425 = None
    view_519: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_55, [1, 512, 256]);  clone_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_426: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_514, [0, 2, 1, 3]);  view_514 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_56: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_426, memory_format = torch.contiguous_format);  permute_426 = None
    view_520: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_56, [1, 512, 256]);  clone_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_521: "f32[512, 256]" = torch.ops.aten.view.default(view_520, [512, 256]);  view_520 = None
    permute_427: "f32[256, 256]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
    mm_106: "f32[512, 256]" = torch.ops.aten.mm.default(view_521, permute_427);  permute_427 = None
    permute_428: "f32[256, 512]" = torch.ops.aten.permute.default(view_521, [1, 0])
    mm_107: "f32[256, 256]" = torch.ops.aten.mm.default(permute_428, view_73);  permute_428 = view_73 = None
    permute_429: "f32[256, 256]" = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
    sum_155: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_521, [0], True);  view_521 = None
    view_522: "f32[256]" = torch.ops.aten.view.default(sum_155, [256]);  sum_155 = None
    permute_430: "f32[256, 256]" = torch.ops.aten.permute.default(permute_429, [1, 0]);  permute_429 = None
    view_523: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_106, [1, 512, 256]);  mm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_157: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_361, view_523);  mul_361 = view_523 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_431: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(permute_424, [0, 2, 1, 3]);  permute_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_524: "f32[1, 512, 256]" = torch.ops.aten.view.default(permute_431, [1, 512, 256]);  permute_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_525: "f32[512, 256]" = torch.ops.aten.view.default(view_524, [512, 256]);  view_524 = None
    permute_432: "f32[256, 256]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
    mm_108: "f32[512, 256]" = torch.ops.aten.mm.default(view_525, permute_432);  permute_432 = None
    permute_433: "f32[256, 512]" = torch.ops.aten.permute.default(view_525, [1, 0])
    mm_109: "f32[256, 256]" = torch.ops.aten.mm.default(permute_433, view_70);  permute_433 = view_70 = None
    permute_434: "f32[256, 256]" = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
    sum_156: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_525, [0], True);  view_525 = None
    view_526: "f32[256]" = torch.ops.aten.view.default(sum_156, [256]);  sum_156 = None
    permute_435: "f32[256, 256]" = torch.ops.aten.permute.default(permute_434, [1, 0]);  permute_434 = None
    view_527: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_108, [1, 512, 256]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_158: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_157, view_527);  add_157 = view_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_528: "f32[512, 256]" = torch.ops.aten.view.default(view_519, [512, 256]);  view_519 = None
    permute_436: "f32[256, 256]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    mm_110: "f32[512, 256]" = torch.ops.aten.mm.default(view_528, permute_436);  permute_436 = None
    permute_437: "f32[256, 512]" = torch.ops.aten.permute.default(view_528, [1, 0])
    mm_111: "f32[256, 256]" = torch.ops.aten.mm.default(permute_437, view_68);  permute_437 = view_68 = None
    permute_438: "f32[256, 256]" = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
    sum_157: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_528, [0], True);  view_528 = None
    view_529: "f32[256]" = torch.ops.aten.view.default(sum_157, [256]);  sum_157 = None
    permute_439: "f32[256, 256]" = torch.ops.aten.permute.default(permute_438, [1, 0]);  permute_438 = None
    view_530: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_110, [1, 512, 256]);  mm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_159: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_158, view_530);  add_158 = view_530 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_108: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_25, getitem_33);  add_25 = getitem_33 = None
    mul_369: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_108, rsqrt_6);  sub_108 = None
    mul_370: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_159, primals_54);  primals_54 = None
    mul_371: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_370, 256)
    sum_158: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_370, [2], True)
    mul_372: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_370, mul_369);  mul_370 = None
    sum_159: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_372, [2], True);  mul_372 = None
    mul_373: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_369, sum_159);  sum_159 = None
    sub_109: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_371, sum_158);  mul_371 = sum_158 = None
    sub_110: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_109, mul_373);  sub_109 = mul_373 = None
    div_54: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 256);  rsqrt_6 = None
    mul_374: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_54, sub_110);  div_54 = sub_110 = None
    mul_375: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_159, mul_369);  mul_369 = None
    sum_160: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_375, [0, 1]);  mul_375 = None
    sum_161: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_159, [0, 1]);  add_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_28: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_31, torch.float32);  getitem_31 = None
    mul_376: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_28, 1.1111111111111112);  convert_element_type_28 = None
    mul_377: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_374, mul_376);  mul_376 = None
    clone_57: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_377, memory_format = torch.contiguous_format);  mul_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_531: "f32[512, 256]" = torch.ops.aten.view.default(clone_57, [512, 256]);  clone_57 = None
    permute_440: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    mm_112: "f32[512, 1024]" = torch.ops.aten.mm.default(view_531, permute_440);  permute_440 = None
    permute_441: "f32[256, 512]" = torch.ops.aten.permute.default(view_531, [1, 0])
    mm_113: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_441, view_66);  permute_441 = view_66 = None
    permute_442: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_113, [1, 0]);  mm_113 = None
    sum_162: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_531, [0], True);  view_531 = None
    view_532: "f32[256]" = torch.ops.aten.view.default(sum_162, [256]);  sum_162 = None
    permute_443: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_442, [1, 0]);  permute_442 = None
    view_533: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_112, [1, 512, 1024]);  mm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_378: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_65, 0.7071067811865476)
    erf_23: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_378);  mul_378 = None
    add_160: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_379: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_160, 0.5);  add_160 = None
    mul_380: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_65, view_65)
    mul_381: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_380, -0.5);  mul_380 = None
    exp_24: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_381);  mul_381 = None
    mul_382: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
    mul_383: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_65, mul_382);  view_65 = mul_382 = None
    add_161: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_379, mul_383);  mul_379 = mul_383 = None
    mul_384: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_533, add_161);  view_533 = add_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_534: "f32[512, 1024]" = torch.ops.aten.view.default(mul_384, [512, 1024]);  mul_384 = None
    permute_444: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    mm_114: "f32[512, 256]" = torch.ops.aten.mm.default(view_534, permute_444);  permute_444 = None
    permute_445: "f32[1024, 512]" = torch.ops.aten.permute.default(view_534, [1, 0])
    mm_115: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_445, view_64);  permute_445 = view_64 = None
    permute_446: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
    sum_163: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_534, [0], True);  view_534 = None
    view_535: "f32[1024]" = torch.ops.aten.view.default(sum_163, [1024]);  sum_163 = None
    permute_447: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_446, [1, 0]);  permute_446 = None
    view_536: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_114, [1, 512, 256]);  mm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_162: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_374, view_536);  mul_374 = view_536 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_111: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_21, getitem_29);  add_21 = getitem_29 = None
    mul_385: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_111, rsqrt_5);  sub_111 = None
    mul_386: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_162, primals_48);  primals_48 = None
    mul_387: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_386, 256)
    sum_164: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_386, [2], True)
    mul_388: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_386, mul_385);  mul_386 = None
    sum_165: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_388, [2], True);  mul_388 = None
    mul_389: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_385, sum_165);  sum_165 = None
    sub_112: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_387, sum_164);  mul_387 = sum_164 = None
    sub_113: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_112, mul_389);  sub_112 = mul_389 = None
    div_55: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 256);  rsqrt_5 = None
    mul_390: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_55, sub_113);  div_55 = sub_113 = None
    mul_391: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_162, mul_385);  mul_385 = None
    sum_166: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_391, [0, 1]);  mul_391 = None
    sum_167: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_162, [0, 1]);  add_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_29: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_27, torch.float32);  getitem_27 = None
    mul_392: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_29, 1.1111111111111112);  convert_element_type_29 = None
    mul_393: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_390, mul_392);  mul_392 = None
    clone_58: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_393, memory_format = torch.contiguous_format);  mul_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_537: "f32[512, 256]" = torch.ops.aten.view.default(clone_58, [512, 256]);  clone_58 = None
    permute_448: "f32[256, 256]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
    mm_116: "f32[512, 256]" = torch.ops.aten.mm.default(view_537, permute_448);  permute_448 = None
    permute_449: "f32[256, 512]" = torch.ops.aten.permute.default(view_537, [1, 0])
    mm_117: "f32[256, 256]" = torch.ops.aten.mm.default(permute_449, view_62);  permute_449 = view_62 = None
    permute_450: "f32[256, 256]" = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
    sum_168: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_537, [0], True);  view_537 = None
    view_538: "f32[256]" = torch.ops.aten.view.default(sum_168, [256]);  sum_168 = None
    permute_451: "f32[256, 256]" = torch.ops.aten.permute.default(permute_450, [1, 0]);  permute_450 = None
    view_539: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_116, [1, 512, 256]);  mm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_540: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_539, [1, 512, 4, 64]);  view_539 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_452: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_540, [0, 2, 1, 3]);  view_540 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_541: "f32[4, 512, 64]" = torch.ops.aten.view.default(permute_452, [4, 512, 64]);  permute_452 = None
    permute_453: "f32[4, 512, 512]" = torch.ops.aten.permute.default(view_58, [0, 2, 1]);  view_58 = None
    bmm_60: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(permute_453, view_541);  permute_453 = None
    permute_454: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_59, [0, 2, 1]);  view_59 = None
    bmm_61: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_541, permute_454);  view_541 = permute_454 = None
    view_542: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_60, [1, 4, 512, 64]);  bmm_60 = None
    view_543: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_61, [1, 4, 512, 512]);  bmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_30: "f32[1, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_25, torch.float32);  getitem_25 = None
    mul_394: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_30, 1.1111111111111112);  convert_element_type_30 = None
    mul_395: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_543, mul_394);  view_543 = mul_394 = None
    clone_59: "f32[1, 4, 512, 512]" = torch.ops.aten.clone.default(mul_395, memory_format = torch.contiguous_format);  mul_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_23: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    mul_396: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(clone_59, alias_23);  clone_59 = None
    sum_169: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_396, [-1], True)
    mul_397: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(alias_23, sum_169);  alias_23 = sum_169 = None
    sub_114: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(mul_396, mul_397);  mul_396 = mul_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_56: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(sub_114, 8.0);  sub_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_544: "f32[4, 512, 512]" = torch.ops.aten.view.default(div_56, [4, 512, 512]);  div_56 = None
    permute_455: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_55, [0, 2, 1]);  view_55 = None
    bmm_62: "f32[4, 64, 512]" = torch.ops.aten.bmm.default(permute_455, view_544);  permute_455 = None
    permute_456: "f32[4, 512, 64]" = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None
    bmm_63: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_544, permute_456);  view_544 = permute_456 = None
    view_545: "f32[1, 4, 64, 512]" = torch.ops.aten.view.default(bmm_62, [1, 4, 64, 512]);  bmm_62 = None
    view_546: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_63, [1, 4, 512, 64]);  bmm_63 = None
    permute_457: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_545, [0, 1, 3, 2]);  view_545 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_458: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_546, [0, 2, 1, 3]);  view_546 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_60: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_458, memory_format = torch.contiguous_format);  permute_458 = None
    view_547: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_60, [1, 512, 256]);  clone_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_459: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_542, [0, 2, 1, 3]);  view_542 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_61: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_459, memory_format = torch.contiguous_format);  permute_459 = None
    view_548: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_61, [1, 512, 256]);  clone_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_549: "f32[512, 256]" = torch.ops.aten.view.default(view_548, [512, 256]);  view_548 = None
    permute_460: "f32[256, 256]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
    mm_118: "f32[512, 256]" = torch.ops.aten.mm.default(view_549, permute_460);  permute_460 = None
    permute_461: "f32[256, 512]" = torch.ops.aten.permute.default(view_549, [1, 0])
    mm_119: "f32[256, 256]" = torch.ops.aten.mm.default(permute_461, view_51);  permute_461 = view_51 = None
    permute_462: "f32[256, 256]" = torch.ops.aten.permute.default(mm_119, [1, 0]);  mm_119 = None
    sum_170: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_549, [0], True);  view_549 = None
    view_550: "f32[256]" = torch.ops.aten.view.default(sum_170, [256]);  sum_170 = None
    permute_463: "f32[256, 256]" = torch.ops.aten.permute.default(permute_462, [1, 0]);  permute_462 = None
    view_551: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_118, [1, 512, 256]);  mm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_163: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_390, view_551);  mul_390 = view_551 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_464: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(permute_457, [0, 2, 1, 3]);  permute_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_552: "f32[1, 512, 256]" = torch.ops.aten.view.default(permute_464, [1, 512, 256]);  permute_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_553: "f32[512, 256]" = torch.ops.aten.view.default(view_552, [512, 256]);  view_552 = None
    permute_465: "f32[256, 256]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
    mm_120: "f32[512, 256]" = torch.ops.aten.mm.default(view_553, permute_465);  permute_465 = None
    permute_466: "f32[256, 512]" = torch.ops.aten.permute.default(view_553, [1, 0])
    mm_121: "f32[256, 256]" = torch.ops.aten.mm.default(permute_466, view_48);  permute_466 = view_48 = None
    permute_467: "f32[256, 256]" = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
    sum_171: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_553, [0], True);  view_553 = None
    view_554: "f32[256]" = torch.ops.aten.view.default(sum_171, [256]);  sum_171 = None
    permute_468: "f32[256, 256]" = torch.ops.aten.permute.default(permute_467, [1, 0]);  permute_467 = None
    view_555: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_120, [1, 512, 256]);  mm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_164: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_163, view_555);  add_163 = view_555 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_556: "f32[512, 256]" = torch.ops.aten.view.default(view_547, [512, 256]);  view_547 = None
    permute_469: "f32[256, 256]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    mm_122: "f32[512, 256]" = torch.ops.aten.mm.default(view_556, permute_469);  permute_469 = None
    permute_470: "f32[256, 512]" = torch.ops.aten.permute.default(view_556, [1, 0])
    mm_123: "f32[256, 256]" = torch.ops.aten.mm.default(permute_470, view_46);  permute_470 = view_46 = None
    permute_471: "f32[256, 256]" = torch.ops.aten.permute.default(mm_123, [1, 0]);  mm_123 = None
    sum_172: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_556, [0], True);  view_556 = None
    view_557: "f32[256]" = torch.ops.aten.view.default(sum_172, [256]);  sum_172 = None
    permute_472: "f32[256, 256]" = torch.ops.aten.permute.default(permute_471, [1, 0]);  permute_471 = None
    view_558: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_122, [1, 512, 256]);  mm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_165: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_164, view_558);  add_164 = view_558 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_115: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_17, getitem_23);  add_17 = getitem_23 = None
    mul_398: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_115, rsqrt_4);  sub_115 = None
    mul_399: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_165, primals_38);  primals_38 = None
    mul_400: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_399, 256)
    sum_173: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_399, [2], True)
    mul_401: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_399, mul_398);  mul_399 = None
    sum_174: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_401, [2], True);  mul_401 = None
    mul_402: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_398, sum_174);  sum_174 = None
    sub_116: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_400, sum_173);  mul_400 = sum_173 = None
    sub_117: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_116, mul_402);  sub_116 = mul_402 = None
    div_57: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 256);  rsqrt_4 = None
    mul_403: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_57, sub_117);  div_57 = sub_117 = None
    mul_404: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_165, mul_398);  mul_398 = None
    sum_175: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_404, [0, 1]);  mul_404 = None
    sum_176: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_165, [0, 1]);  add_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_31: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_21, torch.float32);  getitem_21 = None
    mul_405: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_31, 1.1111111111111112);  convert_element_type_31 = None
    mul_406: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_403, mul_405);  mul_405 = None
    clone_62: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_406, memory_format = torch.contiguous_format);  mul_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_559: "f32[512, 256]" = torch.ops.aten.view.default(clone_62, [512, 256]);  clone_62 = None
    permute_473: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    mm_124: "f32[512, 1024]" = torch.ops.aten.mm.default(view_559, permute_473);  permute_473 = None
    permute_474: "f32[256, 512]" = torch.ops.aten.permute.default(view_559, [1, 0])
    mm_125: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_474, view_44);  permute_474 = view_44 = None
    permute_475: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_125, [1, 0]);  mm_125 = None
    sum_177: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_559, [0], True);  view_559 = None
    view_560: "f32[256]" = torch.ops.aten.view.default(sum_177, [256]);  sum_177 = None
    permute_476: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_475, [1, 0]);  permute_475 = None
    view_561: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_124, [1, 512, 1024]);  mm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_407: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_43, 0.7071067811865476)
    erf_24: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_407);  mul_407 = None
    add_166: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
    mul_408: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_166, 0.5);  add_166 = None
    mul_409: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_43, view_43)
    mul_410: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_409, -0.5);  mul_409 = None
    exp_25: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_410);  mul_410 = None
    mul_411: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
    mul_412: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_43, mul_411);  view_43 = mul_411 = None
    add_167: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_408, mul_412);  mul_408 = mul_412 = None
    mul_413: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_561, add_167);  view_561 = add_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_562: "f32[512, 1024]" = torch.ops.aten.view.default(mul_413, [512, 1024]);  mul_413 = None
    permute_477: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    mm_126: "f32[512, 256]" = torch.ops.aten.mm.default(view_562, permute_477);  permute_477 = None
    permute_478: "f32[1024, 512]" = torch.ops.aten.permute.default(view_562, [1, 0])
    mm_127: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_478, view_42);  permute_478 = view_42 = None
    permute_479: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_127, [1, 0]);  mm_127 = None
    sum_178: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_562, [0], True);  view_562 = None
    view_563: "f32[1024]" = torch.ops.aten.view.default(sum_178, [1024]);  sum_178 = None
    permute_480: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_479, [1, 0]);  permute_479 = None
    view_564: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_126, [1, 512, 256]);  mm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_168: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_403, view_564);  mul_403 = view_564 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_118: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_13, getitem_19);  add_13 = getitem_19 = None
    mul_414: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_118, rsqrt_3);  sub_118 = None
    mul_415: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_168, primals_32);  primals_32 = None
    mul_416: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_415, 256)
    sum_179: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_415, [2], True)
    mul_417: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_415, mul_414);  mul_415 = None
    sum_180: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_417, [2], True);  mul_417 = None
    mul_418: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_414, sum_180);  sum_180 = None
    sub_119: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_416, sum_179);  mul_416 = sum_179 = None
    sub_120: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_119, mul_418);  sub_119 = mul_418 = None
    div_58: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 256);  rsqrt_3 = None
    mul_419: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_58, sub_120);  div_58 = sub_120 = None
    mul_420: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_168, mul_414);  mul_414 = None
    sum_181: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_420, [0, 1]);  mul_420 = None
    sum_182: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_168, [0, 1]);  add_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_32: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_17, torch.float32);  getitem_17 = None
    mul_421: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_32, 1.1111111111111112);  convert_element_type_32 = None
    mul_422: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_419, mul_421);  mul_421 = None
    clone_63: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_422, memory_format = torch.contiguous_format);  mul_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_565: "f32[512, 256]" = torch.ops.aten.view.default(clone_63, [512, 256]);  clone_63 = None
    permute_481: "f32[256, 256]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    mm_128: "f32[512, 256]" = torch.ops.aten.mm.default(view_565, permute_481);  permute_481 = None
    permute_482: "f32[256, 512]" = torch.ops.aten.permute.default(view_565, [1, 0])
    mm_129: "f32[256, 256]" = torch.ops.aten.mm.default(permute_482, view_40);  permute_482 = view_40 = None
    permute_483: "f32[256, 256]" = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
    sum_183: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_565, [0], True);  view_565 = None
    view_566: "f32[256]" = torch.ops.aten.view.default(sum_183, [256]);  sum_183 = None
    permute_484: "f32[256, 256]" = torch.ops.aten.permute.default(permute_483, [1, 0]);  permute_483 = None
    view_567: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_128, [1, 512, 256]);  mm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_568: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_567, [1, 512, 4, 64]);  view_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_485: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_568, [0, 2, 1, 3]);  view_568 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_569: "f32[4, 512, 64]" = torch.ops.aten.view.default(permute_485, [4, 512, 64]);  permute_485 = None
    permute_486: "f32[4, 512, 512]" = torch.ops.aten.permute.default(view_36, [0, 2, 1]);  view_36 = None
    bmm_64: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(permute_486, view_569);  permute_486 = None
    permute_487: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_37, [0, 2, 1]);  view_37 = None
    bmm_65: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_569, permute_487);  view_569 = permute_487 = None
    view_570: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_64, [1, 4, 512, 64]);  bmm_64 = None
    view_571: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_65, [1, 4, 512, 512]);  bmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_33: "f32[1, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_15, torch.float32);  getitem_15 = None
    mul_423: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_33, 1.1111111111111112);  convert_element_type_33 = None
    mul_424: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_571, mul_423);  view_571 = mul_423 = None
    clone_64: "f32[1, 4, 512, 512]" = torch.ops.aten.clone.default(mul_424, memory_format = torch.contiguous_format);  mul_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_24: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    mul_425: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(clone_64, alias_24);  clone_64 = None
    sum_184: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_425, [-1], True)
    mul_426: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(alias_24, sum_184);  alias_24 = sum_184 = None
    sub_121: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(mul_425, mul_426);  mul_425 = mul_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_59: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(sub_121, 8.0);  sub_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_572: "f32[4, 512, 512]" = torch.ops.aten.view.default(div_59, [4, 512, 512]);  div_59 = None
    permute_488: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_33, [0, 2, 1]);  view_33 = None
    bmm_66: "f32[4, 64, 512]" = torch.ops.aten.bmm.default(permute_488, view_572);  permute_488 = None
    permute_489: "f32[4, 512, 64]" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
    bmm_67: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_572, permute_489);  view_572 = permute_489 = None
    view_573: "f32[1, 4, 64, 512]" = torch.ops.aten.view.default(bmm_66, [1, 4, 64, 512]);  bmm_66 = None
    view_574: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_67, [1, 4, 512, 64]);  bmm_67 = None
    permute_490: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_573, [0, 1, 3, 2]);  view_573 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_491: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_574, [0, 2, 1, 3]);  view_574 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_65: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_491, memory_format = torch.contiguous_format);  permute_491 = None
    view_575: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_65, [1, 512, 256]);  clone_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_492: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_570, [0, 2, 1, 3]);  view_570 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_66: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_492, memory_format = torch.contiguous_format);  permute_492 = None
    view_576: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_66, [1, 512, 256]);  clone_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_577: "f32[512, 256]" = torch.ops.aten.view.default(view_576, [512, 256]);  view_576 = None
    permute_493: "f32[256, 256]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
    mm_130: "f32[512, 256]" = torch.ops.aten.mm.default(view_577, permute_493);  permute_493 = None
    permute_494: "f32[256, 512]" = torch.ops.aten.permute.default(view_577, [1, 0])
    mm_131: "f32[256, 256]" = torch.ops.aten.mm.default(permute_494, view_29);  permute_494 = view_29 = None
    permute_495: "f32[256, 256]" = torch.ops.aten.permute.default(mm_131, [1, 0]);  mm_131 = None
    sum_185: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_577, [0], True);  view_577 = None
    view_578: "f32[256]" = torch.ops.aten.view.default(sum_185, [256]);  sum_185 = None
    permute_496: "f32[256, 256]" = torch.ops.aten.permute.default(permute_495, [1, 0]);  permute_495 = None
    view_579: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_130, [1, 512, 256]);  mm_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_169: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_419, view_579);  mul_419 = view_579 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_497: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(permute_490, [0, 2, 1, 3]);  permute_490 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_580: "f32[1, 512, 256]" = torch.ops.aten.view.default(permute_497, [1, 512, 256]);  permute_497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_581: "f32[512, 256]" = torch.ops.aten.view.default(view_580, [512, 256]);  view_580 = None
    permute_498: "f32[256, 256]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    mm_132: "f32[512, 256]" = torch.ops.aten.mm.default(view_581, permute_498);  permute_498 = None
    permute_499: "f32[256, 512]" = torch.ops.aten.permute.default(view_581, [1, 0])
    mm_133: "f32[256, 256]" = torch.ops.aten.mm.default(permute_499, view_26);  permute_499 = view_26 = None
    permute_500: "f32[256, 256]" = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
    sum_186: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_581, [0], True);  view_581 = None
    view_582: "f32[256]" = torch.ops.aten.view.default(sum_186, [256]);  sum_186 = None
    permute_501: "f32[256, 256]" = torch.ops.aten.permute.default(permute_500, [1, 0]);  permute_500 = None
    view_583: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_132, [1, 512, 256]);  mm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_170: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_169, view_583);  add_169 = view_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_584: "f32[512, 256]" = torch.ops.aten.view.default(view_575, [512, 256]);  view_575 = None
    permute_502: "f32[256, 256]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    mm_134: "f32[512, 256]" = torch.ops.aten.mm.default(view_584, permute_502);  permute_502 = None
    permute_503: "f32[256, 512]" = torch.ops.aten.permute.default(view_584, [1, 0])
    mm_135: "f32[256, 256]" = torch.ops.aten.mm.default(permute_503, view_24);  permute_503 = view_24 = None
    permute_504: "f32[256, 256]" = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
    sum_187: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_584, [0], True);  view_584 = None
    view_585: "f32[256]" = torch.ops.aten.view.default(sum_187, [256]);  sum_187 = None
    permute_505: "f32[256, 256]" = torch.ops.aten.permute.default(permute_504, [1, 0]);  permute_504 = None
    view_586: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_134, [1, 512, 256]);  mm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_171: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_170, view_586);  add_170 = view_586 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_122: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_9, getitem_13);  add_9 = getitem_13 = None
    mul_427: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_122, rsqrt_2);  sub_122 = None
    mul_428: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_171, primals_22);  primals_22 = None
    mul_429: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_428, 256)
    sum_188: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_428, [2], True)
    mul_430: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_428, mul_427);  mul_428 = None
    sum_189: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_430, [2], True);  mul_430 = None
    mul_431: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_427, sum_189);  sum_189 = None
    sub_123: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_429, sum_188);  mul_429 = sum_188 = None
    sub_124: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_123, mul_431);  sub_123 = mul_431 = None
    div_60: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 256);  rsqrt_2 = None
    mul_432: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_60, sub_124);  div_60 = sub_124 = None
    mul_433: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_171, mul_427);  mul_427 = None
    sum_190: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_433, [0, 1]);  mul_433 = None
    sum_191: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_171, [0, 1]);  add_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_34: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_11, torch.float32);  getitem_11 = None
    mul_434: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_34, 1.1111111111111112);  convert_element_type_34 = None
    mul_435: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_432, mul_434);  mul_434 = None
    clone_67: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_435, memory_format = torch.contiguous_format);  mul_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_587: "f32[512, 256]" = torch.ops.aten.view.default(clone_67, [512, 256]);  clone_67 = None
    permute_506: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    mm_136: "f32[512, 1024]" = torch.ops.aten.mm.default(view_587, permute_506);  permute_506 = None
    permute_507: "f32[256, 512]" = torch.ops.aten.permute.default(view_587, [1, 0])
    mm_137: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_507, view_22);  permute_507 = view_22 = None
    permute_508: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_137, [1, 0]);  mm_137 = None
    sum_192: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_587, [0], True);  view_587 = None
    view_588: "f32[256]" = torch.ops.aten.view.default(sum_192, [256]);  sum_192 = None
    permute_509: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_508, [1, 0]);  permute_508 = None
    view_589: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_136, [1, 512, 1024]);  mm_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_436: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_21, 0.7071067811865476)
    erf_25: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_436);  mul_436 = None
    add_172: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
    mul_437: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_172, 0.5);  add_172 = None
    mul_438: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_21, view_21)
    mul_439: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_438, -0.5);  mul_438 = None
    exp_26: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_439);  mul_439 = None
    mul_440: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
    mul_441: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_21, mul_440);  view_21 = mul_440 = None
    add_173: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_437, mul_441);  mul_437 = mul_441 = None
    mul_442: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_589, add_173);  view_589 = add_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_590: "f32[512, 1024]" = torch.ops.aten.view.default(mul_442, [512, 1024]);  mul_442 = None
    permute_510: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    mm_138: "f32[512, 256]" = torch.ops.aten.mm.default(view_590, permute_510);  permute_510 = None
    permute_511: "f32[1024, 512]" = torch.ops.aten.permute.default(view_590, [1, 0])
    mm_139: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_511, view_20);  permute_511 = view_20 = None
    permute_512: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
    sum_193: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_590, [0], True);  view_590 = None
    view_591: "f32[1024]" = torch.ops.aten.view.default(sum_193, [1024]);  sum_193 = None
    permute_513: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_512, [1, 0]);  permute_512 = None
    view_592: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_138, [1, 512, 256]);  mm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_174: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_432, view_592);  mul_432 = view_592 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_125: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_5, getitem_9);  add_5 = getitem_9 = None
    mul_443: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_125, rsqrt_1);  sub_125 = None
    mul_444: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_174, primals_16);  primals_16 = None
    mul_445: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_444, 256)
    sum_194: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_444, [2], True)
    mul_446: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_444, mul_443);  mul_444 = None
    sum_195: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_446, [2], True);  mul_446 = None
    mul_447: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_443, sum_195);  sum_195 = None
    sub_126: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_445, sum_194);  mul_445 = sum_194 = None
    sub_127: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_126, mul_447);  sub_126 = mul_447 = None
    div_61: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 256);  rsqrt_1 = None
    mul_448: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_61, sub_127);  div_61 = sub_127 = None
    mul_449: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_174, mul_443);  mul_443 = None
    sum_196: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_449, [0, 1]);  mul_449 = None
    sum_197: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_174, [0, 1]);  add_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_35: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_7, torch.float32);  getitem_7 = None
    mul_450: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_35, 1.1111111111111112);  convert_element_type_35 = None
    mul_451: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_448, mul_450);  mul_450 = None
    clone_68: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_451, memory_format = torch.contiguous_format);  mul_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_593: "f32[512, 256]" = torch.ops.aten.view.default(clone_68, [512, 256]);  clone_68 = None
    permute_514: "f32[256, 256]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    mm_140: "f32[512, 256]" = torch.ops.aten.mm.default(view_593, permute_514);  permute_514 = None
    permute_515: "f32[256, 512]" = torch.ops.aten.permute.default(view_593, [1, 0])
    mm_141: "f32[256, 256]" = torch.ops.aten.mm.default(permute_515, view_18);  permute_515 = view_18 = None
    permute_516: "f32[256, 256]" = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
    sum_198: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_593, [0], True);  view_593 = None
    view_594: "f32[256]" = torch.ops.aten.view.default(sum_198, [256]);  sum_198 = None
    permute_517: "f32[256, 256]" = torch.ops.aten.permute.default(permute_516, [1, 0]);  permute_516 = None
    view_595: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_140, [1, 512, 256]);  mm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_596: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_595, [1, 512, 4, 64]);  view_595 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_518: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_596, [0, 2, 1, 3]);  view_596 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_597: "f32[4, 512, 64]" = torch.ops.aten.view.default(permute_518, [4, 512, 64]);  permute_518 = None
    permute_519: "f32[4, 512, 512]" = torch.ops.aten.permute.default(view_14, [0, 2, 1]);  view_14 = None
    bmm_68: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(permute_519, view_597);  permute_519 = None
    permute_520: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_15, [0, 2, 1]);  view_15 = None
    bmm_69: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_597, permute_520);  view_597 = permute_520 = None
    view_598: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_68, [1, 4, 512, 64]);  bmm_68 = None
    view_599: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_69, [1, 4, 512, 512]);  bmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_36: "f32[1, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_5, torch.float32);  getitem_5 = None
    mul_452: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_36, 1.1111111111111112);  convert_element_type_36 = None
    mul_453: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_599, mul_452);  view_599 = mul_452 = None
    clone_69: "f32[1, 4, 512, 512]" = torch.ops.aten.clone.default(mul_453, memory_format = torch.contiguous_format);  mul_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_25: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(alias);  alias = None
    mul_454: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(clone_69, alias_25);  clone_69 = None
    sum_199: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_454, [-1], True)
    mul_455: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(alias_25, sum_199);  alias_25 = sum_199 = None
    sub_128: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(mul_454, mul_455);  mul_454 = mul_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_62: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(sub_128, 8.0);  sub_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_600: "f32[4, 512, 512]" = torch.ops.aten.view.default(div_62, [4, 512, 512]);  div_62 = None
    permute_521: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
    bmm_70: "f32[4, 64, 512]" = torch.ops.aten.bmm.default(permute_521, view_600);  permute_521 = None
    permute_522: "f32[4, 512, 64]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
    bmm_71: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_600, permute_522);  view_600 = permute_522 = None
    view_601: "f32[1, 4, 64, 512]" = torch.ops.aten.view.default(bmm_70, [1, 4, 64, 512]);  bmm_70 = None
    view_602: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_71, [1, 4, 512, 64]);  bmm_71 = None
    permute_523: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_601, [0, 1, 3, 2]);  view_601 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_524: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_602, [0, 2, 1, 3]);  view_602 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_70: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_524, memory_format = torch.contiguous_format);  permute_524 = None
    view_603: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_70, [1, 512, 256]);  clone_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_525: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_598, [0, 2, 1, 3]);  view_598 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_71: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_525, memory_format = torch.contiguous_format);  permute_525 = None
    view_604: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_71, [1, 512, 256]);  clone_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_605: "f32[512, 256]" = torch.ops.aten.view.default(view_604, [512, 256]);  view_604 = None
    permute_526: "f32[256, 256]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
    mm_142: "f32[512, 256]" = torch.ops.aten.mm.default(view_605, permute_526);  permute_526 = None
    permute_527: "f32[256, 512]" = torch.ops.aten.permute.default(view_605, [1, 0])
    mm_143: "f32[256, 256]" = torch.ops.aten.mm.default(permute_527, view_7);  permute_527 = view_7 = None
    permute_528: "f32[256, 256]" = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
    sum_200: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_605, [0], True);  view_605 = None
    view_606: "f32[256]" = torch.ops.aten.view.default(sum_200, [256]);  sum_200 = None
    permute_529: "f32[256, 256]" = torch.ops.aten.permute.default(permute_528, [1, 0]);  permute_528 = None
    view_607: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_142, [1, 512, 256]);  mm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_175: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_448, view_607);  mul_448 = view_607 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_530: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(permute_523, [0, 2, 1, 3]);  permute_523 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_608: "f32[1, 512, 256]" = torch.ops.aten.view.default(permute_530, [1, 512, 256]);  permute_530 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_609: "f32[512, 256]" = torch.ops.aten.view.default(view_608, [512, 256]);  view_608 = None
    permute_531: "f32[256, 256]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    mm_144: "f32[512, 256]" = torch.ops.aten.mm.default(view_609, permute_531);  permute_531 = None
    permute_532: "f32[256, 512]" = torch.ops.aten.permute.default(view_609, [1, 0])
    mm_145: "f32[256, 256]" = torch.ops.aten.mm.default(permute_532, view_4);  permute_532 = view_4 = None
    permute_533: "f32[256, 256]" = torch.ops.aten.permute.default(mm_145, [1, 0]);  mm_145 = None
    sum_201: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_609, [0], True);  view_609 = None
    view_610: "f32[256]" = torch.ops.aten.view.default(sum_201, [256]);  sum_201 = None
    permute_534: "f32[256, 256]" = torch.ops.aten.permute.default(permute_533, [1, 0]);  permute_533 = None
    view_611: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_144, [1, 512, 256]);  mm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_176: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_175, view_611);  add_175 = view_611 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_612: "f32[512, 256]" = torch.ops.aten.view.default(view_603, [512, 256]);  view_603 = None
    permute_535: "f32[256, 256]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    mm_146: "f32[512, 256]" = torch.ops.aten.mm.default(view_612, permute_535);  permute_535 = None
    permute_536: "f32[256, 512]" = torch.ops.aten.permute.default(view_612, [1, 0])
    mm_147: "f32[256, 256]" = torch.ops.aten.mm.default(permute_536, view_2);  permute_536 = view_2 = None
    permute_537: "f32[256, 256]" = torch.ops.aten.permute.default(mm_147, [1, 0]);  mm_147 = None
    sum_202: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_612, [0], True);  view_612 = None
    view_613: "f32[256]" = torch.ops.aten.view.default(sum_202, [256]);  sum_202 = None
    permute_538: "f32[256, 256]" = torch.ops.aten.permute.default(permute_537, [1, 0]);  permute_537 = None
    view_614: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_146, [1, 512, 256]);  mm_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_177: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_176, view_614);  add_176 = view_614 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:918, code: hidden_states = self.embeddings_project(hidden_states)
    view_615: "f32[512, 256]" = torch.ops.aten.view.default(add_177, [512, 256]);  add_177 = None
    permute_539: "f32[256, 128]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    mm_148: "f32[512, 128]" = torch.ops.aten.mm.default(view_615, permute_539);  permute_539 = None
    permute_540: "f32[256, 512]" = torch.ops.aten.permute.default(view_615, [1, 0])
    mm_149: "f32[256, 128]" = torch.ops.aten.mm.default(permute_540, view);  permute_540 = view = None
    permute_541: "f32[128, 256]" = torch.ops.aten.permute.default(mm_149, [1, 0]);  mm_149 = None
    sum_203: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_615, [0], True);  view_615 = None
    view_616: "f32[256]" = torch.ops.aten.view.default(sum_203, [256]);  sum_203 = None
    permute_542: "f32[256, 128]" = torch.ops.aten.permute.default(permute_541, [1, 0]);  permute_541 = None
    view_617: "f32[1, 512, 128]" = torch.ops.aten.view.default(mm_148, [1, 512, 128]);  mm_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:211, code: embeddings = self.dropout(embeddings)
    convert_element_type_37: "f32[1, 512, 128]" = torch.ops.prims.convert_element_type.default(getitem_3, torch.float32);  getitem_3 = None
    mul_456: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_37, 1.1111111111111112);  convert_element_type_37 = None
    mul_457: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(view_617, mul_456);  view_617 = mul_456 = None
    clone_72: "f32[1, 512, 128]" = torch.ops.aten.clone.default(mul_457, memory_format = torch.contiguous_format);  mul_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:210, code: embeddings = self.LayerNorm(embeddings)
    sub_129: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
    mul_458: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(sub_129, rsqrt);  sub_129 = None
    mul_459: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(clone_72, primals_4);  primals_4 = None
    mul_460: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_459, 128)
    sum_204: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_459, [2], True)
    mul_461: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_459, mul_458);  mul_459 = None
    sum_205: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_461, [2], True);  mul_461 = None
    mul_462: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_458, sum_205);  sum_205 = None
    sub_130: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(mul_460, sum_204);  mul_460 = sum_204 = None
    sub_131: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(sub_130, mul_462);  sub_130 = mul_462 = None
    div_63: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 128);  rsqrt = None
    mul_463: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(div_63, sub_131);  div_63 = sub_131 = None
    mul_464: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(clone_72, mul_458);  mul_458 = None
    sum_206: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_464, [0, 1]);  mul_464 = None
    sum_207: "f32[128]" = torch.ops.aten.sum.dim_IntList(clone_72, [0, 1]);  clone_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:208, code: position_embeddings = self.position_embeddings(position_ids)
    eq: "b8[1, 512]" = torch.ops.aten.eq.Scalar(slice_4, -1)
    unsqueeze_4: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
    scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_4: "f32[1, 512, 128]" = torch.ops.aten.where.self(unsqueeze_4, scalar_tensor_4, mul_463);  unsqueeze_4 = scalar_tensor_4 = None
    full_5: "f32[512, 128]" = torch.ops.aten.full.default([512, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put: "f32[512, 128]" = torch.ops.aten._unsafe_index_put.default(full_5, [slice_4], where_4, True);  full_5 = slice_4 = where_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:204, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    eq_1: "b8[1, 512]" = torch.ops.aten.eq.Scalar(expand, -1)
    unsqueeze_5: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_5: "f32[1, 512, 128]" = torch.ops.aten.where.self(unsqueeze_5, scalar_tensor_5, mul_463);  unsqueeze_5 = scalar_tensor_5 = None
    full_6: "f32[2, 128]" = torch.ops.aten.full.default([2, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put_1: "f32[2, 128]" = torch.ops.aten._unsafe_index_put.default(full_6, [expand], where_5, True);  full_6 = expand = where_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:203, code: inputs_embeds = self.word_embeddings(input_ids)
    eq_2: "b8[1, 512]" = torch.ops.aten.eq.Scalar(primals_209, 0)
    unsqueeze_6: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_2, -1);  eq_2 = None
    scalar_tensor_6: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_6: "f32[1, 512, 128]" = torch.ops.aten.where.self(unsqueeze_6, scalar_tensor_6, mul_463);  unsqueeze_6 = scalar_tensor_6 = mul_463 = None
    full_7: "f32[30522, 128]" = torch.ops.aten.full.default([30522, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put_2: "f32[30522, 128]" = torch.ops.aten._unsafe_index_put.default(full_7, [primals_209], where_6, True);  full_7 = primals_209 = where_6 = None
    return pytree.tree_unflatten([div_24, view_269, _unsafe_index_put_2, _unsafe_index_put_1, _unsafe_index_put, sum_206, sum_207, permute_542, view_616, permute_538, view_613, permute_534, view_610, permute_529, view_606, permute_517, view_594, sum_196, sum_197, permute_513, view_591, permute_509, view_588, sum_190, sum_191, permute_505, view_585, permute_501, view_582, permute_496, view_578, permute_484, view_566, sum_181, sum_182, permute_480, view_563, permute_476, view_560, sum_175, sum_176, permute_472, view_557, permute_468, view_554, permute_463, view_550, permute_451, view_538, sum_166, sum_167, permute_447, view_535, permute_443, view_532, sum_160, sum_161, permute_439, view_529, permute_435, view_526, permute_430, view_522, permute_418, view_510, sum_151, sum_152, permute_414, view_507, permute_410, view_504, sum_145, sum_146, permute_406, view_501, permute_402, view_498, permute_397, view_494, permute_385, view_482, sum_136, sum_137, permute_381, view_479, permute_377, view_476, sum_130, sum_131, permute_373, view_473, permute_369, view_470, permute_364, view_466, permute_352, view_454, sum_121, sum_122, permute_348, view_451, permute_344, view_448, sum_115, sum_116, permute_340, view_445, permute_336, view_442, permute_331, view_438, permute_319, view_426, sum_106, sum_107, permute_315, view_423, permute_311, view_420, sum_100, sum_101, permute_307, view_417, permute_303, view_414, permute_298, view_410, permute_286, view_398, sum_91, sum_92, permute_282, view_395, permute_278, view_392, sum_85, sum_86, permute_274, view_389, permute_270, view_386, permute_265, view_382, permute_253, view_370, sum_76, sum_77, permute_249, view_367, permute_245, view_364, sum_70, sum_71, permute_241, view_361, permute_237, view_358, permute_232, view_354, permute_220, view_342, sum_61, sum_62, permute_216, view_339, permute_212, view_336, sum_55, sum_56, permute_208, view_333, permute_204, view_330, permute_199, view_326, permute_187, view_314, sum_46, sum_47, permute_183, view_311, permute_179, view_308, sum_40, sum_41, permute_175, view_305, permute_171, view_302, permute_166, view_298, permute_154, view_286, sum_31, sum_32, permute_150, view_283, permute_146, view_280, sum_25, sum_26, permute_142, view_277, sum_20, sum_21, permute_138, view_274, None, None, None, None], self._out_spec)
    