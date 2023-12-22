from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[30522, 128]"; primals_2: "f32[2, 128]"; primals_3: "f32[512, 128]"; primals_4: "f32[128]"; primals_5: "f32[128]"; primals_6: "f32[256, 128]"; primals_7: "f32[256]"; primals_8: "f32[256, 256]"; primals_9: "f32[256]"; primals_10: "f32[256, 256]"; primals_11: "f32[256]"; primals_12: "f32[256, 256]"; primals_13: "f32[256]"; primals_14: "f32[256, 256]"; primals_15: "f32[256]"; primals_16: "f32[256]"; primals_17: "f32[256]"; primals_18: "f32[1024, 256]"; primals_19: "f32[1024]"; primals_20: "f32[256, 1024]"; primals_21: "f32[256]"; primals_22: "f32[256]"; primals_23: "f32[256]"; primals_24: "f32[256, 256]"; primals_25: "f32[256]"; primals_26: "f32[256, 256]"; primals_27: "f32[256]"; primals_28: "f32[256, 256]"; primals_29: "f32[256]"; primals_30: "f32[256, 256]"; primals_31: "f32[256]"; primals_32: "f32[256]"; primals_33: "f32[256]"; primals_34: "f32[1024, 256]"; primals_35: "f32[1024]"; primals_36: "f32[256, 1024]"; primals_37: "f32[256]"; primals_38: "f32[256]"; primals_39: "f32[256]"; primals_40: "f32[256, 256]"; primals_41: "f32[256]"; primals_42: "f32[256, 256]"; primals_43: "f32[256]"; primals_44: "f32[256, 256]"; primals_45: "f32[256]"; primals_46: "f32[256, 256]"; primals_47: "f32[256]"; primals_48: "f32[256]"; primals_49: "f32[256]"; primals_50: "f32[1024, 256]"; primals_51: "f32[1024]"; primals_52: "f32[256, 1024]"; primals_53: "f32[256]"; primals_54: "f32[256]"; primals_55: "f32[256]"; primals_56: "f32[256, 256]"; primals_57: "f32[256]"; primals_58: "f32[256, 256]"; primals_59: "f32[256]"; primals_60: "f32[256, 256]"; primals_61: "f32[256]"; primals_62: "f32[256, 256]"; primals_63: "f32[256]"; primals_64: "f32[256]"; primals_65: "f32[256]"; primals_66: "f32[1024, 256]"; primals_67: "f32[1024]"; primals_68: "f32[256, 1024]"; primals_69: "f32[256]"; primals_70: "f32[256]"; primals_71: "f32[256]"; primals_72: "f32[256, 256]"; primals_73: "f32[256]"; primals_74: "f32[256, 256]"; primals_75: "f32[256]"; primals_76: "f32[256, 256]"; primals_77: "f32[256]"; primals_78: "f32[256, 256]"; primals_79: "f32[256]"; primals_80: "f32[256]"; primals_81: "f32[256]"; primals_82: "f32[1024, 256]"; primals_83: "f32[1024]"; primals_84: "f32[256, 1024]"; primals_85: "f32[256]"; primals_86: "f32[256]"; primals_87: "f32[256]"; primals_88: "f32[256, 256]"; primals_89: "f32[256]"; primals_90: "f32[256, 256]"; primals_91: "f32[256]"; primals_92: "f32[256, 256]"; primals_93: "f32[256]"; primals_94: "f32[256, 256]"; primals_95: "f32[256]"; primals_96: "f32[256]"; primals_97: "f32[256]"; primals_98: "f32[1024, 256]"; primals_99: "f32[1024]"; primals_100: "f32[256, 1024]"; primals_101: "f32[256]"; primals_102: "f32[256]"; primals_103: "f32[256]"; primals_104: "f32[256, 256]"; primals_105: "f32[256]"; primals_106: "f32[256, 256]"; primals_107: "f32[256]"; primals_108: "f32[256, 256]"; primals_109: "f32[256]"; primals_110: "f32[256, 256]"; primals_111: "f32[256]"; primals_112: "f32[256]"; primals_113: "f32[256]"; primals_114: "f32[1024, 256]"; primals_115: "f32[1024]"; primals_116: "f32[256, 1024]"; primals_117: "f32[256]"; primals_118: "f32[256]"; primals_119: "f32[256]"; primals_120: "f32[256, 256]"; primals_121: "f32[256]"; primals_122: "f32[256, 256]"; primals_123: "f32[256]"; primals_124: "f32[256, 256]"; primals_125: "f32[256]"; primals_126: "f32[256, 256]"; primals_127: "f32[256]"; primals_128: "f32[256]"; primals_129: "f32[256]"; primals_130: "f32[1024, 256]"; primals_131: "f32[1024]"; primals_132: "f32[256, 1024]"; primals_133: "f32[256]"; primals_134: "f32[256]"; primals_135: "f32[256]"; primals_136: "f32[256, 256]"; primals_137: "f32[256]"; primals_138: "f32[256, 256]"; primals_139: "f32[256]"; primals_140: "f32[256, 256]"; primals_141: "f32[256]"; primals_142: "f32[256, 256]"; primals_143: "f32[256]"; primals_144: "f32[256]"; primals_145: "f32[256]"; primals_146: "f32[1024, 256]"; primals_147: "f32[1024]"; primals_148: "f32[256, 1024]"; primals_149: "f32[256]"; primals_150: "f32[256]"; primals_151: "f32[256]"; primals_152: "f32[256, 256]"; primals_153: "f32[256]"; primals_154: "f32[256, 256]"; primals_155: "f32[256]"; primals_156: "f32[256, 256]"; primals_157: "f32[256]"; primals_158: "f32[256, 256]"; primals_159: "f32[256]"; primals_160: "f32[256]"; primals_161: "f32[256]"; primals_162: "f32[1024, 256]"; primals_163: "f32[1024]"; primals_164: "f32[256, 1024]"; primals_165: "f32[256]"; primals_166: "f32[256]"; primals_167: "f32[256]"; primals_168: "f32[256, 256]"; primals_169: "f32[256]"; primals_170: "f32[256, 256]"; primals_171: "f32[256]"; primals_172: "f32[256, 256]"; primals_173: "f32[256]"; primals_174: "f32[256, 256]"; primals_175: "f32[256]"; primals_176: "f32[256]"; primals_177: "f32[256]"; primals_178: "f32[1024, 256]"; primals_179: "f32[1024]"; primals_180: "f32[256, 1024]"; primals_181: "f32[256]"; primals_182: "f32[256]"; primals_183: "f32[256]"; primals_184: "f32[256, 256]"; primals_185: "f32[256]"; primals_186: "f32[256, 256]"; primals_187: "f32[256]"; primals_188: "f32[256, 256]"; primals_189: "f32[256]"; primals_190: "f32[256, 256]"; primals_191: "f32[256]"; primals_192: "f32[256]"; primals_193: "f32[256]"; primals_194: "f32[1024, 256]"; primals_195: "f32[1024]"; primals_196: "f32[256, 1024]"; primals_197: "f32[256]"; primals_198: "f32[256]"; primals_199: "f32[256]"; primals_200: "f32[2, 256]"; primals_201: "f32[2]"; primals_202: "i64[1, 512]"; primals_203: "i64[1, 512]"; primals_204: "i64[1, 512]"; primals_205: "i64[1]"; primals_206: "i64[1]"; tangents_1: "f32[]"; tangents_2: "f32[1, 512]"; tangents_3: "f32[1, 512]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, tangents_1, tangents_2, tangents_3, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:885, code: attention_mask = torch.ones(input_shape, device=device)
    full: "f32[1, 512]" = torch.ops.aten.full.default([1, 512], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:888, code: buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
    slice_1: "i64[1, 512]" = torch.ops.aten.slice.Tensor(primals_202, 0, 0, 9223372036854775807);  primals_202 = None
    
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
    slice_4: "i64[1, 512]" = torch.ops.aten.slice.Tensor(primals_203, 0, 0, 9223372036854775807);  primals_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:203, code: inputs_embeds = self.word_embeddings(input_ids)
    embedding: "f32[1, 512, 128]" = torch.ops.aten.embedding.default(primals_1, primals_204, 0);  primals_1 = None
    
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1404, code: logits = self.qa_outputs(sequence_output)
    view_266: "f32[512, 256]" = torch.ops.aten.view.default(add_99, [512, 256]);  add_99 = None
    permute_133: "f32[256, 2]" = torch.ops.aten.permute.default(primals_200, [1, 0]);  primals_200 = None
    addmm_73: "f32[512, 2]" = torch.ops.aten.addmm.default(primals_201, view_266, permute_133);  primals_201 = None
    view_267: "f32[1, 512, 2]" = torch.ops.aten.view.default(addmm_73, [1, 512, 2]);  addmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1405, code: start_logits, end_logits = logits.split(1, dim=-1)
    split_with_sizes = torch.ops.aten.split_with_sizes.default(view_267, [1, 1], 2);  view_267 = None
    getitem_124: "f32[1, 512, 1]" = split_with_sizes[0]
    getitem_125: "f32[1, 512, 1]" = split_with_sizes[1];  split_with_sizes = None
    
    # No stacktrace found for following nodes
    squeeze: "f32[1, 512]" = torch.ops.aten.squeeze.dim(getitem_124, -1);  getitem_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1406, code: start_logits = start_logits.squeeze(-1).contiguous()
    clone_12: "f32[1, 512]" = torch.ops.aten.clone.default(squeeze, memory_format = torch.contiguous_format);  squeeze = None
    
    # No stacktrace found for following nodes
    squeeze_1: "f32[1, 512]" = torch.ops.aten.squeeze.dim(getitem_125, -1);  getitem_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1407, code: end_logits = end_logits.squeeze(-1).contiguous()
    clone_13: "f32[1, 512]" = torch.ops.aten.clone.default(squeeze_1, memory_format = torch.contiguous_format);  squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1418, code: start_positions = start_positions.clamp(0, ignored_index)
    clamp_min: "i64[1]" = torch.ops.aten.clamp_min.default(primals_205, 0);  primals_205 = None
    clamp_max: "i64[1]" = torch.ops.aten.clamp_max.default(clamp_min, 512);  clamp_min = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1419, code: end_positions = end_positions.clamp(0, ignored_index)
    clamp_min_1: "i64[1]" = torch.ops.aten.clamp_min.default(primals_206, 0);  primals_206 = None
    clamp_max_1: "i64[1]" = torch.ops.aten.clamp_max.default(clamp_min_1, 512);  clamp_min_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1422, code: start_loss = loss_fct(start_logits, start_positions)
    amax_12: "f32[1, 1]" = torch.ops.aten.amax.default(clone_12, [1], True)
    sub_38: "f32[1, 512]" = torch.ops.aten.sub.Tensor(clone_12, amax_12);  amax_12 = None
    exp_12: "f32[1, 512]" = torch.ops.aten.exp.default(sub_38)
    sum_13: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [1], True);  exp_12 = None
    log: "f32[1, 1]" = torch.ops.aten.log.default(sum_13);  sum_13 = None
    sub_39: "f32[1, 512]" = torch.ops.aten.sub.Tensor(sub_38, log);  sub_38 = log = None
    alias_12: "f32[1, 512]" = torch.ops.aten.alias.default(sub_39)
    ne: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max, 512)
    scalar_tensor: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0))
    where: "i64[1]" = torch.ops.aten.where.self(ne, clamp_max, scalar_tensor);  ne = scalar_tensor = None
    unsqueeze_2: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
    gather: "f32[1, 1]" = torch.ops.aten.gather.default(sub_39, 1, unsqueeze_2);  sub_39 = unsqueeze_2 = None
    squeeze_2: "f32[1]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[1]" = torch.ops.aten.neg.default(squeeze_2);  squeeze_2 = None
    ne_1: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max, 512)
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_1: "f32[1]" = torch.ops.aten.where.self(ne_1, neg, scalar_tensor_1);  ne_1 = neg = scalar_tensor_1 = None
    ne_2: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max, 512)
    sum_14: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None
    sum_15: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
    div_24: "f32[]" = torch.ops.aten.div.Tensor(sum_15, convert_element_type);  sum_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1423, code: end_loss = loss_fct(end_logits, end_positions)
    amax_13: "f32[1, 1]" = torch.ops.aten.amax.default(clone_13, [1], True)
    sub_40: "f32[1, 512]" = torch.ops.aten.sub.Tensor(clone_13, amax_13);  amax_13 = None
    exp_13: "f32[1, 512]" = torch.ops.aten.exp.default(sub_40)
    sum_16: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [1], True);  exp_13 = None
    log_1: "f32[1, 1]" = torch.ops.aten.log.default(sum_16);  sum_16 = None
    sub_41: "f32[1, 512]" = torch.ops.aten.sub.Tensor(sub_40, log_1);  sub_40 = log_1 = None
    alias_13: "f32[1, 512]" = torch.ops.aten.alias.default(sub_41)
    ne_3: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max_1, 512)
    scalar_tensor_2: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0))
    where_2: "i64[1]" = torch.ops.aten.where.self(ne_3, clamp_max_1, scalar_tensor_2);  ne_3 = scalar_tensor_2 = None
    unsqueeze_3: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(where_2, 1);  where_2 = None
    gather_1: "f32[1, 1]" = torch.ops.aten.gather.default(sub_41, 1, unsqueeze_3);  sub_41 = unsqueeze_3 = None
    squeeze_3: "f32[1]" = torch.ops.aten.squeeze.dim(gather_1, 1);  gather_1 = None
    neg_1: "f32[1]" = torch.ops.aten.neg.default(squeeze_3);  squeeze_3 = None
    ne_4: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max_1, 512)
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_3: "f32[1]" = torch.ops.aten.where.self(ne_4, neg_1, scalar_tensor_3);  ne_4 = neg_1 = scalar_tensor_3 = None
    ne_5: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max_1, 512)
    sum_17: "i64[]" = torch.ops.aten.sum.default(ne_5);  ne_5 = None
    convert_element_type_1: "f32[]" = torch.ops.prims.convert_element_type.default(sum_17, torch.float32);  sum_17 = None
    sum_18: "f32[]" = torch.ops.aten.sum.default(where_3);  where_3 = None
    div_25: "f32[]" = torch.ops.aten.div.Tensor(sum_18, convert_element_type_1);  sum_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1424, code: total_loss = (start_loss + end_loss) / 2
    add_100: "f32[]" = torch.ops.aten.add.Tensor(div_24, div_25);  div_24 = div_25 = None
    div_26: "f32[]" = torch.ops.aten.div.Tensor(add_100, 2);  add_100 = None
    div_27: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, 2);  tangents_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1423, code: end_loss = loss_fct(end_logits, end_positions)
    div_28: "f32[]" = torch.ops.aten.div.Tensor(div_27, convert_element_type_1);  convert_element_type_1 = None
    unsqueeze_4: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(clamp_max_1, 1);  clamp_max_1 = None
    ne_6: "b8[1, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_4, 512)
    scalar_tensor_4: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0))
    where_4: "i64[1, 1]" = torch.ops.aten.where.self(ne_6, unsqueeze_4, scalar_tensor_4);  ne_6 = scalar_tensor_4 = None
    full_1: "f32[1, 512]" = torch.ops.aten.full.default([1, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    scatter: "f32[1, 512]" = torch.ops.aten.scatter.value(full_1, 1, where_4, -1.0);  full_1 = where_4 = None
    ne_7: "b8[1, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_4, 512);  unsqueeze_4 = None
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_5: "f32[1, 1]" = torch.ops.aten.where.self(ne_7, div_28, scalar_tensor_5);  ne_7 = div_28 = scalar_tensor_5 = None
    mul_87: "f32[1, 512]" = torch.ops.aten.mul.Tensor(scatter, where_5);  scatter = where_5 = None
    alias_14: "f32[1, 512]" = torch.ops.aten.alias.default(alias_13);  alias_13 = None
    exp_14: "f32[1, 512]" = torch.ops.aten.exp.default(alias_14);  alias_14 = None
    sum_19: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(mul_87, [1], True)
    mul_88: "f32[1, 512]" = torch.ops.aten.mul.Tensor(exp_14, sum_19);  exp_14 = sum_19 = None
    sub_42: "f32[1, 512]" = torch.ops.aten.sub.Tensor(mul_87, mul_88);  mul_87 = mul_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1423, code: end_loss = loss_fct(end_logits, end_positions)
    add_101: "f32[1, 512]" = torch.ops.aten.add.Tensor(tangents_3, sub_42);  tangents_3 = sub_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1422, code: start_loss = loss_fct(start_logits, start_positions)
    div_29: "f32[]" = torch.ops.aten.div.Tensor(div_27, convert_element_type);  div_27 = convert_element_type = None
    unsqueeze_5: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(clamp_max, 1);  clamp_max = None
    ne_8: "b8[1, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_5, 512)
    scalar_tensor_6: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0))
    where_6: "i64[1, 1]" = torch.ops.aten.where.self(ne_8, unsqueeze_5, scalar_tensor_6);  ne_8 = scalar_tensor_6 = None
    full_2: "f32[1, 512]" = torch.ops.aten.full.default([1, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    scatter_1: "f32[1, 512]" = torch.ops.aten.scatter.value(full_2, 1, where_6, -1.0);  full_2 = where_6 = None
    ne_9: "b8[1, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_5, 512);  unsqueeze_5 = None
    scalar_tensor_7: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_7: "f32[1, 1]" = torch.ops.aten.where.self(ne_9, div_29, scalar_tensor_7);  ne_9 = div_29 = scalar_tensor_7 = None
    mul_89: "f32[1, 512]" = torch.ops.aten.mul.Tensor(scatter_1, where_7);  scatter_1 = where_7 = None
    alias_15: "f32[1, 512]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    exp_15: "f32[1, 512]" = torch.ops.aten.exp.default(alias_15);  alias_15 = None
    sum_20: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(mul_89, [1], True)
    mul_90: "f32[1, 512]" = torch.ops.aten.mul.Tensor(exp_15, sum_20);  exp_15 = sum_20 = None
    sub_43: "f32[1, 512]" = torch.ops.aten.sub.Tensor(mul_89, mul_90);  mul_89 = mul_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1422, code: start_loss = loss_fct(start_logits, start_positions)
    add_102: "f32[1, 512]" = torch.ops.aten.add.Tensor(tangents_2, sub_43);  tangents_2 = sub_43 = None
    
    # No stacktrace found for following nodes
    unsqueeze_6: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(add_101, 2);  add_101 = None
    unsqueeze_7: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(add_102, 2);  add_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1405, code: start_logits, end_logits = logits.split(1, dim=-1)
    cat: "f32[1, 512, 2]" = torch.ops.aten.cat.default([unsqueeze_7, unsqueeze_6], 2);  unsqueeze_7 = unsqueeze_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1404, code: logits = self.qa_outputs(sequence_output)
    view_268: "f32[512, 2]" = torch.ops.aten.view.default(cat, [512, 2]);  cat = None
    permute_134: "f32[2, 256]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
    mm: "f32[512, 256]" = torch.ops.aten.mm.default(view_268, permute_134);  permute_134 = None
    permute_135: "f32[2, 512]" = torch.ops.aten.permute.default(view_268, [1, 0])
    mm_1: "f32[2, 256]" = torch.ops.aten.mm.default(permute_135, view_266);  permute_135 = view_266 = None
    permute_136: "f32[256, 2]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_21: "f32[1, 2]" = torch.ops.aten.sum.dim_IntList(view_268, [0], True);  view_268 = None
    view_269: "f32[2]" = torch.ops.aten.view.default(sum_21, [2]);  sum_21 = None
    permute_137: "f32[2, 256]" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
    view_270: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm, [1, 512, 256]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_44: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_97, getitem_123);  add_97 = getitem_123 = None
    mul_91: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_24);  sub_44 = None
    mul_92: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(view_270, primals_198);  primals_198 = None
    mul_93: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_92, 256)
    sum_22: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_92, [2], True)
    mul_94: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_92, mul_91);  mul_92 = None
    sum_23: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_94, [2], True);  mul_94 = None
    mul_95: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_91, sum_23);  sum_23 = None
    sub_45: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_93, sum_22);  mul_93 = sum_22 = None
    sub_46: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_45, mul_95);  sub_45 = mul_95 = None
    div_30: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 256);  rsqrt_24 = None
    mul_96: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_30, sub_46);  div_30 = sub_46 = None
    mul_97: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(view_270, mul_91);  mul_91 = None
    sum_24: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_97, [0, 1]);  mul_97 = None
    sum_25: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_270, [0, 1]);  view_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_2: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_121, torch.float32);  getitem_121 = None
    mul_98: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
    mul_99: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_96, mul_98);  mul_98 = None
    clone_14: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_99, memory_format = torch.contiguous_format);  mul_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_271: "f32[512, 256]" = torch.ops.aten.view.default(clone_14, [512, 256]);  clone_14 = None
    permute_138: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
    mm_2: "f32[512, 1024]" = torch.ops.aten.mm.default(view_271, permute_138);  permute_138 = None
    permute_139: "f32[256, 512]" = torch.ops.aten.permute.default(view_271, [1, 0])
    mm_3: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_139, view_264);  permute_139 = view_264 = None
    permute_140: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_26: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_271, [0], True);  view_271 = None
    view_272: "f32[256]" = torch.ops.aten.view.default(sum_26, [256]);  sum_26 = None
    permute_141: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
    view_273: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_2, [1, 512, 1024]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_100: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_263, 0.7071067811865476)
    erf_12: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_100);  mul_100 = None
    add_103: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_101: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_103, 0.5);  add_103 = None
    mul_102: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_263, view_263)
    mul_103: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_102, -0.5);  mul_102 = None
    exp_16: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_103);  mul_103 = None
    mul_104: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
    mul_105: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_263, mul_104);  view_263 = mul_104 = None
    add_104: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_101, mul_105);  mul_101 = mul_105 = None
    mul_106: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_273, add_104);  view_273 = add_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_274: "f32[512, 1024]" = torch.ops.aten.view.default(mul_106, [512, 1024]);  mul_106 = None
    permute_142: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
    mm_4: "f32[512, 256]" = torch.ops.aten.mm.default(view_274, permute_142);  permute_142 = None
    permute_143: "f32[1024, 512]" = torch.ops.aten.permute.default(view_274, [1, 0])
    mm_5: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_143, view_262);  permute_143 = view_262 = None
    permute_144: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_27: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_274, [0], True);  view_274 = None
    view_275: "f32[1024]" = torch.ops.aten.view.default(sum_27, [1024]);  sum_27 = None
    permute_145: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
    view_276: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_4, [1, 512, 256]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_105: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_96, view_276);  mul_96 = view_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_47: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_93, getitem_119);  add_93 = getitem_119 = None
    mul_107: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_23);  sub_47 = None
    mul_108: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_105, primals_192);  primals_192 = None
    mul_109: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_108, 256)
    sum_28: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_108, [2], True)
    mul_110: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_108, mul_107);  mul_108 = None
    sum_29: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_110, [2], True);  mul_110 = None
    mul_111: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_107, sum_29);  sum_29 = None
    sub_48: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_109, sum_28);  mul_109 = sum_28 = None
    sub_49: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_48, mul_111);  sub_48 = mul_111 = None
    div_31: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 256);  rsqrt_23 = None
    mul_112: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_31, sub_49);  div_31 = sub_49 = None
    mul_113: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_105, mul_107);  mul_107 = None
    sum_30: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_113, [0, 1]);  mul_113 = None
    sum_31: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_105, [0, 1]);  add_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_3: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_117, torch.float32);  getitem_117 = None
    mul_114: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.1111111111111112);  convert_element_type_3 = None
    mul_115: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_112, mul_114);  mul_114 = None
    clone_15: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_115, memory_format = torch.contiguous_format);  mul_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_277: "f32[512, 256]" = torch.ops.aten.view.default(clone_15, [512, 256]);  clone_15 = None
    permute_146: "f32[256, 256]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
    mm_6: "f32[512, 256]" = torch.ops.aten.mm.default(view_277, permute_146);  permute_146 = None
    permute_147: "f32[256, 512]" = torch.ops.aten.permute.default(view_277, [1, 0])
    mm_7: "f32[256, 256]" = torch.ops.aten.mm.default(permute_147, view_260);  permute_147 = view_260 = None
    permute_148: "f32[256, 256]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_32: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_277, [0], True);  view_277 = None
    view_278: "f32[256]" = torch.ops.aten.view.default(sum_32, [256]);  sum_32 = None
    permute_149: "f32[256, 256]" = torch.ops.aten.permute.default(permute_148, [1, 0]);  permute_148 = None
    view_279: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_6, [1, 512, 256]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_280: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_279, [1, 512, 4, 64]);  view_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_150: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_280, [0, 2, 1, 3]);  view_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_281: "f32[4, 512, 64]" = torch.ops.aten.view.default(permute_150, [4, 512, 64]);  permute_150 = None
    permute_151: "f32[4, 512, 512]" = torch.ops.aten.permute.default(view_256, [0, 2, 1]);  view_256 = None
    bmm_24: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(permute_151, view_281);  permute_151 = None
    permute_152: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_257, [0, 2, 1]);  view_257 = None
    bmm_25: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_281, permute_152);  view_281 = permute_152 = None
    view_282: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_24, [1, 4, 512, 64]);  bmm_24 = None
    view_283: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_25, [1, 4, 512, 512]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_4: "f32[1, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_115, torch.float32);  getitem_115 = None
    mul_116: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 1.1111111111111112);  convert_element_type_4 = None
    mul_117: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_283, mul_116);  view_283 = mul_116 = None
    clone_16: "f32[1, 4, 512, 512]" = torch.ops.aten.clone.default(mul_117, memory_format = torch.contiguous_format);  mul_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_16: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    mul_118: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(clone_16, alias_16);  clone_16 = None
    sum_33: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_118, [-1], True)
    mul_119: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(alias_16, sum_33);  alias_16 = sum_33 = None
    sub_50: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(mul_118, mul_119);  mul_118 = mul_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_32: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(sub_50, 8.0);  sub_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_284: "f32[4, 512, 512]" = torch.ops.aten.view.default(div_32, [4, 512, 512]);  div_32 = None
    permute_153: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_253, [0, 2, 1]);  view_253 = None
    bmm_26: "f32[4, 64, 512]" = torch.ops.aten.bmm.default(permute_153, view_284);  permute_153 = None
    permute_154: "f32[4, 512, 64]" = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None
    bmm_27: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_284, permute_154);  view_284 = permute_154 = None
    view_285: "f32[1, 4, 64, 512]" = torch.ops.aten.view.default(bmm_26, [1, 4, 64, 512]);  bmm_26 = None
    view_286: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_27, [1, 4, 512, 64]);  bmm_27 = None
    permute_155: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_285, [0, 1, 3, 2]);  view_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_156: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_286, [0, 2, 1, 3]);  view_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_17: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_156, memory_format = torch.contiguous_format);  permute_156 = None
    view_287: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_17, [1, 512, 256]);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_157: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_282, [0, 2, 1, 3]);  view_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_18: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_157, memory_format = torch.contiguous_format);  permute_157 = None
    view_288: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_18, [1, 512, 256]);  clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_289: "f32[512, 256]" = torch.ops.aten.view.default(view_288, [512, 256]);  view_288 = None
    permute_158: "f32[256, 256]" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
    mm_8: "f32[512, 256]" = torch.ops.aten.mm.default(view_289, permute_158);  permute_158 = None
    permute_159: "f32[256, 512]" = torch.ops.aten.permute.default(view_289, [1, 0])
    mm_9: "f32[256, 256]" = torch.ops.aten.mm.default(permute_159, view_249);  permute_159 = view_249 = None
    permute_160: "f32[256, 256]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_34: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_289, [0], True);  view_289 = None
    view_290: "f32[256]" = torch.ops.aten.view.default(sum_34, [256]);  sum_34 = None
    permute_161: "f32[256, 256]" = torch.ops.aten.permute.default(permute_160, [1, 0]);  permute_160 = None
    view_291: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_8, [1, 512, 256]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_106: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_112, view_291);  mul_112 = view_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_162: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(permute_155, [0, 2, 1, 3]);  permute_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_292: "f32[1, 512, 256]" = torch.ops.aten.view.default(permute_162, [1, 512, 256]);  permute_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_293: "f32[512, 256]" = torch.ops.aten.view.default(view_292, [512, 256]);  view_292 = None
    permute_163: "f32[256, 256]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
    mm_10: "f32[512, 256]" = torch.ops.aten.mm.default(view_293, permute_163);  permute_163 = None
    permute_164: "f32[256, 512]" = torch.ops.aten.permute.default(view_293, [1, 0])
    mm_11: "f32[256, 256]" = torch.ops.aten.mm.default(permute_164, view_246);  permute_164 = view_246 = None
    permute_165: "f32[256, 256]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_35: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_293, [0], True);  view_293 = None
    view_294: "f32[256]" = torch.ops.aten.view.default(sum_35, [256]);  sum_35 = None
    permute_166: "f32[256, 256]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
    view_295: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_10, [1, 512, 256]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_107: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_106, view_295);  add_106 = view_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_296: "f32[512, 256]" = torch.ops.aten.view.default(view_287, [512, 256]);  view_287 = None
    permute_167: "f32[256, 256]" = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
    mm_12: "f32[512, 256]" = torch.ops.aten.mm.default(view_296, permute_167);  permute_167 = None
    permute_168: "f32[256, 512]" = torch.ops.aten.permute.default(view_296, [1, 0])
    mm_13: "f32[256, 256]" = torch.ops.aten.mm.default(permute_168, view_244);  permute_168 = view_244 = None
    permute_169: "f32[256, 256]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_36: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_296, [0], True);  view_296 = None
    view_297: "f32[256]" = torch.ops.aten.view.default(sum_36, [256]);  sum_36 = None
    permute_170: "f32[256, 256]" = torch.ops.aten.permute.default(permute_169, [1, 0]);  permute_169 = None
    view_298: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_12, [1, 512, 256]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_108: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_107, view_298);  add_107 = view_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_51: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_89, getitem_113);  add_89 = getitem_113 = None
    mul_120: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_22);  sub_51 = None
    mul_121: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_108, primals_182);  primals_182 = None
    mul_122: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_121, 256)
    sum_37: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_121, [2], True)
    mul_123: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_121, mul_120);  mul_121 = None
    sum_38: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_123, [2], True);  mul_123 = None
    mul_124: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_120, sum_38);  sum_38 = None
    sub_52: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_122, sum_37);  mul_122 = sum_37 = None
    sub_53: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_52, mul_124);  sub_52 = mul_124 = None
    div_33: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 256);  rsqrt_22 = None
    mul_125: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_33, sub_53);  div_33 = sub_53 = None
    mul_126: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_108, mul_120);  mul_120 = None
    sum_39: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_126, [0, 1]);  mul_126 = None
    sum_40: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_108, [0, 1]);  add_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_5: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_111, torch.float32);  getitem_111 = None
    mul_127: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 1.1111111111111112);  convert_element_type_5 = None
    mul_128: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_125, mul_127);  mul_127 = None
    clone_19: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_128, memory_format = torch.contiguous_format);  mul_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_299: "f32[512, 256]" = torch.ops.aten.view.default(clone_19, [512, 256]);  clone_19 = None
    permute_171: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
    mm_14: "f32[512, 1024]" = torch.ops.aten.mm.default(view_299, permute_171);  permute_171 = None
    permute_172: "f32[256, 512]" = torch.ops.aten.permute.default(view_299, [1, 0])
    mm_15: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_172, view_242);  permute_172 = view_242 = None
    permute_173: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_41: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_299, [0], True);  view_299 = None
    view_300: "f32[256]" = torch.ops.aten.view.default(sum_41, [256]);  sum_41 = None
    permute_174: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
    view_301: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_14, [1, 512, 1024]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_129: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_241, 0.7071067811865476)
    erf_13: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_129);  mul_129 = None
    add_109: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_130: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_109, 0.5);  add_109 = None
    mul_131: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_241, view_241)
    mul_132: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_131, -0.5);  mul_131 = None
    exp_17: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_132);  mul_132 = None
    mul_133: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_17, 0.3989422804014327);  exp_17 = None
    mul_134: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_241, mul_133);  view_241 = mul_133 = None
    add_110: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_130, mul_134);  mul_130 = mul_134 = None
    mul_135: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_301, add_110);  view_301 = add_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_302: "f32[512, 1024]" = torch.ops.aten.view.default(mul_135, [512, 1024]);  mul_135 = None
    permute_175: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
    mm_16: "f32[512, 256]" = torch.ops.aten.mm.default(view_302, permute_175);  permute_175 = None
    permute_176: "f32[1024, 512]" = torch.ops.aten.permute.default(view_302, [1, 0])
    mm_17: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_176, view_240);  permute_176 = view_240 = None
    permute_177: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_42: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_302, [0], True);  view_302 = None
    view_303: "f32[1024]" = torch.ops.aten.view.default(sum_42, [1024]);  sum_42 = None
    permute_178: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_177, [1, 0]);  permute_177 = None
    view_304: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_16, [1, 512, 256]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_111: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_125, view_304);  mul_125 = view_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_54: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_85, getitem_109);  add_85 = getitem_109 = None
    mul_136: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_21);  sub_54 = None
    mul_137: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_111, primals_176);  primals_176 = None
    mul_138: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_137, 256)
    sum_43: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_137, [2], True)
    mul_139: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_137, mul_136);  mul_137 = None
    sum_44: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_139, [2], True);  mul_139 = None
    mul_140: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_136, sum_44);  sum_44 = None
    sub_55: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_138, sum_43);  mul_138 = sum_43 = None
    sub_56: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_55, mul_140);  sub_55 = mul_140 = None
    div_34: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 256);  rsqrt_21 = None
    mul_141: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_34, sub_56);  div_34 = sub_56 = None
    mul_142: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_111, mul_136);  mul_136 = None
    sum_45: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_142, [0, 1]);  mul_142 = None
    sum_46: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_111, [0, 1]);  add_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_6: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_107, torch.float32);  getitem_107 = None
    mul_143: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 1.1111111111111112);  convert_element_type_6 = None
    mul_144: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_141, mul_143);  mul_143 = None
    clone_20: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_144, memory_format = torch.contiguous_format);  mul_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_305: "f32[512, 256]" = torch.ops.aten.view.default(clone_20, [512, 256]);  clone_20 = None
    permute_179: "f32[256, 256]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
    mm_18: "f32[512, 256]" = torch.ops.aten.mm.default(view_305, permute_179);  permute_179 = None
    permute_180: "f32[256, 512]" = torch.ops.aten.permute.default(view_305, [1, 0])
    mm_19: "f32[256, 256]" = torch.ops.aten.mm.default(permute_180, view_238);  permute_180 = view_238 = None
    permute_181: "f32[256, 256]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_47: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_305, [0], True);  view_305 = None
    view_306: "f32[256]" = torch.ops.aten.view.default(sum_47, [256]);  sum_47 = None
    permute_182: "f32[256, 256]" = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None
    view_307: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_18, [1, 512, 256]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_308: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_307, [1, 512, 4, 64]);  view_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_183: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_308, [0, 2, 1, 3]);  view_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_309: "f32[4, 512, 64]" = torch.ops.aten.view.default(permute_183, [4, 512, 64]);  permute_183 = None
    permute_184: "f32[4, 512, 512]" = torch.ops.aten.permute.default(view_234, [0, 2, 1]);  view_234 = None
    bmm_28: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(permute_184, view_309);  permute_184 = None
    permute_185: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_235, [0, 2, 1]);  view_235 = None
    bmm_29: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_309, permute_185);  view_309 = permute_185 = None
    view_310: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_28, [1, 4, 512, 64]);  bmm_28 = None
    view_311: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_29, [1, 4, 512, 512]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_7: "f32[1, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_105, torch.float32);  getitem_105 = None
    mul_145: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_7, 1.1111111111111112);  convert_element_type_7 = None
    mul_146: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_311, mul_145);  view_311 = mul_145 = None
    clone_21: "f32[1, 4, 512, 512]" = torch.ops.aten.clone.default(mul_146, memory_format = torch.contiguous_format);  mul_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_17: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    mul_147: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(clone_21, alias_17);  clone_21 = None
    sum_48: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_147, [-1], True)
    mul_148: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(alias_17, sum_48);  alias_17 = sum_48 = None
    sub_57: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(mul_147, mul_148);  mul_147 = mul_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_35: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(sub_57, 8.0);  sub_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_312: "f32[4, 512, 512]" = torch.ops.aten.view.default(div_35, [4, 512, 512]);  div_35 = None
    permute_186: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_231, [0, 2, 1]);  view_231 = None
    bmm_30: "f32[4, 64, 512]" = torch.ops.aten.bmm.default(permute_186, view_312);  permute_186 = None
    permute_187: "f32[4, 512, 64]" = torch.ops.aten.permute.default(view_232, [0, 2, 1]);  view_232 = None
    bmm_31: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_312, permute_187);  view_312 = permute_187 = None
    view_313: "f32[1, 4, 64, 512]" = torch.ops.aten.view.default(bmm_30, [1, 4, 64, 512]);  bmm_30 = None
    view_314: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_31, [1, 4, 512, 64]);  bmm_31 = None
    permute_188: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_313, [0, 1, 3, 2]);  view_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_189: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_314, [0, 2, 1, 3]);  view_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_22: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None
    view_315: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_22, [1, 512, 256]);  clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_190: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_310, [0, 2, 1, 3]);  view_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_23: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_190, memory_format = torch.contiguous_format);  permute_190 = None
    view_316: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_23, [1, 512, 256]);  clone_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_317: "f32[512, 256]" = torch.ops.aten.view.default(view_316, [512, 256]);  view_316 = None
    permute_191: "f32[256, 256]" = torch.ops.aten.permute.default(permute_114, [1, 0]);  permute_114 = None
    mm_20: "f32[512, 256]" = torch.ops.aten.mm.default(view_317, permute_191);  permute_191 = None
    permute_192: "f32[256, 512]" = torch.ops.aten.permute.default(view_317, [1, 0])
    mm_21: "f32[256, 256]" = torch.ops.aten.mm.default(permute_192, view_227);  permute_192 = view_227 = None
    permute_193: "f32[256, 256]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_49: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_317, [0], True);  view_317 = None
    view_318: "f32[256]" = torch.ops.aten.view.default(sum_49, [256]);  sum_49 = None
    permute_194: "f32[256, 256]" = torch.ops.aten.permute.default(permute_193, [1, 0]);  permute_193 = None
    view_319: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_20, [1, 512, 256]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_112: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_141, view_319);  mul_141 = view_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_195: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(permute_188, [0, 2, 1, 3]);  permute_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_320: "f32[1, 512, 256]" = torch.ops.aten.view.default(permute_195, [1, 512, 256]);  permute_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_321: "f32[512, 256]" = torch.ops.aten.view.default(view_320, [512, 256]);  view_320 = None
    permute_196: "f32[256, 256]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
    mm_22: "f32[512, 256]" = torch.ops.aten.mm.default(view_321, permute_196);  permute_196 = None
    permute_197: "f32[256, 512]" = torch.ops.aten.permute.default(view_321, [1, 0])
    mm_23: "f32[256, 256]" = torch.ops.aten.mm.default(permute_197, view_224);  permute_197 = view_224 = None
    permute_198: "f32[256, 256]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_50: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_321, [0], True);  view_321 = None
    view_322: "f32[256]" = torch.ops.aten.view.default(sum_50, [256]);  sum_50 = None
    permute_199: "f32[256, 256]" = torch.ops.aten.permute.default(permute_198, [1, 0]);  permute_198 = None
    view_323: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_22, [1, 512, 256]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_113: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_112, view_323);  add_112 = view_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_324: "f32[512, 256]" = torch.ops.aten.view.default(view_315, [512, 256]);  view_315 = None
    permute_200: "f32[256, 256]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
    mm_24: "f32[512, 256]" = torch.ops.aten.mm.default(view_324, permute_200);  permute_200 = None
    permute_201: "f32[256, 512]" = torch.ops.aten.permute.default(view_324, [1, 0])
    mm_25: "f32[256, 256]" = torch.ops.aten.mm.default(permute_201, view_222);  permute_201 = view_222 = None
    permute_202: "f32[256, 256]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_51: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_324, [0], True);  view_324 = None
    view_325: "f32[256]" = torch.ops.aten.view.default(sum_51, [256]);  sum_51 = None
    permute_203: "f32[256, 256]" = torch.ops.aten.permute.default(permute_202, [1, 0]);  permute_202 = None
    view_326: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_24, [1, 512, 256]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_114: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_113, view_326);  add_113 = view_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_58: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_81, getitem_103);  add_81 = getitem_103 = None
    mul_149: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_20);  sub_58 = None
    mul_150: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_114, primals_166);  primals_166 = None
    mul_151: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_150, 256)
    sum_52: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_150, [2], True)
    mul_152: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_150, mul_149);  mul_150 = None
    sum_53: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_152, [2], True);  mul_152 = None
    mul_153: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_149, sum_53);  sum_53 = None
    sub_59: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_151, sum_52);  mul_151 = sum_52 = None
    sub_60: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_59, mul_153);  sub_59 = mul_153 = None
    div_36: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 256);  rsqrt_20 = None
    mul_154: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_36, sub_60);  div_36 = sub_60 = None
    mul_155: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_114, mul_149);  mul_149 = None
    sum_54: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_155, [0, 1]);  mul_155 = None
    sum_55: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_114, [0, 1]);  add_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_8: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_101, torch.float32);  getitem_101 = None
    mul_156: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.1111111111111112);  convert_element_type_8 = None
    mul_157: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_154, mul_156);  mul_156 = None
    clone_24: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_157, memory_format = torch.contiguous_format);  mul_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_327: "f32[512, 256]" = torch.ops.aten.view.default(clone_24, [512, 256]);  clone_24 = None
    permute_204: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
    mm_26: "f32[512, 1024]" = torch.ops.aten.mm.default(view_327, permute_204);  permute_204 = None
    permute_205: "f32[256, 512]" = torch.ops.aten.permute.default(view_327, [1, 0])
    mm_27: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_205, view_220);  permute_205 = view_220 = None
    permute_206: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_56: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_327, [0], True);  view_327 = None
    view_328: "f32[256]" = torch.ops.aten.view.default(sum_56, [256]);  sum_56 = None
    permute_207: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None
    view_329: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_26, [1, 512, 1024]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_158: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_219, 0.7071067811865476)
    erf_14: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_158);  mul_158 = None
    add_115: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_159: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_115, 0.5);  add_115 = None
    mul_160: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_219, view_219)
    mul_161: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_160, -0.5);  mul_160 = None
    exp_18: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_161);  mul_161 = None
    mul_162: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
    mul_163: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_219, mul_162);  view_219 = mul_162 = None
    add_116: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_159, mul_163);  mul_159 = mul_163 = None
    mul_164: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_329, add_116);  view_329 = add_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_330: "f32[512, 1024]" = torch.ops.aten.view.default(mul_164, [512, 1024]);  mul_164 = None
    permute_208: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
    mm_28: "f32[512, 256]" = torch.ops.aten.mm.default(view_330, permute_208);  permute_208 = None
    permute_209: "f32[1024, 512]" = torch.ops.aten.permute.default(view_330, [1, 0])
    mm_29: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_209, view_218);  permute_209 = view_218 = None
    permute_210: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_57: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_330, [0], True);  view_330 = None
    view_331: "f32[1024]" = torch.ops.aten.view.default(sum_57, [1024]);  sum_57 = None
    permute_211: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_210, [1, 0]);  permute_210 = None
    view_332: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_28, [1, 512, 256]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_117: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_154, view_332);  mul_154 = view_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_61: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_77, getitem_99);  add_77 = getitem_99 = None
    mul_165: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_19);  sub_61 = None
    mul_166: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_117, primals_160);  primals_160 = None
    mul_167: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_166, 256)
    sum_58: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_166, [2], True)
    mul_168: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_166, mul_165);  mul_166 = None
    sum_59: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_168, [2], True);  mul_168 = None
    mul_169: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_165, sum_59);  sum_59 = None
    sub_62: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_167, sum_58);  mul_167 = sum_58 = None
    sub_63: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_62, mul_169);  sub_62 = mul_169 = None
    div_37: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 256);  rsqrt_19 = None
    mul_170: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_37, sub_63);  div_37 = sub_63 = None
    mul_171: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_117, mul_165);  mul_165 = None
    sum_60: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_171, [0, 1]);  mul_171 = None
    sum_61: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_117, [0, 1]);  add_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_9: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_97, torch.float32);  getitem_97 = None
    mul_172: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_9, 1.1111111111111112);  convert_element_type_9 = None
    mul_173: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_170, mul_172);  mul_172 = None
    clone_25: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_173, memory_format = torch.contiguous_format);  mul_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_333: "f32[512, 256]" = torch.ops.aten.view.default(clone_25, [512, 256]);  clone_25 = None
    permute_212: "f32[256, 256]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
    mm_30: "f32[512, 256]" = torch.ops.aten.mm.default(view_333, permute_212);  permute_212 = None
    permute_213: "f32[256, 512]" = torch.ops.aten.permute.default(view_333, [1, 0])
    mm_31: "f32[256, 256]" = torch.ops.aten.mm.default(permute_213, view_216);  permute_213 = view_216 = None
    permute_214: "f32[256, 256]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_62: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_333, [0], True);  view_333 = None
    view_334: "f32[256]" = torch.ops.aten.view.default(sum_62, [256]);  sum_62 = None
    permute_215: "f32[256, 256]" = torch.ops.aten.permute.default(permute_214, [1, 0]);  permute_214 = None
    view_335: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_30, [1, 512, 256]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_336: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_335, [1, 512, 4, 64]);  view_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_216: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_336, [0, 2, 1, 3]);  view_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_337: "f32[4, 512, 64]" = torch.ops.aten.view.default(permute_216, [4, 512, 64]);  permute_216 = None
    permute_217: "f32[4, 512, 512]" = torch.ops.aten.permute.default(view_212, [0, 2, 1]);  view_212 = None
    bmm_32: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(permute_217, view_337);  permute_217 = None
    permute_218: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_213, [0, 2, 1]);  view_213 = None
    bmm_33: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_337, permute_218);  view_337 = permute_218 = None
    view_338: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_32, [1, 4, 512, 64]);  bmm_32 = None
    view_339: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_33, [1, 4, 512, 512]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_10: "f32[1, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_95, torch.float32);  getitem_95 = None
    mul_174: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.1111111111111112);  convert_element_type_10 = None
    mul_175: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_339, mul_174);  view_339 = mul_174 = None
    clone_26: "f32[1, 4, 512, 512]" = torch.ops.aten.clone.default(mul_175, memory_format = torch.contiguous_format);  mul_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_18: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    mul_176: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(clone_26, alias_18);  clone_26 = None
    sum_63: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_176, [-1], True)
    mul_177: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(alias_18, sum_63);  alias_18 = sum_63 = None
    sub_64: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(mul_176, mul_177);  mul_176 = mul_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_38: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(sub_64, 8.0);  sub_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_340: "f32[4, 512, 512]" = torch.ops.aten.view.default(div_38, [4, 512, 512]);  div_38 = None
    permute_219: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_209, [0, 2, 1]);  view_209 = None
    bmm_34: "f32[4, 64, 512]" = torch.ops.aten.bmm.default(permute_219, view_340);  permute_219 = None
    permute_220: "f32[4, 512, 64]" = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None
    bmm_35: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_340, permute_220);  view_340 = permute_220 = None
    view_341: "f32[1, 4, 64, 512]" = torch.ops.aten.view.default(bmm_34, [1, 4, 64, 512]);  bmm_34 = None
    view_342: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_35, [1, 4, 512, 64]);  bmm_35 = None
    permute_221: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_341, [0, 1, 3, 2]);  view_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_222: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_342, [0, 2, 1, 3]);  view_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_27: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_222, memory_format = torch.contiguous_format);  permute_222 = None
    view_343: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_27, [1, 512, 256]);  clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_223: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_338, [0, 2, 1, 3]);  view_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_28: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_223, memory_format = torch.contiguous_format);  permute_223 = None
    view_344: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_28, [1, 512, 256]);  clone_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_345: "f32[512, 256]" = torch.ops.aten.view.default(view_344, [512, 256]);  view_344 = None
    permute_224: "f32[256, 256]" = torch.ops.aten.permute.default(permute_103, [1, 0]);  permute_103 = None
    mm_32: "f32[512, 256]" = torch.ops.aten.mm.default(view_345, permute_224);  permute_224 = None
    permute_225: "f32[256, 512]" = torch.ops.aten.permute.default(view_345, [1, 0])
    mm_33: "f32[256, 256]" = torch.ops.aten.mm.default(permute_225, view_205);  permute_225 = view_205 = None
    permute_226: "f32[256, 256]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_64: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_345, [0], True);  view_345 = None
    view_346: "f32[256]" = torch.ops.aten.view.default(sum_64, [256]);  sum_64 = None
    permute_227: "f32[256, 256]" = torch.ops.aten.permute.default(permute_226, [1, 0]);  permute_226 = None
    view_347: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_32, [1, 512, 256]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_118: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_170, view_347);  mul_170 = view_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_228: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(permute_221, [0, 2, 1, 3]);  permute_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_348: "f32[1, 512, 256]" = torch.ops.aten.view.default(permute_228, [1, 512, 256]);  permute_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_349: "f32[512, 256]" = torch.ops.aten.view.default(view_348, [512, 256]);  view_348 = None
    permute_229: "f32[256, 256]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
    mm_34: "f32[512, 256]" = torch.ops.aten.mm.default(view_349, permute_229);  permute_229 = None
    permute_230: "f32[256, 512]" = torch.ops.aten.permute.default(view_349, [1, 0])
    mm_35: "f32[256, 256]" = torch.ops.aten.mm.default(permute_230, view_202);  permute_230 = view_202 = None
    permute_231: "f32[256, 256]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_65: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_349, [0], True);  view_349 = None
    view_350: "f32[256]" = torch.ops.aten.view.default(sum_65, [256]);  sum_65 = None
    permute_232: "f32[256, 256]" = torch.ops.aten.permute.default(permute_231, [1, 0]);  permute_231 = None
    view_351: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_34, [1, 512, 256]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_119: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_118, view_351);  add_118 = view_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_352: "f32[512, 256]" = torch.ops.aten.view.default(view_343, [512, 256]);  view_343 = None
    permute_233: "f32[256, 256]" = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
    mm_36: "f32[512, 256]" = torch.ops.aten.mm.default(view_352, permute_233);  permute_233 = None
    permute_234: "f32[256, 512]" = torch.ops.aten.permute.default(view_352, [1, 0])
    mm_37: "f32[256, 256]" = torch.ops.aten.mm.default(permute_234, view_200);  permute_234 = view_200 = None
    permute_235: "f32[256, 256]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_66: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_352, [0], True);  view_352 = None
    view_353: "f32[256]" = torch.ops.aten.view.default(sum_66, [256]);  sum_66 = None
    permute_236: "f32[256, 256]" = torch.ops.aten.permute.default(permute_235, [1, 0]);  permute_235 = None
    view_354: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_36, [1, 512, 256]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_120: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_119, view_354);  add_119 = view_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_65: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_73, getitem_93);  add_73 = getitem_93 = None
    mul_178: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_18);  sub_65 = None
    mul_179: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_120, primals_150);  primals_150 = None
    mul_180: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_179, 256)
    sum_67: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_179, [2], True)
    mul_181: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_179, mul_178);  mul_179 = None
    sum_68: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_181, [2], True);  mul_181 = None
    mul_182: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_178, sum_68);  sum_68 = None
    sub_66: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_180, sum_67);  mul_180 = sum_67 = None
    sub_67: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_66, mul_182);  sub_66 = mul_182 = None
    div_39: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 256);  rsqrt_18 = None
    mul_183: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_39, sub_67);  div_39 = sub_67 = None
    mul_184: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_120, mul_178);  mul_178 = None
    sum_69: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_184, [0, 1]);  mul_184 = None
    sum_70: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_120, [0, 1]);  add_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_11: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_91, torch.float32);  getitem_91 = None
    mul_185: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 1.1111111111111112);  convert_element_type_11 = None
    mul_186: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_183, mul_185);  mul_185 = None
    clone_29: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_186, memory_format = torch.contiguous_format);  mul_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_355: "f32[512, 256]" = torch.ops.aten.view.default(clone_29, [512, 256]);  clone_29 = None
    permute_237: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
    mm_38: "f32[512, 1024]" = torch.ops.aten.mm.default(view_355, permute_237);  permute_237 = None
    permute_238: "f32[256, 512]" = torch.ops.aten.permute.default(view_355, [1, 0])
    mm_39: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_238, view_198);  permute_238 = view_198 = None
    permute_239: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_71: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_355, [0], True);  view_355 = None
    view_356: "f32[256]" = torch.ops.aten.view.default(sum_71, [256]);  sum_71 = None
    permute_240: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_239, [1, 0]);  permute_239 = None
    view_357: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_38, [1, 512, 1024]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_187: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_197, 0.7071067811865476)
    erf_15: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_187);  mul_187 = None
    add_121: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_188: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_121, 0.5);  add_121 = None
    mul_189: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_197, view_197)
    mul_190: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_189, -0.5);  mul_189 = None
    exp_19: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_190);  mul_190 = None
    mul_191: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
    mul_192: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_197, mul_191);  view_197 = mul_191 = None
    add_122: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_188, mul_192);  mul_188 = mul_192 = None
    mul_193: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_357, add_122);  view_357 = add_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_358: "f32[512, 1024]" = torch.ops.aten.view.default(mul_193, [512, 1024]);  mul_193 = None
    permute_241: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
    mm_40: "f32[512, 256]" = torch.ops.aten.mm.default(view_358, permute_241);  permute_241 = None
    permute_242: "f32[1024, 512]" = torch.ops.aten.permute.default(view_358, [1, 0])
    mm_41: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_242, view_196);  permute_242 = view_196 = None
    permute_243: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_72: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_358, [0], True);  view_358 = None
    view_359: "f32[1024]" = torch.ops.aten.view.default(sum_72, [1024]);  sum_72 = None
    permute_244: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
    view_360: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_40, [1, 512, 256]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_123: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_183, view_360);  mul_183 = view_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_68: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_69, getitem_89);  add_69 = getitem_89 = None
    mul_194: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_17);  sub_68 = None
    mul_195: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_123, primals_144);  primals_144 = None
    mul_196: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_195, 256)
    sum_73: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_195, [2], True)
    mul_197: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_195, mul_194);  mul_195 = None
    sum_74: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_197, [2], True);  mul_197 = None
    mul_198: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_194, sum_74);  sum_74 = None
    sub_69: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_196, sum_73);  mul_196 = sum_73 = None
    sub_70: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_69, mul_198);  sub_69 = mul_198 = None
    div_40: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 256);  rsqrt_17 = None
    mul_199: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_40, sub_70);  div_40 = sub_70 = None
    mul_200: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_123, mul_194);  mul_194 = None
    sum_75: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_200, [0, 1]);  mul_200 = None
    sum_76: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_123, [0, 1]);  add_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_12: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_87, torch.float32);  getitem_87 = None
    mul_201: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_12, 1.1111111111111112);  convert_element_type_12 = None
    mul_202: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_199, mul_201);  mul_201 = None
    clone_30: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_202, memory_format = torch.contiguous_format);  mul_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_361: "f32[512, 256]" = torch.ops.aten.view.default(clone_30, [512, 256]);  clone_30 = None
    permute_245: "f32[256, 256]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
    mm_42: "f32[512, 256]" = torch.ops.aten.mm.default(view_361, permute_245);  permute_245 = None
    permute_246: "f32[256, 512]" = torch.ops.aten.permute.default(view_361, [1, 0])
    mm_43: "f32[256, 256]" = torch.ops.aten.mm.default(permute_246, view_194);  permute_246 = view_194 = None
    permute_247: "f32[256, 256]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_77: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_361, [0], True);  view_361 = None
    view_362: "f32[256]" = torch.ops.aten.view.default(sum_77, [256]);  sum_77 = None
    permute_248: "f32[256, 256]" = torch.ops.aten.permute.default(permute_247, [1, 0]);  permute_247 = None
    view_363: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_42, [1, 512, 256]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_364: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_363, [1, 512, 4, 64]);  view_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_249: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_364, [0, 2, 1, 3]);  view_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_365: "f32[4, 512, 64]" = torch.ops.aten.view.default(permute_249, [4, 512, 64]);  permute_249 = None
    permute_250: "f32[4, 512, 512]" = torch.ops.aten.permute.default(view_190, [0, 2, 1]);  view_190 = None
    bmm_36: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(permute_250, view_365);  permute_250 = None
    permute_251: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_191, [0, 2, 1]);  view_191 = None
    bmm_37: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_365, permute_251);  view_365 = permute_251 = None
    view_366: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_36, [1, 4, 512, 64]);  bmm_36 = None
    view_367: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_37, [1, 4, 512, 512]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_13: "f32[1, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_85, torch.float32);  getitem_85 = None
    mul_203: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
    mul_204: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_367, mul_203);  view_367 = mul_203 = None
    clone_31: "f32[1, 4, 512, 512]" = torch.ops.aten.clone.default(mul_204, memory_format = torch.contiguous_format);  mul_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_19: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    mul_205: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(clone_31, alias_19);  clone_31 = None
    sum_78: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_205, [-1], True)
    mul_206: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(alias_19, sum_78);  alias_19 = sum_78 = None
    sub_71: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(mul_205, mul_206);  mul_205 = mul_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_41: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(sub_71, 8.0);  sub_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_368: "f32[4, 512, 512]" = torch.ops.aten.view.default(div_41, [4, 512, 512]);  div_41 = None
    permute_252: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_187, [0, 2, 1]);  view_187 = None
    bmm_38: "f32[4, 64, 512]" = torch.ops.aten.bmm.default(permute_252, view_368);  permute_252 = None
    permute_253: "f32[4, 512, 64]" = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None
    bmm_39: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_368, permute_253);  view_368 = permute_253 = None
    view_369: "f32[1, 4, 64, 512]" = torch.ops.aten.view.default(bmm_38, [1, 4, 64, 512]);  bmm_38 = None
    view_370: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_39, [1, 4, 512, 64]);  bmm_39 = None
    permute_254: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_369, [0, 1, 3, 2]);  view_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_255: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_370, [0, 2, 1, 3]);  view_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_32: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_255, memory_format = torch.contiguous_format);  permute_255 = None
    view_371: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_32, [1, 512, 256]);  clone_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_256: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_366, [0, 2, 1, 3]);  view_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_33: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_256, memory_format = torch.contiguous_format);  permute_256 = None
    view_372: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_33, [1, 512, 256]);  clone_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_373: "f32[512, 256]" = torch.ops.aten.view.default(view_372, [512, 256]);  view_372 = None
    permute_257: "f32[256, 256]" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
    mm_44: "f32[512, 256]" = torch.ops.aten.mm.default(view_373, permute_257);  permute_257 = None
    permute_258: "f32[256, 512]" = torch.ops.aten.permute.default(view_373, [1, 0])
    mm_45: "f32[256, 256]" = torch.ops.aten.mm.default(permute_258, view_183);  permute_258 = view_183 = None
    permute_259: "f32[256, 256]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_79: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_373, [0], True);  view_373 = None
    view_374: "f32[256]" = torch.ops.aten.view.default(sum_79, [256]);  sum_79 = None
    permute_260: "f32[256, 256]" = torch.ops.aten.permute.default(permute_259, [1, 0]);  permute_259 = None
    view_375: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_44, [1, 512, 256]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_124: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_199, view_375);  mul_199 = view_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_261: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(permute_254, [0, 2, 1, 3]);  permute_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_376: "f32[1, 512, 256]" = torch.ops.aten.view.default(permute_261, [1, 512, 256]);  permute_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_377: "f32[512, 256]" = torch.ops.aten.view.default(view_376, [512, 256]);  view_376 = None
    permute_262: "f32[256, 256]" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
    mm_46: "f32[512, 256]" = torch.ops.aten.mm.default(view_377, permute_262);  permute_262 = None
    permute_263: "f32[256, 512]" = torch.ops.aten.permute.default(view_377, [1, 0])
    mm_47: "f32[256, 256]" = torch.ops.aten.mm.default(permute_263, view_180);  permute_263 = view_180 = None
    permute_264: "f32[256, 256]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_80: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_377, [0], True);  view_377 = None
    view_378: "f32[256]" = torch.ops.aten.view.default(sum_80, [256]);  sum_80 = None
    permute_265: "f32[256, 256]" = torch.ops.aten.permute.default(permute_264, [1, 0]);  permute_264 = None
    view_379: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_46, [1, 512, 256]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_125: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_124, view_379);  add_124 = view_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_380: "f32[512, 256]" = torch.ops.aten.view.default(view_371, [512, 256]);  view_371 = None
    permute_266: "f32[256, 256]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
    mm_48: "f32[512, 256]" = torch.ops.aten.mm.default(view_380, permute_266);  permute_266 = None
    permute_267: "f32[256, 512]" = torch.ops.aten.permute.default(view_380, [1, 0])
    mm_49: "f32[256, 256]" = torch.ops.aten.mm.default(permute_267, view_178);  permute_267 = view_178 = None
    permute_268: "f32[256, 256]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_81: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_380, [0], True);  view_380 = None
    view_381: "f32[256]" = torch.ops.aten.view.default(sum_81, [256]);  sum_81 = None
    permute_269: "f32[256, 256]" = torch.ops.aten.permute.default(permute_268, [1, 0]);  permute_268 = None
    view_382: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_48, [1, 512, 256]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_126: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_125, view_382);  add_125 = view_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_72: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_65, getitem_83);  add_65 = getitem_83 = None
    mul_207: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_16);  sub_72 = None
    mul_208: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_126, primals_134);  primals_134 = None
    mul_209: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_208, 256)
    sum_82: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_208, [2], True)
    mul_210: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_208, mul_207);  mul_208 = None
    sum_83: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_210, [2], True);  mul_210 = None
    mul_211: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_207, sum_83);  sum_83 = None
    sub_73: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_209, sum_82);  mul_209 = sum_82 = None
    sub_74: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_73, mul_211);  sub_73 = mul_211 = None
    div_42: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 256);  rsqrt_16 = None
    mul_212: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_42, sub_74);  div_42 = sub_74 = None
    mul_213: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_126, mul_207);  mul_207 = None
    sum_84: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_213, [0, 1]);  mul_213 = None
    sum_85: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_126, [0, 1]);  add_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_14: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_81, torch.float32);  getitem_81 = None
    mul_214: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 1.1111111111111112);  convert_element_type_14 = None
    mul_215: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_212, mul_214);  mul_214 = None
    clone_34: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_215, memory_format = torch.contiguous_format);  mul_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_383: "f32[512, 256]" = torch.ops.aten.view.default(clone_34, [512, 256]);  clone_34 = None
    permute_270: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
    mm_50: "f32[512, 1024]" = torch.ops.aten.mm.default(view_383, permute_270);  permute_270 = None
    permute_271: "f32[256, 512]" = torch.ops.aten.permute.default(view_383, [1, 0])
    mm_51: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_271, view_176);  permute_271 = view_176 = None
    permute_272: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_86: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_383, [0], True);  view_383 = None
    view_384: "f32[256]" = torch.ops.aten.view.default(sum_86, [256]);  sum_86 = None
    permute_273: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_272, [1, 0]);  permute_272 = None
    view_385: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_50, [1, 512, 1024]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_216: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_175, 0.7071067811865476)
    erf_16: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_216);  mul_216 = None
    add_127: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_217: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_127, 0.5);  add_127 = None
    mul_218: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_175, view_175)
    mul_219: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_218, -0.5);  mul_218 = None
    exp_20: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_219);  mul_219 = None
    mul_220: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
    mul_221: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_175, mul_220);  view_175 = mul_220 = None
    add_128: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_217, mul_221);  mul_217 = mul_221 = None
    mul_222: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_385, add_128);  view_385 = add_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_386: "f32[512, 1024]" = torch.ops.aten.view.default(mul_222, [512, 1024]);  mul_222 = None
    permute_274: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
    mm_52: "f32[512, 256]" = torch.ops.aten.mm.default(view_386, permute_274);  permute_274 = None
    permute_275: "f32[1024, 512]" = torch.ops.aten.permute.default(view_386, [1, 0])
    mm_53: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_275, view_174);  permute_275 = view_174 = None
    permute_276: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_87: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_386, [0], True);  view_386 = None
    view_387: "f32[1024]" = torch.ops.aten.view.default(sum_87, [1024]);  sum_87 = None
    permute_277: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_276, [1, 0]);  permute_276 = None
    view_388: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_52, [1, 512, 256]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_129: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_212, view_388);  mul_212 = view_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_75: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_61, getitem_79);  add_61 = getitem_79 = None
    mul_223: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_15);  sub_75 = None
    mul_224: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_129, primals_128);  primals_128 = None
    mul_225: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_224, 256)
    sum_88: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_224, [2], True)
    mul_226: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_224, mul_223);  mul_224 = None
    sum_89: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_226, [2], True);  mul_226 = None
    mul_227: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_223, sum_89);  sum_89 = None
    sub_76: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_225, sum_88);  mul_225 = sum_88 = None
    sub_77: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_76, mul_227);  sub_76 = mul_227 = None
    div_43: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 256);  rsqrt_15 = None
    mul_228: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_43, sub_77);  div_43 = sub_77 = None
    mul_229: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_129, mul_223);  mul_223 = None
    sum_90: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_229, [0, 1]);  mul_229 = None
    sum_91: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_129, [0, 1]);  add_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_15: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_77, torch.float32);  getitem_77 = None
    mul_230: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_15, 1.1111111111111112);  convert_element_type_15 = None
    mul_231: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_228, mul_230);  mul_230 = None
    clone_35: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_231, memory_format = torch.contiguous_format);  mul_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_389: "f32[512, 256]" = torch.ops.aten.view.default(clone_35, [512, 256]);  clone_35 = None
    permute_278: "f32[256, 256]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
    mm_54: "f32[512, 256]" = torch.ops.aten.mm.default(view_389, permute_278);  permute_278 = None
    permute_279: "f32[256, 512]" = torch.ops.aten.permute.default(view_389, [1, 0])
    mm_55: "f32[256, 256]" = torch.ops.aten.mm.default(permute_279, view_172);  permute_279 = view_172 = None
    permute_280: "f32[256, 256]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_92: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_389, [0], True);  view_389 = None
    view_390: "f32[256]" = torch.ops.aten.view.default(sum_92, [256]);  sum_92 = None
    permute_281: "f32[256, 256]" = torch.ops.aten.permute.default(permute_280, [1, 0]);  permute_280 = None
    view_391: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_54, [1, 512, 256]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_392: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_391, [1, 512, 4, 64]);  view_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_282: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_392, [0, 2, 1, 3]);  view_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_393: "f32[4, 512, 64]" = torch.ops.aten.view.default(permute_282, [4, 512, 64]);  permute_282 = None
    permute_283: "f32[4, 512, 512]" = torch.ops.aten.permute.default(view_168, [0, 2, 1]);  view_168 = None
    bmm_40: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(permute_283, view_393);  permute_283 = None
    permute_284: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_169, [0, 2, 1]);  view_169 = None
    bmm_41: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_393, permute_284);  view_393 = permute_284 = None
    view_394: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_40, [1, 4, 512, 64]);  bmm_40 = None
    view_395: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_41, [1, 4, 512, 512]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_16: "f32[1, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_75, torch.float32);  getitem_75 = None
    mul_232: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_16, 1.1111111111111112);  convert_element_type_16 = None
    mul_233: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_395, mul_232);  view_395 = mul_232 = None
    clone_36: "f32[1, 4, 512, 512]" = torch.ops.aten.clone.default(mul_233, memory_format = torch.contiguous_format);  mul_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_20: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    mul_234: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(clone_36, alias_20);  clone_36 = None
    sum_93: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_234, [-1], True)
    mul_235: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(alias_20, sum_93);  alias_20 = sum_93 = None
    sub_78: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(mul_234, mul_235);  mul_234 = mul_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_44: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(sub_78, 8.0);  sub_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_396: "f32[4, 512, 512]" = torch.ops.aten.view.default(div_44, [4, 512, 512]);  div_44 = None
    permute_285: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_165, [0, 2, 1]);  view_165 = None
    bmm_42: "f32[4, 64, 512]" = torch.ops.aten.bmm.default(permute_285, view_396);  permute_285 = None
    permute_286: "f32[4, 512, 64]" = torch.ops.aten.permute.default(view_166, [0, 2, 1]);  view_166 = None
    bmm_43: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_396, permute_286);  view_396 = permute_286 = None
    view_397: "f32[1, 4, 64, 512]" = torch.ops.aten.view.default(bmm_42, [1, 4, 64, 512]);  bmm_42 = None
    view_398: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_43, [1, 4, 512, 64]);  bmm_43 = None
    permute_287: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_397, [0, 1, 3, 2]);  view_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_288: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_398, [0, 2, 1, 3]);  view_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_37: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_288, memory_format = torch.contiguous_format);  permute_288 = None
    view_399: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_37, [1, 512, 256]);  clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_289: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_394, [0, 2, 1, 3]);  view_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_38: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_289, memory_format = torch.contiguous_format);  permute_289 = None
    view_400: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_38, [1, 512, 256]);  clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_401: "f32[512, 256]" = torch.ops.aten.view.default(view_400, [512, 256]);  view_400 = None
    permute_290: "f32[256, 256]" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
    mm_56: "f32[512, 256]" = torch.ops.aten.mm.default(view_401, permute_290);  permute_290 = None
    permute_291: "f32[256, 512]" = torch.ops.aten.permute.default(view_401, [1, 0])
    mm_57: "f32[256, 256]" = torch.ops.aten.mm.default(permute_291, view_161);  permute_291 = view_161 = None
    permute_292: "f32[256, 256]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_94: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_401, [0], True);  view_401 = None
    view_402: "f32[256]" = torch.ops.aten.view.default(sum_94, [256]);  sum_94 = None
    permute_293: "f32[256, 256]" = torch.ops.aten.permute.default(permute_292, [1, 0]);  permute_292 = None
    view_403: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_56, [1, 512, 256]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_130: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_228, view_403);  mul_228 = view_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_294: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(permute_287, [0, 2, 1, 3]);  permute_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_404: "f32[1, 512, 256]" = torch.ops.aten.view.default(permute_294, [1, 512, 256]);  permute_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_405: "f32[512, 256]" = torch.ops.aten.view.default(view_404, [512, 256]);  view_404 = None
    permute_295: "f32[256, 256]" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
    mm_58: "f32[512, 256]" = torch.ops.aten.mm.default(view_405, permute_295);  permute_295 = None
    permute_296: "f32[256, 512]" = torch.ops.aten.permute.default(view_405, [1, 0])
    mm_59: "f32[256, 256]" = torch.ops.aten.mm.default(permute_296, view_158);  permute_296 = view_158 = None
    permute_297: "f32[256, 256]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_95: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_405, [0], True);  view_405 = None
    view_406: "f32[256]" = torch.ops.aten.view.default(sum_95, [256]);  sum_95 = None
    permute_298: "f32[256, 256]" = torch.ops.aten.permute.default(permute_297, [1, 0]);  permute_297 = None
    view_407: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_58, [1, 512, 256]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_131: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_130, view_407);  add_130 = view_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_408: "f32[512, 256]" = torch.ops.aten.view.default(view_399, [512, 256]);  view_399 = None
    permute_299: "f32[256, 256]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
    mm_60: "f32[512, 256]" = torch.ops.aten.mm.default(view_408, permute_299);  permute_299 = None
    permute_300: "f32[256, 512]" = torch.ops.aten.permute.default(view_408, [1, 0])
    mm_61: "f32[256, 256]" = torch.ops.aten.mm.default(permute_300, view_156);  permute_300 = view_156 = None
    permute_301: "f32[256, 256]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_96: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_408, [0], True);  view_408 = None
    view_409: "f32[256]" = torch.ops.aten.view.default(sum_96, [256]);  sum_96 = None
    permute_302: "f32[256, 256]" = torch.ops.aten.permute.default(permute_301, [1, 0]);  permute_301 = None
    view_410: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_60, [1, 512, 256]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_132: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_131, view_410);  add_131 = view_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_79: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_57, getitem_73);  add_57 = getitem_73 = None
    mul_236: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_14);  sub_79 = None
    mul_237: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_132, primals_118);  primals_118 = None
    mul_238: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_237, 256)
    sum_97: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_237, [2], True)
    mul_239: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_237, mul_236);  mul_237 = None
    sum_98: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_239, [2], True);  mul_239 = None
    mul_240: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_236, sum_98);  sum_98 = None
    sub_80: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_238, sum_97);  mul_238 = sum_97 = None
    sub_81: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_80, mul_240);  sub_80 = mul_240 = None
    div_45: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 256);  rsqrt_14 = None
    mul_241: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_45, sub_81);  div_45 = sub_81 = None
    mul_242: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_132, mul_236);  mul_236 = None
    sum_99: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_242, [0, 1]);  mul_242 = None
    sum_100: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_132, [0, 1]);  add_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_17: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_71, torch.float32);  getitem_71 = None
    mul_243: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_17, 1.1111111111111112);  convert_element_type_17 = None
    mul_244: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_241, mul_243);  mul_243 = None
    clone_39: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_244, memory_format = torch.contiguous_format);  mul_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_411: "f32[512, 256]" = torch.ops.aten.view.default(clone_39, [512, 256]);  clone_39 = None
    permute_303: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
    mm_62: "f32[512, 1024]" = torch.ops.aten.mm.default(view_411, permute_303);  permute_303 = None
    permute_304: "f32[256, 512]" = torch.ops.aten.permute.default(view_411, [1, 0])
    mm_63: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_304, view_154);  permute_304 = view_154 = None
    permute_305: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_101: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_411, [0], True);  view_411 = None
    view_412: "f32[256]" = torch.ops.aten.view.default(sum_101, [256]);  sum_101 = None
    permute_306: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_305, [1, 0]);  permute_305 = None
    view_413: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_62, [1, 512, 1024]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_245: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_153, 0.7071067811865476)
    erf_17: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_245);  mul_245 = None
    add_133: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_246: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_133, 0.5);  add_133 = None
    mul_247: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_153, view_153)
    mul_248: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_247, -0.5);  mul_247 = None
    exp_21: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_248);  mul_248 = None
    mul_249: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
    mul_250: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_153, mul_249);  view_153 = mul_249 = None
    add_134: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_246, mul_250);  mul_246 = mul_250 = None
    mul_251: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_413, add_134);  view_413 = add_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_414: "f32[512, 1024]" = torch.ops.aten.view.default(mul_251, [512, 1024]);  mul_251 = None
    permute_307: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
    mm_64: "f32[512, 256]" = torch.ops.aten.mm.default(view_414, permute_307);  permute_307 = None
    permute_308: "f32[1024, 512]" = torch.ops.aten.permute.default(view_414, [1, 0])
    mm_65: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_308, view_152);  permute_308 = view_152 = None
    permute_309: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_102: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_414, [0], True);  view_414 = None
    view_415: "f32[1024]" = torch.ops.aten.view.default(sum_102, [1024]);  sum_102 = None
    permute_310: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_309, [1, 0]);  permute_309 = None
    view_416: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_64, [1, 512, 256]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_135: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_241, view_416);  mul_241 = view_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_82: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_53, getitem_69);  add_53 = getitem_69 = None
    mul_252: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_13);  sub_82 = None
    mul_253: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_135, primals_112);  primals_112 = None
    mul_254: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_253, 256)
    sum_103: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_253, [2], True)
    mul_255: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_253, mul_252);  mul_253 = None
    sum_104: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_255, [2], True);  mul_255 = None
    mul_256: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_252, sum_104);  sum_104 = None
    sub_83: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_254, sum_103);  mul_254 = sum_103 = None
    sub_84: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_83, mul_256);  sub_83 = mul_256 = None
    div_46: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 256);  rsqrt_13 = None
    mul_257: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_46, sub_84);  div_46 = sub_84 = None
    mul_258: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_135, mul_252);  mul_252 = None
    sum_105: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_258, [0, 1]);  mul_258 = None
    sum_106: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_135, [0, 1]);  add_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_18: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_67, torch.float32);  getitem_67 = None
    mul_259: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_18, 1.1111111111111112);  convert_element_type_18 = None
    mul_260: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_257, mul_259);  mul_259 = None
    clone_40: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_260, memory_format = torch.contiguous_format);  mul_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_417: "f32[512, 256]" = torch.ops.aten.view.default(clone_40, [512, 256]);  clone_40 = None
    permute_311: "f32[256, 256]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
    mm_66: "f32[512, 256]" = torch.ops.aten.mm.default(view_417, permute_311);  permute_311 = None
    permute_312: "f32[256, 512]" = torch.ops.aten.permute.default(view_417, [1, 0])
    mm_67: "f32[256, 256]" = torch.ops.aten.mm.default(permute_312, view_150);  permute_312 = view_150 = None
    permute_313: "f32[256, 256]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_107: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_417, [0], True);  view_417 = None
    view_418: "f32[256]" = torch.ops.aten.view.default(sum_107, [256]);  sum_107 = None
    permute_314: "f32[256, 256]" = torch.ops.aten.permute.default(permute_313, [1, 0]);  permute_313 = None
    view_419: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_66, [1, 512, 256]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_420: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_419, [1, 512, 4, 64]);  view_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_315: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_420, [0, 2, 1, 3]);  view_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_421: "f32[4, 512, 64]" = torch.ops.aten.view.default(permute_315, [4, 512, 64]);  permute_315 = None
    permute_316: "f32[4, 512, 512]" = torch.ops.aten.permute.default(view_146, [0, 2, 1]);  view_146 = None
    bmm_44: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(permute_316, view_421);  permute_316 = None
    permute_317: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_147, [0, 2, 1]);  view_147 = None
    bmm_45: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_421, permute_317);  view_421 = permute_317 = None
    view_422: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_44, [1, 4, 512, 64]);  bmm_44 = None
    view_423: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_45, [1, 4, 512, 512]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_19: "f32[1, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_65, torch.float32);  getitem_65 = None
    mul_261: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_19, 1.1111111111111112);  convert_element_type_19 = None
    mul_262: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_423, mul_261);  view_423 = mul_261 = None
    clone_41: "f32[1, 4, 512, 512]" = torch.ops.aten.clone.default(mul_262, memory_format = torch.contiguous_format);  mul_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_21: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    mul_263: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(clone_41, alias_21);  clone_41 = None
    sum_108: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_263, [-1], True)
    mul_264: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(alias_21, sum_108);  alias_21 = sum_108 = None
    sub_85: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(mul_263, mul_264);  mul_263 = mul_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_47: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(sub_85, 8.0);  sub_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_424: "f32[4, 512, 512]" = torch.ops.aten.view.default(div_47, [4, 512, 512]);  div_47 = None
    permute_318: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_143, [0, 2, 1]);  view_143 = None
    bmm_46: "f32[4, 64, 512]" = torch.ops.aten.bmm.default(permute_318, view_424);  permute_318 = None
    permute_319: "f32[4, 512, 64]" = torch.ops.aten.permute.default(view_144, [0, 2, 1]);  view_144 = None
    bmm_47: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_424, permute_319);  view_424 = permute_319 = None
    view_425: "f32[1, 4, 64, 512]" = torch.ops.aten.view.default(bmm_46, [1, 4, 64, 512]);  bmm_46 = None
    view_426: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_47, [1, 4, 512, 64]);  bmm_47 = None
    permute_320: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_425, [0, 1, 3, 2]);  view_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_321: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_426, [0, 2, 1, 3]);  view_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_42: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_321, memory_format = torch.contiguous_format);  permute_321 = None
    view_427: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_42, [1, 512, 256]);  clone_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_322: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_422, [0, 2, 1, 3]);  view_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_43: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_322, memory_format = torch.contiguous_format);  permute_322 = None
    view_428: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_43, [1, 512, 256]);  clone_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_429: "f32[512, 256]" = torch.ops.aten.view.default(view_428, [512, 256]);  view_428 = None
    permute_323: "f32[256, 256]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
    mm_68: "f32[512, 256]" = torch.ops.aten.mm.default(view_429, permute_323);  permute_323 = None
    permute_324: "f32[256, 512]" = torch.ops.aten.permute.default(view_429, [1, 0])
    mm_69: "f32[256, 256]" = torch.ops.aten.mm.default(permute_324, view_139);  permute_324 = view_139 = None
    permute_325: "f32[256, 256]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_109: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_429, [0], True);  view_429 = None
    view_430: "f32[256]" = torch.ops.aten.view.default(sum_109, [256]);  sum_109 = None
    permute_326: "f32[256, 256]" = torch.ops.aten.permute.default(permute_325, [1, 0]);  permute_325 = None
    view_431: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_68, [1, 512, 256]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_136: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_257, view_431);  mul_257 = view_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_327: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(permute_320, [0, 2, 1, 3]);  permute_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_432: "f32[1, 512, 256]" = torch.ops.aten.view.default(permute_327, [1, 512, 256]);  permute_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_433: "f32[512, 256]" = torch.ops.aten.view.default(view_432, [512, 256]);  view_432 = None
    permute_328: "f32[256, 256]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
    mm_70: "f32[512, 256]" = torch.ops.aten.mm.default(view_433, permute_328);  permute_328 = None
    permute_329: "f32[256, 512]" = torch.ops.aten.permute.default(view_433, [1, 0])
    mm_71: "f32[256, 256]" = torch.ops.aten.mm.default(permute_329, view_136);  permute_329 = view_136 = None
    permute_330: "f32[256, 256]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_110: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_433, [0], True);  view_433 = None
    view_434: "f32[256]" = torch.ops.aten.view.default(sum_110, [256]);  sum_110 = None
    permute_331: "f32[256, 256]" = torch.ops.aten.permute.default(permute_330, [1, 0]);  permute_330 = None
    view_435: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_70, [1, 512, 256]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_137: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_136, view_435);  add_136 = view_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_436: "f32[512, 256]" = torch.ops.aten.view.default(view_427, [512, 256]);  view_427 = None
    permute_332: "f32[256, 256]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
    mm_72: "f32[512, 256]" = torch.ops.aten.mm.default(view_436, permute_332);  permute_332 = None
    permute_333: "f32[256, 512]" = torch.ops.aten.permute.default(view_436, [1, 0])
    mm_73: "f32[256, 256]" = torch.ops.aten.mm.default(permute_333, view_134);  permute_333 = view_134 = None
    permute_334: "f32[256, 256]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_111: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_436, [0], True);  view_436 = None
    view_437: "f32[256]" = torch.ops.aten.view.default(sum_111, [256]);  sum_111 = None
    permute_335: "f32[256, 256]" = torch.ops.aten.permute.default(permute_334, [1, 0]);  permute_334 = None
    view_438: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_72, [1, 512, 256]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_138: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_137, view_438);  add_137 = view_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_86: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_49, getitem_63);  add_49 = getitem_63 = None
    mul_265: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_12);  sub_86 = None
    mul_266: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_138, primals_102);  primals_102 = None
    mul_267: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_266, 256)
    sum_112: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_266, [2], True)
    mul_268: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_266, mul_265);  mul_266 = None
    sum_113: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_268, [2], True);  mul_268 = None
    mul_269: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_265, sum_113);  sum_113 = None
    sub_87: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_267, sum_112);  mul_267 = sum_112 = None
    sub_88: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_87, mul_269);  sub_87 = mul_269 = None
    div_48: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 256);  rsqrt_12 = None
    mul_270: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_48, sub_88);  div_48 = sub_88 = None
    mul_271: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_138, mul_265);  mul_265 = None
    sum_114: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_271, [0, 1]);  mul_271 = None
    sum_115: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_138, [0, 1]);  add_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_20: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_61, torch.float32);  getitem_61 = None
    mul_272: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.1111111111111112);  convert_element_type_20 = None
    mul_273: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_270, mul_272);  mul_272 = None
    clone_44: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_273, memory_format = torch.contiguous_format);  mul_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_439: "f32[512, 256]" = torch.ops.aten.view.default(clone_44, [512, 256]);  clone_44 = None
    permute_336: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    mm_74: "f32[512, 1024]" = torch.ops.aten.mm.default(view_439, permute_336);  permute_336 = None
    permute_337: "f32[256, 512]" = torch.ops.aten.permute.default(view_439, [1, 0])
    mm_75: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_337, view_132);  permute_337 = view_132 = None
    permute_338: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_116: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_439, [0], True);  view_439 = None
    view_440: "f32[256]" = torch.ops.aten.view.default(sum_116, [256]);  sum_116 = None
    permute_339: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_338, [1, 0]);  permute_338 = None
    view_441: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_74, [1, 512, 1024]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_274: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_131, 0.7071067811865476)
    erf_18: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_274);  mul_274 = None
    add_139: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_275: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_139, 0.5);  add_139 = None
    mul_276: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_131, view_131)
    mul_277: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_276, -0.5);  mul_276 = None
    exp_22: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_277);  mul_277 = None
    mul_278: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
    mul_279: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_131, mul_278);  view_131 = mul_278 = None
    add_140: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_275, mul_279);  mul_275 = mul_279 = None
    mul_280: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_441, add_140);  view_441 = add_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_442: "f32[512, 1024]" = torch.ops.aten.view.default(mul_280, [512, 1024]);  mul_280 = None
    permute_340: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    mm_76: "f32[512, 256]" = torch.ops.aten.mm.default(view_442, permute_340);  permute_340 = None
    permute_341: "f32[1024, 512]" = torch.ops.aten.permute.default(view_442, [1, 0])
    mm_77: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_341, view_130);  permute_341 = view_130 = None
    permute_342: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_117: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_442, [0], True);  view_442 = None
    view_443: "f32[1024]" = torch.ops.aten.view.default(sum_117, [1024]);  sum_117 = None
    permute_343: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_342, [1, 0]);  permute_342 = None
    view_444: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_76, [1, 512, 256]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_141: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_270, view_444);  mul_270 = view_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_89: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_45, getitem_59);  add_45 = getitem_59 = None
    mul_281: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_89, rsqrt_11);  sub_89 = None
    mul_282: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_141, primals_96);  primals_96 = None
    mul_283: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_282, 256)
    sum_118: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_282, [2], True)
    mul_284: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_282, mul_281);  mul_282 = None
    sum_119: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_284, [2], True);  mul_284 = None
    mul_285: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_281, sum_119);  sum_119 = None
    sub_90: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_283, sum_118);  mul_283 = sum_118 = None
    sub_91: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_90, mul_285);  sub_90 = mul_285 = None
    div_49: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 256);  rsqrt_11 = None
    mul_286: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_49, sub_91);  div_49 = sub_91 = None
    mul_287: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_141, mul_281);  mul_281 = None
    sum_120: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_287, [0, 1]);  mul_287 = None
    sum_121: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_141, [0, 1]);  add_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_21: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_57, torch.float32);  getitem_57 = None
    mul_288: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 1.1111111111111112);  convert_element_type_21 = None
    mul_289: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_286, mul_288);  mul_288 = None
    clone_45: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_289, memory_format = torch.contiguous_format);  mul_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_445: "f32[512, 256]" = torch.ops.aten.view.default(clone_45, [512, 256]);  clone_45 = None
    permute_344: "f32[256, 256]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
    mm_78: "f32[512, 256]" = torch.ops.aten.mm.default(view_445, permute_344);  permute_344 = None
    permute_345: "f32[256, 512]" = torch.ops.aten.permute.default(view_445, [1, 0])
    mm_79: "f32[256, 256]" = torch.ops.aten.mm.default(permute_345, view_128);  permute_345 = view_128 = None
    permute_346: "f32[256, 256]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_122: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_445, [0], True);  view_445 = None
    view_446: "f32[256]" = torch.ops.aten.view.default(sum_122, [256]);  sum_122 = None
    permute_347: "f32[256, 256]" = torch.ops.aten.permute.default(permute_346, [1, 0]);  permute_346 = None
    view_447: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_78, [1, 512, 256]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_448: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_447, [1, 512, 4, 64]);  view_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_348: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_448, [0, 2, 1, 3]);  view_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_449: "f32[4, 512, 64]" = torch.ops.aten.view.default(permute_348, [4, 512, 64]);  permute_348 = None
    permute_349: "f32[4, 512, 512]" = torch.ops.aten.permute.default(view_124, [0, 2, 1]);  view_124 = None
    bmm_48: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(permute_349, view_449);  permute_349 = None
    permute_350: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_125, [0, 2, 1]);  view_125 = None
    bmm_49: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_449, permute_350);  view_449 = permute_350 = None
    view_450: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_48, [1, 4, 512, 64]);  bmm_48 = None
    view_451: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_49, [1, 4, 512, 512]);  bmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_22: "f32[1, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_55, torch.float32);  getitem_55 = None
    mul_290: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 1.1111111111111112);  convert_element_type_22 = None
    mul_291: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_451, mul_290);  view_451 = mul_290 = None
    clone_46: "f32[1, 4, 512, 512]" = torch.ops.aten.clone.default(mul_291, memory_format = torch.contiguous_format);  mul_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_22: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    mul_292: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(clone_46, alias_22);  clone_46 = None
    sum_123: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_292, [-1], True)
    mul_293: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(alias_22, sum_123);  alias_22 = sum_123 = None
    sub_92: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(mul_292, mul_293);  mul_292 = mul_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_50: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(sub_92, 8.0);  sub_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_452: "f32[4, 512, 512]" = torch.ops.aten.view.default(div_50, [4, 512, 512]);  div_50 = None
    permute_351: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_121, [0, 2, 1]);  view_121 = None
    bmm_50: "f32[4, 64, 512]" = torch.ops.aten.bmm.default(permute_351, view_452);  permute_351 = None
    permute_352: "f32[4, 512, 64]" = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None
    bmm_51: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_452, permute_352);  view_452 = permute_352 = None
    view_453: "f32[1, 4, 64, 512]" = torch.ops.aten.view.default(bmm_50, [1, 4, 64, 512]);  bmm_50 = None
    view_454: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_51, [1, 4, 512, 64]);  bmm_51 = None
    permute_353: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_453, [0, 1, 3, 2]);  view_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_354: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_454, [0, 2, 1, 3]);  view_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_47: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_354, memory_format = torch.contiguous_format);  permute_354 = None
    view_455: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_47, [1, 512, 256]);  clone_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_355: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_450, [0, 2, 1, 3]);  view_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_48: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_355, memory_format = torch.contiguous_format);  permute_355 = None
    view_456: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_48, [1, 512, 256]);  clone_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_457: "f32[512, 256]" = torch.ops.aten.view.default(view_456, [512, 256]);  view_456 = None
    permute_356: "f32[256, 256]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
    mm_80: "f32[512, 256]" = torch.ops.aten.mm.default(view_457, permute_356);  permute_356 = None
    permute_357: "f32[256, 512]" = torch.ops.aten.permute.default(view_457, [1, 0])
    mm_81: "f32[256, 256]" = torch.ops.aten.mm.default(permute_357, view_117);  permute_357 = view_117 = None
    permute_358: "f32[256, 256]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_124: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_457, [0], True);  view_457 = None
    view_458: "f32[256]" = torch.ops.aten.view.default(sum_124, [256]);  sum_124 = None
    permute_359: "f32[256, 256]" = torch.ops.aten.permute.default(permute_358, [1, 0]);  permute_358 = None
    view_459: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_80, [1, 512, 256]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_142: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_286, view_459);  mul_286 = view_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_360: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(permute_353, [0, 2, 1, 3]);  permute_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_460: "f32[1, 512, 256]" = torch.ops.aten.view.default(permute_360, [1, 512, 256]);  permute_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_461: "f32[512, 256]" = torch.ops.aten.view.default(view_460, [512, 256]);  view_460 = None
    permute_361: "f32[256, 256]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
    mm_82: "f32[512, 256]" = torch.ops.aten.mm.default(view_461, permute_361);  permute_361 = None
    permute_362: "f32[256, 512]" = torch.ops.aten.permute.default(view_461, [1, 0])
    mm_83: "f32[256, 256]" = torch.ops.aten.mm.default(permute_362, view_114);  permute_362 = view_114 = None
    permute_363: "f32[256, 256]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_125: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_461, [0], True);  view_461 = None
    view_462: "f32[256]" = torch.ops.aten.view.default(sum_125, [256]);  sum_125 = None
    permute_364: "f32[256, 256]" = torch.ops.aten.permute.default(permute_363, [1, 0]);  permute_363 = None
    view_463: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_82, [1, 512, 256]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_143: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_142, view_463);  add_142 = view_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_464: "f32[512, 256]" = torch.ops.aten.view.default(view_455, [512, 256]);  view_455 = None
    permute_365: "f32[256, 256]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    mm_84: "f32[512, 256]" = torch.ops.aten.mm.default(view_464, permute_365);  permute_365 = None
    permute_366: "f32[256, 512]" = torch.ops.aten.permute.default(view_464, [1, 0])
    mm_85: "f32[256, 256]" = torch.ops.aten.mm.default(permute_366, view_112);  permute_366 = view_112 = None
    permute_367: "f32[256, 256]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_126: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_464, [0], True);  view_464 = None
    view_465: "f32[256]" = torch.ops.aten.view.default(sum_126, [256]);  sum_126 = None
    permute_368: "f32[256, 256]" = torch.ops.aten.permute.default(permute_367, [1, 0]);  permute_367 = None
    view_466: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_84, [1, 512, 256]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_144: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_143, view_466);  add_143 = view_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_93: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_41, getitem_53);  add_41 = getitem_53 = None
    mul_294: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_93, rsqrt_10);  sub_93 = None
    mul_295: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_144, primals_86);  primals_86 = None
    mul_296: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_295, 256)
    sum_127: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_295, [2], True)
    mul_297: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_295, mul_294);  mul_295 = None
    sum_128: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_297, [2], True);  mul_297 = None
    mul_298: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_294, sum_128);  sum_128 = None
    sub_94: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_296, sum_127);  mul_296 = sum_127 = None
    sub_95: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_94, mul_298);  sub_94 = mul_298 = None
    div_51: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 256);  rsqrt_10 = None
    mul_299: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_51, sub_95);  div_51 = sub_95 = None
    mul_300: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_144, mul_294);  mul_294 = None
    sum_129: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_300, [0, 1]);  mul_300 = None
    sum_130: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_144, [0, 1]);  add_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_23: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_51, torch.float32);  getitem_51 = None
    mul_301: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_23, 1.1111111111111112);  convert_element_type_23 = None
    mul_302: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_299, mul_301);  mul_301 = None
    clone_49: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_302, memory_format = torch.contiguous_format);  mul_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_467: "f32[512, 256]" = torch.ops.aten.view.default(clone_49, [512, 256]);  clone_49 = None
    permute_369: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    mm_86: "f32[512, 1024]" = torch.ops.aten.mm.default(view_467, permute_369);  permute_369 = None
    permute_370: "f32[256, 512]" = torch.ops.aten.permute.default(view_467, [1, 0])
    mm_87: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_370, view_110);  permute_370 = view_110 = None
    permute_371: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
    sum_131: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_467, [0], True);  view_467 = None
    view_468: "f32[256]" = torch.ops.aten.view.default(sum_131, [256]);  sum_131 = None
    permute_372: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_371, [1, 0]);  permute_371 = None
    view_469: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_86, [1, 512, 1024]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_303: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_109, 0.7071067811865476)
    erf_19: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_303);  mul_303 = None
    add_145: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_304: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_145, 0.5);  add_145 = None
    mul_305: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_109, view_109)
    mul_306: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_305, -0.5);  mul_305 = None
    exp_23: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_306);  mul_306 = None
    mul_307: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
    mul_308: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_109, mul_307);  view_109 = mul_307 = None
    add_146: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_304, mul_308);  mul_304 = mul_308 = None
    mul_309: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_469, add_146);  view_469 = add_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_470: "f32[512, 1024]" = torch.ops.aten.view.default(mul_309, [512, 1024]);  mul_309 = None
    permute_373: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    mm_88: "f32[512, 256]" = torch.ops.aten.mm.default(view_470, permute_373);  permute_373 = None
    permute_374: "f32[1024, 512]" = torch.ops.aten.permute.default(view_470, [1, 0])
    mm_89: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_374, view_108);  permute_374 = view_108 = None
    permute_375: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    sum_132: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_470, [0], True);  view_470 = None
    view_471: "f32[1024]" = torch.ops.aten.view.default(sum_132, [1024]);  sum_132 = None
    permute_376: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_375, [1, 0]);  permute_375 = None
    view_472: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_88, [1, 512, 256]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_147: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_299, view_472);  mul_299 = view_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_96: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_37, getitem_49);  add_37 = getitem_49 = None
    mul_310: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_96, rsqrt_9);  sub_96 = None
    mul_311: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_147, primals_80);  primals_80 = None
    mul_312: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_311, 256)
    sum_133: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_311, [2], True)
    mul_313: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_311, mul_310);  mul_311 = None
    sum_134: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_313, [2], True);  mul_313 = None
    mul_314: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_310, sum_134);  sum_134 = None
    sub_97: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_312, sum_133);  mul_312 = sum_133 = None
    sub_98: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_97, mul_314);  sub_97 = mul_314 = None
    div_52: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 256);  rsqrt_9 = None
    mul_315: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_52, sub_98);  div_52 = sub_98 = None
    mul_316: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_147, mul_310);  mul_310 = None
    sum_135: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_316, [0, 1]);  mul_316 = None
    sum_136: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_147, [0, 1]);  add_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_24: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_47, torch.float32);  getitem_47 = None
    mul_317: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 1.1111111111111112);  convert_element_type_24 = None
    mul_318: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_315, mul_317);  mul_317 = None
    clone_50: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_318, memory_format = torch.contiguous_format);  mul_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_473: "f32[512, 256]" = torch.ops.aten.view.default(clone_50, [512, 256]);  clone_50 = None
    permute_377: "f32[256, 256]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
    mm_90: "f32[512, 256]" = torch.ops.aten.mm.default(view_473, permute_377);  permute_377 = None
    permute_378: "f32[256, 512]" = torch.ops.aten.permute.default(view_473, [1, 0])
    mm_91: "f32[256, 256]" = torch.ops.aten.mm.default(permute_378, view_106);  permute_378 = view_106 = None
    permute_379: "f32[256, 256]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_137: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_473, [0], True);  view_473 = None
    view_474: "f32[256]" = torch.ops.aten.view.default(sum_137, [256]);  sum_137 = None
    permute_380: "f32[256, 256]" = torch.ops.aten.permute.default(permute_379, [1, 0]);  permute_379 = None
    view_475: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_90, [1, 512, 256]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_476: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_475, [1, 512, 4, 64]);  view_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_381: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_476, [0, 2, 1, 3]);  view_476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_477: "f32[4, 512, 64]" = torch.ops.aten.view.default(permute_381, [4, 512, 64]);  permute_381 = None
    permute_382: "f32[4, 512, 512]" = torch.ops.aten.permute.default(view_102, [0, 2, 1]);  view_102 = None
    bmm_52: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(permute_382, view_477);  permute_382 = None
    permute_383: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_103, [0, 2, 1]);  view_103 = None
    bmm_53: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_477, permute_383);  view_477 = permute_383 = None
    view_478: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_52, [1, 4, 512, 64]);  bmm_52 = None
    view_479: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_53, [1, 4, 512, 512]);  bmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_25: "f32[1, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_45, torch.float32);  getitem_45 = None
    mul_319: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.1111111111111112);  convert_element_type_25 = None
    mul_320: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_479, mul_319);  view_479 = mul_319 = None
    clone_51: "f32[1, 4, 512, 512]" = torch.ops.aten.clone.default(mul_320, memory_format = torch.contiguous_format);  mul_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_23: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    mul_321: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(clone_51, alias_23);  clone_51 = None
    sum_138: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_321, [-1], True)
    mul_322: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(alias_23, sum_138);  alias_23 = sum_138 = None
    sub_99: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(mul_321, mul_322);  mul_321 = mul_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_53: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(sub_99, 8.0);  sub_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_480: "f32[4, 512, 512]" = torch.ops.aten.view.default(div_53, [4, 512, 512]);  div_53 = None
    permute_384: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_99, [0, 2, 1]);  view_99 = None
    bmm_54: "f32[4, 64, 512]" = torch.ops.aten.bmm.default(permute_384, view_480);  permute_384 = None
    permute_385: "f32[4, 512, 64]" = torch.ops.aten.permute.default(view_100, [0, 2, 1]);  view_100 = None
    bmm_55: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_480, permute_385);  view_480 = permute_385 = None
    view_481: "f32[1, 4, 64, 512]" = torch.ops.aten.view.default(bmm_54, [1, 4, 64, 512]);  bmm_54 = None
    view_482: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_55, [1, 4, 512, 64]);  bmm_55 = None
    permute_386: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_481, [0, 1, 3, 2]);  view_481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_387: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_482, [0, 2, 1, 3]);  view_482 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_52: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_387, memory_format = torch.contiguous_format);  permute_387 = None
    view_483: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_52, [1, 512, 256]);  clone_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_388: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_478, [0, 2, 1, 3]);  view_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_53: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_388, memory_format = torch.contiguous_format);  permute_388 = None
    view_484: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_53, [1, 512, 256]);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_485: "f32[512, 256]" = torch.ops.aten.view.default(view_484, [512, 256]);  view_484 = None
    permute_389: "f32[256, 256]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
    mm_92: "f32[512, 256]" = torch.ops.aten.mm.default(view_485, permute_389);  permute_389 = None
    permute_390: "f32[256, 512]" = torch.ops.aten.permute.default(view_485, [1, 0])
    mm_93: "f32[256, 256]" = torch.ops.aten.mm.default(permute_390, view_95);  permute_390 = view_95 = None
    permute_391: "f32[256, 256]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_139: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_485, [0], True);  view_485 = None
    view_486: "f32[256]" = torch.ops.aten.view.default(sum_139, [256]);  sum_139 = None
    permute_392: "f32[256, 256]" = torch.ops.aten.permute.default(permute_391, [1, 0]);  permute_391 = None
    view_487: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_92, [1, 512, 256]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_148: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_315, view_487);  mul_315 = view_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_393: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(permute_386, [0, 2, 1, 3]);  permute_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_488: "f32[1, 512, 256]" = torch.ops.aten.view.default(permute_393, [1, 512, 256]);  permute_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_489: "f32[512, 256]" = torch.ops.aten.view.default(view_488, [512, 256]);  view_488 = None
    permute_394: "f32[256, 256]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
    mm_94: "f32[512, 256]" = torch.ops.aten.mm.default(view_489, permute_394);  permute_394 = None
    permute_395: "f32[256, 512]" = torch.ops.aten.permute.default(view_489, [1, 0])
    mm_95: "f32[256, 256]" = torch.ops.aten.mm.default(permute_395, view_92);  permute_395 = view_92 = None
    permute_396: "f32[256, 256]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_140: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_489, [0], True);  view_489 = None
    view_490: "f32[256]" = torch.ops.aten.view.default(sum_140, [256]);  sum_140 = None
    permute_397: "f32[256, 256]" = torch.ops.aten.permute.default(permute_396, [1, 0]);  permute_396 = None
    view_491: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_94, [1, 512, 256]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_149: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_148, view_491);  add_148 = view_491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_492: "f32[512, 256]" = torch.ops.aten.view.default(view_483, [512, 256]);  view_483 = None
    permute_398: "f32[256, 256]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
    mm_96: "f32[512, 256]" = torch.ops.aten.mm.default(view_492, permute_398);  permute_398 = None
    permute_399: "f32[256, 512]" = torch.ops.aten.permute.default(view_492, [1, 0])
    mm_97: "f32[256, 256]" = torch.ops.aten.mm.default(permute_399, view_90);  permute_399 = view_90 = None
    permute_400: "f32[256, 256]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_141: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_492, [0], True);  view_492 = None
    view_493: "f32[256]" = torch.ops.aten.view.default(sum_141, [256]);  sum_141 = None
    permute_401: "f32[256, 256]" = torch.ops.aten.permute.default(permute_400, [1, 0]);  permute_400 = None
    view_494: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_96, [1, 512, 256]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_150: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_149, view_494);  add_149 = view_494 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_100: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_33, getitem_43);  add_33 = getitem_43 = None
    mul_323: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_100, rsqrt_8);  sub_100 = None
    mul_324: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_150, primals_70);  primals_70 = None
    mul_325: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_324, 256)
    sum_142: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_324, [2], True)
    mul_326: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_324, mul_323);  mul_324 = None
    sum_143: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_326, [2], True);  mul_326 = None
    mul_327: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_323, sum_143);  sum_143 = None
    sub_101: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_325, sum_142);  mul_325 = sum_142 = None
    sub_102: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_101, mul_327);  sub_101 = mul_327 = None
    div_54: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 256);  rsqrt_8 = None
    mul_328: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_54, sub_102);  div_54 = sub_102 = None
    mul_329: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_150, mul_323);  mul_323 = None
    sum_144: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_329, [0, 1]);  mul_329 = None
    sum_145: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_150, [0, 1]);  add_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_26: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_41, torch.float32);  getitem_41 = None
    mul_330: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_26, 1.1111111111111112);  convert_element_type_26 = None
    mul_331: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_328, mul_330);  mul_330 = None
    clone_54: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_331, memory_format = torch.contiguous_format);  mul_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_495: "f32[512, 256]" = torch.ops.aten.view.default(clone_54, [512, 256]);  clone_54 = None
    permute_402: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
    mm_98: "f32[512, 1024]" = torch.ops.aten.mm.default(view_495, permute_402);  permute_402 = None
    permute_403: "f32[256, 512]" = torch.ops.aten.permute.default(view_495, [1, 0])
    mm_99: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_403, view_88);  permute_403 = view_88 = None
    permute_404: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    sum_146: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_495, [0], True);  view_495 = None
    view_496: "f32[256]" = torch.ops.aten.view.default(sum_146, [256]);  sum_146 = None
    permute_405: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_404, [1, 0]);  permute_404 = None
    view_497: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_98, [1, 512, 1024]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_332: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_87, 0.7071067811865476)
    erf_20: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_332);  mul_332 = None
    add_151: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_333: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_151, 0.5);  add_151 = None
    mul_334: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_87, view_87)
    mul_335: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_334, -0.5);  mul_334 = None
    exp_24: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_335);  mul_335 = None
    mul_336: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
    mul_337: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_87, mul_336);  view_87 = mul_336 = None
    add_152: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_333, mul_337);  mul_333 = mul_337 = None
    mul_338: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_497, add_152);  view_497 = add_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_498: "f32[512, 1024]" = torch.ops.aten.view.default(mul_338, [512, 1024]);  mul_338 = None
    permute_406: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
    mm_100: "f32[512, 256]" = torch.ops.aten.mm.default(view_498, permute_406);  permute_406 = None
    permute_407: "f32[1024, 512]" = torch.ops.aten.permute.default(view_498, [1, 0])
    mm_101: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_407, view_86);  permute_407 = view_86 = None
    permute_408: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_101, [1, 0]);  mm_101 = None
    sum_147: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_498, [0], True);  view_498 = None
    view_499: "f32[1024]" = torch.ops.aten.view.default(sum_147, [1024]);  sum_147 = None
    permute_409: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_408, [1, 0]);  permute_408 = None
    view_500: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_100, [1, 512, 256]);  mm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_153: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_328, view_500);  mul_328 = view_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_103: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_29, getitem_39);  add_29 = getitem_39 = None
    mul_339: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_103, rsqrt_7);  sub_103 = None
    mul_340: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_153, primals_64);  primals_64 = None
    mul_341: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_340, 256)
    sum_148: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_340, [2], True)
    mul_342: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_340, mul_339);  mul_340 = None
    sum_149: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_342, [2], True);  mul_342 = None
    mul_343: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_339, sum_149);  sum_149 = None
    sub_104: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_341, sum_148);  mul_341 = sum_148 = None
    sub_105: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_104, mul_343);  sub_104 = mul_343 = None
    div_55: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 256);  rsqrt_7 = None
    mul_344: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_55, sub_105);  div_55 = sub_105 = None
    mul_345: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_153, mul_339);  mul_339 = None
    sum_150: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_345, [0, 1]);  mul_345 = None
    sum_151: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_153, [0, 1]);  add_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_27: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_37, torch.float32);  getitem_37 = None
    mul_346: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_27, 1.1111111111111112);  convert_element_type_27 = None
    mul_347: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_344, mul_346);  mul_346 = None
    clone_55: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_347, memory_format = torch.contiguous_format);  mul_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_501: "f32[512, 256]" = torch.ops.aten.view.default(clone_55, [512, 256]);  clone_55 = None
    permute_410: "f32[256, 256]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    mm_102: "f32[512, 256]" = torch.ops.aten.mm.default(view_501, permute_410);  permute_410 = None
    permute_411: "f32[256, 512]" = torch.ops.aten.permute.default(view_501, [1, 0])
    mm_103: "f32[256, 256]" = torch.ops.aten.mm.default(permute_411, view_84);  permute_411 = view_84 = None
    permute_412: "f32[256, 256]" = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
    sum_152: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_501, [0], True);  view_501 = None
    view_502: "f32[256]" = torch.ops.aten.view.default(sum_152, [256]);  sum_152 = None
    permute_413: "f32[256, 256]" = torch.ops.aten.permute.default(permute_412, [1, 0]);  permute_412 = None
    view_503: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_102, [1, 512, 256]);  mm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_504: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_503, [1, 512, 4, 64]);  view_503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_414: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_504, [0, 2, 1, 3]);  view_504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_505: "f32[4, 512, 64]" = torch.ops.aten.view.default(permute_414, [4, 512, 64]);  permute_414 = None
    permute_415: "f32[4, 512, 512]" = torch.ops.aten.permute.default(view_80, [0, 2, 1]);  view_80 = None
    bmm_56: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(permute_415, view_505);  permute_415 = None
    permute_416: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_81, [0, 2, 1]);  view_81 = None
    bmm_57: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_505, permute_416);  view_505 = permute_416 = None
    view_506: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_56, [1, 4, 512, 64]);  bmm_56 = None
    view_507: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_57, [1, 4, 512, 512]);  bmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_28: "f32[1, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_35, torch.float32);  getitem_35 = None
    mul_348: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_28, 1.1111111111111112);  convert_element_type_28 = None
    mul_349: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_507, mul_348);  view_507 = mul_348 = None
    clone_56: "f32[1, 4, 512, 512]" = torch.ops.aten.clone.default(mul_349, memory_format = torch.contiguous_format);  mul_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_24: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    mul_350: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(clone_56, alias_24);  clone_56 = None
    sum_153: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_350, [-1], True)
    mul_351: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(alias_24, sum_153);  alias_24 = sum_153 = None
    sub_106: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(mul_350, mul_351);  mul_350 = mul_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_56: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(sub_106, 8.0);  sub_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_508: "f32[4, 512, 512]" = torch.ops.aten.view.default(div_56, [4, 512, 512]);  div_56 = None
    permute_417: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_77, [0, 2, 1]);  view_77 = None
    bmm_58: "f32[4, 64, 512]" = torch.ops.aten.bmm.default(permute_417, view_508);  permute_417 = None
    permute_418: "f32[4, 512, 64]" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
    bmm_59: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_508, permute_418);  view_508 = permute_418 = None
    view_509: "f32[1, 4, 64, 512]" = torch.ops.aten.view.default(bmm_58, [1, 4, 64, 512]);  bmm_58 = None
    view_510: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_59, [1, 4, 512, 64]);  bmm_59 = None
    permute_419: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_509, [0, 1, 3, 2]);  view_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_420: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_510, [0, 2, 1, 3]);  view_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_57: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_420, memory_format = torch.contiguous_format);  permute_420 = None
    view_511: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_57, [1, 512, 256]);  clone_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_421: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_506, [0, 2, 1, 3]);  view_506 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_58: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_421, memory_format = torch.contiguous_format);  permute_421 = None
    view_512: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_58, [1, 512, 256]);  clone_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_513: "f32[512, 256]" = torch.ops.aten.view.default(view_512, [512, 256]);  view_512 = None
    permute_422: "f32[256, 256]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
    mm_104: "f32[512, 256]" = torch.ops.aten.mm.default(view_513, permute_422);  permute_422 = None
    permute_423: "f32[256, 512]" = torch.ops.aten.permute.default(view_513, [1, 0])
    mm_105: "f32[256, 256]" = torch.ops.aten.mm.default(permute_423, view_73);  permute_423 = view_73 = None
    permute_424: "f32[256, 256]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
    sum_154: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_513, [0], True);  view_513 = None
    view_514: "f32[256]" = torch.ops.aten.view.default(sum_154, [256]);  sum_154 = None
    permute_425: "f32[256, 256]" = torch.ops.aten.permute.default(permute_424, [1, 0]);  permute_424 = None
    view_515: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_104, [1, 512, 256]);  mm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_154: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_344, view_515);  mul_344 = view_515 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_426: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(permute_419, [0, 2, 1, 3]);  permute_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_516: "f32[1, 512, 256]" = torch.ops.aten.view.default(permute_426, [1, 512, 256]);  permute_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_517: "f32[512, 256]" = torch.ops.aten.view.default(view_516, [512, 256]);  view_516 = None
    permute_427: "f32[256, 256]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
    mm_106: "f32[512, 256]" = torch.ops.aten.mm.default(view_517, permute_427);  permute_427 = None
    permute_428: "f32[256, 512]" = torch.ops.aten.permute.default(view_517, [1, 0])
    mm_107: "f32[256, 256]" = torch.ops.aten.mm.default(permute_428, view_70);  permute_428 = view_70 = None
    permute_429: "f32[256, 256]" = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
    sum_155: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_517, [0], True);  view_517 = None
    view_518: "f32[256]" = torch.ops.aten.view.default(sum_155, [256]);  sum_155 = None
    permute_430: "f32[256, 256]" = torch.ops.aten.permute.default(permute_429, [1, 0]);  permute_429 = None
    view_519: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_106, [1, 512, 256]);  mm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_155: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_154, view_519);  add_154 = view_519 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_520: "f32[512, 256]" = torch.ops.aten.view.default(view_511, [512, 256]);  view_511 = None
    permute_431: "f32[256, 256]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    mm_108: "f32[512, 256]" = torch.ops.aten.mm.default(view_520, permute_431);  permute_431 = None
    permute_432: "f32[256, 512]" = torch.ops.aten.permute.default(view_520, [1, 0])
    mm_109: "f32[256, 256]" = torch.ops.aten.mm.default(permute_432, view_68);  permute_432 = view_68 = None
    permute_433: "f32[256, 256]" = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
    sum_156: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_520, [0], True);  view_520 = None
    view_521: "f32[256]" = torch.ops.aten.view.default(sum_156, [256]);  sum_156 = None
    permute_434: "f32[256, 256]" = torch.ops.aten.permute.default(permute_433, [1, 0]);  permute_433 = None
    view_522: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_108, [1, 512, 256]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_156: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_155, view_522);  add_155 = view_522 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_107: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_25, getitem_33);  add_25 = getitem_33 = None
    mul_352: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_107, rsqrt_6);  sub_107 = None
    mul_353: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_156, primals_54);  primals_54 = None
    mul_354: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_353, 256)
    sum_157: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_353, [2], True)
    mul_355: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_353, mul_352);  mul_353 = None
    sum_158: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_355, [2], True);  mul_355 = None
    mul_356: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_352, sum_158);  sum_158 = None
    sub_108: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_354, sum_157);  mul_354 = sum_157 = None
    sub_109: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_108, mul_356);  sub_108 = mul_356 = None
    div_57: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 256);  rsqrt_6 = None
    mul_357: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_57, sub_109);  div_57 = sub_109 = None
    mul_358: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_156, mul_352);  mul_352 = None
    sum_159: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_358, [0, 1]);  mul_358 = None
    sum_160: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_156, [0, 1]);  add_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_29: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_31, torch.float32);  getitem_31 = None
    mul_359: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_29, 1.1111111111111112);  convert_element_type_29 = None
    mul_360: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_357, mul_359);  mul_359 = None
    clone_59: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_360, memory_format = torch.contiguous_format);  mul_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_523: "f32[512, 256]" = torch.ops.aten.view.default(clone_59, [512, 256]);  clone_59 = None
    permute_435: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    mm_110: "f32[512, 1024]" = torch.ops.aten.mm.default(view_523, permute_435);  permute_435 = None
    permute_436: "f32[256, 512]" = torch.ops.aten.permute.default(view_523, [1, 0])
    mm_111: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_436, view_66);  permute_436 = view_66 = None
    permute_437: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
    sum_161: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_523, [0], True);  view_523 = None
    view_524: "f32[256]" = torch.ops.aten.view.default(sum_161, [256]);  sum_161 = None
    permute_438: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_437, [1, 0]);  permute_437 = None
    view_525: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_110, [1, 512, 1024]);  mm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_361: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_65, 0.7071067811865476)
    erf_21: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_361);  mul_361 = None
    add_157: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_362: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_157, 0.5);  add_157 = None
    mul_363: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_65, view_65)
    mul_364: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_363, -0.5);  mul_363 = None
    exp_25: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_364);  mul_364 = None
    mul_365: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
    mul_366: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_65, mul_365);  view_65 = mul_365 = None
    add_158: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_362, mul_366);  mul_362 = mul_366 = None
    mul_367: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_525, add_158);  view_525 = add_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_526: "f32[512, 1024]" = torch.ops.aten.view.default(mul_367, [512, 1024]);  mul_367 = None
    permute_439: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    mm_112: "f32[512, 256]" = torch.ops.aten.mm.default(view_526, permute_439);  permute_439 = None
    permute_440: "f32[1024, 512]" = torch.ops.aten.permute.default(view_526, [1, 0])
    mm_113: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_440, view_64);  permute_440 = view_64 = None
    permute_441: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_113, [1, 0]);  mm_113 = None
    sum_162: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_526, [0], True);  view_526 = None
    view_527: "f32[1024]" = torch.ops.aten.view.default(sum_162, [1024]);  sum_162 = None
    permute_442: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_441, [1, 0]);  permute_441 = None
    view_528: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_112, [1, 512, 256]);  mm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_159: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_357, view_528);  mul_357 = view_528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_110: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_21, getitem_29);  add_21 = getitem_29 = None
    mul_368: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_110, rsqrt_5);  sub_110 = None
    mul_369: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_159, primals_48);  primals_48 = None
    mul_370: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_369, 256)
    sum_163: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_369, [2], True)
    mul_371: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_369, mul_368);  mul_369 = None
    sum_164: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_371, [2], True);  mul_371 = None
    mul_372: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_368, sum_164);  sum_164 = None
    sub_111: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_370, sum_163);  mul_370 = sum_163 = None
    sub_112: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_111, mul_372);  sub_111 = mul_372 = None
    div_58: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 256);  rsqrt_5 = None
    mul_373: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_58, sub_112);  div_58 = sub_112 = None
    mul_374: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_159, mul_368);  mul_368 = None
    sum_165: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_374, [0, 1]);  mul_374 = None
    sum_166: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_159, [0, 1]);  add_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_30: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_27, torch.float32);  getitem_27 = None
    mul_375: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_30, 1.1111111111111112);  convert_element_type_30 = None
    mul_376: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_373, mul_375);  mul_375 = None
    clone_60: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_376, memory_format = torch.contiguous_format);  mul_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_529: "f32[512, 256]" = torch.ops.aten.view.default(clone_60, [512, 256]);  clone_60 = None
    permute_443: "f32[256, 256]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
    mm_114: "f32[512, 256]" = torch.ops.aten.mm.default(view_529, permute_443);  permute_443 = None
    permute_444: "f32[256, 512]" = torch.ops.aten.permute.default(view_529, [1, 0])
    mm_115: "f32[256, 256]" = torch.ops.aten.mm.default(permute_444, view_62);  permute_444 = view_62 = None
    permute_445: "f32[256, 256]" = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
    sum_167: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_529, [0], True);  view_529 = None
    view_530: "f32[256]" = torch.ops.aten.view.default(sum_167, [256]);  sum_167 = None
    permute_446: "f32[256, 256]" = torch.ops.aten.permute.default(permute_445, [1, 0]);  permute_445 = None
    view_531: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_114, [1, 512, 256]);  mm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_532: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_531, [1, 512, 4, 64]);  view_531 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_447: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_532, [0, 2, 1, 3]);  view_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_533: "f32[4, 512, 64]" = torch.ops.aten.view.default(permute_447, [4, 512, 64]);  permute_447 = None
    permute_448: "f32[4, 512, 512]" = torch.ops.aten.permute.default(view_58, [0, 2, 1]);  view_58 = None
    bmm_60: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(permute_448, view_533);  permute_448 = None
    permute_449: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_59, [0, 2, 1]);  view_59 = None
    bmm_61: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_533, permute_449);  view_533 = permute_449 = None
    view_534: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_60, [1, 4, 512, 64]);  bmm_60 = None
    view_535: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_61, [1, 4, 512, 512]);  bmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_31: "f32[1, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_25, torch.float32);  getitem_25 = None
    mul_377: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_31, 1.1111111111111112);  convert_element_type_31 = None
    mul_378: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_535, mul_377);  view_535 = mul_377 = None
    clone_61: "f32[1, 4, 512, 512]" = torch.ops.aten.clone.default(mul_378, memory_format = torch.contiguous_format);  mul_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_25: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    mul_379: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(clone_61, alias_25);  clone_61 = None
    sum_168: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_379, [-1], True)
    mul_380: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(alias_25, sum_168);  alias_25 = sum_168 = None
    sub_113: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(mul_379, mul_380);  mul_379 = mul_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_59: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(sub_113, 8.0);  sub_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_536: "f32[4, 512, 512]" = torch.ops.aten.view.default(div_59, [4, 512, 512]);  div_59 = None
    permute_450: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_55, [0, 2, 1]);  view_55 = None
    bmm_62: "f32[4, 64, 512]" = torch.ops.aten.bmm.default(permute_450, view_536);  permute_450 = None
    permute_451: "f32[4, 512, 64]" = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None
    bmm_63: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_536, permute_451);  view_536 = permute_451 = None
    view_537: "f32[1, 4, 64, 512]" = torch.ops.aten.view.default(bmm_62, [1, 4, 64, 512]);  bmm_62 = None
    view_538: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_63, [1, 4, 512, 64]);  bmm_63 = None
    permute_452: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_537, [0, 1, 3, 2]);  view_537 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_453: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_538, [0, 2, 1, 3]);  view_538 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_62: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_453, memory_format = torch.contiguous_format);  permute_453 = None
    view_539: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_62, [1, 512, 256]);  clone_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_454: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_534, [0, 2, 1, 3]);  view_534 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_63: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_454, memory_format = torch.contiguous_format);  permute_454 = None
    view_540: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_63, [1, 512, 256]);  clone_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_541: "f32[512, 256]" = torch.ops.aten.view.default(view_540, [512, 256]);  view_540 = None
    permute_455: "f32[256, 256]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
    mm_116: "f32[512, 256]" = torch.ops.aten.mm.default(view_541, permute_455);  permute_455 = None
    permute_456: "f32[256, 512]" = torch.ops.aten.permute.default(view_541, [1, 0])
    mm_117: "f32[256, 256]" = torch.ops.aten.mm.default(permute_456, view_51);  permute_456 = view_51 = None
    permute_457: "f32[256, 256]" = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
    sum_169: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_541, [0], True);  view_541 = None
    view_542: "f32[256]" = torch.ops.aten.view.default(sum_169, [256]);  sum_169 = None
    permute_458: "f32[256, 256]" = torch.ops.aten.permute.default(permute_457, [1, 0]);  permute_457 = None
    view_543: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_116, [1, 512, 256]);  mm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_160: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_373, view_543);  mul_373 = view_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_459: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(permute_452, [0, 2, 1, 3]);  permute_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_544: "f32[1, 512, 256]" = torch.ops.aten.view.default(permute_459, [1, 512, 256]);  permute_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_545: "f32[512, 256]" = torch.ops.aten.view.default(view_544, [512, 256]);  view_544 = None
    permute_460: "f32[256, 256]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
    mm_118: "f32[512, 256]" = torch.ops.aten.mm.default(view_545, permute_460);  permute_460 = None
    permute_461: "f32[256, 512]" = torch.ops.aten.permute.default(view_545, [1, 0])
    mm_119: "f32[256, 256]" = torch.ops.aten.mm.default(permute_461, view_48);  permute_461 = view_48 = None
    permute_462: "f32[256, 256]" = torch.ops.aten.permute.default(mm_119, [1, 0]);  mm_119 = None
    sum_170: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_545, [0], True);  view_545 = None
    view_546: "f32[256]" = torch.ops.aten.view.default(sum_170, [256]);  sum_170 = None
    permute_463: "f32[256, 256]" = torch.ops.aten.permute.default(permute_462, [1, 0]);  permute_462 = None
    view_547: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_118, [1, 512, 256]);  mm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_161: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_160, view_547);  add_160 = view_547 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_548: "f32[512, 256]" = torch.ops.aten.view.default(view_539, [512, 256]);  view_539 = None
    permute_464: "f32[256, 256]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    mm_120: "f32[512, 256]" = torch.ops.aten.mm.default(view_548, permute_464);  permute_464 = None
    permute_465: "f32[256, 512]" = torch.ops.aten.permute.default(view_548, [1, 0])
    mm_121: "f32[256, 256]" = torch.ops.aten.mm.default(permute_465, view_46);  permute_465 = view_46 = None
    permute_466: "f32[256, 256]" = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
    sum_171: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_548, [0], True);  view_548 = None
    view_549: "f32[256]" = torch.ops.aten.view.default(sum_171, [256]);  sum_171 = None
    permute_467: "f32[256, 256]" = torch.ops.aten.permute.default(permute_466, [1, 0]);  permute_466 = None
    view_550: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_120, [1, 512, 256]);  mm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_162: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_161, view_550);  add_161 = view_550 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_114: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_17, getitem_23);  add_17 = getitem_23 = None
    mul_381: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_114, rsqrt_4);  sub_114 = None
    mul_382: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_162, primals_38);  primals_38 = None
    mul_383: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_382, 256)
    sum_172: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_382, [2], True)
    mul_384: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_382, mul_381);  mul_382 = None
    sum_173: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_384, [2], True);  mul_384 = None
    mul_385: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_381, sum_173);  sum_173 = None
    sub_115: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_383, sum_172);  mul_383 = sum_172 = None
    sub_116: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_115, mul_385);  sub_115 = mul_385 = None
    div_60: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 256);  rsqrt_4 = None
    mul_386: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_60, sub_116);  div_60 = sub_116 = None
    mul_387: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_162, mul_381);  mul_381 = None
    sum_174: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_387, [0, 1]);  mul_387 = None
    sum_175: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_162, [0, 1]);  add_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_32: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_21, torch.float32);  getitem_21 = None
    mul_388: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_32, 1.1111111111111112);  convert_element_type_32 = None
    mul_389: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_386, mul_388);  mul_388 = None
    clone_64: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_389, memory_format = torch.contiguous_format);  mul_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_551: "f32[512, 256]" = torch.ops.aten.view.default(clone_64, [512, 256]);  clone_64 = None
    permute_468: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    mm_122: "f32[512, 1024]" = torch.ops.aten.mm.default(view_551, permute_468);  permute_468 = None
    permute_469: "f32[256, 512]" = torch.ops.aten.permute.default(view_551, [1, 0])
    mm_123: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_469, view_44);  permute_469 = view_44 = None
    permute_470: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_123, [1, 0]);  mm_123 = None
    sum_176: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_551, [0], True);  view_551 = None
    view_552: "f32[256]" = torch.ops.aten.view.default(sum_176, [256]);  sum_176 = None
    permute_471: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_470, [1, 0]);  permute_470 = None
    view_553: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_122, [1, 512, 1024]);  mm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_390: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_43, 0.7071067811865476)
    erf_22: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_390);  mul_390 = None
    add_163: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_391: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_163, 0.5);  add_163 = None
    mul_392: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_43, view_43)
    mul_393: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_392, -0.5);  mul_392 = None
    exp_26: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_393);  mul_393 = None
    mul_394: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
    mul_395: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_43, mul_394);  view_43 = mul_394 = None
    add_164: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_391, mul_395);  mul_391 = mul_395 = None
    mul_396: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_553, add_164);  view_553 = add_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_554: "f32[512, 1024]" = torch.ops.aten.view.default(mul_396, [512, 1024]);  mul_396 = None
    permute_472: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    mm_124: "f32[512, 256]" = torch.ops.aten.mm.default(view_554, permute_472);  permute_472 = None
    permute_473: "f32[1024, 512]" = torch.ops.aten.permute.default(view_554, [1, 0])
    mm_125: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_473, view_42);  permute_473 = view_42 = None
    permute_474: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_125, [1, 0]);  mm_125 = None
    sum_177: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_554, [0], True);  view_554 = None
    view_555: "f32[1024]" = torch.ops.aten.view.default(sum_177, [1024]);  sum_177 = None
    permute_475: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_474, [1, 0]);  permute_474 = None
    view_556: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_124, [1, 512, 256]);  mm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_165: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_386, view_556);  mul_386 = view_556 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_117: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_13, getitem_19);  add_13 = getitem_19 = None
    mul_397: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_117, rsqrt_3);  sub_117 = None
    mul_398: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_165, primals_32);  primals_32 = None
    mul_399: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_398, 256)
    sum_178: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_398, [2], True)
    mul_400: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_398, mul_397);  mul_398 = None
    sum_179: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_400, [2], True);  mul_400 = None
    mul_401: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_397, sum_179);  sum_179 = None
    sub_118: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_399, sum_178);  mul_399 = sum_178 = None
    sub_119: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_118, mul_401);  sub_118 = mul_401 = None
    div_61: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 256);  rsqrt_3 = None
    mul_402: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_61, sub_119);  div_61 = sub_119 = None
    mul_403: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_165, mul_397);  mul_397 = None
    sum_180: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_403, [0, 1]);  mul_403 = None
    sum_181: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_165, [0, 1]);  add_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_33: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_17, torch.float32);  getitem_17 = None
    mul_404: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_33, 1.1111111111111112);  convert_element_type_33 = None
    mul_405: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_402, mul_404);  mul_404 = None
    clone_65: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_405, memory_format = torch.contiguous_format);  mul_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_557: "f32[512, 256]" = torch.ops.aten.view.default(clone_65, [512, 256]);  clone_65 = None
    permute_476: "f32[256, 256]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    mm_126: "f32[512, 256]" = torch.ops.aten.mm.default(view_557, permute_476);  permute_476 = None
    permute_477: "f32[256, 512]" = torch.ops.aten.permute.default(view_557, [1, 0])
    mm_127: "f32[256, 256]" = torch.ops.aten.mm.default(permute_477, view_40);  permute_477 = view_40 = None
    permute_478: "f32[256, 256]" = torch.ops.aten.permute.default(mm_127, [1, 0]);  mm_127 = None
    sum_182: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_557, [0], True);  view_557 = None
    view_558: "f32[256]" = torch.ops.aten.view.default(sum_182, [256]);  sum_182 = None
    permute_479: "f32[256, 256]" = torch.ops.aten.permute.default(permute_478, [1, 0]);  permute_478 = None
    view_559: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_126, [1, 512, 256]);  mm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_560: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_559, [1, 512, 4, 64]);  view_559 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_480: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_560, [0, 2, 1, 3]);  view_560 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_561: "f32[4, 512, 64]" = torch.ops.aten.view.default(permute_480, [4, 512, 64]);  permute_480 = None
    permute_481: "f32[4, 512, 512]" = torch.ops.aten.permute.default(view_36, [0, 2, 1]);  view_36 = None
    bmm_64: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(permute_481, view_561);  permute_481 = None
    permute_482: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_37, [0, 2, 1]);  view_37 = None
    bmm_65: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_561, permute_482);  view_561 = permute_482 = None
    view_562: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_64, [1, 4, 512, 64]);  bmm_64 = None
    view_563: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_65, [1, 4, 512, 512]);  bmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_34: "f32[1, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_15, torch.float32);  getitem_15 = None
    mul_406: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_34, 1.1111111111111112);  convert_element_type_34 = None
    mul_407: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_563, mul_406);  view_563 = mul_406 = None
    clone_66: "f32[1, 4, 512, 512]" = torch.ops.aten.clone.default(mul_407, memory_format = torch.contiguous_format);  mul_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_26: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    mul_408: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(clone_66, alias_26);  clone_66 = None
    sum_183: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_408, [-1], True)
    mul_409: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(alias_26, sum_183);  alias_26 = sum_183 = None
    sub_120: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(mul_408, mul_409);  mul_408 = mul_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_62: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(sub_120, 8.0);  sub_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_564: "f32[4, 512, 512]" = torch.ops.aten.view.default(div_62, [4, 512, 512]);  div_62 = None
    permute_483: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_33, [0, 2, 1]);  view_33 = None
    bmm_66: "f32[4, 64, 512]" = torch.ops.aten.bmm.default(permute_483, view_564);  permute_483 = None
    permute_484: "f32[4, 512, 64]" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
    bmm_67: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_564, permute_484);  view_564 = permute_484 = None
    view_565: "f32[1, 4, 64, 512]" = torch.ops.aten.view.default(bmm_66, [1, 4, 64, 512]);  bmm_66 = None
    view_566: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_67, [1, 4, 512, 64]);  bmm_67 = None
    permute_485: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_565, [0, 1, 3, 2]);  view_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_486: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_566, [0, 2, 1, 3]);  view_566 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_67: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_486, memory_format = torch.contiguous_format);  permute_486 = None
    view_567: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_67, [1, 512, 256]);  clone_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_487: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_562, [0, 2, 1, 3]);  view_562 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_68: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_487, memory_format = torch.contiguous_format);  permute_487 = None
    view_568: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_68, [1, 512, 256]);  clone_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_569: "f32[512, 256]" = torch.ops.aten.view.default(view_568, [512, 256]);  view_568 = None
    permute_488: "f32[256, 256]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
    mm_128: "f32[512, 256]" = torch.ops.aten.mm.default(view_569, permute_488);  permute_488 = None
    permute_489: "f32[256, 512]" = torch.ops.aten.permute.default(view_569, [1, 0])
    mm_129: "f32[256, 256]" = torch.ops.aten.mm.default(permute_489, view_29);  permute_489 = view_29 = None
    permute_490: "f32[256, 256]" = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
    sum_184: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_569, [0], True);  view_569 = None
    view_570: "f32[256]" = torch.ops.aten.view.default(sum_184, [256]);  sum_184 = None
    permute_491: "f32[256, 256]" = torch.ops.aten.permute.default(permute_490, [1, 0]);  permute_490 = None
    view_571: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_128, [1, 512, 256]);  mm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_166: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_402, view_571);  mul_402 = view_571 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_492: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(permute_485, [0, 2, 1, 3]);  permute_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_572: "f32[1, 512, 256]" = torch.ops.aten.view.default(permute_492, [1, 512, 256]);  permute_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_573: "f32[512, 256]" = torch.ops.aten.view.default(view_572, [512, 256]);  view_572 = None
    permute_493: "f32[256, 256]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    mm_130: "f32[512, 256]" = torch.ops.aten.mm.default(view_573, permute_493);  permute_493 = None
    permute_494: "f32[256, 512]" = torch.ops.aten.permute.default(view_573, [1, 0])
    mm_131: "f32[256, 256]" = torch.ops.aten.mm.default(permute_494, view_26);  permute_494 = view_26 = None
    permute_495: "f32[256, 256]" = torch.ops.aten.permute.default(mm_131, [1, 0]);  mm_131 = None
    sum_185: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_573, [0], True);  view_573 = None
    view_574: "f32[256]" = torch.ops.aten.view.default(sum_185, [256]);  sum_185 = None
    permute_496: "f32[256, 256]" = torch.ops.aten.permute.default(permute_495, [1, 0]);  permute_495 = None
    view_575: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_130, [1, 512, 256]);  mm_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_167: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_166, view_575);  add_166 = view_575 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_576: "f32[512, 256]" = torch.ops.aten.view.default(view_567, [512, 256]);  view_567 = None
    permute_497: "f32[256, 256]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    mm_132: "f32[512, 256]" = torch.ops.aten.mm.default(view_576, permute_497);  permute_497 = None
    permute_498: "f32[256, 512]" = torch.ops.aten.permute.default(view_576, [1, 0])
    mm_133: "f32[256, 256]" = torch.ops.aten.mm.default(permute_498, view_24);  permute_498 = view_24 = None
    permute_499: "f32[256, 256]" = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
    sum_186: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_576, [0], True);  view_576 = None
    view_577: "f32[256]" = torch.ops.aten.view.default(sum_186, [256]);  sum_186 = None
    permute_500: "f32[256, 256]" = torch.ops.aten.permute.default(permute_499, [1, 0]);  permute_499 = None
    view_578: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_132, [1, 512, 256]);  mm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_168: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_167, view_578);  add_167 = view_578 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_121: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_9, getitem_13);  add_9 = getitem_13 = None
    mul_410: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_121, rsqrt_2);  sub_121 = None
    mul_411: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_168, primals_22);  primals_22 = None
    mul_412: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_411, 256)
    sum_187: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_411, [2], True)
    mul_413: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_411, mul_410);  mul_411 = None
    sum_188: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_413, [2], True);  mul_413 = None
    mul_414: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_410, sum_188);  sum_188 = None
    sub_122: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_412, sum_187);  mul_412 = sum_187 = None
    sub_123: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_122, mul_414);  sub_122 = mul_414 = None
    div_63: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 256);  rsqrt_2 = None
    mul_415: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_63, sub_123);  div_63 = sub_123 = None
    mul_416: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_168, mul_410);  mul_410 = None
    sum_189: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_416, [0, 1]);  mul_416 = None
    sum_190: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_168, [0, 1]);  add_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_35: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_11, torch.float32);  getitem_11 = None
    mul_417: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_35, 1.1111111111111112);  convert_element_type_35 = None
    mul_418: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_415, mul_417);  mul_417 = None
    clone_69: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_418, memory_format = torch.contiguous_format);  mul_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_579: "f32[512, 256]" = torch.ops.aten.view.default(clone_69, [512, 256]);  clone_69 = None
    permute_501: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    mm_134: "f32[512, 1024]" = torch.ops.aten.mm.default(view_579, permute_501);  permute_501 = None
    permute_502: "f32[256, 512]" = torch.ops.aten.permute.default(view_579, [1, 0])
    mm_135: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_502, view_22);  permute_502 = view_22 = None
    permute_503: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
    sum_191: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_579, [0], True);  view_579 = None
    view_580: "f32[256]" = torch.ops.aten.view.default(sum_191, [256]);  sum_191 = None
    permute_504: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_503, [1, 0]);  permute_503 = None
    view_581: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_134, [1, 512, 1024]);  mm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_419: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_21, 0.7071067811865476)
    erf_23: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_419);  mul_419 = None
    add_169: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_420: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_169, 0.5);  add_169 = None
    mul_421: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_21, view_21)
    mul_422: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_421, -0.5);  mul_421 = None
    exp_27: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_422);  mul_422 = None
    mul_423: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_27, 0.3989422804014327);  exp_27 = None
    mul_424: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_21, mul_423);  view_21 = mul_423 = None
    add_170: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_420, mul_424);  mul_420 = mul_424 = None
    mul_425: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_581, add_170);  view_581 = add_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_582: "f32[512, 1024]" = torch.ops.aten.view.default(mul_425, [512, 1024]);  mul_425 = None
    permute_505: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    mm_136: "f32[512, 256]" = torch.ops.aten.mm.default(view_582, permute_505);  permute_505 = None
    permute_506: "f32[1024, 512]" = torch.ops.aten.permute.default(view_582, [1, 0])
    mm_137: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_506, view_20);  permute_506 = view_20 = None
    permute_507: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_137, [1, 0]);  mm_137 = None
    sum_192: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_582, [0], True);  view_582 = None
    view_583: "f32[1024]" = torch.ops.aten.view.default(sum_192, [1024]);  sum_192 = None
    permute_508: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_507, [1, 0]);  permute_507 = None
    view_584: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_136, [1, 512, 256]);  mm_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_171: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_415, view_584);  mul_415 = view_584 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    sub_124: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_5, getitem_9);  add_5 = getitem_9 = None
    mul_426: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_124, rsqrt_1);  sub_124 = None
    mul_427: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_171, primals_16);  primals_16 = None
    mul_428: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_427, 256)
    sum_193: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_427, [2], True)
    mul_429: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_427, mul_426);  mul_427 = None
    sum_194: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_429, [2], True);  mul_429 = None
    mul_430: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_426, sum_194);  sum_194 = None
    sub_125: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_428, sum_193);  mul_428 = sum_193 = None
    sub_126: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_125, mul_430);  sub_125 = mul_430 = None
    div_64: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 256);  rsqrt_1 = None
    mul_431: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_64, sub_126);  div_64 = sub_126 = None
    mul_432: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_171, mul_426);  mul_426 = None
    sum_195: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_432, [0, 1]);  mul_432 = None
    sum_196: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_171, [0, 1]);  add_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_36: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_7, torch.float32);  getitem_7 = None
    mul_433: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_36, 1.1111111111111112);  convert_element_type_36 = None
    mul_434: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_431, mul_433);  mul_433 = None
    clone_70: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_434, memory_format = torch.contiguous_format);  mul_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_585: "f32[512, 256]" = torch.ops.aten.view.default(clone_70, [512, 256]);  clone_70 = None
    permute_509: "f32[256, 256]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    mm_138: "f32[512, 256]" = torch.ops.aten.mm.default(view_585, permute_509);  permute_509 = None
    permute_510: "f32[256, 512]" = torch.ops.aten.permute.default(view_585, [1, 0])
    mm_139: "f32[256, 256]" = torch.ops.aten.mm.default(permute_510, view_18);  permute_510 = view_18 = None
    permute_511: "f32[256, 256]" = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
    sum_197: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_585, [0], True);  view_585 = None
    view_586: "f32[256]" = torch.ops.aten.view.default(sum_197, [256]);  sum_197 = None
    permute_512: "f32[256, 256]" = torch.ops.aten.permute.default(permute_511, [1, 0]);  permute_511 = None
    view_587: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_138, [1, 512, 256]);  mm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_588: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_587, [1, 512, 4, 64]);  view_587 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_513: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_588, [0, 2, 1, 3]);  view_588 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:337, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_589: "f32[4, 512, 64]" = torch.ops.aten.view.default(permute_513, [4, 512, 64]);  permute_513 = None
    permute_514: "f32[4, 512, 512]" = torch.ops.aten.permute.default(view_14, [0, 2, 1]);  view_14 = None
    bmm_68: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(permute_514, view_589);  permute_514 = None
    permute_515: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_15, [0, 2, 1]);  view_15 = None
    bmm_69: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_589, permute_515);  view_589 = permute_515 = None
    view_590: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_68, [1, 4, 512, 64]);  bmm_68 = None
    view_591: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_69, [1, 4, 512, 512]);  bmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:331, code: attention_probs = self.dropout(attention_probs)
    convert_element_type_37: "f32[1, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_5, torch.float32);  getitem_5 = None
    mul_435: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_37, 1.1111111111111112);  convert_element_type_37 = None
    mul_436: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_591, mul_435);  view_591 = mul_435 = None
    clone_71: "f32[1, 4, 512, 512]" = torch.ops.aten.clone.default(mul_436, memory_format = torch.contiguous_format);  mul_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:327, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_27: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(alias);  alias = None
    mul_437: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(clone_71, alias_27);  clone_71 = None
    sum_198: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_437, [-1], True)
    mul_438: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(alias_27, sum_198);  alias_27 = sum_198 = None
    sub_127: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(mul_437, mul_438);  mul_437 = mul_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:321, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_65: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(sub_127, 8.0);  sub_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:297, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_592: "f32[4, 512, 512]" = torch.ops.aten.view.default(div_65, [4, 512, 512]);  div_65 = None
    permute_516: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
    bmm_70: "f32[4, 64, 512]" = torch.ops.aten.bmm.default(permute_516, view_592);  permute_516 = None
    permute_517: "f32[4, 512, 64]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
    bmm_71: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_592, permute_517);  view_592 = permute_517 = None
    view_593: "f32[1, 4, 64, 512]" = torch.ops.aten.view.default(bmm_70, [1, 4, 64, 512]);  bmm_70 = None
    view_594: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_71, [1, 4, 512, 64]);  bmm_71 = None
    permute_518: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_593, [0, 1, 3, 2]);  view_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_519: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_594, [0, 2, 1, 3]);  view_594 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_72: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_519, memory_format = torch.contiguous_format);  permute_519 = None
    view_595: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_72, [1, 512, 256]);  clone_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_520: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_590, [0, 2, 1, 3]);  view_590 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_73: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_520, memory_format = torch.contiguous_format);  permute_520 = None
    view_596: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_73, [1, 512, 256]);  clone_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_597: "f32[512, 256]" = torch.ops.aten.view.default(view_596, [512, 256]);  view_596 = None
    permute_521: "f32[256, 256]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
    mm_140: "f32[512, 256]" = torch.ops.aten.mm.default(view_597, permute_521);  permute_521 = None
    permute_522: "f32[256, 512]" = torch.ops.aten.permute.default(view_597, [1, 0])
    mm_141: "f32[256, 256]" = torch.ops.aten.mm.default(permute_522, view_7);  permute_522 = view_7 = None
    permute_523: "f32[256, 256]" = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
    sum_199: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_597, [0], True);  view_597 = None
    view_598: "f32[256]" = torch.ops.aten.view.default(sum_199, [256]);  sum_199 = None
    permute_524: "f32[256, 256]" = torch.ops.aten.permute.default(permute_523, [1, 0]);  permute_523 = None
    view_599: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_140, [1, 512, 256]);  mm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_172: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_431, view_599);  mul_431 = view_599 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_525: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(permute_518, [0, 2, 1, 3]);  permute_518 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_600: "f32[1, 512, 256]" = torch.ops.aten.view.default(permute_525, [1, 512, 256]);  permute_525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_601: "f32[512, 256]" = torch.ops.aten.view.default(view_600, [512, 256]);  view_600 = None
    permute_526: "f32[256, 256]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    mm_142: "f32[512, 256]" = torch.ops.aten.mm.default(view_601, permute_526);  permute_526 = None
    permute_527: "f32[256, 512]" = torch.ops.aten.permute.default(view_601, [1, 0])
    mm_143: "f32[256, 256]" = torch.ops.aten.mm.default(permute_527, view_4);  permute_527 = view_4 = None
    permute_528: "f32[256, 256]" = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
    sum_200: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_601, [0], True);  view_601 = None
    view_602: "f32[256]" = torch.ops.aten.view.default(sum_200, [256]);  sum_200 = None
    permute_529: "f32[256, 256]" = torch.ops.aten.permute.default(permute_528, [1, 0]);  permute_528 = None
    view_603: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_142, [1, 512, 256]);  mm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_173: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_172, view_603);  add_172 = view_603 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_604: "f32[512, 256]" = torch.ops.aten.view.default(view_595, [512, 256]);  view_595 = None
    permute_530: "f32[256, 256]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    mm_144: "f32[512, 256]" = torch.ops.aten.mm.default(view_604, permute_530);  permute_530 = None
    permute_531: "f32[256, 512]" = torch.ops.aten.permute.default(view_604, [1, 0])
    mm_145: "f32[256, 256]" = torch.ops.aten.mm.default(permute_531, view_2);  permute_531 = view_2 = None
    permute_532: "f32[256, 256]" = torch.ops.aten.permute.default(mm_145, [1, 0]);  mm_145 = None
    sum_201: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_604, [0], True);  view_604 = None
    view_605: "f32[256]" = torch.ops.aten.view.default(sum_201, [256]);  sum_201 = None
    permute_533: "f32[256, 256]" = torch.ops.aten.permute.default(permute_532, [1, 0]);  permute_532 = None
    view_606: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_144, [1, 512, 256]);  mm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_174: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_173, view_606);  add_173 = view_606 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:918, code: hidden_states = self.embeddings_project(hidden_states)
    view_607: "f32[512, 256]" = torch.ops.aten.view.default(add_174, [512, 256]);  add_174 = None
    permute_534: "f32[256, 128]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    mm_146: "f32[512, 128]" = torch.ops.aten.mm.default(view_607, permute_534);  permute_534 = None
    permute_535: "f32[256, 512]" = torch.ops.aten.permute.default(view_607, [1, 0])
    mm_147: "f32[256, 128]" = torch.ops.aten.mm.default(permute_535, view);  permute_535 = view = None
    permute_536: "f32[128, 256]" = torch.ops.aten.permute.default(mm_147, [1, 0]);  mm_147 = None
    sum_202: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_607, [0], True);  view_607 = None
    view_608: "f32[256]" = torch.ops.aten.view.default(sum_202, [256]);  sum_202 = None
    permute_537: "f32[256, 128]" = torch.ops.aten.permute.default(permute_536, [1, 0]);  permute_536 = None
    view_609: "f32[1, 512, 128]" = torch.ops.aten.view.default(mm_146, [1, 512, 128]);  mm_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:211, code: embeddings = self.dropout(embeddings)
    convert_element_type_38: "f32[1, 512, 128]" = torch.ops.prims.convert_element_type.default(getitem_3, torch.float32);  getitem_3 = None
    mul_439: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_38, 1.1111111111111112);  convert_element_type_38 = None
    mul_440: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(view_609, mul_439);  view_609 = mul_439 = None
    clone_74: "f32[1, 512, 128]" = torch.ops.aten.clone.default(mul_440, memory_format = torch.contiguous_format);  mul_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:210, code: embeddings = self.LayerNorm(embeddings)
    sub_128: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
    mul_441: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(sub_128, rsqrt);  sub_128 = None
    mul_442: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(clone_74, primals_4);  primals_4 = None
    mul_443: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_442, 128)
    sum_203: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_442, [2], True)
    mul_444: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_442, mul_441);  mul_442 = None
    sum_204: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_444, [2], True);  mul_444 = None
    mul_445: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_441, sum_204);  sum_204 = None
    sub_129: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(mul_443, sum_203);  mul_443 = sum_203 = None
    sub_130: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(sub_129, mul_445);  sub_129 = mul_445 = None
    div_66: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 128);  rsqrt = None
    mul_446: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(div_66, sub_130);  div_66 = sub_130 = None
    mul_447: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(clone_74, mul_441);  mul_441 = None
    sum_205: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_447, [0, 1]);  mul_447 = None
    sum_206: "f32[128]" = torch.ops.aten.sum.dim_IntList(clone_74, [0, 1]);  clone_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:208, code: position_embeddings = self.position_embeddings(position_ids)
    eq: "b8[1, 512]" = torch.ops.aten.eq.Scalar(slice_4, -1)
    unsqueeze_8: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
    scalar_tensor_8: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_8: "f32[1, 512, 128]" = torch.ops.aten.where.self(unsqueeze_8, scalar_tensor_8, mul_446);  unsqueeze_8 = scalar_tensor_8 = None
    full_3: "f32[512, 128]" = torch.ops.aten.full.default([512, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put: "f32[512, 128]" = torch.ops.aten._unsafe_index_put.default(full_3, [slice_4], where_8, True);  full_3 = slice_4 = where_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:204, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    eq_1: "b8[1, 512]" = torch.ops.aten.eq.Scalar(expand, -1)
    unsqueeze_9: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
    scalar_tensor_9: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_9: "f32[1, 512, 128]" = torch.ops.aten.where.self(unsqueeze_9, scalar_tensor_9, mul_446);  unsqueeze_9 = scalar_tensor_9 = None
    full_4: "f32[2, 128]" = torch.ops.aten.full.default([2, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put_1: "f32[2, 128]" = torch.ops.aten._unsafe_index_put.default(full_4, [expand], where_9, True);  full_4 = expand = where_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:203, code: inputs_embeds = self.word_embeddings(input_ids)
    eq_2: "b8[1, 512]" = torch.ops.aten.eq.Scalar(primals_204, 0)
    unsqueeze_10: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_2, -1);  eq_2 = None
    scalar_tensor_10: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_10: "f32[1, 512, 128]" = torch.ops.aten.where.self(unsqueeze_10, scalar_tensor_10, mul_446);  unsqueeze_10 = scalar_tensor_10 = mul_446 = None
    full_5: "f32[30522, 128]" = torch.ops.aten.full.default([30522, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put_2: "f32[30522, 128]" = torch.ops.aten._unsafe_index_put.default(full_5, [primals_204], where_10, True);  full_5 = primals_204 = where_10 = None
    return pytree.tree_unflatten([div_26, clone_12, clone_13, _unsafe_index_put_2, _unsafe_index_put_1, _unsafe_index_put, sum_205, sum_206, permute_537, view_608, permute_533, view_605, permute_529, view_602, permute_524, view_598, permute_512, view_586, sum_195, sum_196, permute_508, view_583, permute_504, view_580, sum_189, sum_190, permute_500, view_577, permute_496, view_574, permute_491, view_570, permute_479, view_558, sum_180, sum_181, permute_475, view_555, permute_471, view_552, sum_174, sum_175, permute_467, view_549, permute_463, view_546, permute_458, view_542, permute_446, view_530, sum_165, sum_166, permute_442, view_527, permute_438, view_524, sum_159, sum_160, permute_434, view_521, permute_430, view_518, permute_425, view_514, permute_413, view_502, sum_150, sum_151, permute_409, view_499, permute_405, view_496, sum_144, sum_145, permute_401, view_493, permute_397, view_490, permute_392, view_486, permute_380, view_474, sum_135, sum_136, permute_376, view_471, permute_372, view_468, sum_129, sum_130, permute_368, view_465, permute_364, view_462, permute_359, view_458, permute_347, view_446, sum_120, sum_121, permute_343, view_443, permute_339, view_440, sum_114, sum_115, permute_335, view_437, permute_331, view_434, permute_326, view_430, permute_314, view_418, sum_105, sum_106, permute_310, view_415, permute_306, view_412, sum_99, sum_100, permute_302, view_409, permute_298, view_406, permute_293, view_402, permute_281, view_390, sum_90, sum_91, permute_277, view_387, permute_273, view_384, sum_84, sum_85, permute_269, view_381, permute_265, view_378, permute_260, view_374, permute_248, view_362, sum_75, sum_76, permute_244, view_359, permute_240, view_356, sum_69, sum_70, permute_236, view_353, permute_232, view_350, permute_227, view_346, permute_215, view_334, sum_60, sum_61, permute_211, view_331, permute_207, view_328, sum_54, sum_55, permute_203, view_325, permute_199, view_322, permute_194, view_318, permute_182, view_306, sum_45, sum_46, permute_178, view_303, permute_174, view_300, sum_39, sum_40, permute_170, view_297, permute_166, view_294, permute_161, view_290, permute_149, view_278, sum_30, sum_31, permute_145, view_275, permute_141, view_272, sum_24, sum_25, permute_137, view_269, None, None, None, None, None], self._out_spec)
    