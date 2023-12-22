from __future__ import annotations



def forward(self, primals_1: "f32[30522, 768]", primals_2: "f32[512, 768]", primals_3: "f32[1024, 768]", primals_4: "f32[1024, 768]", primals_5: "f32[1024, 768]", primals_6: "f32[1024, 768]", primals_7: "f32[2, 768]", primals_8: "f32[768]", primals_9: "f32[768]", primals_10: "f32[768, 768]", primals_11: "f32[768]", primals_12: "f32[768, 768]", primals_13: "f32[768]", primals_14: "f32[768, 768]", primals_15: "f32[768]", primals_16: "f32[768, 768]", primals_17: "f32[768]", primals_18: "f32[768]", primals_19: "f32[768]", primals_20: "f32[3072, 768]", primals_21: "f32[3072]", primals_22: "f32[768, 3072]", primals_23: "f32[768]", primals_24: "f32[768]", primals_25: "f32[768]", primals_26: "f32[768, 768]", primals_27: "f32[768]", primals_28: "f32[768, 768]", primals_29: "f32[768]", primals_30: "f32[768, 768]", primals_31: "f32[768]", primals_32: "f32[768, 768]", primals_33: "f32[768]", primals_34: "f32[768]", primals_35: "f32[768]", primals_36: "f32[3072, 768]", primals_37: "f32[3072]", primals_38: "f32[768, 3072]", primals_39: "f32[768]", primals_40: "f32[768]", primals_41: "f32[768]", primals_42: "f32[768, 768]", primals_43: "f32[768]", primals_44: "f32[768, 768]", primals_45: "f32[768]", primals_46: "f32[768, 768]", primals_47: "f32[768]", primals_48: "f32[768, 768]", primals_49: "f32[768]", primals_50: "f32[768]", primals_51: "f32[768]", primals_52: "f32[3072, 768]", primals_53: "f32[3072]", primals_54: "f32[768, 3072]", primals_55: "f32[768]", primals_56: "f32[768]", primals_57: "f32[768]", primals_58: "f32[768, 768]", primals_59: "f32[768]", primals_60: "f32[768, 768]", primals_61: "f32[768]", primals_62: "f32[768, 768]", primals_63: "f32[768]", primals_64: "f32[768, 768]", primals_65: "f32[768]", primals_66: "f32[768]", primals_67: "f32[768]", primals_68: "f32[3072, 768]", primals_69: "f32[3072]", primals_70: "f32[768, 3072]", primals_71: "f32[768]", primals_72: "f32[768]", primals_73: "f32[768]", primals_74: "f32[768, 768]", primals_75: "f32[768]", primals_76: "f32[768, 768]", primals_77: "f32[768]", primals_78: "f32[768, 768]", primals_79: "f32[768]", primals_80: "f32[768, 768]", primals_81: "f32[768]", primals_82: "f32[768]", primals_83: "f32[768]", primals_84: "f32[3072, 768]", primals_85: "f32[3072]", primals_86: "f32[768, 3072]", primals_87: "f32[768]", primals_88: "f32[768]", primals_89: "f32[768]", primals_90: "f32[768, 768]", primals_91: "f32[768]", primals_92: "f32[768, 768]", primals_93: "f32[768]", primals_94: "f32[768, 768]", primals_95: "f32[768]", primals_96: "f32[768, 768]", primals_97: "f32[768]", primals_98: "f32[768]", primals_99: "f32[768]", primals_100: "f32[3072, 768]", primals_101: "f32[3072]", primals_102: "f32[768, 3072]", primals_103: "f32[768]", primals_104: "f32[768]", primals_105: "f32[768]", primals_106: "f32[768, 768]", primals_107: "f32[768]", primals_108: "f32[768, 768]", primals_109: "f32[768]", primals_110: "f32[768, 768]", primals_111: "f32[768]", primals_112: "f32[768, 768]", primals_113: "f32[768]", primals_114: "f32[768]", primals_115: "f32[768]", primals_116: "f32[3072, 768]", primals_117: "f32[3072]", primals_118: "f32[768, 3072]", primals_119: "f32[768]", primals_120: "f32[768]", primals_121: "f32[768]", primals_122: "f32[768, 768]", primals_123: "f32[768]", primals_124: "f32[768, 768]", primals_125: "f32[768]", primals_126: "f32[768, 768]", primals_127: "f32[768]", primals_128: "f32[768, 768]", primals_129: "f32[768]", primals_130: "f32[768]", primals_131: "f32[768]", primals_132: "f32[3072, 768]", primals_133: "f32[3072]", primals_134: "f32[768, 3072]", primals_135: "f32[768]", primals_136: "f32[768]", primals_137: "f32[768]", primals_138: "f32[768, 768]", primals_139: "f32[768]", primals_140: "f32[768, 768]", primals_141: "f32[768]", primals_142: "f32[768, 768]", primals_143: "f32[768]", primals_144: "f32[768, 768]", primals_145: "f32[768]", primals_146: "f32[768]", primals_147: "f32[768]", primals_148: "f32[3072, 768]", primals_149: "f32[3072]", primals_150: "f32[768, 3072]", primals_151: "f32[768]", primals_152: "f32[768]", primals_153: "f32[768]", primals_154: "f32[768, 768]", primals_155: "f32[768]", primals_156: "f32[768, 768]", primals_157: "f32[768]", primals_158: "f32[768, 768]", primals_159: "f32[768]", primals_160: "f32[768, 768]", primals_161: "f32[768]", primals_162: "f32[768]", primals_163: "f32[768]", primals_164: "f32[3072, 768]", primals_165: "f32[3072]", primals_166: "f32[768, 3072]", primals_167: "f32[768]", primals_168: "f32[768]", primals_169: "f32[768]", primals_170: "f32[768, 768]", primals_171: "f32[768]", primals_172: "f32[768, 768]", primals_173: "f32[768]", primals_174: "f32[768, 768]", primals_175: "f32[768]", primals_176: "f32[768, 768]", primals_177: "f32[768]", primals_178: "f32[768]", primals_179: "f32[768]", primals_180: "f32[3072, 768]", primals_181: "f32[3072]", primals_182: "f32[768, 3072]", primals_183: "f32[768]", primals_184: "f32[768]", primals_185: "f32[768]", primals_186: "f32[768, 768]", primals_187: "f32[768]", primals_188: "f32[768, 768]", primals_189: "f32[768]", primals_190: "f32[768, 768]", primals_191: "f32[768]", primals_192: "f32[768, 768]", primals_193: "f32[768]", primals_194: "f32[768]", primals_195: "f32[768]", primals_196: "f32[3072, 768]", primals_197: "f32[3072]", primals_198: "f32[768, 3072]", primals_199: "f32[768]", primals_200: "f32[768]", primals_201: "f32[768]", primals_202: "f32[768, 768]", primals_203: "f32[768]", primals_204: "f32[2, 768]", primals_205: "f32[2]", primals_206: "i64[1, 512]", primals_207: "i64[1, 512]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:810, code: token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
    full_default: "i64[1, 512]" = torch.ops.aten.full.default([1, 512], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:813, code: bbox = torch.zeros(input_shape + (4,), dtype=torch.long, device=device)
    full_2: "i64[1, 512, 4]" = torch.ops.aten.full.default([1, 512, 4], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:93, code: position_ids = self.position_ids[:, :seq_length]
    slice_1: "i64[1, 512]" = torch.ops.aten.slice.Tensor(primals_206, 0, 0, 9223372036854775807);  primals_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:99, code: inputs_embeds = self.word_embeddings(input_ids)
    embedding: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(primals_1, primals_207, 0);  primals_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:102, code: position_embeddings = self.position_embeddings(position_ids)
    embedding_1: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(primals_2, slice_1);  primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:104, code: left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
    slice_2: "i64[1, 512, 4]" = torch.ops.aten.slice.Tensor(full_2, 0, 0, 9223372036854775807);  full_2 = None
    slice_3: "i64[1, 512, 4]" = torch.ops.aten.slice.Tensor(slice_2, 1, 0, 9223372036854775807);  slice_2 = None
    select: "i64[1, 512]" = torch.ops.aten.select.int(slice_3, 2, 0)
    embedding_2: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(primals_3, select)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:105, code: upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
    select_1: "i64[1, 512]" = torch.ops.aten.select.int(slice_3, 2, 1)
    embedding_3: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(primals_4, select_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:106, code: right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
    select_2: "i64[1, 512]" = torch.ops.aten.select.int(slice_3, 2, 2)
    embedding_4: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(primals_3, select_2);  primals_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:107, code: lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
    select_3: "i64[1, 512]" = torch.ops.aten.select.int(slice_3, 2, 3);  slice_3 = None
    embedding_5: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(primals_4, select_3);  primals_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:111, code: h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
    embedding_6: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(primals_5, full_default);  primals_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:112, code: w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])
    embedding_7: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(primals_6, full_default);  primals_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:113, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    embedding_8: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(primals_7, full_default);  primals_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:116, code: words_embeddings
    add: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
    add_1: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
    add_2: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_1, embedding_3);  add_1 = embedding_3 = None
    add_3: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_2, embedding_4);  add_2 = embedding_4 = None
    add_4: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_3, embedding_5);  add_3 = embedding_5 = None
    add_5: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_4, embedding_6);  add_4 = embedding_6 = None
    add_6: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_5, embedding_7);  add_5 = embedding_7 = None
    add_7: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_6, embedding_8);  add_6 = embedding_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:126, code: embeddings = self.LayerNorm(embeddings)
    var_mean = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 512, 1]" = var_mean[0]
    getitem_1: "f32[1, 512, 1]" = var_mean[1];  var_mean = None
    add_8: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
    rsqrt: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
    sub_3: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_7, getitem_1);  add_7 = getitem_1 = None
    mul_1: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt);  sub_3 = None
    mul_2: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_1, primals_8)
    add_9: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_2, primals_9);  mul_2 = primals_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:127, code: embeddings = self.dropout(embeddings)
    native_dropout = torch.ops.aten.native_dropout.default(add_9, 0.1, True);  add_9 = None
    getitem_2: "f32[1, 512, 768]" = native_dropout[0]
    getitem_3: "b8[1, 512, 768]" = native_dropout[1];  native_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view: "f32[512, 768]" = torch.ops.aten.reshape.default(getitem_2, [512, 768])
    permute: "f32[768, 768]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
    
    # No stacktrace found for following nodes
    mm_default_36: "f32[512, 768]" = torch.ops.aten.mm.default(view, permute)
    add_tensor_36: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_36, primals_11);  mm_default_36 = primals_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_1: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_36, [1, 512, 768]);  add_tensor_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_1: "f32[768, 768]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
    
    # No stacktrace found for following nodes
    mm_default_35: "f32[512, 768]" = torch.ops.aten.mm.default(view, permute_1)
    add_tensor_35: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_35, primals_13);  mm_default_35 = primals_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_3: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_35, [1, 512, 768]);  add_tensor_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_4: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_3, [1, 512, 12, 64]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_2: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_3: "f32[768, 768]" = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None
    
    # No stacktrace found for following nodes
    mm_default_34: "f32[512, 768]" = torch.ops.aten.mm.default(view, permute_3)
    add_tensor_34: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_34, primals_15);  mm_default_34 = primals_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_6: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_34, [1, 512, 768]);  add_tensor_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_7: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_6, [1, 512, 12, 64]);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_4: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_8: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_1, [1, 512, 12, 64]);  view_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_5: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
    
    # No stacktrace found for following nodes
    clone_default_33: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    clone_default_34: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
    clone_default_35: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    _scaled_dot_product_efficient_attention_default_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_33, clone_default_34, clone_default_35, None, True, 0.1, scale = 0.125)
    getitem_203: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default_11[0]
    getitem_204: "f32[1, 12, 512]" = _scaled_dot_product_efficient_attention_default_11[1]
    getitem_205: "i64[]" = _scaled_dot_product_efficient_attention_default_11[2]
    getitem_206: "i64[]" = _scaled_dot_product_efficient_attention_default_11[3];  _scaled_dot_product_efficient_attention_default_11 = None
    alias_default_22: "f32[1, 12, 512, 64]" = torch.ops.aten.alias.default(getitem_203)
    alias_default_23: "f32[1, 12, 512, 64]" = torch.ops.aten.alias.default(alias_default_22);  alias_default_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_7: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_203, [0, 2, 1, 3]);  getitem_203 = None
    clone: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_15: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone, [1, 512, 768]);  clone = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_16: "f32[512, 768]" = torch.ops.aten.reshape.default(view_15, [512, 768]);  view_15 = None
    permute_8: "f32[768, 768]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
    addmm_3: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_17, view_16, permute_8);  primals_17 = None
    view_17: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_3, [1, 512, 768]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    native_dropout_2 = torch.ops.aten.native_dropout.default(view_17, 0.1, True);  view_17 = None
    getitem_6: "f32[1, 512, 768]" = native_dropout_2[0]
    getitem_7: "b8[1, 512, 768]" = native_dropout_2[1];  native_dropout_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_11: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_6, getitem_2);  getitem_6 = getitem_2 = None
    var_mean_1 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
    getitem_8: "f32[1, 512, 1]" = var_mean_1[0]
    getitem_9: "f32[1, 512, 1]" = var_mean_1[1];  var_mean_1 = None
    add_12: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
    rsqrt_1: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
    sub_5: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_11, getitem_9);  add_11 = getitem_9 = None
    mul_3: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_1);  sub_5 = None
    mul_4: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_3, primals_18)
    add_13: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_4, primals_19);  mul_4 = primals_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_18: "f32[512, 768]" = torch.ops.aten.reshape.default(add_13, [512, 768])
    permute_9: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_20, [1, 0]);  primals_20 = None
    addmm_4: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_21, view_18, permute_9);  primals_21 = None
    view_19: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_4, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_5: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.5)
    mul_6: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476);  view_19 = None
    erf: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
    add_14: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_7: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_5, add_14);  mul_5 = add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_20: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_7, [512, 3072]);  mul_7 = None
    permute_10: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_22, [1, 0]);  primals_22 = None
    addmm_5: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_23, view_20, permute_10);  primals_23 = None
    view_21: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_5, [1, 512, 768]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    native_dropout_3 = torch.ops.aten.native_dropout.default(view_21, 0.1, True);  view_21 = None
    getitem_10: "f32[1, 512, 768]" = native_dropout_3[0]
    getitem_11: "b8[1, 512, 768]" = native_dropout_3[1];  native_dropout_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_15: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_10, add_13);  getitem_10 = add_13 = None
    var_mean_2 = torch.ops.aten.var_mean.correction(add_15, [2], correction = 0, keepdim = True)
    getitem_12: "f32[1, 512, 1]" = var_mean_2[0]
    getitem_13: "f32[1, 512, 1]" = var_mean_2[1];  var_mean_2 = None
    add_16: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
    rsqrt_2: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    sub_6: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_15, getitem_13);  add_15 = getitem_13 = None
    mul_8: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_2);  sub_6 = None
    mul_9: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_8, primals_24)
    add_17: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_9, primals_25);  mul_9 = primals_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_22: "f32[512, 768]" = torch.ops.aten.reshape.default(add_17, [512, 768])
    permute_11: "f32[768, 768]" = torch.ops.aten.permute.default(primals_26, [1, 0]);  primals_26 = None
    
    # No stacktrace found for following nodes
    mm_default_33: "f32[512, 768]" = torch.ops.aten.mm.default(view_22, permute_11)
    add_tensor_33: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_33, primals_27);  mm_default_33 = primals_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_23: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_33, [1, 512, 768]);  add_tensor_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_12: "f32[768, 768]" = torch.ops.aten.permute.default(primals_28, [1, 0]);  primals_28 = None
    
    # No stacktrace found for following nodes
    mm_default_32: "f32[512, 768]" = torch.ops.aten.mm.default(view_22, permute_12)
    add_tensor_32: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_32, primals_29);  mm_default_32 = primals_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_25: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_32, [1, 512, 768]);  add_tensor_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_26: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_25, [1, 512, 12, 64]);  view_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_13: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_14: "f32[768, 768]" = torch.ops.aten.permute.default(primals_30, [1, 0]);  primals_30 = None
    
    # No stacktrace found for following nodes
    mm_default_31: "f32[512, 768]" = torch.ops.aten.mm.default(view_22, permute_14)
    add_tensor_31: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_31, primals_31);  mm_default_31 = primals_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_28: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_31, [1, 512, 768]);  add_tensor_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_29: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_28, [1, 512, 12, 64]);  view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_15: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_30: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_23, [1, 512, 12, 64]);  view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_16: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
    
    # No stacktrace found for following nodes
    clone_default_30: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
    clone_default_31: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
    clone_default_32: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
    _scaled_dot_product_efficient_attention_default_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_30, clone_default_31, clone_default_32, None, True, 0.1, scale = 0.125)
    getitem_196: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default_10[0]
    getitem_197: "f32[1, 12, 512]" = _scaled_dot_product_efficient_attention_default_10[1]
    getitem_198: "i64[]" = _scaled_dot_product_efficient_attention_default_10[2]
    getitem_199: "i64[]" = _scaled_dot_product_efficient_attention_default_10[3];  _scaled_dot_product_efficient_attention_default_10 = None
    alias_default_20: "f32[1, 12, 512, 64]" = torch.ops.aten.alias.default(getitem_196)
    alias_default_21: "f32[1, 12, 512, 64]" = torch.ops.aten.alias.default(alias_default_20);  alias_default_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_18: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_196, [0, 2, 1, 3]);  getitem_196 = None
    clone_1: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_37: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_1, [1, 512, 768]);  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_38: "f32[512, 768]" = torch.ops.aten.reshape.default(view_37, [512, 768]);  view_37 = None
    permute_19: "f32[768, 768]" = torch.ops.aten.permute.default(primals_32, [1, 0]);  primals_32 = None
    addmm_9: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_33, view_38, permute_19);  primals_33 = None
    view_39: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_9, [1, 512, 768]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    native_dropout_5 = torch.ops.aten.native_dropout.default(view_39, 0.1, True);  view_39 = None
    getitem_16: "f32[1, 512, 768]" = native_dropout_5[0]
    getitem_17: "b8[1, 512, 768]" = native_dropout_5[1];  native_dropout_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_19: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_16, add_17);  getitem_16 = add_17 = None
    var_mean_3 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
    getitem_18: "f32[1, 512, 1]" = var_mean_3[0]
    getitem_19: "f32[1, 512, 1]" = var_mean_3[1];  var_mean_3 = None
    add_20: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
    rsqrt_3: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
    sub_8: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_19, getitem_19);  add_19 = getitem_19 = None
    mul_10: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_3);  sub_8 = None
    mul_11: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_10, primals_34)
    add_21: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_11, primals_35);  mul_11 = primals_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_40: "f32[512, 768]" = torch.ops.aten.reshape.default(add_21, [512, 768])
    permute_20: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_36, [1, 0]);  primals_36 = None
    addmm_10: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_37, view_40, permute_20);  primals_37 = None
    view_41: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_10, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_12: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, 0.5)
    mul_13: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476);  view_41 = None
    erf_1: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
    add_22: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_14: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_12, add_22);  mul_12 = add_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_42: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_14, [512, 3072]);  mul_14 = None
    permute_21: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_38, [1, 0]);  primals_38 = None
    addmm_11: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_39, view_42, permute_21);  primals_39 = None
    view_43: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_11, [1, 512, 768]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    native_dropout_6 = torch.ops.aten.native_dropout.default(view_43, 0.1, True);  view_43 = None
    getitem_20: "f32[1, 512, 768]" = native_dropout_6[0]
    getitem_21: "b8[1, 512, 768]" = native_dropout_6[1];  native_dropout_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_23: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_20, add_21);  getitem_20 = add_21 = None
    var_mean_4 = torch.ops.aten.var_mean.correction(add_23, [2], correction = 0, keepdim = True)
    getitem_22: "f32[1, 512, 1]" = var_mean_4[0]
    getitem_23: "f32[1, 512, 1]" = var_mean_4[1];  var_mean_4 = None
    add_24: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
    rsqrt_4: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
    sub_9: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_23, getitem_23);  add_23 = getitem_23 = None
    mul_15: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_4);  sub_9 = None
    mul_16: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_15, primals_40)
    add_25: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_16, primals_41);  mul_16 = primals_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_44: "f32[512, 768]" = torch.ops.aten.reshape.default(add_25, [512, 768])
    permute_22: "f32[768, 768]" = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
    
    # No stacktrace found for following nodes
    mm_default_30: "f32[512, 768]" = torch.ops.aten.mm.default(view_44, permute_22)
    add_tensor_30: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_30, primals_43);  mm_default_30 = primals_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_45: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_30, [1, 512, 768]);  add_tensor_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_23: "f32[768, 768]" = torch.ops.aten.permute.default(primals_44, [1, 0]);  primals_44 = None
    
    # No stacktrace found for following nodes
    mm_default_29: "f32[512, 768]" = torch.ops.aten.mm.default(view_44, permute_23)
    add_tensor_29: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_29, primals_45);  mm_default_29 = primals_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_47: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_29, [1, 512, 768]);  add_tensor_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_48: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_47, [1, 512, 12, 64]);  view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_24: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_25: "f32[768, 768]" = torch.ops.aten.permute.default(primals_46, [1, 0]);  primals_46 = None
    
    # No stacktrace found for following nodes
    mm_default_28: "f32[512, 768]" = torch.ops.aten.mm.default(view_44, permute_25)
    add_tensor_28: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_28, primals_47);  mm_default_28 = primals_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_50: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_28, [1, 512, 768]);  add_tensor_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_51: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_50, [1, 512, 12, 64]);  view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_26: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_52: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_45, [1, 512, 12, 64]);  view_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_27: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
    
    # No stacktrace found for following nodes
    clone_default_27: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
    clone_default_28: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
    clone_default_29: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
    _scaled_dot_product_efficient_attention_default_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_27, clone_default_28, clone_default_29, None, True, 0.1, scale = 0.125)
    getitem_189: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default_9[0]
    getitem_190: "f32[1, 12, 512]" = _scaled_dot_product_efficient_attention_default_9[1]
    getitem_191: "i64[]" = _scaled_dot_product_efficient_attention_default_9[2]
    getitem_192: "i64[]" = _scaled_dot_product_efficient_attention_default_9[3];  _scaled_dot_product_efficient_attention_default_9 = None
    alias_default_18: "f32[1, 12, 512, 64]" = torch.ops.aten.alias.default(getitem_189)
    alias_default_19: "f32[1, 12, 512, 64]" = torch.ops.aten.alias.default(alias_default_18);  alias_default_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_29: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_189, [0, 2, 1, 3]);  getitem_189 = None
    clone_2: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_59: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_2, [1, 512, 768]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_60: "f32[512, 768]" = torch.ops.aten.reshape.default(view_59, [512, 768]);  view_59 = None
    permute_30: "f32[768, 768]" = torch.ops.aten.permute.default(primals_48, [1, 0]);  primals_48 = None
    addmm_15: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_49, view_60, permute_30);  primals_49 = None
    view_61: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_15, [1, 512, 768]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    native_dropout_8 = torch.ops.aten.native_dropout.default(view_61, 0.1, True);  view_61 = None
    getitem_26: "f32[1, 512, 768]" = native_dropout_8[0]
    getitem_27: "b8[1, 512, 768]" = native_dropout_8[1];  native_dropout_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_27: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_26, add_25);  getitem_26 = add_25 = None
    var_mean_5 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 512, 1]" = var_mean_5[0]
    getitem_29: "f32[1, 512, 1]" = var_mean_5[1];  var_mean_5 = None
    add_28: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
    rsqrt_5: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    sub_11: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_27, getitem_29);  add_27 = getitem_29 = None
    mul_17: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_5);  sub_11 = None
    mul_18: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_17, primals_50)
    add_29: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_18, primals_51);  mul_18 = primals_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_62: "f32[512, 768]" = torch.ops.aten.reshape.default(add_29, [512, 768])
    permute_31: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_52, [1, 0]);  primals_52 = None
    addmm_16: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_53, view_62, permute_31);  primals_53 = None
    view_63: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_16, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_19: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, 0.5)
    mul_20: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476);  view_63 = None
    erf_2: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_20);  mul_20 = None
    add_30: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_21: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_19, add_30);  mul_19 = add_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_64: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_21, [512, 3072]);  mul_21 = None
    permute_32: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_54, [1, 0]);  primals_54 = None
    addmm_17: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_55, view_64, permute_32);  primals_55 = None
    view_65: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_17, [1, 512, 768]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    native_dropout_9 = torch.ops.aten.native_dropout.default(view_65, 0.1, True);  view_65 = None
    getitem_30: "f32[1, 512, 768]" = native_dropout_9[0]
    getitem_31: "b8[1, 512, 768]" = native_dropout_9[1];  native_dropout_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_31: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_30, add_29);  getitem_30 = add_29 = None
    var_mean_6 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
    getitem_32: "f32[1, 512, 1]" = var_mean_6[0]
    getitem_33: "f32[1, 512, 1]" = var_mean_6[1];  var_mean_6 = None
    add_32: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
    rsqrt_6: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    sub_12: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_31, getitem_33);  add_31 = getitem_33 = None
    mul_22: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_6);  sub_12 = None
    mul_23: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_22, primals_56)
    add_33: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_23, primals_57);  mul_23 = primals_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_66: "f32[512, 768]" = torch.ops.aten.reshape.default(add_33, [512, 768])
    permute_33: "f32[768, 768]" = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
    
    # No stacktrace found for following nodes
    mm_default_27: "f32[512, 768]" = torch.ops.aten.mm.default(view_66, permute_33)
    add_tensor_27: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_27, primals_59);  mm_default_27 = primals_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_67: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_27, [1, 512, 768]);  add_tensor_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_34: "f32[768, 768]" = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
    
    # No stacktrace found for following nodes
    mm_default_26: "f32[512, 768]" = torch.ops.aten.mm.default(view_66, permute_34)
    add_tensor_26: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_26, primals_61);  mm_default_26 = primals_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_69: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_26, [1, 512, 768]);  add_tensor_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_70: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_69, [1, 512, 12, 64]);  view_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_35: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_36: "f32[768, 768]" = torch.ops.aten.permute.default(primals_62, [1, 0]);  primals_62 = None
    
    # No stacktrace found for following nodes
    mm_default_25: "f32[512, 768]" = torch.ops.aten.mm.default(view_66, permute_36)
    add_tensor_25: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_25, primals_63);  mm_default_25 = primals_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_72: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_25, [1, 512, 768]);  add_tensor_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_73: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_72, [1, 512, 12, 64]);  view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_37: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_74: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_67, [1, 512, 12, 64]);  view_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_38: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
    
    # No stacktrace found for following nodes
    clone_default_24: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
    clone_default_25: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
    clone_default_26: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
    _scaled_dot_product_efficient_attention_default_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_24, clone_default_25, clone_default_26, None, True, 0.1, scale = 0.125)
    getitem_182: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default_8[0]
    getitem_183: "f32[1, 12, 512]" = _scaled_dot_product_efficient_attention_default_8[1]
    getitem_184: "i64[]" = _scaled_dot_product_efficient_attention_default_8[2]
    getitem_185: "i64[]" = _scaled_dot_product_efficient_attention_default_8[3];  _scaled_dot_product_efficient_attention_default_8 = None
    alias_default_16: "f32[1, 12, 512, 64]" = torch.ops.aten.alias.default(getitem_182)
    alias_default_17: "f32[1, 12, 512, 64]" = torch.ops.aten.alias.default(alias_default_16);  alias_default_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_40: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_182, [0, 2, 1, 3]);  getitem_182 = None
    clone_3: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_81: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_3, [1, 512, 768]);  clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_82: "f32[512, 768]" = torch.ops.aten.reshape.default(view_81, [512, 768]);  view_81 = None
    permute_41: "f32[768, 768]" = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
    addmm_21: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_65, view_82, permute_41);  primals_65 = None
    view_83: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_21, [1, 512, 768]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    native_dropout_11 = torch.ops.aten.native_dropout.default(view_83, 0.1, True);  view_83 = None
    getitem_36: "f32[1, 512, 768]" = native_dropout_11[0]
    getitem_37: "b8[1, 512, 768]" = native_dropout_11[1];  native_dropout_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_35: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_36, add_33);  getitem_36 = add_33 = None
    var_mean_7 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
    getitem_38: "f32[1, 512, 1]" = var_mean_7[0]
    getitem_39: "f32[1, 512, 1]" = var_mean_7[1];  var_mean_7 = None
    add_36: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
    rsqrt_7: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    sub_14: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_39);  add_35 = getitem_39 = None
    mul_24: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_7);  sub_14 = None
    mul_25: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_24, primals_66)
    add_37: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_25, primals_67);  mul_25 = primals_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_84: "f32[512, 768]" = torch.ops.aten.reshape.default(add_37, [512, 768])
    permute_42: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_68, [1, 0]);  primals_68 = None
    addmm_22: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_69, view_84, permute_42);  primals_69 = None
    view_85: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_22, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_26: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, 0.5)
    mul_27: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476);  view_85 = None
    erf_3: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_27);  mul_27 = None
    add_38: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_28: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_26, add_38);  mul_26 = add_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_86: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_28, [512, 3072]);  mul_28 = None
    permute_43: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_70, [1, 0]);  primals_70 = None
    addmm_23: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_71, view_86, permute_43);  primals_71 = None
    view_87: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_23, [1, 512, 768]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    native_dropout_12 = torch.ops.aten.native_dropout.default(view_87, 0.1, True);  view_87 = None
    getitem_40: "f32[1, 512, 768]" = native_dropout_12[0]
    getitem_41: "b8[1, 512, 768]" = native_dropout_12[1];  native_dropout_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_39: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_40, add_37);  getitem_40 = add_37 = None
    var_mean_8 = torch.ops.aten.var_mean.correction(add_39, [2], correction = 0, keepdim = True)
    getitem_42: "f32[1, 512, 1]" = var_mean_8[0]
    getitem_43: "f32[1, 512, 1]" = var_mean_8[1];  var_mean_8 = None
    add_40: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
    rsqrt_8: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
    sub_15: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_39, getitem_43);  add_39 = getitem_43 = None
    mul_29: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_8);  sub_15 = None
    mul_30: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_29, primals_72)
    add_41: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_30, primals_73);  mul_30 = primals_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_88: "f32[512, 768]" = torch.ops.aten.reshape.default(add_41, [512, 768])
    permute_44: "f32[768, 768]" = torch.ops.aten.permute.default(primals_74, [1, 0]);  primals_74 = None
    
    # No stacktrace found for following nodes
    mm_default_24: "f32[512, 768]" = torch.ops.aten.mm.default(view_88, permute_44)
    add_tensor_24: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_24, primals_75);  mm_default_24 = primals_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_89: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_24, [1, 512, 768]);  add_tensor_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_45: "f32[768, 768]" = torch.ops.aten.permute.default(primals_76, [1, 0]);  primals_76 = None
    
    # No stacktrace found for following nodes
    mm_default_23: "f32[512, 768]" = torch.ops.aten.mm.default(view_88, permute_45)
    add_tensor_23: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_23, primals_77);  mm_default_23 = primals_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_91: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_23, [1, 512, 768]);  add_tensor_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_92: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_91, [1, 512, 12, 64]);  view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_46: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_47: "f32[768, 768]" = torch.ops.aten.permute.default(primals_78, [1, 0]);  primals_78 = None
    
    # No stacktrace found for following nodes
    mm_default_22: "f32[512, 768]" = torch.ops.aten.mm.default(view_88, permute_47)
    add_tensor_22: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_22, primals_79);  mm_default_22 = primals_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_94: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_22, [1, 512, 768]);  add_tensor_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_95: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_94, [1, 512, 12, 64]);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_48: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_96: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_89, [1, 512, 12, 64]);  view_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_49: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
    
    # No stacktrace found for following nodes
    clone_default_21: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
    clone_default_22: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
    clone_default_23: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
    _scaled_dot_product_efficient_attention_default_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_21, clone_default_22, clone_default_23, None, True, 0.1, scale = 0.125)
    getitem_175: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default_7[0]
    getitem_176: "f32[1, 12, 512]" = _scaled_dot_product_efficient_attention_default_7[1]
    getitem_177: "i64[]" = _scaled_dot_product_efficient_attention_default_7[2]
    getitem_178: "i64[]" = _scaled_dot_product_efficient_attention_default_7[3];  _scaled_dot_product_efficient_attention_default_7 = None
    alias_default_14: "f32[1, 12, 512, 64]" = torch.ops.aten.alias.default(getitem_175)
    alias_default_15: "f32[1, 12, 512, 64]" = torch.ops.aten.alias.default(alias_default_14);  alias_default_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_51: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_175, [0, 2, 1, 3]);  getitem_175 = None
    clone_4: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_103: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_4, [1, 512, 768]);  clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_104: "f32[512, 768]" = torch.ops.aten.reshape.default(view_103, [512, 768]);  view_103 = None
    permute_52: "f32[768, 768]" = torch.ops.aten.permute.default(primals_80, [1, 0]);  primals_80 = None
    addmm_27: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_81, view_104, permute_52);  primals_81 = None
    view_105: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_27, [1, 512, 768]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    native_dropout_14 = torch.ops.aten.native_dropout.default(view_105, 0.1, True);  view_105 = None
    getitem_46: "f32[1, 512, 768]" = native_dropout_14[0]
    getitem_47: "b8[1, 512, 768]" = native_dropout_14[1];  native_dropout_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_43: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_46, add_41);  getitem_46 = add_41 = None
    var_mean_9 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
    getitem_48: "f32[1, 512, 1]" = var_mean_9[0]
    getitem_49: "f32[1, 512, 1]" = var_mean_9[1];  var_mean_9 = None
    add_44: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
    rsqrt_9: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
    sub_17: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_43, getitem_49);  add_43 = getitem_49 = None
    mul_31: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_9);  sub_17 = None
    mul_32: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_31, primals_82)
    add_45: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_32, primals_83);  mul_32 = primals_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_106: "f32[512, 768]" = torch.ops.aten.reshape.default(add_45, [512, 768])
    permute_53: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_84, [1, 0]);  primals_84 = None
    addmm_28: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_85, view_106, permute_53);  primals_85 = None
    view_107: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_28, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_33: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.5)
    mul_34: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476);  view_107 = None
    erf_4: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_34);  mul_34 = None
    add_46: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_35: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_33, add_46);  mul_33 = add_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_108: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_35, [512, 3072]);  mul_35 = None
    permute_54: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_86, [1, 0]);  primals_86 = None
    addmm_29: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_87, view_108, permute_54);  primals_87 = None
    view_109: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_29, [1, 512, 768]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    native_dropout_15 = torch.ops.aten.native_dropout.default(view_109, 0.1, True);  view_109 = None
    getitem_50: "f32[1, 512, 768]" = native_dropout_15[0]
    getitem_51: "b8[1, 512, 768]" = native_dropout_15[1];  native_dropout_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_47: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_50, add_45);  getitem_50 = add_45 = None
    var_mean_10 = torch.ops.aten.var_mean.correction(add_47, [2], correction = 0, keepdim = True)
    getitem_52: "f32[1, 512, 1]" = var_mean_10[0]
    getitem_53: "f32[1, 512, 1]" = var_mean_10[1];  var_mean_10 = None
    add_48: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-12);  getitem_52 = None
    rsqrt_10: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
    sub_18: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_47, getitem_53);  add_47 = getitem_53 = None
    mul_36: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_10);  sub_18 = None
    mul_37: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_36, primals_88)
    add_49: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_37, primals_89);  mul_37 = primals_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_110: "f32[512, 768]" = torch.ops.aten.reshape.default(add_49, [512, 768])
    permute_55: "f32[768, 768]" = torch.ops.aten.permute.default(primals_90, [1, 0]);  primals_90 = None
    
    # No stacktrace found for following nodes
    mm_default_21: "f32[512, 768]" = torch.ops.aten.mm.default(view_110, permute_55)
    add_tensor_21: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_21, primals_91);  mm_default_21 = primals_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_111: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_21, [1, 512, 768]);  add_tensor_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_56: "f32[768, 768]" = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
    
    # No stacktrace found for following nodes
    mm_default_20: "f32[512, 768]" = torch.ops.aten.mm.default(view_110, permute_56)
    add_tensor_20: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_20, primals_93);  mm_default_20 = primals_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_113: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_20, [1, 512, 768]);  add_tensor_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_114: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_113, [1, 512, 12, 64]);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_57: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_58: "f32[768, 768]" = torch.ops.aten.permute.default(primals_94, [1, 0]);  primals_94 = None
    
    # No stacktrace found for following nodes
    mm_default_19: "f32[512, 768]" = torch.ops.aten.mm.default(view_110, permute_58)
    add_tensor_19: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_19, primals_95);  mm_default_19 = primals_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_116: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_19, [1, 512, 768]);  add_tensor_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_117: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_116, [1, 512, 12, 64]);  view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_59: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_118: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_111, [1, 512, 12, 64]);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_60: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
    
    # No stacktrace found for following nodes
    clone_default_18: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
    clone_default_19: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_57, memory_format = torch.contiguous_format);  permute_57 = None
    clone_default_20: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
    _scaled_dot_product_efficient_attention_default_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_18, clone_default_19, clone_default_20, None, True, 0.1, scale = 0.125)
    getitem_168: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default_6[0]
    getitem_169: "f32[1, 12, 512]" = _scaled_dot_product_efficient_attention_default_6[1]
    getitem_170: "i64[]" = _scaled_dot_product_efficient_attention_default_6[2]
    getitem_171: "i64[]" = _scaled_dot_product_efficient_attention_default_6[3];  _scaled_dot_product_efficient_attention_default_6 = None
    alias_default_12: "f32[1, 12, 512, 64]" = torch.ops.aten.alias.default(getitem_168)
    alias_default_13: "f32[1, 12, 512, 64]" = torch.ops.aten.alias.default(alias_default_12);  alias_default_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_62: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_168, [0, 2, 1, 3]);  getitem_168 = None
    clone_5: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_125: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_5, [1, 512, 768]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_126: "f32[512, 768]" = torch.ops.aten.reshape.default(view_125, [512, 768]);  view_125 = None
    permute_63: "f32[768, 768]" = torch.ops.aten.permute.default(primals_96, [1, 0]);  primals_96 = None
    addmm_33: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_97, view_126, permute_63);  primals_97 = None
    view_127: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_33, [1, 512, 768]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    native_dropout_17 = torch.ops.aten.native_dropout.default(view_127, 0.1, True);  view_127 = None
    getitem_56: "f32[1, 512, 768]" = native_dropout_17[0]
    getitem_57: "b8[1, 512, 768]" = native_dropout_17[1];  native_dropout_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_51: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_56, add_49);  getitem_56 = add_49 = None
    var_mean_11 = torch.ops.aten.var_mean.correction(add_51, [2], correction = 0, keepdim = True)
    getitem_58: "f32[1, 512, 1]" = var_mean_11[0]
    getitem_59: "f32[1, 512, 1]" = var_mean_11[1];  var_mean_11 = None
    add_52: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-12);  getitem_58 = None
    rsqrt_11: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    sub_20: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_51, getitem_59);  add_51 = getitem_59 = None
    mul_38: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_11);  sub_20 = None
    mul_39: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_38, primals_98)
    add_53: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_39, primals_99);  mul_39 = primals_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_128: "f32[512, 768]" = torch.ops.aten.reshape.default(add_53, [512, 768])
    permute_64: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_100, [1, 0]);  primals_100 = None
    addmm_34: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_101, view_128, permute_64);  primals_101 = None
    view_129: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_34, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_40: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, 0.5)
    mul_41: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476);  view_129 = None
    erf_5: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_41);  mul_41 = None
    add_54: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_42: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_40, add_54);  mul_40 = add_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_130: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_42, [512, 3072]);  mul_42 = None
    permute_65: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_102, [1, 0]);  primals_102 = None
    addmm_35: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_103, view_130, permute_65);  primals_103 = None
    view_131: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_35, [1, 512, 768]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    native_dropout_18 = torch.ops.aten.native_dropout.default(view_131, 0.1, True);  view_131 = None
    getitem_60: "f32[1, 512, 768]" = native_dropout_18[0]
    getitem_61: "b8[1, 512, 768]" = native_dropout_18[1];  native_dropout_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_55: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_60, add_53);  getitem_60 = add_53 = None
    var_mean_12 = torch.ops.aten.var_mean.correction(add_55, [2], correction = 0, keepdim = True)
    getitem_62: "f32[1, 512, 1]" = var_mean_12[0]
    getitem_63: "f32[1, 512, 1]" = var_mean_12[1];  var_mean_12 = None
    add_56: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-12);  getitem_62 = None
    rsqrt_12: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    sub_21: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_55, getitem_63);  add_55 = getitem_63 = None
    mul_43: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_12);  sub_21 = None
    mul_44: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_43, primals_104)
    add_57: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_44, primals_105);  mul_44 = primals_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_132: "f32[512, 768]" = torch.ops.aten.reshape.default(add_57, [512, 768])
    permute_66: "f32[768, 768]" = torch.ops.aten.permute.default(primals_106, [1, 0]);  primals_106 = None
    
    # No stacktrace found for following nodes
    mm_default_18: "f32[512, 768]" = torch.ops.aten.mm.default(view_132, permute_66)
    add_tensor_18: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_18, primals_107);  mm_default_18 = primals_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_133: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_18, [1, 512, 768]);  add_tensor_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_67: "f32[768, 768]" = torch.ops.aten.permute.default(primals_108, [1, 0]);  primals_108 = None
    
    # No stacktrace found for following nodes
    mm_default_17: "f32[512, 768]" = torch.ops.aten.mm.default(view_132, permute_67)
    add_tensor_17: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_17, primals_109);  mm_default_17 = primals_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_135: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_17, [1, 512, 768]);  add_tensor_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_136: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_135, [1, 512, 12, 64]);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_68: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_69: "f32[768, 768]" = torch.ops.aten.permute.default(primals_110, [1, 0]);  primals_110 = None
    
    # No stacktrace found for following nodes
    mm_default_16: "f32[512, 768]" = torch.ops.aten.mm.default(view_132, permute_69)
    add_tensor_16: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_16, primals_111);  mm_default_16 = primals_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_138: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_16, [1, 512, 768]);  add_tensor_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_139: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_138, [1, 512, 12, 64]);  view_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_70: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_139, [0, 2, 1, 3]);  view_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_140: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_133, [1, 512, 12, 64]);  view_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_71: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None
    
    # No stacktrace found for following nodes
    clone_default_15: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
    clone_default_16: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
    clone_default_17: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
    _scaled_dot_product_efficient_attention_default_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_15, clone_default_16, clone_default_17, None, True, 0.1, scale = 0.125)
    getitem_161: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default_5[0]
    getitem_162: "f32[1, 12, 512]" = _scaled_dot_product_efficient_attention_default_5[1]
    getitem_163: "i64[]" = _scaled_dot_product_efficient_attention_default_5[2]
    getitem_164: "i64[]" = _scaled_dot_product_efficient_attention_default_5[3];  _scaled_dot_product_efficient_attention_default_5 = None
    alias_default_10: "f32[1, 12, 512, 64]" = torch.ops.aten.alias.default(getitem_161)
    alias_default_11: "f32[1, 12, 512, 64]" = torch.ops.aten.alias.default(alias_default_10);  alias_default_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_73: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_161, [0, 2, 1, 3]);  getitem_161 = None
    clone_6: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_147: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_6, [1, 512, 768]);  clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_148: "f32[512, 768]" = torch.ops.aten.reshape.default(view_147, [512, 768]);  view_147 = None
    permute_74: "f32[768, 768]" = torch.ops.aten.permute.default(primals_112, [1, 0]);  primals_112 = None
    addmm_39: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_113, view_148, permute_74);  primals_113 = None
    view_149: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_39, [1, 512, 768]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    native_dropout_20 = torch.ops.aten.native_dropout.default(view_149, 0.1, True);  view_149 = None
    getitem_66: "f32[1, 512, 768]" = native_dropout_20[0]
    getitem_67: "b8[1, 512, 768]" = native_dropout_20[1];  native_dropout_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_59: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_66, add_57);  getitem_66 = add_57 = None
    var_mean_13 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
    getitem_68: "f32[1, 512, 1]" = var_mean_13[0]
    getitem_69: "f32[1, 512, 1]" = var_mean_13[1];  var_mean_13 = None
    add_60: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-12);  getitem_68 = None
    rsqrt_13: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    sub_23: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_59, getitem_69);  add_59 = getitem_69 = None
    mul_45: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_13);  sub_23 = None
    mul_46: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_45, primals_114)
    add_61: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_46, primals_115);  mul_46 = primals_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_150: "f32[512, 768]" = torch.ops.aten.reshape.default(add_61, [512, 768])
    permute_75: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_116, [1, 0]);  primals_116 = None
    addmm_40: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_117, view_150, permute_75);  primals_117 = None
    view_151: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_40, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_47: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, 0.5)
    mul_48: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476);  view_151 = None
    erf_6: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_48);  mul_48 = None
    add_62: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_49: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_47, add_62);  mul_47 = add_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_152: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_49, [512, 3072]);  mul_49 = None
    permute_76: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_118, [1, 0]);  primals_118 = None
    addmm_41: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_119, view_152, permute_76);  primals_119 = None
    view_153: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_41, [1, 512, 768]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    native_dropout_21 = torch.ops.aten.native_dropout.default(view_153, 0.1, True);  view_153 = None
    getitem_70: "f32[1, 512, 768]" = native_dropout_21[0]
    getitem_71: "b8[1, 512, 768]" = native_dropout_21[1];  native_dropout_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_63: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_70, add_61);  getitem_70 = add_61 = None
    var_mean_14 = torch.ops.aten.var_mean.correction(add_63, [2], correction = 0, keepdim = True)
    getitem_72: "f32[1, 512, 1]" = var_mean_14[0]
    getitem_73: "f32[1, 512, 1]" = var_mean_14[1];  var_mean_14 = None
    add_64: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-12);  getitem_72 = None
    rsqrt_14: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
    sub_24: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_63, getitem_73);  add_63 = getitem_73 = None
    mul_50: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_14);  sub_24 = None
    mul_51: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_50, primals_120)
    add_65: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_51, primals_121);  mul_51 = primals_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_154: "f32[512, 768]" = torch.ops.aten.reshape.default(add_65, [512, 768])
    permute_77: "f32[768, 768]" = torch.ops.aten.permute.default(primals_122, [1, 0]);  primals_122 = None
    
    # No stacktrace found for following nodes
    mm_default_15: "f32[512, 768]" = torch.ops.aten.mm.default(view_154, permute_77)
    add_tensor_15: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_15, primals_123);  mm_default_15 = primals_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_155: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_15, [1, 512, 768]);  add_tensor_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_78: "f32[768, 768]" = torch.ops.aten.permute.default(primals_124, [1, 0]);  primals_124 = None
    
    # No stacktrace found for following nodes
    mm_default_14: "f32[512, 768]" = torch.ops.aten.mm.default(view_154, permute_78)
    add_tensor_14: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_14, primals_125);  mm_default_14 = primals_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_157: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_14, [1, 512, 768]);  add_tensor_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_158: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_157, [1, 512, 12, 64]);  view_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_79: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_158, [0, 2, 1, 3]);  view_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_80: "f32[768, 768]" = torch.ops.aten.permute.default(primals_126, [1, 0]);  primals_126 = None
    
    # No stacktrace found for following nodes
    mm_default_13: "f32[512, 768]" = torch.ops.aten.mm.default(view_154, permute_80)
    add_tensor_13: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_13, primals_127);  mm_default_13 = primals_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_160: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_13, [1, 512, 768]);  add_tensor_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_161: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_160, [1, 512, 12, 64]);  view_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_81: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_161, [0, 2, 1, 3]);  view_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_162: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_155, [1, 512, 12, 64]);  view_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_82: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
    
    # No stacktrace found for following nodes
    clone_default_12: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
    clone_default_13: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None
    clone_default_14: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_81, memory_format = torch.contiguous_format);  permute_81 = None
    _scaled_dot_product_efficient_attention_default_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_12, clone_default_13, clone_default_14, None, True, 0.1, scale = 0.125)
    getitem_154: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default_4[0]
    getitem_155: "f32[1, 12, 512]" = _scaled_dot_product_efficient_attention_default_4[1]
    getitem_156: "i64[]" = _scaled_dot_product_efficient_attention_default_4[2]
    getitem_157: "i64[]" = _scaled_dot_product_efficient_attention_default_4[3];  _scaled_dot_product_efficient_attention_default_4 = None
    alias_default_8: "f32[1, 12, 512, 64]" = torch.ops.aten.alias.default(getitem_154)
    alias_default_9: "f32[1, 12, 512, 64]" = torch.ops.aten.alias.default(alias_default_8);  alias_default_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_84: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_154, [0, 2, 1, 3]);  getitem_154 = None
    clone_7: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_169: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_7, [1, 512, 768]);  clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_170: "f32[512, 768]" = torch.ops.aten.reshape.default(view_169, [512, 768]);  view_169 = None
    permute_85: "f32[768, 768]" = torch.ops.aten.permute.default(primals_128, [1, 0]);  primals_128 = None
    addmm_45: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_129, view_170, permute_85);  primals_129 = None
    view_171: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_45, [1, 512, 768]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    native_dropout_23 = torch.ops.aten.native_dropout.default(view_171, 0.1, True);  view_171 = None
    getitem_76: "f32[1, 512, 768]" = native_dropout_23[0]
    getitem_77: "b8[1, 512, 768]" = native_dropout_23[1];  native_dropout_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_67: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_76, add_65);  getitem_76 = add_65 = None
    var_mean_15 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
    getitem_78: "f32[1, 512, 1]" = var_mean_15[0]
    getitem_79: "f32[1, 512, 1]" = var_mean_15[1];  var_mean_15 = None
    add_68: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-12);  getitem_78 = None
    rsqrt_15: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
    sub_26: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_67, getitem_79);  add_67 = getitem_79 = None
    mul_52: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_15);  sub_26 = None
    mul_53: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_52, primals_130)
    add_69: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_53, primals_131);  mul_53 = primals_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_172: "f32[512, 768]" = torch.ops.aten.reshape.default(add_69, [512, 768])
    permute_86: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_132, [1, 0]);  primals_132 = None
    addmm_46: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_133, view_172, permute_86);  primals_133 = None
    view_173: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_46, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_54: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.5)
    mul_55: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476);  view_173 = None
    erf_7: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
    add_70: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_56: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_54, add_70);  mul_54 = add_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_174: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_56, [512, 3072]);  mul_56 = None
    permute_87: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_134, [1, 0]);  primals_134 = None
    addmm_47: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_135, view_174, permute_87);  primals_135 = None
    view_175: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_47, [1, 512, 768]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    native_dropout_24 = torch.ops.aten.native_dropout.default(view_175, 0.1, True);  view_175 = None
    getitem_80: "f32[1, 512, 768]" = native_dropout_24[0]
    getitem_81: "b8[1, 512, 768]" = native_dropout_24[1];  native_dropout_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_71: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_80, add_69);  getitem_80 = add_69 = None
    var_mean_16 = torch.ops.aten.var_mean.correction(add_71, [2], correction = 0, keepdim = True)
    getitem_82: "f32[1, 512, 1]" = var_mean_16[0]
    getitem_83: "f32[1, 512, 1]" = var_mean_16[1];  var_mean_16 = None
    add_72: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-12);  getitem_82 = None
    rsqrt_16: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
    sub_27: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_71, getitem_83);  add_71 = getitem_83 = None
    mul_57: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_16);  sub_27 = None
    mul_58: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_57, primals_136)
    add_73: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_58, primals_137);  mul_58 = primals_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_176: "f32[512, 768]" = torch.ops.aten.reshape.default(add_73, [512, 768])
    permute_88: "f32[768, 768]" = torch.ops.aten.permute.default(primals_138, [1, 0]);  primals_138 = None
    
    # No stacktrace found for following nodes
    mm_default_12: "f32[512, 768]" = torch.ops.aten.mm.default(view_176, permute_88)
    add_tensor_12: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_12, primals_139);  mm_default_12 = primals_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_177: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_12, [1, 512, 768]);  add_tensor_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_89: "f32[768, 768]" = torch.ops.aten.permute.default(primals_140, [1, 0]);  primals_140 = None
    
    # No stacktrace found for following nodes
    mm_default_11: "f32[512, 768]" = torch.ops.aten.mm.default(view_176, permute_89)
    add_tensor_11: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_11, primals_141);  mm_default_11 = primals_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_179: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_11, [1, 512, 768]);  add_tensor_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_180: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_179, [1, 512, 12, 64]);  view_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_90: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_180, [0, 2, 1, 3]);  view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_91: "f32[768, 768]" = torch.ops.aten.permute.default(primals_142, [1, 0]);  primals_142 = None
    
    # No stacktrace found for following nodes
    mm_default_10: "f32[512, 768]" = torch.ops.aten.mm.default(view_176, permute_91)
    add_tensor_10: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_10, primals_143);  mm_default_10 = primals_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_182: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_10, [1, 512, 768]);  add_tensor_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_183: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_182, [1, 512, 12, 64]);  view_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_92: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_184: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_177, [1, 512, 12, 64]);  view_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_93: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
    
    # No stacktrace found for following nodes
    clone_default_9: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
    clone_default_10: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None
    clone_default_11: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
    _scaled_dot_product_efficient_attention_default_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_9, clone_default_10, clone_default_11, None, True, 0.1, scale = 0.125)
    getitem_147: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default_3[0]
    getitem_148: "f32[1, 12, 512]" = _scaled_dot_product_efficient_attention_default_3[1]
    getitem_149: "i64[]" = _scaled_dot_product_efficient_attention_default_3[2]
    getitem_150: "i64[]" = _scaled_dot_product_efficient_attention_default_3[3];  _scaled_dot_product_efficient_attention_default_3 = None
    alias_default_6: "f32[1, 12, 512, 64]" = torch.ops.aten.alias.default(getitem_147)
    alias_default_7: "f32[1, 12, 512, 64]" = torch.ops.aten.alias.default(alias_default_6);  alias_default_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_95: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_147, [0, 2, 1, 3]);  getitem_147 = None
    clone_8: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_191: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_8, [1, 512, 768]);  clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_192: "f32[512, 768]" = torch.ops.aten.reshape.default(view_191, [512, 768]);  view_191 = None
    permute_96: "f32[768, 768]" = torch.ops.aten.permute.default(primals_144, [1, 0]);  primals_144 = None
    addmm_51: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_145, view_192, permute_96);  primals_145 = None
    view_193: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_51, [1, 512, 768]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    native_dropout_26 = torch.ops.aten.native_dropout.default(view_193, 0.1, True);  view_193 = None
    getitem_86: "f32[1, 512, 768]" = native_dropout_26[0]
    getitem_87: "b8[1, 512, 768]" = native_dropout_26[1];  native_dropout_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_75: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_86, add_73);  getitem_86 = add_73 = None
    var_mean_17 = torch.ops.aten.var_mean.correction(add_75, [2], correction = 0, keepdim = True)
    getitem_88: "f32[1, 512, 1]" = var_mean_17[0]
    getitem_89: "f32[1, 512, 1]" = var_mean_17[1];  var_mean_17 = None
    add_76: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-12);  getitem_88 = None
    rsqrt_17: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
    sub_29: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_75, getitem_89);  add_75 = getitem_89 = None
    mul_59: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_17);  sub_29 = None
    mul_60: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_59, primals_146)
    add_77: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_60, primals_147);  mul_60 = primals_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_194: "f32[512, 768]" = torch.ops.aten.reshape.default(add_77, [512, 768])
    permute_97: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_148, [1, 0]);  primals_148 = None
    addmm_52: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_149, view_194, permute_97);  primals_149 = None
    view_195: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_52, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_61: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, 0.5)
    mul_62: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476);  view_195 = None
    erf_8: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_62);  mul_62 = None
    add_78: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_63: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_61, add_78);  mul_61 = add_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_196: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_63, [512, 3072]);  mul_63 = None
    permute_98: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_150, [1, 0]);  primals_150 = None
    addmm_53: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_151, view_196, permute_98);  primals_151 = None
    view_197: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_53, [1, 512, 768]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    native_dropout_27 = torch.ops.aten.native_dropout.default(view_197, 0.1, True);  view_197 = None
    getitem_90: "f32[1, 512, 768]" = native_dropout_27[0]
    getitem_91: "b8[1, 512, 768]" = native_dropout_27[1];  native_dropout_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_79: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_90, add_77);  getitem_90 = add_77 = None
    var_mean_18 = torch.ops.aten.var_mean.correction(add_79, [2], correction = 0, keepdim = True)
    getitem_92: "f32[1, 512, 1]" = var_mean_18[0]
    getitem_93: "f32[1, 512, 1]" = var_mean_18[1];  var_mean_18 = None
    add_80: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-12);  getitem_92 = None
    rsqrt_18: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
    sub_30: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_79, getitem_93);  add_79 = getitem_93 = None
    mul_64: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_18);  sub_30 = None
    mul_65: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_64, primals_152)
    add_81: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_65, primals_153);  mul_65 = primals_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_198: "f32[512, 768]" = torch.ops.aten.reshape.default(add_81, [512, 768])
    permute_99: "f32[768, 768]" = torch.ops.aten.permute.default(primals_154, [1, 0]);  primals_154 = None
    
    # No stacktrace found for following nodes
    mm_default_9: "f32[512, 768]" = torch.ops.aten.mm.default(view_198, permute_99)
    add_tensor_9: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_9, primals_155);  mm_default_9 = primals_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_199: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_9, [1, 512, 768]);  add_tensor_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_100: "f32[768, 768]" = torch.ops.aten.permute.default(primals_156, [1, 0]);  primals_156 = None
    
    # No stacktrace found for following nodes
    mm_default_8: "f32[512, 768]" = torch.ops.aten.mm.default(view_198, permute_100)
    add_tensor_8: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_8, primals_157);  mm_default_8 = primals_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_201: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_8, [1, 512, 768]);  add_tensor_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_202: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_201, [1, 512, 12, 64]);  view_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_101: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_202, [0, 2, 1, 3]);  view_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_102: "f32[768, 768]" = torch.ops.aten.permute.default(primals_158, [1, 0]);  primals_158 = None
    
    # No stacktrace found for following nodes
    mm_default_7: "f32[512, 768]" = torch.ops.aten.mm.default(view_198, permute_102)
    add_tensor_7: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_7, primals_159);  mm_default_7 = primals_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_204: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_7, [1, 512, 768]);  add_tensor_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_205: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_204, [1, 512, 12, 64]);  view_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_103: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_205, [0, 2, 1, 3]);  view_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_206: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_199, [1, 512, 12, 64]);  view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_104: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
    
    # No stacktrace found for following nodes
    clone_default_6: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
    clone_default_7: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_101, memory_format = torch.contiguous_format);  permute_101 = None
    clone_default_8: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_103, memory_format = torch.contiguous_format);  permute_103 = None
    _scaled_dot_product_efficient_attention_default_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_6, clone_default_7, clone_default_8, None, True, 0.1, scale = 0.125)
    getitem_140: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default_2[0]
    getitem_141: "f32[1, 12, 512]" = _scaled_dot_product_efficient_attention_default_2[1]
    getitem_142: "i64[]" = _scaled_dot_product_efficient_attention_default_2[2]
    getitem_143: "i64[]" = _scaled_dot_product_efficient_attention_default_2[3];  _scaled_dot_product_efficient_attention_default_2 = None
    alias_default_4: "f32[1, 12, 512, 64]" = torch.ops.aten.alias.default(getitem_140)
    alias_default_5: "f32[1, 12, 512, 64]" = torch.ops.aten.alias.default(alias_default_4);  alias_default_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_106: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_140, [0, 2, 1, 3]);  getitem_140 = None
    clone_9: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_213: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_9, [1, 512, 768]);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_214: "f32[512, 768]" = torch.ops.aten.reshape.default(view_213, [512, 768]);  view_213 = None
    permute_107: "f32[768, 768]" = torch.ops.aten.permute.default(primals_160, [1, 0]);  primals_160 = None
    addmm_57: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_161, view_214, permute_107);  primals_161 = None
    view_215: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_57, [1, 512, 768]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    native_dropout_29 = torch.ops.aten.native_dropout.default(view_215, 0.1, True);  view_215 = None
    getitem_96: "f32[1, 512, 768]" = native_dropout_29[0]
    getitem_97: "b8[1, 512, 768]" = native_dropout_29[1];  native_dropout_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_83: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_96, add_81);  getitem_96 = add_81 = None
    var_mean_19 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
    getitem_98: "f32[1, 512, 1]" = var_mean_19[0]
    getitem_99: "f32[1, 512, 1]" = var_mean_19[1];  var_mean_19 = None
    add_84: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-12);  getitem_98 = None
    rsqrt_19: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
    sub_32: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_83, getitem_99);  add_83 = getitem_99 = None
    mul_66: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_19);  sub_32 = None
    mul_67: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_66, primals_162)
    add_85: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_67, primals_163);  mul_67 = primals_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_216: "f32[512, 768]" = torch.ops.aten.reshape.default(add_85, [512, 768])
    permute_108: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_164, [1, 0]);  primals_164 = None
    addmm_58: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_165, view_216, permute_108);  primals_165 = None
    view_217: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_58, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_68: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, 0.5)
    mul_69: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476);  view_217 = None
    erf_9: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
    add_86: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_70: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_68, add_86);  mul_68 = add_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_218: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_70, [512, 3072]);  mul_70 = None
    permute_109: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_166, [1, 0]);  primals_166 = None
    addmm_59: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_167, view_218, permute_109);  primals_167 = None
    view_219: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_59, [1, 512, 768]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    native_dropout_30 = torch.ops.aten.native_dropout.default(view_219, 0.1, True);  view_219 = None
    getitem_100: "f32[1, 512, 768]" = native_dropout_30[0]
    getitem_101: "b8[1, 512, 768]" = native_dropout_30[1];  native_dropout_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_87: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_100, add_85);  getitem_100 = add_85 = None
    var_mean_20 = torch.ops.aten.var_mean.correction(add_87, [2], correction = 0, keepdim = True)
    getitem_102: "f32[1, 512, 1]" = var_mean_20[0]
    getitem_103: "f32[1, 512, 1]" = var_mean_20[1];  var_mean_20 = None
    add_88: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-12);  getitem_102 = None
    rsqrt_20: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
    sub_33: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_87, getitem_103);  add_87 = getitem_103 = None
    mul_71: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_20);  sub_33 = None
    mul_72: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_71, primals_168)
    add_89: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_72, primals_169);  mul_72 = primals_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_220: "f32[512, 768]" = torch.ops.aten.reshape.default(add_89, [512, 768])
    permute_110: "f32[768, 768]" = torch.ops.aten.permute.default(primals_170, [1, 0]);  primals_170 = None
    
    # No stacktrace found for following nodes
    mm_default_6: "f32[512, 768]" = torch.ops.aten.mm.default(view_220, permute_110)
    add_tensor_6: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_6, primals_171);  mm_default_6 = primals_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_221: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_6, [1, 512, 768]);  add_tensor_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_111: "f32[768, 768]" = torch.ops.aten.permute.default(primals_172, [1, 0]);  primals_172 = None
    
    # No stacktrace found for following nodes
    mm_default_5: "f32[512, 768]" = torch.ops.aten.mm.default(view_220, permute_111)
    add_tensor_5: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_5, primals_173);  mm_default_5 = primals_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_223: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_5, [1, 512, 768]);  add_tensor_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_224: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_223, [1, 512, 12, 64]);  view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_112: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_113: "f32[768, 768]" = torch.ops.aten.permute.default(primals_174, [1, 0]);  primals_174 = None
    
    # No stacktrace found for following nodes
    mm_default_4: "f32[512, 768]" = torch.ops.aten.mm.default(view_220, permute_113)
    add_tensor_4: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_4, primals_175);  mm_default_4 = primals_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_226: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_4, [1, 512, 768]);  add_tensor_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_227: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_226, [1, 512, 12, 64]);  view_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_114: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_227, [0, 2, 1, 3]);  view_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_228: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_221, [1, 512, 12, 64]);  view_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_115: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None
    
    # No stacktrace found for following nodes
    clone_default_3: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
    clone_default_4: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_112, memory_format = torch.contiguous_format);  permute_112 = None
    clone_default_5: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
    _scaled_dot_product_efficient_attention_default_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_3, clone_default_4, clone_default_5, None, True, 0.1, scale = 0.125)
    getitem_133: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default_1[0]
    getitem_134: "f32[1, 12, 512]" = _scaled_dot_product_efficient_attention_default_1[1]
    getitem_135: "i64[]" = _scaled_dot_product_efficient_attention_default_1[2]
    getitem_136: "i64[]" = _scaled_dot_product_efficient_attention_default_1[3];  _scaled_dot_product_efficient_attention_default_1 = None
    alias_default_2: "f32[1, 12, 512, 64]" = torch.ops.aten.alias.default(getitem_133)
    alias_default_3: "f32[1, 12, 512, 64]" = torch.ops.aten.alias.default(alias_default_2);  alias_default_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_117: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_133, [0, 2, 1, 3]);  getitem_133 = None
    clone_10: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_235: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_10, [1, 512, 768]);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_236: "f32[512, 768]" = torch.ops.aten.reshape.default(view_235, [512, 768]);  view_235 = None
    permute_118: "f32[768, 768]" = torch.ops.aten.permute.default(primals_176, [1, 0]);  primals_176 = None
    addmm_63: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_177, view_236, permute_118);  primals_177 = None
    view_237: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_63, [1, 512, 768]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    native_dropout_32 = torch.ops.aten.native_dropout.default(view_237, 0.1, True);  view_237 = None
    getitem_106: "f32[1, 512, 768]" = native_dropout_32[0]
    getitem_107: "b8[1, 512, 768]" = native_dropout_32[1];  native_dropout_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_91: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_106, add_89);  getitem_106 = add_89 = None
    var_mean_21 = torch.ops.aten.var_mean.correction(add_91, [2], correction = 0, keepdim = True)
    getitem_108: "f32[1, 512, 1]" = var_mean_21[0]
    getitem_109: "f32[1, 512, 1]" = var_mean_21[1];  var_mean_21 = None
    add_92: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-12);  getitem_108 = None
    rsqrt_21: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
    sub_35: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_91, getitem_109);  add_91 = getitem_109 = None
    mul_73: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_21);  sub_35 = None
    mul_74: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_73, primals_178)
    add_93: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_74, primals_179);  mul_74 = primals_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_238: "f32[512, 768]" = torch.ops.aten.reshape.default(add_93, [512, 768])
    permute_119: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_180, [1, 0]);  primals_180 = None
    addmm_64: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_181, view_238, permute_119);  primals_181 = None
    view_239: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_64, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_75: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, 0.5)
    mul_76: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, 0.7071067811865476);  view_239 = None
    erf_10: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_76);  mul_76 = None
    add_94: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_77: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_75, add_94);  mul_75 = add_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_240: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_77, [512, 3072]);  mul_77 = None
    permute_120: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_182, [1, 0]);  primals_182 = None
    addmm_65: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_183, view_240, permute_120);  primals_183 = None
    view_241: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_65, [1, 512, 768]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    native_dropout_33 = torch.ops.aten.native_dropout.default(view_241, 0.1, True);  view_241 = None
    getitem_110: "f32[1, 512, 768]" = native_dropout_33[0]
    getitem_111: "b8[1, 512, 768]" = native_dropout_33[1];  native_dropout_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_95: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_110, add_93);  getitem_110 = add_93 = None
    var_mean_22 = torch.ops.aten.var_mean.correction(add_95, [2], correction = 0, keepdim = True)
    getitem_112: "f32[1, 512, 1]" = var_mean_22[0]
    getitem_113: "f32[1, 512, 1]" = var_mean_22[1];  var_mean_22 = None
    add_96: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-12);  getitem_112 = None
    rsqrt_22: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
    sub_36: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_95, getitem_113);  add_95 = getitem_113 = None
    mul_78: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_22);  sub_36 = None
    mul_79: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_78, primals_184)
    add_97: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_79, primals_185);  mul_79 = primals_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_242: "f32[512, 768]" = torch.ops.aten.reshape.default(add_97, [512, 768])
    permute_121: "f32[768, 768]" = torch.ops.aten.permute.default(primals_186, [1, 0]);  primals_186 = None
    
    # No stacktrace found for following nodes
    mm_default_3: "f32[512, 768]" = torch.ops.aten.mm.default(view_242, permute_121)
    add_tensor_3: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_3, primals_187);  mm_default_3 = primals_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_243: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_3, [1, 512, 768]);  add_tensor_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_122: "f32[768, 768]" = torch.ops.aten.permute.default(primals_188, [1, 0]);  primals_188 = None
    
    # No stacktrace found for following nodes
    mm_default_2: "f32[512, 768]" = torch.ops.aten.mm.default(view_242, permute_122)
    add_tensor_2: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_2, primals_189);  mm_default_2 = primals_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_245: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_2, [1, 512, 768]);  add_tensor_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_246: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_245, [1, 512, 12, 64]);  view_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_123: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_124: "f32[768, 768]" = torch.ops.aten.permute.default(primals_190, [1, 0]);  primals_190 = None
    
    # No stacktrace found for following nodes
    mm_default_1: "f32[512, 768]" = torch.ops.aten.mm.default(view_242, permute_124)
    add_tensor_1: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_1, primals_191);  mm_default_1 = primals_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_248: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_1, [1, 512, 768]);  add_tensor_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_249: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_248, [1, 512, 12, 64]);  view_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_125: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_249, [0, 2, 1, 3]);  view_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_250: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_243, [1, 512, 12, 64]);  view_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_126: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None
    
    # No stacktrace found for following nodes
    clone_default: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
    clone_default_1: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_123, memory_format = torch.contiguous_format);  permute_123 = None
    clone_default_2: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
    _scaled_dot_product_efficient_attention_default = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default, clone_default_1, clone_default_2, None, True, 0.1, scale = 0.125)
    getitem_126: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_default[0]
    getitem_127: "f32[1, 12, 512]" = _scaled_dot_product_efficient_attention_default[1]
    getitem_128: "i64[]" = _scaled_dot_product_efficient_attention_default[2]
    getitem_129: "i64[]" = _scaled_dot_product_efficient_attention_default[3];  _scaled_dot_product_efficient_attention_default = None
    alias_default: "f32[1, 12, 512, 64]" = torch.ops.aten.alias.default(getitem_126)
    alias_default_1: "f32[1, 12, 512, 64]" = torch.ops.aten.alias.default(alias_default);  alias_default = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_128: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3]);  getitem_126 = None
    clone_11: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_257: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_11, [1, 512, 768]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_258: "f32[512, 768]" = torch.ops.aten.reshape.default(view_257, [512, 768]);  view_257 = None
    permute_129: "f32[768, 768]" = torch.ops.aten.permute.default(primals_192, [1, 0]);  primals_192 = None
    addmm_69: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_193, view_258, permute_129);  primals_193 = None
    view_259: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_69, [1, 512, 768]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    native_dropout_35 = torch.ops.aten.native_dropout.default(view_259, 0.1, True);  view_259 = None
    getitem_116: "f32[1, 512, 768]" = native_dropout_35[0]
    getitem_117: "b8[1, 512, 768]" = native_dropout_35[1];  native_dropout_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_99: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_116, add_97);  getitem_116 = add_97 = None
    var_mean_23 = torch.ops.aten.var_mean.correction(add_99, [2], correction = 0, keepdim = True)
    getitem_118: "f32[1, 512, 1]" = var_mean_23[0]
    getitem_119: "f32[1, 512, 1]" = var_mean_23[1];  var_mean_23 = None
    add_100: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-12);  getitem_118 = None
    rsqrt_23: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
    sub_38: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_99, getitem_119);  add_99 = getitem_119 = None
    mul_80: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_23);  sub_38 = None
    mul_81: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_80, primals_194)
    add_101: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_81, primals_195);  mul_81 = primals_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_260: "f32[512, 768]" = torch.ops.aten.reshape.default(add_101, [512, 768])
    permute_130: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_196, [1, 0]);  primals_196 = None
    addmm_70: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_197, view_260, permute_130);  primals_197 = None
    view_261: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_70, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_82: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, 0.5)
    mul_83: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476);  view_261 = None
    erf_11: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
    add_102: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_84: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_82, add_102);  mul_82 = add_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_262: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_84, [512, 3072]);  mul_84 = None
    permute_131: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_198, [1, 0]);  primals_198 = None
    addmm_71: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_199, view_262, permute_131);  primals_199 = None
    view_263: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_71, [1, 512, 768]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    native_dropout_36 = torch.ops.aten.native_dropout.default(view_263, 0.1, True);  view_263 = None
    getitem_120: "f32[1, 512, 768]" = native_dropout_36[0]
    getitem_121: "b8[1, 512, 768]" = native_dropout_36[1];  native_dropout_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_103: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_120, add_101);  getitem_120 = add_101 = None
    var_mean_24 = torch.ops.aten.var_mean.correction(add_103, [2], correction = 0, keepdim = True)
    getitem_122: "f32[1, 512, 1]" = var_mean_24[0]
    getitem_123: "f32[1, 512, 1]" = var_mean_24[1];  var_mean_24 = None
    add_104: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_122, 1e-12);  getitem_122 = None
    rsqrt_24: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
    sub_39: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_103, getitem_123);  add_103 = getitem_123 = None
    mul_85: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_24);  sub_39 = None
    mul_86: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_85, primals_200)
    add_105: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_86, primals_201);  mul_86 = primals_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:558, code: first_token_tensor = hidden_states[:, 0]
    slice_18: "f32[1, 512, 768]" = torch.ops.aten.slice.Tensor(add_105, 0, 0, 9223372036854775807)
    select_8: "f32[1, 768]" = torch.ops.aten.select.int(slice_18, 1, 0);  slice_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:559, code: pooled_output = self.dense(first_token_tensor)
    permute_132: "f32[768, 768]" = torch.ops.aten.permute.default(primals_202, [1, 0]);  primals_202 = None
    
    # No stacktrace found for following nodes
    mm_default: "f32[1, 768]" = torch.ops.aten.mm.default(select_8, permute_132)
    add_tensor: "f32[1, 768]" = torch.ops.aten.add.Tensor(mm_default, primals_203);  mm_default = primals_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:560, code: pooled_output = self.activation(pooled_output)
    tanh: "f32[1, 768]" = torch.ops.aten.tanh.default(add_tensor);  add_tensor = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:1084, code: pooled_output = self.dropout(pooled_output)
    native_dropout_37 = torch.ops.aten.native_dropout.default(tanh, 0.1, True)
    getitem_124: "f32[1, 768]" = native_dropout_37[0]
    getitem_125: "b8[1, 768]" = native_dropout_37[1];  native_dropout_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:1085, code: logits = self.classifier(pooled_output)
    permute_133: "f32[768, 2]" = torch.ops.aten.permute.default(primals_204, [1, 0]);  primals_204 = None
    addmm_73: "f32[1, 2]" = torch.ops.aten.addmm.default(primals_205, getitem_124, permute_133);  primals_205 = None
    permute_134: "f32[2, 768]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:559, code: pooled_output = self.dense(first_token_tensor)
    permute_138: "f32[768, 768]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_24: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    permute_142: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    permute_146: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_25: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    permute_150: "f32[768, 768]" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_162: "f32[768, 768]" = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_167: "f32[768, 768]" = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    permute_171: "f32[768, 768]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_27: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    permute_175: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    permute_179: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_28: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    permute_183: "f32[768, 768]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_195: "f32[768, 768]" = torch.ops.aten.permute.default(permute_113, [1, 0]);  permute_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_200: "f32[768, 768]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    permute_204: "f32[768, 768]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_30: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    permute_208: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    permute_212: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_31: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    permute_216: "f32[768, 768]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_228: "f32[768, 768]" = torch.ops.aten.permute.default(permute_102, [1, 0]);  permute_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_233: "f32[768, 768]" = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    permute_237: "f32[768, 768]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_33: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    permute_241: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    permute_245: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_34: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    permute_249: "f32[768, 768]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_261: "f32[768, 768]" = torch.ops.aten.permute.default(permute_91, [1, 0]);  permute_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_266: "f32[768, 768]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    permute_270: "f32[768, 768]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_36: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    permute_274: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    permute_278: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_37: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    permute_282: "f32[768, 768]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_294: "f32[768, 768]" = torch.ops.aten.permute.default(permute_80, [1, 0]);  permute_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_299: "f32[768, 768]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    permute_303: "f32[768, 768]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_39: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    permute_307: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    permute_311: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_40: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    permute_315: "f32[768, 768]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_327: "f32[768, 768]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_332: "f32[768, 768]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    permute_336: "f32[768, 768]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_42: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    permute_340: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    permute_344: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_43: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    permute_348: "f32[768, 768]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_360: "f32[768, 768]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_365: "f32[768, 768]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    permute_369: "f32[768, 768]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_45: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    permute_373: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    permute_377: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_46: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    permute_381: "f32[768, 768]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_393: "f32[768, 768]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_398: "f32[768, 768]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    permute_402: "f32[768, 768]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_48: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    permute_406: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    permute_410: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_49: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    permute_414: "f32[768, 768]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_426: "f32[768, 768]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_431: "f32[768, 768]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    permute_435: "f32[768, 768]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_51: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    permute_439: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    permute_443: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_52: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    permute_447: "f32[768, 768]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_459: "f32[768, 768]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_464: "f32[768, 768]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    permute_468: "f32[768, 768]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_54: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    permute_472: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    permute_476: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_55: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    permute_480: "f32[768, 768]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_492: "f32[768, 768]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_497: "f32[768, 768]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    permute_501: "f32[768, 768]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_57: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    permute_505: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    permute_509: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_58: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    permute_513: "f32[768, 768]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_525: "f32[768, 768]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_530: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    permute_534: "f32[768, 768]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:126, code: embeddings = self.LayerNorm(embeddings)
    div_60: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
    return [add_105, tanh, addmm_73, primals_8, primals_18, primals_24, primals_34, primals_40, primals_50, primals_56, primals_66, primals_72, primals_82, primals_88, primals_98, primals_104, primals_114, primals_120, primals_130, primals_136, primals_146, primals_152, primals_162, primals_168, primals_178, primals_184, primals_194, primals_200, primals_207, full_default, slice_1, select, select_1, select_2, select_3, mul_1, getitem_3, view, clone_default_33, clone_default_34, clone_default_35, getitem_204, getitem_205, getitem_206, alias_default_23, view_16, getitem_7, mul_3, view_18, addmm_4, view_20, getitem_11, mul_8, view_22, clone_default_30, clone_default_31, clone_default_32, getitem_197, getitem_198, getitem_199, alias_default_21, view_38, getitem_17, mul_10, view_40, addmm_10, view_42, getitem_21, mul_15, view_44, clone_default_27, clone_default_28, clone_default_29, getitem_190, getitem_191, getitem_192, alias_default_19, view_60, getitem_27, mul_17, view_62, addmm_16, view_64, getitem_31, mul_22, view_66, clone_default_24, clone_default_25, clone_default_26, getitem_183, getitem_184, getitem_185, alias_default_17, view_82, getitem_37, mul_24, view_84, addmm_22, view_86, getitem_41, mul_29, view_88, clone_default_21, clone_default_22, clone_default_23, getitem_176, getitem_177, getitem_178, alias_default_15, view_104, getitem_47, mul_31, view_106, addmm_28, view_108, getitem_51, mul_36, view_110, clone_default_18, clone_default_19, clone_default_20, getitem_169, getitem_170, getitem_171, alias_default_13, view_126, getitem_57, mul_38, view_128, addmm_34, view_130, getitem_61, mul_43, view_132, clone_default_15, clone_default_16, clone_default_17, getitem_162, getitem_163, getitem_164, alias_default_11, view_148, getitem_67, mul_45, view_150, addmm_40, view_152, getitem_71, mul_50, view_154, clone_default_12, clone_default_13, clone_default_14, getitem_155, getitem_156, getitem_157, alias_default_9, view_170, getitem_77, mul_52, view_172, addmm_46, view_174, getitem_81, mul_57, view_176, clone_default_9, clone_default_10, clone_default_11, getitem_148, getitem_149, getitem_150, alias_default_7, view_192, getitem_87, mul_59, view_194, addmm_52, view_196, getitem_91, mul_64, view_198, clone_default_6, clone_default_7, clone_default_8, getitem_141, getitem_142, getitem_143, alias_default_5, view_214, getitem_97, mul_66, view_216, addmm_58, view_218, getitem_101, mul_71, view_220, clone_default_3, clone_default_4, clone_default_5, getitem_134, getitem_135, getitem_136, alias_default_3, view_236, getitem_107, mul_73, view_238, addmm_64, view_240, getitem_111, mul_78, view_242, clone_default, clone_default_1, clone_default_2, getitem_127, getitem_128, getitem_129, alias_default_1, view_258, getitem_117, mul_80, view_260, addmm_70, view_262, getitem_121, mul_85, select_8, tanh, getitem_124, getitem_125, permute_134, permute_138, div_24, permute_142, permute_146, div_25, permute_150, permute_162, permute_167, permute_171, div_27, permute_175, permute_179, div_28, permute_183, permute_195, permute_200, permute_204, div_30, permute_208, permute_212, div_31, permute_216, permute_228, permute_233, permute_237, div_33, permute_241, permute_245, div_34, permute_249, permute_261, permute_266, permute_270, div_36, permute_274, permute_278, div_37, permute_282, permute_294, permute_299, permute_303, div_39, permute_307, permute_311, div_40, permute_315, permute_327, permute_332, permute_336, div_42, permute_340, permute_344, div_43, permute_348, permute_360, permute_365, permute_369, div_45, permute_373, permute_377, div_46, permute_381, permute_393, permute_398, permute_402, div_48, permute_406, permute_410, div_49, permute_414, permute_426, permute_431, permute_435, div_51, permute_439, permute_443, div_52, permute_447, permute_459, permute_464, permute_468, div_54, permute_472, permute_476, div_55, permute_480, permute_492, permute_497, permute_501, div_57, permute_505, permute_509, div_58, permute_513, permute_525, permute_530, permute_534, div_60]
    