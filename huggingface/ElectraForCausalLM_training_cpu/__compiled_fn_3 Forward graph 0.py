from __future__ import annotations



def forward(self, primals_1: "f32[30522, 128]", primals_2: "f32[2, 128]", primals_3: "f32[512, 128]", primals_4: "f32[128]", primals_5: "f32[128]", primals_6: "f32[256, 128]", primals_7: "f32[256]", primals_8: "f32[256, 256]", primals_9: "f32[256]", primals_10: "f32[256, 256]", primals_11: "f32[256]", primals_12: "f32[256, 256]", primals_13: "f32[256]", primals_14: "f32[256, 256]", primals_15: "f32[256]", primals_16: "f32[256]", primals_17: "f32[256]", primals_18: "f32[1024, 256]", primals_19: "f32[1024]", primals_20: "f32[256, 1024]", primals_21: "f32[256]", primals_22: "f32[256]", primals_23: "f32[256]", primals_24: "f32[256, 256]", primals_25: "f32[256]", primals_26: "f32[256, 256]", primals_27: "f32[256]", primals_28: "f32[256, 256]", primals_29: "f32[256]", primals_30: "f32[256, 256]", primals_31: "f32[256]", primals_32: "f32[256]", primals_33: "f32[256]", primals_34: "f32[1024, 256]", primals_35: "f32[1024]", primals_36: "f32[256, 1024]", primals_37: "f32[256]", primals_38: "f32[256]", primals_39: "f32[256]", primals_40: "f32[256, 256]", primals_41: "f32[256]", primals_42: "f32[256, 256]", primals_43: "f32[256]", primals_44: "f32[256, 256]", primals_45: "f32[256]", primals_46: "f32[256, 256]", primals_47: "f32[256]", primals_48: "f32[256]", primals_49: "f32[256]", primals_50: "f32[1024, 256]", primals_51: "f32[1024]", primals_52: "f32[256, 1024]", primals_53: "f32[256]", primals_54: "f32[256]", primals_55: "f32[256]", primals_56: "f32[256, 256]", primals_57: "f32[256]", primals_58: "f32[256, 256]", primals_59: "f32[256]", primals_60: "f32[256, 256]", primals_61: "f32[256]", primals_62: "f32[256, 256]", primals_63: "f32[256]", primals_64: "f32[256]", primals_65: "f32[256]", primals_66: "f32[1024, 256]", primals_67: "f32[1024]", primals_68: "f32[256, 1024]", primals_69: "f32[256]", primals_70: "f32[256]", primals_71: "f32[256]", primals_72: "f32[256, 256]", primals_73: "f32[256]", primals_74: "f32[256, 256]", primals_75: "f32[256]", primals_76: "f32[256, 256]", primals_77: "f32[256]", primals_78: "f32[256, 256]", primals_79: "f32[256]", primals_80: "f32[256]", primals_81: "f32[256]", primals_82: "f32[1024, 256]", primals_83: "f32[1024]", primals_84: "f32[256, 1024]", primals_85: "f32[256]", primals_86: "f32[256]", primals_87: "f32[256]", primals_88: "f32[256, 256]", primals_89: "f32[256]", primals_90: "f32[256, 256]", primals_91: "f32[256]", primals_92: "f32[256, 256]", primals_93: "f32[256]", primals_94: "f32[256, 256]", primals_95: "f32[256]", primals_96: "f32[256]", primals_97: "f32[256]", primals_98: "f32[1024, 256]", primals_99: "f32[1024]", primals_100: "f32[256, 1024]", primals_101: "f32[256]", primals_102: "f32[256]", primals_103: "f32[256]", primals_104: "f32[256, 256]", primals_105: "f32[256]", primals_106: "f32[256, 256]", primals_107: "f32[256]", primals_108: "f32[256, 256]", primals_109: "f32[256]", primals_110: "f32[256, 256]", primals_111: "f32[256]", primals_112: "f32[256]", primals_113: "f32[256]", primals_114: "f32[1024, 256]", primals_115: "f32[1024]", primals_116: "f32[256, 1024]", primals_117: "f32[256]", primals_118: "f32[256]", primals_119: "f32[256]", primals_120: "f32[256, 256]", primals_121: "f32[256]", primals_122: "f32[256, 256]", primals_123: "f32[256]", primals_124: "f32[256, 256]", primals_125: "f32[256]", primals_126: "f32[256, 256]", primals_127: "f32[256]", primals_128: "f32[256]", primals_129: "f32[256]", primals_130: "f32[1024, 256]", primals_131: "f32[1024]", primals_132: "f32[256, 1024]", primals_133: "f32[256]", primals_134: "f32[256]", primals_135: "f32[256]", primals_136: "f32[256, 256]", primals_137: "f32[256]", primals_138: "f32[256, 256]", primals_139: "f32[256]", primals_140: "f32[256, 256]", primals_141: "f32[256]", primals_142: "f32[256, 256]", primals_143: "f32[256]", primals_144: "f32[256]", primals_145: "f32[256]", primals_146: "f32[1024, 256]", primals_147: "f32[1024]", primals_148: "f32[256, 1024]", primals_149: "f32[256]", primals_150: "f32[256]", primals_151: "f32[256]", primals_152: "f32[256, 256]", primals_153: "f32[256]", primals_154: "f32[256, 256]", primals_155: "f32[256]", primals_156: "f32[256, 256]", primals_157: "f32[256]", primals_158: "f32[256, 256]", primals_159: "f32[256]", primals_160: "f32[256]", primals_161: "f32[256]", primals_162: "f32[1024, 256]", primals_163: "f32[1024]", primals_164: "f32[256, 1024]", primals_165: "f32[256]", primals_166: "f32[256]", primals_167: "f32[256]", primals_168: "f32[256, 256]", primals_169: "f32[256]", primals_170: "f32[256, 256]", primals_171: "f32[256]", primals_172: "f32[256, 256]", primals_173: "f32[256]", primals_174: "f32[256, 256]", primals_175: "f32[256]", primals_176: "f32[256]", primals_177: "f32[256]", primals_178: "f32[1024, 256]", primals_179: "f32[1024]", primals_180: "f32[256, 1024]", primals_181: "f32[256]", primals_182: "f32[256]", primals_183: "f32[256]", primals_184: "f32[256, 256]", primals_185: "f32[256]", primals_186: "f32[256, 256]", primals_187: "f32[256]", primals_188: "f32[256, 256]", primals_189: "f32[256]", primals_190: "f32[256, 256]", primals_191: "f32[256]", primals_192: "f32[256]", primals_193: "f32[256]", primals_194: "f32[1024, 256]", primals_195: "f32[1024]", primals_196: "f32[256, 1024]", primals_197: "f32[256]", primals_198: "f32[256]", primals_199: "f32[256]", primals_200: "f32[128, 256]", primals_201: "f32[128]", primals_202: "f32[128]", primals_203: "f32[128]", primals_204: "f32[30522, 128]", primals_205: "f32[30522]", primals_206: "i64[1, 512]", primals_207: "i64[1, 512]", primals_208: "i64[1, 512]", primals_209: "i64[1, 512]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:888, code: buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
    slice_1: "i64[1, 512]" = torch.ops.aten.slice.Tensor(primals_206, 0, 0, 9223372036854775807);  primals_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:889, code: buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
    expand: "i64[1, 512]" = torch.ops.aten.expand.default(slice_1, [1, 512]);  slice_1 = None
    
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
    sub_1: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
    mul_1: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = None
    mul_2: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_1, primals_4)
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
    permute_2: "f32[256, 256]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
    addmm_2: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_11, view_2, permute_2);  primals_11 = None
    view_5: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_2, [1, 512, 256]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_6: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_5, [1, 512, 4, 64]);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_3: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_4: "f32[256, 256]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
    addmm_3: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_13, view_2, permute_4);  primals_13 = None
    view_8: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_3, [1, 512, 256]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_9: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_8, [1, 512, 4, 64]);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_5: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_10: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_3, [1, 512, 4, 64]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_6: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
    
    # No stacktrace found for following nodes
    clone_default_44: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format);  permute_6 = None
    clone_default_45: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None
    clone_default_46: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    mul_scalar_44: "f32[1, 4, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default_44, 0.3535533905932738);  clone_default_44 = None
    permute_default_66: "f32[1, 4, 64, 512]" = torch.ops.aten.permute.default(clone_default_45, [0, 1, 3, 2]);  clone_default_45 = None
    mul_scalar_45: "f32[1, 4, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_66, 0.3535533905932738);  permute_default_66 = None
    expand_default_44: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(mul_scalar_44, [1, 4, 512, 64]);  mul_scalar_44 = None
    view_default_132: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_default_44, [4, 512, 64]);  expand_default_44 = None
    expand_default_45: "f32[1, 4, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_45, [1, 4, 64, 512]);  mul_scalar_45 = None
    view_default_133: "f32[4, 64, 512]" = torch.ops.aten.view.default(expand_default_45, [4, 64, 512]);  expand_default_45 = None
    bmm_default_66: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_default_132, view_default_133)
    view_default_134: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_default_66, [1, 4, 512, 512]);  bmm_default_66 = None
    amax_default_11: "f32[1, 4, 512, 1]" = torch.ops.aten.amax.default(view_default_134, [-1], True)
    sub_tensor_22: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_134, amax_default_11);  view_default_134 = amax_default_11 = None
    exp_default_11: "f32[1, 4, 512, 512]" = torch.ops.aten.exp.default(sub_tensor_22);  sub_tensor_22 = None
    sum_dim_int_list_22: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_11, [-1], True)
    div_tensor_11: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_default_11, sum_dim_int_list_22);  exp_default_11 = sum_dim_int_list_22 = None
    alias_default_22: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(div_tensor_11)
    native_dropout_default_11 = torch.ops.aten.native_dropout.default(div_tensor_11, 0.1, True);  div_tensor_11 = None
    getitem_148: "f32[1, 4, 512, 512]" = native_dropout_default_11[0]
    getitem_149: "b8[1, 4, 512, 512]" = native_dropout_default_11[1];  native_dropout_default_11 = None
    expand_default_46: "f32[1, 4, 512, 512]" = torch.ops.aten.expand.default(getitem_148, [1, 4, 512, 512]);  getitem_148 = None
    view_default_135: "f32[4, 512, 512]" = torch.ops.aten.view.default(expand_default_46, [4, 512, 512]);  expand_default_46 = None
    expand_default_47: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(clone_default_46, [1, 4, 512, 64]);  clone_default_46 = None
    view_default_136: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_default_47, [4, 512, 64]);  expand_default_47 = None
    bmm_default_67: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_default_135, view_default_136)
    view_default_137: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_default_67, [1, 4, 512, 64]);  bmm_default_67 = None
    permute_default_67: "f32[4, 512, 512]" = torch.ops.aten.permute.default(view_default_135, [0, 2, 1]);  view_default_135 = None
    permute_default_68: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_default_136, [0, 2, 1]);  view_default_136 = None
    alias_default_23: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(alias_default_22);  alias_default_22 = None
    permute_default_69: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_default_132, [0, 2, 1]);  view_default_132 = None
    permute_default_70: "f32[4, 512, 64]" = torch.ops.aten.permute.default(view_default_133, [0, 2, 1]);  view_default_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_8: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_default_137, [0, 2, 1, 3]);  view_default_137 = None
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
    sub_3: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_5, getitem_9);  add_5 = getitem_9 = None
    mul_3: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = None
    mul_4: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_3, primals_16)
    add_7: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_4, primals_17);  mul_4 = primals_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_20: "f32[512, 256]" = torch.ops.aten.view.default(add_7, [512, 256])
    permute_10: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
    addmm_5: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_19, view_20, permute_10);  primals_19 = None
    view_21: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_5, [1, 512, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_5: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_21, 0.5)
    mul_6: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_21, 0.7071067811865476);  view_21 = None
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
    sub_4: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_9, getitem_13);  add_9 = getitem_13 = None
    mul_8: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = None
    mul_9: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_8, primals_22)
    add_11: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_9, primals_23);  mul_9 = primals_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_24: "f32[512, 256]" = torch.ops.aten.view.default(add_11, [512, 256])
    permute_12: "f32[256, 256]" = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
    addmm_7: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_25, view_24, permute_12);  primals_25 = None
    view_25: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_7, [1, 512, 256]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_13: "f32[256, 256]" = torch.ops.aten.permute.default(primals_26, [1, 0]);  primals_26 = None
    addmm_8: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_27, view_24, permute_13);  primals_27 = None
    view_27: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_8, [1, 512, 256]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_28: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_27, [1, 512, 4, 64]);  view_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_14: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_15: "f32[256, 256]" = torch.ops.aten.permute.default(primals_28, [1, 0]);  primals_28 = None
    addmm_9: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_29, view_24, permute_15);  primals_29 = None
    view_30: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_9, [1, 512, 256]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_31: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_30, [1, 512, 4, 64]);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_16: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_32: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_25, [1, 512, 4, 64]);  view_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_17: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None
    
    # No stacktrace found for following nodes
    clone_default_40: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_17, memory_format = torch.contiguous_format);  permute_17 = None
    clone_default_41: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
    clone_default_42: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
    mul_scalar_40: "f32[1, 4, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default_40, 0.3535533905932738);  clone_default_40 = None
    permute_default_60: "f32[1, 4, 64, 512]" = torch.ops.aten.permute.default(clone_default_41, [0, 1, 3, 2]);  clone_default_41 = None
    mul_scalar_41: "f32[1, 4, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_60, 0.3535533905932738);  permute_default_60 = None
    expand_default_40: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(mul_scalar_40, [1, 4, 512, 64]);  mul_scalar_40 = None
    view_default_120: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_default_40, [4, 512, 64]);  expand_default_40 = None
    expand_default_41: "f32[1, 4, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_41, [1, 4, 64, 512]);  mul_scalar_41 = None
    view_default_121: "f32[4, 64, 512]" = torch.ops.aten.view.default(expand_default_41, [4, 64, 512]);  expand_default_41 = None
    bmm_default_60: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_default_120, view_default_121)
    view_default_122: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_default_60, [1, 4, 512, 512]);  bmm_default_60 = None
    amax_default_10: "f32[1, 4, 512, 1]" = torch.ops.aten.amax.default(view_default_122, [-1], True)
    sub_tensor_20: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_122, amax_default_10);  view_default_122 = amax_default_10 = None
    exp_default_10: "f32[1, 4, 512, 512]" = torch.ops.aten.exp.default(sub_tensor_20);  sub_tensor_20 = None
    sum_dim_int_list_20: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_10, [-1], True)
    div_tensor_10: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_default_10, sum_dim_int_list_20);  exp_default_10 = sum_dim_int_list_20 = None
    alias_default_20: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(div_tensor_10)
    native_dropout_default_10 = torch.ops.aten.native_dropout.default(div_tensor_10, 0.1, True);  div_tensor_10 = None
    getitem_146: "f32[1, 4, 512, 512]" = native_dropout_default_10[0]
    getitem_147: "b8[1, 4, 512, 512]" = native_dropout_default_10[1];  native_dropout_default_10 = None
    expand_default_42: "f32[1, 4, 512, 512]" = torch.ops.aten.expand.default(getitem_146, [1, 4, 512, 512]);  getitem_146 = None
    view_default_123: "f32[4, 512, 512]" = torch.ops.aten.view.default(expand_default_42, [4, 512, 512]);  expand_default_42 = None
    expand_default_43: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(clone_default_42, [1, 4, 512, 64]);  clone_default_42 = None
    view_default_124: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_default_43, [4, 512, 64]);  expand_default_43 = None
    bmm_default_61: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_default_123, view_default_124)
    view_default_125: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_default_61, [1, 4, 512, 64]);  bmm_default_61 = None
    permute_default_61: "f32[4, 512, 512]" = torch.ops.aten.permute.default(view_default_123, [0, 2, 1]);  view_default_123 = None
    permute_default_62: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_default_124, [0, 2, 1]);  view_default_124 = None
    alias_default_21: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(alias_default_20);  alias_default_20 = None
    permute_default_63: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_default_120, [0, 2, 1]);  view_default_120 = None
    permute_default_64: "f32[4, 512, 64]" = torch.ops.aten.permute.default(view_default_121, [0, 2, 1]);  view_default_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_19: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_default_125, [0, 2, 1, 3]);  view_default_125 = None
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
    sub_6: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_13, getitem_19);  add_13 = getitem_19 = None
    mul_10: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_3);  sub_6 = None
    mul_11: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_10, primals_32)
    add_15: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_11, primals_33);  mul_11 = primals_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_42: "f32[512, 256]" = torch.ops.aten.view.default(add_15, [512, 256])
    permute_21: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
    addmm_11: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_35, view_42, permute_21);  primals_35 = None
    view_43: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_11, [1, 512, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_12: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_43, 0.5)
    mul_13: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_43, 0.7071067811865476);  view_43 = None
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
    sub_7: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_17, getitem_23);  add_17 = getitem_23 = None
    mul_15: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_4);  sub_7 = None
    mul_16: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_15, primals_38)
    add_19: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_16, primals_39);  mul_16 = primals_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_46: "f32[512, 256]" = torch.ops.aten.view.default(add_19, [512, 256])
    permute_23: "f32[256, 256]" = torch.ops.aten.permute.default(primals_40, [1, 0]);  primals_40 = None
    addmm_13: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_41, view_46, permute_23);  primals_41 = None
    view_47: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_13, [1, 512, 256]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_24: "f32[256, 256]" = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
    addmm_14: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_43, view_46, permute_24);  primals_43 = None
    view_49: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_14, [1, 512, 256]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_50: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_49, [1, 512, 4, 64]);  view_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_25: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_26: "f32[256, 256]" = torch.ops.aten.permute.default(primals_44, [1, 0]);  primals_44 = None
    addmm_15: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_45, view_46, permute_26);  primals_45 = None
    view_52: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_15, [1, 512, 256]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_53: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_52, [1, 512, 4, 64]);  view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_27: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_54: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_47, [1, 512, 4, 64]);  view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_28: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
    
    # No stacktrace found for following nodes
    clone_default_36: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_28, memory_format = torch.contiguous_format);  permute_28 = None
    clone_default_37: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_25, memory_format = torch.contiguous_format);  permute_25 = None
    clone_default_38: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
    mul_scalar_36: "f32[1, 4, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default_36, 0.3535533905932738);  clone_default_36 = None
    permute_default_54: "f32[1, 4, 64, 512]" = torch.ops.aten.permute.default(clone_default_37, [0, 1, 3, 2]);  clone_default_37 = None
    mul_scalar_37: "f32[1, 4, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_54, 0.3535533905932738);  permute_default_54 = None
    expand_default_36: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(mul_scalar_36, [1, 4, 512, 64]);  mul_scalar_36 = None
    view_default_108: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_default_36, [4, 512, 64]);  expand_default_36 = None
    expand_default_37: "f32[1, 4, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_37, [1, 4, 64, 512]);  mul_scalar_37 = None
    view_default_109: "f32[4, 64, 512]" = torch.ops.aten.view.default(expand_default_37, [4, 64, 512]);  expand_default_37 = None
    bmm_default_54: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_default_108, view_default_109)
    view_default_110: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_default_54, [1, 4, 512, 512]);  bmm_default_54 = None
    amax_default_9: "f32[1, 4, 512, 1]" = torch.ops.aten.amax.default(view_default_110, [-1], True)
    sub_tensor_18: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_110, amax_default_9);  view_default_110 = amax_default_9 = None
    exp_default_9: "f32[1, 4, 512, 512]" = torch.ops.aten.exp.default(sub_tensor_18);  sub_tensor_18 = None
    sum_dim_int_list_18: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_9, [-1], True)
    div_tensor_9: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_default_9, sum_dim_int_list_18);  exp_default_9 = sum_dim_int_list_18 = None
    alias_default_18: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(div_tensor_9)
    native_dropout_default_9 = torch.ops.aten.native_dropout.default(div_tensor_9, 0.1, True);  div_tensor_9 = None
    getitem_144: "f32[1, 4, 512, 512]" = native_dropout_default_9[0]
    getitem_145: "b8[1, 4, 512, 512]" = native_dropout_default_9[1];  native_dropout_default_9 = None
    expand_default_38: "f32[1, 4, 512, 512]" = torch.ops.aten.expand.default(getitem_144, [1, 4, 512, 512]);  getitem_144 = None
    view_default_111: "f32[4, 512, 512]" = torch.ops.aten.view.default(expand_default_38, [4, 512, 512]);  expand_default_38 = None
    expand_default_39: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(clone_default_38, [1, 4, 512, 64]);  clone_default_38 = None
    view_default_112: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_default_39, [4, 512, 64]);  expand_default_39 = None
    bmm_default_55: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_default_111, view_default_112)
    view_default_113: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_default_55, [1, 4, 512, 64]);  bmm_default_55 = None
    permute_default_55: "f32[4, 512, 512]" = torch.ops.aten.permute.default(view_default_111, [0, 2, 1]);  view_default_111 = None
    permute_default_56: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_default_112, [0, 2, 1]);  view_default_112 = None
    alias_default_19: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(alias_default_18);  alias_default_18 = None
    permute_default_57: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_default_108, [0, 2, 1]);  view_default_108 = None
    permute_default_58: "f32[4, 512, 64]" = torch.ops.aten.permute.default(view_default_109, [0, 2, 1]);  view_default_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_30: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_default_113, [0, 2, 1, 3]);  view_default_113 = None
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
    sub_9: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_21, getitem_29);  add_21 = getitem_29 = None
    mul_17: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_5);  sub_9 = None
    mul_18: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_17, primals_48)
    add_23: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_18, primals_49);  mul_18 = primals_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_64: "f32[512, 256]" = torch.ops.aten.view.default(add_23, [512, 256])
    permute_32: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_50, [1, 0]);  primals_50 = None
    addmm_17: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_51, view_64, permute_32);  primals_51 = None
    view_65: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_17, [1, 512, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_19: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_65, 0.5)
    mul_20: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_65, 0.7071067811865476);  view_65 = None
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
    sub_10: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_25, getitem_33);  add_25 = getitem_33 = None
    mul_22: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_6);  sub_10 = None
    mul_23: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_22, primals_54)
    add_27: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_23, primals_55);  mul_23 = primals_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_68: "f32[512, 256]" = torch.ops.aten.view.default(add_27, [512, 256])
    permute_34: "f32[256, 256]" = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
    addmm_19: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_57, view_68, permute_34);  primals_57 = None
    view_69: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_19, [1, 512, 256]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_35: "f32[256, 256]" = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
    addmm_20: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_59, view_68, permute_35);  primals_59 = None
    view_71: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_20, [1, 512, 256]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_72: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_71, [1, 512, 4, 64]);  view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_36: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_37: "f32[256, 256]" = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
    addmm_21: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_61, view_68, permute_37);  primals_61 = None
    view_74: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_21, [1, 512, 256]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_75: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_74, [1, 512, 4, 64]);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_38: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_76: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_69, [1, 512, 4, 64]);  view_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_39: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
    
    # No stacktrace found for following nodes
    clone_default_32: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_39, memory_format = torch.contiguous_format);  permute_39 = None
    clone_default_33: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_36, memory_format = torch.contiguous_format);  permute_36 = None
    clone_default_34: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
    mul_scalar_32: "f32[1, 4, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default_32, 0.3535533905932738);  clone_default_32 = None
    permute_default_48: "f32[1, 4, 64, 512]" = torch.ops.aten.permute.default(clone_default_33, [0, 1, 3, 2]);  clone_default_33 = None
    mul_scalar_33: "f32[1, 4, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_48, 0.3535533905932738);  permute_default_48 = None
    expand_default_32: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(mul_scalar_32, [1, 4, 512, 64]);  mul_scalar_32 = None
    view_default_96: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_default_32, [4, 512, 64]);  expand_default_32 = None
    expand_default_33: "f32[1, 4, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_33, [1, 4, 64, 512]);  mul_scalar_33 = None
    view_default_97: "f32[4, 64, 512]" = torch.ops.aten.view.default(expand_default_33, [4, 64, 512]);  expand_default_33 = None
    bmm_default_48: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_default_96, view_default_97)
    view_default_98: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_default_48, [1, 4, 512, 512]);  bmm_default_48 = None
    amax_default_8: "f32[1, 4, 512, 1]" = torch.ops.aten.amax.default(view_default_98, [-1], True)
    sub_tensor_16: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_98, amax_default_8);  view_default_98 = amax_default_8 = None
    exp_default_8: "f32[1, 4, 512, 512]" = torch.ops.aten.exp.default(sub_tensor_16);  sub_tensor_16 = None
    sum_dim_int_list_16: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_8, [-1], True)
    div_tensor_8: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_default_8, sum_dim_int_list_16);  exp_default_8 = sum_dim_int_list_16 = None
    alias_default_16: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(div_tensor_8)
    native_dropout_default_8 = torch.ops.aten.native_dropout.default(div_tensor_8, 0.1, True);  div_tensor_8 = None
    getitem_142: "f32[1, 4, 512, 512]" = native_dropout_default_8[0]
    getitem_143: "b8[1, 4, 512, 512]" = native_dropout_default_8[1];  native_dropout_default_8 = None
    expand_default_34: "f32[1, 4, 512, 512]" = torch.ops.aten.expand.default(getitem_142, [1, 4, 512, 512]);  getitem_142 = None
    view_default_99: "f32[4, 512, 512]" = torch.ops.aten.view.default(expand_default_34, [4, 512, 512]);  expand_default_34 = None
    expand_default_35: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(clone_default_34, [1, 4, 512, 64]);  clone_default_34 = None
    view_default_100: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_default_35, [4, 512, 64]);  expand_default_35 = None
    bmm_default_49: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_default_99, view_default_100)
    view_default_101: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_default_49, [1, 4, 512, 64]);  bmm_default_49 = None
    permute_default_49: "f32[4, 512, 512]" = torch.ops.aten.permute.default(view_default_99, [0, 2, 1]);  view_default_99 = None
    permute_default_50: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_default_100, [0, 2, 1]);  view_default_100 = None
    alias_default_17: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(alias_default_16);  alias_default_16 = None
    permute_default_51: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_default_96, [0, 2, 1]);  view_default_96 = None
    permute_default_52: "f32[4, 512, 64]" = torch.ops.aten.permute.default(view_default_97, [0, 2, 1]);  view_default_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_41: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_default_101, [0, 2, 1, 3]);  view_default_101 = None
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
    sub_12: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_29, getitem_39);  add_29 = getitem_39 = None
    mul_24: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_7);  sub_12 = None
    mul_25: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_24, primals_64)
    add_31: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_25, primals_65);  mul_25 = primals_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_86: "f32[512, 256]" = torch.ops.aten.view.default(add_31, [512, 256])
    permute_43: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
    addmm_23: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_67, view_86, permute_43);  primals_67 = None
    view_87: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_23, [1, 512, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_26: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_87, 0.5)
    mul_27: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_87, 0.7071067811865476);  view_87 = None
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
    sub_13: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_33, getitem_43);  add_33 = getitem_43 = None
    mul_29: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_8);  sub_13 = None
    mul_30: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_29, primals_70)
    add_35: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_30, primals_71);  mul_30 = primals_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_90: "f32[512, 256]" = torch.ops.aten.view.default(add_35, [512, 256])
    permute_45: "f32[256, 256]" = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
    addmm_25: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_73, view_90, permute_45);  primals_73 = None
    view_91: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_25, [1, 512, 256]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_46: "f32[256, 256]" = torch.ops.aten.permute.default(primals_74, [1, 0]);  primals_74 = None
    addmm_26: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_75, view_90, permute_46);  primals_75 = None
    view_93: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_26, [1, 512, 256]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_94: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_93, [1, 512, 4, 64]);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_47: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_48: "f32[256, 256]" = torch.ops.aten.permute.default(primals_76, [1, 0]);  primals_76 = None
    addmm_27: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_77, view_90, permute_48);  primals_77 = None
    view_96: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_27, [1, 512, 256]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_97: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_96, [1, 512, 4, 64]);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_49: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_97, [0, 2, 1, 3]);  view_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_98: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_91, [1, 512, 4, 64]);  view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_50: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None
    
    # No stacktrace found for following nodes
    clone_default_28: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_50, memory_format = torch.contiguous_format);  permute_50 = None
    clone_default_29: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_47, memory_format = torch.contiguous_format);  permute_47 = None
    clone_default_30: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
    mul_scalar_28: "f32[1, 4, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default_28, 0.3535533905932738);  clone_default_28 = None
    permute_default_42: "f32[1, 4, 64, 512]" = torch.ops.aten.permute.default(clone_default_29, [0, 1, 3, 2]);  clone_default_29 = None
    mul_scalar_29: "f32[1, 4, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_42, 0.3535533905932738);  permute_default_42 = None
    expand_default_28: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(mul_scalar_28, [1, 4, 512, 64]);  mul_scalar_28 = None
    view_default_84: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_default_28, [4, 512, 64]);  expand_default_28 = None
    expand_default_29: "f32[1, 4, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_29, [1, 4, 64, 512]);  mul_scalar_29 = None
    view_default_85: "f32[4, 64, 512]" = torch.ops.aten.view.default(expand_default_29, [4, 64, 512]);  expand_default_29 = None
    bmm_default_42: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_default_84, view_default_85)
    view_default_86: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_default_42, [1, 4, 512, 512]);  bmm_default_42 = None
    amax_default_7: "f32[1, 4, 512, 1]" = torch.ops.aten.amax.default(view_default_86, [-1], True)
    sub_tensor_14: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_86, amax_default_7);  view_default_86 = amax_default_7 = None
    exp_default_7: "f32[1, 4, 512, 512]" = torch.ops.aten.exp.default(sub_tensor_14);  sub_tensor_14 = None
    sum_dim_int_list_14: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_7, [-1], True)
    div_tensor_7: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_default_7, sum_dim_int_list_14);  exp_default_7 = sum_dim_int_list_14 = None
    alias_default_14: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(div_tensor_7)
    native_dropout_default_7 = torch.ops.aten.native_dropout.default(div_tensor_7, 0.1, True);  div_tensor_7 = None
    getitem_140: "f32[1, 4, 512, 512]" = native_dropout_default_7[0]
    getitem_141: "b8[1, 4, 512, 512]" = native_dropout_default_7[1];  native_dropout_default_7 = None
    expand_default_30: "f32[1, 4, 512, 512]" = torch.ops.aten.expand.default(getitem_140, [1, 4, 512, 512]);  getitem_140 = None
    view_default_87: "f32[4, 512, 512]" = torch.ops.aten.view.default(expand_default_30, [4, 512, 512]);  expand_default_30 = None
    expand_default_31: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(clone_default_30, [1, 4, 512, 64]);  clone_default_30 = None
    view_default_88: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_default_31, [4, 512, 64]);  expand_default_31 = None
    bmm_default_43: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_default_87, view_default_88)
    view_default_89: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_default_43, [1, 4, 512, 64]);  bmm_default_43 = None
    permute_default_43: "f32[4, 512, 512]" = torch.ops.aten.permute.default(view_default_87, [0, 2, 1]);  view_default_87 = None
    permute_default_44: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_default_88, [0, 2, 1]);  view_default_88 = None
    alias_default_15: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(alias_default_14);  alias_default_14 = None
    permute_default_45: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_default_84, [0, 2, 1]);  view_default_84 = None
    permute_default_46: "f32[4, 512, 64]" = torch.ops.aten.permute.default(view_default_85, [0, 2, 1]);  view_default_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_52: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_default_89, [0, 2, 1, 3]);  view_default_89 = None
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
    sub_15: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_37, getitem_49);  add_37 = getitem_49 = None
    mul_31: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_9);  sub_15 = None
    mul_32: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_31, primals_80)
    add_39: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_32, primals_81);  mul_32 = primals_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_108: "f32[512, 256]" = torch.ops.aten.view.default(add_39, [512, 256])
    permute_54: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_82, [1, 0]);  primals_82 = None
    addmm_29: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_83, view_108, permute_54);  primals_83 = None
    view_109: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_29, [1, 512, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_33: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_109, 0.5)
    mul_34: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_109, 0.7071067811865476);  view_109 = None
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
    sub_16: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_41, getitem_53);  add_41 = getitem_53 = None
    mul_36: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_10);  sub_16 = None
    mul_37: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_36, primals_86)
    add_43: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_37, primals_87);  mul_37 = primals_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_112: "f32[512, 256]" = torch.ops.aten.view.default(add_43, [512, 256])
    permute_56: "f32[256, 256]" = torch.ops.aten.permute.default(primals_88, [1, 0]);  primals_88 = None
    addmm_31: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_89, view_112, permute_56);  primals_89 = None
    view_113: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_31, [1, 512, 256]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_57: "f32[256, 256]" = torch.ops.aten.permute.default(primals_90, [1, 0]);  primals_90 = None
    addmm_32: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_91, view_112, permute_57);  primals_91 = None
    view_115: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_32, [1, 512, 256]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_116: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_115, [1, 512, 4, 64]);  view_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_58: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_116, [0, 2, 1, 3]);  view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_59: "f32[256, 256]" = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
    addmm_33: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_93, view_112, permute_59);  primals_93 = None
    view_118: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_33, [1, 512, 256]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_119: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_118, [1, 512, 4, 64]);  view_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_60: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_119, [0, 2, 1, 3]);  view_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_120: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_113, [1, 512, 4, 64]);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_61: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
    
    # No stacktrace found for following nodes
    clone_default_24: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_61, memory_format = torch.contiguous_format);  permute_61 = None
    clone_default_25: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_58, memory_format = torch.contiguous_format);  permute_58 = None
    clone_default_26: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
    mul_scalar_24: "f32[1, 4, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default_24, 0.3535533905932738);  clone_default_24 = None
    permute_default_36: "f32[1, 4, 64, 512]" = torch.ops.aten.permute.default(clone_default_25, [0, 1, 3, 2]);  clone_default_25 = None
    mul_scalar_25: "f32[1, 4, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_36, 0.3535533905932738);  permute_default_36 = None
    expand_default_24: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(mul_scalar_24, [1, 4, 512, 64]);  mul_scalar_24 = None
    view_default_72: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_default_24, [4, 512, 64]);  expand_default_24 = None
    expand_default_25: "f32[1, 4, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_25, [1, 4, 64, 512]);  mul_scalar_25 = None
    view_default_73: "f32[4, 64, 512]" = torch.ops.aten.view.default(expand_default_25, [4, 64, 512]);  expand_default_25 = None
    bmm_default_36: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_default_72, view_default_73)
    view_default_74: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_default_36, [1, 4, 512, 512]);  bmm_default_36 = None
    amax_default_6: "f32[1, 4, 512, 1]" = torch.ops.aten.amax.default(view_default_74, [-1], True)
    sub_tensor_12: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_74, amax_default_6);  view_default_74 = amax_default_6 = None
    exp_default_6: "f32[1, 4, 512, 512]" = torch.ops.aten.exp.default(sub_tensor_12);  sub_tensor_12 = None
    sum_dim_int_list_12: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_6, [-1], True)
    div_tensor_6: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_default_6, sum_dim_int_list_12);  exp_default_6 = sum_dim_int_list_12 = None
    alias_default_12: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(div_tensor_6)
    native_dropout_default_6 = torch.ops.aten.native_dropout.default(div_tensor_6, 0.1, True);  div_tensor_6 = None
    getitem_138: "f32[1, 4, 512, 512]" = native_dropout_default_6[0]
    getitem_139: "b8[1, 4, 512, 512]" = native_dropout_default_6[1];  native_dropout_default_6 = None
    expand_default_26: "f32[1, 4, 512, 512]" = torch.ops.aten.expand.default(getitem_138, [1, 4, 512, 512]);  getitem_138 = None
    view_default_75: "f32[4, 512, 512]" = torch.ops.aten.view.default(expand_default_26, [4, 512, 512]);  expand_default_26 = None
    expand_default_27: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(clone_default_26, [1, 4, 512, 64]);  clone_default_26 = None
    view_default_76: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_default_27, [4, 512, 64]);  expand_default_27 = None
    bmm_default_37: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_default_75, view_default_76)
    view_default_77: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_default_37, [1, 4, 512, 64]);  bmm_default_37 = None
    permute_default_37: "f32[4, 512, 512]" = torch.ops.aten.permute.default(view_default_75, [0, 2, 1]);  view_default_75 = None
    permute_default_38: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_default_76, [0, 2, 1]);  view_default_76 = None
    alias_default_13: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(alias_default_12);  alias_default_12 = None
    permute_default_39: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_default_72, [0, 2, 1]);  view_default_72 = None
    permute_default_40: "f32[4, 512, 64]" = torch.ops.aten.permute.default(view_default_73, [0, 2, 1]);  view_default_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_63: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_default_77, [0, 2, 1, 3]);  view_default_77 = None
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
    sub_18: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_45, getitem_59);  add_45 = getitem_59 = None
    mul_38: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_11);  sub_18 = None
    mul_39: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_38, primals_96)
    add_47: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_39, primals_97);  mul_39 = primals_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_130: "f32[512, 256]" = torch.ops.aten.view.default(add_47, [512, 256])
    permute_65: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_98, [1, 0]);  primals_98 = None
    addmm_35: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_99, view_130, permute_65);  primals_99 = None
    view_131: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_35, [1, 512, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_40: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_131, 0.5)
    mul_41: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_131, 0.7071067811865476);  view_131 = None
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
    sub_19: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_49, getitem_63);  add_49 = getitem_63 = None
    mul_43: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_12);  sub_19 = None
    mul_44: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_43, primals_102)
    add_51: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_44, primals_103);  mul_44 = primals_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_134: "f32[512, 256]" = torch.ops.aten.view.default(add_51, [512, 256])
    permute_67: "f32[256, 256]" = torch.ops.aten.permute.default(primals_104, [1, 0]);  primals_104 = None
    addmm_37: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_105, view_134, permute_67);  primals_105 = None
    view_135: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_37, [1, 512, 256]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_68: "f32[256, 256]" = torch.ops.aten.permute.default(primals_106, [1, 0]);  primals_106 = None
    addmm_38: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_107, view_134, permute_68);  primals_107 = None
    view_137: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_38, [1, 512, 256]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_138: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_137, [1, 512, 4, 64]);  view_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_69: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_70: "f32[256, 256]" = torch.ops.aten.permute.default(primals_108, [1, 0]);  primals_108 = None
    addmm_39: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_109, view_134, permute_70);  primals_109 = None
    view_140: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_39, [1, 512, 256]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_141: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_140, [1, 512, 4, 64]);  view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_71: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_141, [0, 2, 1, 3]);  view_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_142: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_135, [1, 512, 4, 64]);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_72: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_142, [0, 2, 1, 3]);  view_142 = None
    
    # No stacktrace found for following nodes
    clone_default_20: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_72, memory_format = torch.contiguous_format);  permute_72 = None
    clone_default_21: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_69, memory_format = torch.contiguous_format);  permute_69 = None
    clone_default_22: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
    mul_scalar_20: "f32[1, 4, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default_20, 0.3535533905932738);  clone_default_20 = None
    permute_default_30: "f32[1, 4, 64, 512]" = torch.ops.aten.permute.default(clone_default_21, [0, 1, 3, 2]);  clone_default_21 = None
    mul_scalar_21: "f32[1, 4, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_30, 0.3535533905932738);  permute_default_30 = None
    expand_default_20: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(mul_scalar_20, [1, 4, 512, 64]);  mul_scalar_20 = None
    view_default_60: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_default_20, [4, 512, 64]);  expand_default_20 = None
    expand_default_21: "f32[1, 4, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_21, [1, 4, 64, 512]);  mul_scalar_21 = None
    view_default_61: "f32[4, 64, 512]" = torch.ops.aten.view.default(expand_default_21, [4, 64, 512]);  expand_default_21 = None
    bmm_default_30: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_default_60, view_default_61)
    view_default_62: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_default_30, [1, 4, 512, 512]);  bmm_default_30 = None
    amax_default_5: "f32[1, 4, 512, 1]" = torch.ops.aten.amax.default(view_default_62, [-1], True)
    sub_tensor_10: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_62, amax_default_5);  view_default_62 = amax_default_5 = None
    exp_default_5: "f32[1, 4, 512, 512]" = torch.ops.aten.exp.default(sub_tensor_10);  sub_tensor_10 = None
    sum_dim_int_list_10: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_5, [-1], True)
    div_tensor_5: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_default_5, sum_dim_int_list_10);  exp_default_5 = sum_dim_int_list_10 = None
    alias_default_10: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(div_tensor_5)
    native_dropout_default_5 = torch.ops.aten.native_dropout.default(div_tensor_5, 0.1, True);  div_tensor_5 = None
    getitem_136: "f32[1, 4, 512, 512]" = native_dropout_default_5[0]
    getitem_137: "b8[1, 4, 512, 512]" = native_dropout_default_5[1];  native_dropout_default_5 = None
    expand_default_22: "f32[1, 4, 512, 512]" = torch.ops.aten.expand.default(getitem_136, [1, 4, 512, 512]);  getitem_136 = None
    view_default_63: "f32[4, 512, 512]" = torch.ops.aten.view.default(expand_default_22, [4, 512, 512]);  expand_default_22 = None
    expand_default_23: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(clone_default_22, [1, 4, 512, 64]);  clone_default_22 = None
    view_default_64: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_default_23, [4, 512, 64]);  expand_default_23 = None
    bmm_default_31: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_default_63, view_default_64)
    view_default_65: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_default_31, [1, 4, 512, 64]);  bmm_default_31 = None
    permute_default_31: "f32[4, 512, 512]" = torch.ops.aten.permute.default(view_default_63, [0, 2, 1]);  view_default_63 = None
    permute_default_32: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_default_64, [0, 2, 1]);  view_default_64 = None
    alias_default_11: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(alias_default_10);  alias_default_10 = None
    permute_default_33: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_default_60, [0, 2, 1]);  view_default_60 = None
    permute_default_34: "f32[4, 512, 64]" = torch.ops.aten.permute.default(view_default_61, [0, 2, 1]);  view_default_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_74: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_default_65, [0, 2, 1, 3]);  view_default_65 = None
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
    sub_21: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_53, getitem_69);  add_53 = getitem_69 = None
    mul_45: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_13);  sub_21 = None
    mul_46: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_45, primals_112)
    add_55: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_46, primals_113);  mul_46 = primals_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_152: "f32[512, 256]" = torch.ops.aten.view.default(add_55, [512, 256])
    permute_76: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_114, [1, 0]);  primals_114 = None
    addmm_41: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_115, view_152, permute_76);  primals_115 = None
    view_153: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_41, [1, 512, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_47: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_153, 0.5)
    mul_48: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_153, 0.7071067811865476);  view_153 = None
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
    sub_22: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_57, getitem_73);  add_57 = getitem_73 = None
    mul_50: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_14);  sub_22 = None
    mul_51: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_50, primals_118)
    add_59: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_51, primals_119);  mul_51 = primals_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_156: "f32[512, 256]" = torch.ops.aten.view.default(add_59, [512, 256])
    permute_78: "f32[256, 256]" = torch.ops.aten.permute.default(primals_120, [1, 0]);  primals_120 = None
    addmm_43: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_121, view_156, permute_78);  primals_121 = None
    view_157: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_43, [1, 512, 256]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_79: "f32[256, 256]" = torch.ops.aten.permute.default(primals_122, [1, 0]);  primals_122 = None
    addmm_44: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_123, view_156, permute_79);  primals_123 = None
    view_159: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_44, [1, 512, 256]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_160: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_159, [1, 512, 4, 64]);  view_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_80: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_160, [0, 2, 1, 3]);  view_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_81: "f32[256, 256]" = torch.ops.aten.permute.default(primals_124, [1, 0]);  primals_124 = None
    addmm_45: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_125, view_156, permute_81);  primals_125 = None
    view_162: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_45, [1, 512, 256]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_163: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_162, [1, 512, 4, 64]);  view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_82: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_163, [0, 2, 1, 3]);  view_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_164: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_157, [1, 512, 4, 64]);  view_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_83: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_164, [0, 2, 1, 3]);  view_164 = None
    
    # No stacktrace found for following nodes
    clone_default_16: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_83, memory_format = torch.contiguous_format);  permute_83 = None
    clone_default_17: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format);  permute_80 = None
    clone_default_18: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
    mul_scalar_16: "f32[1, 4, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default_16, 0.3535533905932738);  clone_default_16 = None
    permute_default_24: "f32[1, 4, 64, 512]" = torch.ops.aten.permute.default(clone_default_17, [0, 1, 3, 2]);  clone_default_17 = None
    mul_scalar_17: "f32[1, 4, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_24, 0.3535533905932738);  permute_default_24 = None
    expand_default_16: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(mul_scalar_16, [1, 4, 512, 64]);  mul_scalar_16 = None
    view_default_48: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_default_16, [4, 512, 64]);  expand_default_16 = None
    expand_default_17: "f32[1, 4, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_17, [1, 4, 64, 512]);  mul_scalar_17 = None
    view_default_49: "f32[4, 64, 512]" = torch.ops.aten.view.default(expand_default_17, [4, 64, 512]);  expand_default_17 = None
    bmm_default_24: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_default_48, view_default_49)
    view_default_50: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_default_24, [1, 4, 512, 512]);  bmm_default_24 = None
    amax_default_4: "f32[1, 4, 512, 1]" = torch.ops.aten.amax.default(view_default_50, [-1], True)
    sub_tensor_8: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_50, amax_default_4);  view_default_50 = amax_default_4 = None
    exp_default_4: "f32[1, 4, 512, 512]" = torch.ops.aten.exp.default(sub_tensor_8);  sub_tensor_8 = None
    sum_dim_int_list_8: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_4, [-1], True)
    div_tensor_4: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_default_4, sum_dim_int_list_8);  exp_default_4 = sum_dim_int_list_8 = None
    alias_default_8: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(div_tensor_4)
    native_dropout_default_4 = torch.ops.aten.native_dropout.default(div_tensor_4, 0.1, True);  div_tensor_4 = None
    getitem_134: "f32[1, 4, 512, 512]" = native_dropout_default_4[0]
    getitem_135: "b8[1, 4, 512, 512]" = native_dropout_default_4[1];  native_dropout_default_4 = None
    expand_default_18: "f32[1, 4, 512, 512]" = torch.ops.aten.expand.default(getitem_134, [1, 4, 512, 512]);  getitem_134 = None
    view_default_51: "f32[4, 512, 512]" = torch.ops.aten.view.default(expand_default_18, [4, 512, 512]);  expand_default_18 = None
    expand_default_19: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(clone_default_18, [1, 4, 512, 64]);  clone_default_18 = None
    view_default_52: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_default_19, [4, 512, 64]);  expand_default_19 = None
    bmm_default_25: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_default_51, view_default_52)
    view_default_53: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_default_25, [1, 4, 512, 64]);  bmm_default_25 = None
    permute_default_25: "f32[4, 512, 512]" = torch.ops.aten.permute.default(view_default_51, [0, 2, 1]);  view_default_51 = None
    permute_default_26: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_default_52, [0, 2, 1]);  view_default_52 = None
    alias_default_9: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(alias_default_8);  alias_default_8 = None
    permute_default_27: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_default_48, [0, 2, 1]);  view_default_48 = None
    permute_default_28: "f32[4, 512, 64]" = torch.ops.aten.permute.default(view_default_49, [0, 2, 1]);  view_default_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_85: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_default_53, [0, 2, 1, 3]);  view_default_53 = None
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
    sub_24: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_61, getitem_79);  add_61 = getitem_79 = None
    mul_52: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_15);  sub_24 = None
    mul_53: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_52, primals_128)
    add_63: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_53, primals_129);  mul_53 = primals_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_174: "f32[512, 256]" = torch.ops.aten.view.default(add_63, [512, 256])
    permute_87: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_130, [1, 0]);  primals_130 = None
    addmm_47: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_131, view_174, permute_87);  primals_131 = None
    view_175: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_47, [1, 512, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_54: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_175, 0.5)
    mul_55: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_175, 0.7071067811865476);  view_175 = None
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
    sub_25: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_65, getitem_83);  add_65 = getitem_83 = None
    mul_57: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_16);  sub_25 = None
    mul_58: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_57, primals_134)
    add_67: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_58, primals_135);  mul_58 = primals_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_178: "f32[512, 256]" = torch.ops.aten.view.default(add_67, [512, 256])
    permute_89: "f32[256, 256]" = torch.ops.aten.permute.default(primals_136, [1, 0]);  primals_136 = None
    addmm_49: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_137, view_178, permute_89);  primals_137 = None
    view_179: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_49, [1, 512, 256]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_90: "f32[256, 256]" = torch.ops.aten.permute.default(primals_138, [1, 0]);  primals_138 = None
    addmm_50: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_139, view_178, permute_90);  primals_139 = None
    view_181: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_50, [1, 512, 256]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_182: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_181, [1, 512, 4, 64]);  view_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_91: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_92: "f32[256, 256]" = torch.ops.aten.permute.default(primals_140, [1, 0]);  primals_140 = None
    addmm_51: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_141, view_178, permute_92);  primals_141 = None
    view_184: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_51, [1, 512, 256]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_185: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_184, [1, 512, 4, 64]);  view_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_93: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_186: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_179, [1, 512, 4, 64]);  view_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_94: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
    
    # No stacktrace found for following nodes
    clone_default_12: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_94, memory_format = torch.contiguous_format);  permute_94 = None
    clone_default_13: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_91, memory_format = torch.contiguous_format);  permute_91 = None
    clone_default_14: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
    mul_scalar_12: "f32[1, 4, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default_12, 0.3535533905932738);  clone_default_12 = None
    permute_default_18: "f32[1, 4, 64, 512]" = torch.ops.aten.permute.default(clone_default_13, [0, 1, 3, 2]);  clone_default_13 = None
    mul_scalar_13: "f32[1, 4, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_18, 0.3535533905932738);  permute_default_18 = None
    expand_default_12: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(mul_scalar_12, [1, 4, 512, 64]);  mul_scalar_12 = None
    view_default_36: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_default_12, [4, 512, 64]);  expand_default_12 = None
    expand_default_13: "f32[1, 4, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_13, [1, 4, 64, 512]);  mul_scalar_13 = None
    view_default_37: "f32[4, 64, 512]" = torch.ops.aten.view.default(expand_default_13, [4, 64, 512]);  expand_default_13 = None
    bmm_default_18: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_default_36, view_default_37)
    view_default_38: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_default_18, [1, 4, 512, 512]);  bmm_default_18 = None
    amax_default_3: "f32[1, 4, 512, 1]" = torch.ops.aten.amax.default(view_default_38, [-1], True)
    sub_tensor_6: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_38, amax_default_3);  view_default_38 = amax_default_3 = None
    exp_default_3: "f32[1, 4, 512, 512]" = torch.ops.aten.exp.default(sub_tensor_6);  sub_tensor_6 = None
    sum_dim_int_list_6: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_3, [-1], True)
    div_tensor_3: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_default_3, sum_dim_int_list_6);  exp_default_3 = sum_dim_int_list_6 = None
    alias_default_6: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(div_tensor_3)
    native_dropout_default_3 = torch.ops.aten.native_dropout.default(div_tensor_3, 0.1, True);  div_tensor_3 = None
    getitem_132: "f32[1, 4, 512, 512]" = native_dropout_default_3[0]
    getitem_133: "b8[1, 4, 512, 512]" = native_dropout_default_3[1];  native_dropout_default_3 = None
    expand_default_14: "f32[1, 4, 512, 512]" = torch.ops.aten.expand.default(getitem_132, [1, 4, 512, 512]);  getitem_132 = None
    view_default_39: "f32[4, 512, 512]" = torch.ops.aten.view.default(expand_default_14, [4, 512, 512]);  expand_default_14 = None
    expand_default_15: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(clone_default_14, [1, 4, 512, 64]);  clone_default_14 = None
    view_default_40: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_default_15, [4, 512, 64]);  expand_default_15 = None
    bmm_default_19: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_default_39, view_default_40)
    view_default_41: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_default_19, [1, 4, 512, 64]);  bmm_default_19 = None
    permute_default_19: "f32[4, 512, 512]" = torch.ops.aten.permute.default(view_default_39, [0, 2, 1]);  view_default_39 = None
    permute_default_20: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_default_40, [0, 2, 1]);  view_default_40 = None
    alias_default_7: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(alias_default_6);  alias_default_6 = None
    permute_default_21: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_default_36, [0, 2, 1]);  view_default_36 = None
    permute_default_22: "f32[4, 512, 64]" = torch.ops.aten.permute.default(view_default_37, [0, 2, 1]);  view_default_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_96: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_default_41, [0, 2, 1, 3]);  view_default_41 = None
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
    sub_27: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_69, getitem_89);  add_69 = getitem_89 = None
    mul_59: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_17);  sub_27 = None
    mul_60: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_59, primals_144)
    add_71: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_60, primals_145);  mul_60 = primals_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_196: "f32[512, 256]" = torch.ops.aten.view.default(add_71, [512, 256])
    permute_98: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_146, [1, 0]);  primals_146 = None
    addmm_53: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_147, view_196, permute_98);  primals_147 = None
    view_197: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_53, [1, 512, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_61: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_197, 0.5)
    mul_62: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_197, 0.7071067811865476);  view_197 = None
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
    sub_28: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_73, getitem_93);  add_73 = getitem_93 = None
    mul_64: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_18);  sub_28 = None
    mul_65: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_64, primals_150)
    add_75: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_65, primals_151);  mul_65 = primals_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_200: "f32[512, 256]" = torch.ops.aten.view.default(add_75, [512, 256])
    permute_100: "f32[256, 256]" = torch.ops.aten.permute.default(primals_152, [1, 0]);  primals_152 = None
    addmm_55: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_153, view_200, permute_100);  primals_153 = None
    view_201: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_55, [1, 512, 256]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_101: "f32[256, 256]" = torch.ops.aten.permute.default(primals_154, [1, 0]);  primals_154 = None
    addmm_56: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_155, view_200, permute_101);  primals_155 = None
    view_203: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_56, [1, 512, 256]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_204: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_203, [1, 512, 4, 64]);  view_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_102: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_204, [0, 2, 1, 3]);  view_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_103: "f32[256, 256]" = torch.ops.aten.permute.default(primals_156, [1, 0]);  primals_156 = None
    addmm_57: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_157, view_200, permute_103);  primals_157 = None
    view_206: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_57, [1, 512, 256]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_207: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_206, [1, 512, 4, 64]);  view_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_104: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_207, [0, 2, 1, 3]);  view_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_208: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_201, [1, 512, 4, 64]);  view_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_105: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None
    
    # No stacktrace found for following nodes
    clone_default_8: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_105, memory_format = torch.contiguous_format);  permute_105 = None
    clone_default_9: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_102, memory_format = torch.contiguous_format);  permute_102 = None
    clone_default_10: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
    mul_scalar_8: "f32[1, 4, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default_8, 0.3535533905932738);  clone_default_8 = None
    permute_default_12: "f32[1, 4, 64, 512]" = torch.ops.aten.permute.default(clone_default_9, [0, 1, 3, 2]);  clone_default_9 = None
    mul_scalar_9: "f32[1, 4, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_12, 0.3535533905932738);  permute_default_12 = None
    expand_default_8: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(mul_scalar_8, [1, 4, 512, 64]);  mul_scalar_8 = None
    view_default_24: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_default_8, [4, 512, 64]);  expand_default_8 = None
    expand_default_9: "f32[1, 4, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_9, [1, 4, 64, 512]);  mul_scalar_9 = None
    view_default_25: "f32[4, 64, 512]" = torch.ops.aten.view.default(expand_default_9, [4, 64, 512]);  expand_default_9 = None
    bmm_default_12: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_default_24, view_default_25)
    view_default_26: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_default_12, [1, 4, 512, 512]);  bmm_default_12 = None
    amax_default_2: "f32[1, 4, 512, 1]" = torch.ops.aten.amax.default(view_default_26, [-1], True)
    sub_tensor_4: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_26, amax_default_2);  view_default_26 = amax_default_2 = None
    exp_default_2: "f32[1, 4, 512, 512]" = torch.ops.aten.exp.default(sub_tensor_4);  sub_tensor_4 = None
    sum_dim_int_list_4: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_2, [-1], True)
    div_tensor_2: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_default_2, sum_dim_int_list_4);  exp_default_2 = sum_dim_int_list_4 = None
    alias_default_4: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(div_tensor_2)
    native_dropout_default_2 = torch.ops.aten.native_dropout.default(div_tensor_2, 0.1, True);  div_tensor_2 = None
    getitem_130: "f32[1, 4, 512, 512]" = native_dropout_default_2[0]
    getitem_131: "b8[1, 4, 512, 512]" = native_dropout_default_2[1];  native_dropout_default_2 = None
    expand_default_10: "f32[1, 4, 512, 512]" = torch.ops.aten.expand.default(getitem_130, [1, 4, 512, 512]);  getitem_130 = None
    view_default_27: "f32[4, 512, 512]" = torch.ops.aten.view.default(expand_default_10, [4, 512, 512]);  expand_default_10 = None
    expand_default_11: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(clone_default_10, [1, 4, 512, 64]);  clone_default_10 = None
    view_default_28: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_default_11, [4, 512, 64]);  expand_default_11 = None
    bmm_default_13: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_default_27, view_default_28)
    view_default_29: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_default_13, [1, 4, 512, 64]);  bmm_default_13 = None
    permute_default_13: "f32[4, 512, 512]" = torch.ops.aten.permute.default(view_default_27, [0, 2, 1]);  view_default_27 = None
    permute_default_14: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_default_28, [0, 2, 1]);  view_default_28 = None
    alias_default_5: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(alias_default_4);  alias_default_4 = None
    permute_default_15: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_default_24, [0, 2, 1]);  view_default_24 = None
    permute_default_16: "f32[4, 512, 64]" = torch.ops.aten.permute.default(view_default_25, [0, 2, 1]);  view_default_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_107: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_default_29, [0, 2, 1, 3]);  view_default_29 = None
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
    sub_30: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_77, getitem_99);  add_77 = getitem_99 = None
    mul_66: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_19);  sub_30 = None
    mul_67: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_66, primals_160)
    add_79: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_67, primals_161);  mul_67 = primals_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_218: "f32[512, 256]" = torch.ops.aten.view.default(add_79, [512, 256])
    permute_109: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_162, [1, 0]);  primals_162 = None
    addmm_59: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_163, view_218, permute_109);  primals_163 = None
    view_219: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_59, [1, 512, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_68: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_219, 0.5)
    mul_69: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_219, 0.7071067811865476);  view_219 = None
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
    sub_31: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_81, getitem_103);  add_81 = getitem_103 = None
    mul_71: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_20);  sub_31 = None
    mul_72: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_71, primals_166)
    add_83: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_72, primals_167);  mul_72 = primals_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_222: "f32[512, 256]" = torch.ops.aten.view.default(add_83, [512, 256])
    permute_111: "f32[256, 256]" = torch.ops.aten.permute.default(primals_168, [1, 0]);  primals_168 = None
    addmm_61: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_169, view_222, permute_111);  primals_169 = None
    view_223: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_61, [1, 512, 256]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_112: "f32[256, 256]" = torch.ops.aten.permute.default(primals_170, [1, 0]);  primals_170 = None
    addmm_62: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_171, view_222, permute_112);  primals_171 = None
    view_225: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_62, [1, 512, 256]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_226: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_225, [1, 512, 4, 64]);  view_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_113: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_114: "f32[256, 256]" = torch.ops.aten.permute.default(primals_172, [1, 0]);  primals_172 = None
    addmm_63: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_173, view_222, permute_114);  primals_173 = None
    view_228: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_63, [1, 512, 256]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_229: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_228, [1, 512, 4, 64]);  view_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_115: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_229, [0, 2, 1, 3]);  view_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_230: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_223, [1, 512, 4, 64]);  view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_116: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_230, [0, 2, 1, 3]);  view_230 = None
    
    # No stacktrace found for following nodes
    clone_default_4: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_116, memory_format = torch.contiguous_format);  permute_116 = None
    clone_default_5: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_113, memory_format = torch.contiguous_format);  permute_113 = None
    clone_default_6: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
    mul_scalar_4: "f32[1, 4, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default_4, 0.3535533905932738);  clone_default_4 = None
    permute_default_6: "f32[1, 4, 64, 512]" = torch.ops.aten.permute.default(clone_default_5, [0, 1, 3, 2]);  clone_default_5 = None
    mul_scalar_5: "f32[1, 4, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_6, 0.3535533905932738);  permute_default_6 = None
    expand_default_4: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(mul_scalar_4, [1, 4, 512, 64]);  mul_scalar_4 = None
    view_default_12: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_default_4, [4, 512, 64]);  expand_default_4 = None
    expand_default_5: "f32[1, 4, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_5, [1, 4, 64, 512]);  mul_scalar_5 = None
    view_default_13: "f32[4, 64, 512]" = torch.ops.aten.view.default(expand_default_5, [4, 64, 512]);  expand_default_5 = None
    bmm_default_6: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_default_12, view_default_13)
    view_default_14: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_default_6, [1, 4, 512, 512]);  bmm_default_6 = None
    amax_default_1: "f32[1, 4, 512, 1]" = torch.ops.aten.amax.default(view_default_14, [-1], True)
    sub_tensor_2: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_14, amax_default_1);  view_default_14 = amax_default_1 = None
    exp_default_1: "f32[1, 4, 512, 512]" = torch.ops.aten.exp.default(sub_tensor_2);  sub_tensor_2 = None
    sum_dim_int_list_2: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_1, [-1], True)
    div_tensor_1: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_default_1, sum_dim_int_list_2);  exp_default_1 = sum_dim_int_list_2 = None
    alias_default_2: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(div_tensor_1)
    native_dropout_default_1 = torch.ops.aten.native_dropout.default(div_tensor_1, 0.1, True);  div_tensor_1 = None
    getitem_128: "f32[1, 4, 512, 512]" = native_dropout_default_1[0]
    getitem_129: "b8[1, 4, 512, 512]" = native_dropout_default_1[1];  native_dropout_default_1 = None
    expand_default_6: "f32[1, 4, 512, 512]" = torch.ops.aten.expand.default(getitem_128, [1, 4, 512, 512]);  getitem_128 = None
    view_default_15: "f32[4, 512, 512]" = torch.ops.aten.view.default(expand_default_6, [4, 512, 512]);  expand_default_6 = None
    expand_default_7: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(clone_default_6, [1, 4, 512, 64]);  clone_default_6 = None
    view_default_16: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_default_7, [4, 512, 64]);  expand_default_7 = None
    bmm_default_7: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_default_15, view_default_16)
    view_default_17: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_default_7, [1, 4, 512, 64]);  bmm_default_7 = None
    permute_default_7: "f32[4, 512, 512]" = torch.ops.aten.permute.default(view_default_15, [0, 2, 1]);  view_default_15 = None
    permute_default_8: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_default_16, [0, 2, 1]);  view_default_16 = None
    alias_default_3: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(alias_default_2);  alias_default_2 = None
    permute_default_9: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_default_12, [0, 2, 1]);  view_default_12 = None
    permute_default_10: "f32[4, 512, 64]" = torch.ops.aten.permute.default(view_default_13, [0, 2, 1]);  view_default_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_118: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_default_17, [0, 2, 1, 3]);  view_default_17 = None
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
    sub_33: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_85, getitem_109);  add_85 = getitem_109 = None
    mul_73: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_21);  sub_33 = None
    mul_74: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_73, primals_176)
    add_87: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_74, primals_177);  mul_74 = primals_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_240: "f32[512, 256]" = torch.ops.aten.view.default(add_87, [512, 256])
    permute_120: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_178, [1, 0]);  primals_178 = None
    addmm_65: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_179, view_240, permute_120);  primals_179 = None
    view_241: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_65, [1, 512, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_75: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_241, 0.5)
    mul_76: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_241, 0.7071067811865476);  view_241 = None
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
    sub_34: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_89, getitem_113);  add_89 = getitem_113 = None
    mul_78: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_22);  sub_34 = None
    mul_79: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_78, primals_182)
    add_91: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_79, primals_183);  mul_79 = primals_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_244: "f32[512, 256]" = torch.ops.aten.view.default(add_91, [512, 256])
    permute_122: "f32[256, 256]" = torch.ops.aten.permute.default(primals_184, [1, 0]);  primals_184 = None
    addmm_67: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_185, view_244, permute_122);  primals_185 = None
    view_245: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_67, [1, 512, 256]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_123: "f32[256, 256]" = torch.ops.aten.permute.default(primals_186, [1, 0]);  primals_186 = None
    addmm_68: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_187, view_244, permute_123);  primals_187 = None
    view_247: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_68, [1, 512, 256]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_248: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_247, [1, 512, 4, 64]);  view_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_124: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_248, [0, 2, 1, 3]);  view_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_125: "f32[256, 256]" = torch.ops.aten.permute.default(primals_188, [1, 0]);  primals_188 = None
    addmm_69: "f32[512, 256]" = torch.ops.aten.addmm.default(primals_189, view_244, permute_125);  primals_189 = None
    view_250: "f32[1, 512, 256]" = torch.ops.aten.view.default(addmm_69, [1, 512, 256]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_251: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_250, [1, 512, 4, 64]);  view_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_126: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_251, [0, 2, 1, 3]);  view_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_252: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_245, [1, 512, 4, 64]);  view_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_127: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None
    
    # No stacktrace found for following nodes
    clone_default: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_127, memory_format = torch.contiguous_format);  permute_127 = None
    clone_default_1: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_124, memory_format = torch.contiguous_format);  permute_124 = None
    clone_default_2: "f32[1, 4, 512, 64]" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
    mul_scalar: "f32[1, 4, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default, 0.3535533905932738);  clone_default = None
    permute_default: "f32[1, 4, 64, 512]" = torch.ops.aten.permute.default(clone_default_1, [0, 1, 3, 2]);  clone_default_1 = None
    mul_scalar_1: "f32[1, 4, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default, 0.3535533905932738);  permute_default = None
    expand_default: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(mul_scalar, [1, 4, 512, 64]);  mul_scalar = None
    view_default: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_default, [4, 512, 64]);  expand_default = None
    expand_default_1: "f32[1, 4, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_1, [1, 4, 64, 512]);  mul_scalar_1 = None
    view_default_1: "f32[4, 64, 512]" = torch.ops.aten.view.default(expand_default_1, [4, 64, 512]);  expand_default_1 = None
    bmm_default: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_default, view_default_1)
    view_default_2: "f32[1, 4, 512, 512]" = torch.ops.aten.view.default(bmm_default, [1, 4, 512, 512]);  bmm_default = None
    amax_default: "f32[1, 4, 512, 1]" = torch.ops.aten.amax.default(view_default_2, [-1], True)
    sub_tensor: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_2, amax_default);  view_default_2 = amax_default = None
    exp_default: "f32[1, 4, 512, 512]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
    sum_dim_int_list: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
    div_tensor: "f32[1, 4, 512, 512]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
    alias_default: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(div_tensor)
    native_dropout_default = torch.ops.aten.native_dropout.default(div_tensor, 0.1, True);  div_tensor = None
    getitem_126: "f32[1, 4, 512, 512]" = native_dropout_default[0]
    getitem_127: "b8[1, 4, 512, 512]" = native_dropout_default[1];  native_dropout_default = None
    expand_default_2: "f32[1, 4, 512, 512]" = torch.ops.aten.expand.default(getitem_126, [1, 4, 512, 512]);  getitem_126 = None
    view_default_3: "f32[4, 512, 512]" = torch.ops.aten.view.default(expand_default_2, [4, 512, 512]);  expand_default_2 = None
    expand_default_3: "f32[1, 4, 512, 64]" = torch.ops.aten.expand.default(clone_default_2, [1, 4, 512, 64]);  clone_default_2 = None
    view_default_4: "f32[4, 512, 64]" = torch.ops.aten.view.default(expand_default_3, [4, 512, 64]);  expand_default_3 = None
    bmm_default_1: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_default_3, view_default_4)
    view_default_5: "f32[1, 4, 512, 64]" = torch.ops.aten.view.default(bmm_default_1, [1, 4, 512, 64]);  bmm_default_1 = None
    permute_default_1: "f32[4, 512, 512]" = torch.ops.aten.permute.default(view_default_3, [0, 2, 1]);  view_default_3 = None
    permute_default_2: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_default_4, [0, 2, 1]);  view_default_4 = None
    alias_default_1: "f32[1, 4, 512, 512]" = torch.ops.aten.alias.default(alias_default);  alias_default = None
    permute_default_3: "f32[4, 64, 512]" = torch.ops.aten.permute.default(view_default, [0, 2, 1]);  view_default = None
    permute_default_4: "f32[4, 512, 64]" = torch.ops.aten.permute.default(view_default_1, [0, 2, 1]);  view_default_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_129: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_default_5, [0, 2, 1, 3]);  view_default_5 = None
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
    sub_36: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_93, getitem_119);  add_93 = getitem_119 = None
    mul_80: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_23);  sub_36 = None
    mul_81: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_80, primals_192)
    add_95: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_81, primals_193);  mul_81 = primals_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_262: "f32[512, 256]" = torch.ops.aten.view.default(add_95, [512, 256])
    permute_131: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_194, [1, 0]);  primals_194 = None
    addmm_71: "f32[512, 1024]" = torch.ops.aten.addmm.default(primals_195, view_262, permute_131);  primals_195 = None
    view_263: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_71, [1, 512, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_82: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_263, 0.5)
    mul_83: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_263, 0.7071067811865476);  view_263 = None
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
    sub_37: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(add_97, getitem_123);  add_97 = getitem_123 = None
    mul_85: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_24);  sub_37 = None
    mul_86: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_85, primals_198)
    add_99: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_86, primals_199);  mul_86 = primals_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:660, code: hidden_states = self.dense(generator_hidden_states)
    view_266: "f32[512, 256]" = torch.ops.aten.view.default(add_99, [512, 256]);  add_99 = None
    permute_133: "f32[256, 128]" = torch.ops.aten.permute.default(primals_200, [1, 0]);  primals_200 = None
    addmm_73: "f32[512, 128]" = torch.ops.aten.addmm.default(primals_201, view_266, permute_133);  primals_201 = None
    view_267: "f32[1, 512, 128]" = torch.ops.aten.view.default(addmm_73, [1, 512, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_87: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(view_267, 0.5)
    mul_88: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(view_267, 0.7071067811865476);  view_267 = None
    erf_12: "f32[1, 512, 128]" = torch.ops.aten.erf.default(mul_88);  mul_88 = None
    add_100: "f32[1, 512, 128]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_89: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_87, add_100);  mul_87 = add_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:662, code: hidden_states = self.LayerNorm(hidden_states)
    var_mean_25 = torch.ops.aten.var_mean.correction(mul_89, [2], correction = 0, keepdim = True)
    getitem_124: "f32[1, 512, 1]" = var_mean_25[0]
    getitem_125: "f32[1, 512, 1]" = var_mean_25[1];  var_mean_25 = None
    add_101: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_124, 1e-12);  getitem_124 = None
    rsqrt_25: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
    sub_38: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(mul_89, getitem_125);  mul_89 = getitem_125 = None
    mul_90: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_25);  sub_38 = None
    mul_91: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_90, primals_202)
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
    ne: "b8[511]" = torch.ops.aten.ne.Scalar(view_271, -100)
    full_default_1: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where: "i64[511]" = torch.ops.aten.where.self(ne, view_271, full_default_1)
    unsqueeze_2: "i64[511, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
    gather: "f32[511, 1]" = torch.ops.aten.gather.default(sub_40, 1, unsqueeze_2);  unsqueeze_2 = None
    squeeze: "f32[511]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[511]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
    full_default_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where_1: "f32[511]" = torch.ops.aten.where.self(ne, neg, full_default_2);  neg = full_default_2 = None
    sum_14: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None
    sum_15: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
    div_24: "f32[]" = torch.ops.aten.div.Tensor(sum_15, convert_element_type);  sum_15 = None
    unsqueeze_3: "i64[511, 1]" = torch.ops.aten.unsqueeze.default(view_271, 1);  view_271 = None
    ne_3: "b8[511, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_3, -100)
    where_2: "i64[511, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_3, full_default_1);  unsqueeze_3 = full_default_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1640, code: prediction_scores = self.generator_lm_head(self.generator_predictions(sequence_output))
    permute_135: "f32[30522, 128]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:662, code: hidden_states = self.LayerNorm(hidden_states)
    div_26: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 128);  rsqrt_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:660, code: hidden_states = self.dense(generator_hidden_states)
    permute_139: "f32[128, 256]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_27: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 256);  rsqrt_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    permute_143: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    permute_147: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_28: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 256);  rsqrt_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    permute_151: "f32[256, 256]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_163: "f32[256, 256]" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_168: "f32[256, 256]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    permute_172: "f32[256, 256]" = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_30: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 256);  rsqrt_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    permute_176: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    permute_180: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_31: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 256);  rsqrt_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    permute_184: "f32[256, 256]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_196: "f32[256, 256]" = torch.ops.aten.permute.default(permute_114, [1, 0]);  permute_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_201: "f32[256, 256]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    permute_205: "f32[256, 256]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_33: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 256);  rsqrt_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    permute_209: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    permute_213: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_34: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 256);  rsqrt_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    permute_217: "f32[256, 256]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_229: "f32[256, 256]" = torch.ops.aten.permute.default(permute_103, [1, 0]);  permute_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_234: "f32[256, 256]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    permute_238: "f32[256, 256]" = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_36: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 256);  rsqrt_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    permute_242: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    permute_246: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_37: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 256);  rsqrt_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    permute_250: "f32[256, 256]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_262: "f32[256, 256]" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_267: "f32[256, 256]" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    permute_271: "f32[256, 256]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_39: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 256);  rsqrt_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    permute_275: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    permute_279: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_40: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 256);  rsqrt_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    permute_283: "f32[256, 256]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_295: "f32[256, 256]" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_300: "f32[256, 256]" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    permute_304: "f32[256, 256]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_42: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 256);  rsqrt_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    permute_308: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    permute_312: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_43: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 256);  rsqrt_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    permute_316: "f32[256, 256]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_328: "f32[256, 256]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_333: "f32[256, 256]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    permute_337: "f32[256, 256]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_45: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 256);  rsqrt_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    permute_341: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    permute_345: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_46: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 256);  rsqrt_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    permute_349: "f32[256, 256]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_361: "f32[256, 256]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_366: "f32[256, 256]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    permute_370: "f32[256, 256]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_48: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 256);  rsqrt_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    permute_374: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    permute_378: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_49: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 256);  rsqrt_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    permute_382: "f32[256, 256]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_394: "f32[256, 256]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_399: "f32[256, 256]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    permute_403: "f32[256, 256]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_51: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 256);  rsqrt_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    permute_407: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    permute_411: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_52: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 256);  rsqrt_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    permute_415: "f32[256, 256]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_427: "f32[256, 256]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_432: "f32[256, 256]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    permute_436: "f32[256, 256]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_54: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 256);  rsqrt_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    permute_440: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    permute_444: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_55: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 256);  rsqrt_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    permute_448: "f32[256, 256]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_460: "f32[256, 256]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_465: "f32[256, 256]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    permute_469: "f32[256, 256]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_57: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 256);  rsqrt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    permute_473: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    permute_477: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_58: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 256);  rsqrt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    permute_481: "f32[256, 256]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_493: "f32[256, 256]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_498: "f32[256, 256]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    permute_502: "f32[256, 256]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_60: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 256);  rsqrt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    permute_506: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    permute_510: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_61: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 256);  rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    permute_514: "f32[256, 256]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_526: "f32[256, 256]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_531: "f32[256, 256]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    permute_535: "f32[256, 256]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:918, code: hidden_states = self.embeddings_project(hidden_states)
    permute_539: "f32[256, 128]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:210, code: embeddings = self.LayerNorm(embeddings)
    div_63: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 128);  rsqrt = None
    return [div_24, view_269, primals_4, primals_16, primals_22, primals_32, primals_38, primals_48, primals_54, primals_64, primals_70, primals_80, primals_86, primals_96, primals_102, primals_112, primals_118, primals_128, primals_134, primals_144, primals_150, primals_160, primals_166, primals_176, primals_182, primals_192, primals_198, primals_202, primals_209, expand, slice_4, mul_1, getitem_3, view, view_2, getitem_149, permute_default_67, permute_default_68, alias_default_23, permute_default_69, permute_default_70, view_18, getitem_7, mul_3, view_20, addmm_5, view_22, getitem_11, mul_8, view_24, getitem_147, permute_default_61, permute_default_62, alias_default_21, permute_default_63, permute_default_64, view_40, getitem_17, mul_10, view_42, addmm_11, view_44, getitem_21, mul_15, view_46, getitem_145, permute_default_55, permute_default_56, alias_default_19, permute_default_57, permute_default_58, view_62, getitem_27, mul_17, view_64, addmm_17, view_66, getitem_31, mul_22, view_68, getitem_143, permute_default_49, permute_default_50, alias_default_17, permute_default_51, permute_default_52, view_84, getitem_37, mul_24, view_86, addmm_23, view_88, getitem_41, mul_29, view_90, getitem_141, permute_default_43, permute_default_44, alias_default_15, permute_default_45, permute_default_46, view_106, getitem_47, mul_31, view_108, addmm_29, view_110, getitem_51, mul_36, view_112, getitem_139, permute_default_37, permute_default_38, alias_default_13, permute_default_39, permute_default_40, view_128, getitem_57, mul_38, view_130, addmm_35, view_132, getitem_61, mul_43, view_134, getitem_137, permute_default_31, permute_default_32, alias_default_11, permute_default_33, permute_default_34, view_150, getitem_67, mul_45, view_152, addmm_41, view_154, getitem_71, mul_50, view_156, getitem_135, permute_default_25, permute_default_26, alias_default_9, permute_default_27, permute_default_28, view_172, getitem_77, mul_52, view_174, addmm_47, view_176, getitem_81, mul_57, view_178, getitem_133, permute_default_19, permute_default_20, alias_default_7, permute_default_21, permute_default_22, view_194, getitem_87, mul_59, view_196, addmm_53, view_198, getitem_91, mul_64, view_200, getitem_131, permute_default_13, permute_default_14, alias_default_5, permute_default_15, permute_default_16, view_216, getitem_97, mul_66, view_218, addmm_59, view_220, getitem_101, mul_71, view_222, getitem_129, permute_default_7, permute_default_8, alias_default_3, permute_default_9, permute_default_10, view_238, getitem_107, mul_73, view_240, addmm_65, view_242, getitem_111, mul_78, view_244, getitem_127, permute_default_1, permute_default_2, alias_default_1, permute_default_3, permute_default_4, view_260, getitem_117, mul_80, view_262, addmm_71, view_264, getitem_121, mul_85, view_266, addmm_73, mul_90, view_268, sub_40, convert_element_type, ne_3, where_2, permute_135, div_26, permute_139, div_27, permute_143, permute_147, div_28, permute_151, permute_163, permute_168, permute_172, div_30, permute_176, permute_180, div_31, permute_184, permute_196, permute_201, permute_205, div_33, permute_209, permute_213, div_34, permute_217, permute_229, permute_234, permute_238, div_36, permute_242, permute_246, div_37, permute_250, permute_262, permute_267, permute_271, div_39, permute_275, permute_279, div_40, permute_283, permute_295, permute_300, permute_304, div_42, permute_308, permute_312, div_43, permute_316, permute_328, permute_333, permute_337, div_45, permute_341, permute_345, div_46, permute_349, permute_361, permute_366, permute_370, div_48, permute_374, permute_378, div_49, permute_382, permute_394, permute_399, permute_403, div_51, permute_407, permute_411, div_52, permute_415, permute_427, permute_432, permute_436, div_54, permute_440, permute_444, div_55, permute_448, permute_460, permute_465, permute_469, div_57, permute_473, permute_477, div_58, permute_481, permute_493, permute_498, permute_502, div_60, permute_506, permute_510, div_61, permute_514, permute_526, permute_531, permute_535, permute_539, div_63]
    