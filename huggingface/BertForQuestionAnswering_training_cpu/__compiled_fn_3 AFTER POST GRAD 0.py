from __future__ import annotations



def forward(self, primals_1: "f32[30522, 768]", primals_2: "f32[2, 768]", primals_3: "f32[512, 768]", primals_4: "f32[768]", primals_5: "f32[768]", primals_6: "f32[768, 768]", primals_7: "f32[768]", primals_8: "f32[768, 768]", primals_9: "f32[768]", primals_10: "f32[768, 768]", primals_11: "f32[768]", primals_12: "f32[768, 768]", primals_13: "f32[768]", primals_14: "f32[768]", primals_15: "f32[768]", primals_16: "f32[3072, 768]", primals_17: "f32[3072]", primals_18: "f32[768, 3072]", primals_19: "f32[768]", primals_20: "f32[768]", primals_21: "f32[768]", primals_22: "f32[768, 768]", primals_23: "f32[768]", primals_24: "f32[768, 768]", primals_25: "f32[768]", primals_26: "f32[768, 768]", primals_27: "f32[768]", primals_28: "f32[768, 768]", primals_29: "f32[768]", primals_30: "f32[768]", primals_31: "f32[768]", primals_32: "f32[3072, 768]", primals_33: "f32[3072]", primals_34: "f32[768, 3072]", primals_35: "f32[768]", primals_36: "f32[768]", primals_37: "f32[768]", primals_38: "f32[768, 768]", primals_39: "f32[768]", primals_40: "f32[768, 768]", primals_41: "f32[768]", primals_42: "f32[768, 768]", primals_43: "f32[768]", primals_44: "f32[768, 768]", primals_45: "f32[768]", primals_46: "f32[768]", primals_47: "f32[768]", primals_48: "f32[3072, 768]", primals_49: "f32[3072]", primals_50: "f32[768, 3072]", primals_51: "f32[768]", primals_52: "f32[768]", primals_53: "f32[768]", primals_54: "f32[768, 768]", primals_55: "f32[768]", primals_56: "f32[768, 768]", primals_57: "f32[768]", primals_58: "f32[768, 768]", primals_59: "f32[768]", primals_60: "f32[768, 768]", primals_61: "f32[768]", primals_62: "f32[768]", primals_63: "f32[768]", primals_64: "f32[3072, 768]", primals_65: "f32[3072]", primals_66: "f32[768, 3072]", primals_67: "f32[768]", primals_68: "f32[768]", primals_69: "f32[768]", primals_70: "f32[768, 768]", primals_71: "f32[768]", primals_72: "f32[768, 768]", primals_73: "f32[768]", primals_74: "f32[768, 768]", primals_75: "f32[768]", primals_76: "f32[768, 768]", primals_77: "f32[768]", primals_78: "f32[768]", primals_79: "f32[768]", primals_80: "f32[3072, 768]", primals_81: "f32[3072]", primals_82: "f32[768, 3072]", primals_83: "f32[768]", primals_84: "f32[768]", primals_85: "f32[768]", primals_86: "f32[768, 768]", primals_87: "f32[768]", primals_88: "f32[768, 768]", primals_89: "f32[768]", primals_90: "f32[768, 768]", primals_91: "f32[768]", primals_92: "f32[768, 768]", primals_93: "f32[768]", primals_94: "f32[768]", primals_95: "f32[768]", primals_96: "f32[3072, 768]", primals_97: "f32[3072]", primals_98: "f32[768, 3072]", primals_99: "f32[768]", primals_100: "f32[768]", primals_101: "f32[768]", primals_102: "f32[768, 768]", primals_103: "f32[768]", primals_104: "f32[768, 768]", primals_105: "f32[768]", primals_106: "f32[768, 768]", primals_107: "f32[768]", primals_108: "f32[768, 768]", primals_109: "f32[768]", primals_110: "f32[768]", primals_111: "f32[768]", primals_112: "f32[3072, 768]", primals_113: "f32[3072]", primals_114: "f32[768, 3072]", primals_115: "f32[768]", primals_116: "f32[768]", primals_117: "f32[768]", primals_118: "f32[768, 768]", primals_119: "f32[768]", primals_120: "f32[768, 768]", primals_121: "f32[768]", primals_122: "f32[768, 768]", primals_123: "f32[768]", primals_124: "f32[768, 768]", primals_125: "f32[768]", primals_126: "f32[768]", primals_127: "f32[768]", primals_128: "f32[3072, 768]", primals_129: "f32[3072]", primals_130: "f32[768, 3072]", primals_131: "f32[768]", primals_132: "f32[768]", primals_133: "f32[768]", primals_134: "f32[768, 768]", primals_135: "f32[768]", primals_136: "f32[768, 768]", primals_137: "f32[768]", primals_138: "f32[768, 768]", primals_139: "f32[768]", primals_140: "f32[768, 768]", primals_141: "f32[768]", primals_142: "f32[768]", primals_143: "f32[768]", primals_144: "f32[3072, 768]", primals_145: "f32[3072]", primals_146: "f32[768, 3072]", primals_147: "f32[768]", primals_148: "f32[768]", primals_149: "f32[768]", primals_150: "f32[768, 768]", primals_151: "f32[768]", primals_152: "f32[768, 768]", primals_153: "f32[768]", primals_154: "f32[768, 768]", primals_155: "f32[768]", primals_156: "f32[768, 768]", primals_157: "f32[768]", primals_158: "f32[768]", primals_159: "f32[768]", primals_160: "f32[3072, 768]", primals_161: "f32[3072]", primals_162: "f32[768, 3072]", primals_163: "f32[768]", primals_164: "f32[768]", primals_165: "f32[768]", primals_166: "f32[768, 768]", primals_167: "f32[768]", primals_168: "f32[768, 768]", primals_169: "f32[768]", primals_170: "f32[768, 768]", primals_171: "f32[768]", primals_172: "f32[768, 768]", primals_173: "f32[768]", primals_174: "f32[768]", primals_175: "f32[768]", primals_176: "f32[3072, 768]", primals_177: "f32[3072]", primals_178: "f32[768, 3072]", primals_179: "f32[768]", primals_180: "f32[768]", primals_181: "f32[768]", primals_182: "f32[768, 768]", primals_183: "f32[768]", primals_184: "f32[768, 768]", primals_185: "f32[768]", primals_186: "f32[768, 768]", primals_187: "f32[768]", primals_188: "f32[768, 768]", primals_189: "f32[768]", primals_190: "f32[768]", primals_191: "f32[768]", primals_192: "f32[3072, 768]", primals_193: "f32[3072]", primals_194: "f32[768, 3072]", primals_195: "f32[768]", primals_196: "f32[768]", primals_197: "f32[768]", primals_198: "f32[2, 768]", primals_199: "f32[2]", primals_200: "i64[1, 512]", primals_201: "i64[1, 512]", primals_202: "i64[1, 512]", primals_203: "i64[1]", primals_204: "i64[1]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:987, code: buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
    slice_1: "i64[1, 512]" = torch.ops.aten.slice.Tensor(primals_200, 0, 0, 9223372036854775807);  primals_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:988, code: buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
    expand: "i64[1, 512]" = torch.ops.aten.expand.default(slice_1, [1, 512]);  slice_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:218, code: position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
    slice_4: "i64[1, 512]" = torch.ops.aten.slice.Tensor(primals_201, 0, 0, 9223372036854775807);  primals_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:232, code: inputs_embeds = self.word_embeddings(input_ids)
    embedding: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(primals_1, primals_202, 0);  primals_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:233, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    embedding_1: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(primals_2, expand);  primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:235, code: embeddings = inputs_embeds + token_type_embeddings
    add: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:237, code: position_embeddings = self.position_embeddings(position_ids)
    embedding_2: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(primals_3, slice_4);  primals_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:238, code: embeddings += position_embeddings
    add_1: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:239, code: embeddings = self.LayerNorm(embeddings)
    var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 512, 1]" = var_mean[0]
    getitem_1: "f32[1, 512, 1]" = var_mean[1];  var_mean = None
    add_2: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
    rsqrt: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
    sub_1: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
    mul_1: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = None
    mul_2: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_1, primals_4)
    add_3: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_2, primals_5);  mul_2 = primals_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:240, code: embeddings = self.dropout(embeddings)
    native_dropout = torch.ops.aten.native_dropout.default(add_3, 0.1, True);  add_3 = None
    getitem_2: "f32[1, 512, 768]" = native_dropout[0]
    getitem_3: "b8[1, 512, 768]" = native_dropout[1];  native_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view: "f32[512, 768]" = torch.ops.aten.reshape.default(getitem_2, [512, 768])
    permute: "f32[768, 768]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
    addmm: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_7, view, permute);  primals_7 = None
    view_1: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm, [1, 512, 768]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_1: "f32[768, 768]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
    addmm_1: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_9, view, permute_1);  primals_9 = None
    view_3: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_1, [1, 512, 768]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_4: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_3, [1, 512, 12, 64]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_2: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_3: "f32[768, 768]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
    addmm_2: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_11, view, permute_3);  primals_11 = None
    view_6: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_2, [1, 512, 768]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_7: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_6, [1, 512, 12, 64]);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_4: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_8: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_1, [1, 512, 12, 64]);  view_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_5: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
    
    # No stacktrace found for following nodes
    clone_default_44: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    clone_default_45: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
    clone_default_46: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    mul_scalar_44: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default_44, 0.3535533905932738);  clone_default_44 = None
    permute_default_66: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(clone_default_45, [0, 1, 3, 2]);  clone_default_45 = None
    mul_scalar_45: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_66, 0.3535533905932738);  permute_default_66 = None
    expand_default_44: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(mul_scalar_44, [1, 12, 512, 64]);  mul_scalar_44 = None
    view_default_132: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_default_44, [12, 512, 64]);  expand_default_44 = None
    expand_default_45: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_45, [1, 12, 64, 512]);  mul_scalar_45 = None
    view_default_133: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_default_45, [12, 64, 512]);  expand_default_45 = None
    bmm_default_66: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_default_132, view_default_133)
    view_default_134: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_66, [1, 12, 512, 512]);  bmm_default_66 = None
    amax_default_11: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(view_default_134, [-1], True)
    sub_tensor_22: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_134, amax_default_11);  view_default_134 = amax_default_11 = None
    exp_default_11: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_tensor_22);  sub_tensor_22 = None
    sum_dim_int_list_22: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_11, [-1], True)
    div_tensor_11: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_default_11, sum_dim_int_list_22);  exp_default_11 = sum_dim_int_list_22 = None
    alias_default_22: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(div_tensor_11)
    native_dropout_default_11 = torch.ops.aten.native_dropout.default(div_tensor_11, 0.1, True);  div_tensor_11 = None
    getitem_148: "f32[1, 12, 512, 512]" = native_dropout_default_11[0]
    getitem_149: "b8[1, 12, 512, 512]" = native_dropout_default_11[1];  native_dropout_default_11 = None
    expand_default_46: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(getitem_148, [1, 12, 512, 512]);  getitem_148 = None
    view_default_135: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_default_46, [12, 512, 512]);  expand_default_46 = None
    expand_default_47: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(clone_default_46, [1, 12, 512, 64]);  clone_default_46 = None
    view_default_136: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_default_47, [12, 512, 64]);  expand_default_47 = None
    bmm_default_67: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_default_135, view_default_136)
    view_default_137: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_67, [1, 12, 512, 64]);  bmm_default_67 = None
    permute_default_67: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_default_135, [0, 2, 1]);  view_default_135 = None
    permute_default_68: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_default_136, [0, 2, 1]);  view_default_136 = None
    alias_default_23: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_default_22);  alias_default_22 = None
    permute_default_69: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_default_132, [0, 2, 1]);  view_default_132 = None
    permute_default_70: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_default_133, [0, 2, 1]);  view_default_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_7: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_default_137, [0, 2, 1, 3]);  view_default_137 = None
    clone: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_15: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone, [1, 512, 768]);  clone = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_16: "f32[512, 768]" = torch.ops.aten.reshape.default(view_15, [512, 768]);  view_15 = None
    permute_8: "f32[768, 768]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
    addmm_3: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_13, view_16, permute_8);  primals_13 = None
    view_17: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_3, [1, 512, 768]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:387, code: hidden_states = self.dropout(hidden_states)
    native_dropout_2 = torch.ops.aten.native_dropout.default(view_17, 0.1, True);  view_17 = None
    getitem_6: "f32[1, 512, 768]" = native_dropout_2[0]
    getitem_7: "b8[1, 512, 768]" = native_dropout_2[1];  native_dropout_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_5: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_6, getitem_2);  getitem_6 = getitem_2 = None
    var_mean_1 = torch.ops.aten.var_mean.correction(add_5, [2], correction = 0, keepdim = True)
    getitem_8: "f32[1, 512, 1]" = var_mean_1[0]
    getitem_9: "f32[1, 512, 1]" = var_mean_1[1];  var_mean_1 = None
    add_6: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
    rsqrt_1: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_3: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_5, getitem_9);  add_5 = getitem_9 = None
    mul_3: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = None
    mul_4: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_3, primals_14)
    add_7: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_4, primals_15);  mul_4 = primals_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_18: "f32[512, 768]" = torch.ops.aten.reshape.default(add_7, [512, 768])
    permute_9: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
    addmm_4: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_17, view_18, permute_9);  primals_17 = None
    view_19: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_4, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_5: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.5)
    mul_6: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476);  view_19 = None
    erf: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
    add_8: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_7: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_5, add_8);  mul_5 = add_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_20: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_7, [512, 3072]);  mul_7 = None
    permute_10: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
    addmm_5: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_19, view_20, permute_10);  primals_19 = None
    view_21: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_5, [1, 512, 768]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:465, code: hidden_states = self.dropout(hidden_states)
    native_dropout_3 = torch.ops.aten.native_dropout.default(view_21, 0.1, True);  view_21 = None
    getitem_10: "f32[1, 512, 768]" = native_dropout_3[0]
    getitem_11: "b8[1, 512, 768]" = native_dropout_3[1];  native_dropout_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_9: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_10, add_7);  getitem_10 = add_7 = None
    var_mean_2 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
    getitem_12: "f32[1, 512, 1]" = var_mean_2[0]
    getitem_13: "f32[1, 512, 1]" = var_mean_2[1];  var_mean_2 = None
    add_10: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
    rsqrt_2: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
    sub_4: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_9, getitem_13);  add_9 = getitem_13 = None
    mul_8: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = None
    mul_9: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_8, primals_20)
    add_11: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_9, primals_21);  mul_9 = primals_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_22: "f32[512, 768]" = torch.ops.aten.reshape.default(add_11, [512, 768])
    permute_11: "f32[768, 768]" = torch.ops.aten.permute.default(primals_22, [1, 0]);  primals_22 = None
    addmm_6: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_23, view_22, permute_11);  primals_23 = None
    view_23: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_6, [1, 512, 768]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_12: "f32[768, 768]" = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
    addmm_7: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_25, view_22, permute_12);  primals_25 = None
    view_25: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_7, [1, 512, 768]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_26: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_25, [1, 512, 12, 64]);  view_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_13: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_14: "f32[768, 768]" = torch.ops.aten.permute.default(primals_26, [1, 0]);  primals_26 = None
    addmm_8: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_27, view_22, permute_14);  primals_27 = None
    view_28: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_8, [1, 512, 768]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_29: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_28, [1, 512, 12, 64]);  view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_15: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_30: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_23, [1, 512, 12, 64]);  view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_16: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
    
    # No stacktrace found for following nodes
    clone_default_40: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
    clone_default_41: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
    clone_default_42: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
    mul_scalar_40: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default_40, 0.3535533905932738);  clone_default_40 = None
    permute_default_60: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(clone_default_41, [0, 1, 3, 2]);  clone_default_41 = None
    mul_scalar_41: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_60, 0.3535533905932738);  permute_default_60 = None
    expand_default_40: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(mul_scalar_40, [1, 12, 512, 64]);  mul_scalar_40 = None
    view_default_120: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_default_40, [12, 512, 64]);  expand_default_40 = None
    expand_default_41: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_41, [1, 12, 64, 512]);  mul_scalar_41 = None
    view_default_121: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_default_41, [12, 64, 512]);  expand_default_41 = None
    bmm_default_60: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_default_120, view_default_121)
    view_default_122: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_60, [1, 12, 512, 512]);  bmm_default_60 = None
    amax_default_10: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(view_default_122, [-1], True)
    sub_tensor_20: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_122, amax_default_10);  view_default_122 = amax_default_10 = None
    exp_default_10: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_tensor_20);  sub_tensor_20 = None
    sum_dim_int_list_20: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_10, [-1], True)
    div_tensor_10: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_default_10, sum_dim_int_list_20);  exp_default_10 = sum_dim_int_list_20 = None
    alias_default_20: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(div_tensor_10)
    native_dropout_default_10 = torch.ops.aten.native_dropout.default(div_tensor_10, 0.1, True);  div_tensor_10 = None
    getitem_146: "f32[1, 12, 512, 512]" = native_dropout_default_10[0]
    getitem_147: "b8[1, 12, 512, 512]" = native_dropout_default_10[1];  native_dropout_default_10 = None
    expand_default_42: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(getitem_146, [1, 12, 512, 512]);  getitem_146 = None
    view_default_123: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_default_42, [12, 512, 512]);  expand_default_42 = None
    expand_default_43: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(clone_default_42, [1, 12, 512, 64]);  clone_default_42 = None
    view_default_124: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_default_43, [12, 512, 64]);  expand_default_43 = None
    bmm_default_61: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_default_123, view_default_124)
    view_default_125: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_61, [1, 12, 512, 64]);  bmm_default_61 = None
    permute_default_61: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_default_123, [0, 2, 1]);  view_default_123 = None
    permute_default_62: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_default_124, [0, 2, 1]);  view_default_124 = None
    alias_default_21: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_default_20);  alias_default_20 = None
    permute_default_63: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_default_120, [0, 2, 1]);  view_default_120 = None
    permute_default_64: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_default_121, [0, 2, 1]);  view_default_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_18: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_default_125, [0, 2, 1, 3]);  view_default_125 = None
    clone_1: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_37: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_1, [1, 512, 768]);  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_38: "f32[512, 768]" = torch.ops.aten.reshape.default(view_37, [512, 768]);  view_37 = None
    permute_19: "f32[768, 768]" = torch.ops.aten.permute.default(primals_28, [1, 0]);  primals_28 = None
    addmm_9: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_29, view_38, permute_19);  primals_29 = None
    view_39: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_9, [1, 512, 768]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:387, code: hidden_states = self.dropout(hidden_states)
    native_dropout_5 = torch.ops.aten.native_dropout.default(view_39, 0.1, True);  view_39 = None
    getitem_16: "f32[1, 512, 768]" = native_dropout_5[0]
    getitem_17: "b8[1, 512, 768]" = native_dropout_5[1];  native_dropout_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_13: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_16, add_11);  getitem_16 = add_11 = None
    var_mean_3 = torch.ops.aten.var_mean.correction(add_13, [2], correction = 0, keepdim = True)
    getitem_18: "f32[1, 512, 1]" = var_mean_3[0]
    getitem_19: "f32[1, 512, 1]" = var_mean_3[1];  var_mean_3 = None
    add_14: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
    rsqrt_3: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
    sub_6: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_13, getitem_19);  add_13 = getitem_19 = None
    mul_10: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_3);  sub_6 = None
    mul_11: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_10, primals_30)
    add_15: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_11, primals_31);  mul_11 = primals_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_40: "f32[512, 768]" = torch.ops.aten.reshape.default(add_15, [512, 768])
    permute_20: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_32, [1, 0]);  primals_32 = None
    addmm_10: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_33, view_40, permute_20);  primals_33 = None
    view_41: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_10, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_12: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, 0.5)
    mul_13: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476);  view_41 = None
    erf_1: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
    add_16: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_14: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_12, add_16);  mul_12 = add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_42: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_14, [512, 3072]);  mul_14 = None
    permute_21: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
    addmm_11: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_35, view_42, permute_21);  primals_35 = None
    view_43: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_11, [1, 512, 768]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:465, code: hidden_states = self.dropout(hidden_states)
    native_dropout_6 = torch.ops.aten.native_dropout.default(view_43, 0.1, True);  view_43 = None
    getitem_20: "f32[1, 512, 768]" = native_dropout_6[0]
    getitem_21: "b8[1, 512, 768]" = native_dropout_6[1];  native_dropout_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_17: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_20, add_15);  getitem_20 = add_15 = None
    var_mean_4 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
    getitem_22: "f32[1, 512, 1]" = var_mean_4[0]
    getitem_23: "f32[1, 512, 1]" = var_mean_4[1];  var_mean_4 = None
    add_18: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
    rsqrt_4: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    sub_7: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_17, getitem_23);  add_17 = getitem_23 = None
    mul_15: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_4);  sub_7 = None
    mul_16: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_15, primals_36)
    add_19: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_16, primals_37);  mul_16 = primals_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_44: "f32[512, 768]" = torch.ops.aten.reshape.default(add_19, [512, 768])
    permute_22: "f32[768, 768]" = torch.ops.aten.permute.default(primals_38, [1, 0]);  primals_38 = None
    addmm_12: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_39, view_44, permute_22);  primals_39 = None
    view_45: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_12, [1, 512, 768]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_23: "f32[768, 768]" = torch.ops.aten.permute.default(primals_40, [1, 0]);  primals_40 = None
    addmm_13: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_41, view_44, permute_23);  primals_41 = None
    view_47: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_13, [1, 512, 768]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_48: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_47, [1, 512, 12, 64]);  view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_24: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_25: "f32[768, 768]" = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
    addmm_14: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_43, view_44, permute_25);  primals_43 = None
    view_50: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_14, [1, 512, 768]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_51: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_50, [1, 512, 12, 64]);  view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_26: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_52: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_45, [1, 512, 12, 64]);  view_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_27: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
    
    # No stacktrace found for following nodes
    clone_default_36: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
    clone_default_37: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
    clone_default_38: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
    mul_scalar_36: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default_36, 0.3535533905932738);  clone_default_36 = None
    permute_default_54: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(clone_default_37, [0, 1, 3, 2]);  clone_default_37 = None
    mul_scalar_37: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_54, 0.3535533905932738);  permute_default_54 = None
    expand_default_36: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(mul_scalar_36, [1, 12, 512, 64]);  mul_scalar_36 = None
    view_default_108: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_default_36, [12, 512, 64]);  expand_default_36 = None
    expand_default_37: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_37, [1, 12, 64, 512]);  mul_scalar_37 = None
    view_default_109: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_default_37, [12, 64, 512]);  expand_default_37 = None
    bmm_default_54: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_default_108, view_default_109)
    view_default_110: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_54, [1, 12, 512, 512]);  bmm_default_54 = None
    amax_default_9: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(view_default_110, [-1], True)
    sub_tensor_18: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_110, amax_default_9);  view_default_110 = amax_default_9 = None
    exp_default_9: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_tensor_18);  sub_tensor_18 = None
    sum_dim_int_list_18: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_9, [-1], True)
    div_tensor_9: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_default_9, sum_dim_int_list_18);  exp_default_9 = sum_dim_int_list_18 = None
    alias_default_18: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(div_tensor_9)
    native_dropout_default_9 = torch.ops.aten.native_dropout.default(div_tensor_9, 0.1, True);  div_tensor_9 = None
    getitem_144: "f32[1, 12, 512, 512]" = native_dropout_default_9[0]
    getitem_145: "b8[1, 12, 512, 512]" = native_dropout_default_9[1];  native_dropout_default_9 = None
    expand_default_38: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(getitem_144, [1, 12, 512, 512]);  getitem_144 = None
    view_default_111: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_default_38, [12, 512, 512]);  expand_default_38 = None
    expand_default_39: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(clone_default_38, [1, 12, 512, 64]);  clone_default_38 = None
    view_default_112: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_default_39, [12, 512, 64]);  expand_default_39 = None
    bmm_default_55: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_default_111, view_default_112)
    view_default_113: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_55, [1, 12, 512, 64]);  bmm_default_55 = None
    permute_default_55: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_default_111, [0, 2, 1]);  view_default_111 = None
    permute_default_56: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_default_112, [0, 2, 1]);  view_default_112 = None
    alias_default_19: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_default_18);  alias_default_18 = None
    permute_default_57: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_default_108, [0, 2, 1]);  view_default_108 = None
    permute_default_58: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_default_109, [0, 2, 1]);  view_default_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_29: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_default_113, [0, 2, 1, 3]);  view_default_113 = None
    clone_2: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_59: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_2, [1, 512, 768]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_60: "f32[512, 768]" = torch.ops.aten.reshape.default(view_59, [512, 768]);  view_59 = None
    permute_30: "f32[768, 768]" = torch.ops.aten.permute.default(primals_44, [1, 0]);  primals_44 = None
    addmm_15: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_45, view_60, permute_30);  primals_45 = None
    view_61: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_15, [1, 512, 768]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:387, code: hidden_states = self.dropout(hidden_states)
    native_dropout_8 = torch.ops.aten.native_dropout.default(view_61, 0.1, True);  view_61 = None
    getitem_26: "f32[1, 512, 768]" = native_dropout_8[0]
    getitem_27: "b8[1, 512, 768]" = native_dropout_8[1];  native_dropout_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_21: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_26, add_19);  getitem_26 = add_19 = None
    var_mean_5 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 512, 1]" = var_mean_5[0]
    getitem_29: "f32[1, 512, 1]" = var_mean_5[1];  var_mean_5 = None
    add_22: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
    rsqrt_5: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
    sub_9: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_21, getitem_29);  add_21 = getitem_29 = None
    mul_17: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_5);  sub_9 = None
    mul_18: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_17, primals_46)
    add_23: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_18, primals_47);  mul_18 = primals_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_62: "f32[512, 768]" = torch.ops.aten.reshape.default(add_23, [512, 768])
    permute_31: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_48, [1, 0]);  primals_48 = None
    addmm_16: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_49, view_62, permute_31);  primals_49 = None
    view_63: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_16, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_19: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, 0.5)
    mul_20: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476);  view_63 = None
    erf_2: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_20);  mul_20 = None
    add_24: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_21: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_19, add_24);  mul_19 = add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_64: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_21, [512, 3072]);  mul_21 = None
    permute_32: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_50, [1, 0]);  primals_50 = None
    addmm_17: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_51, view_64, permute_32);  primals_51 = None
    view_65: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_17, [1, 512, 768]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:465, code: hidden_states = self.dropout(hidden_states)
    native_dropout_9 = torch.ops.aten.native_dropout.default(view_65, 0.1, True);  view_65 = None
    getitem_30: "f32[1, 512, 768]" = native_dropout_9[0]
    getitem_31: "b8[1, 512, 768]" = native_dropout_9[1];  native_dropout_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_25: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_30, add_23);  getitem_30 = add_23 = None
    var_mean_6 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
    getitem_32: "f32[1, 512, 1]" = var_mean_6[0]
    getitem_33: "f32[1, 512, 1]" = var_mean_6[1];  var_mean_6 = None
    add_26: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
    rsqrt_6: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
    sub_10: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_25, getitem_33);  add_25 = getitem_33 = None
    mul_22: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_6);  sub_10 = None
    mul_23: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_22, primals_52)
    add_27: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_23, primals_53);  mul_23 = primals_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_66: "f32[512, 768]" = torch.ops.aten.reshape.default(add_27, [512, 768])
    permute_33: "f32[768, 768]" = torch.ops.aten.permute.default(primals_54, [1, 0]);  primals_54 = None
    addmm_18: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_55, view_66, permute_33);  primals_55 = None
    view_67: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_18, [1, 512, 768]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_34: "f32[768, 768]" = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
    addmm_19: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_57, view_66, permute_34);  primals_57 = None
    view_69: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_19, [1, 512, 768]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_70: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_69, [1, 512, 12, 64]);  view_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_35: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_36: "f32[768, 768]" = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
    addmm_20: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_59, view_66, permute_36);  primals_59 = None
    view_72: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_20, [1, 512, 768]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_73: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_72, [1, 512, 12, 64]);  view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_37: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_74: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_67, [1, 512, 12, 64]);  view_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_38: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
    
    # No stacktrace found for following nodes
    clone_default_32: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
    clone_default_33: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
    clone_default_34: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
    mul_scalar_32: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default_32, 0.3535533905932738);  clone_default_32 = None
    permute_default_48: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(clone_default_33, [0, 1, 3, 2]);  clone_default_33 = None
    mul_scalar_33: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_48, 0.3535533905932738);  permute_default_48 = None
    expand_default_32: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(mul_scalar_32, [1, 12, 512, 64]);  mul_scalar_32 = None
    view_default_96: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_default_32, [12, 512, 64]);  expand_default_32 = None
    expand_default_33: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_33, [1, 12, 64, 512]);  mul_scalar_33 = None
    view_default_97: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_default_33, [12, 64, 512]);  expand_default_33 = None
    bmm_default_48: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_default_96, view_default_97)
    view_default_98: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_48, [1, 12, 512, 512]);  bmm_default_48 = None
    amax_default_8: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(view_default_98, [-1], True)
    sub_tensor_16: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_98, amax_default_8);  view_default_98 = amax_default_8 = None
    exp_default_8: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_tensor_16);  sub_tensor_16 = None
    sum_dim_int_list_16: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_8, [-1], True)
    div_tensor_8: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_default_8, sum_dim_int_list_16);  exp_default_8 = sum_dim_int_list_16 = None
    alias_default_16: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(div_tensor_8)
    native_dropout_default_8 = torch.ops.aten.native_dropout.default(div_tensor_8, 0.1, True);  div_tensor_8 = None
    getitem_142: "f32[1, 12, 512, 512]" = native_dropout_default_8[0]
    getitem_143: "b8[1, 12, 512, 512]" = native_dropout_default_8[1];  native_dropout_default_8 = None
    expand_default_34: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(getitem_142, [1, 12, 512, 512]);  getitem_142 = None
    view_default_99: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_default_34, [12, 512, 512]);  expand_default_34 = None
    expand_default_35: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(clone_default_34, [1, 12, 512, 64]);  clone_default_34 = None
    view_default_100: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_default_35, [12, 512, 64]);  expand_default_35 = None
    bmm_default_49: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_default_99, view_default_100)
    view_default_101: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_49, [1, 12, 512, 64]);  bmm_default_49 = None
    permute_default_49: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_default_99, [0, 2, 1]);  view_default_99 = None
    permute_default_50: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_default_100, [0, 2, 1]);  view_default_100 = None
    alias_default_17: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_default_16);  alias_default_16 = None
    permute_default_51: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_default_96, [0, 2, 1]);  view_default_96 = None
    permute_default_52: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_default_97, [0, 2, 1]);  view_default_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_40: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_default_101, [0, 2, 1, 3]);  view_default_101 = None
    clone_3: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_81: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_3, [1, 512, 768]);  clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_82: "f32[512, 768]" = torch.ops.aten.reshape.default(view_81, [512, 768]);  view_81 = None
    permute_41: "f32[768, 768]" = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
    addmm_21: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_61, view_82, permute_41);  primals_61 = None
    view_83: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_21, [1, 512, 768]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:387, code: hidden_states = self.dropout(hidden_states)
    native_dropout_11 = torch.ops.aten.native_dropout.default(view_83, 0.1, True);  view_83 = None
    getitem_36: "f32[1, 512, 768]" = native_dropout_11[0]
    getitem_37: "b8[1, 512, 768]" = native_dropout_11[1];  native_dropout_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_29: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_36, add_27);  getitem_36 = add_27 = None
    var_mean_7 = torch.ops.aten.var_mean.correction(add_29, [2], correction = 0, keepdim = True)
    getitem_38: "f32[1, 512, 1]" = var_mean_7[0]
    getitem_39: "f32[1, 512, 1]" = var_mean_7[1];  var_mean_7 = None
    add_30: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
    rsqrt_7: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
    sub_12: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_29, getitem_39);  add_29 = getitem_39 = None
    mul_24: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_7);  sub_12 = None
    mul_25: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_24, primals_62)
    add_31: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_25, primals_63);  mul_25 = primals_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_84: "f32[512, 768]" = torch.ops.aten.reshape.default(add_31, [512, 768])
    permute_42: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
    addmm_22: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_65, view_84, permute_42);  primals_65 = None
    view_85: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_22, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_26: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, 0.5)
    mul_27: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476);  view_85 = None
    erf_3: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_27);  mul_27 = None
    add_32: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_28: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_26, add_32);  mul_26 = add_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_86: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_28, [512, 3072]);  mul_28 = None
    permute_43: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
    addmm_23: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_67, view_86, permute_43);  primals_67 = None
    view_87: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_23, [1, 512, 768]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:465, code: hidden_states = self.dropout(hidden_states)
    native_dropout_12 = torch.ops.aten.native_dropout.default(view_87, 0.1, True);  view_87 = None
    getitem_40: "f32[1, 512, 768]" = native_dropout_12[0]
    getitem_41: "b8[1, 512, 768]" = native_dropout_12[1];  native_dropout_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_33: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_40, add_31);  getitem_40 = add_31 = None
    var_mean_8 = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
    getitem_42: "f32[1, 512, 1]" = var_mean_8[0]
    getitem_43: "f32[1, 512, 1]" = var_mean_8[1];  var_mean_8 = None
    add_34: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
    rsqrt_8: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
    sub_13: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_33, getitem_43);  add_33 = getitem_43 = None
    mul_29: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_8);  sub_13 = None
    mul_30: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_29, primals_68)
    add_35: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_30, primals_69);  mul_30 = primals_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_88: "f32[512, 768]" = torch.ops.aten.reshape.default(add_35, [512, 768])
    permute_44: "f32[768, 768]" = torch.ops.aten.permute.default(primals_70, [1, 0]);  primals_70 = None
    addmm_24: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_71, view_88, permute_44);  primals_71 = None
    view_89: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_24, [1, 512, 768]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_45: "f32[768, 768]" = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
    addmm_25: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_73, view_88, permute_45);  primals_73 = None
    view_91: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_25, [1, 512, 768]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_92: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_91, [1, 512, 12, 64]);  view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_46: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_47: "f32[768, 768]" = torch.ops.aten.permute.default(primals_74, [1, 0]);  primals_74 = None
    addmm_26: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_75, view_88, permute_47);  primals_75 = None
    view_94: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_26, [1, 512, 768]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_95: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_94, [1, 512, 12, 64]);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_48: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_96: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_89, [1, 512, 12, 64]);  view_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_49: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
    
    # No stacktrace found for following nodes
    clone_default_28: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
    clone_default_29: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
    clone_default_30: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
    mul_scalar_28: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default_28, 0.3535533905932738);  clone_default_28 = None
    permute_default_42: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(clone_default_29, [0, 1, 3, 2]);  clone_default_29 = None
    mul_scalar_29: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_42, 0.3535533905932738);  permute_default_42 = None
    expand_default_28: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(mul_scalar_28, [1, 12, 512, 64]);  mul_scalar_28 = None
    view_default_84: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_default_28, [12, 512, 64]);  expand_default_28 = None
    expand_default_29: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_29, [1, 12, 64, 512]);  mul_scalar_29 = None
    view_default_85: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_default_29, [12, 64, 512]);  expand_default_29 = None
    bmm_default_42: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_default_84, view_default_85)
    view_default_86: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_42, [1, 12, 512, 512]);  bmm_default_42 = None
    amax_default_7: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(view_default_86, [-1], True)
    sub_tensor_14: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_86, amax_default_7);  view_default_86 = amax_default_7 = None
    exp_default_7: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_tensor_14);  sub_tensor_14 = None
    sum_dim_int_list_14: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_7, [-1], True)
    div_tensor_7: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_default_7, sum_dim_int_list_14);  exp_default_7 = sum_dim_int_list_14 = None
    alias_default_14: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(div_tensor_7)
    native_dropout_default_7 = torch.ops.aten.native_dropout.default(div_tensor_7, 0.1, True);  div_tensor_7 = None
    getitem_140: "f32[1, 12, 512, 512]" = native_dropout_default_7[0]
    getitem_141: "b8[1, 12, 512, 512]" = native_dropout_default_7[1];  native_dropout_default_7 = None
    expand_default_30: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(getitem_140, [1, 12, 512, 512]);  getitem_140 = None
    view_default_87: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_default_30, [12, 512, 512]);  expand_default_30 = None
    expand_default_31: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(clone_default_30, [1, 12, 512, 64]);  clone_default_30 = None
    view_default_88: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_default_31, [12, 512, 64]);  expand_default_31 = None
    bmm_default_43: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_default_87, view_default_88)
    view_default_89: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_43, [1, 12, 512, 64]);  bmm_default_43 = None
    permute_default_43: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_default_87, [0, 2, 1]);  view_default_87 = None
    permute_default_44: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_default_88, [0, 2, 1]);  view_default_88 = None
    alias_default_15: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_default_14);  alias_default_14 = None
    permute_default_45: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_default_84, [0, 2, 1]);  view_default_84 = None
    permute_default_46: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_default_85, [0, 2, 1]);  view_default_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_51: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_default_89, [0, 2, 1, 3]);  view_default_89 = None
    clone_4: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_103: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_4, [1, 512, 768]);  clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_104: "f32[512, 768]" = torch.ops.aten.reshape.default(view_103, [512, 768]);  view_103 = None
    permute_52: "f32[768, 768]" = torch.ops.aten.permute.default(primals_76, [1, 0]);  primals_76 = None
    addmm_27: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_77, view_104, permute_52);  primals_77 = None
    view_105: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_27, [1, 512, 768]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:387, code: hidden_states = self.dropout(hidden_states)
    native_dropout_14 = torch.ops.aten.native_dropout.default(view_105, 0.1, True);  view_105 = None
    getitem_46: "f32[1, 512, 768]" = native_dropout_14[0]
    getitem_47: "b8[1, 512, 768]" = native_dropout_14[1];  native_dropout_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_37: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_46, add_35);  getitem_46 = add_35 = None
    var_mean_9 = torch.ops.aten.var_mean.correction(add_37, [2], correction = 0, keepdim = True)
    getitem_48: "f32[1, 512, 1]" = var_mean_9[0]
    getitem_49: "f32[1, 512, 1]" = var_mean_9[1];  var_mean_9 = None
    add_38: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
    rsqrt_9: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
    sub_15: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_37, getitem_49);  add_37 = getitem_49 = None
    mul_31: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_9);  sub_15 = None
    mul_32: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_31, primals_78)
    add_39: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_32, primals_79);  mul_32 = primals_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_106: "f32[512, 768]" = torch.ops.aten.reshape.default(add_39, [512, 768])
    permute_53: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_80, [1, 0]);  primals_80 = None
    addmm_28: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_81, view_106, permute_53);  primals_81 = None
    view_107: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_28, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_33: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.5)
    mul_34: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476);  view_107 = None
    erf_4: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_34);  mul_34 = None
    add_40: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_35: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_33, add_40);  mul_33 = add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_108: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_35, [512, 3072]);  mul_35 = None
    permute_54: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_82, [1, 0]);  primals_82 = None
    addmm_29: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_83, view_108, permute_54);  primals_83 = None
    view_109: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_29, [1, 512, 768]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:465, code: hidden_states = self.dropout(hidden_states)
    native_dropout_15 = torch.ops.aten.native_dropout.default(view_109, 0.1, True);  view_109 = None
    getitem_50: "f32[1, 512, 768]" = native_dropout_15[0]
    getitem_51: "b8[1, 512, 768]" = native_dropout_15[1];  native_dropout_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_41: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_50, add_39);  getitem_50 = add_39 = None
    var_mean_10 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
    getitem_52: "f32[1, 512, 1]" = var_mean_10[0]
    getitem_53: "f32[1, 512, 1]" = var_mean_10[1];  var_mean_10 = None
    add_42: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-12);  getitem_52 = None
    rsqrt_10: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_16: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_41, getitem_53);  add_41 = getitem_53 = None
    mul_36: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_10);  sub_16 = None
    mul_37: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_36, primals_84)
    add_43: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_37, primals_85);  mul_37 = primals_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_110: "f32[512, 768]" = torch.ops.aten.reshape.default(add_43, [512, 768])
    permute_55: "f32[768, 768]" = torch.ops.aten.permute.default(primals_86, [1, 0]);  primals_86 = None
    addmm_30: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_87, view_110, permute_55);  primals_87 = None
    view_111: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_30, [1, 512, 768]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_56: "f32[768, 768]" = torch.ops.aten.permute.default(primals_88, [1, 0]);  primals_88 = None
    addmm_31: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_89, view_110, permute_56);  primals_89 = None
    view_113: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_31, [1, 512, 768]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_114: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_113, [1, 512, 12, 64]);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_57: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_58: "f32[768, 768]" = torch.ops.aten.permute.default(primals_90, [1, 0]);  primals_90 = None
    addmm_32: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_91, view_110, permute_58);  primals_91 = None
    view_116: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_32, [1, 512, 768]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_117: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_116, [1, 512, 12, 64]);  view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_59: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_118: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_111, [1, 512, 12, 64]);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_60: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
    
    # No stacktrace found for following nodes
    clone_default_24: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
    clone_default_25: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_57, memory_format = torch.contiguous_format);  permute_57 = None
    clone_default_26: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
    mul_scalar_24: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default_24, 0.3535533905932738);  clone_default_24 = None
    permute_default_36: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(clone_default_25, [0, 1, 3, 2]);  clone_default_25 = None
    mul_scalar_25: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_36, 0.3535533905932738);  permute_default_36 = None
    expand_default_24: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(mul_scalar_24, [1, 12, 512, 64]);  mul_scalar_24 = None
    view_default_72: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_default_24, [12, 512, 64]);  expand_default_24 = None
    expand_default_25: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_25, [1, 12, 64, 512]);  mul_scalar_25 = None
    view_default_73: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_default_25, [12, 64, 512]);  expand_default_25 = None
    bmm_default_36: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_default_72, view_default_73)
    view_default_74: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_36, [1, 12, 512, 512]);  bmm_default_36 = None
    amax_default_6: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(view_default_74, [-1], True)
    sub_tensor_12: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_74, amax_default_6);  view_default_74 = amax_default_6 = None
    exp_default_6: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_tensor_12);  sub_tensor_12 = None
    sum_dim_int_list_12: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_6, [-1], True)
    div_tensor_6: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_default_6, sum_dim_int_list_12);  exp_default_6 = sum_dim_int_list_12 = None
    alias_default_12: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(div_tensor_6)
    native_dropout_default_6 = torch.ops.aten.native_dropout.default(div_tensor_6, 0.1, True);  div_tensor_6 = None
    getitem_138: "f32[1, 12, 512, 512]" = native_dropout_default_6[0]
    getitem_139: "b8[1, 12, 512, 512]" = native_dropout_default_6[1];  native_dropout_default_6 = None
    expand_default_26: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(getitem_138, [1, 12, 512, 512]);  getitem_138 = None
    view_default_75: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_default_26, [12, 512, 512]);  expand_default_26 = None
    expand_default_27: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(clone_default_26, [1, 12, 512, 64]);  clone_default_26 = None
    view_default_76: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_default_27, [12, 512, 64]);  expand_default_27 = None
    bmm_default_37: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_default_75, view_default_76)
    view_default_77: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_37, [1, 12, 512, 64]);  bmm_default_37 = None
    permute_default_37: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_default_75, [0, 2, 1]);  view_default_75 = None
    permute_default_38: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_default_76, [0, 2, 1]);  view_default_76 = None
    alias_default_13: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_default_12);  alias_default_12 = None
    permute_default_39: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_default_72, [0, 2, 1]);  view_default_72 = None
    permute_default_40: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_default_73, [0, 2, 1]);  view_default_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_62: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_default_77, [0, 2, 1, 3]);  view_default_77 = None
    clone_5: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_125: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_5, [1, 512, 768]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_126: "f32[512, 768]" = torch.ops.aten.reshape.default(view_125, [512, 768]);  view_125 = None
    permute_63: "f32[768, 768]" = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
    addmm_33: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_93, view_126, permute_63);  primals_93 = None
    view_127: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_33, [1, 512, 768]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:387, code: hidden_states = self.dropout(hidden_states)
    native_dropout_17 = torch.ops.aten.native_dropout.default(view_127, 0.1, True);  view_127 = None
    getitem_56: "f32[1, 512, 768]" = native_dropout_17[0]
    getitem_57: "b8[1, 512, 768]" = native_dropout_17[1];  native_dropout_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_45: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_56, add_43);  getitem_56 = add_43 = None
    var_mean_11 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
    getitem_58: "f32[1, 512, 1]" = var_mean_11[0]
    getitem_59: "f32[1, 512, 1]" = var_mean_11[1];  var_mean_11 = None
    add_46: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-12);  getitem_58 = None
    rsqrt_11: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
    sub_18: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_45, getitem_59);  add_45 = getitem_59 = None
    mul_38: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_11);  sub_18 = None
    mul_39: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_38, primals_94)
    add_47: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_39, primals_95);  mul_39 = primals_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_128: "f32[512, 768]" = torch.ops.aten.reshape.default(add_47, [512, 768])
    permute_64: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_96, [1, 0]);  primals_96 = None
    addmm_34: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_97, view_128, permute_64);  primals_97 = None
    view_129: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_34, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_40: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, 0.5)
    mul_41: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476);  view_129 = None
    erf_5: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_41);  mul_41 = None
    add_48: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_42: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_40, add_48);  mul_40 = add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_130: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_42, [512, 3072]);  mul_42 = None
    permute_65: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_98, [1, 0]);  primals_98 = None
    addmm_35: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_99, view_130, permute_65);  primals_99 = None
    view_131: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_35, [1, 512, 768]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:465, code: hidden_states = self.dropout(hidden_states)
    native_dropout_18 = torch.ops.aten.native_dropout.default(view_131, 0.1, True);  view_131 = None
    getitem_60: "f32[1, 512, 768]" = native_dropout_18[0]
    getitem_61: "b8[1, 512, 768]" = native_dropout_18[1];  native_dropout_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_49: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_60, add_47);  getitem_60 = add_47 = None
    var_mean_12 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
    getitem_62: "f32[1, 512, 1]" = var_mean_12[0]
    getitem_63: "f32[1, 512, 1]" = var_mean_12[1];  var_mean_12 = None
    add_50: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-12);  getitem_62 = None
    rsqrt_12: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
    sub_19: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_49, getitem_63);  add_49 = getitem_63 = None
    mul_43: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_12);  sub_19 = None
    mul_44: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_43, primals_100)
    add_51: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_44, primals_101);  mul_44 = primals_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_132: "f32[512, 768]" = torch.ops.aten.reshape.default(add_51, [512, 768])
    permute_66: "f32[768, 768]" = torch.ops.aten.permute.default(primals_102, [1, 0]);  primals_102 = None
    addmm_36: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_103, view_132, permute_66);  primals_103 = None
    view_133: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_36, [1, 512, 768]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_67: "f32[768, 768]" = torch.ops.aten.permute.default(primals_104, [1, 0]);  primals_104 = None
    addmm_37: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_105, view_132, permute_67);  primals_105 = None
    view_135: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_37, [1, 512, 768]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_136: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_135, [1, 512, 12, 64]);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_68: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_69: "f32[768, 768]" = torch.ops.aten.permute.default(primals_106, [1, 0]);  primals_106 = None
    addmm_38: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_107, view_132, permute_69);  primals_107 = None
    view_138: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_38, [1, 512, 768]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_139: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_138, [1, 512, 12, 64]);  view_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_70: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_139, [0, 2, 1, 3]);  view_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_140: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_133, [1, 512, 12, 64]);  view_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_71: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None
    
    # No stacktrace found for following nodes
    clone_default_20: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
    clone_default_21: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
    clone_default_22: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
    mul_scalar_20: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default_20, 0.3535533905932738);  clone_default_20 = None
    permute_default_30: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(clone_default_21, [0, 1, 3, 2]);  clone_default_21 = None
    mul_scalar_21: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_30, 0.3535533905932738);  permute_default_30 = None
    expand_default_20: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(mul_scalar_20, [1, 12, 512, 64]);  mul_scalar_20 = None
    view_default_60: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_default_20, [12, 512, 64]);  expand_default_20 = None
    expand_default_21: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_21, [1, 12, 64, 512]);  mul_scalar_21 = None
    view_default_61: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_default_21, [12, 64, 512]);  expand_default_21 = None
    bmm_default_30: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_default_60, view_default_61)
    view_default_62: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_30, [1, 12, 512, 512]);  bmm_default_30 = None
    amax_default_5: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(view_default_62, [-1], True)
    sub_tensor_10: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_62, amax_default_5);  view_default_62 = amax_default_5 = None
    exp_default_5: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_tensor_10);  sub_tensor_10 = None
    sum_dim_int_list_10: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_5, [-1], True)
    div_tensor_5: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_default_5, sum_dim_int_list_10);  exp_default_5 = sum_dim_int_list_10 = None
    alias_default_10: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(div_tensor_5)
    native_dropout_default_5 = torch.ops.aten.native_dropout.default(div_tensor_5, 0.1, True);  div_tensor_5 = None
    getitem_136: "f32[1, 12, 512, 512]" = native_dropout_default_5[0]
    getitem_137: "b8[1, 12, 512, 512]" = native_dropout_default_5[1];  native_dropout_default_5 = None
    expand_default_22: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(getitem_136, [1, 12, 512, 512]);  getitem_136 = None
    view_default_63: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_default_22, [12, 512, 512]);  expand_default_22 = None
    expand_default_23: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(clone_default_22, [1, 12, 512, 64]);  clone_default_22 = None
    view_default_64: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_default_23, [12, 512, 64]);  expand_default_23 = None
    bmm_default_31: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_default_63, view_default_64)
    view_default_65: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_31, [1, 12, 512, 64]);  bmm_default_31 = None
    permute_default_31: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_default_63, [0, 2, 1]);  view_default_63 = None
    permute_default_32: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_default_64, [0, 2, 1]);  view_default_64 = None
    alias_default_11: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_default_10);  alias_default_10 = None
    permute_default_33: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_default_60, [0, 2, 1]);  view_default_60 = None
    permute_default_34: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_default_61, [0, 2, 1]);  view_default_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_73: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_default_65, [0, 2, 1, 3]);  view_default_65 = None
    clone_6: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_147: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_6, [1, 512, 768]);  clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_148: "f32[512, 768]" = torch.ops.aten.reshape.default(view_147, [512, 768]);  view_147 = None
    permute_74: "f32[768, 768]" = torch.ops.aten.permute.default(primals_108, [1, 0]);  primals_108 = None
    addmm_39: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_109, view_148, permute_74);  primals_109 = None
    view_149: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_39, [1, 512, 768]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:387, code: hidden_states = self.dropout(hidden_states)
    native_dropout_20 = torch.ops.aten.native_dropout.default(view_149, 0.1, True);  view_149 = None
    getitem_66: "f32[1, 512, 768]" = native_dropout_20[0]
    getitem_67: "b8[1, 512, 768]" = native_dropout_20[1];  native_dropout_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_53: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_66, add_51);  getitem_66 = add_51 = None
    var_mean_13 = torch.ops.aten.var_mean.correction(add_53, [2], correction = 0, keepdim = True)
    getitem_68: "f32[1, 512, 1]" = var_mean_13[0]
    getitem_69: "f32[1, 512, 1]" = var_mean_13[1];  var_mean_13 = None
    add_54: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-12);  getitem_68 = None
    rsqrt_13: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
    sub_21: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_53, getitem_69);  add_53 = getitem_69 = None
    mul_45: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_13);  sub_21 = None
    mul_46: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_45, primals_110)
    add_55: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_46, primals_111);  mul_46 = primals_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_150: "f32[512, 768]" = torch.ops.aten.reshape.default(add_55, [512, 768])
    permute_75: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_112, [1, 0]);  primals_112 = None
    addmm_40: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_113, view_150, permute_75);  primals_113 = None
    view_151: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_40, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_47: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, 0.5)
    mul_48: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476);  view_151 = None
    erf_6: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_48);  mul_48 = None
    add_56: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_49: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_47, add_56);  mul_47 = add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_152: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_49, [512, 3072]);  mul_49 = None
    permute_76: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_114, [1, 0]);  primals_114 = None
    addmm_41: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_115, view_152, permute_76);  primals_115 = None
    view_153: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_41, [1, 512, 768]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:465, code: hidden_states = self.dropout(hidden_states)
    native_dropout_21 = torch.ops.aten.native_dropout.default(view_153, 0.1, True);  view_153 = None
    getitem_70: "f32[1, 512, 768]" = native_dropout_21[0]
    getitem_71: "b8[1, 512, 768]" = native_dropout_21[1];  native_dropout_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_57: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_70, add_55);  getitem_70 = add_55 = None
    var_mean_14 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
    getitem_72: "f32[1, 512, 1]" = var_mean_14[0]
    getitem_73: "f32[1, 512, 1]" = var_mean_14[1];  var_mean_14 = None
    add_58: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-12);  getitem_72 = None
    rsqrt_14: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
    sub_22: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_57, getitem_73);  add_57 = getitem_73 = None
    mul_50: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_14);  sub_22 = None
    mul_51: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_50, primals_116)
    add_59: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_51, primals_117);  mul_51 = primals_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_154: "f32[512, 768]" = torch.ops.aten.reshape.default(add_59, [512, 768])
    permute_77: "f32[768, 768]" = torch.ops.aten.permute.default(primals_118, [1, 0]);  primals_118 = None
    addmm_42: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_119, view_154, permute_77);  primals_119 = None
    view_155: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_42, [1, 512, 768]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_78: "f32[768, 768]" = torch.ops.aten.permute.default(primals_120, [1, 0]);  primals_120 = None
    addmm_43: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_121, view_154, permute_78);  primals_121 = None
    view_157: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_43, [1, 512, 768]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_158: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_157, [1, 512, 12, 64]);  view_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_79: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_158, [0, 2, 1, 3]);  view_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_80: "f32[768, 768]" = torch.ops.aten.permute.default(primals_122, [1, 0]);  primals_122 = None
    addmm_44: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_123, view_154, permute_80);  primals_123 = None
    view_160: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_44, [1, 512, 768]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_161: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_160, [1, 512, 12, 64]);  view_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_81: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_161, [0, 2, 1, 3]);  view_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_162: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_155, [1, 512, 12, 64]);  view_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_82: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
    
    # No stacktrace found for following nodes
    clone_default_16: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
    clone_default_17: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None
    clone_default_18: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_81, memory_format = torch.contiguous_format);  permute_81 = None
    mul_scalar_16: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default_16, 0.3535533905932738);  clone_default_16 = None
    permute_default_24: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(clone_default_17, [0, 1, 3, 2]);  clone_default_17 = None
    mul_scalar_17: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_24, 0.3535533905932738);  permute_default_24 = None
    expand_default_16: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(mul_scalar_16, [1, 12, 512, 64]);  mul_scalar_16 = None
    view_default_48: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_default_16, [12, 512, 64]);  expand_default_16 = None
    expand_default_17: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_17, [1, 12, 64, 512]);  mul_scalar_17 = None
    view_default_49: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_default_17, [12, 64, 512]);  expand_default_17 = None
    bmm_default_24: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_default_48, view_default_49)
    view_default_50: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_24, [1, 12, 512, 512]);  bmm_default_24 = None
    amax_default_4: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(view_default_50, [-1], True)
    sub_tensor_8: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_50, amax_default_4);  view_default_50 = amax_default_4 = None
    exp_default_4: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_tensor_8);  sub_tensor_8 = None
    sum_dim_int_list_8: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_4, [-1], True)
    div_tensor_4: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_default_4, sum_dim_int_list_8);  exp_default_4 = sum_dim_int_list_8 = None
    alias_default_8: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(div_tensor_4)
    native_dropout_default_4 = torch.ops.aten.native_dropout.default(div_tensor_4, 0.1, True);  div_tensor_4 = None
    getitem_134: "f32[1, 12, 512, 512]" = native_dropout_default_4[0]
    getitem_135: "b8[1, 12, 512, 512]" = native_dropout_default_4[1];  native_dropout_default_4 = None
    expand_default_18: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(getitem_134, [1, 12, 512, 512]);  getitem_134 = None
    view_default_51: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_default_18, [12, 512, 512]);  expand_default_18 = None
    expand_default_19: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(clone_default_18, [1, 12, 512, 64]);  clone_default_18 = None
    view_default_52: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_default_19, [12, 512, 64]);  expand_default_19 = None
    bmm_default_25: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_default_51, view_default_52)
    view_default_53: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_25, [1, 12, 512, 64]);  bmm_default_25 = None
    permute_default_25: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_default_51, [0, 2, 1]);  view_default_51 = None
    permute_default_26: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_default_52, [0, 2, 1]);  view_default_52 = None
    alias_default_9: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_default_8);  alias_default_8 = None
    permute_default_27: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_default_48, [0, 2, 1]);  view_default_48 = None
    permute_default_28: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_default_49, [0, 2, 1]);  view_default_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_84: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_default_53, [0, 2, 1, 3]);  view_default_53 = None
    clone_7: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_169: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_7, [1, 512, 768]);  clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_170: "f32[512, 768]" = torch.ops.aten.reshape.default(view_169, [512, 768]);  view_169 = None
    permute_85: "f32[768, 768]" = torch.ops.aten.permute.default(primals_124, [1, 0]);  primals_124 = None
    addmm_45: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_125, view_170, permute_85);  primals_125 = None
    view_171: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_45, [1, 512, 768]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:387, code: hidden_states = self.dropout(hidden_states)
    native_dropout_23 = torch.ops.aten.native_dropout.default(view_171, 0.1, True);  view_171 = None
    getitem_76: "f32[1, 512, 768]" = native_dropout_23[0]
    getitem_77: "b8[1, 512, 768]" = native_dropout_23[1];  native_dropout_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_61: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_76, add_59);  getitem_76 = add_59 = None
    var_mean_15 = torch.ops.aten.var_mean.correction(add_61, [2], correction = 0, keepdim = True)
    getitem_78: "f32[1, 512, 1]" = var_mean_15[0]
    getitem_79: "f32[1, 512, 1]" = var_mean_15[1];  var_mean_15 = None
    add_62: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-12);  getitem_78 = None
    rsqrt_15: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
    sub_24: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_61, getitem_79);  add_61 = getitem_79 = None
    mul_52: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_15);  sub_24 = None
    mul_53: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_52, primals_126)
    add_63: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_53, primals_127);  mul_53 = primals_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_172: "f32[512, 768]" = torch.ops.aten.reshape.default(add_63, [512, 768])
    permute_86: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_128, [1, 0]);  primals_128 = None
    addmm_46: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_129, view_172, permute_86);  primals_129 = None
    view_173: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_46, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_54: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.5)
    mul_55: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476);  view_173 = None
    erf_7: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
    add_64: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_56: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_54, add_64);  mul_54 = add_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_174: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_56, [512, 3072]);  mul_56 = None
    permute_87: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_130, [1, 0]);  primals_130 = None
    addmm_47: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_131, view_174, permute_87);  primals_131 = None
    view_175: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_47, [1, 512, 768]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:465, code: hidden_states = self.dropout(hidden_states)
    native_dropout_24 = torch.ops.aten.native_dropout.default(view_175, 0.1, True);  view_175 = None
    getitem_80: "f32[1, 512, 768]" = native_dropout_24[0]
    getitem_81: "b8[1, 512, 768]" = native_dropout_24[1];  native_dropout_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_65: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_80, add_63);  getitem_80 = add_63 = None
    var_mean_16 = torch.ops.aten.var_mean.correction(add_65, [2], correction = 0, keepdim = True)
    getitem_82: "f32[1, 512, 1]" = var_mean_16[0]
    getitem_83: "f32[1, 512, 1]" = var_mean_16[1];  var_mean_16 = None
    add_66: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-12);  getitem_82 = None
    rsqrt_16: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    sub_25: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_65, getitem_83);  add_65 = getitem_83 = None
    mul_57: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_16);  sub_25 = None
    mul_58: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_57, primals_132)
    add_67: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_58, primals_133);  mul_58 = primals_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_176: "f32[512, 768]" = torch.ops.aten.reshape.default(add_67, [512, 768])
    permute_88: "f32[768, 768]" = torch.ops.aten.permute.default(primals_134, [1, 0]);  primals_134 = None
    addmm_48: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_135, view_176, permute_88);  primals_135 = None
    view_177: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_48, [1, 512, 768]);  addmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_89: "f32[768, 768]" = torch.ops.aten.permute.default(primals_136, [1, 0]);  primals_136 = None
    addmm_49: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_137, view_176, permute_89);  primals_137 = None
    view_179: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_49, [1, 512, 768]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_180: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_179, [1, 512, 12, 64]);  view_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_90: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_180, [0, 2, 1, 3]);  view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_91: "f32[768, 768]" = torch.ops.aten.permute.default(primals_138, [1, 0]);  primals_138 = None
    addmm_50: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_139, view_176, permute_91);  primals_139 = None
    view_182: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_50, [1, 512, 768]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_183: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_182, [1, 512, 12, 64]);  view_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_92: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_184: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_177, [1, 512, 12, 64]);  view_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_93: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
    
    # No stacktrace found for following nodes
    clone_default_12: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
    clone_default_13: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None
    clone_default_14: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
    mul_scalar_12: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default_12, 0.3535533905932738);  clone_default_12 = None
    permute_default_18: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(clone_default_13, [0, 1, 3, 2]);  clone_default_13 = None
    mul_scalar_13: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_18, 0.3535533905932738);  permute_default_18 = None
    expand_default_12: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(mul_scalar_12, [1, 12, 512, 64]);  mul_scalar_12 = None
    view_default_36: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_default_12, [12, 512, 64]);  expand_default_12 = None
    expand_default_13: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_13, [1, 12, 64, 512]);  mul_scalar_13 = None
    view_default_37: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_default_13, [12, 64, 512]);  expand_default_13 = None
    bmm_default_18: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_default_36, view_default_37)
    view_default_38: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_18, [1, 12, 512, 512]);  bmm_default_18 = None
    amax_default_3: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(view_default_38, [-1], True)
    sub_tensor_6: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_38, amax_default_3);  view_default_38 = amax_default_3 = None
    exp_default_3: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_tensor_6);  sub_tensor_6 = None
    sum_dim_int_list_6: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_3, [-1], True)
    div_tensor_3: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_default_3, sum_dim_int_list_6);  exp_default_3 = sum_dim_int_list_6 = None
    alias_default_6: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(div_tensor_3)
    native_dropout_default_3 = torch.ops.aten.native_dropout.default(div_tensor_3, 0.1, True);  div_tensor_3 = None
    getitem_132: "f32[1, 12, 512, 512]" = native_dropout_default_3[0]
    getitem_133: "b8[1, 12, 512, 512]" = native_dropout_default_3[1];  native_dropout_default_3 = None
    expand_default_14: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(getitem_132, [1, 12, 512, 512]);  getitem_132 = None
    view_default_39: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_default_14, [12, 512, 512]);  expand_default_14 = None
    expand_default_15: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(clone_default_14, [1, 12, 512, 64]);  clone_default_14 = None
    view_default_40: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_default_15, [12, 512, 64]);  expand_default_15 = None
    bmm_default_19: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_default_39, view_default_40)
    view_default_41: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_19, [1, 12, 512, 64]);  bmm_default_19 = None
    permute_default_19: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_default_39, [0, 2, 1]);  view_default_39 = None
    permute_default_20: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_default_40, [0, 2, 1]);  view_default_40 = None
    alias_default_7: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_default_6);  alias_default_6 = None
    permute_default_21: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_default_36, [0, 2, 1]);  view_default_36 = None
    permute_default_22: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_default_37, [0, 2, 1]);  view_default_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_95: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_default_41, [0, 2, 1, 3]);  view_default_41 = None
    clone_8: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_191: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_8, [1, 512, 768]);  clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_192: "f32[512, 768]" = torch.ops.aten.reshape.default(view_191, [512, 768]);  view_191 = None
    permute_96: "f32[768, 768]" = torch.ops.aten.permute.default(primals_140, [1, 0]);  primals_140 = None
    addmm_51: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_141, view_192, permute_96);  primals_141 = None
    view_193: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_51, [1, 512, 768]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:387, code: hidden_states = self.dropout(hidden_states)
    native_dropout_26 = torch.ops.aten.native_dropout.default(view_193, 0.1, True);  view_193 = None
    getitem_86: "f32[1, 512, 768]" = native_dropout_26[0]
    getitem_87: "b8[1, 512, 768]" = native_dropout_26[1];  native_dropout_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_69: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_86, add_67);  getitem_86 = add_67 = None
    var_mean_17 = torch.ops.aten.var_mean.correction(add_69, [2], correction = 0, keepdim = True)
    getitem_88: "f32[1, 512, 1]" = var_mean_17[0]
    getitem_89: "f32[1, 512, 1]" = var_mean_17[1];  var_mean_17 = None
    add_70: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-12);  getitem_88 = None
    rsqrt_17: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
    sub_27: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_69, getitem_89);  add_69 = getitem_89 = None
    mul_59: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_17);  sub_27 = None
    mul_60: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_59, primals_142)
    add_71: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_60, primals_143);  mul_60 = primals_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_194: "f32[512, 768]" = torch.ops.aten.reshape.default(add_71, [512, 768])
    permute_97: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_144, [1, 0]);  primals_144 = None
    addmm_52: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_145, view_194, permute_97);  primals_145 = None
    view_195: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_52, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_61: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, 0.5)
    mul_62: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476);  view_195 = None
    erf_8: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_62);  mul_62 = None
    add_72: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_63: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_61, add_72);  mul_61 = add_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_196: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_63, [512, 3072]);  mul_63 = None
    permute_98: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_146, [1, 0]);  primals_146 = None
    addmm_53: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_147, view_196, permute_98);  primals_147 = None
    view_197: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_53, [1, 512, 768]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:465, code: hidden_states = self.dropout(hidden_states)
    native_dropout_27 = torch.ops.aten.native_dropout.default(view_197, 0.1, True);  view_197 = None
    getitem_90: "f32[1, 512, 768]" = native_dropout_27[0]
    getitem_91: "b8[1, 512, 768]" = native_dropout_27[1];  native_dropout_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_73: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_90, add_71);  getitem_90 = add_71 = None
    var_mean_18 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
    getitem_92: "f32[1, 512, 1]" = var_mean_18[0]
    getitem_93: "f32[1, 512, 1]" = var_mean_18[1];  var_mean_18 = None
    add_74: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-12);  getitem_92 = None
    rsqrt_18: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    sub_28: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_73, getitem_93);  add_73 = getitem_93 = None
    mul_64: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_18);  sub_28 = None
    mul_65: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_64, primals_148)
    add_75: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_65, primals_149);  mul_65 = primals_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_198: "f32[512, 768]" = torch.ops.aten.reshape.default(add_75, [512, 768])
    permute_99: "f32[768, 768]" = torch.ops.aten.permute.default(primals_150, [1, 0]);  primals_150 = None
    addmm_54: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_151, view_198, permute_99);  primals_151 = None
    view_199: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_54, [1, 512, 768]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_100: "f32[768, 768]" = torch.ops.aten.permute.default(primals_152, [1, 0]);  primals_152 = None
    addmm_55: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_153, view_198, permute_100);  primals_153 = None
    view_201: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_55, [1, 512, 768]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_202: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_201, [1, 512, 12, 64]);  view_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_101: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_202, [0, 2, 1, 3]);  view_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_102: "f32[768, 768]" = torch.ops.aten.permute.default(primals_154, [1, 0]);  primals_154 = None
    addmm_56: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_155, view_198, permute_102);  primals_155 = None
    view_204: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_56, [1, 512, 768]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_205: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_204, [1, 512, 12, 64]);  view_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_103: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_205, [0, 2, 1, 3]);  view_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_206: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_199, [1, 512, 12, 64]);  view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_104: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
    
    # No stacktrace found for following nodes
    clone_default_8: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
    clone_default_9: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_101, memory_format = torch.contiguous_format);  permute_101 = None
    clone_default_10: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_103, memory_format = torch.contiguous_format);  permute_103 = None
    mul_scalar_8: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default_8, 0.3535533905932738);  clone_default_8 = None
    permute_default_12: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(clone_default_9, [0, 1, 3, 2]);  clone_default_9 = None
    mul_scalar_9: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_12, 0.3535533905932738);  permute_default_12 = None
    expand_default_8: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(mul_scalar_8, [1, 12, 512, 64]);  mul_scalar_8 = None
    view_default_24: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_default_8, [12, 512, 64]);  expand_default_8 = None
    expand_default_9: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_9, [1, 12, 64, 512]);  mul_scalar_9 = None
    view_default_25: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_default_9, [12, 64, 512]);  expand_default_9 = None
    bmm_default_12: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_default_24, view_default_25)
    view_default_26: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_12, [1, 12, 512, 512]);  bmm_default_12 = None
    amax_default_2: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(view_default_26, [-1], True)
    sub_tensor_4: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_26, amax_default_2);  view_default_26 = amax_default_2 = None
    exp_default_2: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_tensor_4);  sub_tensor_4 = None
    sum_dim_int_list_4: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_2, [-1], True)
    div_tensor_2: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_default_2, sum_dim_int_list_4);  exp_default_2 = sum_dim_int_list_4 = None
    alias_default_4: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(div_tensor_2)
    native_dropout_default_2 = torch.ops.aten.native_dropout.default(div_tensor_2, 0.1, True);  div_tensor_2 = None
    getitem_130: "f32[1, 12, 512, 512]" = native_dropout_default_2[0]
    getitem_131: "b8[1, 12, 512, 512]" = native_dropout_default_2[1];  native_dropout_default_2 = None
    expand_default_10: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(getitem_130, [1, 12, 512, 512]);  getitem_130 = None
    view_default_27: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_default_10, [12, 512, 512]);  expand_default_10 = None
    expand_default_11: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(clone_default_10, [1, 12, 512, 64]);  clone_default_10 = None
    view_default_28: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_default_11, [12, 512, 64]);  expand_default_11 = None
    bmm_default_13: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_default_27, view_default_28)
    view_default_29: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_13, [1, 12, 512, 64]);  bmm_default_13 = None
    permute_default_13: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_default_27, [0, 2, 1]);  view_default_27 = None
    permute_default_14: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_default_28, [0, 2, 1]);  view_default_28 = None
    alias_default_5: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_default_4);  alias_default_4 = None
    permute_default_15: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_default_24, [0, 2, 1]);  view_default_24 = None
    permute_default_16: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_default_25, [0, 2, 1]);  view_default_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_106: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_default_29, [0, 2, 1, 3]);  view_default_29 = None
    clone_9: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_213: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_9, [1, 512, 768]);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_214: "f32[512, 768]" = torch.ops.aten.reshape.default(view_213, [512, 768]);  view_213 = None
    permute_107: "f32[768, 768]" = torch.ops.aten.permute.default(primals_156, [1, 0]);  primals_156 = None
    addmm_57: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_157, view_214, permute_107);  primals_157 = None
    view_215: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_57, [1, 512, 768]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:387, code: hidden_states = self.dropout(hidden_states)
    native_dropout_29 = torch.ops.aten.native_dropout.default(view_215, 0.1, True);  view_215 = None
    getitem_96: "f32[1, 512, 768]" = native_dropout_29[0]
    getitem_97: "b8[1, 512, 768]" = native_dropout_29[1];  native_dropout_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_77: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_96, add_75);  getitem_96 = add_75 = None
    var_mean_19 = torch.ops.aten.var_mean.correction(add_77, [2], correction = 0, keepdim = True)
    getitem_98: "f32[1, 512, 1]" = var_mean_19[0]
    getitem_99: "f32[1, 512, 1]" = var_mean_19[1];  var_mean_19 = None
    add_78: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-12);  getitem_98 = None
    rsqrt_19: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    sub_30: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_77, getitem_99);  add_77 = getitem_99 = None
    mul_66: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_19);  sub_30 = None
    mul_67: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_66, primals_158)
    add_79: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_67, primals_159);  mul_67 = primals_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_216: "f32[512, 768]" = torch.ops.aten.reshape.default(add_79, [512, 768])
    permute_108: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_160, [1, 0]);  primals_160 = None
    addmm_58: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_161, view_216, permute_108);  primals_161 = None
    view_217: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_58, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_68: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, 0.5)
    mul_69: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476);  view_217 = None
    erf_9: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
    add_80: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_70: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_68, add_80);  mul_68 = add_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_218: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_70, [512, 3072]);  mul_70 = None
    permute_109: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_162, [1, 0]);  primals_162 = None
    addmm_59: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_163, view_218, permute_109);  primals_163 = None
    view_219: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_59, [1, 512, 768]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:465, code: hidden_states = self.dropout(hidden_states)
    native_dropout_30 = torch.ops.aten.native_dropout.default(view_219, 0.1, True);  view_219 = None
    getitem_100: "f32[1, 512, 768]" = native_dropout_30[0]
    getitem_101: "b8[1, 512, 768]" = native_dropout_30[1];  native_dropout_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_81: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_100, add_79);  getitem_100 = add_79 = None
    var_mean_20 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
    getitem_102: "f32[1, 512, 1]" = var_mean_20[0]
    getitem_103: "f32[1, 512, 1]" = var_mean_20[1];  var_mean_20 = None
    add_82: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-12);  getitem_102 = None
    rsqrt_20: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
    sub_31: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_81, getitem_103);  add_81 = getitem_103 = None
    mul_71: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_20);  sub_31 = None
    mul_72: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_71, primals_164)
    add_83: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_72, primals_165);  mul_72 = primals_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_220: "f32[512, 768]" = torch.ops.aten.reshape.default(add_83, [512, 768])
    permute_110: "f32[768, 768]" = torch.ops.aten.permute.default(primals_166, [1, 0]);  primals_166 = None
    addmm_60: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_167, view_220, permute_110);  primals_167 = None
    view_221: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_60, [1, 512, 768]);  addmm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_111: "f32[768, 768]" = torch.ops.aten.permute.default(primals_168, [1, 0]);  primals_168 = None
    addmm_61: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_169, view_220, permute_111);  primals_169 = None
    view_223: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_61, [1, 512, 768]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_224: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_223, [1, 512, 12, 64]);  view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_112: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_113: "f32[768, 768]" = torch.ops.aten.permute.default(primals_170, [1, 0]);  primals_170 = None
    addmm_62: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_171, view_220, permute_113);  primals_171 = None
    view_226: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_62, [1, 512, 768]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_227: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_226, [1, 512, 12, 64]);  view_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_114: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_227, [0, 2, 1, 3]);  view_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_228: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_221, [1, 512, 12, 64]);  view_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_115: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None
    
    # No stacktrace found for following nodes
    clone_default_4: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
    clone_default_5: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_112, memory_format = torch.contiguous_format);  permute_112 = None
    clone_default_6: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
    mul_scalar_4: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default_4, 0.3535533905932738);  clone_default_4 = None
    permute_default_6: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(clone_default_5, [0, 1, 3, 2]);  clone_default_5 = None
    mul_scalar_5: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_6, 0.3535533905932738);  permute_default_6 = None
    expand_default_4: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(mul_scalar_4, [1, 12, 512, 64]);  mul_scalar_4 = None
    view_default_12: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_default_4, [12, 512, 64]);  expand_default_4 = None
    expand_default_5: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_5, [1, 12, 64, 512]);  mul_scalar_5 = None
    view_default_13: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_default_5, [12, 64, 512]);  expand_default_5 = None
    bmm_default_6: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_default_12, view_default_13)
    view_default_14: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_6, [1, 12, 512, 512]);  bmm_default_6 = None
    amax_default_1: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(view_default_14, [-1], True)
    sub_tensor_2: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_14, amax_default_1);  view_default_14 = amax_default_1 = None
    exp_default_1: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_tensor_2);  sub_tensor_2 = None
    sum_dim_int_list_2: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_1, [-1], True)
    div_tensor_1: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_default_1, sum_dim_int_list_2);  exp_default_1 = sum_dim_int_list_2 = None
    alias_default_2: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(div_tensor_1)
    native_dropout_default_1 = torch.ops.aten.native_dropout.default(div_tensor_1, 0.1, True);  div_tensor_1 = None
    getitem_128: "f32[1, 12, 512, 512]" = native_dropout_default_1[0]
    getitem_129: "b8[1, 12, 512, 512]" = native_dropout_default_1[1];  native_dropout_default_1 = None
    expand_default_6: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(getitem_128, [1, 12, 512, 512]);  getitem_128 = None
    view_default_15: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_default_6, [12, 512, 512]);  expand_default_6 = None
    expand_default_7: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(clone_default_6, [1, 12, 512, 64]);  clone_default_6 = None
    view_default_16: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_default_7, [12, 512, 64]);  expand_default_7 = None
    bmm_default_7: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_default_15, view_default_16)
    view_default_17: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_7, [1, 12, 512, 64]);  bmm_default_7 = None
    permute_default_7: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_default_15, [0, 2, 1]);  view_default_15 = None
    permute_default_8: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_default_16, [0, 2, 1]);  view_default_16 = None
    alias_default_3: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_default_2);  alias_default_2 = None
    permute_default_9: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_default_12, [0, 2, 1]);  view_default_12 = None
    permute_default_10: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_default_13, [0, 2, 1]);  view_default_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_117: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_default_17, [0, 2, 1, 3]);  view_default_17 = None
    clone_10: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_235: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_10, [1, 512, 768]);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_236: "f32[512, 768]" = torch.ops.aten.reshape.default(view_235, [512, 768]);  view_235 = None
    permute_118: "f32[768, 768]" = torch.ops.aten.permute.default(primals_172, [1, 0]);  primals_172 = None
    addmm_63: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_173, view_236, permute_118);  primals_173 = None
    view_237: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_63, [1, 512, 768]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:387, code: hidden_states = self.dropout(hidden_states)
    native_dropout_32 = torch.ops.aten.native_dropout.default(view_237, 0.1, True);  view_237 = None
    getitem_106: "f32[1, 512, 768]" = native_dropout_32[0]
    getitem_107: "b8[1, 512, 768]" = native_dropout_32[1];  native_dropout_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_85: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_106, add_83);  getitem_106 = add_83 = None
    var_mean_21 = torch.ops.aten.var_mean.correction(add_85, [2], correction = 0, keepdim = True)
    getitem_108: "f32[1, 512, 1]" = var_mean_21[0]
    getitem_109: "f32[1, 512, 1]" = var_mean_21[1];  var_mean_21 = None
    add_86: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-12);  getitem_108 = None
    rsqrt_21: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
    sub_33: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_85, getitem_109);  add_85 = getitem_109 = None
    mul_73: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_21);  sub_33 = None
    mul_74: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_73, primals_174)
    add_87: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_74, primals_175);  mul_74 = primals_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_238: "f32[512, 768]" = torch.ops.aten.reshape.default(add_87, [512, 768])
    permute_119: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_176, [1, 0]);  primals_176 = None
    addmm_64: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_177, view_238, permute_119);  primals_177 = None
    view_239: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_64, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_75: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, 0.5)
    mul_76: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, 0.7071067811865476);  view_239 = None
    erf_10: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_76);  mul_76 = None
    add_88: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_77: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_75, add_88);  mul_75 = add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_240: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_77, [512, 3072]);  mul_77 = None
    permute_120: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_178, [1, 0]);  primals_178 = None
    addmm_65: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_179, view_240, permute_120);  primals_179 = None
    view_241: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_65, [1, 512, 768]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:465, code: hidden_states = self.dropout(hidden_states)
    native_dropout_33 = torch.ops.aten.native_dropout.default(view_241, 0.1, True);  view_241 = None
    getitem_110: "f32[1, 512, 768]" = native_dropout_33[0]
    getitem_111: "b8[1, 512, 768]" = native_dropout_33[1];  native_dropout_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_89: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_110, add_87);  getitem_110 = add_87 = None
    var_mean_22 = torch.ops.aten.var_mean.correction(add_89, [2], correction = 0, keepdim = True)
    getitem_112: "f32[1, 512, 1]" = var_mean_22[0]
    getitem_113: "f32[1, 512, 1]" = var_mean_22[1];  var_mean_22 = None
    add_90: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-12);  getitem_112 = None
    rsqrt_22: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
    sub_34: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_89, getitem_113);  add_89 = getitem_113 = None
    mul_78: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_22);  sub_34 = None
    mul_79: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_78, primals_180)
    add_91: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_79, primals_181);  mul_79 = primals_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_242: "f32[512, 768]" = torch.ops.aten.reshape.default(add_91, [512, 768])
    permute_121: "f32[768, 768]" = torch.ops.aten.permute.default(primals_182, [1, 0]);  primals_182 = None
    addmm_66: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_183, view_242, permute_121);  primals_183 = None
    view_243: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_66, [1, 512, 768]);  addmm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_122: "f32[768, 768]" = torch.ops.aten.permute.default(primals_184, [1, 0]);  primals_184 = None
    addmm_67: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_185, view_242, permute_122);  primals_185 = None
    view_245: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_67, [1, 512, 768]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_246: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_245, [1, 512, 12, 64]);  view_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_123: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_124: "f32[768, 768]" = torch.ops.aten.permute.default(primals_186, [1, 0]);  primals_186 = None
    addmm_68: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_187, view_242, permute_124);  primals_187 = None
    view_248: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_68, [1, 512, 768]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_249: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_248, [1, 512, 12, 64]);  view_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_125: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_249, [0, 2, 1, 3]);  view_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_250: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_243, [1, 512, 12, 64]);  view_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_126: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None
    
    # No stacktrace found for following nodes
    clone_default: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
    clone_default_1: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_123, memory_format = torch.contiguous_format);  permute_123 = None
    clone_default_2: "f32[1, 12, 512, 64]" = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
    mul_scalar: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(clone_default, 0.3535533905932738);  clone_default = None
    permute_default: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(clone_default_1, [0, 1, 3, 2]);  clone_default_1 = None
    mul_scalar_1: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default, 0.3535533905932738);  permute_default = None
    expand_default: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(mul_scalar, [1, 12, 512, 64]);  mul_scalar = None
    view_default: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_default, [12, 512, 64]);  expand_default = None
    expand_default_1: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_1, [1, 12, 64, 512]);  mul_scalar_1 = None
    view_default_1: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_default_1, [12, 64, 512]);  expand_default_1 = None
    bmm_default: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_default, view_default_1)
    view_default_2: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_default, [1, 12, 512, 512]);  bmm_default = None
    amax_default: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(view_default_2, [-1], True)
    sub_tensor: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(view_default_2, amax_default);  view_default_2 = amax_default = None
    exp_default: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
    sum_dim_int_list: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
    div_tensor: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
    alias_default: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(div_tensor)
    native_dropout_default = torch.ops.aten.native_dropout.default(div_tensor, 0.1, True);  div_tensor = None
    getitem_126: "f32[1, 12, 512, 512]" = native_dropout_default[0]
    getitem_127: "b8[1, 12, 512, 512]" = native_dropout_default[1];  native_dropout_default = None
    expand_default_2: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(getitem_126, [1, 12, 512, 512]);  getitem_126 = None
    view_default_3: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_default_2, [12, 512, 512]);  expand_default_2 = None
    expand_default_3: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(clone_default_2, [1, 12, 512, 64]);  clone_default_2 = None
    view_default_4: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_default_3, [12, 512, 64]);  expand_default_3 = None
    bmm_default_1: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_default_3, view_default_4)
    view_default_5: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_1, [1, 12, 512, 64]);  bmm_default_1 = None
    permute_default_1: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_default_3, [0, 2, 1]);  view_default_3 = None
    permute_default_2: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_default_4, [0, 2, 1]);  view_default_4 = None
    alias_default_1: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_default);  alias_default = None
    permute_default_3: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_default, [0, 2, 1]);  view_default = None
    permute_default_4: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_default_1, [0, 2, 1]);  view_default_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_128: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_default_5, [0, 2, 1, 3]);  view_default_5 = None
    clone_11: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_257: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_11, [1, 512, 768]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_258: "f32[512, 768]" = torch.ops.aten.reshape.default(view_257, [512, 768]);  view_257 = None
    permute_129: "f32[768, 768]" = torch.ops.aten.permute.default(primals_188, [1, 0]);  primals_188 = None
    addmm_69: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_189, view_258, permute_129);  primals_189 = None
    view_259: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_69, [1, 512, 768]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:387, code: hidden_states = self.dropout(hidden_states)
    native_dropout_35 = torch.ops.aten.native_dropout.default(view_259, 0.1, True);  view_259 = None
    getitem_116: "f32[1, 512, 768]" = native_dropout_35[0]
    getitem_117: "b8[1, 512, 768]" = native_dropout_35[1];  native_dropout_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_93: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_116, add_91);  getitem_116 = add_91 = None
    var_mean_23 = torch.ops.aten.var_mean.correction(add_93, [2], correction = 0, keepdim = True)
    getitem_118: "f32[1, 512, 1]" = var_mean_23[0]
    getitem_119: "f32[1, 512, 1]" = var_mean_23[1];  var_mean_23 = None
    add_94: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-12);  getitem_118 = None
    rsqrt_23: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
    sub_36: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_93, getitem_119);  add_93 = getitem_119 = None
    mul_80: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_23);  sub_36 = None
    mul_81: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_80, primals_190)
    add_95: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_81, primals_191);  mul_81 = primals_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_260: "f32[512, 768]" = torch.ops.aten.reshape.default(add_95, [512, 768])
    permute_130: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_192, [1, 0]);  primals_192 = None
    addmm_70: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_193, view_260, permute_130);  primals_193 = None
    view_261: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_70, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_82: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, 0.5)
    mul_83: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476);  view_261 = None
    erf_11: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
    add_96: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_84: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_82, add_96);  mul_82 = add_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_262: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_84, [512, 3072]);  mul_84 = None
    permute_131: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_194, [1, 0]);  primals_194 = None
    addmm_71: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_195, view_262, permute_131);  primals_195 = None
    view_263: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_71, [1, 512, 768]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:465, code: hidden_states = self.dropout(hidden_states)
    native_dropout_36 = torch.ops.aten.native_dropout.default(view_263, 0.1, True);  view_263 = None
    getitem_120: "f32[1, 512, 768]" = native_dropout_36[0]
    getitem_121: "b8[1, 512, 768]" = native_dropout_36[1];  native_dropout_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_97: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(getitem_120, add_95);  getitem_120 = add_95 = None
    var_mean_24 = torch.ops.aten.var_mean.correction(add_97, [2], correction = 0, keepdim = True)
    getitem_122: "f32[1, 512, 1]" = var_mean_24[0]
    getitem_123: "f32[1, 512, 1]" = var_mean_24[1];  var_mean_24 = None
    add_98: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_122, 1e-12);  getitem_122 = None
    rsqrt_24: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
    sub_37: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_97, getitem_123);  add_97 = getitem_123 = None
    mul_85: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_24);  sub_37 = None
    mul_86: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_85, primals_196)
    add_99: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_86, primals_197);  mul_86 = primals_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:1858, code: logits = self.qa_outputs(sequence_output)
    view_264: "f32[512, 768]" = torch.ops.aten.reshape.default(add_99, [512, 768]);  add_99 = None
    permute_132: "f32[768, 2]" = torch.ops.aten.permute.default(primals_198, [1, 0]);  primals_198 = None
    addmm_72: "f32[512, 2]" = torch.ops.aten.addmm.default(primals_199, view_264, permute_132);  primals_199 = None
    view_265: "f32[1, 512, 2]" = torch.ops.aten.reshape.default(addmm_72, [1, 512, 2]);  addmm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:1859, code: start_logits, end_logits = logits.split(1, dim=-1)
    split_with_sizes = torch.ops.aten.split_with_sizes.default(view_265, [1, 1], 2);  view_265 = None
    getitem_124: "f32[1, 512, 1]" = split_with_sizes[0]
    getitem_125: "f32[1, 512, 1]" = split_with_sizes[1];  split_with_sizes = None
    
    # No stacktrace found for following nodes
    squeeze: "f32[1, 512]" = torch.ops.aten.squeeze.dim(getitem_124, -1);  getitem_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:1860, code: start_logits = start_logits.squeeze(-1).contiguous()
    clone_12: "f32[1, 512]" = torch.ops.aten.clone.default(squeeze, memory_format = torch.contiguous_format);  squeeze = None
    
    # No stacktrace found for following nodes
    squeeze_1: "f32[1, 512]" = torch.ops.aten.squeeze.dim(getitem_125, -1);  getitem_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:1861, code: end_logits = end_logits.squeeze(-1).contiguous()
    clone_13: "f32[1, 512]" = torch.ops.aten.clone.default(squeeze_1, memory_format = torch.contiguous_format);  squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:1872, code: start_positions = start_positions.clamp(0, ignored_index)
    clamp_min: "i64[1]" = torch.ops.aten.clamp_min.default(primals_203, 0);  primals_203 = None
    clamp_max: "i64[1]" = torch.ops.aten.clamp_max.default(clamp_min, 512);  clamp_min = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:1873, code: end_positions = end_positions.clamp(0, ignored_index)
    clamp_min_1: "i64[1]" = torch.ops.aten.clamp_min.default(primals_204, 0);  primals_204 = None
    clamp_max_1: "i64[1]" = torch.ops.aten.clamp_max.default(clamp_min_1, 512);  clamp_min_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:1876, code: start_loss = loss_fct(start_logits, start_positions)
    amax_12: "f32[1, 1]" = torch.ops.aten.amax.default(clone_12, [1], True)
    sub_38: "f32[1, 512]" = torch.ops.aten.sub.Tensor(clone_12, amax_12);  amax_12 = None
    exp_12: "f32[1, 512]" = torch.ops.aten.exp.default(sub_38)
    sum_13: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [1], True);  exp_12 = None
    log: "f32[1, 1]" = torch.ops.aten.log.default(sum_13);  sum_13 = None
    sub_39: "f32[1, 512]" = torch.ops.aten.sub.Tensor(sub_38, log);  sub_38 = log = None
    ne: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max, 512)
    full_default_1: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where: "i64[1]" = torch.ops.aten.where.self(ne, clamp_max, full_default_1)
    unsqueeze_2: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
    gather: "f32[1, 1]" = torch.ops.aten.gather.default(sub_39, 1, unsqueeze_2);  unsqueeze_2 = None
    squeeze_2: "f32[1]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[1]" = torch.ops.aten.neg.default(squeeze_2);  squeeze_2 = None
    full_default_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where_1: "f32[1]" = torch.ops.aten.where.self(ne, neg, full_default_2);  neg = None
    sum_14: "i64[]" = torch.ops.aten.sum.default(ne)
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None
    sum_15: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
    div_24: "f32[]" = torch.ops.aten.div.Tensor(sum_15, convert_element_type);  sum_15 = convert_element_type = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:1877, code: end_loss = loss_fct(end_logits, end_positions)
    amax_13: "f32[1, 1]" = torch.ops.aten.amax.default(clone_13, [1], True)
    sub_40: "f32[1, 512]" = torch.ops.aten.sub.Tensor(clone_13, amax_13);  amax_13 = None
    exp_13: "f32[1, 512]" = torch.ops.aten.exp.default(sub_40)
    sum_16: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [1], True);  exp_13 = None
    log_1: "f32[1, 1]" = torch.ops.aten.log.default(sum_16);  sum_16 = None
    sub_41: "f32[1, 512]" = torch.ops.aten.sub.Tensor(sub_40, log_1);  sub_40 = log_1 = None
    ne_3: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max_1, 512)
    where_2: "i64[1]" = torch.ops.aten.where.self(ne_3, clamp_max_1, full_default_1)
    unsqueeze_3: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(where_2, 1);  where_2 = None
    gather_1: "f32[1, 1]" = torch.ops.aten.gather.default(sub_41, 1, unsqueeze_3);  unsqueeze_3 = None
    squeeze_3: "f32[1]" = torch.ops.aten.squeeze.dim(gather_1, 1);  gather_1 = None
    neg_1: "f32[1]" = torch.ops.aten.neg.default(squeeze_3);  squeeze_3 = None
    where_3: "f32[1]" = torch.ops.aten.where.self(ne_3, neg_1, full_default_2);  neg_1 = full_default_2 = None
    sum_17: "i64[]" = torch.ops.aten.sum.default(ne_3)
    convert_element_type_1: "f32[]" = torch.ops.prims.convert_element_type.default(sum_17, torch.float32);  sum_17 = None
    sum_18: "f32[]" = torch.ops.aten.sum.default(where_3);  where_3 = None
    div_25: "f32[]" = torch.ops.aten.div.Tensor(sum_18, convert_element_type_1);  sum_18 = convert_element_type_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:1878, code: total_loss = (start_loss + end_loss) / 2
    add_100: "f32[]" = torch.ops.aten.add.Tensor(div_24, div_25);  div_24 = div_25 = None
    div_26: "f32[]" = torch.ops.aten.div.Tensor(add_100, 2);  add_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:1877, code: end_loss = loss_fct(end_logits, end_positions)
    unsqueeze_4: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(clamp_max_1, 1);  clamp_max_1 = None
    ne_6: "b8[1, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_4, 512)
    where_4: "i64[1, 1]" = torch.ops.aten.where.self(ne_6, unsqueeze_4, full_default_1);  unsqueeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:1876, code: start_loss = loss_fct(start_logits, start_positions)
    unsqueeze_5: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(clamp_max, 1);  clamp_max = None
    ne_8: "b8[1, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_5, 512)
    where_6: "i64[1, 1]" = torch.ops.aten.where.self(ne_8, unsqueeze_5, full_default_1);  unsqueeze_5 = full_default_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:1858, code: logits = self.qa_outputs(sequence_output)
    permute_133: "f32[2, 768]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_30: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    permute_137: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    permute_141: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_31: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    permute_145: "f32[768, 768]" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_157: "f32[768, 768]" = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_162: "f32[768, 768]" = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    permute_166: "f32[768, 768]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_33: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    permute_170: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    permute_174: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_34: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    permute_178: "f32[768, 768]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_190: "f32[768, 768]" = torch.ops.aten.permute.default(permute_113, [1, 0]);  permute_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_195: "f32[768, 768]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    permute_199: "f32[768, 768]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_36: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    permute_203: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    permute_207: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_37: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    permute_211: "f32[768, 768]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_223: "f32[768, 768]" = torch.ops.aten.permute.default(permute_102, [1, 0]);  permute_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_228: "f32[768, 768]" = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    permute_232: "f32[768, 768]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_39: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    permute_236: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    permute_240: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_40: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    permute_244: "f32[768, 768]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_256: "f32[768, 768]" = torch.ops.aten.permute.default(permute_91, [1, 0]);  permute_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_261: "f32[768, 768]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    permute_265: "f32[768, 768]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_42: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    permute_269: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    permute_273: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_43: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    permute_277: "f32[768, 768]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_289: "f32[768, 768]" = torch.ops.aten.permute.default(permute_80, [1, 0]);  permute_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_294: "f32[768, 768]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    permute_298: "f32[768, 768]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_45: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    permute_302: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    permute_306: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_46: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    permute_310: "f32[768, 768]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_322: "f32[768, 768]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_327: "f32[768, 768]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    permute_331: "f32[768, 768]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_48: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    permute_335: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    permute_339: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_49: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    permute_343: "f32[768, 768]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_355: "f32[768, 768]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_360: "f32[768, 768]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    permute_364: "f32[768, 768]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_51: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    permute_368: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    permute_372: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_52: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    permute_376: "f32[768, 768]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_388: "f32[768, 768]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_393: "f32[768, 768]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    permute_397: "f32[768, 768]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_54: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    permute_401: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    permute_405: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_55: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    permute_409: "f32[768, 768]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_421: "f32[768, 768]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_426: "f32[768, 768]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    permute_430: "f32[768, 768]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_57: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    permute_434: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    permute_438: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_58: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    permute_442: "f32[768, 768]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_454: "f32[768, 768]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_459: "f32[768, 768]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    permute_463: "f32[768, 768]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_60: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    permute_467: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    permute_471: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_61: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    permute_475: "f32[768, 768]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_487: "f32[768, 768]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_492: "f32[768, 768]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    permute_496: "f32[768, 768]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_63: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    permute_500: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    permute_504: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_64: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    permute_508: "f32[768, 768]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_520: "f32[768, 768]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_525: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    permute_529: "f32[768, 768]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:239, code: embeddings = self.LayerNorm(embeddings)
    div_66: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
    return [div_26, clone_12, clone_13, primals_4, primals_14, primals_20, primals_30, primals_36, primals_46, primals_52, primals_62, primals_68, primals_78, primals_84, primals_94, primals_100, primals_110, primals_116, primals_126, primals_132, primals_142, primals_148, primals_158, primals_164, primals_174, primals_180, primals_190, primals_196, primals_202, expand, slice_4, mul_1, getitem_3, view, getitem_149, permute_default_67, permute_default_68, alias_default_23, permute_default_69, permute_default_70, view_16, getitem_7, mul_3, view_18, addmm_4, view_20, getitem_11, mul_8, view_22, getitem_147, permute_default_61, permute_default_62, alias_default_21, permute_default_63, permute_default_64, view_38, getitem_17, mul_10, view_40, addmm_10, view_42, getitem_21, mul_15, view_44, getitem_145, permute_default_55, permute_default_56, alias_default_19, permute_default_57, permute_default_58, view_60, getitem_27, mul_17, view_62, addmm_16, view_64, getitem_31, mul_22, view_66, getitem_143, permute_default_49, permute_default_50, alias_default_17, permute_default_51, permute_default_52, view_82, getitem_37, mul_24, view_84, addmm_22, view_86, getitem_41, mul_29, view_88, getitem_141, permute_default_43, permute_default_44, alias_default_15, permute_default_45, permute_default_46, view_104, getitem_47, mul_31, view_106, addmm_28, view_108, getitem_51, mul_36, view_110, getitem_139, permute_default_37, permute_default_38, alias_default_13, permute_default_39, permute_default_40, view_126, getitem_57, mul_38, view_128, addmm_34, view_130, getitem_61, mul_43, view_132, getitem_137, permute_default_31, permute_default_32, alias_default_11, permute_default_33, permute_default_34, view_148, getitem_67, mul_45, view_150, addmm_40, view_152, getitem_71, mul_50, view_154, getitem_135, permute_default_25, permute_default_26, alias_default_9, permute_default_27, permute_default_28, view_170, getitem_77, mul_52, view_172, addmm_46, view_174, getitem_81, mul_57, view_176, getitem_133, permute_default_19, permute_default_20, alias_default_7, permute_default_21, permute_default_22, view_192, getitem_87, mul_59, view_194, addmm_52, view_196, getitem_91, mul_64, view_198, getitem_131, permute_default_13, permute_default_14, alias_default_5, permute_default_15, permute_default_16, view_214, getitem_97, mul_66, view_216, addmm_58, view_218, getitem_101, mul_71, view_220, getitem_129, permute_default_7, permute_default_8, alias_default_3, permute_default_9, permute_default_10, view_236, getitem_107, mul_73, view_238, addmm_64, view_240, getitem_111, mul_78, view_242, getitem_127, permute_default_1, permute_default_2, alias_default_1, permute_default_3, permute_default_4, view_258, getitem_117, mul_80, view_260, addmm_70, view_262, getitem_121, mul_85, view_264, sub_39, ne, sub_41, ne_3, ne_6, where_4, ne_8, where_6, permute_133, div_30, permute_137, permute_141, div_31, permute_145, permute_157, permute_162, permute_166, div_33, permute_170, permute_174, div_34, permute_178, permute_190, permute_195, permute_199, div_36, permute_203, permute_207, div_37, permute_211, permute_223, permute_228, permute_232, div_39, permute_236, permute_240, div_40, permute_244, permute_256, permute_261, permute_265, div_42, permute_269, permute_273, div_43, permute_277, permute_289, permute_294, permute_298, div_45, permute_302, permute_306, div_46, permute_310, permute_322, permute_327, permute_331, div_48, permute_335, permute_339, div_49, permute_343, permute_355, permute_360, permute_364, div_51, permute_368, permute_372, div_52, permute_376, permute_388, permute_393, permute_397, div_54, permute_401, permute_405, div_55, permute_409, permute_421, permute_426, permute_430, div_57, permute_434, permute_438, div_58, permute_442, permute_454, permute_459, permute_463, div_60, permute_467, permute_471, div_61, permute_475, permute_487, permute_492, permute_496, div_63, permute_500, permute_504, div_64, permute_508, permute_520, permute_525, permute_529, div_66]
    