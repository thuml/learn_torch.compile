from __future__ import annotations



def forward(self, primals_1: "f32[30522, 768]", primals_2: "f32[2, 768]", primals_3: "f32[512, 768]", primals_4: "f32[768]", primals_5: "f32[768]", primals_6: "f32[768, 768]", primals_7: "f32[768]", primals_8: "f32[768, 768]", primals_9: "f32[768]", primals_10: "f32[768, 768]", primals_11: "f32[768]", primals_12: "f32[768, 768]", primals_13: "f32[768]", primals_14: "f32[768]", primals_15: "f32[768]", primals_16: "f32[3072, 768]", primals_17: "f32[3072]", primals_18: "f32[768, 3072]", primals_19: "f32[768]", primals_20: "f32[768]", primals_21: "f32[768]", primals_22: "f32[768, 768]", primals_23: "f32[768]", primals_24: "f32[768, 768]", primals_25: "f32[768]", primals_26: "f32[768, 768]", primals_27: "f32[768]", primals_28: "f32[768, 768]", primals_29: "f32[768]", primals_30: "f32[768]", primals_31: "f32[768]", primals_32: "f32[3072, 768]", primals_33: "f32[3072]", primals_34: "f32[768, 3072]", primals_35: "f32[768]", primals_36: "f32[768]", primals_37: "f32[768]", primals_38: "f32[768, 768]", primals_39: "f32[768]", primals_40: "f32[768, 768]", primals_41: "f32[768]", primals_42: "f32[768, 768]", primals_43: "f32[768]", primals_44: "f32[768, 768]", primals_45: "f32[768]", primals_46: "f32[768]", primals_47: "f32[768]", primals_48: "f32[3072, 768]", primals_49: "f32[3072]", primals_50: "f32[768, 3072]", primals_51: "f32[768]", primals_52: "f32[768]", primals_53: "f32[768]", primals_54: "f32[768, 768]", primals_55: "f32[768]", primals_56: "f32[768, 768]", primals_57: "f32[768]", primals_58: "f32[768, 768]", primals_59: "f32[768]", primals_60: "f32[768, 768]", primals_61: "f32[768]", primals_62: "f32[768]", primals_63: "f32[768]", primals_64: "f32[3072, 768]", primals_65: "f32[3072]", primals_66: "f32[768, 3072]", primals_67: "f32[768]", primals_68: "f32[768]", primals_69: "f32[768]", primals_70: "f32[768, 768]", primals_71: "f32[768]", primals_72: "f32[768, 768]", primals_73: "f32[768]", primals_74: "f32[768, 768]", primals_75: "f32[768]", primals_76: "f32[768, 768]", primals_77: "f32[768]", primals_78: "f32[768]", primals_79: "f32[768]", primals_80: "f32[3072, 768]", primals_81: "f32[3072]", primals_82: "f32[768, 3072]", primals_83: "f32[768]", primals_84: "f32[768]", primals_85: "f32[768]", primals_86: "f32[768, 768]", primals_87: "f32[768]", primals_88: "f32[768, 768]", primals_89: "f32[768]", primals_90: "f32[768, 768]", primals_91: "f32[768]", primals_92: "f32[768, 768]", primals_93: "f32[768]", primals_94: "f32[768]", primals_95: "f32[768]", primals_96: "f32[3072, 768]", primals_97: "f32[3072]", primals_98: "f32[768, 3072]", primals_99: "f32[768]", primals_100: "f32[768]", primals_101: "f32[768]", primals_102: "f32[768, 768]", primals_103: "f32[768]", primals_104: "f32[768, 768]", primals_105: "f32[768]", primals_106: "f32[768, 768]", primals_107: "f32[768]", primals_108: "f32[768, 768]", primals_109: "f32[768]", primals_110: "f32[768]", primals_111: "f32[768]", primals_112: "f32[3072, 768]", primals_113: "f32[3072]", primals_114: "f32[768, 3072]", primals_115: "f32[768]", primals_116: "f32[768]", primals_117: "f32[768]", primals_118: "f32[768, 768]", primals_119: "f32[768]", primals_120: "f32[768, 768]", primals_121: "f32[768]", primals_122: "f32[768, 768]", primals_123: "f32[768]", primals_124: "f32[768, 768]", primals_125: "f32[768]", primals_126: "f32[768]", primals_127: "f32[768]", primals_128: "f32[3072, 768]", primals_129: "f32[3072]", primals_130: "f32[768, 3072]", primals_131: "f32[768]", primals_132: "f32[768]", primals_133: "f32[768]", primals_134: "f32[768, 768]", primals_135: "f32[768]", primals_136: "f32[768, 768]", primals_137: "f32[768]", primals_138: "f32[768, 768]", primals_139: "f32[768]", primals_140: "f32[768, 768]", primals_141: "f32[768]", primals_142: "f32[768]", primals_143: "f32[768]", primals_144: "f32[3072, 768]", primals_145: "f32[3072]", primals_146: "f32[768, 3072]", primals_147: "f32[768]", primals_148: "f32[768]", primals_149: "f32[768]", primals_150: "f32[768, 768]", primals_151: "f32[768]", primals_152: "f32[768, 768]", primals_153: "f32[768]", primals_154: "f32[768, 768]", primals_155: "f32[768]", primals_156: "f32[768, 768]", primals_157: "f32[768]", primals_158: "f32[768]", primals_159: "f32[768]", primals_160: "f32[3072, 768]", primals_161: "f32[3072]", primals_162: "f32[768, 3072]", primals_163: "f32[768]", primals_164: "f32[768]", primals_165: "f32[768]", primals_166: "f32[768, 768]", primals_167: "f32[768]", primals_168: "f32[768, 768]", primals_169: "f32[768]", primals_170: "f32[768, 768]", primals_171: "f32[768]", primals_172: "f32[768, 768]", primals_173: "f32[768]", primals_174: "f32[768]", primals_175: "f32[768]", primals_176: "f32[3072, 768]", primals_177: "f32[3072]", primals_178: "f32[768, 3072]", primals_179: "f32[768]", primals_180: "f32[768]", primals_181: "f32[768]", primals_182: "f32[768, 768]", primals_183: "f32[768]", primals_184: "f32[768, 768]", primals_185: "f32[768]", primals_186: "f32[768, 768]", primals_187: "f32[768]", primals_188: "f32[768, 768]", primals_189: "f32[768]", primals_190: "f32[768]", primals_191: "f32[768]", primals_192: "f32[3072, 768]", primals_193: "f32[3072]", primals_194: "f32[768, 3072]", primals_195: "f32[768]", primals_196: "f32[768]", primals_197: "f32[768]", primals_198: "f32[768, 768]", primals_199: "f32[768]", primals_200: "f32[768]", primals_201: "f32[768]", primals_202: "f32[30522, 768]", primals_203: "f32[30522]", primals_204: "i64[1, 512]", primals_205: "i64[1, 512]", primals_206: "i64[4, 512]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:987, code: buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
    slice_1: "i64[1, 512]" = torch.ops.aten.slice.Tensor(primals_204, 0, 0, 9223372036854775807);  primals_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:988, code: buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
    expand: "i64[4, 512]" = torch.ops.aten.expand.default(slice_1, [4, 512]);  slice_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:218, code: position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
    slice_4: "i64[1, 512]" = torch.ops.aten.slice.Tensor(primals_205, 0, 0, 9223372036854775807);  primals_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:232, code: inputs_embeds = self.word_embeddings(input_ids)
    embedding: "f32[4, 512, 768]" = torch.ops.aten.embedding.default(primals_1, primals_206, 0);  primals_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:233, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    embedding_1: "f32[4, 512, 768]" = torch.ops.aten.embedding.default(primals_2, expand);  primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:235, code: embeddings = inputs_embeds + token_type_embeddings
    add: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:237, code: position_embeddings = self.position_embeddings(position_ids)
    embedding_2: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(primals_3, slice_4);  primals_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:238, code: embeddings += position_embeddings
    add_1: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:239, code: embeddings = self.LayerNorm(embeddings)
    var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
    getitem: "f32[4, 512, 1]" = var_mean[0]
    getitem_1: "f32[4, 512, 1]" = var_mean[1];  var_mean = None
    add_2: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
    rsqrt: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
    sub_1: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
    mul_1: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = None
    mul_2: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_1, primals_4)
    add_3: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_2, primals_5);  mul_2 = primals_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_3, [2048, 768])
    permute: "f32[768, 768]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
    
    # No stacktrace found for following nodes
    mm_default_59: "f32[2048, 768]" = torch.ops.aten.mm.default(view, permute)
    add_tensor_59: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_59, primals_7);  mm_default_59 = primals_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_1: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_59, [4, 512, 768]);  add_tensor_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_1: "f32[768, 768]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
    
    # No stacktrace found for following nodes
    mm_default_58: "f32[2048, 768]" = torch.ops.aten.mm.default(view, permute_1)
    add_tensor_58: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_58, primals_9);  mm_default_58 = primals_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_3: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_58, [4, 512, 768]);  add_tensor_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_4: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_3, [4, 512, 12, 64]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_2: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_3: "f32[768, 768]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
    
    # No stacktrace found for following nodes
    mm_default_57: "f32[2048, 768]" = torch.ops.aten.mm.default(view, permute_3)
    add_tensor_57: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_57, primals_11);  mm_default_57 = primals_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_6: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_57, [4, 512, 768]);  add_tensor_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_7: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_6, [4, 512, 12, 64]);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_4: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_8: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_1, [4, 512, 12, 64]);  view_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_5: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_6: "f32[4, 12, 64, 512]" = torch.ops.aten.permute.default(permute_2, [0, 1, 3, 2]);  permute_2 = None
    expand_1: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_5, [4, 12, 512, 64]);  permute_5 = None
    clone_1: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
    view_9: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_1, [48, 512, 64]);  clone_1 = None
    expand_2: "f32[4, 12, 64, 512]" = torch.ops.aten.expand.default(permute_6, [4, 12, 64, 512]);  permute_6 = None
    clone_2: "f32[4, 12, 64, 512]" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
    view_10: "f32[48, 64, 512]" = torch.ops.aten.reshape.default(clone_2, [48, 64, 512]);  clone_2 = None
    bmm: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_9, view_10)
    view_11: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm, [4, 12, 512, 512]);  bmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:352, code: attention_scores = attention_scores + attention_mask
    div: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_11, 8.0);  view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax: "f32[4, 12, 512, 1]" = torch.ops.aten.amax.default(div, [-1], True)
    sub_2: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(div, amax);  div = amax = None
    exp: "f32[4, 12, 512, 512]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
    sum_1: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div_1: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(div_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: attention_probs = self.dropout(attention_probs)
    clone_3: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_1);  div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_3: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_3, [4, 12, 512, 512]);  clone_3 = None
    view_12: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(expand_3, [48, 512, 512]);  expand_3 = None
    expand_4: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_4, [4, 12, 512, 64]);  permute_4 = None
    clone_4: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
    view_13: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_4, [48, 512, 64]);  clone_4 = None
    bmm_1: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_12, view_13)
    view_14: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_1, [4, 12, 512, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_7: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
    clone_5: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_15: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_5, [4, 512, 768]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_16: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_15, [2048, 768]);  view_15 = None
    permute_8: "f32[768, 768]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
    
    # No stacktrace found for following nodes
    mm_default_56: "f32[2048, 768]" = torch.ops.aten.mm.default(view_16, permute_8)
    add_tensor_56: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_56, primals_13);  mm_default_56 = primals_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_17: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_56, [4, 512, 768]);  add_tensor_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_5: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_17, add_3);  view_17 = add_3 = None
    var_mean_1 = torch.ops.aten.var_mean.correction(add_5, [2], correction = 0, keepdim = True)
    getitem_2: "f32[4, 512, 1]" = var_mean_1[0]
    getitem_3: "f32[4, 512, 1]" = var_mean_1[1];  var_mean_1 = None
    add_6: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
    rsqrt_1: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_3: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_5, getitem_3);  add_5 = getitem_3 = None
    mul_3: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = None
    mul_4: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_3, primals_14)
    add_7: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_4, primals_15);  mul_4 = primals_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_18: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_7, [2048, 768])
    permute_9: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
    addmm_4: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_17, view_18, permute_9);  primals_17 = None
    view_19: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_4, [4, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_5: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.5)
    mul_6: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476);  view_19 = None
    erf: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
    add_8: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_7: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_5, add_8);  mul_5 = add_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_20: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_7, [2048, 3072]);  mul_7 = None
    permute_10: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
    
    # No stacktrace found for following nodes
    mm_default_55: "f32[2048, 768]" = torch.ops.aten.mm.default(view_20, permute_10)
    add_tensor_55: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_55, primals_19);  mm_default_55 = primals_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_21: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_55, [4, 512, 768]);  add_tensor_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_9: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_21, add_7);  view_21 = add_7 = None
    var_mean_2 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
    getitem_4: "f32[4, 512, 1]" = var_mean_2[0]
    getitem_5: "f32[4, 512, 1]" = var_mean_2[1];  var_mean_2 = None
    add_10: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
    rsqrt_2: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
    sub_4: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_9, getitem_5);  add_9 = getitem_5 = None
    mul_8: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = None
    mul_9: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_8, primals_20)
    add_11: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_9, primals_21);  mul_9 = primals_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_22: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_11, [2048, 768])
    permute_11: "f32[768, 768]" = torch.ops.aten.permute.default(primals_22, [1, 0]);  primals_22 = None
    
    # No stacktrace found for following nodes
    mm_default_54: "f32[2048, 768]" = torch.ops.aten.mm.default(view_22, permute_11)
    add_tensor_54: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_54, primals_23);  mm_default_54 = primals_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_23: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_54, [4, 512, 768]);  add_tensor_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_12: "f32[768, 768]" = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
    
    # No stacktrace found for following nodes
    mm_default_53: "f32[2048, 768]" = torch.ops.aten.mm.default(view_22, permute_12)
    add_tensor_53: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_53, primals_25);  mm_default_53 = primals_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_25: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_53, [4, 512, 768]);  add_tensor_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_26: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_25, [4, 512, 12, 64]);  view_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_13: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_14: "f32[768, 768]" = torch.ops.aten.permute.default(primals_26, [1, 0]);  primals_26 = None
    
    # No stacktrace found for following nodes
    mm_default_52: "f32[2048, 768]" = torch.ops.aten.mm.default(view_22, permute_14)
    add_tensor_52: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_52, primals_27);  mm_default_52 = primals_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_28: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_52, [4, 512, 768]);  add_tensor_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_29: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_28, [4, 512, 12, 64]);  view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_15: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_30: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_23, [4, 512, 12, 64]);  view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_16: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_17: "f32[4, 12, 64, 512]" = torch.ops.aten.permute.default(permute_13, [0, 1, 3, 2]);  permute_13 = None
    expand_5: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_16, [4, 12, 512, 64]);  permute_16 = None
    clone_8: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
    view_31: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_8, [48, 512, 64]);  clone_8 = None
    expand_6: "f32[4, 12, 64, 512]" = torch.ops.aten.expand.default(permute_17, [4, 12, 64, 512]);  permute_17 = None
    clone_9: "f32[4, 12, 64, 512]" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
    view_32: "f32[48, 64, 512]" = torch.ops.aten.reshape.default(clone_9, [48, 64, 512]);  clone_9 = None
    bmm_2: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_31, view_32)
    view_33: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_2, [4, 12, 512, 512]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:352, code: attention_scores = attention_scores + attention_mask
    div_2: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_33, 8.0);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_1: "f32[4, 12, 512, 1]" = torch.ops.aten.amax.default(div_2, [-1], True)
    sub_5: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(div_2, amax_1);  div_2 = amax_1 = None
    exp_1: "f32[4, 12, 512, 512]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
    sum_2: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_3: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    alias_1: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(div_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: attention_probs = self.dropout(attention_probs)
    clone_10: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_3);  div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_7: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_10, [4, 12, 512, 512]);  clone_10 = None
    view_34: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(expand_7, [48, 512, 512]);  expand_7 = None
    expand_8: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_15, [4, 12, 512, 64]);  permute_15 = None
    clone_11: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
    view_35: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_11, [48, 512, 64]);  clone_11 = None
    bmm_3: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_34, view_35)
    view_36: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_3, [4, 12, 512, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_18: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None
    clone_12: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_37: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_12, [4, 512, 768]);  clone_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_38: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_37, [2048, 768]);  view_37 = None
    permute_19: "f32[768, 768]" = torch.ops.aten.permute.default(primals_28, [1, 0]);  primals_28 = None
    
    # No stacktrace found for following nodes
    mm_default_51: "f32[2048, 768]" = torch.ops.aten.mm.default(view_38, permute_19)
    add_tensor_51: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_51, primals_29);  mm_default_51 = primals_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_39: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_51, [4, 512, 768]);  add_tensor_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_13: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_39, add_11);  view_39 = add_11 = None
    var_mean_3 = torch.ops.aten.var_mean.correction(add_13, [2], correction = 0, keepdim = True)
    getitem_6: "f32[4, 512, 1]" = var_mean_3[0]
    getitem_7: "f32[4, 512, 1]" = var_mean_3[1];  var_mean_3 = None
    add_14: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
    rsqrt_3: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
    sub_6: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_13, getitem_7);  add_13 = getitem_7 = None
    mul_10: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_3);  sub_6 = None
    mul_11: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_10, primals_30)
    add_15: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_11, primals_31);  mul_11 = primals_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_40: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_15, [2048, 768])
    permute_20: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_32, [1, 0]);  primals_32 = None
    addmm_10: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_33, view_40, permute_20);  primals_33 = None
    view_41: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_10, [4, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_12: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, 0.5)
    mul_13: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476);  view_41 = None
    erf_1: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
    add_16: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_14: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_12, add_16);  mul_12 = add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_42: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_14, [2048, 3072]);  mul_14 = None
    permute_21: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
    
    # No stacktrace found for following nodes
    mm_default_50: "f32[2048, 768]" = torch.ops.aten.mm.default(view_42, permute_21)
    add_tensor_50: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_50, primals_35);  mm_default_50 = primals_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_43: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_50, [4, 512, 768]);  add_tensor_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_17: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_43, add_15);  view_43 = add_15 = None
    var_mean_4 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
    getitem_8: "f32[4, 512, 1]" = var_mean_4[0]
    getitem_9: "f32[4, 512, 1]" = var_mean_4[1];  var_mean_4 = None
    add_18: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
    rsqrt_4: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    sub_7: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_17, getitem_9);  add_17 = getitem_9 = None
    mul_15: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_4);  sub_7 = None
    mul_16: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_15, primals_36)
    add_19: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_16, primals_37);  mul_16 = primals_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_44: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_19, [2048, 768])
    permute_22: "f32[768, 768]" = torch.ops.aten.permute.default(primals_38, [1, 0]);  primals_38 = None
    
    # No stacktrace found for following nodes
    mm_default_49: "f32[2048, 768]" = torch.ops.aten.mm.default(view_44, permute_22)
    add_tensor_49: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_49, primals_39);  mm_default_49 = primals_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_45: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_49, [4, 512, 768]);  add_tensor_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_23: "f32[768, 768]" = torch.ops.aten.permute.default(primals_40, [1, 0]);  primals_40 = None
    
    # No stacktrace found for following nodes
    mm_default_48: "f32[2048, 768]" = torch.ops.aten.mm.default(view_44, permute_23)
    add_tensor_48: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_48, primals_41);  mm_default_48 = primals_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_47: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_48, [4, 512, 768]);  add_tensor_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_48: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_47, [4, 512, 12, 64]);  view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_24: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_25: "f32[768, 768]" = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
    
    # No stacktrace found for following nodes
    mm_default_47: "f32[2048, 768]" = torch.ops.aten.mm.default(view_44, permute_25)
    add_tensor_47: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_47, primals_43);  mm_default_47 = primals_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_50: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_47, [4, 512, 768]);  add_tensor_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_51: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_50, [4, 512, 12, 64]);  view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_26: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_52: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_45, [4, 512, 12, 64]);  view_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_27: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_28: "f32[4, 12, 64, 512]" = torch.ops.aten.permute.default(permute_24, [0, 1, 3, 2]);  permute_24 = None
    expand_9: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_27, [4, 12, 512, 64]);  permute_27 = None
    clone_15: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
    view_53: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_15, [48, 512, 64]);  clone_15 = None
    expand_10: "f32[4, 12, 64, 512]" = torch.ops.aten.expand.default(permute_28, [4, 12, 64, 512]);  permute_28 = None
    clone_16: "f32[4, 12, 64, 512]" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
    view_54: "f32[48, 64, 512]" = torch.ops.aten.reshape.default(clone_16, [48, 64, 512]);  clone_16 = None
    bmm_4: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_53, view_54)
    view_55: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_4, [4, 12, 512, 512]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:352, code: attention_scores = attention_scores + attention_mask
    div_4: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_55, 8.0);  view_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_2: "f32[4, 12, 512, 1]" = torch.ops.aten.amax.default(div_4, [-1], True)
    sub_8: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(div_4, amax_2);  div_4 = amax_2 = None
    exp_2: "f32[4, 12, 512, 512]" = torch.ops.aten.exp.default(sub_8);  sub_8 = None
    sum_3: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_5: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    alias_2: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(div_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: attention_probs = self.dropout(attention_probs)
    clone_17: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_5);  div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_11: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_17, [4, 12, 512, 512]);  clone_17 = None
    view_56: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(expand_11, [48, 512, 512]);  expand_11 = None
    expand_12: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_26, [4, 12, 512, 64]);  permute_26 = None
    clone_18: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
    view_57: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_18, [48, 512, 64]);  clone_18 = None
    bmm_5: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_56, view_57)
    view_58: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_5, [4, 12, 512, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_29: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
    clone_19: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_59: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_19, [4, 512, 768]);  clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_60: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_59, [2048, 768]);  view_59 = None
    permute_30: "f32[768, 768]" = torch.ops.aten.permute.default(primals_44, [1, 0]);  primals_44 = None
    
    # No stacktrace found for following nodes
    mm_default_46: "f32[2048, 768]" = torch.ops.aten.mm.default(view_60, permute_30)
    add_tensor_46: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_46, primals_45);  mm_default_46 = primals_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_61: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_46, [4, 512, 768]);  add_tensor_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_21: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_61, add_19);  view_61 = add_19 = None
    var_mean_5 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
    getitem_10: "f32[4, 512, 1]" = var_mean_5[0]
    getitem_11: "f32[4, 512, 1]" = var_mean_5[1];  var_mean_5 = None
    add_22: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
    rsqrt_5: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
    sub_9: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_21, getitem_11);  add_21 = getitem_11 = None
    mul_17: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_5);  sub_9 = None
    mul_18: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_17, primals_46)
    add_23: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_18, primals_47);  mul_18 = primals_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_62: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_23, [2048, 768])
    permute_31: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_48, [1, 0]);  primals_48 = None
    addmm_16: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_49, view_62, permute_31);  primals_49 = None
    view_63: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_16, [4, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_19: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, 0.5)
    mul_20: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476);  view_63 = None
    erf_2: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_20);  mul_20 = None
    add_24: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_21: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_19, add_24);  mul_19 = add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_64: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_21, [2048, 3072]);  mul_21 = None
    permute_32: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_50, [1, 0]);  primals_50 = None
    
    # No stacktrace found for following nodes
    mm_default_45: "f32[2048, 768]" = torch.ops.aten.mm.default(view_64, permute_32)
    add_tensor_45: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_45, primals_51);  mm_default_45 = primals_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_65: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_45, [4, 512, 768]);  add_tensor_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_25: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_65, add_23);  view_65 = add_23 = None
    var_mean_6 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
    getitem_12: "f32[4, 512, 1]" = var_mean_6[0]
    getitem_13: "f32[4, 512, 1]" = var_mean_6[1];  var_mean_6 = None
    add_26: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
    rsqrt_6: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
    sub_10: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_25, getitem_13);  add_25 = getitem_13 = None
    mul_22: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_6);  sub_10 = None
    mul_23: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_22, primals_52)
    add_27: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_23, primals_53);  mul_23 = primals_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_66: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_27, [2048, 768])
    permute_33: "f32[768, 768]" = torch.ops.aten.permute.default(primals_54, [1, 0]);  primals_54 = None
    
    # No stacktrace found for following nodes
    mm_default_44: "f32[2048, 768]" = torch.ops.aten.mm.default(view_66, permute_33)
    add_tensor_44: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_44, primals_55);  mm_default_44 = primals_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_67: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_44, [4, 512, 768]);  add_tensor_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_34: "f32[768, 768]" = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
    
    # No stacktrace found for following nodes
    mm_default_43: "f32[2048, 768]" = torch.ops.aten.mm.default(view_66, permute_34)
    add_tensor_43: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_43, primals_57);  mm_default_43 = primals_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_69: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_43, [4, 512, 768]);  add_tensor_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_70: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_69, [4, 512, 12, 64]);  view_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_35: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_36: "f32[768, 768]" = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
    
    # No stacktrace found for following nodes
    mm_default_42: "f32[2048, 768]" = torch.ops.aten.mm.default(view_66, permute_36)
    add_tensor_42: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_42, primals_59);  mm_default_42 = primals_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_72: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_42, [4, 512, 768]);  add_tensor_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_73: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_72, [4, 512, 12, 64]);  view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_37: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_74: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_67, [4, 512, 12, 64]);  view_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_38: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_39: "f32[4, 12, 64, 512]" = torch.ops.aten.permute.default(permute_35, [0, 1, 3, 2]);  permute_35 = None
    expand_13: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_38, [4, 12, 512, 64]);  permute_38 = None
    clone_22: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
    view_75: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_22, [48, 512, 64]);  clone_22 = None
    expand_14: "f32[4, 12, 64, 512]" = torch.ops.aten.expand.default(permute_39, [4, 12, 64, 512]);  permute_39 = None
    clone_23: "f32[4, 12, 64, 512]" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
    view_76: "f32[48, 64, 512]" = torch.ops.aten.reshape.default(clone_23, [48, 64, 512]);  clone_23 = None
    bmm_6: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_75, view_76)
    view_77: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_6, [4, 12, 512, 512]);  bmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:352, code: attention_scores = attention_scores + attention_mask
    div_6: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_77, 8.0);  view_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_3: "f32[4, 12, 512, 1]" = torch.ops.aten.amax.default(div_6, [-1], True)
    sub_11: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(div_6, amax_3);  div_6 = amax_3 = None
    exp_3: "f32[4, 12, 512, 512]" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
    sum_4: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_7: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    alias_3: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(div_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: attention_probs = self.dropout(attention_probs)
    clone_24: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_7);  div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_15: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_24, [4, 12, 512, 512]);  clone_24 = None
    view_78: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(expand_15, [48, 512, 512]);  expand_15 = None
    expand_16: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_37, [4, 12, 512, 64]);  permute_37 = None
    clone_25: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
    view_79: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_25, [48, 512, 64]);  clone_25 = None
    bmm_7: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_78, view_79)
    view_80: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_7, [4, 12, 512, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_40: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_80, [0, 2, 1, 3]);  view_80 = None
    clone_26: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_81: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_26, [4, 512, 768]);  clone_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_82: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_81, [2048, 768]);  view_81 = None
    permute_41: "f32[768, 768]" = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
    
    # No stacktrace found for following nodes
    mm_default_41: "f32[2048, 768]" = torch.ops.aten.mm.default(view_82, permute_41)
    add_tensor_41: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_41, primals_61);  mm_default_41 = primals_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_83: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_41, [4, 512, 768]);  add_tensor_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_29: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_83, add_27);  view_83 = add_27 = None
    var_mean_7 = torch.ops.aten.var_mean.correction(add_29, [2], correction = 0, keepdim = True)
    getitem_14: "f32[4, 512, 1]" = var_mean_7[0]
    getitem_15: "f32[4, 512, 1]" = var_mean_7[1];  var_mean_7 = None
    add_30: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
    rsqrt_7: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
    sub_12: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_29, getitem_15);  add_29 = getitem_15 = None
    mul_24: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_7);  sub_12 = None
    mul_25: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_24, primals_62)
    add_31: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_25, primals_63);  mul_25 = primals_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_84: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_31, [2048, 768])
    permute_42: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
    addmm_22: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_65, view_84, permute_42);  primals_65 = None
    view_85: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_22, [4, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_26: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, 0.5)
    mul_27: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476);  view_85 = None
    erf_3: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_27);  mul_27 = None
    add_32: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_28: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_26, add_32);  mul_26 = add_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_86: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_28, [2048, 3072]);  mul_28 = None
    permute_43: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
    
    # No stacktrace found for following nodes
    mm_default_40: "f32[2048, 768]" = torch.ops.aten.mm.default(view_86, permute_43)
    add_tensor_40: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_40, primals_67);  mm_default_40 = primals_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_87: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_40, [4, 512, 768]);  add_tensor_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_33: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_87, add_31);  view_87 = add_31 = None
    var_mean_8 = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
    getitem_16: "f32[4, 512, 1]" = var_mean_8[0]
    getitem_17: "f32[4, 512, 1]" = var_mean_8[1];  var_mean_8 = None
    add_34: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
    rsqrt_8: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
    sub_13: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_33, getitem_17);  add_33 = getitem_17 = None
    mul_29: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_8);  sub_13 = None
    mul_30: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_29, primals_68)
    add_35: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_30, primals_69);  mul_30 = primals_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_88: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_35, [2048, 768])
    permute_44: "f32[768, 768]" = torch.ops.aten.permute.default(primals_70, [1, 0]);  primals_70 = None
    
    # No stacktrace found for following nodes
    mm_default_39: "f32[2048, 768]" = torch.ops.aten.mm.default(view_88, permute_44)
    add_tensor_39: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_39, primals_71);  mm_default_39 = primals_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_89: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_39, [4, 512, 768]);  add_tensor_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_45: "f32[768, 768]" = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
    
    # No stacktrace found for following nodes
    mm_default_38: "f32[2048, 768]" = torch.ops.aten.mm.default(view_88, permute_45)
    add_tensor_38: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_38, primals_73);  mm_default_38 = primals_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_91: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_38, [4, 512, 768]);  add_tensor_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_92: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_91, [4, 512, 12, 64]);  view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_46: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_47: "f32[768, 768]" = torch.ops.aten.permute.default(primals_74, [1, 0]);  primals_74 = None
    
    # No stacktrace found for following nodes
    mm_default_37: "f32[2048, 768]" = torch.ops.aten.mm.default(view_88, permute_47)
    add_tensor_37: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_37, primals_75);  mm_default_37 = primals_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_94: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_37, [4, 512, 768]);  add_tensor_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_95: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_94, [4, 512, 12, 64]);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_48: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_96: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_89, [4, 512, 12, 64]);  view_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_49: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_50: "f32[4, 12, 64, 512]" = torch.ops.aten.permute.default(permute_46, [0, 1, 3, 2]);  permute_46 = None
    expand_17: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_49, [4, 12, 512, 64]);  permute_49 = None
    clone_29: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
    view_97: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_29, [48, 512, 64]);  clone_29 = None
    expand_18: "f32[4, 12, 64, 512]" = torch.ops.aten.expand.default(permute_50, [4, 12, 64, 512]);  permute_50 = None
    clone_30: "f32[4, 12, 64, 512]" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
    view_98: "f32[48, 64, 512]" = torch.ops.aten.reshape.default(clone_30, [48, 64, 512]);  clone_30 = None
    bmm_8: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_97, view_98)
    view_99: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_8, [4, 12, 512, 512]);  bmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:352, code: attention_scores = attention_scores + attention_mask
    div_8: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_99, 8.0);  view_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_4: "f32[4, 12, 512, 1]" = torch.ops.aten.amax.default(div_8, [-1], True)
    sub_14: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(div_8, amax_4);  div_8 = amax_4 = None
    exp_4: "f32[4, 12, 512, 512]" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
    sum_5: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_9: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    alias_4: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(div_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: attention_probs = self.dropout(attention_probs)
    clone_31: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_9);  div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_19: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_31, [4, 12, 512, 512]);  clone_31 = None
    view_100: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(expand_19, [48, 512, 512]);  expand_19 = None
    expand_20: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_48, [4, 12, 512, 64]);  permute_48 = None
    clone_32: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
    view_101: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_32, [48, 512, 64]);  clone_32 = None
    bmm_9: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_100, view_101)
    view_102: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_9, [4, 12, 512, 64]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_51: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
    clone_33: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_103: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_33, [4, 512, 768]);  clone_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_104: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_103, [2048, 768]);  view_103 = None
    permute_52: "f32[768, 768]" = torch.ops.aten.permute.default(primals_76, [1, 0]);  primals_76 = None
    
    # No stacktrace found for following nodes
    mm_default_36: "f32[2048, 768]" = torch.ops.aten.mm.default(view_104, permute_52)
    add_tensor_36: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_36, primals_77);  mm_default_36 = primals_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_105: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_36, [4, 512, 768]);  add_tensor_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_37: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_105, add_35);  view_105 = add_35 = None
    var_mean_9 = torch.ops.aten.var_mean.correction(add_37, [2], correction = 0, keepdim = True)
    getitem_18: "f32[4, 512, 1]" = var_mean_9[0]
    getitem_19: "f32[4, 512, 1]" = var_mean_9[1];  var_mean_9 = None
    add_38: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
    rsqrt_9: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
    sub_15: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_37, getitem_19);  add_37 = getitem_19 = None
    mul_31: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_9);  sub_15 = None
    mul_32: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_31, primals_78)
    add_39: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_32, primals_79);  mul_32 = primals_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_106: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_39, [2048, 768])
    permute_53: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_80, [1, 0]);  primals_80 = None
    addmm_28: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_81, view_106, permute_53);  primals_81 = None
    view_107: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_28, [4, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_33: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.5)
    mul_34: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476);  view_107 = None
    erf_4: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_34);  mul_34 = None
    add_40: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_35: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_33, add_40);  mul_33 = add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_108: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_35, [2048, 3072]);  mul_35 = None
    permute_54: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_82, [1, 0]);  primals_82 = None
    
    # No stacktrace found for following nodes
    mm_default_35: "f32[2048, 768]" = torch.ops.aten.mm.default(view_108, permute_54)
    add_tensor_35: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_35, primals_83);  mm_default_35 = primals_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_109: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_35, [4, 512, 768]);  add_tensor_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_41: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_109, add_39);  view_109 = add_39 = None
    var_mean_10 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
    getitem_20: "f32[4, 512, 1]" = var_mean_10[0]
    getitem_21: "f32[4, 512, 1]" = var_mean_10[1];  var_mean_10 = None
    add_42: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
    rsqrt_10: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_16: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_41, getitem_21);  add_41 = getitem_21 = None
    mul_36: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_10);  sub_16 = None
    mul_37: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_36, primals_84)
    add_43: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_37, primals_85);  mul_37 = primals_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_110: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_43, [2048, 768])
    permute_55: "f32[768, 768]" = torch.ops.aten.permute.default(primals_86, [1, 0]);  primals_86 = None
    
    # No stacktrace found for following nodes
    mm_default_34: "f32[2048, 768]" = torch.ops.aten.mm.default(view_110, permute_55)
    add_tensor_34: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_34, primals_87);  mm_default_34 = primals_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_111: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_34, [4, 512, 768]);  add_tensor_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_56: "f32[768, 768]" = torch.ops.aten.permute.default(primals_88, [1, 0]);  primals_88 = None
    
    # No stacktrace found for following nodes
    mm_default_33: "f32[2048, 768]" = torch.ops.aten.mm.default(view_110, permute_56)
    add_tensor_33: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_33, primals_89);  mm_default_33 = primals_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_113: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_33, [4, 512, 768]);  add_tensor_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_114: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_113, [4, 512, 12, 64]);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_57: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_58: "f32[768, 768]" = torch.ops.aten.permute.default(primals_90, [1, 0]);  primals_90 = None
    
    # No stacktrace found for following nodes
    mm_default_32: "f32[2048, 768]" = torch.ops.aten.mm.default(view_110, permute_58)
    add_tensor_32: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_32, primals_91);  mm_default_32 = primals_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_116: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_32, [4, 512, 768]);  add_tensor_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_117: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_116, [4, 512, 12, 64]);  view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_59: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_118: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_111, [4, 512, 12, 64]);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_60: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_61: "f32[4, 12, 64, 512]" = torch.ops.aten.permute.default(permute_57, [0, 1, 3, 2]);  permute_57 = None
    expand_21: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_60, [4, 12, 512, 64]);  permute_60 = None
    clone_36: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
    view_119: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_36, [48, 512, 64]);  clone_36 = None
    expand_22: "f32[4, 12, 64, 512]" = torch.ops.aten.expand.default(permute_61, [4, 12, 64, 512]);  permute_61 = None
    clone_37: "f32[4, 12, 64, 512]" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
    view_120: "f32[48, 64, 512]" = torch.ops.aten.reshape.default(clone_37, [48, 64, 512]);  clone_37 = None
    bmm_10: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_119, view_120)
    view_121: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_10, [4, 12, 512, 512]);  bmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:352, code: attention_scores = attention_scores + attention_mask
    div_10: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_121, 8.0);  view_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_5: "f32[4, 12, 512, 1]" = torch.ops.aten.amax.default(div_10, [-1], True)
    sub_17: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(div_10, amax_5);  div_10 = amax_5 = None
    exp_5: "f32[4, 12, 512, 512]" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
    sum_6: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_11: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    alias_5: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(div_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: attention_probs = self.dropout(attention_probs)
    clone_38: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_11);  div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_23: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_38, [4, 12, 512, 512]);  clone_38 = None
    view_122: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(expand_23, [48, 512, 512]);  expand_23 = None
    expand_24: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_59, [4, 12, 512, 64]);  permute_59 = None
    clone_39: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
    view_123: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_39, [48, 512, 64]);  clone_39 = None
    bmm_11: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_122, view_123)
    view_124: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_11, [4, 12, 512, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_62: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None
    clone_40: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_125: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_40, [4, 512, 768]);  clone_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_126: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_125, [2048, 768]);  view_125 = None
    permute_63: "f32[768, 768]" = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
    
    # No stacktrace found for following nodes
    mm_default_31: "f32[2048, 768]" = torch.ops.aten.mm.default(view_126, permute_63)
    add_tensor_31: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_31, primals_93);  mm_default_31 = primals_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_127: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_31, [4, 512, 768]);  add_tensor_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_45: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_127, add_43);  view_127 = add_43 = None
    var_mean_11 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
    getitem_22: "f32[4, 512, 1]" = var_mean_11[0]
    getitem_23: "f32[4, 512, 1]" = var_mean_11[1];  var_mean_11 = None
    add_46: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
    rsqrt_11: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
    sub_18: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_45, getitem_23);  add_45 = getitem_23 = None
    mul_38: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_11);  sub_18 = None
    mul_39: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_38, primals_94)
    add_47: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_39, primals_95);  mul_39 = primals_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_128: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_47, [2048, 768])
    permute_64: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_96, [1, 0]);  primals_96 = None
    addmm_34: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_97, view_128, permute_64);  primals_97 = None
    view_129: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_34, [4, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_40: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, 0.5)
    mul_41: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476);  view_129 = None
    erf_5: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_41);  mul_41 = None
    add_48: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_42: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_40, add_48);  mul_40 = add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_130: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_42, [2048, 3072]);  mul_42 = None
    permute_65: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_98, [1, 0]);  primals_98 = None
    
    # No stacktrace found for following nodes
    mm_default_30: "f32[2048, 768]" = torch.ops.aten.mm.default(view_130, permute_65)
    add_tensor_30: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_30, primals_99);  mm_default_30 = primals_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_131: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_30, [4, 512, 768]);  add_tensor_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_49: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_131, add_47);  view_131 = add_47 = None
    var_mean_12 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
    getitem_24: "f32[4, 512, 1]" = var_mean_12[0]
    getitem_25: "f32[4, 512, 1]" = var_mean_12[1];  var_mean_12 = None
    add_50: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
    rsqrt_12: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
    sub_19: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_49, getitem_25);  add_49 = getitem_25 = None
    mul_43: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_12);  sub_19 = None
    mul_44: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_43, primals_100)
    add_51: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_44, primals_101);  mul_44 = primals_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_132: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_51, [2048, 768])
    permute_66: "f32[768, 768]" = torch.ops.aten.permute.default(primals_102, [1, 0]);  primals_102 = None
    
    # No stacktrace found for following nodes
    mm_default_29: "f32[2048, 768]" = torch.ops.aten.mm.default(view_132, permute_66)
    add_tensor_29: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_29, primals_103);  mm_default_29 = primals_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_133: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_29, [4, 512, 768]);  add_tensor_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_67: "f32[768, 768]" = torch.ops.aten.permute.default(primals_104, [1, 0]);  primals_104 = None
    
    # No stacktrace found for following nodes
    mm_default_28: "f32[2048, 768]" = torch.ops.aten.mm.default(view_132, permute_67)
    add_tensor_28: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_28, primals_105);  mm_default_28 = primals_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_135: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_28, [4, 512, 768]);  add_tensor_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_136: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_135, [4, 512, 12, 64]);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_68: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_69: "f32[768, 768]" = torch.ops.aten.permute.default(primals_106, [1, 0]);  primals_106 = None
    
    # No stacktrace found for following nodes
    mm_default_27: "f32[2048, 768]" = torch.ops.aten.mm.default(view_132, permute_69)
    add_tensor_27: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_27, primals_107);  mm_default_27 = primals_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_138: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_27, [4, 512, 768]);  add_tensor_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_139: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_138, [4, 512, 12, 64]);  view_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_70: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_139, [0, 2, 1, 3]);  view_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_140: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_133, [4, 512, 12, 64]);  view_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_71: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_72: "f32[4, 12, 64, 512]" = torch.ops.aten.permute.default(permute_68, [0, 1, 3, 2]);  permute_68 = None
    expand_25: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_71, [4, 12, 512, 64]);  permute_71 = None
    clone_43: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
    view_141: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_43, [48, 512, 64]);  clone_43 = None
    expand_26: "f32[4, 12, 64, 512]" = torch.ops.aten.expand.default(permute_72, [4, 12, 64, 512]);  permute_72 = None
    clone_44: "f32[4, 12, 64, 512]" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
    view_142: "f32[48, 64, 512]" = torch.ops.aten.reshape.default(clone_44, [48, 64, 512]);  clone_44 = None
    bmm_12: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_141, view_142)
    view_143: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_12, [4, 12, 512, 512]);  bmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:352, code: attention_scores = attention_scores + attention_mask
    div_12: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_143, 8.0);  view_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_6: "f32[4, 12, 512, 1]" = torch.ops.aten.amax.default(div_12, [-1], True)
    sub_20: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(div_12, amax_6);  div_12 = amax_6 = None
    exp_6: "f32[4, 12, 512, 512]" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
    sum_7: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_13: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    alias_6: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(div_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: attention_probs = self.dropout(attention_probs)
    clone_45: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_13);  div_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_27: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_45, [4, 12, 512, 512]);  clone_45 = None
    view_144: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(expand_27, [48, 512, 512]);  expand_27 = None
    expand_28: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_70, [4, 12, 512, 64]);  permute_70 = None
    clone_46: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
    view_145: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_46, [48, 512, 64]);  clone_46 = None
    bmm_13: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_144, view_145)
    view_146: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_13, [4, 12, 512, 64]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_73: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
    clone_47: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_147: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_47, [4, 512, 768]);  clone_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_148: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_147, [2048, 768]);  view_147 = None
    permute_74: "f32[768, 768]" = torch.ops.aten.permute.default(primals_108, [1, 0]);  primals_108 = None
    
    # No stacktrace found for following nodes
    mm_default_26: "f32[2048, 768]" = torch.ops.aten.mm.default(view_148, permute_74)
    add_tensor_26: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_26, primals_109);  mm_default_26 = primals_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_149: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_26, [4, 512, 768]);  add_tensor_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_53: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_149, add_51);  view_149 = add_51 = None
    var_mean_13 = torch.ops.aten.var_mean.correction(add_53, [2], correction = 0, keepdim = True)
    getitem_26: "f32[4, 512, 1]" = var_mean_13[0]
    getitem_27: "f32[4, 512, 1]" = var_mean_13[1];  var_mean_13 = None
    add_54: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
    rsqrt_13: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
    sub_21: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_53, getitem_27);  add_53 = getitem_27 = None
    mul_45: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_13);  sub_21 = None
    mul_46: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_45, primals_110)
    add_55: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_46, primals_111);  mul_46 = primals_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_150: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_55, [2048, 768])
    permute_75: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_112, [1, 0]);  primals_112 = None
    addmm_40: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_113, view_150, permute_75);  primals_113 = None
    view_151: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_40, [4, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_47: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, 0.5)
    mul_48: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476);  view_151 = None
    erf_6: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_48);  mul_48 = None
    add_56: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_49: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_47, add_56);  mul_47 = add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_152: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_49, [2048, 3072]);  mul_49 = None
    permute_76: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_114, [1, 0]);  primals_114 = None
    
    # No stacktrace found for following nodes
    mm_default_25: "f32[2048, 768]" = torch.ops.aten.mm.default(view_152, permute_76)
    add_tensor_25: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_25, primals_115);  mm_default_25 = primals_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_153: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_25, [4, 512, 768]);  add_tensor_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_57: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_153, add_55);  view_153 = add_55 = None
    var_mean_14 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
    getitem_28: "f32[4, 512, 1]" = var_mean_14[0]
    getitem_29: "f32[4, 512, 1]" = var_mean_14[1];  var_mean_14 = None
    add_58: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
    rsqrt_14: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
    sub_22: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_57, getitem_29);  add_57 = getitem_29 = None
    mul_50: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_14);  sub_22 = None
    mul_51: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_50, primals_116)
    add_59: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_51, primals_117);  mul_51 = primals_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_154: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_59, [2048, 768])
    permute_77: "f32[768, 768]" = torch.ops.aten.permute.default(primals_118, [1, 0]);  primals_118 = None
    
    # No stacktrace found for following nodes
    mm_default_24: "f32[2048, 768]" = torch.ops.aten.mm.default(view_154, permute_77)
    add_tensor_24: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_24, primals_119);  mm_default_24 = primals_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_155: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_24, [4, 512, 768]);  add_tensor_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_78: "f32[768, 768]" = torch.ops.aten.permute.default(primals_120, [1, 0]);  primals_120 = None
    
    # No stacktrace found for following nodes
    mm_default_23: "f32[2048, 768]" = torch.ops.aten.mm.default(view_154, permute_78)
    add_tensor_23: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_23, primals_121);  mm_default_23 = primals_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_157: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_23, [4, 512, 768]);  add_tensor_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_158: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_157, [4, 512, 12, 64]);  view_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_79: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_158, [0, 2, 1, 3]);  view_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_80: "f32[768, 768]" = torch.ops.aten.permute.default(primals_122, [1, 0]);  primals_122 = None
    
    # No stacktrace found for following nodes
    mm_default_22: "f32[2048, 768]" = torch.ops.aten.mm.default(view_154, permute_80)
    add_tensor_22: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_22, primals_123);  mm_default_22 = primals_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_160: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_22, [4, 512, 768]);  add_tensor_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_161: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_160, [4, 512, 12, 64]);  view_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_81: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_161, [0, 2, 1, 3]);  view_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_162: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_155, [4, 512, 12, 64]);  view_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_82: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_83: "f32[4, 12, 64, 512]" = torch.ops.aten.permute.default(permute_79, [0, 1, 3, 2]);  permute_79 = None
    expand_29: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_82, [4, 12, 512, 64]);  permute_82 = None
    clone_50: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
    view_163: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_50, [48, 512, 64]);  clone_50 = None
    expand_30: "f32[4, 12, 64, 512]" = torch.ops.aten.expand.default(permute_83, [4, 12, 64, 512]);  permute_83 = None
    clone_51: "f32[4, 12, 64, 512]" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
    view_164: "f32[48, 64, 512]" = torch.ops.aten.reshape.default(clone_51, [48, 64, 512]);  clone_51 = None
    bmm_14: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_163, view_164)
    view_165: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_14, [4, 12, 512, 512]);  bmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:352, code: attention_scores = attention_scores + attention_mask
    div_14: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_165, 8.0);  view_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_7: "f32[4, 12, 512, 1]" = torch.ops.aten.amax.default(div_14, [-1], True)
    sub_23: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(div_14, amax_7);  div_14 = amax_7 = None
    exp_7: "f32[4, 12, 512, 512]" = torch.ops.aten.exp.default(sub_23);  sub_23 = None
    sum_8: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_15: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    alias_7: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(div_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: attention_probs = self.dropout(attention_probs)
    clone_52: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_15);  div_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_31: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_52, [4, 12, 512, 512]);  clone_52 = None
    view_166: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(expand_31, [48, 512, 512]);  expand_31 = None
    expand_32: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_81, [4, 12, 512, 64]);  permute_81 = None
    clone_53: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
    view_167: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_53, [48, 512, 64]);  clone_53 = None
    bmm_15: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_166, view_167)
    view_168: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_15, [4, 12, 512, 64]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_84: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
    clone_54: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_169: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_54, [4, 512, 768]);  clone_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_170: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_169, [2048, 768]);  view_169 = None
    permute_85: "f32[768, 768]" = torch.ops.aten.permute.default(primals_124, [1, 0]);  primals_124 = None
    
    # No stacktrace found for following nodes
    mm_default_21: "f32[2048, 768]" = torch.ops.aten.mm.default(view_170, permute_85)
    add_tensor_21: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_21, primals_125);  mm_default_21 = primals_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_171: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_21, [4, 512, 768]);  add_tensor_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_61: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_171, add_59);  view_171 = add_59 = None
    var_mean_15 = torch.ops.aten.var_mean.correction(add_61, [2], correction = 0, keepdim = True)
    getitem_30: "f32[4, 512, 1]" = var_mean_15[0]
    getitem_31: "f32[4, 512, 1]" = var_mean_15[1];  var_mean_15 = None
    add_62: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
    rsqrt_15: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
    sub_24: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_61, getitem_31);  add_61 = getitem_31 = None
    mul_52: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_15);  sub_24 = None
    mul_53: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_52, primals_126)
    add_63: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_53, primals_127);  mul_53 = primals_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_172: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_63, [2048, 768])
    permute_86: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_128, [1, 0]);  primals_128 = None
    addmm_46: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_129, view_172, permute_86);  primals_129 = None
    view_173: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_46, [4, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_54: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.5)
    mul_55: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476);  view_173 = None
    erf_7: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
    add_64: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_56: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_54, add_64);  mul_54 = add_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_174: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_56, [2048, 3072]);  mul_56 = None
    permute_87: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_130, [1, 0]);  primals_130 = None
    
    # No stacktrace found for following nodes
    mm_default_20: "f32[2048, 768]" = torch.ops.aten.mm.default(view_174, permute_87)
    add_tensor_20: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_20, primals_131);  mm_default_20 = primals_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_175: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_20, [4, 512, 768]);  add_tensor_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_65: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_175, add_63);  view_175 = add_63 = None
    var_mean_16 = torch.ops.aten.var_mean.correction(add_65, [2], correction = 0, keepdim = True)
    getitem_32: "f32[4, 512, 1]" = var_mean_16[0]
    getitem_33: "f32[4, 512, 1]" = var_mean_16[1];  var_mean_16 = None
    add_66: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
    rsqrt_16: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    sub_25: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_65, getitem_33);  add_65 = getitem_33 = None
    mul_57: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_16);  sub_25 = None
    mul_58: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_57, primals_132)
    add_67: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_58, primals_133);  mul_58 = primals_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_176: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_67, [2048, 768])
    permute_88: "f32[768, 768]" = torch.ops.aten.permute.default(primals_134, [1, 0]);  primals_134 = None
    
    # No stacktrace found for following nodes
    mm_default_19: "f32[2048, 768]" = torch.ops.aten.mm.default(view_176, permute_88)
    add_tensor_19: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_19, primals_135);  mm_default_19 = primals_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_177: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_19, [4, 512, 768]);  add_tensor_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_89: "f32[768, 768]" = torch.ops.aten.permute.default(primals_136, [1, 0]);  primals_136 = None
    
    # No stacktrace found for following nodes
    mm_default_18: "f32[2048, 768]" = torch.ops.aten.mm.default(view_176, permute_89)
    add_tensor_18: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_18, primals_137);  mm_default_18 = primals_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_179: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_18, [4, 512, 768]);  add_tensor_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_180: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_179, [4, 512, 12, 64]);  view_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_90: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_180, [0, 2, 1, 3]);  view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_91: "f32[768, 768]" = torch.ops.aten.permute.default(primals_138, [1, 0]);  primals_138 = None
    
    # No stacktrace found for following nodes
    mm_default_17: "f32[2048, 768]" = torch.ops.aten.mm.default(view_176, permute_91)
    add_tensor_17: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_17, primals_139);  mm_default_17 = primals_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_182: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_17, [4, 512, 768]);  add_tensor_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_183: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_182, [4, 512, 12, 64]);  view_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_92: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_184: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_177, [4, 512, 12, 64]);  view_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_93: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_94: "f32[4, 12, 64, 512]" = torch.ops.aten.permute.default(permute_90, [0, 1, 3, 2]);  permute_90 = None
    expand_33: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_93, [4, 12, 512, 64]);  permute_93 = None
    clone_57: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
    view_185: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_57, [48, 512, 64]);  clone_57 = None
    expand_34: "f32[4, 12, 64, 512]" = torch.ops.aten.expand.default(permute_94, [4, 12, 64, 512]);  permute_94 = None
    clone_58: "f32[4, 12, 64, 512]" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
    view_186: "f32[48, 64, 512]" = torch.ops.aten.reshape.default(clone_58, [48, 64, 512]);  clone_58 = None
    bmm_16: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_185, view_186)
    view_187: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_16, [4, 12, 512, 512]);  bmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:352, code: attention_scores = attention_scores + attention_mask
    div_16: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_187, 8.0);  view_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_8: "f32[4, 12, 512, 1]" = torch.ops.aten.amax.default(div_16, [-1], True)
    sub_26: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(div_16, amax_8);  div_16 = amax_8 = None
    exp_8: "f32[4, 12, 512, 512]" = torch.ops.aten.exp.default(sub_26);  sub_26 = None
    sum_9: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_17: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    alias_8: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(div_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: attention_probs = self.dropout(attention_probs)
    clone_59: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_17);  div_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_35: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_59, [4, 12, 512, 512]);  clone_59 = None
    view_188: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(expand_35, [48, 512, 512]);  expand_35 = None
    expand_36: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_92, [4, 12, 512, 64]);  permute_92 = None
    clone_60: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
    view_189: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_60, [48, 512, 64]);  clone_60 = None
    bmm_17: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_188, view_189)
    view_190: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_17, [4, 12, 512, 64]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_95: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_190, [0, 2, 1, 3]);  view_190 = None
    clone_61: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_191: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_61, [4, 512, 768]);  clone_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_192: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_191, [2048, 768]);  view_191 = None
    permute_96: "f32[768, 768]" = torch.ops.aten.permute.default(primals_140, [1, 0]);  primals_140 = None
    
    # No stacktrace found for following nodes
    mm_default_16: "f32[2048, 768]" = torch.ops.aten.mm.default(view_192, permute_96)
    add_tensor_16: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_16, primals_141);  mm_default_16 = primals_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_193: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_16, [4, 512, 768]);  add_tensor_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_69: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_193, add_67);  view_193 = add_67 = None
    var_mean_17 = torch.ops.aten.var_mean.correction(add_69, [2], correction = 0, keepdim = True)
    getitem_34: "f32[4, 512, 1]" = var_mean_17[0]
    getitem_35: "f32[4, 512, 1]" = var_mean_17[1];  var_mean_17 = None
    add_70: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
    rsqrt_17: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
    sub_27: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_69, getitem_35);  add_69 = getitem_35 = None
    mul_59: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_17);  sub_27 = None
    mul_60: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_59, primals_142)
    add_71: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_60, primals_143);  mul_60 = primals_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_194: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_71, [2048, 768])
    permute_97: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_144, [1, 0]);  primals_144 = None
    addmm_52: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_145, view_194, permute_97);  primals_145 = None
    view_195: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_52, [4, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_61: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, 0.5)
    mul_62: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476);  view_195 = None
    erf_8: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_62);  mul_62 = None
    add_72: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_63: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_61, add_72);  mul_61 = add_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_196: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_63, [2048, 3072]);  mul_63 = None
    permute_98: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_146, [1, 0]);  primals_146 = None
    
    # No stacktrace found for following nodes
    mm_default_15: "f32[2048, 768]" = torch.ops.aten.mm.default(view_196, permute_98)
    add_tensor_15: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_15, primals_147);  mm_default_15 = primals_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_197: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_15, [4, 512, 768]);  add_tensor_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_73: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_197, add_71);  view_197 = add_71 = None
    var_mean_18 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
    getitem_36: "f32[4, 512, 1]" = var_mean_18[0]
    getitem_37: "f32[4, 512, 1]" = var_mean_18[1];  var_mean_18 = None
    add_74: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
    rsqrt_18: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    sub_28: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_73, getitem_37);  add_73 = getitem_37 = None
    mul_64: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_18);  sub_28 = None
    mul_65: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_64, primals_148)
    add_75: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_65, primals_149);  mul_65 = primals_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_198: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_75, [2048, 768])
    permute_99: "f32[768, 768]" = torch.ops.aten.permute.default(primals_150, [1, 0]);  primals_150 = None
    
    # No stacktrace found for following nodes
    mm_default_14: "f32[2048, 768]" = torch.ops.aten.mm.default(view_198, permute_99)
    add_tensor_14: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_14, primals_151);  mm_default_14 = primals_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_199: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_14, [4, 512, 768]);  add_tensor_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_100: "f32[768, 768]" = torch.ops.aten.permute.default(primals_152, [1, 0]);  primals_152 = None
    
    # No stacktrace found for following nodes
    mm_default_13: "f32[2048, 768]" = torch.ops.aten.mm.default(view_198, permute_100)
    add_tensor_13: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_13, primals_153);  mm_default_13 = primals_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_201: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_13, [4, 512, 768]);  add_tensor_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_202: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_201, [4, 512, 12, 64]);  view_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_101: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_202, [0, 2, 1, 3]);  view_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_102: "f32[768, 768]" = torch.ops.aten.permute.default(primals_154, [1, 0]);  primals_154 = None
    
    # No stacktrace found for following nodes
    mm_default_12: "f32[2048, 768]" = torch.ops.aten.mm.default(view_198, permute_102)
    add_tensor_12: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_12, primals_155);  mm_default_12 = primals_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_204: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_12, [4, 512, 768]);  add_tensor_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_205: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_204, [4, 512, 12, 64]);  view_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_103: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_205, [0, 2, 1, 3]);  view_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_206: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_199, [4, 512, 12, 64]);  view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_104: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_105: "f32[4, 12, 64, 512]" = torch.ops.aten.permute.default(permute_101, [0, 1, 3, 2]);  permute_101 = None
    expand_37: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_104, [4, 12, 512, 64]);  permute_104 = None
    clone_64: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
    view_207: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_64, [48, 512, 64]);  clone_64 = None
    expand_38: "f32[4, 12, 64, 512]" = torch.ops.aten.expand.default(permute_105, [4, 12, 64, 512]);  permute_105 = None
    clone_65: "f32[4, 12, 64, 512]" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
    view_208: "f32[48, 64, 512]" = torch.ops.aten.reshape.default(clone_65, [48, 64, 512]);  clone_65 = None
    bmm_18: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_207, view_208)
    view_209: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_18, [4, 12, 512, 512]);  bmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:352, code: attention_scores = attention_scores + attention_mask
    div_18: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_209, 8.0);  view_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_9: "f32[4, 12, 512, 1]" = torch.ops.aten.amax.default(div_18, [-1], True)
    sub_29: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(div_18, amax_9);  div_18 = amax_9 = None
    exp_9: "f32[4, 12, 512, 512]" = torch.ops.aten.exp.default(sub_29);  sub_29 = None
    sum_10: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_19: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    alias_9: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(div_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: attention_probs = self.dropout(attention_probs)
    clone_66: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_19);  div_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_39: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_66, [4, 12, 512, 512]);  clone_66 = None
    view_210: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(expand_39, [48, 512, 512]);  expand_39 = None
    expand_40: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_103, [4, 12, 512, 64]);  permute_103 = None
    clone_67: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
    view_211: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_67, [48, 512, 64]);  clone_67 = None
    bmm_19: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_210, view_211)
    view_212: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_19, [4, 12, 512, 64]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_106: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None
    clone_68: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_213: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_68, [4, 512, 768]);  clone_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_214: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_213, [2048, 768]);  view_213 = None
    permute_107: "f32[768, 768]" = torch.ops.aten.permute.default(primals_156, [1, 0]);  primals_156 = None
    
    # No stacktrace found for following nodes
    mm_default_11: "f32[2048, 768]" = torch.ops.aten.mm.default(view_214, permute_107)
    add_tensor_11: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_11, primals_157);  mm_default_11 = primals_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_215: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_11, [4, 512, 768]);  add_tensor_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_77: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_215, add_75);  view_215 = add_75 = None
    var_mean_19 = torch.ops.aten.var_mean.correction(add_77, [2], correction = 0, keepdim = True)
    getitem_38: "f32[4, 512, 1]" = var_mean_19[0]
    getitem_39: "f32[4, 512, 1]" = var_mean_19[1];  var_mean_19 = None
    add_78: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
    rsqrt_19: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    sub_30: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_77, getitem_39);  add_77 = getitem_39 = None
    mul_66: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_19);  sub_30 = None
    mul_67: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_66, primals_158)
    add_79: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_67, primals_159);  mul_67 = primals_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_216: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_79, [2048, 768])
    permute_108: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_160, [1, 0]);  primals_160 = None
    addmm_58: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_161, view_216, permute_108);  primals_161 = None
    view_217: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_58, [4, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_68: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, 0.5)
    mul_69: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476);  view_217 = None
    erf_9: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
    add_80: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_70: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_68, add_80);  mul_68 = add_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_218: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_70, [2048, 3072]);  mul_70 = None
    permute_109: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_162, [1, 0]);  primals_162 = None
    
    # No stacktrace found for following nodes
    mm_default_10: "f32[2048, 768]" = torch.ops.aten.mm.default(view_218, permute_109)
    add_tensor_10: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_10, primals_163);  mm_default_10 = primals_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_219: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_10, [4, 512, 768]);  add_tensor_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_81: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_219, add_79);  view_219 = add_79 = None
    var_mean_20 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
    getitem_40: "f32[4, 512, 1]" = var_mean_20[0]
    getitem_41: "f32[4, 512, 1]" = var_mean_20[1];  var_mean_20 = None
    add_82: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
    rsqrt_20: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
    sub_31: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_81, getitem_41);  add_81 = getitem_41 = None
    mul_71: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_20);  sub_31 = None
    mul_72: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_71, primals_164)
    add_83: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_72, primals_165);  mul_72 = primals_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_220: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_83, [2048, 768])
    permute_110: "f32[768, 768]" = torch.ops.aten.permute.default(primals_166, [1, 0]);  primals_166 = None
    
    # No stacktrace found for following nodes
    mm_default_9: "f32[2048, 768]" = torch.ops.aten.mm.default(view_220, permute_110)
    add_tensor_9: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_9, primals_167);  mm_default_9 = primals_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_221: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_9, [4, 512, 768]);  add_tensor_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_111: "f32[768, 768]" = torch.ops.aten.permute.default(primals_168, [1, 0]);  primals_168 = None
    
    # No stacktrace found for following nodes
    mm_default_8: "f32[2048, 768]" = torch.ops.aten.mm.default(view_220, permute_111)
    add_tensor_8: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_8, primals_169);  mm_default_8 = primals_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_223: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_8, [4, 512, 768]);  add_tensor_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_224: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_223, [4, 512, 12, 64]);  view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_112: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_113: "f32[768, 768]" = torch.ops.aten.permute.default(primals_170, [1, 0]);  primals_170 = None
    
    # No stacktrace found for following nodes
    mm_default_7: "f32[2048, 768]" = torch.ops.aten.mm.default(view_220, permute_113)
    add_tensor_7: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_7, primals_171);  mm_default_7 = primals_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_226: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_7, [4, 512, 768]);  add_tensor_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_227: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_226, [4, 512, 12, 64]);  view_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_114: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_227, [0, 2, 1, 3]);  view_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_228: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_221, [4, 512, 12, 64]);  view_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_115: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_116: "f32[4, 12, 64, 512]" = torch.ops.aten.permute.default(permute_112, [0, 1, 3, 2]);  permute_112 = None
    expand_41: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_115, [4, 12, 512, 64]);  permute_115 = None
    clone_71: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
    view_229: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_71, [48, 512, 64]);  clone_71 = None
    expand_42: "f32[4, 12, 64, 512]" = torch.ops.aten.expand.default(permute_116, [4, 12, 64, 512]);  permute_116 = None
    clone_72: "f32[4, 12, 64, 512]" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
    view_230: "f32[48, 64, 512]" = torch.ops.aten.reshape.default(clone_72, [48, 64, 512]);  clone_72 = None
    bmm_20: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_229, view_230)
    view_231: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_20, [4, 12, 512, 512]);  bmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:352, code: attention_scores = attention_scores + attention_mask
    div_20: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_231, 8.0);  view_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_10: "f32[4, 12, 512, 1]" = torch.ops.aten.amax.default(div_20, [-1], True)
    sub_32: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(div_20, amax_10);  div_20 = amax_10 = None
    exp_10: "f32[4, 12, 512, 512]" = torch.ops.aten.exp.default(sub_32);  sub_32 = None
    sum_11: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_21: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    alias_10: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(div_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: attention_probs = self.dropout(attention_probs)
    clone_73: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_21);  div_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_43: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_73, [4, 12, 512, 512]);  clone_73 = None
    view_232: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(expand_43, [48, 512, 512]);  expand_43 = None
    expand_44: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_114, [4, 12, 512, 64]);  permute_114 = None
    clone_74: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
    view_233: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_74, [48, 512, 64]);  clone_74 = None
    bmm_21: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_232, view_233)
    view_234: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_21, [4, 12, 512, 64]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_117: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_234, [0, 2, 1, 3]);  view_234 = None
    clone_75: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_235: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_75, [4, 512, 768]);  clone_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_236: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_235, [2048, 768]);  view_235 = None
    permute_118: "f32[768, 768]" = torch.ops.aten.permute.default(primals_172, [1, 0]);  primals_172 = None
    
    # No stacktrace found for following nodes
    mm_default_6: "f32[2048, 768]" = torch.ops.aten.mm.default(view_236, permute_118)
    add_tensor_6: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_6, primals_173);  mm_default_6 = primals_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_237: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_6, [4, 512, 768]);  add_tensor_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_85: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_237, add_83);  view_237 = add_83 = None
    var_mean_21 = torch.ops.aten.var_mean.correction(add_85, [2], correction = 0, keepdim = True)
    getitem_42: "f32[4, 512, 1]" = var_mean_21[0]
    getitem_43: "f32[4, 512, 1]" = var_mean_21[1];  var_mean_21 = None
    add_86: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
    rsqrt_21: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
    sub_33: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_85, getitem_43);  add_85 = getitem_43 = None
    mul_73: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_21);  sub_33 = None
    mul_74: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_73, primals_174)
    add_87: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_74, primals_175);  mul_74 = primals_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_238: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_87, [2048, 768])
    permute_119: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_176, [1, 0]);  primals_176 = None
    addmm_64: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_177, view_238, permute_119);  primals_177 = None
    view_239: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_64, [4, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_75: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, 0.5)
    mul_76: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, 0.7071067811865476);  view_239 = None
    erf_10: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_76);  mul_76 = None
    add_88: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_77: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_75, add_88);  mul_75 = add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_240: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_77, [2048, 3072]);  mul_77 = None
    permute_120: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_178, [1, 0]);  primals_178 = None
    
    # No stacktrace found for following nodes
    mm_default_5: "f32[2048, 768]" = torch.ops.aten.mm.default(view_240, permute_120)
    add_tensor_5: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_5, primals_179);  mm_default_5 = primals_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_241: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_5, [4, 512, 768]);  add_tensor_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_89: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_241, add_87);  view_241 = add_87 = None
    var_mean_22 = torch.ops.aten.var_mean.correction(add_89, [2], correction = 0, keepdim = True)
    getitem_44: "f32[4, 512, 1]" = var_mean_22[0]
    getitem_45: "f32[4, 512, 1]" = var_mean_22[1];  var_mean_22 = None
    add_90: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
    rsqrt_22: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
    sub_34: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_89, getitem_45);  add_89 = getitem_45 = None
    mul_78: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_22);  sub_34 = None
    mul_79: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_78, primals_180)
    add_91: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_79, primals_181);  mul_79 = primals_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_242: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_91, [2048, 768])
    permute_121: "f32[768, 768]" = torch.ops.aten.permute.default(primals_182, [1, 0]);  primals_182 = None
    
    # No stacktrace found for following nodes
    mm_default_4: "f32[2048, 768]" = torch.ops.aten.mm.default(view_242, permute_121)
    add_tensor_4: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_4, primals_183);  mm_default_4 = primals_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_243: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_4, [4, 512, 768]);  add_tensor_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_122: "f32[768, 768]" = torch.ops.aten.permute.default(primals_184, [1, 0]);  primals_184 = None
    
    # No stacktrace found for following nodes
    mm_default_3: "f32[2048, 768]" = torch.ops.aten.mm.default(view_242, permute_122)
    add_tensor_3: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_3, primals_185);  mm_default_3 = primals_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_245: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_3, [4, 512, 768]);  add_tensor_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_246: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_245, [4, 512, 12, 64]);  view_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_123: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_124: "f32[768, 768]" = torch.ops.aten.permute.default(primals_186, [1, 0]);  primals_186 = None
    
    # No stacktrace found for following nodes
    mm_default_2: "f32[2048, 768]" = torch.ops.aten.mm.default(view_242, permute_124)
    add_tensor_2: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_2, primals_187);  mm_default_2 = primals_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_248: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_2, [4, 512, 768]);  add_tensor_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_249: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_248, [4, 512, 12, 64]);  view_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_125: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_249, [0, 2, 1, 3]);  view_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_250: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_243, [4, 512, 12, 64]);  view_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_126: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_127: "f32[4, 12, 64, 512]" = torch.ops.aten.permute.default(permute_123, [0, 1, 3, 2]);  permute_123 = None
    expand_45: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_126, [4, 12, 512, 64]);  permute_126 = None
    clone_78: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
    view_251: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_78, [48, 512, 64]);  clone_78 = None
    expand_46: "f32[4, 12, 64, 512]" = torch.ops.aten.expand.default(permute_127, [4, 12, 64, 512]);  permute_127 = None
    clone_79: "f32[4, 12, 64, 512]" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
    view_252: "f32[48, 64, 512]" = torch.ops.aten.reshape.default(clone_79, [48, 64, 512]);  clone_79 = None
    bmm_22: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_251, view_252)
    view_253: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_22, [4, 12, 512, 512]);  bmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:352, code: attention_scores = attention_scores + attention_mask
    div_22: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(view_253, 8.0);  view_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    amax_11: "f32[4, 12, 512, 1]" = torch.ops.aten.amax.default(div_22, [-1], True)
    sub_35: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(div_22, amax_11);  div_22 = amax_11 = None
    exp_11: "f32[4, 12, 512, 512]" = torch.ops.aten.exp.default(sub_35);  sub_35 = None
    sum_12: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_23: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    alias_11: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(div_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:359, code: attention_probs = self.dropout(attention_probs)
    clone_80: "f32[4, 12, 512, 512]" = torch.ops.aten.clone.default(div_23);  div_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_47: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(clone_80, [4, 12, 512, 512]);  clone_80 = None
    view_254: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(expand_47, [48, 512, 512]);  expand_47 = None
    expand_48: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_125, [4, 12, 512, 64]);  permute_125 = None
    clone_81: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
    view_255: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_81, [48, 512, 64]);  clone_81 = None
    bmm_23: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_254, view_255)
    view_256: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_23, [4, 12, 512, 64]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_128: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_256, [0, 2, 1, 3]);  view_256 = None
    clone_82: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_257: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_82, [4, 512, 768]);  clone_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_258: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_257, [2048, 768]);  view_257 = None
    permute_129: "f32[768, 768]" = torch.ops.aten.permute.default(primals_188, [1, 0]);  primals_188 = None
    
    # No stacktrace found for following nodes
    mm_default_1: "f32[2048, 768]" = torch.ops.aten.mm.default(view_258, permute_129)
    add_tensor_1: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default_1, primals_189);  mm_default_1 = primals_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_259: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_1, [4, 512, 768]);  add_tensor_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_93: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_259, add_91);  view_259 = add_91 = None
    var_mean_23 = torch.ops.aten.var_mean.correction(add_93, [2], correction = 0, keepdim = True)
    getitem_46: "f32[4, 512, 1]" = var_mean_23[0]
    getitem_47: "f32[4, 512, 1]" = var_mean_23[1];  var_mean_23 = None
    add_94: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
    rsqrt_23: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
    sub_36: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_93, getitem_47);  add_93 = getitem_47 = None
    mul_80: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_23);  sub_36 = None
    mul_81: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_80, primals_190)
    add_95: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_81, primals_191);  mul_81 = primals_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_260: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_95, [2048, 768])
    permute_130: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_192, [1, 0]);  primals_192 = None
    addmm_70: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_193, view_260, permute_130);  primals_193 = None
    view_261: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_70, [4, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_82: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, 0.5)
    mul_83: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476);  view_261 = None
    erf_11: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
    add_96: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_84: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_82, add_96);  mul_82 = add_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_262: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_84, [2048, 3072]);  mul_84 = None
    permute_131: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_194, [1, 0]);  primals_194 = None
    
    # No stacktrace found for following nodes
    mm_default: "f32[2048, 768]" = torch.ops.aten.mm.default(view_262, permute_131)
    add_tensor: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mm_default, primals_195);  mm_default = primals_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_263: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(add_tensor, [4, 512, 768]);  add_tensor = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_97: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(view_263, add_95);  view_263 = add_95 = None
    var_mean_24 = torch.ops.aten.var_mean.correction(add_97, [2], correction = 0, keepdim = True)
    getitem_48: "f32[4, 512, 1]" = var_mean_24[0]
    getitem_49: "f32[4, 512, 1]" = var_mean_24[1];  var_mean_24 = None
    add_98: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
    rsqrt_24: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
    sub_37: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_97, getitem_49);  add_97 = getitem_49 = None
    mul_85: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_24);  sub_37 = None
    mul_86: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_85, primals_196)
    add_99: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_86, primals_197);  mul_86 = primals_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:680, code: hidden_states = self.dense(hidden_states)
    view_264: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_99, [2048, 768]);  add_99 = None
    permute_132: "f32[768, 768]" = torch.ops.aten.permute.default(primals_198, [1, 0]);  primals_198 = None
    addmm_72: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_199, view_264, permute_132);  primals_199 = None
    view_265: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_72, [4, 512, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_87: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_265, 0.5)
    mul_88: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_265, 0.7071067811865476);  view_265 = None
    erf_12: "f32[4, 512, 768]" = torch.ops.aten.erf.default(mul_88);  mul_88 = None
    add_100: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_89: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_87, add_100);  mul_87 = add_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:682, code: hidden_states = self.LayerNorm(hidden_states)
    var_mean_25 = torch.ops.aten.var_mean.correction(mul_89, [2], correction = 0, keepdim = True)
    getitem_50: "f32[4, 512, 1]" = var_mean_25[0]
    getitem_51: "f32[4, 512, 1]" = var_mean_25[1];  var_mean_25 = None
    add_101: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
    rsqrt_25: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
    sub_38: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_89, getitem_51);  mul_89 = getitem_51 = None
    mul_90: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_25);  sub_38 = None
    mul_91: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_90, primals_200)
    add_102: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_91, primals_201);  mul_91 = primals_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:702, code: hidden_states = self.decoder(hidden_states)
    view_266: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_102, [2048, 768]);  add_102 = None
    permute_133: "f32[768, 30522]" = torch.ops.aten.permute.default(primals_202, [1, 0]);  primals_202 = None
    addmm_73: "f32[2048, 30522]" = torch.ops.aten.addmm.default(primals_203, view_266, permute_133);  primals_203 = None
    view_267: "f32[4, 512, 30522]" = torch.ops.aten.reshape.default(addmm_73, [4, 512, 30522]);  addmm_73 = None
    permute_134: "f32[30522, 768]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:682, code: hidden_states = self.LayerNorm(hidden_states)
    div_24: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 768);  rsqrt_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:680, code: hidden_states = self.dense(hidden_states)
    permute_138: "f32[768, 768]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_25: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    permute_142: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    permute_146: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_26: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    permute_150: "f32[768, 768]" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    permute_155: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None
    permute_156: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_255, [0, 2, 1]);  view_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_12: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_157: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_251, [0, 2, 1]);  view_251 = None
    permute_158: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_252, [0, 2, 1]);  view_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_162: "f32[768, 768]" = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_167: "f32[768, 768]" = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    permute_171: "f32[768, 768]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_28: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    permute_175: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    permute_179: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_29: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    permute_183: "f32[768, 768]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    permute_188: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_232, [0, 2, 1]);  view_232 = None
    permute_189: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_233, [0, 2, 1]);  view_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_13: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_190: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_229, [0, 2, 1]);  view_229 = None
    permute_191: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_230, [0, 2, 1]);  view_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_195: "f32[768, 768]" = torch.ops.aten.permute.default(permute_113, [1, 0]);  permute_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_200: "f32[768, 768]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    permute_204: "f32[768, 768]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_31: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    permute_208: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    permute_212: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_32: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    permute_216: "f32[768, 768]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    permute_221: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None
    permute_222: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_211, [0, 2, 1]);  view_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_14: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_223: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_207, [0, 2, 1]);  view_207 = None
    permute_224: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_208, [0, 2, 1]);  view_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_228: "f32[768, 768]" = torch.ops.aten.permute.default(permute_102, [1, 0]);  permute_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_233: "f32[768, 768]" = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    permute_237: "f32[768, 768]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_34: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    permute_241: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    permute_245: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_35: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    permute_249: "f32[768, 768]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    permute_254: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None
    permute_255: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_189, [0, 2, 1]);  view_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_15: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_256: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_185, [0, 2, 1]);  view_185 = None
    permute_257: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1]);  view_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_261: "f32[768, 768]" = torch.ops.aten.permute.default(permute_91, [1, 0]);  permute_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_266: "f32[768, 768]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    permute_270: "f32[768, 768]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_37: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    permute_274: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    permute_278: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_38: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    permute_282: "f32[768, 768]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    permute_287: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_166, [0, 2, 1]);  view_166 = None
    permute_288: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_167, [0, 2, 1]);  view_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_16: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_289: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_163, [0, 2, 1]);  view_163 = None
    permute_290: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_164, [0, 2, 1]);  view_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_294: "f32[768, 768]" = torch.ops.aten.permute.default(permute_80, [1, 0]);  permute_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_299: "f32[768, 768]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    permute_303: "f32[768, 768]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_40: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    permute_307: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    permute_311: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_41: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    permute_315: "f32[768, 768]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    permute_320: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_144, [0, 2, 1]);  view_144 = None
    permute_321: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_145, [0, 2, 1]);  view_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_17: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_322: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_141, [0, 2, 1]);  view_141 = None
    permute_323: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_142, [0, 2, 1]);  view_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_327: "f32[768, 768]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_332: "f32[768, 768]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    permute_336: "f32[768, 768]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_43: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    permute_340: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    permute_344: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_44: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    permute_348: "f32[768, 768]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    permute_353: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None
    permute_354: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_18: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_355: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_119, [0, 2, 1]);  view_119 = None
    permute_356: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1]);  view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_360: "f32[768, 768]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_365: "f32[768, 768]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    permute_369: "f32[768, 768]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_46: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    permute_373: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    permute_377: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_47: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    permute_381: "f32[768, 768]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    permute_386: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_100, [0, 2, 1]);  view_100 = None
    permute_387: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_101, [0, 2, 1]);  view_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_19: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_388: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_97, [0, 2, 1]);  view_97 = None
    permute_389: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_98, [0, 2, 1]);  view_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_393: "f32[768, 768]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_398: "f32[768, 768]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    permute_402: "f32[768, 768]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_49: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    permute_406: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    permute_410: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_50: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    permute_414: "f32[768, 768]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    permute_419: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
    permute_420: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_79, [0, 2, 1]);  view_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_20: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_421: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_75, [0, 2, 1]);  view_75 = None
    permute_422: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1]);  view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_426: "f32[768, 768]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_431: "f32[768, 768]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    permute_435: "f32[768, 768]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_52: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    permute_439: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    permute_443: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_53: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    permute_447: "f32[768, 768]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    permute_452: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None
    permute_453: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_57, [0, 2, 1]);  view_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_21: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_454: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_53, [0, 2, 1]);  view_53 = None
    permute_455: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1]);  view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_459: "f32[768, 768]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_464: "f32[768, 768]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    permute_468: "f32[768, 768]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_55: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    permute_472: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    permute_476: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_56: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    permute_480: "f32[768, 768]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    permute_485: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
    permute_486: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_35, [0, 2, 1]);  view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_22: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_487: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_31, [0, 2, 1]);  view_31 = None
    permute_488: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_32, [0, 2, 1]);  view_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_492: "f32[768, 768]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_497: "f32[768, 768]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    permute_501: "f32[768, 768]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_58: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    permute_505: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    permute_509: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    div_59: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    permute_513: "f32[768, 768]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    permute_518: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
    permute_519: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    alias_23: "f32[4, 12, 512, 512]" = torch.ops.aten.alias.default(alias);  alias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_520: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
    permute_521: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    permute_525: "f32[768, 768]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    permute_530: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    permute_534: "f32[768, 768]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:239, code: embeddings = self.LayerNorm(embeddings)
    div_61: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
    return [view_267, primals_4, primals_14, primals_20, primals_30, primals_36, primals_46, primals_52, primals_62, primals_68, primals_78, primals_84, primals_94, primals_100, primals_110, primals_116, primals_126, primals_132, primals_142, primals_148, primals_158, primals_164, primals_174, primals_180, primals_190, primals_196, primals_200, primals_206, expand, slice_4, mul_1, view, view_16, mul_3, view_18, addmm_4, view_20, mul_8, view_22, view_38, mul_10, view_40, addmm_10, view_42, mul_15, view_44, view_60, mul_17, view_62, addmm_16, view_64, mul_22, view_66, view_82, mul_24, view_84, addmm_22, view_86, mul_29, view_88, view_104, mul_31, view_106, addmm_28, view_108, mul_36, view_110, view_126, mul_38, view_128, addmm_34, view_130, mul_43, view_132, view_148, mul_45, view_150, addmm_40, view_152, mul_50, view_154, view_170, mul_52, view_172, addmm_46, view_174, mul_57, view_176, view_192, mul_59, view_194, addmm_52, view_196, mul_64, view_198, view_214, mul_66, view_216, addmm_58, view_218, mul_71, view_220, view_236, mul_73, view_238, addmm_64, view_240, mul_78, view_242, view_258, mul_80, view_260, addmm_70, view_262, mul_85, view_264, addmm_72, mul_90, view_266, permute_134, div_24, permute_138, div_25, permute_142, permute_146, div_26, permute_150, permute_155, permute_156, alias_12, permute_157, permute_158, permute_162, permute_167, permute_171, div_28, permute_175, permute_179, div_29, permute_183, permute_188, permute_189, alias_13, permute_190, permute_191, permute_195, permute_200, permute_204, div_31, permute_208, permute_212, div_32, permute_216, permute_221, permute_222, alias_14, permute_223, permute_224, permute_228, permute_233, permute_237, div_34, permute_241, permute_245, div_35, permute_249, permute_254, permute_255, alias_15, permute_256, permute_257, permute_261, permute_266, permute_270, div_37, permute_274, permute_278, div_38, permute_282, permute_287, permute_288, alias_16, permute_289, permute_290, permute_294, permute_299, permute_303, div_40, permute_307, permute_311, div_41, permute_315, permute_320, permute_321, alias_17, permute_322, permute_323, permute_327, permute_332, permute_336, div_43, permute_340, permute_344, div_44, permute_348, permute_353, permute_354, alias_18, permute_355, permute_356, permute_360, permute_365, permute_369, div_46, permute_373, permute_377, div_47, permute_381, permute_386, permute_387, alias_19, permute_388, permute_389, permute_393, permute_398, permute_402, div_49, permute_406, permute_410, div_50, permute_414, permute_419, permute_420, alias_20, permute_421, permute_422, permute_426, permute_431, permute_435, div_52, permute_439, permute_443, div_53, permute_447, permute_452, permute_453, alias_21, permute_454, permute_455, permute_459, permute_464, permute_468, div_55, permute_472, permute_476, div_56, permute_480, permute_485, permute_486, alias_22, permute_487, permute_488, permute_492, permute_497, permute_501, div_58, permute_505, permute_509, div_59, permute_513, permute_518, permute_519, alias_23, permute_520, permute_521, permute_525, permute_530, permute_534, div_61]
    