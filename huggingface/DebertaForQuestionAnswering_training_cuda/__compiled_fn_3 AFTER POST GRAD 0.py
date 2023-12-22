from __future__ import annotations



def forward(self, primals_1: "f32[768]", primals_2: "f32[768]", primals_3: "f32[768]", primals_4: "f32[768]", primals_5: "f32[768]", primals_6: "f32[768]", primals_7: "f32[768]", primals_8: "f32[768]", primals_9: "f32[768]", primals_10: "f32[768]", primals_11: "f32[768]", primals_12: "f32[768]", primals_13: "f32[768]", primals_14: "f32[768]", primals_15: "f32[768]", primals_16: "f32[768]", primals_17: "f32[768]", primals_18: "f32[768]", primals_19: "f32[768]", primals_20: "f32[768]", primals_21: "f32[768]", primals_22: "f32[768]", primals_23: "f32[768]", primals_24: "f32[768]", primals_25: "f32[768]", primals_26: "f32[768]", primals_27: "f32[768]", primals_28: "f32[768]", primals_29: "f32[768]", primals_30: "f32[768]", primals_31: "f32[768]", primals_32: "f32[768]", primals_33: "f32[768]", primals_34: "f32[768]", primals_35: "f32[768]", primals_36: "f32[768]", primals_37: "f32[768]", primals_38: "f32[768]", primals_39: "f32[768]", primals_40: "f32[768]", primals_41: "f32[768]", primals_42: "f32[768]", primals_43: "f32[768]", primals_44: "f32[768]", primals_45: "f32[768]", primals_46: "f32[768]", primals_47: "f32[768]", primals_48: "f32[768]", primals_49: "f32[768]", primals_50: "f32[768]", primals_51: "f32[768]", primals_52: "f32[768]", primals_53: "f32[768]", primals_54: "f32[768]", primals_55: "f32[768]", primals_56: "f32[768]", primals_57: "f32[768]", primals_58: "f32[768]", primals_59: "f32[768]", primals_60: "f32[768]", primals_61: "f32[768]", primals_62: "f32[768]", primals_63: "f32[768]", primals_64: "f32[768]", primals_65: "f32[768]", primals_66: "f32[768]", primals_67: "f32[768]", primals_68: "f32[768]", primals_69: "f32[768]", primals_70: "f32[768]", primals_71: "f32[768]", primals_72: "f32[768]", primals_73: "f32[768]", primals_74: "f32[768]", primals_75: "f32[50265, 768]", primals_76: "f32[512, 768]", primals_77: "f32[2304, 768]", primals_78: "f32[768, 768]", primals_79: "f32[768]", primals_80: "f32[3072, 768]", primals_81: "f32[3072]", primals_82: "f32[768, 3072]", primals_83: "f32[768]", primals_84: "f32[2304, 768]", primals_85: "f32[768, 768]", primals_86: "f32[768]", primals_87: "f32[3072, 768]", primals_88: "f32[3072]", primals_89: "f32[768, 3072]", primals_90: "f32[768]", primals_91: "f32[2304, 768]", primals_92: "f32[768, 768]", primals_93: "f32[768]", primals_94: "f32[3072, 768]", primals_95: "f32[3072]", primals_96: "f32[768, 3072]", primals_97: "f32[768]", primals_98: "f32[2304, 768]", primals_99: "f32[768, 768]", primals_100: "f32[768]", primals_101: "f32[3072, 768]", primals_102: "f32[3072]", primals_103: "f32[768, 3072]", primals_104: "f32[768]", primals_105: "f32[2304, 768]", primals_106: "f32[768, 768]", primals_107: "f32[768]", primals_108: "f32[3072, 768]", primals_109: "f32[3072]", primals_110: "f32[768, 3072]", primals_111: "f32[768]", primals_112: "f32[2304, 768]", primals_113: "f32[768, 768]", primals_114: "f32[768]", primals_115: "f32[3072, 768]", primals_116: "f32[3072]", primals_117: "f32[768, 3072]", primals_118: "f32[768]", primals_119: "f32[2304, 768]", primals_120: "f32[768, 768]", primals_121: "f32[768]", primals_122: "f32[3072, 768]", primals_123: "f32[3072]", primals_124: "f32[768, 3072]", primals_125: "f32[768]", primals_126: "f32[2304, 768]", primals_127: "f32[768, 768]", primals_128: "f32[768]", primals_129: "f32[3072, 768]", primals_130: "f32[3072]", primals_131: "f32[768, 3072]", primals_132: "f32[768]", primals_133: "f32[2304, 768]", primals_134: "f32[768, 768]", primals_135: "f32[768]", primals_136: "f32[3072, 768]", primals_137: "f32[3072]", primals_138: "f32[768, 3072]", primals_139: "f32[768]", primals_140: "f32[2304, 768]", primals_141: "f32[768, 768]", primals_142: "f32[768]", primals_143: "f32[3072, 768]", primals_144: "f32[3072]", primals_145: "f32[768, 3072]", primals_146: "f32[768]", primals_147: "f32[2304, 768]", primals_148: "f32[768, 768]", primals_149: "f32[768]", primals_150: "f32[3072, 768]", primals_151: "f32[3072]", primals_152: "f32[768, 3072]", primals_153: "f32[768]", primals_154: "f32[2304, 768]", primals_155: "f32[768, 768]", primals_156: "f32[768]", primals_157: "f32[3072, 768]", primals_158: "f32[3072]", primals_159: "f32[768, 3072]", primals_160: "f32[768]", primals_161: "f32[2, 768]", primals_162: "f32[2]", primals_163: "i64[1, 512]", primals_164: "i64[1, 512]", primals_165: "i64[1]", primals_166: "i64[1]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:780, code: position_ids = self.position_ids[:, :seq_length]
    slice_1: "i64[1, 512]" = torch.ops.aten.slice.Tensor(primals_163, 0, 0, 9223372036854775807);  primals_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:786, code: inputs_embeds = self.word_embeddings(input_ids)
    embedding: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(primals_75, primals_164, 0);  primals_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:789, code: position_embeddings = self.position_embeddings(position_ids.long())
    embedding_1: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(primals_76, slice_1);  primals_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:795, code: embeddings += position_embeddings
    add: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add, mean);  add = mean = None
    pow_1: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub, 2)
    mean_1: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_1: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_1, 1e-07);  mean_1 = None
    sqrt: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_1);  add_1 = None
    div: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub, sqrt)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_1, div);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:812, code: embeddings = embeddings * mask
    add_2: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul, primals_2);  mul = primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    permute: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty, [0, 1, 2]);  empty = None
    bernoulli: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_2: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli);  bernoulli = None
    convert_element_type: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_2, torch.bool);  sub_2 = None
    full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type, full_default_1, add_2);  add_2 = None
    mul_2: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where, 1.1111111111111112);  where = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_1: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_77, [1, 0]);  primals_77 = None
    view: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_2, [512, 768])
    mm: "f32[512, 2304]" = torch.ops.aten.mm.default(view, permute_1)
    view_1: "f32[1, 512, 2304]" = torch.ops.aten.reshape.default(mm, [1, 512, 2304]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_2: "f32[1, 512, 12, 192]" = torch.ops.aten.reshape.default(view_1, [1, 512, 12, -1]);  view_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_2: "f32[1, 12, 512, 192]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    split = torch.ops.aten.split.Tensor(permute_2, 64, -1);  permute_2 = None
    getitem: "f32[1, 12, 512, 64]" = split[0]
    getitem_1: "f32[1, 12, 512, 64]" = split[1]
    getitem_2: "f32[1, 12, 512, 64]" = split[2];  split = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    unsqueeze_4: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_3, 0);  primals_3 = None
    unsqueeze_5: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 1);  unsqueeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_3: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_5, [1, 1, 12, -1]);  unsqueeze_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_3: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    add_3: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem, permute_3);  getitem = permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    unsqueeze_6: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_4, 0);  primals_4 = None
    unsqueeze_7: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 1);  unsqueeze_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_4: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_7, [1, 1, 12, -1]);  unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_4: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    add_4: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_2, permute_4);  getitem_2 = permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    full_default_2: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_1: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(add_3, full_default_2);  add_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_5: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(getitem_1, [0, 1, 3, 2]);  getitem_1 = None
    expand: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(div_1, [1, 12, 512, 64]);  div_1 = None
    view_5: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand, [12, 512, 64]);  expand = None
    expand_1: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_5, [1, 12, 64, 512]);  permute_5 = None
    view_6: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_1, [12, 64, 512]);  expand_1 = None
    bmm: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_5, view_6)
    view_7: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm, [1, 12, 512, 512]);  bmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    full_default_3: "b8[1, 1, 512, 512]" = torch.ops.aten.full.default([1, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    full_default_4: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where_1: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_4, view_7);  view_7 = None
    amax: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where_1, [-1], True)
    sub_3: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where_1, amax);  where_1 = amax = None
    exp: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
    sum_1: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div_2: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    where_2: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_1, div_2);  div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_1: "f32[1, 12, 512, 512]" = torch.ops.aten.empty.memory_format([1, 12, 512, 512], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    permute_6: "f32[1, 12, 512, 512]" = torch.ops.aten.permute.default(empty_1, [0, 1, 2, 3]);  empty_1 = None
    bernoulli_1: "f32[1, 12, 512, 512]" = torch.ops.aten.bernoulli.p(permute_6, 0.9)
    sub_4: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_1);  bernoulli_1 = None
    convert_element_type_2: "b8[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_4, torch.bool);  sub_4 = None
    where_3: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_2, full_default_1, where_2)
    mul_5: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_3, 1.1111111111111112);  where_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_2: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(mul_5, [1, 12, 512, 512]);  mul_5 = None
    view_8: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_2, [12, 512, 512]);  expand_2 = None
    expand_3: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(add_4, [1, 12, 512, 64]);  add_4 = None
    view_9: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_3, [12, 512, 64]);  expand_3 = None
    bmm_1: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_8, view_9)
    view_10: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_1, [1, 12, 512, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_7: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
    clone: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_11: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone, [1, 512, -1]);  clone = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_12: "f32[512, 768]" = torch.ops.aten.reshape.default(view_11, [512, 768]);  view_11 = None
    permute_8: "f32[768, 768]" = torch.ops.aten.permute.default(primals_78, [1, 0]);  primals_78 = None
    
    # No stacktrace found for following nodes
    mm_default_24: "f32[512, 768]" = torch.ops.aten.mm.default(view_12, permute_8)
    add_tensor_24: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_24, primals_79);  mm_default_24 = primals_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_13: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_24, [1, 512, 768]);  add_tensor_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    bernoulli_2: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_5: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_2);  bernoulli_2 = None
    convert_element_type_3: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_5, torch.bool);  sub_5 = None
    where_4: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_3, full_default_1, view_13);  view_13 = None
    mul_6: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_4, 1.1111111111111112);  where_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_5: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_6, mul_2);  mul_6 = mul_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_2: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_5, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_6: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_5, mean_2);  add_5 = mean_2 = None
    pow_2: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_6, 2)
    mean_3: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_6: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_3, 1e-07);  mean_3 = None
    sqrt_2: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_6);  add_6 = None
    div_3: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_6, sqrt_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_7: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_5, div_3);  div_3 = None
    add_7: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_7, primals_6);  mul_7 = primals_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_14: "f32[512, 768]" = torch.ops.aten.reshape.default(add_7, [512, 768])
    permute_10: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_80, [1, 0]);  primals_80 = None
    addmm_1: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_81, view_14, permute_10);  primals_81 = None
    view_15: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_1, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_8: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_15, 0.5)
    mul_9: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_15, 0.7071067811865476);  view_15 = None
    erf: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_9);  mul_9 = None
    add_8: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_10: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_8, add_8);  mul_8 = add_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_16: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_10, [512, 3072]);  mul_10 = None
    permute_11: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_82, [1, 0]);  primals_82 = None
    
    # No stacktrace found for following nodes
    mm_default_23: "f32[512, 768]" = torch.ops.aten.mm.default(view_16, permute_11)
    add_tensor_23: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_23, primals_83);  mm_default_23 = primals_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_17: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_23, [1, 512, 768]);  add_tensor_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    bernoulli_3: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_8: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_3);  bernoulli_3 = None
    convert_element_type_4: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_8, torch.bool);  sub_8 = None
    where_5: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_4, full_default_1, view_17);  view_17 = None
    mul_11: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_5, 1.1111111111111112);  where_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_9: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_11, add_7);  mul_11 = add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_4: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_9, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_9: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_9, mean_4);  add_9 = mean_4 = None
    pow_3: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_9, 2)
    mean_5: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_3, [-1], True);  pow_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_10: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_5, 1e-07);  mean_5 = None
    sqrt_3: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_10);  add_10 = None
    div_4: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_9, sqrt_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_12: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_7, div_4);  div_4 = None
    add_11: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_12, primals_8);  mul_12 = primals_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_13: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_84, [1, 0]);  primals_84 = None
    view_18: "f32[512, 768]" = torch.ops.aten.reshape.default(add_11, [512, 768])
    mm_1: "f32[512, 2304]" = torch.ops.aten.mm.default(view_18, permute_13)
    view_19: "f32[1, 512, 2304]" = torch.ops.aten.reshape.default(mm_1, [1, 512, 2304]);  mm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_20: "f32[1, 512, 12, 192]" = torch.ops.aten.reshape.default(view_19, [1, 512, 12, -1]);  view_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_14: "f32[1, 12, 512, 192]" = torch.ops.aten.permute.default(view_20, [0, 2, 1, 3]);  view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    split_1 = torch.ops.aten.split.Tensor(permute_14, 64, -1);  permute_14 = None
    getitem_3: "f32[1, 12, 512, 64]" = split_1[0]
    getitem_4: "f32[1, 12, 512, 64]" = split_1[1]
    getitem_5: "f32[1, 12, 512, 64]" = split_1[2];  split_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    unsqueeze_8: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_9, 0);  primals_9 = None
    unsqueeze_9: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, 1);  unsqueeze_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_21: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_9, [1, 1, 12, -1]);  unsqueeze_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_15: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_21, [0, 2, 1, 3]);  view_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    add_12: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_3, permute_15);  getitem_3 = permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    unsqueeze_10: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_10, 0);  primals_10 = None
    unsqueeze_11: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 1);  unsqueeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_22: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_11, [1, 1, 12, -1]);  unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_16: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_22, [0, 2, 1, 3]);  view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    add_13: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_5, permute_16);  getitem_5 = permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_5: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(add_12, full_default_2);  add_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_17: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(getitem_4, [0, 1, 3, 2]);  getitem_4 = None
    expand_4: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(div_5, [1, 12, 512, 64]);  div_5 = None
    view_23: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_4, [12, 512, 64]);  expand_4 = None
    expand_5: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_17, [1, 12, 64, 512]);  permute_17 = None
    view_24: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_5, [12, 64, 512]);  expand_5 = None
    bmm_2: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_23, view_24)
    view_25: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_2, [1, 12, 512, 512]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    where_6: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_4, view_25);  view_25 = None
    amax_1: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where_6, [-1], True)
    sub_11: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where_6, amax_1);  where_6 = amax_1 = None
    exp_1: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
    sum_2: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_6: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    where_7: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_1, div_6);  div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    bernoulli_4: "f32[1, 12, 512, 512]" = torch.ops.aten.bernoulli.p(permute_6, 0.9)
    sub_12: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_4);  bernoulli_4 = None
    convert_element_type_6: "b8[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_12, torch.bool);  sub_12 = None
    where_8: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_6, full_default_1, where_7)
    mul_14: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_8, 1.1111111111111112);  where_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_6: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(mul_14, [1, 12, 512, 512]);  mul_14 = None
    view_26: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_6, [12, 512, 512]);  expand_6 = None
    expand_7: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(add_13, [1, 12, 512, 64]);  add_13 = None
    view_27: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_7, [12, 512, 64]);  expand_7 = None
    bmm_3: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_26, view_27)
    view_28: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_3, [1, 12, 512, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_19: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
    clone_1: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_29: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_1, [1, 512, -1]);  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_30: "f32[512, 768]" = torch.ops.aten.reshape.default(view_29, [512, 768]);  view_29 = None
    permute_20: "f32[768, 768]" = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
    
    # No stacktrace found for following nodes
    mm_default_22: "f32[512, 768]" = torch.ops.aten.mm.default(view_30, permute_20)
    add_tensor_22: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_22, primals_86);  mm_default_22 = primals_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_31: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_22, [1, 512, 768]);  add_tensor_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    bernoulli_5: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_13: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_5);  bernoulli_5 = None
    convert_element_type_7: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_13, torch.bool);  sub_13 = None
    where_9: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_7, full_default_1, view_31);  view_31 = None
    mul_15: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_9, 1.1111111111111112);  where_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_14: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_15, add_11);  mul_15 = add_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_6: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_14, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_14: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_14, mean_6);  add_14 = mean_6 = None
    pow_4: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_14, 2)
    mean_7: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_4, [-1], True);  pow_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_15: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_7, 1e-07);  mean_7 = None
    sqrt_5: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_15);  add_15 = None
    div_7: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_14, sqrt_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_16: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_11, div_7);  div_7 = None
    add_16: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_16, primals_12);  mul_16 = primals_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_32: "f32[512, 768]" = torch.ops.aten.reshape.default(add_16, [512, 768])
    permute_22: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
    addmm_4: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_88, view_32, permute_22);  primals_88 = None
    view_33: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_4, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_17: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_33, 0.5)
    mul_18: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_33, 0.7071067811865476);  view_33 = None
    erf_1: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_18);  mul_18 = None
    add_17: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_19: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_17, add_17);  mul_17 = add_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_34: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_19, [512, 3072]);  mul_19 = None
    permute_23: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_89, [1, 0]);  primals_89 = None
    
    # No stacktrace found for following nodes
    mm_default_21: "f32[512, 768]" = torch.ops.aten.mm.default(view_34, permute_23)
    add_tensor_21: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_21, primals_90);  mm_default_21 = primals_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_35: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_21, [1, 512, 768]);  add_tensor_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    bernoulli_6: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_16: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_6);  bernoulli_6 = None
    convert_element_type_8: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_16, torch.bool);  sub_16 = None
    where_10: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_8, full_default_1, view_35);  view_35 = None
    mul_20: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_10, 1.1111111111111112);  where_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_18: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_20, add_16);  mul_20 = add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_8: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_18, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_17: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_18, mean_8);  add_18 = mean_8 = None
    pow_5: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_17, 2)
    mean_9: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_5, [-1], True);  pow_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_19: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_9, 1e-07);  mean_9 = None
    sqrt_6: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_19);  add_19 = None
    div_8: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_17, sqrt_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_21: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_13, div_8);  div_8 = None
    add_20: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_21, primals_14);  mul_21 = primals_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_25: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_91, [1, 0]);  primals_91 = None
    view_36: "f32[512, 768]" = torch.ops.aten.reshape.default(add_20, [512, 768])
    mm_2: "f32[512, 2304]" = torch.ops.aten.mm.default(view_36, permute_25)
    view_37: "f32[1, 512, 2304]" = torch.ops.aten.reshape.default(mm_2, [1, 512, 2304]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_38: "f32[1, 512, 12, 192]" = torch.ops.aten.reshape.default(view_37, [1, 512, 12, -1]);  view_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_26: "f32[1, 12, 512, 192]" = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    split_2 = torch.ops.aten.split.Tensor(permute_26, 64, -1);  permute_26 = None
    getitem_6: "f32[1, 12, 512, 64]" = split_2[0]
    getitem_7: "f32[1, 12, 512, 64]" = split_2[1]
    getitem_8: "f32[1, 12, 512, 64]" = split_2[2];  split_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    unsqueeze_12: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_15, 0);  primals_15 = None
    unsqueeze_13: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, 1);  unsqueeze_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_39: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_13, [1, 1, 12, -1]);  unsqueeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_27: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_39, [0, 2, 1, 3]);  view_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    add_21: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_6, permute_27);  getitem_6 = permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    unsqueeze_14: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_16, 0);  primals_16 = None
    unsqueeze_15: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, 1);  unsqueeze_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_40: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_15, [1, 1, 12, -1]);  unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_28: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    add_22: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_8, permute_28);  getitem_8 = permute_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_9: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(add_21, full_default_2);  add_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_29: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(getitem_7, [0, 1, 3, 2]);  getitem_7 = None
    expand_8: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(div_9, [1, 12, 512, 64]);  div_9 = None
    view_41: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_8, [12, 512, 64]);  expand_8 = None
    expand_9: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_29, [1, 12, 64, 512]);  permute_29 = None
    view_42: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_9, [12, 64, 512]);  expand_9 = None
    bmm_4: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_41, view_42)
    view_43: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_4, [1, 12, 512, 512]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    where_11: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_4, view_43);  view_43 = None
    amax_2: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where_11, [-1], True)
    sub_19: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where_11, amax_2);  where_11 = amax_2 = None
    exp_2: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
    sum_3: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_10: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    where_12: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_1, div_10);  div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    bernoulli_7: "f32[1, 12, 512, 512]" = torch.ops.aten.bernoulli.p(permute_6, 0.9)
    sub_20: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_7);  bernoulli_7 = None
    convert_element_type_10: "b8[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_20, torch.bool);  sub_20 = None
    where_13: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_10, full_default_1, where_12)
    mul_23: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_13, 1.1111111111111112);  where_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_10: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(mul_23, [1, 12, 512, 512]);  mul_23 = None
    view_44: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_10, [12, 512, 512]);  expand_10 = None
    expand_11: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(add_22, [1, 12, 512, 64]);  add_22 = None
    view_45: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_11, [12, 512, 64]);  expand_11 = None
    bmm_5: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_44, view_45)
    view_46: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_5, [1, 12, 512, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_31: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None
    clone_2: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_31, memory_format = torch.contiguous_format);  permute_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_47: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_2, [1, 512, -1]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_48: "f32[512, 768]" = torch.ops.aten.reshape.default(view_47, [512, 768]);  view_47 = None
    permute_32: "f32[768, 768]" = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
    
    # No stacktrace found for following nodes
    mm_default_20: "f32[512, 768]" = torch.ops.aten.mm.default(view_48, permute_32)
    add_tensor_20: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_20, primals_93);  mm_default_20 = primals_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_49: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_20, [1, 512, 768]);  add_tensor_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    bernoulli_8: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_21: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_8);  bernoulli_8 = None
    convert_element_type_11: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_21, torch.bool);  sub_21 = None
    where_14: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_11, full_default_1, view_49);  view_49 = None
    mul_24: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_14, 1.1111111111111112);  where_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_23: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_24, add_20);  mul_24 = add_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_10: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_23, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_22: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_23, mean_10);  add_23 = mean_10 = None
    pow_6: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_22, 2)
    mean_11: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_6, [-1], True);  pow_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_24: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_11, 1e-07);  mean_11 = None
    sqrt_8: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_24);  add_24 = None
    div_11: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_22, sqrt_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_25: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_17, div_11);  div_11 = None
    add_25: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_25, primals_18);  mul_25 = primals_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_50: "f32[512, 768]" = torch.ops.aten.reshape.default(add_25, [512, 768])
    permute_34: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_94, [1, 0]);  primals_94 = None
    addmm_7: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_95, view_50, permute_34);  primals_95 = None
    view_51: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_7, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_26: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_51, 0.5)
    mul_27: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_51, 0.7071067811865476);  view_51 = None
    erf_2: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_27);  mul_27 = None
    add_26: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_28: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_26, add_26);  mul_26 = add_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_52: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_28, [512, 3072]);  mul_28 = None
    permute_35: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_96, [1, 0]);  primals_96 = None
    
    # No stacktrace found for following nodes
    mm_default_19: "f32[512, 768]" = torch.ops.aten.mm.default(view_52, permute_35)
    add_tensor_19: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_19, primals_97);  mm_default_19 = primals_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_53: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_19, [1, 512, 768]);  add_tensor_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    bernoulli_9: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_24: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_9);  bernoulli_9 = None
    convert_element_type_12: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_24, torch.bool);  sub_24 = None
    where_15: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_12, full_default_1, view_53);  view_53 = None
    mul_29: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_15, 1.1111111111111112);  where_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_27: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_29, add_25);  mul_29 = add_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_12: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_27, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_25: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_27, mean_12);  add_27 = mean_12 = None
    pow_7: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_25, 2)
    mean_13: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_7, [-1], True);  pow_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_28: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_13, 1e-07);  mean_13 = None
    sqrt_9: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_28);  add_28 = None
    div_12: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_25, sqrt_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_30: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_19, div_12);  div_12 = None
    add_29: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_30, primals_20);  mul_30 = primals_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_37: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_98, [1, 0]);  primals_98 = None
    view_54: "f32[512, 768]" = torch.ops.aten.reshape.default(add_29, [512, 768])
    mm_3: "f32[512, 2304]" = torch.ops.aten.mm.default(view_54, permute_37)
    view_55: "f32[1, 512, 2304]" = torch.ops.aten.reshape.default(mm_3, [1, 512, 2304]);  mm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_56: "f32[1, 512, 12, 192]" = torch.ops.aten.reshape.default(view_55, [1, 512, 12, -1]);  view_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_38: "f32[1, 12, 512, 192]" = torch.ops.aten.permute.default(view_56, [0, 2, 1, 3]);  view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    split_3 = torch.ops.aten.split.Tensor(permute_38, 64, -1);  permute_38 = None
    getitem_9: "f32[1, 12, 512, 64]" = split_3[0]
    getitem_10: "f32[1, 12, 512, 64]" = split_3[1]
    getitem_11: "f32[1, 12, 512, 64]" = split_3[2];  split_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    unsqueeze_16: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_21, 0);  primals_21 = None
    unsqueeze_17: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, 1);  unsqueeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_57: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_17, [1, 1, 12, -1]);  unsqueeze_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_39: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_57, [0, 2, 1, 3]);  view_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    add_30: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_9, permute_39);  getitem_9 = permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    unsqueeze_18: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_22, 0);  primals_22 = None
    unsqueeze_19: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, 1);  unsqueeze_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_58: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_19, [1, 1, 12, -1]);  unsqueeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_40: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    add_31: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_11, permute_40);  getitem_11 = permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_13: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(add_30, full_default_2);  add_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_41: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(getitem_10, [0, 1, 3, 2]);  getitem_10 = None
    expand_12: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(div_13, [1, 12, 512, 64]);  div_13 = None
    view_59: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_12, [12, 512, 64]);  expand_12 = None
    expand_13: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_41, [1, 12, 64, 512]);  permute_41 = None
    view_60: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_13, [12, 64, 512]);  expand_13 = None
    bmm_6: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_59, view_60)
    view_61: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_6, [1, 12, 512, 512]);  bmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    where_16: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_4, view_61);  view_61 = None
    amax_3: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where_16, [-1], True)
    sub_27: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where_16, amax_3);  where_16 = amax_3 = None
    exp_3: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_27);  sub_27 = None
    sum_4: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_14: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    where_17: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_1, div_14);  div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    bernoulli_10: "f32[1, 12, 512, 512]" = torch.ops.aten.bernoulli.p(permute_6, 0.9)
    sub_28: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_10);  bernoulli_10 = None
    convert_element_type_14: "b8[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_28, torch.bool);  sub_28 = None
    where_18: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_14, full_default_1, where_17)
    mul_32: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_18, 1.1111111111111112);  where_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_14: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(mul_32, [1, 12, 512, 512]);  mul_32 = None
    view_62: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_14, [12, 512, 512]);  expand_14 = None
    expand_15: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(add_31, [1, 12, 512, 64]);  add_31 = None
    view_63: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_15, [12, 512, 64]);  expand_15 = None
    bmm_7: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_62, view_63)
    view_64: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_7, [1, 12, 512, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_43: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_64, [0, 2, 1, 3]);  view_64 = None
    clone_3: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_43, memory_format = torch.contiguous_format);  permute_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_65: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_3, [1, 512, -1]);  clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_66: "f32[512, 768]" = torch.ops.aten.reshape.default(view_65, [512, 768]);  view_65 = None
    permute_44: "f32[768, 768]" = torch.ops.aten.permute.default(primals_99, [1, 0]);  primals_99 = None
    
    # No stacktrace found for following nodes
    mm_default_18: "f32[512, 768]" = torch.ops.aten.mm.default(view_66, permute_44)
    add_tensor_18: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_18, primals_100);  mm_default_18 = primals_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_67: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_18, [1, 512, 768]);  add_tensor_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    bernoulli_11: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_29: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_11);  bernoulli_11 = None
    convert_element_type_15: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_29, torch.bool);  sub_29 = None
    where_19: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_15, full_default_1, view_67);  view_67 = None
    mul_33: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_19, 1.1111111111111112);  where_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_32: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_33, add_29);  mul_33 = add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_14: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_32, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_30: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_32, mean_14);  add_32 = mean_14 = None
    pow_8: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_30, 2)
    mean_15: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_8, [-1], True);  pow_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_33: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_15, 1e-07);  mean_15 = None
    sqrt_11: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_33);  add_33 = None
    div_15: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_30, sqrt_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_34: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_23, div_15);  div_15 = None
    add_34: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_34, primals_24);  mul_34 = primals_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_68: "f32[512, 768]" = torch.ops.aten.reshape.default(add_34, [512, 768])
    permute_46: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_101, [1, 0]);  primals_101 = None
    addmm_10: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_102, view_68, permute_46);  primals_102 = None
    view_69: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_10, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_35: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_69, 0.5)
    mul_36: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_69, 0.7071067811865476);  view_69 = None
    erf_3: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_36);  mul_36 = None
    add_35: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_37: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_35, add_35);  mul_35 = add_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_70: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_37, [512, 3072]);  mul_37 = None
    permute_47: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_103, [1, 0]);  primals_103 = None
    
    # No stacktrace found for following nodes
    mm_default_17: "f32[512, 768]" = torch.ops.aten.mm.default(view_70, permute_47)
    add_tensor_17: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_17, primals_104);  mm_default_17 = primals_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_71: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_17, [1, 512, 768]);  add_tensor_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    bernoulli_12: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_32: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_12);  bernoulli_12 = None
    convert_element_type_16: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_32, torch.bool);  sub_32 = None
    where_20: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_16, full_default_1, view_71);  view_71 = None
    mul_38: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_20, 1.1111111111111112);  where_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_36: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_38, add_34);  mul_38 = add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_16: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_36, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_33: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_36, mean_16);  add_36 = mean_16 = None
    pow_9: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_33, 2)
    mean_17: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_9, [-1], True);  pow_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_37: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_17, 1e-07);  mean_17 = None
    sqrt_12: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_37);  add_37 = None
    div_16: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_33, sqrt_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_39: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_25, div_16);  div_16 = None
    add_38: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_39, primals_26);  mul_39 = primals_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_49: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
    view_72: "f32[512, 768]" = torch.ops.aten.reshape.default(add_38, [512, 768])
    mm_4: "f32[512, 2304]" = torch.ops.aten.mm.default(view_72, permute_49)
    view_73: "f32[1, 512, 2304]" = torch.ops.aten.reshape.default(mm_4, [1, 512, 2304]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_74: "f32[1, 512, 12, 192]" = torch.ops.aten.reshape.default(view_73, [1, 512, 12, -1]);  view_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_50: "f32[1, 12, 512, 192]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    split_4 = torch.ops.aten.split.Tensor(permute_50, 64, -1);  permute_50 = None
    getitem_12: "f32[1, 12, 512, 64]" = split_4[0]
    getitem_13: "f32[1, 12, 512, 64]" = split_4[1]
    getitem_14: "f32[1, 12, 512, 64]" = split_4[2];  split_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    unsqueeze_20: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_27, 0);  primals_27 = None
    unsqueeze_21: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, 1);  unsqueeze_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_75: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_21, [1, 1, 12, -1]);  unsqueeze_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_51: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    add_39: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_12, permute_51);  getitem_12 = permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    unsqueeze_22: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_28, 0);  primals_28 = None
    unsqueeze_23: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, 1);  unsqueeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_76: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_23, [1, 1, 12, -1]);  unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_52: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    add_40: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_14, permute_52);  getitem_14 = permute_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_17: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(add_39, full_default_2);  add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_53: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(getitem_13, [0, 1, 3, 2]);  getitem_13 = None
    expand_16: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(div_17, [1, 12, 512, 64]);  div_17 = None
    view_77: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_16, [12, 512, 64]);  expand_16 = None
    expand_17: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_53, [1, 12, 64, 512]);  permute_53 = None
    view_78: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_17, [12, 64, 512]);  expand_17 = None
    bmm_8: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_77, view_78)
    view_79: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_8, [1, 12, 512, 512]);  bmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    where_21: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_4, view_79);  view_79 = None
    amax_4: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where_21, [-1], True)
    sub_35: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where_21, amax_4);  where_21 = amax_4 = None
    exp_4: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_35);  sub_35 = None
    sum_5: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_18: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    where_22: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_1, div_18);  div_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    bernoulli_13: "f32[1, 12, 512, 512]" = torch.ops.aten.bernoulli.p(permute_6, 0.9)
    sub_36: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_13);  bernoulli_13 = None
    convert_element_type_18: "b8[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_36, torch.bool);  sub_36 = None
    where_23: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_18, full_default_1, where_22)
    mul_41: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_23, 1.1111111111111112);  where_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_18: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(mul_41, [1, 12, 512, 512]);  mul_41 = None
    view_80: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_18, [12, 512, 512]);  expand_18 = None
    expand_19: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(add_40, [1, 12, 512, 64]);  add_40 = None
    view_81: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_19, [12, 512, 64]);  expand_19 = None
    bmm_9: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_80, view_81)
    view_82: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_9, [1, 12, 512, 64]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_55: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
    clone_4: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_55, memory_format = torch.contiguous_format);  permute_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_83: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_4, [1, 512, -1]);  clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_84: "f32[512, 768]" = torch.ops.aten.reshape.default(view_83, [512, 768]);  view_83 = None
    permute_56: "f32[768, 768]" = torch.ops.aten.permute.default(primals_106, [1, 0]);  primals_106 = None
    
    # No stacktrace found for following nodes
    mm_default_16: "f32[512, 768]" = torch.ops.aten.mm.default(view_84, permute_56)
    add_tensor_16: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_16, primals_107);  mm_default_16 = primals_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_85: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_16, [1, 512, 768]);  add_tensor_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    bernoulli_14: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_37: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_14);  bernoulli_14 = None
    convert_element_type_19: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_37, torch.bool);  sub_37 = None
    where_24: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_19, full_default_1, view_85);  view_85 = None
    mul_42: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_24, 1.1111111111111112);  where_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_41: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_42, add_38);  mul_42 = add_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_18: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_41, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_38: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_41, mean_18);  add_41 = mean_18 = None
    pow_10: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_38, 2)
    mean_19: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_10, [-1], True);  pow_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_42: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_19, 1e-07);  mean_19 = None
    sqrt_14: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_42);  add_42 = None
    div_19: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_38, sqrt_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_43: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_29, div_19);  div_19 = None
    add_43: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_43, primals_30);  mul_43 = primals_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_86: "f32[512, 768]" = torch.ops.aten.reshape.default(add_43, [512, 768])
    permute_58: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_108, [1, 0]);  primals_108 = None
    addmm_13: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_109, view_86, permute_58);  primals_109 = None
    view_87: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_13, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_44: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_87, 0.5)
    mul_45: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_87, 0.7071067811865476);  view_87 = None
    erf_4: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_45);  mul_45 = None
    add_44: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_46: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_44, add_44);  mul_44 = add_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_88: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_46, [512, 3072]);  mul_46 = None
    permute_59: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_110, [1, 0]);  primals_110 = None
    
    # No stacktrace found for following nodes
    mm_default_15: "f32[512, 768]" = torch.ops.aten.mm.default(view_88, permute_59)
    add_tensor_15: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_15, primals_111);  mm_default_15 = primals_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_89: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_15, [1, 512, 768]);  add_tensor_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    bernoulli_15: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_40: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_15);  bernoulli_15 = None
    convert_element_type_20: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_40, torch.bool);  sub_40 = None
    where_25: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_20, full_default_1, view_89);  view_89 = None
    mul_47: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_25, 1.1111111111111112);  where_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_45: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_47, add_43);  mul_47 = add_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_20: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_45, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_41: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_45, mean_20);  add_45 = mean_20 = None
    pow_11: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_41, 2)
    mean_21: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_11, [-1], True);  pow_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_46: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_21, 1e-07);  mean_21 = None
    sqrt_15: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_46);  add_46 = None
    div_20: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_41, sqrt_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_48: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_31, div_20);  div_20 = None
    add_47: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_48, primals_32);  mul_48 = primals_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_61: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_112, [1, 0]);  primals_112 = None
    view_90: "f32[512, 768]" = torch.ops.aten.reshape.default(add_47, [512, 768])
    mm_5: "f32[512, 2304]" = torch.ops.aten.mm.default(view_90, permute_61)
    view_91: "f32[1, 512, 2304]" = torch.ops.aten.reshape.default(mm_5, [1, 512, 2304]);  mm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_92: "f32[1, 512, 12, 192]" = torch.ops.aten.reshape.default(view_91, [1, 512, 12, -1]);  view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_62: "f32[1, 12, 512, 192]" = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    split_5 = torch.ops.aten.split.Tensor(permute_62, 64, -1);  permute_62 = None
    getitem_15: "f32[1, 12, 512, 64]" = split_5[0]
    getitem_16: "f32[1, 12, 512, 64]" = split_5[1]
    getitem_17: "f32[1, 12, 512, 64]" = split_5[2];  split_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    unsqueeze_24: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_33, 0);  primals_33 = None
    unsqueeze_25: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, 1);  unsqueeze_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_93: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_25, [1, 1, 12, -1]);  unsqueeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_63: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    add_48: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_15, permute_63);  getitem_15 = permute_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    unsqueeze_26: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_34, 0);  primals_34 = None
    unsqueeze_27: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, 1);  unsqueeze_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_94: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_27, [1, 1, 12, -1]);  unsqueeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_64: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    add_49: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_17, permute_64);  getitem_17 = permute_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_21: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(add_48, full_default_2);  add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_65: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(getitem_16, [0, 1, 3, 2]);  getitem_16 = None
    expand_20: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(div_21, [1, 12, 512, 64]);  div_21 = None
    view_95: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_20, [12, 512, 64]);  expand_20 = None
    expand_21: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_65, [1, 12, 64, 512]);  permute_65 = None
    view_96: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_21, [12, 64, 512]);  expand_21 = None
    bmm_10: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_95, view_96)
    view_97: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_10, [1, 12, 512, 512]);  bmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    where_26: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_4, view_97);  view_97 = None
    amax_5: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where_26, [-1], True)
    sub_43: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where_26, amax_5);  where_26 = amax_5 = None
    exp_5: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_43);  sub_43 = None
    sum_6: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_22: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    where_27: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_1, div_22);  div_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    bernoulli_16: "f32[1, 12, 512, 512]" = torch.ops.aten.bernoulli.p(permute_6, 0.9)
    sub_44: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_16);  bernoulli_16 = None
    convert_element_type_22: "b8[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_44, torch.bool);  sub_44 = None
    where_28: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_22, full_default_1, where_27)
    mul_50: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_28, 1.1111111111111112);  where_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_22: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(mul_50, [1, 12, 512, 512]);  mul_50 = None
    view_98: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_22, [12, 512, 512]);  expand_22 = None
    expand_23: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(add_49, [1, 12, 512, 64]);  add_49 = None
    view_99: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_23, [12, 512, 64]);  expand_23 = None
    bmm_11: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_98, view_99)
    view_100: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_11, [1, 12, 512, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_67: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_100, [0, 2, 1, 3]);  view_100 = None
    clone_5: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_67, memory_format = torch.contiguous_format);  permute_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_101: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_5, [1, 512, -1]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_102: "f32[512, 768]" = torch.ops.aten.reshape.default(view_101, [512, 768]);  view_101 = None
    permute_68: "f32[768, 768]" = torch.ops.aten.permute.default(primals_113, [1, 0]);  primals_113 = None
    
    # No stacktrace found for following nodes
    mm_default_14: "f32[512, 768]" = torch.ops.aten.mm.default(view_102, permute_68)
    add_tensor_14: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_14, primals_114);  mm_default_14 = primals_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_103: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_14, [1, 512, 768]);  add_tensor_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    bernoulli_17: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_45: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_17);  bernoulli_17 = None
    convert_element_type_23: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_45, torch.bool);  sub_45 = None
    where_29: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_23, full_default_1, view_103);  view_103 = None
    mul_51: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_29, 1.1111111111111112);  where_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_50: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_51, add_47);  mul_51 = add_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_22: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_50, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_46: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_50, mean_22);  add_50 = mean_22 = None
    pow_12: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_46, 2)
    mean_23: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_12, [-1], True);  pow_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_51: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_23, 1e-07);  mean_23 = None
    sqrt_17: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_51);  add_51 = None
    div_23: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_46, sqrt_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_52: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_35, div_23);  div_23 = None
    add_52: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_52, primals_36);  mul_52 = primals_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_104: "f32[512, 768]" = torch.ops.aten.reshape.default(add_52, [512, 768])
    permute_70: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
    addmm_16: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_116, view_104, permute_70);  primals_116 = None
    view_105: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_16, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_53: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_105, 0.5)
    mul_54: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_105, 0.7071067811865476);  view_105 = None
    erf_5: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_54);  mul_54 = None
    add_53: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_55: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_53, add_53);  mul_53 = add_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_106: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_55, [512, 3072]);  mul_55 = None
    permute_71: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_117, [1, 0]);  primals_117 = None
    
    # No stacktrace found for following nodes
    mm_default_13: "f32[512, 768]" = torch.ops.aten.mm.default(view_106, permute_71)
    add_tensor_13: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_13, primals_118);  mm_default_13 = primals_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_107: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_13, [1, 512, 768]);  add_tensor_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    bernoulli_18: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_48: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_18);  bernoulli_18 = None
    convert_element_type_24: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_48, torch.bool);  sub_48 = None
    where_30: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_24, full_default_1, view_107);  view_107 = None
    mul_56: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_30, 1.1111111111111112);  where_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_54: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_56, add_52);  mul_56 = add_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_24: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_54, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_49: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_54, mean_24);  add_54 = mean_24 = None
    pow_13: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_49, 2)
    mean_25: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_13, [-1], True);  pow_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_55: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_25, 1e-07);  mean_25 = None
    sqrt_18: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_55);  add_55 = None
    div_24: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_49, sqrt_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_57: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_37, div_24);  div_24 = None
    add_56: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_57, primals_38);  mul_57 = primals_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_73: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_119, [1, 0]);  primals_119 = None
    view_108: "f32[512, 768]" = torch.ops.aten.reshape.default(add_56, [512, 768])
    mm_6: "f32[512, 2304]" = torch.ops.aten.mm.default(view_108, permute_73)
    view_109: "f32[1, 512, 2304]" = torch.ops.aten.reshape.default(mm_6, [1, 512, 2304]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_110: "f32[1, 512, 12, 192]" = torch.ops.aten.reshape.default(view_109, [1, 512, 12, -1]);  view_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_74: "f32[1, 12, 512, 192]" = torch.ops.aten.permute.default(view_110, [0, 2, 1, 3]);  view_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    split_6 = torch.ops.aten.split.Tensor(permute_74, 64, -1);  permute_74 = None
    getitem_18: "f32[1, 12, 512, 64]" = split_6[0]
    getitem_19: "f32[1, 12, 512, 64]" = split_6[1]
    getitem_20: "f32[1, 12, 512, 64]" = split_6[2];  split_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    unsqueeze_28: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_39, 0);  primals_39 = None
    unsqueeze_29: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, 1);  unsqueeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_111: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_29, [1, 1, 12, -1]);  unsqueeze_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_75: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_111, [0, 2, 1, 3]);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    add_57: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_18, permute_75);  getitem_18 = permute_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    unsqueeze_30: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_40, 0);  primals_40 = None
    unsqueeze_31: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, 1);  unsqueeze_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_112: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_31, [1, 1, 12, -1]);  unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_76: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    add_58: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_20, permute_76);  getitem_20 = permute_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_25: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(add_57, full_default_2);  add_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_77: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(getitem_19, [0, 1, 3, 2]);  getitem_19 = None
    expand_24: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(div_25, [1, 12, 512, 64]);  div_25 = None
    view_113: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_24, [12, 512, 64]);  expand_24 = None
    expand_25: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_77, [1, 12, 64, 512]);  permute_77 = None
    view_114: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_25, [12, 64, 512]);  expand_25 = None
    bmm_12: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_113, view_114)
    view_115: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_12, [1, 12, 512, 512]);  bmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    where_31: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_4, view_115);  view_115 = None
    amax_6: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where_31, [-1], True)
    sub_51: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where_31, amax_6);  where_31 = amax_6 = None
    exp_6: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_51);  sub_51 = None
    sum_7: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_26: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    where_32: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_1, div_26);  div_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    bernoulli_19: "f32[1, 12, 512, 512]" = torch.ops.aten.bernoulli.p(permute_6, 0.9)
    sub_52: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_19);  bernoulli_19 = None
    convert_element_type_26: "b8[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_52, torch.bool);  sub_52 = None
    where_33: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_26, full_default_1, where_32)
    mul_59: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_33, 1.1111111111111112);  where_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_26: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(mul_59, [1, 12, 512, 512]);  mul_59 = None
    view_116: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_26, [12, 512, 512]);  expand_26 = None
    expand_27: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(add_58, [1, 12, 512, 64]);  add_58 = None
    view_117: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_27, [12, 512, 64]);  expand_27 = None
    bmm_13: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_116, view_117)
    view_118: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_13, [1, 12, 512, 64]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_79: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
    clone_6: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_119: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_6, [1, 512, -1]);  clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_120: "f32[512, 768]" = torch.ops.aten.reshape.default(view_119, [512, 768]);  view_119 = None
    permute_80: "f32[768, 768]" = torch.ops.aten.permute.default(primals_120, [1, 0]);  primals_120 = None
    
    # No stacktrace found for following nodes
    mm_default_12: "f32[512, 768]" = torch.ops.aten.mm.default(view_120, permute_80)
    add_tensor_12: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_12, primals_121);  mm_default_12 = primals_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_121: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_12, [1, 512, 768]);  add_tensor_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    bernoulli_20: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_53: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_20);  bernoulli_20 = None
    convert_element_type_27: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_53, torch.bool);  sub_53 = None
    where_34: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_27, full_default_1, view_121);  view_121 = None
    mul_60: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_34, 1.1111111111111112);  where_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_59: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_60, add_56);  mul_60 = add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_26: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_59, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_54: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_59, mean_26);  add_59 = mean_26 = None
    pow_14: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_54, 2)
    mean_27: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_14, [-1], True);  pow_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_60: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_27, 1e-07);  mean_27 = None
    sqrt_20: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_60);  add_60 = None
    div_27: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_54, sqrt_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_61: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_41, div_27);  div_27 = None
    add_61: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_61, primals_42);  mul_61 = primals_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_122: "f32[512, 768]" = torch.ops.aten.reshape.default(add_61, [512, 768])
    permute_82: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_122, [1, 0]);  primals_122 = None
    addmm_19: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_123, view_122, permute_82);  primals_123 = None
    view_123: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_19, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_62: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_123, 0.5)
    mul_63: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_123, 0.7071067811865476);  view_123 = None
    erf_6: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_63);  mul_63 = None
    add_62: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_64: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_62, add_62);  mul_62 = add_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_124: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_64, [512, 3072]);  mul_64 = None
    permute_83: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_124, [1, 0]);  primals_124 = None
    
    # No stacktrace found for following nodes
    mm_default_11: "f32[512, 768]" = torch.ops.aten.mm.default(view_124, permute_83)
    add_tensor_11: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_11, primals_125);  mm_default_11 = primals_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_125: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_11, [1, 512, 768]);  add_tensor_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    bernoulli_21: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_56: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_21);  bernoulli_21 = None
    convert_element_type_28: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_56, torch.bool);  sub_56 = None
    where_35: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_28, full_default_1, view_125);  view_125 = None
    mul_65: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_35, 1.1111111111111112);  where_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_63: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_65, add_61);  mul_65 = add_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_28: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_63, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_57: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_63, mean_28);  add_63 = mean_28 = None
    pow_15: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_57, 2)
    mean_29: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_15, [-1], True);  pow_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_64: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_29, 1e-07);  mean_29 = None
    sqrt_21: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_64);  add_64 = None
    div_28: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_57, sqrt_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_66: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_43, div_28);  div_28 = None
    add_65: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_66, primals_44);  mul_66 = primals_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_85: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_126, [1, 0]);  primals_126 = None
    view_126: "f32[512, 768]" = torch.ops.aten.reshape.default(add_65, [512, 768])
    mm_7: "f32[512, 2304]" = torch.ops.aten.mm.default(view_126, permute_85)
    view_127: "f32[1, 512, 2304]" = torch.ops.aten.reshape.default(mm_7, [1, 512, 2304]);  mm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_128: "f32[1, 512, 12, 192]" = torch.ops.aten.reshape.default(view_127, [1, 512, 12, -1]);  view_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_86: "f32[1, 12, 512, 192]" = torch.ops.aten.permute.default(view_128, [0, 2, 1, 3]);  view_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    split_7 = torch.ops.aten.split.Tensor(permute_86, 64, -1);  permute_86 = None
    getitem_21: "f32[1, 12, 512, 64]" = split_7[0]
    getitem_22: "f32[1, 12, 512, 64]" = split_7[1]
    getitem_23: "f32[1, 12, 512, 64]" = split_7[2];  split_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    unsqueeze_32: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_45, 0);  primals_45 = None
    unsqueeze_33: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, 1);  unsqueeze_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_129: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_33, [1, 1, 12, -1]);  unsqueeze_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_87: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_129, [0, 2, 1, 3]);  view_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    add_66: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_21, permute_87);  getitem_21 = permute_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    unsqueeze_34: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_46, 0);  primals_46 = None
    unsqueeze_35: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, 1);  unsqueeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_130: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_35, [1, 1, 12, -1]);  unsqueeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_88: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    add_67: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_23, permute_88);  getitem_23 = permute_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_29: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(add_66, full_default_2);  add_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_89: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(getitem_22, [0, 1, 3, 2]);  getitem_22 = None
    expand_28: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(div_29, [1, 12, 512, 64]);  div_29 = None
    view_131: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_28, [12, 512, 64]);  expand_28 = None
    expand_29: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_89, [1, 12, 64, 512]);  permute_89 = None
    view_132: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_29, [12, 64, 512]);  expand_29 = None
    bmm_14: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_131, view_132)
    view_133: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_14, [1, 12, 512, 512]);  bmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    where_36: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_4, view_133);  view_133 = None
    amax_7: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where_36, [-1], True)
    sub_59: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where_36, amax_7);  where_36 = amax_7 = None
    exp_7: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_59);  sub_59 = None
    sum_8: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_30: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    where_37: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_1, div_30);  div_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    bernoulli_22: "f32[1, 12, 512, 512]" = torch.ops.aten.bernoulli.p(permute_6, 0.9)
    sub_60: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_22);  bernoulli_22 = None
    convert_element_type_30: "b8[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_60, torch.bool);  sub_60 = None
    where_38: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_30, full_default_1, where_37)
    mul_68: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_38, 1.1111111111111112);  where_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_30: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(mul_68, [1, 12, 512, 512]);  mul_68 = None
    view_134: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_30, [12, 512, 512]);  expand_30 = None
    expand_31: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(add_67, [1, 12, 512, 64]);  add_67 = None
    view_135: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_31, [12, 512, 64]);  expand_31 = None
    bmm_15: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_134, view_135)
    view_136: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_15, [1, 12, 512, 64]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_91: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
    clone_7: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_91, memory_format = torch.contiguous_format);  permute_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_137: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_7, [1, 512, -1]);  clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_138: "f32[512, 768]" = torch.ops.aten.reshape.default(view_137, [512, 768]);  view_137 = None
    permute_92: "f32[768, 768]" = torch.ops.aten.permute.default(primals_127, [1, 0]);  primals_127 = None
    
    # No stacktrace found for following nodes
    mm_default_10: "f32[512, 768]" = torch.ops.aten.mm.default(view_138, permute_92)
    add_tensor_10: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_10, primals_128);  mm_default_10 = primals_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_139: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_10, [1, 512, 768]);  add_tensor_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    bernoulli_23: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_61: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_23);  bernoulli_23 = None
    convert_element_type_31: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_61, torch.bool);  sub_61 = None
    where_39: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_31, full_default_1, view_139);  view_139 = None
    mul_69: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_39, 1.1111111111111112);  where_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_68: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_69, add_65);  mul_69 = add_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_30: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_68, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_62: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_68, mean_30);  add_68 = mean_30 = None
    pow_16: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_62, 2)
    mean_31: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_16, [-1], True);  pow_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_69: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_31, 1e-07);  mean_31 = None
    sqrt_23: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_69);  add_69 = None
    div_31: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_62, sqrt_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_70: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_47, div_31);  div_31 = None
    add_70: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_70, primals_48);  mul_70 = primals_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_140: "f32[512, 768]" = torch.ops.aten.reshape.default(add_70, [512, 768])
    permute_94: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_129, [1, 0]);  primals_129 = None
    addmm_22: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_130, view_140, permute_94);  primals_130 = None
    view_141: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_22, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_71: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_141, 0.5)
    mul_72: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_141, 0.7071067811865476);  view_141 = None
    erf_7: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_72);  mul_72 = None
    add_71: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_73: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_71, add_71);  mul_71 = add_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_142: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_73, [512, 3072]);  mul_73 = None
    permute_95: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_131, [1, 0]);  primals_131 = None
    
    # No stacktrace found for following nodes
    mm_default_9: "f32[512, 768]" = torch.ops.aten.mm.default(view_142, permute_95)
    add_tensor_9: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_9, primals_132);  mm_default_9 = primals_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_143: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_9, [1, 512, 768]);  add_tensor_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    bernoulli_24: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_64: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_24);  bernoulli_24 = None
    convert_element_type_32: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_64, torch.bool);  sub_64 = None
    where_40: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_32, full_default_1, view_143);  view_143 = None
    mul_74: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_40, 1.1111111111111112);  where_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_72: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_74, add_70);  mul_74 = add_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_32: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_72, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_65: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_72, mean_32);  add_72 = mean_32 = None
    pow_17: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_65, 2)
    mean_33: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_17, [-1], True);  pow_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_73: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_33, 1e-07);  mean_33 = None
    sqrt_24: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_73);  add_73 = None
    div_32: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_65, sqrt_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_75: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_49, div_32);  div_32 = None
    add_74: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_75, primals_50);  mul_75 = primals_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_97: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_133, [1, 0]);  primals_133 = None
    view_144: "f32[512, 768]" = torch.ops.aten.reshape.default(add_74, [512, 768])
    mm_8: "f32[512, 2304]" = torch.ops.aten.mm.default(view_144, permute_97)
    view_145: "f32[1, 512, 2304]" = torch.ops.aten.reshape.default(mm_8, [1, 512, 2304]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_146: "f32[1, 512, 12, 192]" = torch.ops.aten.reshape.default(view_145, [1, 512, 12, -1]);  view_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_98: "f32[1, 12, 512, 192]" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    split_8 = torch.ops.aten.split.Tensor(permute_98, 64, -1);  permute_98 = None
    getitem_24: "f32[1, 12, 512, 64]" = split_8[0]
    getitem_25: "f32[1, 12, 512, 64]" = split_8[1]
    getitem_26: "f32[1, 12, 512, 64]" = split_8[2];  split_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    unsqueeze_36: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_51, 0);  primals_51 = None
    unsqueeze_37: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, 1);  unsqueeze_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_147: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_37, [1, 1, 12, -1]);  unsqueeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_99: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_147, [0, 2, 1, 3]);  view_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    add_75: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_24, permute_99);  getitem_24 = permute_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    unsqueeze_38: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_52, 0);  primals_52 = None
    unsqueeze_39: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, 1);  unsqueeze_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_148: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_39, [1, 1, 12, -1]);  unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_100: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    add_76: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_26, permute_100);  getitem_26 = permute_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_33: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(add_75, full_default_2);  add_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_101: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(getitem_25, [0, 1, 3, 2]);  getitem_25 = None
    expand_32: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(div_33, [1, 12, 512, 64]);  div_33 = None
    view_149: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_32, [12, 512, 64]);  expand_32 = None
    expand_33: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_101, [1, 12, 64, 512]);  permute_101 = None
    view_150: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_33, [12, 64, 512]);  expand_33 = None
    bmm_16: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_149, view_150)
    view_151: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_16, [1, 12, 512, 512]);  bmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    where_41: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_4, view_151);  view_151 = None
    amax_8: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where_41, [-1], True)
    sub_67: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where_41, amax_8);  where_41 = amax_8 = None
    exp_8: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_67);  sub_67 = None
    sum_9: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_34: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    where_42: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_1, div_34);  div_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    bernoulli_25: "f32[1, 12, 512, 512]" = torch.ops.aten.bernoulli.p(permute_6, 0.9)
    sub_68: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_25);  bernoulli_25 = None
    convert_element_type_34: "b8[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_68, torch.bool);  sub_68 = None
    where_43: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_34, full_default_1, where_42)
    mul_77: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_43, 1.1111111111111112);  where_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_34: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(mul_77, [1, 12, 512, 512]);  mul_77 = None
    view_152: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_34, [12, 512, 512]);  expand_34 = None
    expand_35: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(add_76, [1, 12, 512, 64]);  add_76 = None
    view_153: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_35, [12, 512, 64]);  expand_35 = None
    bmm_17: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_152, view_153)
    view_154: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_17, [1, 12, 512, 64]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_103: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_154, [0, 2, 1, 3]);  view_154 = None
    clone_8: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_103, memory_format = torch.contiguous_format);  permute_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_155: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_8, [1, 512, -1]);  clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_156: "f32[512, 768]" = torch.ops.aten.reshape.default(view_155, [512, 768]);  view_155 = None
    permute_104: "f32[768, 768]" = torch.ops.aten.permute.default(primals_134, [1, 0]);  primals_134 = None
    
    # No stacktrace found for following nodes
    mm_default_8: "f32[512, 768]" = torch.ops.aten.mm.default(view_156, permute_104)
    add_tensor_8: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_8, primals_135);  mm_default_8 = primals_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_157: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_8, [1, 512, 768]);  add_tensor_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    bernoulli_26: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_69: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_26);  bernoulli_26 = None
    convert_element_type_35: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_69, torch.bool);  sub_69 = None
    where_44: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_35, full_default_1, view_157);  view_157 = None
    mul_78: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_44, 1.1111111111111112);  where_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_77: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_78, add_74);  mul_78 = add_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_34: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_77, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_70: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_77, mean_34);  add_77 = mean_34 = None
    pow_18: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_70, 2)
    mean_35: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_18, [-1], True);  pow_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_78: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_35, 1e-07);  mean_35 = None
    sqrt_26: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_78);  add_78 = None
    div_35: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_70, sqrt_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_79: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_53, div_35);  div_35 = None
    add_79: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_79, primals_54);  mul_79 = primals_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_158: "f32[512, 768]" = torch.ops.aten.reshape.default(add_79, [512, 768])
    permute_106: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_136, [1, 0]);  primals_136 = None
    addmm_25: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_137, view_158, permute_106);  primals_137 = None
    view_159: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_25, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_80: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_159, 0.5)
    mul_81: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_159, 0.7071067811865476);  view_159 = None
    erf_8: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_81);  mul_81 = None
    add_80: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_82: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_80, add_80);  mul_80 = add_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_160: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_82, [512, 3072]);  mul_82 = None
    permute_107: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_138, [1, 0]);  primals_138 = None
    
    # No stacktrace found for following nodes
    mm_default_7: "f32[512, 768]" = torch.ops.aten.mm.default(view_160, permute_107)
    add_tensor_7: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_7, primals_139);  mm_default_7 = primals_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_161: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_7, [1, 512, 768]);  add_tensor_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    bernoulli_27: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_72: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_27);  bernoulli_27 = None
    convert_element_type_36: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_72, torch.bool);  sub_72 = None
    where_45: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_36, full_default_1, view_161);  view_161 = None
    mul_83: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_45, 1.1111111111111112);  where_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_81: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_83, add_79);  mul_83 = add_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_36: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_81, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_73: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_81, mean_36);  add_81 = mean_36 = None
    pow_19: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_73, 2)
    mean_37: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_19, [-1], True);  pow_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_82: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_37, 1e-07);  mean_37 = None
    sqrt_27: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_82);  add_82 = None
    div_36: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_73, sqrt_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_84: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_55, div_36);  div_36 = None
    add_83: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_84, primals_56);  mul_84 = primals_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_109: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_140, [1, 0]);  primals_140 = None
    view_162: "f32[512, 768]" = torch.ops.aten.reshape.default(add_83, [512, 768])
    mm_9: "f32[512, 2304]" = torch.ops.aten.mm.default(view_162, permute_109)
    view_163: "f32[1, 512, 2304]" = torch.ops.aten.reshape.default(mm_9, [1, 512, 2304]);  mm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_164: "f32[1, 512, 12, 192]" = torch.ops.aten.reshape.default(view_163, [1, 512, 12, -1]);  view_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_110: "f32[1, 12, 512, 192]" = torch.ops.aten.permute.default(view_164, [0, 2, 1, 3]);  view_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    split_9 = torch.ops.aten.split.Tensor(permute_110, 64, -1);  permute_110 = None
    getitem_27: "f32[1, 12, 512, 64]" = split_9[0]
    getitem_28: "f32[1, 12, 512, 64]" = split_9[1]
    getitem_29: "f32[1, 12, 512, 64]" = split_9[2];  split_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    unsqueeze_40: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_57, 0);  primals_57 = None
    unsqueeze_41: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, 1);  unsqueeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_165: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_41, [1, 1, 12, -1]);  unsqueeze_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_111: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_165, [0, 2, 1, 3]);  view_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    add_84: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_27, permute_111);  getitem_27 = permute_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    unsqueeze_42: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_58, 0);  primals_58 = None
    unsqueeze_43: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, 1);  unsqueeze_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_166: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_43, [1, 1, 12, -1]);  unsqueeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_112: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_166, [0, 2, 1, 3]);  view_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    add_85: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_29, permute_112);  getitem_29 = permute_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_37: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(add_84, full_default_2);  add_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_113: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(getitem_28, [0, 1, 3, 2]);  getitem_28 = None
    expand_36: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(div_37, [1, 12, 512, 64]);  div_37 = None
    view_167: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_36, [12, 512, 64]);  expand_36 = None
    expand_37: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_113, [1, 12, 64, 512]);  permute_113 = None
    view_168: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_37, [12, 64, 512]);  expand_37 = None
    bmm_18: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_167, view_168)
    view_169: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_18, [1, 12, 512, 512]);  bmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    where_46: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_4, view_169);  view_169 = None
    amax_9: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where_46, [-1], True)
    sub_75: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where_46, amax_9);  where_46 = amax_9 = None
    exp_9: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_75);  sub_75 = None
    sum_10: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_38: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    where_47: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_1, div_38);  div_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    bernoulli_28: "f32[1, 12, 512, 512]" = torch.ops.aten.bernoulli.p(permute_6, 0.9)
    sub_76: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_28);  bernoulli_28 = None
    convert_element_type_38: "b8[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_76, torch.bool);  sub_76 = None
    where_48: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_38, full_default_1, where_47)
    mul_86: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_48, 1.1111111111111112);  where_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_38: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(mul_86, [1, 12, 512, 512]);  mul_86 = None
    view_170: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_38, [12, 512, 512]);  expand_38 = None
    expand_39: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(add_85, [1, 12, 512, 64]);  add_85 = None
    view_171: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_39, [12, 512, 64]);  expand_39 = None
    bmm_19: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_170, view_171)
    view_172: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_19, [1, 12, 512, 64]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_115: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_172, [0, 2, 1, 3]);  view_172 = None
    clone_9: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_173: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_9, [1, 512, -1]);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_174: "f32[512, 768]" = torch.ops.aten.reshape.default(view_173, [512, 768]);  view_173 = None
    permute_116: "f32[768, 768]" = torch.ops.aten.permute.default(primals_141, [1, 0]);  primals_141 = None
    
    # No stacktrace found for following nodes
    mm_default_6: "f32[512, 768]" = torch.ops.aten.mm.default(view_174, permute_116)
    add_tensor_6: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_6, primals_142);  mm_default_6 = primals_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_175: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_6, [1, 512, 768]);  add_tensor_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    bernoulli_29: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_77: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_29);  bernoulli_29 = None
    convert_element_type_39: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_77, torch.bool);  sub_77 = None
    where_49: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_39, full_default_1, view_175);  view_175 = None
    mul_87: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_49, 1.1111111111111112);  where_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_86: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_87, add_83);  mul_87 = add_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_38: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_86, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_78: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_86, mean_38);  add_86 = mean_38 = None
    pow_20: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_78, 2)
    mean_39: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_20, [-1], True);  pow_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_87: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_39, 1e-07);  mean_39 = None
    sqrt_29: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_87);  add_87 = None
    div_39: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_78, sqrt_29)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_88: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_59, div_39);  div_39 = None
    add_88: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_88, primals_60);  mul_88 = primals_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_176: "f32[512, 768]" = torch.ops.aten.reshape.default(add_88, [512, 768])
    permute_118: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_143, [1, 0]);  primals_143 = None
    addmm_28: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_144, view_176, permute_118);  primals_144 = None
    view_177: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_28, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_89: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_177, 0.5)
    mul_90: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_177, 0.7071067811865476);  view_177 = None
    erf_9: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_90);  mul_90 = None
    add_89: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_91: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_89, add_89);  mul_89 = add_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_178: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_91, [512, 3072]);  mul_91 = None
    permute_119: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_145, [1, 0]);  primals_145 = None
    
    # No stacktrace found for following nodes
    mm_default_5: "f32[512, 768]" = torch.ops.aten.mm.default(view_178, permute_119)
    add_tensor_5: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_5, primals_146);  mm_default_5 = primals_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_179: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_5, [1, 512, 768]);  add_tensor_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    bernoulli_30: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_80: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_30);  bernoulli_30 = None
    convert_element_type_40: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_80, torch.bool);  sub_80 = None
    where_50: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_40, full_default_1, view_179);  view_179 = None
    mul_92: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_50, 1.1111111111111112);  where_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_90: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_92, add_88);  mul_92 = add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_40: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_90, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_81: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_90, mean_40);  add_90 = mean_40 = None
    pow_21: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_81, 2)
    mean_41: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_21, [-1], True);  pow_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_91: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_41, 1e-07);  mean_41 = None
    sqrt_30: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_91);  add_91 = None
    div_40: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_81, sqrt_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_93: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_61, div_40);  div_40 = None
    add_92: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_93, primals_62);  mul_93 = primals_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_121: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_147, [1, 0]);  primals_147 = None
    view_180: "f32[512, 768]" = torch.ops.aten.reshape.default(add_92, [512, 768])
    mm_10: "f32[512, 2304]" = torch.ops.aten.mm.default(view_180, permute_121)
    view_181: "f32[1, 512, 2304]" = torch.ops.aten.reshape.default(mm_10, [1, 512, 2304]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_182: "f32[1, 512, 12, 192]" = torch.ops.aten.reshape.default(view_181, [1, 512, 12, -1]);  view_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_122: "f32[1, 12, 512, 192]" = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    split_10 = torch.ops.aten.split.Tensor(permute_122, 64, -1);  permute_122 = None
    getitem_30: "f32[1, 12, 512, 64]" = split_10[0]
    getitem_31: "f32[1, 12, 512, 64]" = split_10[1]
    getitem_32: "f32[1, 12, 512, 64]" = split_10[2];  split_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    unsqueeze_44: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_63, 0);  primals_63 = None
    unsqueeze_45: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, 1);  unsqueeze_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_183: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_45, [1, 1, 12, -1]);  unsqueeze_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_123: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    add_93: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_30, permute_123);  getitem_30 = permute_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    unsqueeze_46: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_64, 0);  primals_64 = None
    unsqueeze_47: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, 1);  unsqueeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_184: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_47, [1, 1, 12, -1]);  unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_124: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    add_94: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_32, permute_124);  getitem_32 = permute_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_41: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(add_93, full_default_2);  add_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_125: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(getitem_31, [0, 1, 3, 2]);  getitem_31 = None
    expand_40: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(div_41, [1, 12, 512, 64]);  div_41 = None
    view_185: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_40, [12, 512, 64]);  expand_40 = None
    expand_41: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_125, [1, 12, 64, 512]);  permute_125 = None
    view_186: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_41, [12, 64, 512]);  expand_41 = None
    bmm_20: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_185, view_186)
    view_187: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_20, [1, 12, 512, 512]);  bmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    where_51: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_4, view_187);  view_187 = None
    amax_10: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where_51, [-1], True)
    sub_83: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where_51, amax_10);  where_51 = amax_10 = None
    exp_10: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_83);  sub_83 = None
    sum_11: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_42: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    where_52: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_1, div_42);  div_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    bernoulli_31: "f32[1, 12, 512, 512]" = torch.ops.aten.bernoulli.p(permute_6, 0.9)
    sub_84: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_31);  bernoulli_31 = None
    convert_element_type_42: "b8[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_84, torch.bool);  sub_84 = None
    where_53: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_42, full_default_1, where_52)
    mul_95: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_53, 1.1111111111111112);  where_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_42: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(mul_95, [1, 12, 512, 512]);  mul_95 = None
    view_188: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_42, [12, 512, 512]);  expand_42 = None
    expand_43: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(add_94, [1, 12, 512, 64]);  add_94 = None
    view_189: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_43, [12, 512, 64]);  expand_43 = None
    bmm_21: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_188, view_189)
    view_190: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_21, [1, 12, 512, 64]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_127: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_190, [0, 2, 1, 3]);  view_190 = None
    clone_10: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_127, memory_format = torch.contiguous_format);  permute_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_191: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_10, [1, 512, -1]);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_192: "f32[512, 768]" = torch.ops.aten.reshape.default(view_191, [512, 768]);  view_191 = None
    permute_128: "f32[768, 768]" = torch.ops.aten.permute.default(primals_148, [1, 0]);  primals_148 = None
    
    # No stacktrace found for following nodes
    mm_default_4: "f32[512, 768]" = torch.ops.aten.mm.default(view_192, permute_128)
    add_tensor_4: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_4, primals_149);  mm_default_4 = primals_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_193: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_4, [1, 512, 768]);  add_tensor_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    bernoulli_32: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_85: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_32);  bernoulli_32 = None
    convert_element_type_43: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_85, torch.bool);  sub_85 = None
    where_54: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_43, full_default_1, view_193);  view_193 = None
    mul_96: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_54, 1.1111111111111112);  where_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_95: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_96, add_92);  mul_96 = add_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_42: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_95, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_86: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_95, mean_42);  add_95 = mean_42 = None
    pow_22: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_86, 2)
    mean_43: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_22, [-1], True);  pow_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_96: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_43, 1e-07);  mean_43 = None
    sqrt_32: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_96);  add_96 = None
    div_43: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_86, sqrt_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_97: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_65, div_43);  div_43 = None
    add_97: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_97, primals_66);  mul_97 = primals_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_194: "f32[512, 768]" = torch.ops.aten.reshape.default(add_97, [512, 768])
    permute_130: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_150, [1, 0]);  primals_150 = None
    addmm_31: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_151, view_194, permute_130);  primals_151 = None
    view_195: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_31, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_98: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, 0.5)
    mul_99: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476);  view_195 = None
    erf_10: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_99);  mul_99 = None
    add_98: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_100: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_98, add_98);  mul_98 = add_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_196: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_100, [512, 3072]);  mul_100 = None
    permute_131: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_152, [1, 0]);  primals_152 = None
    
    # No stacktrace found for following nodes
    mm_default_3: "f32[512, 768]" = torch.ops.aten.mm.default(view_196, permute_131)
    add_tensor_3: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_3, primals_153);  mm_default_3 = primals_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_197: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_3, [1, 512, 768]);  add_tensor_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    bernoulli_33: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_88: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_33);  bernoulli_33 = None
    convert_element_type_44: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_88, torch.bool);  sub_88 = None
    where_55: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_44, full_default_1, view_197);  view_197 = None
    mul_101: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_55, 1.1111111111111112);  where_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_99: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_101, add_97);  mul_101 = add_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_44: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_99, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_89: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_99, mean_44);  add_99 = mean_44 = None
    pow_23: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_89, 2)
    mean_45: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_23, [-1], True);  pow_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_100: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_45, 1e-07);  mean_45 = None
    sqrt_33: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_100);  add_100 = None
    div_44: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_89, sqrt_33)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_102: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_67, div_44);  div_44 = None
    add_101: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_102, primals_68);  mul_102 = primals_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_133: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_154, [1, 0]);  primals_154 = None
    view_198: "f32[512, 768]" = torch.ops.aten.reshape.default(add_101, [512, 768])
    mm_11: "f32[512, 2304]" = torch.ops.aten.mm.default(view_198, permute_133)
    view_199: "f32[1, 512, 2304]" = torch.ops.aten.reshape.default(mm_11, [1, 512, 2304]);  mm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_200: "f32[1, 512, 12, 192]" = torch.ops.aten.reshape.default(view_199, [1, 512, 12, -1]);  view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_134: "f32[1, 12, 512, 192]" = torch.ops.aten.permute.default(view_200, [0, 2, 1, 3]);  view_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    split_11 = torch.ops.aten.split.Tensor(permute_134, 64, -1);  permute_134 = None
    getitem_33: "f32[1, 12, 512, 64]" = split_11[0]
    getitem_34: "f32[1, 12, 512, 64]" = split_11[1]
    getitem_35: "f32[1, 12, 512, 64]" = split_11[2];  split_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    unsqueeze_48: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_69, 0);  primals_69 = None
    unsqueeze_49: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, 1);  unsqueeze_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_201: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_49, [1, 1, 12, -1]);  unsqueeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_135: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_201, [0, 2, 1, 3]);  view_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    add_102: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_33, permute_135);  getitem_33 = permute_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    unsqueeze_50: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_70, 0);  primals_70 = None
    unsqueeze_51: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, 1);  unsqueeze_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_202: "f32[1, 1, 12, 64]" = torch.ops.aten.reshape.default(unsqueeze_51, [1, 1, 12, -1]);  unsqueeze_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_136: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_202, [0, 2, 1, 3]);  view_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    add_103: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_35, permute_136);  getitem_35 = permute_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_45: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(add_102, full_default_2);  add_102 = full_default_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_137: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(getitem_34, [0, 1, 3, 2]);  getitem_34 = None
    expand_44: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(div_45, [1, 12, 512, 64]);  div_45 = None
    view_203: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_44, [12, 512, 64]);  expand_44 = None
    expand_45: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_137, [1, 12, 64, 512]);  permute_137 = None
    view_204: "f32[12, 64, 512]" = torch.ops.aten.reshape.default(expand_45, [12, 64, 512]);  expand_45 = None
    bmm_22: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_203, view_204)
    view_205: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_22, [1, 12, 512, 512]);  bmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    where_56: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_4, view_205);  full_default_4 = view_205 = None
    amax_11: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where_56, [-1], True)
    sub_91: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where_56, amax_11);  where_56 = amax_11 = None
    exp_11: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_91);  sub_91 = None
    sum_12: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_46: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    where_57: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(full_default_3, full_default_1, div_46);  full_default_3 = div_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    bernoulli_34: "f32[1, 12, 512, 512]" = torch.ops.aten.bernoulli.p(permute_6, 0.9);  permute_6 = None
    sub_92: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_34);  bernoulli_34 = None
    convert_element_type_46: "b8[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_92, torch.bool);  sub_92 = None
    where_58: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_46, full_default_1, where_57)
    mul_104: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_58, 1.1111111111111112);  where_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_46: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(mul_104, [1, 12, 512, 512]);  mul_104 = None
    view_206: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(expand_46, [12, 512, 512]);  expand_46 = None
    expand_47: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(add_103, [1, 12, 512, 64]);  add_103 = None
    view_207: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(expand_47, [12, 512, 64]);  expand_47 = None
    bmm_23: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_206, view_207)
    view_208: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_23, [1, 12, 512, 64]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_139: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None
    clone_11: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_209: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_11, [1, 512, -1]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_210: "f32[512, 768]" = torch.ops.aten.reshape.default(view_209, [512, 768]);  view_209 = None
    permute_140: "f32[768, 768]" = torch.ops.aten.permute.default(primals_155, [1, 0]);  primals_155 = None
    
    # No stacktrace found for following nodes
    mm_default_2: "f32[512, 768]" = torch.ops.aten.mm.default(view_210, permute_140)
    add_tensor_2: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_2, primals_156);  mm_default_2 = primals_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_211: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_2, [1, 512, 768]);  add_tensor_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    bernoulli_35: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute, 0.9)
    sub_93: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_35);  bernoulli_35 = None
    convert_element_type_47: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_93, torch.bool);  sub_93 = None
    where_59: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_47, full_default_1, view_211);  view_211 = None
    mul_105: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_59, 1.1111111111111112);  where_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_104: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_105, add_101);  mul_105 = add_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_46: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_104, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_94: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_104, mean_46);  add_104 = mean_46 = None
    pow_24: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_94, 2)
    mean_47: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_24, [-1], True);  pow_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_105: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_47, 1e-07);  mean_47 = None
    sqrt_35: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_105);  add_105 = None
    div_47: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_94, sqrt_35)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_106: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_71, div_47);  div_47 = None
    add_106: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_106, primals_72);  mul_106 = primals_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_212: "f32[512, 768]" = torch.ops.aten.reshape.default(add_106, [512, 768])
    permute_142: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_157, [1, 0]);  primals_157 = None
    addmm_34: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_158, view_212, permute_142);  primals_158 = None
    view_213: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_34, [1, 512, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_107: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_213, 0.5)
    mul_108: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_213, 0.7071067811865476);  view_213 = None
    erf_11: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_108);  mul_108 = None
    add_107: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_109: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_107, add_107);  mul_107 = add_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_214: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_109, [512, 3072]);  mul_109 = None
    permute_143: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_159, [1, 0]);  primals_159 = None
    
    # No stacktrace found for following nodes
    mm_default_1: "f32[512, 768]" = torch.ops.aten.mm.default(view_214, permute_143)
    add_tensor_1: "f32[512, 768]" = torch.ops.aten.add.Tensor(mm_default_1, primals_160);  mm_default_1 = primals_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_215: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(add_tensor_1, [1, 512, 768]);  add_tensor_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    bernoulli_36: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute, 0.9);  permute = None
    sub_96: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_36);  bernoulli_36 = None
    convert_element_type_48: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_96, torch.bool);  sub_96 = None
    where_60: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_48, full_default_1, view_215);  view_215 = None
    mul_110: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_60, 1.1111111111111112);  where_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_108: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_110, add_106);  mul_110 = add_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_48: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_108, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_97: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_108, mean_48);  add_108 = mean_48 = None
    pow_25: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_97, 2)
    mean_49: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_25, [-1], True);  pow_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    add_109: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_49, 1e-07);  mean_49 = None
    sqrt_36: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_109);  add_109 = None
    div_48: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_97, sqrt_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_111: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_73, div_48);  div_48 = None
    add_110: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_111, primals_74);  mul_111 = primals_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1411, code: logits = self.qa_outputs(sequence_output)
    view_216: "f32[512, 768]" = torch.ops.aten.reshape.default(add_110, [512, 768]);  add_110 = None
    permute_145: "f32[768, 2]" = torch.ops.aten.permute.default(primals_161, [1, 0]);  primals_161 = None
    
    # No stacktrace found for following nodes
    mm_default: "f32[512, 2]" = torch.ops.aten.mm.default(view_216, permute_145)
    add_tensor: "f32[512, 2]" = torch.ops.aten.add.Tensor(mm_default, primals_162);  mm_default = primals_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1411, code: logits = self.qa_outputs(sequence_output)
    view_217: "f32[1, 512, 2]" = torch.ops.aten.reshape.default(add_tensor, [1, 512, 2]);  add_tensor = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1412, code: start_logits, end_logits = logits.split(1, dim=-1)
    split_with_sizes = torch.ops.aten.split_with_sizes.default(view_217, [1, 1], 2);  view_217 = None
    getitem_36: "f32[1, 512, 1]" = split_with_sizes[0]
    getitem_37: "f32[1, 512, 1]" = split_with_sizes[1];  split_with_sizes = None
    
    # No stacktrace found for following nodes
    squeeze_1: "f32[1, 512]" = torch.ops.aten.squeeze.dim(getitem_36, -1);  getitem_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1413, code: start_logits = start_logits.squeeze(-1).contiguous()
    clone_12: "f32[1, 512]" = torch.ops.aten.clone.default(squeeze_1, memory_format = torch.contiguous_format);  squeeze_1 = None
    
    # No stacktrace found for following nodes
    squeeze_2: "f32[1, 512]" = torch.ops.aten.squeeze.dim(getitem_37, -1);  getitem_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1414, code: end_logits = end_logits.squeeze(-1).contiguous()
    clone_13: "f32[1, 512]" = torch.ops.aten.clone.default(squeeze_2, memory_format = torch.contiguous_format);  squeeze_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1425, code: start_positions = start_positions.clamp(0, ignored_index)
    clamp_min: "i64[1]" = torch.ops.aten.clamp_min.default(primals_165, 0);  primals_165 = None
    clamp_max: "i64[1]" = torch.ops.aten.clamp_max.default(clamp_min, 512);  clamp_min = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1426, code: end_positions = end_positions.clamp(0, ignored_index)
    clamp_min_1: "i64[1]" = torch.ops.aten.clamp_min.default(primals_166, 0);  primals_166 = None
    clamp_max_1: "i64[1]" = torch.ops.aten.clamp_max.default(clamp_min_1, 512);  clamp_min_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1429, code: start_loss = loss_fct(start_logits, start_positions)
    amax_12: "f32[1, 1]" = torch.ops.aten.amax.default(clone_12, [1], True)
    sub_99: "f32[1, 512]" = torch.ops.aten.sub.Tensor(clone_12, amax_12);  amax_12 = None
    exp_12: "f32[1, 512]" = torch.ops.aten.exp.default(sub_99)
    sum_13: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [1], True);  exp_12 = None
    log: "f32[1, 1]" = torch.ops.aten.log.default(sum_13);  sum_13 = None
    sub_100: "f32[1, 512]" = torch.ops.aten.sub.Tensor(sub_99, log);  sub_99 = log = None
    ne: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max, 512)
    full_default_86: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where_61: "i64[1]" = torch.ops.aten.where.self(ne, clamp_max, full_default_86)
    unsqueeze_52: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(where_61, 1);  where_61 = None
    gather: "f32[1, 1]" = torch.ops.aten.gather.default(sub_100, 1, unsqueeze_52);  unsqueeze_52 = None
    squeeze_3: "f32[1]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[1]" = torch.ops.aten.neg.default(squeeze_3);  squeeze_3 = None
    where_62: "f32[1]" = torch.ops.aten.where.self(ne, neg, full_default_1);  neg = None
    sum_14: "i64[]" = torch.ops.aten.sum.default(ne)
    convert_element_type_49: "f32[]" = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None
    sum_15: "f32[]" = torch.ops.aten.sum.default(where_62);  where_62 = None
    div_49: "f32[]" = torch.ops.aten.div.Tensor(sum_15, convert_element_type_49);  sum_15 = convert_element_type_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1430, code: end_loss = loss_fct(end_logits, end_positions)
    amax_13: "f32[1, 1]" = torch.ops.aten.amax.default(clone_13, [1], True)
    sub_101: "f32[1, 512]" = torch.ops.aten.sub.Tensor(clone_13, amax_13);  amax_13 = None
    exp_13: "f32[1, 512]" = torch.ops.aten.exp.default(sub_101)
    sum_16: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [1], True);  exp_13 = None
    log_1: "f32[1, 1]" = torch.ops.aten.log.default(sum_16);  sum_16 = None
    sub_102: "f32[1, 512]" = torch.ops.aten.sub.Tensor(sub_101, log_1);  sub_101 = log_1 = None
    ne_3: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max_1, 512)
    where_63: "i64[1]" = torch.ops.aten.where.self(ne_3, clamp_max_1, full_default_86)
    unsqueeze_53: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(where_63, 1);  where_63 = None
    gather_1: "f32[1, 1]" = torch.ops.aten.gather.default(sub_102, 1, unsqueeze_53);  unsqueeze_53 = None
    squeeze_4: "f32[1]" = torch.ops.aten.squeeze.dim(gather_1, 1);  gather_1 = None
    neg_1: "f32[1]" = torch.ops.aten.neg.default(squeeze_4);  squeeze_4 = None
    where_64: "f32[1]" = torch.ops.aten.where.self(ne_3, neg_1, full_default_1);  neg_1 = full_default_1 = None
    sum_17: "i64[]" = torch.ops.aten.sum.default(ne_3)
    convert_element_type_50: "f32[]" = torch.ops.prims.convert_element_type.default(sum_17, torch.float32);  sum_17 = None
    sum_18: "f32[]" = torch.ops.aten.sum.default(where_64);  where_64 = None
    div_50: "f32[]" = torch.ops.aten.div.Tensor(sum_18, convert_element_type_50);  sum_18 = convert_element_type_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1431, code: total_loss = (start_loss + end_loss) / 2
    add_111: "f32[]" = torch.ops.aten.add.Tensor(div_49, div_50);  div_49 = div_50 = None
    div_51: "f32[]" = torch.ops.aten.div.Tensor(add_111, 2);  add_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1430, code: end_loss = loss_fct(end_logits, end_positions)
    unsqueeze_54: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(clamp_max_1, 1);  clamp_max_1 = None
    ne_6: "b8[1, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_54, 512)
    where_65: "i64[1, 1]" = torch.ops.aten.where.self(ne_6, unsqueeze_54, full_default_86);  unsqueeze_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1429, code: start_loss = loss_fct(start_logits, start_positions)
    unsqueeze_55: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(clamp_max, 1);  clamp_max = None
    ne_8: "b8[1, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_55, 512)
    where_67: "i64[1, 1]" = torch.ops.aten.where.self(ne_8, unsqueeze_55, full_default_86);  unsqueeze_55 = full_default_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1411, code: logits = self.qa_outputs(sequence_output)
    permute_146: "f32[2, 768]" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    permute_150: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    permute_154: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_142, [1, 0]);  permute_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    permute_158: "f32[768, 768]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    permute_163: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_206, [0, 2, 1]);  view_206 = None
    permute_164: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_207, [0, 2, 1]);  view_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_44: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(where_57);  where_57 = None
    alias_45: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_44);  alias_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_165: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_203, [0, 2, 1]);  view_203 = None
    permute_166: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_204, [0, 2, 1]);  view_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_173: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    permute_175: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    permute_179: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    permute_183: "f32[768, 768]" = torch.ops.aten.permute.default(permute_128, [1, 0]);  permute_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    permute_188: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None
    permute_189: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_189, [0, 2, 1]);  view_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_49: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(where_52);  where_52 = None
    alias_50: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_49);  alias_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_190: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_185, [0, 2, 1]);  view_185 = None
    permute_191: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1]);  view_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_198: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    permute_200: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    permute_204: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    permute_208: "f32[768, 768]" = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    permute_213: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_170, [0, 2, 1]);  view_170 = None
    permute_214: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_171, [0, 2, 1]);  view_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_54: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(where_47);  where_47 = None
    alias_55: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_54);  alias_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_215: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_167, [0, 2, 1]);  view_167 = None
    permute_216: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1]);  view_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_223: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    permute_225: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    permute_229: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_106, [1, 0]);  permute_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    permute_233: "f32[768, 768]" = torch.ops.aten.permute.default(permute_104, [1, 0]);  permute_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    permute_238: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_152, [0, 2, 1]);  view_152 = None
    permute_239: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_153, [0, 2, 1]);  view_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_59: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(where_42);  where_42 = None
    alias_60: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_59);  alias_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_240: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_149, [0, 2, 1]);  view_149 = None
    permute_241: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_150, [0, 2, 1]);  view_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_248: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    permute_250: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_95, [1, 0]);  permute_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    permute_254: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_94, [1, 0]);  permute_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    permute_258: "f32[768, 768]" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    permute_263: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_134, [0, 2, 1]);  view_134 = None
    permute_264: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_135, [0, 2, 1]);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_64: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(where_37);  where_37 = None
    alias_65: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_64);  alias_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_265: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_131, [0, 2, 1]);  view_131 = None
    permute_266: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_132, [0, 2, 1]);  view_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_273: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    permute_275: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_83, [1, 0]);  permute_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    permute_279: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_82, [1, 0]);  permute_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    permute_283: "f32[768, 768]" = torch.ops.aten.permute.default(permute_80, [1, 0]);  permute_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    permute_288: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_116, [0, 2, 1]);  view_116 = None
    permute_289: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_117, [0, 2, 1]);  view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_69: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(where_32);  where_32 = None
    alias_70: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_69);  alias_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_290: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_113, [0, 2, 1]);  view_113 = None
    permute_291: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_114, [0, 2, 1]);  view_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_298: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_73, [1, 0]);  permute_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    permute_300: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    permute_304: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    permute_308: "f32[768, 768]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    permute_313: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_98, [0, 2, 1]);  view_98 = None
    permute_314: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_99, [0, 2, 1]);  view_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_74: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(where_27);  where_27 = None
    alias_75: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_74);  alias_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_315: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_95, [0, 2, 1]);  view_95 = None
    permute_316: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_96, [0, 2, 1]);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_323: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_61, [1, 0]);  permute_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    permute_325: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    permute_329: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    permute_333: "f32[768, 768]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    permute_338: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_80, [0, 2, 1]);  view_80 = None
    permute_339: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_81, [0, 2, 1]);  view_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_79: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(where_22);  where_22 = None
    alias_80: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_79);  alias_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_340: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_77, [0, 2, 1]);  view_77 = None
    permute_341: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_348: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    permute_350: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    permute_354: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    permute_358: "f32[768, 768]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    permute_363: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_62, [0, 2, 1]);  view_62 = None
    permute_364: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_63, [0, 2, 1]);  view_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_84: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(where_17);  where_17 = None
    alias_85: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_84);  alias_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_365: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_59, [0, 2, 1]);  view_59 = None
    permute_366: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_60, [0, 2, 1]);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_373: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    permute_375: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    permute_379: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    permute_383: "f32[768, 768]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    permute_388: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_44, [0, 2, 1]);  view_44 = None
    permute_389: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_45, [0, 2, 1]);  view_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_89: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(where_12);  where_12 = None
    alias_90: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_89);  alias_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_390: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_41, [0, 2, 1]);  view_41 = None
    permute_391: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_42, [0, 2, 1]);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_398: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    permute_400: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    permute_404: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    permute_408: "f32[768, 768]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    permute_413: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_26, [0, 2, 1]);  view_26 = None
    permute_414: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_27, [0, 2, 1]);  view_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_94: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(where_7);  where_7 = None
    alias_95: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_94);  alias_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_415: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_23, [0, 2, 1]);  view_23 = None
    permute_416: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_24, [0, 2, 1]);  view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_423: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    permute_425: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    permute_429: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    permute_433: "f32[768, 768]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    permute_438: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_8, [0, 2, 1]);  view_8 = None
    permute_439: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_99: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(where_2);  where_2 = None
    alias_100: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_99);  alias_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_440: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_5, [0, 2, 1]);  view_5 = None
    permute_441: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1]);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_448: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    return [div_51, clone_12, clone_13, primals_1, primals_5, primals_7, primals_11, primals_13, primals_17, primals_19, primals_23, primals_25, primals_29, primals_31, primals_35, primals_37, primals_41, primals_43, primals_47, primals_49, primals_53, primals_55, primals_59, primals_61, primals_65, primals_67, primals_71, primals_73, primals_164, slice_1, sub, sqrt, convert_element_type, view, convert_element_type_2, view_12, convert_element_type_3, sub_6, sqrt_2, view_14, addmm_1, view_16, convert_element_type_4, sub_9, sqrt_3, view_18, convert_element_type_6, view_30, convert_element_type_7, sub_14, sqrt_5, view_32, addmm_4, view_34, convert_element_type_8, sub_17, sqrt_6, view_36, convert_element_type_10, view_48, convert_element_type_11, sub_22, sqrt_8, view_50, addmm_7, view_52, convert_element_type_12, sub_25, sqrt_9, view_54, convert_element_type_14, view_66, convert_element_type_15, sub_30, sqrt_11, view_68, addmm_10, view_70, convert_element_type_16, sub_33, sqrt_12, view_72, convert_element_type_18, view_84, convert_element_type_19, sub_38, sqrt_14, view_86, addmm_13, view_88, convert_element_type_20, sub_41, sqrt_15, view_90, convert_element_type_22, view_102, convert_element_type_23, sub_46, sqrt_17, view_104, addmm_16, view_106, convert_element_type_24, sub_49, sqrt_18, view_108, convert_element_type_26, view_120, convert_element_type_27, sub_54, sqrt_20, view_122, addmm_19, view_124, convert_element_type_28, sub_57, sqrt_21, view_126, convert_element_type_30, view_138, convert_element_type_31, sub_62, sqrt_23, view_140, addmm_22, view_142, convert_element_type_32, sub_65, sqrt_24, view_144, convert_element_type_34, view_156, convert_element_type_35, sub_70, sqrt_26, view_158, addmm_25, view_160, convert_element_type_36, sub_73, sqrt_27, view_162, convert_element_type_38, view_174, convert_element_type_39, sub_78, sqrt_29, view_176, addmm_28, view_178, convert_element_type_40, sub_81, sqrt_30, view_180, convert_element_type_42, view_192, convert_element_type_43, sub_86, sqrt_32, view_194, addmm_31, view_196, convert_element_type_44, sub_89, sqrt_33, view_198, convert_element_type_46, view_210, convert_element_type_47, sub_94, sqrt_35, view_212, addmm_34, view_214, convert_element_type_48, sub_97, sqrt_36, view_216, sub_100, ne, sub_102, ne_3, ne_6, where_65, ne_8, where_67, permute_146, permute_150, permute_154, permute_158, permute_163, permute_164, alias_45, permute_165, permute_166, permute_173, permute_175, permute_179, permute_183, permute_188, permute_189, alias_50, permute_190, permute_191, permute_198, permute_200, permute_204, permute_208, permute_213, permute_214, alias_55, permute_215, permute_216, permute_223, permute_225, permute_229, permute_233, permute_238, permute_239, alias_60, permute_240, permute_241, permute_248, permute_250, permute_254, permute_258, permute_263, permute_264, alias_65, permute_265, permute_266, permute_273, permute_275, permute_279, permute_283, permute_288, permute_289, alias_70, permute_290, permute_291, permute_298, permute_300, permute_304, permute_308, permute_313, permute_314, alias_75, permute_315, permute_316, permute_323, permute_325, permute_329, permute_333, permute_338, permute_339, alias_80, permute_340, permute_341, permute_348, permute_350, permute_354, permute_358, permute_363, permute_364, alias_85, permute_365, permute_366, permute_373, permute_375, permute_379, permute_383, permute_388, permute_389, alias_90, permute_390, permute_391, permute_398, permute_400, permute_404, permute_408, permute_413, permute_414, alias_95, permute_415, permute_416, permute_423, permute_425, permute_429, permute_433, permute_438, permute_439, alias_100, permute_440, permute_441, permute_448]
    