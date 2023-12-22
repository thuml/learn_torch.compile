from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[768]"; primals_2: "f32[768]"; primals_3: "f32[768]"; primals_4: "f32[768]"; primals_5: "f32[768]"; primals_6: "f32[768]"; primals_7: "f32[768]"; primals_8: "f32[768]"; primals_9: "f32[768]"; primals_10: "f32[768]"; primals_11: "f32[768]"; primals_12: "f32[768]"; primals_13: "f32[768]"; primals_14: "f32[768]"; primals_15: "f32[768]"; primals_16: "f32[768]"; primals_17: "f32[768]"; primals_18: "f32[768]"; primals_19: "f32[768]"; primals_20: "f32[768]"; primals_21: "f32[768]"; primals_22: "f32[768]"; primals_23: "f32[768]"; primals_24: "f32[768]"; primals_25: "f32[768]"; primals_26: "f32[768]"; primals_27: "f32[768]"; primals_28: "f32[768]"; primals_29: "f32[768]"; primals_30: "f32[768]"; primals_31: "f32[768]"; primals_32: "f32[768]"; primals_33: "f32[768]"; primals_34: "f32[768]"; primals_35: "f32[768]"; primals_36: "f32[768]"; primals_37: "f32[768]"; primals_38: "f32[768]"; primals_39: "f32[768]"; primals_40: "f32[768]"; primals_41: "f32[768]"; primals_42: "f32[768]"; primals_43: "f32[768]"; primals_44: "f32[768]"; primals_45: "f32[768]"; primals_46: "f32[768]"; primals_47: "f32[768]"; primals_48: "f32[768]"; primals_49: "f32[768]"; primals_50: "f32[768]"; primals_51: "f32[768]"; primals_52: "f32[768]"; primals_53: "f32[768]"; primals_54: "f32[768]"; primals_55: "f32[768]"; primals_56: "f32[768]"; primals_57: "f32[768]"; primals_58: "f32[768]"; primals_59: "f32[768]"; primals_60: "f32[768]"; primals_61: "f32[768]"; primals_62: "f32[768]"; primals_63: "f32[768]"; primals_64: "f32[768]"; primals_65: "f32[768]"; primals_66: "f32[768]"; primals_67: "f32[768]"; primals_68: "f32[768]"; primals_69: "f32[768]"; primals_70: "f32[768]"; primals_71: "f32[768]"; primals_72: "f32[768]"; primals_73: "f32[768]"; primals_74: "f32[768]"; primals_75: "f32[50265, 768]"; primals_76: "f32[512, 768]"; primals_77: "f32[2304, 768]"; primals_78: "f32[768, 768]"; primals_79: "f32[768]"; primals_80: "f32[3072, 768]"; primals_81: "f32[3072]"; primals_82: "f32[768, 3072]"; primals_83: "f32[768]"; primals_84: "f32[2304, 768]"; primals_85: "f32[768, 768]"; primals_86: "f32[768]"; primals_87: "f32[3072, 768]"; primals_88: "f32[3072]"; primals_89: "f32[768, 3072]"; primals_90: "f32[768]"; primals_91: "f32[2304, 768]"; primals_92: "f32[768, 768]"; primals_93: "f32[768]"; primals_94: "f32[3072, 768]"; primals_95: "f32[3072]"; primals_96: "f32[768, 3072]"; primals_97: "f32[768]"; primals_98: "f32[2304, 768]"; primals_99: "f32[768, 768]"; primals_100: "f32[768]"; primals_101: "f32[3072, 768]"; primals_102: "f32[3072]"; primals_103: "f32[768, 3072]"; primals_104: "f32[768]"; primals_105: "f32[2304, 768]"; primals_106: "f32[768, 768]"; primals_107: "f32[768]"; primals_108: "f32[3072, 768]"; primals_109: "f32[3072]"; primals_110: "f32[768, 3072]"; primals_111: "f32[768]"; primals_112: "f32[2304, 768]"; primals_113: "f32[768, 768]"; primals_114: "f32[768]"; primals_115: "f32[3072, 768]"; primals_116: "f32[3072]"; primals_117: "f32[768, 3072]"; primals_118: "f32[768]"; primals_119: "f32[2304, 768]"; primals_120: "f32[768, 768]"; primals_121: "f32[768]"; primals_122: "f32[3072, 768]"; primals_123: "f32[3072]"; primals_124: "f32[768, 3072]"; primals_125: "f32[768]"; primals_126: "f32[2304, 768]"; primals_127: "f32[768, 768]"; primals_128: "f32[768]"; primals_129: "f32[3072, 768]"; primals_130: "f32[3072]"; primals_131: "f32[768, 3072]"; primals_132: "f32[768]"; primals_133: "f32[2304, 768]"; primals_134: "f32[768, 768]"; primals_135: "f32[768]"; primals_136: "f32[3072, 768]"; primals_137: "f32[3072]"; primals_138: "f32[768, 3072]"; primals_139: "f32[768]"; primals_140: "f32[2304, 768]"; primals_141: "f32[768, 768]"; primals_142: "f32[768]"; primals_143: "f32[3072, 768]"; primals_144: "f32[3072]"; primals_145: "f32[768, 3072]"; primals_146: "f32[768]"; primals_147: "f32[2304, 768]"; primals_148: "f32[768, 768]"; primals_149: "f32[768]"; primals_150: "f32[3072, 768]"; primals_151: "f32[3072]"; primals_152: "f32[768, 3072]"; primals_153: "f32[768]"; primals_154: "f32[2304, 768]"; primals_155: "f32[768, 768]"; primals_156: "f32[768]"; primals_157: "f32[3072, 768]"; primals_158: "f32[3072]"; primals_159: "f32[768, 3072]"; primals_160: "f32[768]"; primals_161: "f32[2, 768]"; primals_162: "f32[2]"; primals_163: "i64[1, 512]"; primals_164: "i64[1, 512]"; primals_165: "i64[1]"; primals_166: "i64[1]"; tangents_1: "f32[]"; tangents_2: "f32[1, 512]"; tangents_3: "f32[1, 512]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, tangents_1, tangents_2, tangents_3, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:972, code: attention_mask = torch.ones(input_shape, device=device)
    full: "f32[1, 512]" = torch.ops.aten.full.default([1, 512], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
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
    sub: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add, mean)
    pow_1: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub, 2)
    mean_1: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_1: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add, mean);  add = mean = None
    add_1: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_1, 1e-07);  mean_1 = None
    sqrt: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_1);  add_1 = None
    alias: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt)
    div: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_1, sqrt)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_1, div)
    add_2: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul, primals_2);  mul = primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:809, code: mask = mask.unsqueeze(2)
    unsqueeze: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(full, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:812, code: embeddings = embeddings * mask
    mul_1: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_2, unsqueeze);  add_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    permute: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty, [0, 1, 2]);  empty = None
    bernoulli: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute, 0.9);  permute = None
    sub_2: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli);  bernoulli = None
    convert_element_type: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_2, torch.bool);  sub_2 = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type, scalar_tensor, mul_1);  scalar_tensor = mul_1 = None
    mul_2: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where, 1.1111111111111112);  where = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:421, code: extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    unsqueeze_1: "f32[1, 1, 512]" = torch.ops.aten.unsqueeze.default(full, 1);  full = None
    unsqueeze_2: "f32[1, 1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 2);  unsqueeze_1 = None
    
    # No stacktrace found for following nodes
    squeeze: "f32[1, 1, 512]" = torch.ops.aten.squeeze.dim(unsqueeze_2, -2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:422, code: attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
    unsqueeze_3: "f32[1, 1, 512, 1]" = torch.ops.aten.unsqueeze.default(squeeze, -1);  squeeze = None
    mul_3: "f32[1, 1, 512, 512]" = torch.ops.aten.mul.Tensor(unsqueeze_2, unsqueeze_3);  unsqueeze_2 = unsqueeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_1: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_77, [1, 0]);  primals_77 = None
    view: "f32[512, 768]" = torch.ops.aten.view.default(mul_2, [512, 768])
    mm: "f32[512, 2304]" = torch.ops.aten.mm.default(view, permute_1)
    view_1: "f32[1, 512, 2304]" = torch.ops.aten.view.default(mm, [1, 512, 2304]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_2: "f32[1, 512, 12, 192]" = torch.ops.aten.view.default(view_1, [1, 512, 12, -1]);  view_1 = None
    
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
    slice_2: "f32[1, 1, 768]" = torch.ops.aten.slice.Tensor(unsqueeze_5, 2, 0, 9223372036854775807);  unsqueeze_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_3: "f32[1, 1, 12, 64]" = torch.ops.aten.view.default(slice_2, [1, 1, 12, -1]);  slice_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_3: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    add_3: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem, permute_3);  getitem = permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    unsqueeze_6: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_4, 0);  primals_4 = None
    unsqueeze_7: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 1);  unsqueeze_6 = None
    slice_3: "f32[1, 1, 768]" = torch.ops.aten.slice.Tensor(unsqueeze_7, 2, 0, 9223372036854775807);  unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_4: "f32[1, 1, 12, 64]" = torch.ops.aten.view.default(slice_3, [1, 1, 12, -1]);  slice_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_4: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    add_4: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_2, permute_4);  getitem_2 = permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant0 = self._tensor_constant0
    lift_fresh_copy: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
    mul_4: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy, 1);  lift_fresh_copy = None
    sqrt_1: "f32[]" = torch.ops.aten.sqrt.default(mul_4);  mul_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_1: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(add_3, sqrt_1);  add_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_5: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(getitem_1, [0, 1, 3, 2]);  getitem_1 = None
    expand: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(div_1, [1, 12, 512, 64]);  div_1 = None
    view_5: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand, [12, 512, 64]);  expand = None
    expand_1: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_5, [1, 12, 64, 512]);  permute_5 = None
    view_6: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_1, [12, 64, 512]);  expand_1 = None
    bmm: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_5, view_6)
    view_7: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm, [1, 12, 512, 512]);  bmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    convert_element_type_1: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool)
    bitwise_not: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_not.default(convert_element_type_1);  convert_element_type_1 = None
    _tensor_constant1 = self._tensor_constant1
    lift_fresh_copy_1: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant1);  _tensor_constant1 = None
    where_1: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(bitwise_not, lift_fresh_copy_1, view_7);  lift_fresh_copy_1 = view_7 = None
    amax: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where_1, [-1], True)
    sub_3: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where_1, amax);  where_1 = amax = None
    exp: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
    sum_1: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div_2: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_2: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(bitwise_not, scalar_tensor_1, div_2);  bitwise_not = scalar_tensor_1 = div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_1: "f32[1, 12, 512, 512]" = torch.ops.aten.empty.memory_format([1, 12, 512, 512], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    permute_6: "f32[1, 12, 512, 512]" = torch.ops.aten.permute.default(empty_1, [0, 1, 2, 3]);  empty_1 = None
    bernoulli_1: "f32[1, 12, 512, 512]" = torch.ops.aten.bernoulli.p(permute_6, 0.9);  permute_6 = None
    sub_4: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_1);  bernoulli_1 = None
    convert_element_type_2: "b8[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_4, torch.bool);  sub_4 = None
    scalar_tensor_2: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_3: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_2, scalar_tensor_2, where_2);  scalar_tensor_2 = None
    mul_5: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_3, 1.1111111111111112);  where_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_2: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(mul_5, [1, 12, 512, 512]);  mul_5 = None
    view_8: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_2, [12, 512, 512]);  expand_2 = None
    expand_3: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(add_4, [1, 12, 512, 64]);  add_4 = None
    view_9: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_3, [12, 512, 64]);  expand_3 = None
    bmm_1: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_8, view_9)
    view_10: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_1, [1, 12, 512, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_7: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
    clone: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_11: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone, [1, 512, -1]);  clone = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_12: "f32[512, 768]" = torch.ops.aten.view.default(view_11, [512, 768]);  view_11 = None
    permute_8: "f32[768, 768]" = torch.ops.aten.permute.default(primals_78, [1, 0]);  primals_78 = None
    addmm: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_79, view_12, permute_8);  primals_79 = None
    view_13: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm, [1, 512, 768]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_2: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    permute_9: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_2, [0, 1, 2]);  empty_2 = None
    bernoulli_2: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_9, 0.9);  permute_9 = None
    sub_5: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_2);  bernoulli_2 = None
    convert_element_type_3: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_5, torch.bool);  sub_5 = None
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_4: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_3, scalar_tensor_3, view_13);  scalar_tensor_3 = view_13 = None
    mul_6: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_4, 1.1111111111111112);  where_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_5: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_6, mul_2);  mul_6 = mul_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_2: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_5, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_6: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_5, mean_2)
    pow_2: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_6, 2)
    mean_3: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_7: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_5, mean_2);  add_5 = mean_2 = None
    add_6: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_3, 1e-07);  mean_3 = None
    sqrt_2: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_6);  add_6 = None
    alias_2: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_2)
    div_3: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_7, sqrt_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_7: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_5, div_3)
    add_7: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_7, primals_6);  mul_7 = primals_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_14: "f32[512, 768]" = torch.ops.aten.view.default(add_7, [512, 768])
    permute_10: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_80, [1, 0]);  primals_80 = None
    addmm_1: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_81, view_14, permute_10);  primals_81 = None
    view_15: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_1, [1, 512, 3072]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_8: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_15, 0.5)
    mul_9: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_15, 0.7071067811865476)
    erf: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_9);  mul_9 = None
    add_8: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_10: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_8, add_8);  mul_8 = add_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_16: "f32[512, 3072]" = torch.ops.aten.view.default(mul_10, [512, 3072]);  mul_10 = None
    permute_11: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_82, [1, 0]);  primals_82 = None
    addmm_2: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_83, view_16, permute_11);  primals_83 = None
    view_17: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_2, [1, 512, 768]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_3: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    permute_12: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_3, [0, 1, 2]);  empty_3 = None
    bernoulli_3: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_12, 0.9);  permute_12 = None
    sub_8: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_3);  bernoulli_3 = None
    convert_element_type_4: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_8, torch.bool);  sub_8 = None
    scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_5: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_4, scalar_tensor_4, view_17);  scalar_tensor_4 = view_17 = None
    mul_11: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_5, 1.1111111111111112);  where_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_9: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_11, add_7);  mul_11 = add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_4: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_9, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_9: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_9, mean_4)
    pow_3: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_9, 2)
    mean_5: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_3, [-1], True);  pow_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_10: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_9, mean_4);  add_9 = mean_4 = None
    add_10: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_5, 1e-07);  mean_5 = None
    sqrt_3: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_10);  add_10 = None
    alias_3: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_3)
    div_4: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_10, sqrt_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_12: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_7, div_4)
    add_11: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_12, primals_8);  mul_12 = primals_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_13: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_84, [1, 0]);  primals_84 = None
    view_18: "f32[512, 768]" = torch.ops.aten.view.default(add_11, [512, 768])
    mm_1: "f32[512, 2304]" = torch.ops.aten.mm.default(view_18, permute_13)
    view_19: "f32[1, 512, 2304]" = torch.ops.aten.view.default(mm_1, [1, 512, 2304]);  mm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_20: "f32[1, 512, 12, 192]" = torch.ops.aten.view.default(view_19, [1, 512, 12, -1]);  view_19 = None
    
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
    slice_4: "f32[1, 1, 768]" = torch.ops.aten.slice.Tensor(unsqueeze_9, 2, 0, 9223372036854775807);  unsqueeze_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_21: "f32[1, 1, 12, 64]" = torch.ops.aten.view.default(slice_4, [1, 1, 12, -1]);  slice_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_15: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_21, [0, 2, 1, 3]);  view_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    add_12: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_3, permute_15);  getitem_3 = permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    unsqueeze_10: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_10, 0);  primals_10 = None
    unsqueeze_11: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 1);  unsqueeze_10 = None
    slice_5: "f32[1, 1, 768]" = torch.ops.aten.slice.Tensor(unsqueeze_11, 2, 0, 9223372036854775807);  unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_22: "f32[1, 1, 12, 64]" = torch.ops.aten.view.default(slice_5, [1, 1, 12, -1]);  slice_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_16: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_22, [0, 2, 1, 3]);  view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    add_13: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_5, permute_16);  getitem_5 = permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant2 = self._tensor_constant2
    lift_fresh_copy_2: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant2);  _tensor_constant2 = None
    mul_13: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_2, 1);  lift_fresh_copy_2 = None
    sqrt_4: "f32[]" = torch.ops.aten.sqrt.default(mul_13);  mul_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_5: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(add_12, sqrt_4);  add_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_17: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(getitem_4, [0, 1, 3, 2]);  getitem_4 = None
    expand_4: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(div_5, [1, 12, 512, 64]);  div_5 = None
    view_23: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_4, [12, 512, 64]);  expand_4 = None
    expand_5: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_17, [1, 12, 64, 512]);  permute_17 = None
    view_24: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_5, [12, 64, 512]);  expand_5 = None
    bmm_2: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_23, view_24)
    view_25: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_2, [1, 12, 512, 512]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    convert_element_type_5: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool)
    bitwise_not_1: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_not.default(convert_element_type_5);  convert_element_type_5 = None
    _tensor_constant3 = self._tensor_constant3
    lift_fresh_copy_3: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant3);  _tensor_constant3 = None
    where_6: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(bitwise_not_1, lift_fresh_copy_3, view_25);  lift_fresh_copy_3 = view_25 = None
    amax_1: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where_6, [-1], True)
    sub_11: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where_6, amax_1);  where_6 = amax_1 = None
    exp_1: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
    sum_2: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_6: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_7: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(bitwise_not_1, scalar_tensor_5, div_6);  bitwise_not_1 = scalar_tensor_5 = div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_4: "f32[1, 12, 512, 512]" = torch.ops.aten.empty.memory_format([1, 12, 512, 512], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    permute_18: "f32[1, 12, 512, 512]" = torch.ops.aten.permute.default(empty_4, [0, 1, 2, 3]);  empty_4 = None
    bernoulli_4: "f32[1, 12, 512, 512]" = torch.ops.aten.bernoulli.p(permute_18, 0.9);  permute_18 = None
    sub_12: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_4);  bernoulli_4 = None
    convert_element_type_6: "b8[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_12, torch.bool);  sub_12 = None
    scalar_tensor_6: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_8: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_6, scalar_tensor_6, where_7);  scalar_tensor_6 = None
    mul_14: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_8, 1.1111111111111112);  where_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_6: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(mul_14, [1, 12, 512, 512]);  mul_14 = None
    view_26: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_6, [12, 512, 512]);  expand_6 = None
    expand_7: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(add_13, [1, 12, 512, 64]);  add_13 = None
    view_27: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_7, [12, 512, 64]);  expand_7 = None
    bmm_3: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_26, view_27)
    view_28: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_3, [1, 12, 512, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_19: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
    clone_1: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_29: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_1, [1, 512, -1]);  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_30: "f32[512, 768]" = torch.ops.aten.view.default(view_29, [512, 768]);  view_29 = None
    permute_20: "f32[768, 768]" = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
    addmm_3: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_86, view_30, permute_20);  primals_86 = None
    view_31: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_3, [1, 512, 768]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_5: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    permute_21: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_5, [0, 1, 2]);  empty_5 = None
    bernoulli_5: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_21, 0.9);  permute_21 = None
    sub_13: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_5);  bernoulli_5 = None
    convert_element_type_7: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_13, torch.bool);  sub_13 = None
    scalar_tensor_7: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_9: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_7, scalar_tensor_7, view_31);  scalar_tensor_7 = view_31 = None
    mul_15: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_9, 1.1111111111111112);  where_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_14: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_15, add_11);  mul_15 = add_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_6: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_14, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_14: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_14, mean_6)
    pow_4: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_14, 2)
    mean_7: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_4, [-1], True);  pow_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_15: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_14, mean_6);  add_14 = mean_6 = None
    add_15: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_7, 1e-07);  mean_7 = None
    sqrt_5: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_15);  add_15 = None
    alias_5: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_5)
    div_7: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_15, sqrt_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_16: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_11, div_7)
    add_16: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_16, primals_12);  mul_16 = primals_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_32: "f32[512, 768]" = torch.ops.aten.view.default(add_16, [512, 768])
    permute_22: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
    addmm_4: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_88, view_32, permute_22);  primals_88 = None
    view_33: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_4, [1, 512, 3072]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_17: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_33, 0.5)
    mul_18: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_33, 0.7071067811865476)
    erf_1: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_18);  mul_18 = None
    add_17: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_19: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_17, add_17);  mul_17 = add_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_34: "f32[512, 3072]" = torch.ops.aten.view.default(mul_19, [512, 3072]);  mul_19 = None
    permute_23: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_89, [1, 0]);  primals_89 = None
    addmm_5: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_90, view_34, permute_23);  primals_90 = None
    view_35: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_5, [1, 512, 768]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_6: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    permute_24: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_6, [0, 1, 2]);  empty_6 = None
    bernoulli_6: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_24, 0.9);  permute_24 = None
    sub_16: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_6);  bernoulli_6 = None
    convert_element_type_8: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_16, torch.bool);  sub_16 = None
    scalar_tensor_8: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_10: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_8, scalar_tensor_8, view_35);  scalar_tensor_8 = view_35 = None
    mul_20: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_10, 1.1111111111111112);  where_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_18: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_20, add_16);  mul_20 = add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_8: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_18, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_17: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_18, mean_8)
    pow_5: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_17, 2)
    mean_9: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_5, [-1], True);  pow_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_18: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_18, mean_8);  add_18 = mean_8 = None
    add_19: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_9, 1e-07);  mean_9 = None
    sqrt_6: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_19);  add_19 = None
    alias_6: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_6)
    div_8: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_18, sqrt_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_21: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_13, div_8)
    add_20: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_21, primals_14);  mul_21 = primals_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_25: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_91, [1, 0]);  primals_91 = None
    view_36: "f32[512, 768]" = torch.ops.aten.view.default(add_20, [512, 768])
    mm_2: "f32[512, 2304]" = torch.ops.aten.mm.default(view_36, permute_25)
    view_37: "f32[1, 512, 2304]" = torch.ops.aten.view.default(mm_2, [1, 512, 2304]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_38: "f32[1, 512, 12, 192]" = torch.ops.aten.view.default(view_37, [1, 512, 12, -1]);  view_37 = None
    
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
    slice_6: "f32[1, 1, 768]" = torch.ops.aten.slice.Tensor(unsqueeze_13, 2, 0, 9223372036854775807);  unsqueeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_39: "f32[1, 1, 12, 64]" = torch.ops.aten.view.default(slice_6, [1, 1, 12, -1]);  slice_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_27: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_39, [0, 2, 1, 3]);  view_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    add_21: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_6, permute_27);  getitem_6 = permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    unsqueeze_14: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_16, 0);  primals_16 = None
    unsqueeze_15: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, 1);  unsqueeze_14 = None
    slice_7: "f32[1, 1, 768]" = torch.ops.aten.slice.Tensor(unsqueeze_15, 2, 0, 9223372036854775807);  unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_40: "f32[1, 1, 12, 64]" = torch.ops.aten.view.default(slice_7, [1, 1, 12, -1]);  slice_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_28: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    add_22: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_8, permute_28);  getitem_8 = permute_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant4 = self._tensor_constant4
    lift_fresh_copy_4: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant4);  _tensor_constant4 = None
    mul_22: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_4, 1);  lift_fresh_copy_4 = None
    sqrt_7: "f32[]" = torch.ops.aten.sqrt.default(mul_22);  mul_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_9: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(add_21, sqrt_7);  add_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_29: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(getitem_7, [0, 1, 3, 2]);  getitem_7 = None
    expand_8: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(div_9, [1, 12, 512, 64]);  div_9 = None
    view_41: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_8, [12, 512, 64]);  expand_8 = None
    expand_9: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_29, [1, 12, 64, 512]);  permute_29 = None
    view_42: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_9, [12, 64, 512]);  expand_9 = None
    bmm_4: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_41, view_42)
    view_43: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_4, [1, 12, 512, 512]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    convert_element_type_9: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool)
    bitwise_not_2: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_not.default(convert_element_type_9);  convert_element_type_9 = None
    _tensor_constant5 = self._tensor_constant5
    lift_fresh_copy_5: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant5);  _tensor_constant5 = None
    where_11: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(bitwise_not_2, lift_fresh_copy_5, view_43);  lift_fresh_copy_5 = view_43 = None
    amax_2: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where_11, [-1], True)
    sub_19: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where_11, amax_2);  where_11 = amax_2 = None
    exp_2: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
    sum_3: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_10: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    scalar_tensor_9: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_12: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(bitwise_not_2, scalar_tensor_9, div_10);  bitwise_not_2 = scalar_tensor_9 = div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_7: "f32[1, 12, 512, 512]" = torch.ops.aten.empty.memory_format([1, 12, 512, 512], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    permute_30: "f32[1, 12, 512, 512]" = torch.ops.aten.permute.default(empty_7, [0, 1, 2, 3]);  empty_7 = None
    bernoulli_7: "f32[1, 12, 512, 512]" = torch.ops.aten.bernoulli.p(permute_30, 0.9);  permute_30 = None
    sub_20: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_7);  bernoulli_7 = None
    convert_element_type_10: "b8[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_20, torch.bool);  sub_20 = None
    scalar_tensor_10: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_13: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_10, scalar_tensor_10, where_12);  scalar_tensor_10 = None
    mul_23: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_13, 1.1111111111111112);  where_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_10: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(mul_23, [1, 12, 512, 512]);  mul_23 = None
    view_44: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_10, [12, 512, 512]);  expand_10 = None
    expand_11: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(add_22, [1, 12, 512, 64]);  add_22 = None
    view_45: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_11, [12, 512, 64]);  expand_11 = None
    bmm_5: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_44, view_45)
    view_46: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_5, [1, 12, 512, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_31: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None
    clone_2: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_31, memory_format = torch.contiguous_format);  permute_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_47: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_2, [1, 512, -1]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_48: "f32[512, 768]" = torch.ops.aten.view.default(view_47, [512, 768]);  view_47 = None
    permute_32: "f32[768, 768]" = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
    addmm_6: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_93, view_48, permute_32);  primals_93 = None
    view_49: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_6, [1, 512, 768]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_8: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    permute_33: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_8, [0, 1, 2]);  empty_8 = None
    bernoulli_8: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_33, 0.9);  permute_33 = None
    sub_21: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_8);  bernoulli_8 = None
    convert_element_type_11: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_21, torch.bool);  sub_21 = None
    scalar_tensor_11: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_14: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_11, scalar_tensor_11, view_49);  scalar_tensor_11 = view_49 = None
    mul_24: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_14, 1.1111111111111112);  where_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_23: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_24, add_20);  mul_24 = add_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_10: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_23, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_22: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_23, mean_10)
    pow_6: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_22, 2)
    mean_11: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_6, [-1], True);  pow_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_23: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_23, mean_10);  add_23 = mean_10 = None
    add_24: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_11, 1e-07);  mean_11 = None
    sqrt_8: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_24);  add_24 = None
    alias_8: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_8)
    div_11: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_23, sqrt_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_25: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_17, div_11)
    add_25: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_25, primals_18);  mul_25 = primals_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_50: "f32[512, 768]" = torch.ops.aten.view.default(add_25, [512, 768])
    permute_34: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_94, [1, 0]);  primals_94 = None
    addmm_7: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_95, view_50, permute_34);  primals_95 = None
    view_51: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_7, [1, 512, 3072]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_26: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_51, 0.5)
    mul_27: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_51, 0.7071067811865476)
    erf_2: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_27);  mul_27 = None
    add_26: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_28: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_26, add_26);  mul_26 = add_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_52: "f32[512, 3072]" = torch.ops.aten.view.default(mul_28, [512, 3072]);  mul_28 = None
    permute_35: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_96, [1, 0]);  primals_96 = None
    addmm_8: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_97, view_52, permute_35);  primals_97 = None
    view_53: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_8, [1, 512, 768]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_9: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    permute_36: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_9, [0, 1, 2]);  empty_9 = None
    bernoulli_9: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_36, 0.9);  permute_36 = None
    sub_24: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_9);  bernoulli_9 = None
    convert_element_type_12: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_24, torch.bool);  sub_24 = None
    scalar_tensor_12: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_15: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_12, scalar_tensor_12, view_53);  scalar_tensor_12 = view_53 = None
    mul_29: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_15, 1.1111111111111112);  where_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_27: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_29, add_25);  mul_29 = add_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_12: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_27, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_25: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_27, mean_12)
    pow_7: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_25, 2)
    mean_13: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_7, [-1], True);  pow_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_26: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_27, mean_12);  add_27 = mean_12 = None
    add_28: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_13, 1e-07);  mean_13 = None
    sqrt_9: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_28);  add_28 = None
    alias_9: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_9)
    div_12: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_26, sqrt_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_30: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_19, div_12)
    add_29: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_30, primals_20);  mul_30 = primals_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_37: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_98, [1, 0]);  primals_98 = None
    view_54: "f32[512, 768]" = torch.ops.aten.view.default(add_29, [512, 768])
    mm_3: "f32[512, 2304]" = torch.ops.aten.mm.default(view_54, permute_37)
    view_55: "f32[1, 512, 2304]" = torch.ops.aten.view.default(mm_3, [1, 512, 2304]);  mm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_56: "f32[1, 512, 12, 192]" = torch.ops.aten.view.default(view_55, [1, 512, 12, -1]);  view_55 = None
    
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
    slice_8: "f32[1, 1, 768]" = torch.ops.aten.slice.Tensor(unsqueeze_17, 2, 0, 9223372036854775807);  unsqueeze_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_57: "f32[1, 1, 12, 64]" = torch.ops.aten.view.default(slice_8, [1, 1, 12, -1]);  slice_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_39: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_57, [0, 2, 1, 3]);  view_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    add_30: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_9, permute_39);  getitem_9 = permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    unsqueeze_18: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_22, 0);  primals_22 = None
    unsqueeze_19: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, 1);  unsqueeze_18 = None
    slice_9: "f32[1, 1, 768]" = torch.ops.aten.slice.Tensor(unsqueeze_19, 2, 0, 9223372036854775807);  unsqueeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_58: "f32[1, 1, 12, 64]" = torch.ops.aten.view.default(slice_9, [1, 1, 12, -1]);  slice_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_40: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    add_31: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_11, permute_40);  getitem_11 = permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant6 = self._tensor_constant6
    lift_fresh_copy_6: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant6);  _tensor_constant6 = None
    mul_31: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_6, 1);  lift_fresh_copy_6 = None
    sqrt_10: "f32[]" = torch.ops.aten.sqrt.default(mul_31);  mul_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_13: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(add_30, sqrt_10);  add_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_41: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(getitem_10, [0, 1, 3, 2]);  getitem_10 = None
    expand_12: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(div_13, [1, 12, 512, 64]);  div_13 = None
    view_59: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_12, [12, 512, 64]);  expand_12 = None
    expand_13: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_41, [1, 12, 64, 512]);  permute_41 = None
    view_60: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_13, [12, 64, 512]);  expand_13 = None
    bmm_6: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_59, view_60)
    view_61: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_6, [1, 12, 512, 512]);  bmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    convert_element_type_13: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool)
    bitwise_not_3: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_not.default(convert_element_type_13);  convert_element_type_13 = None
    _tensor_constant7 = self._tensor_constant7
    lift_fresh_copy_7: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant7);  _tensor_constant7 = None
    where_16: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(bitwise_not_3, lift_fresh_copy_7, view_61);  lift_fresh_copy_7 = view_61 = None
    amax_3: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where_16, [-1], True)
    sub_27: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where_16, amax_3);  where_16 = amax_3 = None
    exp_3: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_27);  sub_27 = None
    sum_4: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_14: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    scalar_tensor_13: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_17: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(bitwise_not_3, scalar_tensor_13, div_14);  bitwise_not_3 = scalar_tensor_13 = div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_10: "f32[1, 12, 512, 512]" = torch.ops.aten.empty.memory_format([1, 12, 512, 512], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    permute_42: "f32[1, 12, 512, 512]" = torch.ops.aten.permute.default(empty_10, [0, 1, 2, 3]);  empty_10 = None
    bernoulli_10: "f32[1, 12, 512, 512]" = torch.ops.aten.bernoulli.p(permute_42, 0.9);  permute_42 = None
    sub_28: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_10);  bernoulli_10 = None
    convert_element_type_14: "b8[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_28, torch.bool);  sub_28 = None
    scalar_tensor_14: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_18: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_14, scalar_tensor_14, where_17);  scalar_tensor_14 = None
    mul_32: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_18, 1.1111111111111112);  where_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_14: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(mul_32, [1, 12, 512, 512]);  mul_32 = None
    view_62: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_14, [12, 512, 512]);  expand_14 = None
    expand_15: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(add_31, [1, 12, 512, 64]);  add_31 = None
    view_63: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_15, [12, 512, 64]);  expand_15 = None
    bmm_7: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_62, view_63)
    view_64: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_7, [1, 12, 512, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_43: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_64, [0, 2, 1, 3]);  view_64 = None
    clone_3: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_43, memory_format = torch.contiguous_format);  permute_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_65: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_3, [1, 512, -1]);  clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_66: "f32[512, 768]" = torch.ops.aten.view.default(view_65, [512, 768]);  view_65 = None
    permute_44: "f32[768, 768]" = torch.ops.aten.permute.default(primals_99, [1, 0]);  primals_99 = None
    addmm_9: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_100, view_66, permute_44);  primals_100 = None
    view_67: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_9, [1, 512, 768]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_11: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    permute_45: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_11, [0, 1, 2]);  empty_11 = None
    bernoulli_11: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_45, 0.9);  permute_45 = None
    sub_29: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_11);  bernoulli_11 = None
    convert_element_type_15: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_29, torch.bool);  sub_29 = None
    scalar_tensor_15: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_19: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_15, scalar_tensor_15, view_67);  scalar_tensor_15 = view_67 = None
    mul_33: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_19, 1.1111111111111112);  where_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_32: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_33, add_29);  mul_33 = add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_14: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_32, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_30: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_32, mean_14)
    pow_8: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_30, 2)
    mean_15: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_8, [-1], True);  pow_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_31: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_32, mean_14);  add_32 = mean_14 = None
    add_33: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_15, 1e-07);  mean_15 = None
    sqrt_11: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_33);  add_33 = None
    alias_11: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_11)
    div_15: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_31, sqrt_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_34: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_23, div_15)
    add_34: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_34, primals_24);  mul_34 = primals_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_68: "f32[512, 768]" = torch.ops.aten.view.default(add_34, [512, 768])
    permute_46: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_101, [1, 0]);  primals_101 = None
    addmm_10: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_102, view_68, permute_46);  primals_102 = None
    view_69: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_10, [1, 512, 3072]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_35: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_69, 0.5)
    mul_36: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_69, 0.7071067811865476)
    erf_3: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_36);  mul_36 = None
    add_35: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_37: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_35, add_35);  mul_35 = add_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_70: "f32[512, 3072]" = torch.ops.aten.view.default(mul_37, [512, 3072]);  mul_37 = None
    permute_47: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_103, [1, 0]);  primals_103 = None
    addmm_11: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_104, view_70, permute_47);  primals_104 = None
    view_71: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_11, [1, 512, 768]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_12: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    permute_48: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_12, [0, 1, 2]);  empty_12 = None
    bernoulli_12: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_48, 0.9);  permute_48 = None
    sub_32: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_12);  bernoulli_12 = None
    convert_element_type_16: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_32, torch.bool);  sub_32 = None
    scalar_tensor_16: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_20: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_16, scalar_tensor_16, view_71);  scalar_tensor_16 = view_71 = None
    mul_38: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_20, 1.1111111111111112);  where_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_36: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_38, add_34);  mul_38 = add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_16: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_36, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_33: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_36, mean_16)
    pow_9: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_33, 2)
    mean_17: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_9, [-1], True);  pow_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_34: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_36, mean_16);  add_36 = mean_16 = None
    add_37: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_17, 1e-07);  mean_17 = None
    sqrt_12: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_37);  add_37 = None
    alias_12: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_12)
    div_16: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_34, sqrt_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_39: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_25, div_16)
    add_38: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_39, primals_26);  mul_39 = primals_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_49: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
    view_72: "f32[512, 768]" = torch.ops.aten.view.default(add_38, [512, 768])
    mm_4: "f32[512, 2304]" = torch.ops.aten.mm.default(view_72, permute_49)
    view_73: "f32[1, 512, 2304]" = torch.ops.aten.view.default(mm_4, [1, 512, 2304]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_74: "f32[1, 512, 12, 192]" = torch.ops.aten.view.default(view_73, [1, 512, 12, -1]);  view_73 = None
    
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
    slice_10: "f32[1, 1, 768]" = torch.ops.aten.slice.Tensor(unsqueeze_21, 2, 0, 9223372036854775807);  unsqueeze_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_75: "f32[1, 1, 12, 64]" = torch.ops.aten.view.default(slice_10, [1, 1, 12, -1]);  slice_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_51: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    add_39: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_12, permute_51);  getitem_12 = permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    unsqueeze_22: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_28, 0);  primals_28 = None
    unsqueeze_23: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, 1);  unsqueeze_22 = None
    slice_11: "f32[1, 1, 768]" = torch.ops.aten.slice.Tensor(unsqueeze_23, 2, 0, 9223372036854775807);  unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_76: "f32[1, 1, 12, 64]" = torch.ops.aten.view.default(slice_11, [1, 1, 12, -1]);  slice_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_52: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    add_40: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_14, permute_52);  getitem_14 = permute_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant8 = self._tensor_constant8
    lift_fresh_copy_8: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant8);  _tensor_constant8 = None
    mul_40: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_8, 1);  lift_fresh_copy_8 = None
    sqrt_13: "f32[]" = torch.ops.aten.sqrt.default(mul_40);  mul_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_17: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(add_39, sqrt_13);  add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_53: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(getitem_13, [0, 1, 3, 2]);  getitem_13 = None
    expand_16: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(div_17, [1, 12, 512, 64]);  div_17 = None
    view_77: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_16, [12, 512, 64]);  expand_16 = None
    expand_17: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_53, [1, 12, 64, 512]);  permute_53 = None
    view_78: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_17, [12, 64, 512]);  expand_17 = None
    bmm_8: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_77, view_78)
    view_79: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_8, [1, 12, 512, 512]);  bmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    convert_element_type_17: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool)
    bitwise_not_4: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_not.default(convert_element_type_17);  convert_element_type_17 = None
    _tensor_constant9 = self._tensor_constant9
    lift_fresh_copy_9: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant9);  _tensor_constant9 = None
    where_21: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(bitwise_not_4, lift_fresh_copy_9, view_79);  lift_fresh_copy_9 = view_79 = None
    amax_4: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where_21, [-1], True)
    sub_35: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where_21, amax_4);  where_21 = amax_4 = None
    exp_4: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_35);  sub_35 = None
    sum_5: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_18: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    scalar_tensor_17: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_22: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(bitwise_not_4, scalar_tensor_17, div_18);  bitwise_not_4 = scalar_tensor_17 = div_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_13: "f32[1, 12, 512, 512]" = torch.ops.aten.empty.memory_format([1, 12, 512, 512], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    permute_54: "f32[1, 12, 512, 512]" = torch.ops.aten.permute.default(empty_13, [0, 1, 2, 3]);  empty_13 = None
    bernoulli_13: "f32[1, 12, 512, 512]" = torch.ops.aten.bernoulli.p(permute_54, 0.9);  permute_54 = None
    sub_36: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_13);  bernoulli_13 = None
    convert_element_type_18: "b8[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_36, torch.bool);  sub_36 = None
    scalar_tensor_18: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_23: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_18, scalar_tensor_18, where_22);  scalar_tensor_18 = None
    mul_41: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_23, 1.1111111111111112);  where_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_18: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(mul_41, [1, 12, 512, 512]);  mul_41 = None
    view_80: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_18, [12, 512, 512]);  expand_18 = None
    expand_19: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(add_40, [1, 12, 512, 64]);  add_40 = None
    view_81: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_19, [12, 512, 64]);  expand_19 = None
    bmm_9: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_80, view_81)
    view_82: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_9, [1, 12, 512, 64]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_55: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
    clone_4: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_55, memory_format = torch.contiguous_format);  permute_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_83: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_4, [1, 512, -1]);  clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_84: "f32[512, 768]" = torch.ops.aten.view.default(view_83, [512, 768]);  view_83 = None
    permute_56: "f32[768, 768]" = torch.ops.aten.permute.default(primals_106, [1, 0]);  primals_106 = None
    addmm_12: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_107, view_84, permute_56);  primals_107 = None
    view_85: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_12, [1, 512, 768]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_14: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    permute_57: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_14, [0, 1, 2]);  empty_14 = None
    bernoulli_14: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_57, 0.9);  permute_57 = None
    sub_37: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_14);  bernoulli_14 = None
    convert_element_type_19: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_37, torch.bool);  sub_37 = None
    scalar_tensor_19: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_24: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_19, scalar_tensor_19, view_85);  scalar_tensor_19 = view_85 = None
    mul_42: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_24, 1.1111111111111112);  where_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_41: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_42, add_38);  mul_42 = add_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_18: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_41, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_38: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_41, mean_18)
    pow_10: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_38, 2)
    mean_19: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_10, [-1], True);  pow_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_39: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_41, mean_18);  add_41 = mean_18 = None
    add_42: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_19, 1e-07);  mean_19 = None
    sqrt_14: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_42);  add_42 = None
    alias_14: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_14)
    div_19: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_39, sqrt_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_43: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_29, div_19)
    add_43: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_43, primals_30);  mul_43 = primals_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_86: "f32[512, 768]" = torch.ops.aten.view.default(add_43, [512, 768])
    permute_58: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_108, [1, 0]);  primals_108 = None
    addmm_13: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_109, view_86, permute_58);  primals_109 = None
    view_87: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_13, [1, 512, 3072]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_44: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_87, 0.5)
    mul_45: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_87, 0.7071067811865476)
    erf_4: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_45);  mul_45 = None
    add_44: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_46: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_44, add_44);  mul_44 = add_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_88: "f32[512, 3072]" = torch.ops.aten.view.default(mul_46, [512, 3072]);  mul_46 = None
    permute_59: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_110, [1, 0]);  primals_110 = None
    addmm_14: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_111, view_88, permute_59);  primals_111 = None
    view_89: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_14, [1, 512, 768]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_15: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    permute_60: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_15, [0, 1, 2]);  empty_15 = None
    bernoulli_15: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_60, 0.9);  permute_60 = None
    sub_40: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_15);  bernoulli_15 = None
    convert_element_type_20: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_40, torch.bool);  sub_40 = None
    scalar_tensor_20: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_25: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_20, scalar_tensor_20, view_89);  scalar_tensor_20 = view_89 = None
    mul_47: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_25, 1.1111111111111112);  where_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_45: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_47, add_43);  mul_47 = add_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_20: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_45, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_41: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_45, mean_20)
    pow_11: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_41, 2)
    mean_21: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_11, [-1], True);  pow_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_42: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_45, mean_20);  add_45 = mean_20 = None
    add_46: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_21, 1e-07);  mean_21 = None
    sqrt_15: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_46);  add_46 = None
    alias_15: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_15)
    div_20: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_42, sqrt_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_48: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_31, div_20)
    add_47: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_48, primals_32);  mul_48 = primals_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_61: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_112, [1, 0]);  primals_112 = None
    view_90: "f32[512, 768]" = torch.ops.aten.view.default(add_47, [512, 768])
    mm_5: "f32[512, 2304]" = torch.ops.aten.mm.default(view_90, permute_61)
    view_91: "f32[1, 512, 2304]" = torch.ops.aten.view.default(mm_5, [1, 512, 2304]);  mm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_92: "f32[1, 512, 12, 192]" = torch.ops.aten.view.default(view_91, [1, 512, 12, -1]);  view_91 = None
    
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
    slice_12: "f32[1, 1, 768]" = torch.ops.aten.slice.Tensor(unsqueeze_25, 2, 0, 9223372036854775807);  unsqueeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_93: "f32[1, 1, 12, 64]" = torch.ops.aten.view.default(slice_12, [1, 1, 12, -1]);  slice_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_63: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    add_48: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_15, permute_63);  getitem_15 = permute_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    unsqueeze_26: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_34, 0);  primals_34 = None
    unsqueeze_27: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, 1);  unsqueeze_26 = None
    slice_13: "f32[1, 1, 768]" = torch.ops.aten.slice.Tensor(unsqueeze_27, 2, 0, 9223372036854775807);  unsqueeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_94: "f32[1, 1, 12, 64]" = torch.ops.aten.view.default(slice_13, [1, 1, 12, -1]);  slice_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_64: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    add_49: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_17, permute_64);  getitem_17 = permute_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant10 = self._tensor_constant10
    lift_fresh_copy_10: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant10);  _tensor_constant10 = None
    mul_49: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_10, 1);  lift_fresh_copy_10 = None
    sqrt_16: "f32[]" = torch.ops.aten.sqrt.default(mul_49);  mul_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_21: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(add_48, sqrt_16);  add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_65: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(getitem_16, [0, 1, 3, 2]);  getitem_16 = None
    expand_20: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(div_21, [1, 12, 512, 64]);  div_21 = None
    view_95: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_20, [12, 512, 64]);  expand_20 = None
    expand_21: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_65, [1, 12, 64, 512]);  permute_65 = None
    view_96: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_21, [12, 64, 512]);  expand_21 = None
    bmm_10: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_95, view_96)
    view_97: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_10, [1, 12, 512, 512]);  bmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    convert_element_type_21: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool)
    bitwise_not_5: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_not.default(convert_element_type_21);  convert_element_type_21 = None
    _tensor_constant11 = self._tensor_constant11
    lift_fresh_copy_11: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant11);  _tensor_constant11 = None
    where_26: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(bitwise_not_5, lift_fresh_copy_11, view_97);  lift_fresh_copy_11 = view_97 = None
    amax_5: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where_26, [-1], True)
    sub_43: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where_26, amax_5);  where_26 = amax_5 = None
    exp_5: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_43);  sub_43 = None
    sum_6: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_22: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    scalar_tensor_21: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_27: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(bitwise_not_5, scalar_tensor_21, div_22);  bitwise_not_5 = scalar_tensor_21 = div_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_16: "f32[1, 12, 512, 512]" = torch.ops.aten.empty.memory_format([1, 12, 512, 512], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    permute_66: "f32[1, 12, 512, 512]" = torch.ops.aten.permute.default(empty_16, [0, 1, 2, 3]);  empty_16 = None
    bernoulli_16: "f32[1, 12, 512, 512]" = torch.ops.aten.bernoulli.p(permute_66, 0.9);  permute_66 = None
    sub_44: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_16);  bernoulli_16 = None
    convert_element_type_22: "b8[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_44, torch.bool);  sub_44 = None
    scalar_tensor_22: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_28: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_22, scalar_tensor_22, where_27);  scalar_tensor_22 = None
    mul_50: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_28, 1.1111111111111112);  where_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_22: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(mul_50, [1, 12, 512, 512]);  mul_50 = None
    view_98: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_22, [12, 512, 512]);  expand_22 = None
    expand_23: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(add_49, [1, 12, 512, 64]);  add_49 = None
    view_99: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_23, [12, 512, 64]);  expand_23 = None
    bmm_11: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_98, view_99)
    view_100: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_11, [1, 12, 512, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_67: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_100, [0, 2, 1, 3]);  view_100 = None
    clone_5: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_67, memory_format = torch.contiguous_format);  permute_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_101: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_5, [1, 512, -1]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_102: "f32[512, 768]" = torch.ops.aten.view.default(view_101, [512, 768]);  view_101 = None
    permute_68: "f32[768, 768]" = torch.ops.aten.permute.default(primals_113, [1, 0]);  primals_113 = None
    addmm_15: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_114, view_102, permute_68);  primals_114 = None
    view_103: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_15, [1, 512, 768]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_17: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    permute_69: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_17, [0, 1, 2]);  empty_17 = None
    bernoulli_17: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_69, 0.9);  permute_69 = None
    sub_45: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_17);  bernoulli_17 = None
    convert_element_type_23: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_45, torch.bool);  sub_45 = None
    scalar_tensor_23: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_29: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_23, scalar_tensor_23, view_103);  scalar_tensor_23 = view_103 = None
    mul_51: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_29, 1.1111111111111112);  where_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_50: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_51, add_47);  mul_51 = add_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_22: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_50, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_46: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_50, mean_22)
    pow_12: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_46, 2)
    mean_23: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_12, [-1], True);  pow_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_47: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_50, mean_22);  add_50 = mean_22 = None
    add_51: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_23, 1e-07);  mean_23 = None
    sqrt_17: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_51);  add_51 = None
    alias_17: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_17)
    div_23: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_47, sqrt_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_52: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_35, div_23)
    add_52: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_52, primals_36);  mul_52 = primals_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_104: "f32[512, 768]" = torch.ops.aten.view.default(add_52, [512, 768])
    permute_70: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
    addmm_16: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_116, view_104, permute_70);  primals_116 = None
    view_105: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_16, [1, 512, 3072]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_53: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_105, 0.5)
    mul_54: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_105, 0.7071067811865476)
    erf_5: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_54);  mul_54 = None
    add_53: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_55: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_53, add_53);  mul_53 = add_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_106: "f32[512, 3072]" = torch.ops.aten.view.default(mul_55, [512, 3072]);  mul_55 = None
    permute_71: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_117, [1, 0]);  primals_117 = None
    addmm_17: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_118, view_106, permute_71);  primals_118 = None
    view_107: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_17, [1, 512, 768]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_18: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    permute_72: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_18, [0, 1, 2]);  empty_18 = None
    bernoulli_18: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_72, 0.9);  permute_72 = None
    sub_48: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_18);  bernoulli_18 = None
    convert_element_type_24: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_48, torch.bool);  sub_48 = None
    scalar_tensor_24: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_30: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_24, scalar_tensor_24, view_107);  scalar_tensor_24 = view_107 = None
    mul_56: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_30, 1.1111111111111112);  where_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_54: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_56, add_52);  mul_56 = add_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_24: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_54, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_49: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_54, mean_24)
    pow_13: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_49, 2)
    mean_25: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_13, [-1], True);  pow_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_50: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_54, mean_24);  add_54 = mean_24 = None
    add_55: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_25, 1e-07);  mean_25 = None
    sqrt_18: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_55);  add_55 = None
    alias_18: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_18)
    div_24: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_50, sqrt_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_57: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_37, div_24)
    add_56: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_57, primals_38);  mul_57 = primals_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_73: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_119, [1, 0]);  primals_119 = None
    view_108: "f32[512, 768]" = torch.ops.aten.view.default(add_56, [512, 768])
    mm_6: "f32[512, 2304]" = torch.ops.aten.mm.default(view_108, permute_73)
    view_109: "f32[1, 512, 2304]" = torch.ops.aten.view.default(mm_6, [1, 512, 2304]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_110: "f32[1, 512, 12, 192]" = torch.ops.aten.view.default(view_109, [1, 512, 12, -1]);  view_109 = None
    
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
    slice_14: "f32[1, 1, 768]" = torch.ops.aten.slice.Tensor(unsqueeze_29, 2, 0, 9223372036854775807);  unsqueeze_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_111: "f32[1, 1, 12, 64]" = torch.ops.aten.view.default(slice_14, [1, 1, 12, -1]);  slice_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_75: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_111, [0, 2, 1, 3]);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    add_57: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_18, permute_75);  getitem_18 = permute_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    unsqueeze_30: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_40, 0);  primals_40 = None
    unsqueeze_31: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, 1);  unsqueeze_30 = None
    slice_15: "f32[1, 1, 768]" = torch.ops.aten.slice.Tensor(unsqueeze_31, 2, 0, 9223372036854775807);  unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_112: "f32[1, 1, 12, 64]" = torch.ops.aten.view.default(slice_15, [1, 1, 12, -1]);  slice_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_76: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    add_58: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_20, permute_76);  getitem_20 = permute_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant12 = self._tensor_constant12
    lift_fresh_copy_12: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant12);  _tensor_constant12 = None
    mul_58: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_12, 1);  lift_fresh_copy_12 = None
    sqrt_19: "f32[]" = torch.ops.aten.sqrt.default(mul_58);  mul_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_25: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(add_57, sqrt_19);  add_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_77: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(getitem_19, [0, 1, 3, 2]);  getitem_19 = None
    expand_24: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(div_25, [1, 12, 512, 64]);  div_25 = None
    view_113: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_24, [12, 512, 64]);  expand_24 = None
    expand_25: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_77, [1, 12, 64, 512]);  permute_77 = None
    view_114: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_25, [12, 64, 512]);  expand_25 = None
    bmm_12: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_113, view_114)
    view_115: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_12, [1, 12, 512, 512]);  bmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    convert_element_type_25: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool)
    bitwise_not_6: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_not.default(convert_element_type_25);  convert_element_type_25 = None
    _tensor_constant13 = self._tensor_constant13
    lift_fresh_copy_13: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant13);  _tensor_constant13 = None
    where_31: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(bitwise_not_6, lift_fresh_copy_13, view_115);  lift_fresh_copy_13 = view_115 = None
    amax_6: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where_31, [-1], True)
    sub_51: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where_31, amax_6);  where_31 = amax_6 = None
    exp_6: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_51);  sub_51 = None
    sum_7: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_26: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    scalar_tensor_25: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_32: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(bitwise_not_6, scalar_tensor_25, div_26);  bitwise_not_6 = scalar_tensor_25 = div_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_19: "f32[1, 12, 512, 512]" = torch.ops.aten.empty.memory_format([1, 12, 512, 512], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    permute_78: "f32[1, 12, 512, 512]" = torch.ops.aten.permute.default(empty_19, [0, 1, 2, 3]);  empty_19 = None
    bernoulli_19: "f32[1, 12, 512, 512]" = torch.ops.aten.bernoulli.p(permute_78, 0.9);  permute_78 = None
    sub_52: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_19);  bernoulli_19 = None
    convert_element_type_26: "b8[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_52, torch.bool);  sub_52 = None
    scalar_tensor_26: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_33: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_26, scalar_tensor_26, where_32);  scalar_tensor_26 = None
    mul_59: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_33, 1.1111111111111112);  where_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_26: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(mul_59, [1, 12, 512, 512]);  mul_59 = None
    view_116: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_26, [12, 512, 512]);  expand_26 = None
    expand_27: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(add_58, [1, 12, 512, 64]);  add_58 = None
    view_117: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_27, [12, 512, 64]);  expand_27 = None
    bmm_13: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_116, view_117)
    view_118: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_13, [1, 12, 512, 64]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_79: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
    clone_6: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_119: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_6, [1, 512, -1]);  clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_120: "f32[512, 768]" = torch.ops.aten.view.default(view_119, [512, 768]);  view_119 = None
    permute_80: "f32[768, 768]" = torch.ops.aten.permute.default(primals_120, [1, 0]);  primals_120 = None
    addmm_18: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_121, view_120, permute_80);  primals_121 = None
    view_121: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_18, [1, 512, 768]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_20: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    permute_81: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_20, [0, 1, 2]);  empty_20 = None
    bernoulli_20: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_81, 0.9);  permute_81 = None
    sub_53: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_20);  bernoulli_20 = None
    convert_element_type_27: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_53, torch.bool);  sub_53 = None
    scalar_tensor_27: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_34: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_27, scalar_tensor_27, view_121);  scalar_tensor_27 = view_121 = None
    mul_60: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_34, 1.1111111111111112);  where_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_59: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_60, add_56);  mul_60 = add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_26: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_59, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_54: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_59, mean_26)
    pow_14: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_54, 2)
    mean_27: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_14, [-1], True);  pow_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_55: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_59, mean_26);  add_59 = mean_26 = None
    add_60: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_27, 1e-07);  mean_27 = None
    sqrt_20: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_60);  add_60 = None
    alias_20: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_20)
    div_27: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_55, sqrt_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_61: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_41, div_27)
    add_61: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_61, primals_42);  mul_61 = primals_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_122: "f32[512, 768]" = torch.ops.aten.view.default(add_61, [512, 768])
    permute_82: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_122, [1, 0]);  primals_122 = None
    addmm_19: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_123, view_122, permute_82);  primals_123 = None
    view_123: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_19, [1, 512, 3072]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_62: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_123, 0.5)
    mul_63: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_123, 0.7071067811865476)
    erf_6: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_63);  mul_63 = None
    add_62: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_64: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_62, add_62);  mul_62 = add_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_124: "f32[512, 3072]" = torch.ops.aten.view.default(mul_64, [512, 3072]);  mul_64 = None
    permute_83: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_124, [1, 0]);  primals_124 = None
    addmm_20: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_125, view_124, permute_83);  primals_125 = None
    view_125: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_20, [1, 512, 768]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_21: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    permute_84: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_21, [0, 1, 2]);  empty_21 = None
    bernoulli_21: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_84, 0.9);  permute_84 = None
    sub_56: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_21);  bernoulli_21 = None
    convert_element_type_28: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_56, torch.bool);  sub_56 = None
    scalar_tensor_28: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_35: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_28, scalar_tensor_28, view_125);  scalar_tensor_28 = view_125 = None
    mul_65: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_35, 1.1111111111111112);  where_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_63: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_65, add_61);  mul_65 = add_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_28: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_63, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_57: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_63, mean_28)
    pow_15: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_57, 2)
    mean_29: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_15, [-1], True);  pow_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_58: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_63, mean_28);  add_63 = mean_28 = None
    add_64: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_29, 1e-07);  mean_29 = None
    sqrt_21: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_64);  add_64 = None
    alias_21: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_21)
    div_28: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_58, sqrt_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_66: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_43, div_28)
    add_65: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_66, primals_44);  mul_66 = primals_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_85: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_126, [1, 0]);  primals_126 = None
    view_126: "f32[512, 768]" = torch.ops.aten.view.default(add_65, [512, 768])
    mm_7: "f32[512, 2304]" = torch.ops.aten.mm.default(view_126, permute_85)
    view_127: "f32[1, 512, 2304]" = torch.ops.aten.view.default(mm_7, [1, 512, 2304]);  mm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_128: "f32[1, 512, 12, 192]" = torch.ops.aten.view.default(view_127, [1, 512, 12, -1]);  view_127 = None
    
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
    slice_16: "f32[1, 1, 768]" = torch.ops.aten.slice.Tensor(unsqueeze_33, 2, 0, 9223372036854775807);  unsqueeze_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_129: "f32[1, 1, 12, 64]" = torch.ops.aten.view.default(slice_16, [1, 1, 12, -1]);  slice_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_87: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_129, [0, 2, 1, 3]);  view_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    add_66: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_21, permute_87);  getitem_21 = permute_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    unsqueeze_34: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_46, 0);  primals_46 = None
    unsqueeze_35: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, 1);  unsqueeze_34 = None
    slice_17: "f32[1, 1, 768]" = torch.ops.aten.slice.Tensor(unsqueeze_35, 2, 0, 9223372036854775807);  unsqueeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_130: "f32[1, 1, 12, 64]" = torch.ops.aten.view.default(slice_17, [1, 1, 12, -1]);  slice_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_88: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    add_67: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_23, permute_88);  getitem_23 = permute_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant14 = self._tensor_constant14
    lift_fresh_copy_14: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant14);  _tensor_constant14 = None
    mul_67: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_14, 1);  lift_fresh_copy_14 = None
    sqrt_22: "f32[]" = torch.ops.aten.sqrt.default(mul_67);  mul_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_29: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(add_66, sqrt_22);  add_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_89: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(getitem_22, [0, 1, 3, 2]);  getitem_22 = None
    expand_28: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(div_29, [1, 12, 512, 64]);  div_29 = None
    view_131: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_28, [12, 512, 64]);  expand_28 = None
    expand_29: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_89, [1, 12, 64, 512]);  permute_89 = None
    view_132: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_29, [12, 64, 512]);  expand_29 = None
    bmm_14: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_131, view_132)
    view_133: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_14, [1, 12, 512, 512]);  bmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    convert_element_type_29: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool)
    bitwise_not_7: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_not.default(convert_element_type_29);  convert_element_type_29 = None
    _tensor_constant15 = self._tensor_constant15
    lift_fresh_copy_15: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant15);  _tensor_constant15 = None
    where_36: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(bitwise_not_7, lift_fresh_copy_15, view_133);  lift_fresh_copy_15 = view_133 = None
    amax_7: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where_36, [-1], True)
    sub_59: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where_36, amax_7);  where_36 = amax_7 = None
    exp_7: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_59);  sub_59 = None
    sum_8: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_30: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    scalar_tensor_29: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_37: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(bitwise_not_7, scalar_tensor_29, div_30);  bitwise_not_7 = scalar_tensor_29 = div_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_22: "f32[1, 12, 512, 512]" = torch.ops.aten.empty.memory_format([1, 12, 512, 512], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    permute_90: "f32[1, 12, 512, 512]" = torch.ops.aten.permute.default(empty_22, [0, 1, 2, 3]);  empty_22 = None
    bernoulli_22: "f32[1, 12, 512, 512]" = torch.ops.aten.bernoulli.p(permute_90, 0.9);  permute_90 = None
    sub_60: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_22);  bernoulli_22 = None
    convert_element_type_30: "b8[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_60, torch.bool);  sub_60 = None
    scalar_tensor_30: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_38: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_30, scalar_tensor_30, where_37);  scalar_tensor_30 = None
    mul_68: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_38, 1.1111111111111112);  where_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_30: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(mul_68, [1, 12, 512, 512]);  mul_68 = None
    view_134: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_30, [12, 512, 512]);  expand_30 = None
    expand_31: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(add_67, [1, 12, 512, 64]);  add_67 = None
    view_135: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_31, [12, 512, 64]);  expand_31 = None
    bmm_15: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_134, view_135)
    view_136: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_15, [1, 12, 512, 64]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_91: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
    clone_7: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_91, memory_format = torch.contiguous_format);  permute_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_137: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_7, [1, 512, -1]);  clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_138: "f32[512, 768]" = torch.ops.aten.view.default(view_137, [512, 768]);  view_137 = None
    permute_92: "f32[768, 768]" = torch.ops.aten.permute.default(primals_127, [1, 0]);  primals_127 = None
    addmm_21: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_128, view_138, permute_92);  primals_128 = None
    view_139: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_21, [1, 512, 768]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_23: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    permute_93: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_23, [0, 1, 2]);  empty_23 = None
    bernoulli_23: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_93, 0.9);  permute_93 = None
    sub_61: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_23);  bernoulli_23 = None
    convert_element_type_31: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_61, torch.bool);  sub_61 = None
    scalar_tensor_31: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_39: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_31, scalar_tensor_31, view_139);  scalar_tensor_31 = view_139 = None
    mul_69: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_39, 1.1111111111111112);  where_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_68: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_69, add_65);  mul_69 = add_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_30: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_68, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_62: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_68, mean_30)
    pow_16: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_62, 2)
    mean_31: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_16, [-1], True);  pow_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_63: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_68, mean_30);  add_68 = mean_30 = None
    add_69: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_31, 1e-07);  mean_31 = None
    sqrt_23: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_69);  add_69 = None
    alias_23: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_23)
    div_31: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_63, sqrt_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_70: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_47, div_31)
    add_70: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_70, primals_48);  mul_70 = primals_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_140: "f32[512, 768]" = torch.ops.aten.view.default(add_70, [512, 768])
    permute_94: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_129, [1, 0]);  primals_129 = None
    addmm_22: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_130, view_140, permute_94);  primals_130 = None
    view_141: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_22, [1, 512, 3072]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_71: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_141, 0.5)
    mul_72: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_141, 0.7071067811865476)
    erf_7: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_72);  mul_72 = None
    add_71: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_73: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_71, add_71);  mul_71 = add_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_142: "f32[512, 3072]" = torch.ops.aten.view.default(mul_73, [512, 3072]);  mul_73 = None
    permute_95: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_131, [1, 0]);  primals_131 = None
    addmm_23: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_132, view_142, permute_95);  primals_132 = None
    view_143: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_23, [1, 512, 768]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_24: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    permute_96: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_24, [0, 1, 2]);  empty_24 = None
    bernoulli_24: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_96, 0.9);  permute_96 = None
    sub_64: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_24);  bernoulli_24 = None
    convert_element_type_32: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_64, torch.bool);  sub_64 = None
    scalar_tensor_32: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_40: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_32, scalar_tensor_32, view_143);  scalar_tensor_32 = view_143 = None
    mul_74: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_40, 1.1111111111111112);  where_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_72: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_74, add_70);  mul_74 = add_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_32: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_72, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_65: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_72, mean_32)
    pow_17: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_65, 2)
    mean_33: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_17, [-1], True);  pow_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_66: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_72, mean_32);  add_72 = mean_32 = None
    add_73: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_33, 1e-07);  mean_33 = None
    sqrt_24: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_73);  add_73 = None
    alias_24: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_24)
    div_32: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_66, sqrt_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_75: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_49, div_32)
    add_74: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_75, primals_50);  mul_75 = primals_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_97: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_133, [1, 0]);  primals_133 = None
    view_144: "f32[512, 768]" = torch.ops.aten.view.default(add_74, [512, 768])
    mm_8: "f32[512, 2304]" = torch.ops.aten.mm.default(view_144, permute_97)
    view_145: "f32[1, 512, 2304]" = torch.ops.aten.view.default(mm_8, [1, 512, 2304]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_146: "f32[1, 512, 12, 192]" = torch.ops.aten.view.default(view_145, [1, 512, 12, -1]);  view_145 = None
    
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
    slice_18: "f32[1, 1, 768]" = torch.ops.aten.slice.Tensor(unsqueeze_37, 2, 0, 9223372036854775807);  unsqueeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_147: "f32[1, 1, 12, 64]" = torch.ops.aten.view.default(slice_18, [1, 1, 12, -1]);  slice_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_99: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_147, [0, 2, 1, 3]);  view_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    add_75: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_24, permute_99);  getitem_24 = permute_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    unsqueeze_38: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_52, 0);  primals_52 = None
    unsqueeze_39: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, 1);  unsqueeze_38 = None
    slice_19: "f32[1, 1, 768]" = torch.ops.aten.slice.Tensor(unsqueeze_39, 2, 0, 9223372036854775807);  unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_148: "f32[1, 1, 12, 64]" = torch.ops.aten.view.default(slice_19, [1, 1, 12, -1]);  slice_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_100: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    add_76: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_26, permute_100);  getitem_26 = permute_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant16 = self._tensor_constant16
    lift_fresh_copy_16: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant16);  _tensor_constant16 = None
    mul_76: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_16, 1);  lift_fresh_copy_16 = None
    sqrt_25: "f32[]" = torch.ops.aten.sqrt.default(mul_76);  mul_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_33: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(add_75, sqrt_25);  add_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_101: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(getitem_25, [0, 1, 3, 2]);  getitem_25 = None
    expand_32: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(div_33, [1, 12, 512, 64]);  div_33 = None
    view_149: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_32, [12, 512, 64]);  expand_32 = None
    expand_33: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_101, [1, 12, 64, 512]);  permute_101 = None
    view_150: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_33, [12, 64, 512]);  expand_33 = None
    bmm_16: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_149, view_150)
    view_151: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_16, [1, 12, 512, 512]);  bmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    convert_element_type_33: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool)
    bitwise_not_8: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_not.default(convert_element_type_33);  convert_element_type_33 = None
    _tensor_constant17 = self._tensor_constant17
    lift_fresh_copy_17: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant17);  _tensor_constant17 = None
    where_41: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(bitwise_not_8, lift_fresh_copy_17, view_151);  lift_fresh_copy_17 = view_151 = None
    amax_8: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where_41, [-1], True)
    sub_67: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where_41, amax_8);  where_41 = amax_8 = None
    exp_8: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_67);  sub_67 = None
    sum_9: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_34: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    scalar_tensor_33: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_42: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(bitwise_not_8, scalar_tensor_33, div_34);  bitwise_not_8 = scalar_tensor_33 = div_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_25: "f32[1, 12, 512, 512]" = torch.ops.aten.empty.memory_format([1, 12, 512, 512], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    permute_102: "f32[1, 12, 512, 512]" = torch.ops.aten.permute.default(empty_25, [0, 1, 2, 3]);  empty_25 = None
    bernoulli_25: "f32[1, 12, 512, 512]" = torch.ops.aten.bernoulli.p(permute_102, 0.9);  permute_102 = None
    sub_68: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_25);  bernoulli_25 = None
    convert_element_type_34: "b8[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_68, torch.bool);  sub_68 = None
    scalar_tensor_34: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_43: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_34, scalar_tensor_34, where_42);  scalar_tensor_34 = None
    mul_77: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_43, 1.1111111111111112);  where_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_34: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(mul_77, [1, 12, 512, 512]);  mul_77 = None
    view_152: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_34, [12, 512, 512]);  expand_34 = None
    expand_35: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(add_76, [1, 12, 512, 64]);  add_76 = None
    view_153: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_35, [12, 512, 64]);  expand_35 = None
    bmm_17: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_152, view_153)
    view_154: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_17, [1, 12, 512, 64]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_103: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_154, [0, 2, 1, 3]);  view_154 = None
    clone_8: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_103, memory_format = torch.contiguous_format);  permute_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_155: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_8, [1, 512, -1]);  clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_156: "f32[512, 768]" = torch.ops.aten.view.default(view_155, [512, 768]);  view_155 = None
    permute_104: "f32[768, 768]" = torch.ops.aten.permute.default(primals_134, [1, 0]);  primals_134 = None
    addmm_24: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_135, view_156, permute_104);  primals_135 = None
    view_157: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_24, [1, 512, 768]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_26: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    permute_105: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_26, [0, 1, 2]);  empty_26 = None
    bernoulli_26: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_105, 0.9);  permute_105 = None
    sub_69: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_26);  bernoulli_26 = None
    convert_element_type_35: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_69, torch.bool);  sub_69 = None
    scalar_tensor_35: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_44: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_35, scalar_tensor_35, view_157);  scalar_tensor_35 = view_157 = None
    mul_78: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_44, 1.1111111111111112);  where_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_77: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_78, add_74);  mul_78 = add_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_34: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_77, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_70: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_77, mean_34)
    pow_18: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_70, 2)
    mean_35: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_18, [-1], True);  pow_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_71: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_77, mean_34);  add_77 = mean_34 = None
    add_78: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_35, 1e-07);  mean_35 = None
    sqrt_26: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_78);  add_78 = None
    alias_26: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_26)
    div_35: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_71, sqrt_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_79: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_53, div_35)
    add_79: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_79, primals_54);  mul_79 = primals_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_158: "f32[512, 768]" = torch.ops.aten.view.default(add_79, [512, 768])
    permute_106: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_136, [1, 0]);  primals_136 = None
    addmm_25: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_137, view_158, permute_106);  primals_137 = None
    view_159: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_25, [1, 512, 3072]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_80: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_159, 0.5)
    mul_81: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_159, 0.7071067811865476)
    erf_8: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_81);  mul_81 = None
    add_80: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_82: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_80, add_80);  mul_80 = add_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_160: "f32[512, 3072]" = torch.ops.aten.view.default(mul_82, [512, 3072]);  mul_82 = None
    permute_107: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_138, [1, 0]);  primals_138 = None
    addmm_26: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_139, view_160, permute_107);  primals_139 = None
    view_161: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_26, [1, 512, 768]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_27: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    permute_108: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_27, [0, 1, 2]);  empty_27 = None
    bernoulli_27: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_108, 0.9);  permute_108 = None
    sub_72: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_27);  bernoulli_27 = None
    convert_element_type_36: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_72, torch.bool);  sub_72 = None
    scalar_tensor_36: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_45: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_36, scalar_tensor_36, view_161);  scalar_tensor_36 = view_161 = None
    mul_83: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_45, 1.1111111111111112);  where_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_81: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_83, add_79);  mul_83 = add_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_36: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_81, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_73: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_81, mean_36)
    pow_19: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_73, 2)
    mean_37: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_19, [-1], True);  pow_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_74: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_81, mean_36);  add_81 = mean_36 = None
    add_82: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_37, 1e-07);  mean_37 = None
    sqrt_27: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_82);  add_82 = None
    alias_27: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_27)
    div_36: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_74, sqrt_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_84: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_55, div_36)
    add_83: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_84, primals_56);  mul_84 = primals_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_109: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_140, [1, 0]);  primals_140 = None
    view_162: "f32[512, 768]" = torch.ops.aten.view.default(add_83, [512, 768])
    mm_9: "f32[512, 2304]" = torch.ops.aten.mm.default(view_162, permute_109)
    view_163: "f32[1, 512, 2304]" = torch.ops.aten.view.default(mm_9, [1, 512, 2304]);  mm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_164: "f32[1, 512, 12, 192]" = torch.ops.aten.view.default(view_163, [1, 512, 12, -1]);  view_163 = None
    
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
    slice_20: "f32[1, 1, 768]" = torch.ops.aten.slice.Tensor(unsqueeze_41, 2, 0, 9223372036854775807);  unsqueeze_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_165: "f32[1, 1, 12, 64]" = torch.ops.aten.view.default(slice_20, [1, 1, 12, -1]);  slice_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_111: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_165, [0, 2, 1, 3]);  view_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    add_84: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_27, permute_111);  getitem_27 = permute_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    unsqueeze_42: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_58, 0);  primals_58 = None
    unsqueeze_43: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, 1);  unsqueeze_42 = None
    slice_21: "f32[1, 1, 768]" = torch.ops.aten.slice.Tensor(unsqueeze_43, 2, 0, 9223372036854775807);  unsqueeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_166: "f32[1, 1, 12, 64]" = torch.ops.aten.view.default(slice_21, [1, 1, 12, -1]);  slice_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_112: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_166, [0, 2, 1, 3]);  view_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    add_85: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_29, permute_112);  getitem_29 = permute_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant18 = self._tensor_constant18
    lift_fresh_copy_18: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant18);  _tensor_constant18 = None
    mul_85: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_18, 1);  lift_fresh_copy_18 = None
    sqrt_28: "f32[]" = torch.ops.aten.sqrt.default(mul_85);  mul_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_37: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(add_84, sqrt_28);  add_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_113: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(getitem_28, [0, 1, 3, 2]);  getitem_28 = None
    expand_36: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(div_37, [1, 12, 512, 64]);  div_37 = None
    view_167: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_36, [12, 512, 64]);  expand_36 = None
    expand_37: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_113, [1, 12, 64, 512]);  permute_113 = None
    view_168: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_37, [12, 64, 512]);  expand_37 = None
    bmm_18: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_167, view_168)
    view_169: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_18, [1, 12, 512, 512]);  bmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    convert_element_type_37: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool)
    bitwise_not_9: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_not.default(convert_element_type_37);  convert_element_type_37 = None
    _tensor_constant19 = self._tensor_constant19
    lift_fresh_copy_19: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant19);  _tensor_constant19 = None
    where_46: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(bitwise_not_9, lift_fresh_copy_19, view_169);  lift_fresh_copy_19 = view_169 = None
    amax_9: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where_46, [-1], True)
    sub_75: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where_46, amax_9);  where_46 = amax_9 = None
    exp_9: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_75);  sub_75 = None
    sum_10: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_38: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    scalar_tensor_37: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_47: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(bitwise_not_9, scalar_tensor_37, div_38);  bitwise_not_9 = scalar_tensor_37 = div_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_28: "f32[1, 12, 512, 512]" = torch.ops.aten.empty.memory_format([1, 12, 512, 512], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    permute_114: "f32[1, 12, 512, 512]" = torch.ops.aten.permute.default(empty_28, [0, 1, 2, 3]);  empty_28 = None
    bernoulli_28: "f32[1, 12, 512, 512]" = torch.ops.aten.bernoulli.p(permute_114, 0.9);  permute_114 = None
    sub_76: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_28);  bernoulli_28 = None
    convert_element_type_38: "b8[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_76, torch.bool);  sub_76 = None
    scalar_tensor_38: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_48: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_38, scalar_tensor_38, where_47);  scalar_tensor_38 = None
    mul_86: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_48, 1.1111111111111112);  where_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_38: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(mul_86, [1, 12, 512, 512]);  mul_86 = None
    view_170: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_38, [12, 512, 512]);  expand_38 = None
    expand_39: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(add_85, [1, 12, 512, 64]);  add_85 = None
    view_171: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_39, [12, 512, 64]);  expand_39 = None
    bmm_19: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_170, view_171)
    view_172: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_19, [1, 12, 512, 64]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_115: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_172, [0, 2, 1, 3]);  view_172 = None
    clone_9: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_173: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_9, [1, 512, -1]);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_174: "f32[512, 768]" = torch.ops.aten.view.default(view_173, [512, 768]);  view_173 = None
    permute_116: "f32[768, 768]" = torch.ops.aten.permute.default(primals_141, [1, 0]);  primals_141 = None
    addmm_27: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_142, view_174, permute_116);  primals_142 = None
    view_175: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_27, [1, 512, 768]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_29: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    permute_117: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_29, [0, 1, 2]);  empty_29 = None
    bernoulli_29: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_117, 0.9);  permute_117 = None
    sub_77: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_29);  bernoulli_29 = None
    convert_element_type_39: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_77, torch.bool);  sub_77 = None
    scalar_tensor_39: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_49: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_39, scalar_tensor_39, view_175);  scalar_tensor_39 = view_175 = None
    mul_87: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_49, 1.1111111111111112);  where_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_86: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_87, add_83);  mul_87 = add_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_38: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_86, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_78: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_86, mean_38)
    pow_20: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_78, 2)
    mean_39: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_20, [-1], True);  pow_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_79: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_86, mean_38);  add_86 = mean_38 = None
    add_87: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_39, 1e-07);  mean_39 = None
    sqrt_29: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_87);  add_87 = None
    alias_29: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_29)
    div_39: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_79, sqrt_29)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_88: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_59, div_39)
    add_88: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_88, primals_60);  mul_88 = primals_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_176: "f32[512, 768]" = torch.ops.aten.view.default(add_88, [512, 768])
    permute_118: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_143, [1, 0]);  primals_143 = None
    addmm_28: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_144, view_176, permute_118);  primals_144 = None
    view_177: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_28, [1, 512, 3072]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_89: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_177, 0.5)
    mul_90: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_177, 0.7071067811865476)
    erf_9: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_90);  mul_90 = None
    add_89: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_91: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_89, add_89);  mul_89 = add_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_178: "f32[512, 3072]" = torch.ops.aten.view.default(mul_91, [512, 3072]);  mul_91 = None
    permute_119: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_145, [1, 0]);  primals_145 = None
    addmm_29: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_146, view_178, permute_119);  primals_146 = None
    view_179: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_29, [1, 512, 768]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_30: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    permute_120: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_30, [0, 1, 2]);  empty_30 = None
    bernoulli_30: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_120, 0.9);  permute_120 = None
    sub_80: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_30);  bernoulli_30 = None
    convert_element_type_40: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_80, torch.bool);  sub_80 = None
    scalar_tensor_40: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_50: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_40, scalar_tensor_40, view_179);  scalar_tensor_40 = view_179 = None
    mul_92: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_50, 1.1111111111111112);  where_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_90: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_92, add_88);  mul_92 = add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_40: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_90, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_81: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_90, mean_40)
    pow_21: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_81, 2)
    mean_41: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_21, [-1], True);  pow_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_82: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_90, mean_40);  add_90 = mean_40 = None
    add_91: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_41, 1e-07);  mean_41 = None
    sqrt_30: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_91);  add_91 = None
    alias_30: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_30)
    div_40: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_82, sqrt_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_93: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_61, div_40)
    add_92: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_93, primals_62);  mul_93 = primals_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_121: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_147, [1, 0]);  primals_147 = None
    view_180: "f32[512, 768]" = torch.ops.aten.view.default(add_92, [512, 768])
    mm_10: "f32[512, 2304]" = torch.ops.aten.mm.default(view_180, permute_121)
    view_181: "f32[1, 512, 2304]" = torch.ops.aten.view.default(mm_10, [1, 512, 2304]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_182: "f32[1, 512, 12, 192]" = torch.ops.aten.view.default(view_181, [1, 512, 12, -1]);  view_181 = None
    
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
    slice_22: "f32[1, 1, 768]" = torch.ops.aten.slice.Tensor(unsqueeze_45, 2, 0, 9223372036854775807);  unsqueeze_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_183: "f32[1, 1, 12, 64]" = torch.ops.aten.view.default(slice_22, [1, 1, 12, -1]);  slice_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_123: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    add_93: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_30, permute_123);  getitem_30 = permute_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    unsqueeze_46: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_64, 0);  primals_64 = None
    unsqueeze_47: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, 1);  unsqueeze_46 = None
    slice_23: "f32[1, 1, 768]" = torch.ops.aten.slice.Tensor(unsqueeze_47, 2, 0, 9223372036854775807);  unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_184: "f32[1, 1, 12, 64]" = torch.ops.aten.view.default(slice_23, [1, 1, 12, -1]);  slice_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_124: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    add_94: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_32, permute_124);  getitem_32 = permute_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant20 = self._tensor_constant20
    lift_fresh_copy_20: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant20);  _tensor_constant20 = None
    mul_94: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_20, 1);  lift_fresh_copy_20 = None
    sqrt_31: "f32[]" = torch.ops.aten.sqrt.default(mul_94);  mul_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_41: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(add_93, sqrt_31);  add_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_125: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(getitem_31, [0, 1, 3, 2]);  getitem_31 = None
    expand_40: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(div_41, [1, 12, 512, 64]);  div_41 = None
    view_185: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_40, [12, 512, 64]);  expand_40 = None
    expand_41: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_125, [1, 12, 64, 512]);  permute_125 = None
    view_186: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_41, [12, 64, 512]);  expand_41 = None
    bmm_20: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_185, view_186)
    view_187: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_20, [1, 12, 512, 512]);  bmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    convert_element_type_41: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool)
    bitwise_not_10: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_not.default(convert_element_type_41);  convert_element_type_41 = None
    _tensor_constant21 = self._tensor_constant21
    lift_fresh_copy_21: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant21);  _tensor_constant21 = None
    where_51: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(bitwise_not_10, lift_fresh_copy_21, view_187);  lift_fresh_copy_21 = view_187 = None
    amax_10: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where_51, [-1], True)
    sub_83: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where_51, amax_10);  where_51 = amax_10 = None
    exp_10: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_83);  sub_83 = None
    sum_11: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_42: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    scalar_tensor_41: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_52: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(bitwise_not_10, scalar_tensor_41, div_42);  bitwise_not_10 = scalar_tensor_41 = div_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_31: "f32[1, 12, 512, 512]" = torch.ops.aten.empty.memory_format([1, 12, 512, 512], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    permute_126: "f32[1, 12, 512, 512]" = torch.ops.aten.permute.default(empty_31, [0, 1, 2, 3]);  empty_31 = None
    bernoulli_31: "f32[1, 12, 512, 512]" = torch.ops.aten.bernoulli.p(permute_126, 0.9);  permute_126 = None
    sub_84: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_31);  bernoulli_31 = None
    convert_element_type_42: "b8[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_84, torch.bool);  sub_84 = None
    scalar_tensor_42: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_53: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_42, scalar_tensor_42, where_52);  scalar_tensor_42 = None
    mul_95: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_53, 1.1111111111111112);  where_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_42: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(mul_95, [1, 12, 512, 512]);  mul_95 = None
    view_188: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_42, [12, 512, 512]);  expand_42 = None
    expand_43: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(add_94, [1, 12, 512, 64]);  add_94 = None
    view_189: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_43, [12, 512, 64]);  expand_43 = None
    bmm_21: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_188, view_189)
    view_190: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_21, [1, 12, 512, 64]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_127: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_190, [0, 2, 1, 3]);  view_190 = None
    clone_10: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_127, memory_format = torch.contiguous_format);  permute_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_191: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_10, [1, 512, -1]);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_192: "f32[512, 768]" = torch.ops.aten.view.default(view_191, [512, 768]);  view_191 = None
    permute_128: "f32[768, 768]" = torch.ops.aten.permute.default(primals_148, [1, 0]);  primals_148 = None
    addmm_30: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_149, view_192, permute_128);  primals_149 = None
    view_193: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_30, [1, 512, 768]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_32: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    permute_129: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_32, [0, 1, 2]);  empty_32 = None
    bernoulli_32: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_129, 0.9);  permute_129 = None
    sub_85: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_32);  bernoulli_32 = None
    convert_element_type_43: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_85, torch.bool);  sub_85 = None
    scalar_tensor_43: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_54: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_43, scalar_tensor_43, view_193);  scalar_tensor_43 = view_193 = None
    mul_96: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_54, 1.1111111111111112);  where_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_95: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_96, add_92);  mul_96 = add_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_42: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_95, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_86: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_95, mean_42)
    pow_22: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_86, 2)
    mean_43: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_22, [-1], True);  pow_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_87: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_95, mean_42);  add_95 = mean_42 = None
    add_96: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_43, 1e-07);  mean_43 = None
    sqrt_32: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_96);  add_96 = None
    alias_32: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_32)
    div_43: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_87, sqrt_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_97: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_65, div_43)
    add_97: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_97, primals_66);  mul_97 = primals_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_194: "f32[512, 768]" = torch.ops.aten.view.default(add_97, [512, 768])
    permute_130: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_150, [1, 0]);  primals_150 = None
    addmm_31: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_151, view_194, permute_130);  primals_151 = None
    view_195: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_31, [1, 512, 3072]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_98: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, 0.5)
    mul_99: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476)
    erf_10: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_99);  mul_99 = None
    add_98: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_100: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_98, add_98);  mul_98 = add_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_196: "f32[512, 3072]" = torch.ops.aten.view.default(mul_100, [512, 3072]);  mul_100 = None
    permute_131: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_152, [1, 0]);  primals_152 = None
    addmm_32: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_153, view_196, permute_131);  primals_153 = None
    view_197: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_32, [1, 512, 768]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_33: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    permute_132: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_33, [0, 1, 2]);  empty_33 = None
    bernoulli_33: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_132, 0.9);  permute_132 = None
    sub_88: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_33);  bernoulli_33 = None
    convert_element_type_44: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_88, torch.bool);  sub_88 = None
    scalar_tensor_44: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_55: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_44, scalar_tensor_44, view_197);  scalar_tensor_44 = view_197 = None
    mul_101: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_55, 1.1111111111111112);  where_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_99: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_101, add_97);  mul_101 = add_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_44: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_99, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_89: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_99, mean_44)
    pow_23: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_89, 2)
    mean_45: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_23, [-1], True);  pow_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_90: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_99, mean_44);  add_99 = mean_44 = None
    add_100: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_45, 1e-07);  mean_45 = None
    sqrt_33: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_100);  add_100 = None
    alias_33: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_33)
    div_44: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_90, sqrt_33)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_102: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_67, div_44)
    add_101: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_102, primals_68);  mul_102 = primals_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_133: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_154, [1, 0]);  primals_154 = None
    view_198: "f32[512, 768]" = torch.ops.aten.view.default(add_101, [512, 768])
    mm_11: "f32[512, 2304]" = torch.ops.aten.mm.default(view_198, permute_133)
    view_199: "f32[1, 512, 2304]" = torch.ops.aten.view.default(mm_11, [1, 512, 2304]);  mm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_200: "f32[1, 512, 12, 192]" = torch.ops.aten.view.default(view_199, [1, 512, 12, -1]);  view_199 = None
    
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
    slice_24: "f32[1, 1, 768]" = torch.ops.aten.slice.Tensor(unsqueeze_49, 2, 0, 9223372036854775807);  unsqueeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_201: "f32[1, 1, 12, 64]" = torch.ops.aten.view.default(slice_24, [1, 1, 12, -1]);  slice_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_135: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_201, [0, 2, 1, 3]);  view_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    add_102: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_33, permute_135);  getitem_33 = permute_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    unsqueeze_50: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(primals_70, 0);  primals_70 = None
    unsqueeze_51: "f32[1, 1, 768]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, 1);  unsqueeze_50 = None
    slice_25: "f32[1, 1, 768]" = torch.ops.aten.slice.Tensor(unsqueeze_51, 2, 0, 9223372036854775807);  unsqueeze_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_202: "f32[1, 1, 12, 64]" = torch.ops.aten.view.default(slice_25, [1, 1, 12, -1]);  slice_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_136: "f32[1, 12, 1, 64]" = torch.ops.aten.permute.default(view_202, [0, 2, 1, 3]);  view_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    add_103: "f32[1, 12, 512, 64]" = torch.ops.aten.add.Tensor(getitem_35, permute_136);  getitem_35 = permute_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    _tensor_constant22 = self._tensor_constant22
    lift_fresh_copy_22: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant22);  _tensor_constant22 = None
    mul_103: "f32[]" = torch.ops.aten.mul.Tensor(lift_fresh_copy_22, 1);  lift_fresh_copy_22 = None
    sqrt_34: "f32[]" = torch.ops.aten.sqrt.default(mul_103);  mul_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_45: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(add_102, sqrt_34);  add_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    permute_137: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(getitem_34, [0, 1, 3, 2]);  getitem_34 = None
    expand_44: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(div_45, [1, 12, 512, 64]);  div_45 = None
    view_203: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_44, [12, 512, 64]);  expand_44 = None
    expand_45: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(permute_137, [1, 12, 64, 512]);  permute_137 = None
    view_204: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_45, [12, 64, 512]);  expand_45 = None
    bmm_22: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_203, view_204)
    view_205: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_22, [1, 12, 512, 512]);  bmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    convert_element_type_45: "b8[1, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bool);  mul_3 = None
    bitwise_not_11: "b8[1, 1, 512, 512]" = torch.ops.aten.bitwise_not.default(convert_element_type_45);  convert_element_type_45 = None
    _tensor_constant23 = self._tensor_constant23
    lift_fresh_copy_23: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant23);  _tensor_constant23 = None
    where_56: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(bitwise_not_11, lift_fresh_copy_23, view_205);  lift_fresh_copy_23 = view_205 = None
    amax_11: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(where_56, [-1], True)
    sub_91: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(where_56, amax_11);  where_56 = amax_11 = None
    exp_11: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(sub_91);  sub_91 = None
    sum_12: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_46: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    scalar_tensor_45: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_57: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(bitwise_not_11, scalar_tensor_45, div_46);  bitwise_not_11 = scalar_tensor_45 = div_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_34: "f32[1, 12, 512, 512]" = torch.ops.aten.empty.memory_format([1, 12, 512, 512], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    permute_138: "f32[1, 12, 512, 512]" = torch.ops.aten.permute.default(empty_34, [0, 1, 2, 3]);  empty_34 = None
    bernoulli_34: "f32[1, 12, 512, 512]" = torch.ops.aten.bernoulli.p(permute_138, 0.9);  permute_138 = None
    sub_92: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_34);  bernoulli_34 = None
    convert_element_type_46: "b8[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_92, torch.bool);  sub_92 = None
    scalar_tensor_46: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_58: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_46, scalar_tensor_46, where_57);  scalar_tensor_46 = None
    mul_104: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_58, 1.1111111111111112);  where_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    expand_46: "f32[1, 12, 512, 512]" = torch.ops.aten.expand.default(mul_104, [1, 12, 512, 512]);  mul_104 = None
    view_206: "f32[12, 512, 512]" = torch.ops.aten.view.default(expand_46, [12, 512, 512]);  expand_46 = None
    expand_47: "f32[1, 12, 512, 64]" = torch.ops.aten.expand.default(add_103, [1, 12, 512, 64]);  add_103 = None
    view_207: "f32[12, 512, 64]" = torch.ops.aten.view.default(expand_47, [12, 512, 64]);  expand_47 = None
    bmm_23: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_206, view_207)
    view_208: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_23, [1, 12, 512, 64]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_139: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None
    clone_11: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_209: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_11, [1, 512, -1]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_210: "f32[512, 768]" = torch.ops.aten.view.default(view_209, [512, 768]);  view_209 = None
    permute_140: "f32[768, 768]" = torch.ops.aten.permute.default(primals_155, [1, 0]);  primals_155 = None
    addmm_33: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_156, view_210, permute_140);  primals_156 = None
    view_211: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_33, [1, 512, 768]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_35: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    permute_141: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_35, [0, 1, 2]);  empty_35 = None
    bernoulli_35: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_141, 0.9);  permute_141 = None
    sub_93: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_35);  bernoulli_35 = None
    convert_element_type_47: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_93, torch.bool);  sub_93 = None
    scalar_tensor_47: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_59: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_47, scalar_tensor_47, view_211);  scalar_tensor_47 = view_211 = None
    mul_105: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_59, 1.1111111111111112);  where_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:296, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_104: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_105, add_101);  mul_105 = add_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_46: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_104, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_94: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_104, mean_46)
    pow_24: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_94, 2)
    mean_47: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_24, [-1], True);  pow_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_95: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_104, mean_46);  add_104 = mean_46 = None
    add_105: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_47, 1e-07);  mean_47 = None
    sqrt_35: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_105);  add_105 = None
    alias_35: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_35)
    div_47: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_95, sqrt_35)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_106: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_71, div_47)
    add_106: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_106, primals_72);  mul_106 = primals_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_212: "f32[512, 768]" = torch.ops.aten.view.default(add_106, [512, 768])
    permute_142: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_157, [1, 0]);  primals_157 = None
    addmm_34: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_158, view_212, permute_142);  primals_158 = None
    view_213: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_34, [1, 512, 3072]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_107: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_213, 0.5)
    mul_108: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_213, 0.7071067811865476)
    erf_11: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_108);  mul_108 = None
    add_107: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_109: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_107, add_107);  mul_107 = add_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_214: "f32[512, 3072]" = torch.ops.aten.view.default(mul_109, [512, 3072]);  mul_109 = None
    permute_143: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_159, [1, 0]);  primals_159 = None
    addmm_35: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_160, view_214, permute_143);  primals_160 = None
    view_215: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_35, [1, 512, 768]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_36: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    permute_144: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_36, [0, 1, 2]);  empty_36 = None
    bernoulli_36: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_144, 0.9);  permute_144 = None
    sub_96: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_36);  bernoulli_36 = None
    convert_element_type_48: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_96, torch.bool);  sub_96 = None
    scalar_tensor_48: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_60: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_48, scalar_tensor_48, view_215);  scalar_tensor_48 = view_215 = None
    mul_110: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_60, 1.1111111111111112);  where_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:363, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    add_108: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_110, add_106);  mul_110 = add_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    mean_48: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(add_108, [-1], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    sub_97: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_108, mean_48)
    pow_25: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_97, 2)
    mean_49: "f32[1, 512, 1]" = torch.ops.aten.mean.dim(pow_25, [-1], True);  pow_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    sub_98: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_108, mean_48);  add_108 = mean_48 = None
    add_109: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(mean_49, 1e-07);  mean_49 = None
    sqrt_36: "f32[1, 512, 1]" = torch.ops.aten.sqrt.default(add_109);  add_109 = None
    alias_36: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_36)
    div_48: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_98, sqrt_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    mul_111: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(primals_73, div_48)
    add_110: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_111, primals_74);  mul_111 = primals_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1411, code: logits = self.qa_outputs(sequence_output)
    view_216: "f32[512, 768]" = torch.ops.aten.view.default(add_110, [512, 768]);  add_110 = None
    permute_145: "f32[768, 2]" = torch.ops.aten.permute.default(primals_161, [1, 0]);  primals_161 = None
    addmm_36: "f32[512, 2]" = torch.ops.aten.addmm.default(primals_162, view_216, permute_145);  primals_162 = None
    view_217: "f32[1, 512, 2]" = torch.ops.aten.view.default(addmm_36, [1, 512, 2]);  addmm_36 = None
    
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
    alias_37: "f32[1, 512]" = torch.ops.aten.alias.default(sub_100)
    ne: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max, 512)
    scalar_tensor_49: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'))
    where_61: "i64[1]" = torch.ops.aten.where.self(ne, clamp_max, scalar_tensor_49);  ne = scalar_tensor_49 = None
    unsqueeze_52: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(where_61, 1);  where_61 = None
    gather: "f32[1, 1]" = torch.ops.aten.gather.default(sub_100, 1, unsqueeze_52);  sub_100 = unsqueeze_52 = None
    squeeze_3: "f32[1]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[1]" = torch.ops.aten.neg.default(squeeze_3);  squeeze_3 = None
    ne_1: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max, 512)
    scalar_tensor_50: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_62: "f32[1]" = torch.ops.aten.where.self(ne_1, neg, scalar_tensor_50);  ne_1 = neg = scalar_tensor_50 = None
    ne_2: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max, 512)
    sum_14: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
    convert_element_type_49: "f32[]" = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None
    sum_15: "f32[]" = torch.ops.aten.sum.default(where_62);  where_62 = None
    div_49: "f32[]" = torch.ops.aten.div.Tensor(sum_15, convert_element_type_49);  sum_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1430, code: end_loss = loss_fct(end_logits, end_positions)
    amax_13: "f32[1, 1]" = torch.ops.aten.amax.default(clone_13, [1], True)
    sub_101: "f32[1, 512]" = torch.ops.aten.sub.Tensor(clone_13, amax_13);  amax_13 = None
    exp_13: "f32[1, 512]" = torch.ops.aten.exp.default(sub_101)
    sum_16: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [1], True);  exp_13 = None
    log_1: "f32[1, 1]" = torch.ops.aten.log.default(sum_16);  sum_16 = None
    sub_102: "f32[1, 512]" = torch.ops.aten.sub.Tensor(sub_101, log_1);  sub_101 = log_1 = None
    alias_38: "f32[1, 512]" = torch.ops.aten.alias.default(sub_102)
    ne_3: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max_1, 512)
    scalar_tensor_51: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'))
    where_63: "i64[1]" = torch.ops.aten.where.self(ne_3, clamp_max_1, scalar_tensor_51);  ne_3 = scalar_tensor_51 = None
    unsqueeze_53: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(where_63, 1);  where_63 = None
    gather_1: "f32[1, 1]" = torch.ops.aten.gather.default(sub_102, 1, unsqueeze_53);  sub_102 = unsqueeze_53 = None
    squeeze_4: "f32[1]" = torch.ops.aten.squeeze.dim(gather_1, 1);  gather_1 = None
    neg_1: "f32[1]" = torch.ops.aten.neg.default(squeeze_4);  squeeze_4 = None
    ne_4: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max_1, 512)
    scalar_tensor_52: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_64: "f32[1]" = torch.ops.aten.where.self(ne_4, neg_1, scalar_tensor_52);  ne_4 = neg_1 = scalar_tensor_52 = None
    ne_5: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max_1, 512)
    sum_17: "i64[]" = torch.ops.aten.sum.default(ne_5);  ne_5 = None
    convert_element_type_50: "f32[]" = torch.ops.prims.convert_element_type.default(sum_17, torch.float32);  sum_17 = None
    sum_18: "f32[]" = torch.ops.aten.sum.default(where_64);  where_64 = None
    div_50: "f32[]" = torch.ops.aten.div.Tensor(sum_18, convert_element_type_50);  sum_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1431, code: total_loss = (start_loss + end_loss) / 2
    add_111: "f32[]" = torch.ops.aten.add.Tensor(div_49, div_50);  div_49 = div_50 = None
    div_51: "f32[]" = torch.ops.aten.div.Tensor(add_111, 2);  add_111 = None
    div_52: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, 2);  tangents_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1430, code: end_loss = loss_fct(end_logits, end_positions)
    div_53: "f32[]" = torch.ops.aten.div.Tensor(div_52, convert_element_type_50);  convert_element_type_50 = None
    unsqueeze_54: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(clamp_max_1, 1);  clamp_max_1 = None
    ne_6: "b8[1, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_54, 512)
    scalar_tensor_53: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'))
    where_65: "i64[1, 1]" = torch.ops.aten.where.self(ne_6, unsqueeze_54, scalar_tensor_53);  ne_6 = scalar_tensor_53 = None
    full_2: "f32[1, 512]" = torch.ops.aten.full.default([1, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    scatter: "f32[1, 512]" = torch.ops.aten.scatter.value(full_2, 1, where_65, -1.0);  full_2 = where_65 = None
    ne_7: "b8[1, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_54, 512);  unsqueeze_54 = None
    scalar_tensor_54: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_66: "f32[1, 1]" = torch.ops.aten.where.self(ne_7, div_53, scalar_tensor_54);  ne_7 = div_53 = scalar_tensor_54 = None
    mul_112: "f32[1, 512]" = torch.ops.aten.mul.Tensor(scatter, where_66);  scatter = where_66 = None
    alias_39: "f32[1, 512]" = torch.ops.aten.alias.default(alias_38);  alias_38 = None
    exp_14: "f32[1, 512]" = torch.ops.aten.exp.default(alias_39);  alias_39 = None
    sum_19: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(mul_112, [1], True)
    mul_113: "f32[1, 512]" = torch.ops.aten.mul.Tensor(exp_14, sum_19);  exp_14 = sum_19 = None
    sub_103: "f32[1, 512]" = torch.ops.aten.sub.Tensor(mul_112, mul_113);  mul_112 = mul_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1430, code: end_loss = loss_fct(end_logits, end_positions)
    add_112: "f32[1, 512]" = torch.ops.aten.add.Tensor(tangents_3, sub_103);  tangents_3 = sub_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1429, code: start_loss = loss_fct(start_logits, start_positions)
    div_54: "f32[]" = torch.ops.aten.div.Tensor(div_52, convert_element_type_49);  div_52 = convert_element_type_49 = None
    unsqueeze_55: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(clamp_max, 1);  clamp_max = None
    ne_8: "b8[1, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_55, 512)
    scalar_tensor_55: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'))
    where_67: "i64[1, 1]" = torch.ops.aten.where.self(ne_8, unsqueeze_55, scalar_tensor_55);  ne_8 = scalar_tensor_55 = None
    full_3: "f32[1, 512]" = torch.ops.aten.full.default([1, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    scatter_1: "f32[1, 512]" = torch.ops.aten.scatter.value(full_3, 1, where_67, -1.0);  full_3 = where_67 = None
    ne_9: "b8[1, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_55, 512);  unsqueeze_55 = None
    scalar_tensor_56: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_68: "f32[1, 1]" = torch.ops.aten.where.self(ne_9, div_54, scalar_tensor_56);  ne_9 = div_54 = scalar_tensor_56 = None
    mul_114: "f32[1, 512]" = torch.ops.aten.mul.Tensor(scatter_1, where_68);  scatter_1 = where_68 = None
    alias_40: "f32[1, 512]" = torch.ops.aten.alias.default(alias_37);  alias_37 = None
    exp_15: "f32[1, 512]" = torch.ops.aten.exp.default(alias_40);  alias_40 = None
    sum_20: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(mul_114, [1], True)
    mul_115: "f32[1, 512]" = torch.ops.aten.mul.Tensor(exp_15, sum_20);  exp_15 = sum_20 = None
    sub_104: "f32[1, 512]" = torch.ops.aten.sub.Tensor(mul_114, mul_115);  mul_114 = mul_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1429, code: start_loss = loss_fct(start_logits, start_positions)
    add_113: "f32[1, 512]" = torch.ops.aten.add.Tensor(tangents_2, sub_104);  tangents_2 = sub_104 = None
    
    # No stacktrace found for following nodes
    unsqueeze_56: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(add_112, 2);  add_112 = None
    unsqueeze_57: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(add_113, 2);  add_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1412, code: start_logits, end_logits = logits.split(1, dim=-1)
    cat: "f32[1, 512, 2]" = torch.ops.aten.cat.default([unsqueeze_57, unsqueeze_56], 2);  unsqueeze_57 = unsqueeze_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1411, code: logits = self.qa_outputs(sequence_output)
    view_218: "f32[512, 2]" = torch.ops.aten.view.default(cat, [512, 2]);  cat = None
    permute_146: "f32[2, 768]" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
    mm_12: "f32[512, 768]" = torch.ops.aten.mm.default(view_218, permute_146);  permute_146 = None
    permute_147: "f32[2, 512]" = torch.ops.aten.permute.default(view_218, [1, 0])
    mm_13: "f32[2, 768]" = torch.ops.aten.mm.default(permute_147, view_216);  permute_147 = view_216 = None
    permute_148: "f32[768, 2]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_21: "f32[1, 2]" = torch.ops.aten.sum.dim_IntList(view_218, [0], True);  view_218 = None
    view_219: "f32[2]" = torch.ops.aten.view.default(sum_21, [2]);  sum_21 = None
    permute_149: "f32[2, 768]" = torch.ops.aten.permute.default(permute_148, [1, 0]);  permute_148 = None
    view_220: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_12, [1, 512, 768]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_22: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_220, [0, 1], True)
    view_221: "f32[768]" = torch.ops.aten.view.default(sum_22, [768]);  sum_22 = None
    mul_116: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_220, primals_73);  primals_73 = None
    mul_117: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_220, div_48);  view_220 = div_48 = None
    sum_23: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_117, [0, 1], True);  mul_117 = None
    view_222: "f32[768]" = torch.ops.aten.view.default(sum_23, [768]);  sum_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_55: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_98, sqrt_36);  sub_98 = None
    div_56: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_55, sqrt_36);  div_55 = None
    neg_2: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_116)
    mul_118: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_2, div_56);  neg_2 = div_56 = None
    div_57: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_116, sqrt_36);  mul_116 = sqrt_36 = None
    sum_24: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_118, [2], True);  mul_118 = None
    alias_41: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_36);  alias_36 = None
    mul_119: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_41, 2);  alias_41 = None
    div_58: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_24, mul_119);  sum_24 = mul_119 = None
    neg_3: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_57)
    sum_25: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_3, [2], True);  neg_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_48: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_58, [1, 512, 768]);  div_58 = None
    div_59: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_48, 768);  expand_48 = None
    pow_26: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_97, 1.0);  sub_97 = None
    mul_120: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_26, 2.0);  pow_26 = None
    mul_121: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_59, mul_120);  div_59 = mul_120 = None
    neg_4: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_121)
    sum_26: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_4, [2], True);  neg_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_114: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_57, mul_121);  div_57 = mul_121 = None
    add_115: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_25, sum_26);  sum_25 = sum_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_49: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_115, [1, 512, 768]);  add_115 = None
    div_60: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_49, 768);  expand_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_116: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_114, div_60);  add_114 = div_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_57: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_69: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_48, scalar_tensor_57, add_116);  convert_element_type_48 = scalar_tensor_57 = None
    mul_122: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_69, 1.1111111111111112);  where_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_223: "f32[512, 768]" = torch.ops.aten.view.default(mul_122, [512, 768]);  mul_122 = None
    permute_150: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
    mm_14: "f32[512, 3072]" = torch.ops.aten.mm.default(view_223, permute_150);  permute_150 = None
    permute_151: "f32[768, 512]" = torch.ops.aten.permute.default(view_223, [1, 0])
    mm_15: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_151, view_214);  permute_151 = view_214 = None
    permute_152: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_27: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_223, [0], True);  view_223 = None
    view_224: "f32[768]" = torch.ops.aten.view.default(sum_27, [768]);  sum_27 = None
    permute_153: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
    view_225: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_14, [1, 512, 3072]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_123: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_213, 0.7071067811865476)
    erf_12: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_123);  mul_123 = None
    add_117: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_124: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_117, 0.5);  add_117 = None
    mul_125: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_213, view_213)
    mul_126: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_125, -0.5);  mul_125 = None
    exp_16: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_126);  mul_126 = None
    mul_127: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
    mul_128: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_213, mul_127);  view_213 = mul_127 = None
    add_118: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_124, mul_128);  mul_124 = mul_128 = None
    mul_129: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_225, add_118);  view_225 = add_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_226: "f32[512, 3072]" = torch.ops.aten.view.default(mul_129, [512, 3072]);  mul_129 = None
    permute_154: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_142, [1, 0]);  permute_142 = None
    mm_16: "f32[512, 768]" = torch.ops.aten.mm.default(view_226, permute_154);  permute_154 = None
    permute_155: "f32[3072, 512]" = torch.ops.aten.permute.default(view_226, [1, 0])
    mm_17: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_155, view_212);  permute_155 = view_212 = None
    permute_156: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_28: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_226, [0], True);  view_226 = None
    view_227: "f32[3072]" = torch.ops.aten.view.default(sum_28, [3072]);  sum_28 = None
    permute_157: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None
    view_228: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_16, [1, 512, 768]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    add_119: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_116, view_228);  add_116 = view_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_29: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_119, [0, 1], True)
    view_229: "f32[768]" = torch.ops.aten.view.default(sum_29, [768]);  sum_29 = None
    mul_130: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_119, primals_71);  primals_71 = None
    mul_131: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_119, div_47);  add_119 = div_47 = None
    sum_30: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_131, [0, 1], True);  mul_131 = None
    view_230: "f32[768]" = torch.ops.aten.view.default(sum_30, [768]);  sum_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_61: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_95, sqrt_35);  sub_95 = None
    div_62: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_61, sqrt_35);  div_61 = None
    neg_5: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_130)
    mul_132: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_5, div_62);  neg_5 = div_62 = None
    div_63: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_130, sqrt_35);  mul_130 = sqrt_35 = None
    sum_31: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_132, [2], True);  mul_132 = None
    alias_42: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_35);  alias_35 = None
    mul_133: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_42, 2);  alias_42 = None
    div_64: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_31, mul_133);  sum_31 = mul_133 = None
    neg_6: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_63)
    sum_32: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_6, [2], True);  neg_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_50: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_64, [1, 512, 768]);  div_64 = None
    div_65: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_50, 768);  expand_50 = None
    pow_27: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_94, 1.0);  sub_94 = None
    mul_134: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_27, 2.0);  pow_27 = None
    mul_135: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_65, mul_134);  div_65 = mul_134 = None
    neg_7: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_135)
    sum_33: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_7, [2], True);  neg_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_120: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_63, mul_135);  div_63 = mul_135 = None
    add_121: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_32, sum_33);  sum_32 = sum_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_51: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_121, [1, 512, 768]);  add_121 = None
    div_66: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_51, 768);  expand_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_122: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_120, div_66);  add_120 = div_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_58: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_70: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_47, scalar_tensor_58, add_122);  convert_element_type_47 = scalar_tensor_58 = None
    mul_136: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_70, 1.1111111111111112);  where_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_231: "f32[512, 768]" = torch.ops.aten.view.default(mul_136, [512, 768]);  mul_136 = None
    permute_158: "f32[768, 768]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
    mm_18: "f32[512, 768]" = torch.ops.aten.mm.default(view_231, permute_158);  permute_158 = None
    permute_159: "f32[768, 512]" = torch.ops.aten.permute.default(view_231, [1, 0])
    mm_19: "f32[768, 768]" = torch.ops.aten.mm.default(permute_159, view_210);  permute_159 = view_210 = None
    permute_160: "f32[768, 768]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_34: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_231, [0], True);  view_231 = None
    view_232: "f32[768]" = torch.ops.aten.view.default(sum_34, [768]);  sum_34 = None
    permute_161: "f32[768, 768]" = torch.ops.aten.permute.default(permute_160, [1, 0]);  permute_160 = None
    view_233: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_18, [1, 512, 768]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_234: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_233, [1, 512, 12, 64]);  view_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_162: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_234, [0, 2, 1, 3]);  view_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_235: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_162, [12, 512, 64]);  permute_162 = None
    permute_163: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_206, [0, 2, 1]);  view_206 = None
    bmm_24: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_163, view_235);  permute_163 = None
    permute_164: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_207, [0, 2, 1]);  view_207 = None
    bmm_25: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_235, permute_164);  view_235 = permute_164 = None
    view_236: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_24, [1, 12, 512, 64]);  bmm_24 = None
    view_237: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_25, [1, 12, 512, 512]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_59: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_71: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_46, scalar_tensor_59, view_237);  convert_element_type_46 = scalar_tensor_59 = view_237 = None
    mul_137: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_71, 1.1111111111111112);  where_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_44: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(where_57);  where_57 = None
    alias_45: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_44);  alias_44 = None
    mul_138: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_137, alias_45);  mul_137 = None
    sum_35: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_138, [-1], True)
    mul_139: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_45, sum_35);  alias_45 = sum_35 = None
    sub_105: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_138, mul_139);  mul_138 = mul_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_238: "f32[12, 512, 512]" = torch.ops.aten.view.default(sub_105, [12, 512, 512]);  sub_105 = None
    permute_165: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_203, [0, 2, 1]);  view_203 = None
    bmm_26: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_165, view_238);  permute_165 = None
    permute_166: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_204, [0, 2, 1]);  view_204 = None
    bmm_27: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_238, permute_166);  view_238 = permute_166 = None
    view_239: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_26, [1, 12, 64, 512]);  bmm_26 = None
    view_240: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_27, [1, 12, 512, 64]);  bmm_27 = None
    permute_167: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_239, [0, 1, 3, 2]);  view_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_67: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(view_240, sqrt_34);  view_240 = sqrt_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    sum_36: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(view_236, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_168: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_36, [0, 2, 1, 3]);  sum_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_241: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_168, [1, 1, 768]);  permute_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    full_4: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_4, view_241, 2, 0, 9223372036854775807);  full_4 = view_241 = None
    squeeze_5: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter, 1);  slice_scatter = None
    squeeze_6: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_5, 0);  squeeze_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    sum_37: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(div_67, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_169: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_37, [0, 2, 1, 3]);  sum_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_242: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_169, [1, 1, 768]);  permute_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    full_5: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_1: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_5, view_242, 2, 0, 9223372036854775807);  full_5 = view_242 = None
    squeeze_7: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_1, 1);  slice_scatter_1 = None
    squeeze_8: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_7, 0);  squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    cat_1: "f32[1, 12, 512, 192]" = torch.ops.aten.cat.default([div_67, permute_167, view_236], 3);  div_67 = permute_167 = view_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_170: "f32[1, 512, 12, 192]" = torch.ops.aten.permute.default(cat_1, [0, 2, 1, 3]);  cat_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    clone_14: "f32[1, 512, 12, 192]" = torch.ops.aten.clone.default(permute_170, memory_format = torch.contiguous_format);  permute_170 = None
    view_243: "f32[1, 512, 2304]" = torch.ops.aten.view.default(clone_14, [1, 512, 2304]);  clone_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_244: "f32[512, 2304]" = torch.ops.aten.view.default(view_243, [512, 2304]);  view_243 = None
    permute_171: "f32[2304, 512]" = torch.ops.aten.permute.default(view_244, [1, 0])
    mm_20: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_171, view_198);  permute_171 = view_198 = None
    permute_172: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_20, [1, 0]);  mm_20 = None
    permute_173: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
    mm_21: "f32[512, 768]" = torch.ops.aten.mm.default(view_244, permute_173);  view_244 = permute_173 = None
    view_245: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_21, [1, 512, 768]);  mm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    add_123: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_122, view_245);  add_122 = view_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_174: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_38: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_123, [0, 1], True)
    view_246: "f32[768]" = torch.ops.aten.view.default(sum_38, [768]);  sum_38 = None
    mul_140: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_123, primals_67);  primals_67 = None
    mul_141: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_123, div_44);  add_123 = div_44 = None
    sum_39: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_141, [0, 1], True);  mul_141 = None
    view_247: "f32[768]" = torch.ops.aten.view.default(sum_39, [768]);  sum_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_68: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_90, sqrt_33);  sub_90 = None
    div_69: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_68, sqrt_33);  div_68 = None
    neg_8: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_140)
    mul_142: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_8, div_69);  neg_8 = div_69 = None
    div_70: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_140, sqrt_33);  mul_140 = sqrt_33 = None
    sum_40: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_142, [2], True);  mul_142 = None
    alias_46: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_33);  alias_33 = None
    mul_143: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_46, 2);  alias_46 = None
    div_71: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_40, mul_143);  sum_40 = mul_143 = None
    neg_9: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_70)
    sum_41: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_9, [2], True);  neg_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_52: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_71, [1, 512, 768]);  div_71 = None
    div_72: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_52, 768);  expand_52 = None
    pow_28: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_89, 1.0);  sub_89 = None
    mul_144: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_28, 2.0);  pow_28 = None
    mul_145: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_72, mul_144);  div_72 = mul_144 = None
    neg_10: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_145)
    sum_42: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_10, [2], True);  neg_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_124: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_70, mul_145);  div_70 = mul_145 = None
    add_125: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_41, sum_42);  sum_41 = sum_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_53: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_125, [1, 512, 768]);  add_125 = None
    div_73: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_53, 768);  expand_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_126: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_124, div_73);  add_124 = div_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_60: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_72: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_44, scalar_tensor_60, add_126);  convert_element_type_44 = scalar_tensor_60 = None
    mul_146: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_72, 1.1111111111111112);  where_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_248: "f32[512, 768]" = torch.ops.aten.view.default(mul_146, [512, 768]);  mul_146 = None
    permute_175: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
    mm_22: "f32[512, 3072]" = torch.ops.aten.mm.default(view_248, permute_175);  permute_175 = None
    permute_176: "f32[768, 512]" = torch.ops.aten.permute.default(view_248, [1, 0])
    mm_23: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_176, view_196);  permute_176 = view_196 = None
    permute_177: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_43: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_248, [0], True);  view_248 = None
    view_249: "f32[768]" = torch.ops.aten.view.default(sum_43, [768]);  sum_43 = None
    permute_178: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_177, [1, 0]);  permute_177 = None
    view_250: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_22, [1, 512, 3072]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_147: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476)
    erf_13: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_147);  mul_147 = None
    add_127: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_148: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_127, 0.5);  add_127 = None
    mul_149: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, view_195)
    mul_150: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_149, -0.5);  mul_149 = None
    exp_17: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_150);  mul_150 = None
    mul_151: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_17, 0.3989422804014327);  exp_17 = None
    mul_152: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, mul_151);  view_195 = mul_151 = None
    add_128: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_148, mul_152);  mul_148 = mul_152 = None
    mul_153: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_250, add_128);  view_250 = add_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_251: "f32[512, 3072]" = torch.ops.aten.view.default(mul_153, [512, 3072]);  mul_153 = None
    permute_179: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
    mm_24: "f32[512, 768]" = torch.ops.aten.mm.default(view_251, permute_179);  permute_179 = None
    permute_180: "f32[3072, 512]" = torch.ops.aten.permute.default(view_251, [1, 0])
    mm_25: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_180, view_194);  permute_180 = view_194 = None
    permute_181: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_44: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_251, [0], True);  view_251 = None
    view_252: "f32[3072]" = torch.ops.aten.view.default(sum_44, [3072]);  sum_44 = None
    permute_182: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None
    view_253: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_24, [1, 512, 768]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    add_129: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_126, view_253);  add_126 = view_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_45: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_129, [0, 1], True)
    view_254: "f32[768]" = torch.ops.aten.view.default(sum_45, [768]);  sum_45 = None
    mul_154: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_129, primals_65);  primals_65 = None
    mul_155: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_129, div_43);  add_129 = div_43 = None
    sum_46: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_155, [0, 1], True);  mul_155 = None
    view_255: "f32[768]" = torch.ops.aten.view.default(sum_46, [768]);  sum_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_74: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_87, sqrt_32);  sub_87 = None
    div_75: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_74, sqrt_32);  div_74 = None
    neg_11: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_154)
    mul_156: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_11, div_75);  neg_11 = div_75 = None
    div_76: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_154, sqrt_32);  mul_154 = sqrt_32 = None
    sum_47: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_156, [2], True);  mul_156 = None
    alias_47: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_32);  alias_32 = None
    mul_157: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_47, 2);  alias_47 = None
    div_77: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_47, mul_157);  sum_47 = mul_157 = None
    neg_12: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_76)
    sum_48: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_12, [2], True);  neg_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_54: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_77, [1, 512, 768]);  div_77 = None
    div_78: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_54, 768);  expand_54 = None
    pow_29: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_86, 1.0);  sub_86 = None
    mul_158: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_29, 2.0);  pow_29 = None
    mul_159: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_78, mul_158);  div_78 = mul_158 = None
    neg_13: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_159)
    sum_49: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_13, [2], True);  neg_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_130: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_76, mul_159);  div_76 = mul_159 = None
    add_131: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_48, sum_49);  sum_48 = sum_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_55: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_131, [1, 512, 768]);  add_131 = None
    div_79: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_55, 768);  expand_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_132: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_130, div_79);  add_130 = div_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_61: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_73: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_43, scalar_tensor_61, add_132);  convert_element_type_43 = scalar_tensor_61 = None
    mul_160: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_73, 1.1111111111111112);  where_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_256: "f32[512, 768]" = torch.ops.aten.view.default(mul_160, [512, 768]);  mul_160 = None
    permute_183: "f32[768, 768]" = torch.ops.aten.permute.default(permute_128, [1, 0]);  permute_128 = None
    mm_26: "f32[512, 768]" = torch.ops.aten.mm.default(view_256, permute_183);  permute_183 = None
    permute_184: "f32[768, 512]" = torch.ops.aten.permute.default(view_256, [1, 0])
    mm_27: "f32[768, 768]" = torch.ops.aten.mm.default(permute_184, view_192);  permute_184 = view_192 = None
    permute_185: "f32[768, 768]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_50: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_256, [0], True);  view_256 = None
    view_257: "f32[768]" = torch.ops.aten.view.default(sum_50, [768]);  sum_50 = None
    permute_186: "f32[768, 768]" = torch.ops.aten.permute.default(permute_185, [1, 0]);  permute_185 = None
    view_258: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_26, [1, 512, 768]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_259: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_258, [1, 512, 12, 64]);  view_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_187: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_259, [0, 2, 1, 3]);  view_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_260: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_187, [12, 512, 64]);  permute_187 = None
    permute_188: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None
    bmm_28: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_188, view_260);  permute_188 = None
    permute_189: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_189, [0, 2, 1]);  view_189 = None
    bmm_29: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_260, permute_189);  view_260 = permute_189 = None
    view_261: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_28, [1, 12, 512, 64]);  bmm_28 = None
    view_262: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_29, [1, 12, 512, 512]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_62: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_74: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_42, scalar_tensor_62, view_262);  convert_element_type_42 = scalar_tensor_62 = view_262 = None
    mul_161: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_74, 1.1111111111111112);  where_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_49: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(where_52);  where_52 = None
    alias_50: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_49);  alias_49 = None
    mul_162: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_161, alias_50);  mul_161 = None
    sum_51: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_162, [-1], True)
    mul_163: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_50, sum_51);  alias_50 = sum_51 = None
    sub_106: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_162, mul_163);  mul_162 = mul_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_263: "f32[12, 512, 512]" = torch.ops.aten.view.default(sub_106, [12, 512, 512]);  sub_106 = None
    permute_190: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_185, [0, 2, 1]);  view_185 = None
    bmm_30: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_190, view_263);  permute_190 = None
    permute_191: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1]);  view_186 = None
    bmm_31: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_263, permute_191);  view_263 = permute_191 = None
    view_264: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_30, [1, 12, 64, 512]);  bmm_30 = None
    view_265: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_31, [1, 12, 512, 64]);  bmm_31 = None
    permute_192: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_264, [0, 1, 3, 2]);  view_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_80: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(view_265, sqrt_31);  view_265 = sqrt_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    sum_52: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(view_261, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_193: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_52, [0, 2, 1, 3]);  sum_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_266: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_193, [1, 1, 768]);  permute_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    full_6: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_2: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_6, view_266, 2, 0, 9223372036854775807);  full_6 = view_266 = None
    squeeze_9: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_2, 1);  slice_scatter_2 = None
    squeeze_10: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_9, 0);  squeeze_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    sum_53: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(div_80, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_194: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_53, [0, 2, 1, 3]);  sum_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_267: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_194, [1, 1, 768]);  permute_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    full_7: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_3: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_7, view_267, 2, 0, 9223372036854775807);  full_7 = view_267 = None
    squeeze_11: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_3, 1);  slice_scatter_3 = None
    squeeze_12: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_11, 0);  squeeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    cat_2: "f32[1, 12, 512, 192]" = torch.ops.aten.cat.default([div_80, permute_192, view_261], 3);  div_80 = permute_192 = view_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_195: "f32[1, 512, 12, 192]" = torch.ops.aten.permute.default(cat_2, [0, 2, 1, 3]);  cat_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    clone_15: "f32[1, 512, 12, 192]" = torch.ops.aten.clone.default(permute_195, memory_format = torch.contiguous_format);  permute_195 = None
    view_268: "f32[1, 512, 2304]" = torch.ops.aten.view.default(clone_15, [1, 512, 2304]);  clone_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_269: "f32[512, 2304]" = torch.ops.aten.view.default(view_268, [512, 2304]);  view_268 = None
    permute_196: "f32[2304, 512]" = torch.ops.aten.permute.default(view_269, [1, 0])
    mm_28: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_196, view_180);  permute_196 = view_180 = None
    permute_197: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_28, [1, 0]);  mm_28 = None
    permute_198: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
    mm_29: "f32[512, 768]" = torch.ops.aten.mm.default(view_269, permute_198);  view_269 = permute_198 = None
    view_270: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_29, [1, 512, 768]);  mm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    add_133: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_132, view_270);  add_132 = view_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_199: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_197, [1, 0]);  permute_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_54: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_133, [0, 1], True)
    view_271: "f32[768]" = torch.ops.aten.view.default(sum_54, [768]);  sum_54 = None
    mul_164: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_133, primals_61);  primals_61 = None
    mul_165: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_133, div_40);  add_133 = div_40 = None
    sum_55: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_165, [0, 1], True);  mul_165 = None
    view_272: "f32[768]" = torch.ops.aten.view.default(sum_55, [768]);  sum_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_81: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_82, sqrt_30);  sub_82 = None
    div_82: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_81, sqrt_30);  div_81 = None
    neg_14: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_164)
    mul_166: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_14, div_82);  neg_14 = div_82 = None
    div_83: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_164, sqrt_30);  mul_164 = sqrt_30 = None
    sum_56: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_166, [2], True);  mul_166 = None
    alias_51: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_30);  alias_30 = None
    mul_167: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_51, 2);  alias_51 = None
    div_84: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_56, mul_167);  sum_56 = mul_167 = None
    neg_15: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_83)
    sum_57: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_15, [2], True);  neg_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_56: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_84, [1, 512, 768]);  div_84 = None
    div_85: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_56, 768);  expand_56 = None
    pow_30: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_81, 1.0);  sub_81 = None
    mul_168: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_30, 2.0);  pow_30 = None
    mul_169: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_85, mul_168);  div_85 = mul_168 = None
    neg_16: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_169)
    sum_58: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_16, [2], True);  neg_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_134: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_83, mul_169);  div_83 = mul_169 = None
    add_135: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_57, sum_58);  sum_57 = sum_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_57: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_135, [1, 512, 768]);  add_135 = None
    div_86: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_57, 768);  expand_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_136: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_134, div_86);  add_134 = div_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_63: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_75: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_40, scalar_tensor_63, add_136);  convert_element_type_40 = scalar_tensor_63 = None
    mul_170: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_75, 1.1111111111111112);  where_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_273: "f32[512, 768]" = torch.ops.aten.view.default(mul_170, [512, 768]);  mul_170 = None
    permute_200: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
    mm_30: "f32[512, 3072]" = torch.ops.aten.mm.default(view_273, permute_200);  permute_200 = None
    permute_201: "f32[768, 512]" = torch.ops.aten.permute.default(view_273, [1, 0])
    mm_31: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_201, view_178);  permute_201 = view_178 = None
    permute_202: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_59: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_273, [0], True);  view_273 = None
    view_274: "f32[768]" = torch.ops.aten.view.default(sum_59, [768]);  sum_59 = None
    permute_203: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_202, [1, 0]);  permute_202 = None
    view_275: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_30, [1, 512, 3072]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_171: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_177, 0.7071067811865476)
    erf_14: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_171);  mul_171 = None
    add_137: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_172: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_137, 0.5);  add_137 = None
    mul_173: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_177, view_177)
    mul_174: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_173, -0.5);  mul_173 = None
    exp_18: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_174);  mul_174 = None
    mul_175: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
    mul_176: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_177, mul_175);  view_177 = mul_175 = None
    add_138: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_172, mul_176);  mul_172 = mul_176 = None
    mul_177: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_275, add_138);  view_275 = add_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_276: "f32[512, 3072]" = torch.ops.aten.view.default(mul_177, [512, 3072]);  mul_177 = None
    permute_204: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
    mm_32: "f32[512, 768]" = torch.ops.aten.mm.default(view_276, permute_204);  permute_204 = None
    permute_205: "f32[3072, 512]" = torch.ops.aten.permute.default(view_276, [1, 0])
    mm_33: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_205, view_176);  permute_205 = view_176 = None
    permute_206: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_60: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_276, [0], True);  view_276 = None
    view_277: "f32[3072]" = torch.ops.aten.view.default(sum_60, [3072]);  sum_60 = None
    permute_207: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None
    view_278: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_32, [1, 512, 768]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    add_139: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_136, view_278);  add_136 = view_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_61: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_139, [0, 1], True)
    view_279: "f32[768]" = torch.ops.aten.view.default(sum_61, [768]);  sum_61 = None
    mul_178: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_139, primals_59);  primals_59 = None
    mul_179: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_139, div_39);  add_139 = div_39 = None
    sum_62: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_179, [0, 1], True);  mul_179 = None
    view_280: "f32[768]" = torch.ops.aten.view.default(sum_62, [768]);  sum_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_87: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_79, sqrt_29);  sub_79 = None
    div_88: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_87, sqrt_29);  div_87 = None
    neg_17: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_178)
    mul_180: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_17, div_88);  neg_17 = div_88 = None
    div_89: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_178, sqrt_29);  mul_178 = sqrt_29 = None
    sum_63: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_180, [2], True);  mul_180 = None
    alias_52: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_29);  alias_29 = None
    mul_181: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_52, 2);  alias_52 = None
    div_90: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_63, mul_181);  sum_63 = mul_181 = None
    neg_18: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_89)
    sum_64: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_18, [2], True);  neg_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_58: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_90, [1, 512, 768]);  div_90 = None
    div_91: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_58, 768);  expand_58 = None
    pow_31: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_78, 1.0);  sub_78 = None
    mul_182: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_31, 2.0);  pow_31 = None
    mul_183: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_91, mul_182);  div_91 = mul_182 = None
    neg_19: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_183)
    sum_65: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_19, [2], True);  neg_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_140: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_89, mul_183);  div_89 = mul_183 = None
    add_141: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_64, sum_65);  sum_64 = sum_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_59: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_141, [1, 512, 768]);  add_141 = None
    div_92: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_59, 768);  expand_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_142: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_140, div_92);  add_140 = div_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_64: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_76: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_39, scalar_tensor_64, add_142);  convert_element_type_39 = scalar_tensor_64 = None
    mul_184: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_76, 1.1111111111111112);  where_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_281: "f32[512, 768]" = torch.ops.aten.view.default(mul_184, [512, 768]);  mul_184 = None
    permute_208: "f32[768, 768]" = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None
    mm_34: "f32[512, 768]" = torch.ops.aten.mm.default(view_281, permute_208);  permute_208 = None
    permute_209: "f32[768, 512]" = torch.ops.aten.permute.default(view_281, [1, 0])
    mm_35: "f32[768, 768]" = torch.ops.aten.mm.default(permute_209, view_174);  permute_209 = view_174 = None
    permute_210: "f32[768, 768]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_66: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_281, [0], True);  view_281 = None
    view_282: "f32[768]" = torch.ops.aten.view.default(sum_66, [768]);  sum_66 = None
    permute_211: "f32[768, 768]" = torch.ops.aten.permute.default(permute_210, [1, 0]);  permute_210 = None
    view_283: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_34, [1, 512, 768]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_284: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_283, [1, 512, 12, 64]);  view_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_212: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_284, [0, 2, 1, 3]);  view_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_285: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_212, [12, 512, 64]);  permute_212 = None
    permute_213: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_170, [0, 2, 1]);  view_170 = None
    bmm_32: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_213, view_285);  permute_213 = None
    permute_214: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_171, [0, 2, 1]);  view_171 = None
    bmm_33: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_285, permute_214);  view_285 = permute_214 = None
    view_286: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_32, [1, 12, 512, 64]);  bmm_32 = None
    view_287: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_33, [1, 12, 512, 512]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_65: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_77: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_38, scalar_tensor_65, view_287);  convert_element_type_38 = scalar_tensor_65 = view_287 = None
    mul_185: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_77, 1.1111111111111112);  where_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_54: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(where_47);  where_47 = None
    alias_55: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_54);  alias_54 = None
    mul_186: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_185, alias_55);  mul_185 = None
    sum_67: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_186, [-1], True)
    mul_187: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_55, sum_67);  alias_55 = sum_67 = None
    sub_107: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_186, mul_187);  mul_186 = mul_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_288: "f32[12, 512, 512]" = torch.ops.aten.view.default(sub_107, [12, 512, 512]);  sub_107 = None
    permute_215: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_167, [0, 2, 1]);  view_167 = None
    bmm_34: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_215, view_288);  permute_215 = None
    permute_216: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1]);  view_168 = None
    bmm_35: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_288, permute_216);  view_288 = permute_216 = None
    view_289: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_34, [1, 12, 64, 512]);  bmm_34 = None
    view_290: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_35, [1, 12, 512, 64]);  bmm_35 = None
    permute_217: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_289, [0, 1, 3, 2]);  view_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_93: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(view_290, sqrt_28);  view_290 = sqrt_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    sum_68: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(view_286, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_218: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_68, [0, 2, 1, 3]);  sum_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_291: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_218, [1, 1, 768]);  permute_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    full_8: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_4: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_8, view_291, 2, 0, 9223372036854775807);  full_8 = view_291 = None
    squeeze_13: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_4, 1);  slice_scatter_4 = None
    squeeze_14: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_13, 0);  squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    sum_69: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(div_93, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_219: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_69, [0, 2, 1, 3]);  sum_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_292: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_219, [1, 1, 768]);  permute_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    full_9: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_5: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_9, view_292, 2, 0, 9223372036854775807);  full_9 = view_292 = None
    squeeze_15: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_5, 1);  slice_scatter_5 = None
    squeeze_16: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_15, 0);  squeeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    cat_3: "f32[1, 12, 512, 192]" = torch.ops.aten.cat.default([div_93, permute_217, view_286], 3);  div_93 = permute_217 = view_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_220: "f32[1, 512, 12, 192]" = torch.ops.aten.permute.default(cat_3, [0, 2, 1, 3]);  cat_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    clone_16: "f32[1, 512, 12, 192]" = torch.ops.aten.clone.default(permute_220, memory_format = torch.contiguous_format);  permute_220 = None
    view_293: "f32[1, 512, 2304]" = torch.ops.aten.view.default(clone_16, [1, 512, 2304]);  clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_294: "f32[512, 2304]" = torch.ops.aten.view.default(view_293, [512, 2304]);  view_293 = None
    permute_221: "f32[2304, 512]" = torch.ops.aten.permute.default(view_294, [1, 0])
    mm_36: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_221, view_162);  permute_221 = view_162 = None
    permute_222: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_36, [1, 0]);  mm_36 = None
    permute_223: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
    mm_37: "f32[512, 768]" = torch.ops.aten.mm.default(view_294, permute_223);  view_294 = permute_223 = None
    view_295: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_37, [1, 512, 768]);  mm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    add_143: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_142, view_295);  add_142 = view_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_224: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_222, [1, 0]);  permute_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_70: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_143, [0, 1], True)
    view_296: "f32[768]" = torch.ops.aten.view.default(sum_70, [768]);  sum_70 = None
    mul_188: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_143, primals_55);  primals_55 = None
    mul_189: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_143, div_36);  add_143 = div_36 = None
    sum_71: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_189, [0, 1], True);  mul_189 = None
    view_297: "f32[768]" = torch.ops.aten.view.default(sum_71, [768]);  sum_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_94: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_74, sqrt_27);  sub_74 = None
    div_95: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_94, sqrt_27);  div_94 = None
    neg_20: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_188)
    mul_190: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_20, div_95);  neg_20 = div_95 = None
    div_96: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_188, sqrt_27);  mul_188 = sqrt_27 = None
    sum_72: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_190, [2], True);  mul_190 = None
    alias_56: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_27);  alias_27 = None
    mul_191: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_56, 2);  alias_56 = None
    div_97: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_72, mul_191);  sum_72 = mul_191 = None
    neg_21: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_96)
    sum_73: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_21, [2], True);  neg_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_60: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_97, [1, 512, 768]);  div_97 = None
    div_98: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_60, 768);  expand_60 = None
    pow_32: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_73, 1.0);  sub_73 = None
    mul_192: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_32, 2.0);  pow_32 = None
    mul_193: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_98, mul_192);  div_98 = mul_192 = None
    neg_22: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_193)
    sum_74: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_22, [2], True);  neg_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_144: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_96, mul_193);  div_96 = mul_193 = None
    add_145: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_73, sum_74);  sum_73 = sum_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_61: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_145, [1, 512, 768]);  add_145 = None
    div_99: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_61, 768);  expand_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_146: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_144, div_99);  add_144 = div_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_66: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_78: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_36, scalar_tensor_66, add_146);  convert_element_type_36 = scalar_tensor_66 = None
    mul_194: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_78, 1.1111111111111112);  where_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_298: "f32[512, 768]" = torch.ops.aten.view.default(mul_194, [512, 768]);  mul_194 = None
    permute_225: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
    mm_38: "f32[512, 3072]" = torch.ops.aten.mm.default(view_298, permute_225);  permute_225 = None
    permute_226: "f32[768, 512]" = torch.ops.aten.permute.default(view_298, [1, 0])
    mm_39: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_226, view_160);  permute_226 = view_160 = None
    permute_227: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_75: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_298, [0], True);  view_298 = None
    view_299: "f32[768]" = torch.ops.aten.view.default(sum_75, [768]);  sum_75 = None
    permute_228: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_227, [1, 0]);  permute_227 = None
    view_300: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_38, [1, 512, 3072]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_195: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_159, 0.7071067811865476)
    erf_15: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_195);  mul_195 = None
    add_147: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_196: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_147, 0.5);  add_147 = None
    mul_197: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_159, view_159)
    mul_198: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_197, -0.5);  mul_197 = None
    exp_19: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_198);  mul_198 = None
    mul_199: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
    mul_200: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_159, mul_199);  view_159 = mul_199 = None
    add_148: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_196, mul_200);  mul_196 = mul_200 = None
    mul_201: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_300, add_148);  view_300 = add_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_301: "f32[512, 3072]" = torch.ops.aten.view.default(mul_201, [512, 3072]);  mul_201 = None
    permute_229: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_106, [1, 0]);  permute_106 = None
    mm_40: "f32[512, 768]" = torch.ops.aten.mm.default(view_301, permute_229);  permute_229 = None
    permute_230: "f32[3072, 512]" = torch.ops.aten.permute.default(view_301, [1, 0])
    mm_41: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_230, view_158);  permute_230 = view_158 = None
    permute_231: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_76: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_301, [0], True);  view_301 = None
    view_302: "f32[3072]" = torch.ops.aten.view.default(sum_76, [3072]);  sum_76 = None
    permute_232: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_231, [1, 0]);  permute_231 = None
    view_303: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_40, [1, 512, 768]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    add_149: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_146, view_303);  add_146 = view_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_77: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_149, [0, 1], True)
    view_304: "f32[768]" = torch.ops.aten.view.default(sum_77, [768]);  sum_77 = None
    mul_202: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_149, primals_53);  primals_53 = None
    mul_203: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_149, div_35);  add_149 = div_35 = None
    sum_78: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_203, [0, 1], True);  mul_203 = None
    view_305: "f32[768]" = torch.ops.aten.view.default(sum_78, [768]);  sum_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_100: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_71, sqrt_26);  sub_71 = None
    div_101: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_100, sqrt_26);  div_100 = None
    neg_23: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_202)
    mul_204: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_23, div_101);  neg_23 = div_101 = None
    div_102: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_202, sqrt_26);  mul_202 = sqrt_26 = None
    sum_79: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_204, [2], True);  mul_204 = None
    alias_57: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_26);  alias_26 = None
    mul_205: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_57, 2);  alias_57 = None
    div_103: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_79, mul_205);  sum_79 = mul_205 = None
    neg_24: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_102)
    sum_80: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_24, [2], True);  neg_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_62: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_103, [1, 512, 768]);  div_103 = None
    div_104: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_62, 768);  expand_62 = None
    pow_33: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_70, 1.0);  sub_70 = None
    mul_206: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_33, 2.0);  pow_33 = None
    mul_207: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_104, mul_206);  div_104 = mul_206 = None
    neg_25: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_207)
    sum_81: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_25, [2], True);  neg_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_150: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_102, mul_207);  div_102 = mul_207 = None
    add_151: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_80, sum_81);  sum_80 = sum_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_63: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_151, [1, 512, 768]);  add_151 = None
    div_105: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_63, 768);  expand_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_152: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_150, div_105);  add_150 = div_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_67: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_79: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_35, scalar_tensor_67, add_152);  convert_element_type_35 = scalar_tensor_67 = None
    mul_208: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_79, 1.1111111111111112);  where_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_306: "f32[512, 768]" = torch.ops.aten.view.default(mul_208, [512, 768]);  mul_208 = None
    permute_233: "f32[768, 768]" = torch.ops.aten.permute.default(permute_104, [1, 0]);  permute_104 = None
    mm_42: "f32[512, 768]" = torch.ops.aten.mm.default(view_306, permute_233);  permute_233 = None
    permute_234: "f32[768, 512]" = torch.ops.aten.permute.default(view_306, [1, 0])
    mm_43: "f32[768, 768]" = torch.ops.aten.mm.default(permute_234, view_156);  permute_234 = view_156 = None
    permute_235: "f32[768, 768]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_82: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_306, [0], True);  view_306 = None
    view_307: "f32[768]" = torch.ops.aten.view.default(sum_82, [768]);  sum_82 = None
    permute_236: "f32[768, 768]" = torch.ops.aten.permute.default(permute_235, [1, 0]);  permute_235 = None
    view_308: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_42, [1, 512, 768]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_309: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_308, [1, 512, 12, 64]);  view_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_237: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_309, [0, 2, 1, 3]);  view_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_310: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_237, [12, 512, 64]);  permute_237 = None
    permute_238: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_152, [0, 2, 1]);  view_152 = None
    bmm_36: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_238, view_310);  permute_238 = None
    permute_239: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_153, [0, 2, 1]);  view_153 = None
    bmm_37: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_310, permute_239);  view_310 = permute_239 = None
    view_311: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_36, [1, 12, 512, 64]);  bmm_36 = None
    view_312: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_37, [1, 12, 512, 512]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_68: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_80: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_34, scalar_tensor_68, view_312);  convert_element_type_34 = scalar_tensor_68 = view_312 = None
    mul_209: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_80, 1.1111111111111112);  where_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_59: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(where_42);  where_42 = None
    alias_60: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_59);  alias_59 = None
    mul_210: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_209, alias_60);  mul_209 = None
    sum_83: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_210, [-1], True)
    mul_211: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_60, sum_83);  alias_60 = sum_83 = None
    sub_108: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_210, mul_211);  mul_210 = mul_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_313: "f32[12, 512, 512]" = torch.ops.aten.view.default(sub_108, [12, 512, 512]);  sub_108 = None
    permute_240: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_149, [0, 2, 1]);  view_149 = None
    bmm_38: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_240, view_313);  permute_240 = None
    permute_241: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_150, [0, 2, 1]);  view_150 = None
    bmm_39: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_313, permute_241);  view_313 = permute_241 = None
    view_314: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_38, [1, 12, 64, 512]);  bmm_38 = None
    view_315: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_39, [1, 12, 512, 64]);  bmm_39 = None
    permute_242: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_314, [0, 1, 3, 2]);  view_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_106: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(view_315, sqrt_25);  view_315 = sqrt_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    sum_84: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(view_311, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_243: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_84, [0, 2, 1, 3]);  sum_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_316: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_243, [1, 1, 768]);  permute_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    full_10: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_6: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_10, view_316, 2, 0, 9223372036854775807);  full_10 = view_316 = None
    squeeze_17: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_6, 1);  slice_scatter_6 = None
    squeeze_18: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_17, 0);  squeeze_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    sum_85: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(div_106, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_244: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_85, [0, 2, 1, 3]);  sum_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_317: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_244, [1, 1, 768]);  permute_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    full_11: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_7: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_11, view_317, 2, 0, 9223372036854775807);  full_11 = view_317 = None
    squeeze_19: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_7, 1);  slice_scatter_7 = None
    squeeze_20: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_19, 0);  squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    cat_4: "f32[1, 12, 512, 192]" = torch.ops.aten.cat.default([div_106, permute_242, view_311], 3);  div_106 = permute_242 = view_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_245: "f32[1, 512, 12, 192]" = torch.ops.aten.permute.default(cat_4, [0, 2, 1, 3]);  cat_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    clone_17: "f32[1, 512, 12, 192]" = torch.ops.aten.clone.default(permute_245, memory_format = torch.contiguous_format);  permute_245 = None
    view_318: "f32[1, 512, 2304]" = torch.ops.aten.view.default(clone_17, [1, 512, 2304]);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_319: "f32[512, 2304]" = torch.ops.aten.view.default(view_318, [512, 2304]);  view_318 = None
    permute_246: "f32[2304, 512]" = torch.ops.aten.permute.default(view_319, [1, 0])
    mm_44: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_246, view_144);  permute_246 = view_144 = None
    permute_247: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_44, [1, 0]);  mm_44 = None
    permute_248: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
    mm_45: "f32[512, 768]" = torch.ops.aten.mm.default(view_319, permute_248);  view_319 = permute_248 = None
    view_320: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_45, [1, 512, 768]);  mm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    add_153: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_152, view_320);  add_152 = view_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_249: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_247, [1, 0]);  permute_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_86: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_153, [0, 1], True)
    view_321: "f32[768]" = torch.ops.aten.view.default(sum_86, [768]);  sum_86 = None
    mul_212: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_153, primals_49);  primals_49 = None
    mul_213: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_153, div_32);  add_153 = div_32 = None
    sum_87: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_213, [0, 1], True);  mul_213 = None
    view_322: "f32[768]" = torch.ops.aten.view.default(sum_87, [768]);  sum_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_107: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_66, sqrt_24);  sub_66 = None
    div_108: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_107, sqrt_24);  div_107 = None
    neg_26: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_212)
    mul_214: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_26, div_108);  neg_26 = div_108 = None
    div_109: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_212, sqrt_24);  mul_212 = sqrt_24 = None
    sum_88: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_214, [2], True);  mul_214 = None
    alias_61: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_24);  alias_24 = None
    mul_215: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_61, 2);  alias_61 = None
    div_110: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_88, mul_215);  sum_88 = mul_215 = None
    neg_27: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_109)
    sum_89: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_27, [2], True);  neg_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_64: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_110, [1, 512, 768]);  div_110 = None
    div_111: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_64, 768);  expand_64 = None
    pow_34: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_65, 1.0);  sub_65 = None
    mul_216: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_34, 2.0);  pow_34 = None
    mul_217: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_111, mul_216);  div_111 = mul_216 = None
    neg_28: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_217)
    sum_90: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_28, [2], True);  neg_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_154: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_109, mul_217);  div_109 = mul_217 = None
    add_155: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_89, sum_90);  sum_89 = sum_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_65: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_155, [1, 512, 768]);  add_155 = None
    div_112: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_65, 768);  expand_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_156: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_154, div_112);  add_154 = div_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_69: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_81: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_32, scalar_tensor_69, add_156);  convert_element_type_32 = scalar_tensor_69 = None
    mul_218: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_81, 1.1111111111111112);  where_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_323: "f32[512, 768]" = torch.ops.aten.view.default(mul_218, [512, 768]);  mul_218 = None
    permute_250: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_95, [1, 0]);  permute_95 = None
    mm_46: "f32[512, 3072]" = torch.ops.aten.mm.default(view_323, permute_250);  permute_250 = None
    permute_251: "f32[768, 512]" = torch.ops.aten.permute.default(view_323, [1, 0])
    mm_47: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_251, view_142);  permute_251 = view_142 = None
    permute_252: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_91: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_323, [0], True);  view_323 = None
    view_324: "f32[768]" = torch.ops.aten.view.default(sum_91, [768]);  sum_91 = None
    permute_253: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_252, [1, 0]);  permute_252 = None
    view_325: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_46, [1, 512, 3072]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_219: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_141, 0.7071067811865476)
    erf_16: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_219);  mul_219 = None
    add_157: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_220: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_157, 0.5);  add_157 = None
    mul_221: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_141, view_141)
    mul_222: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_221, -0.5);  mul_221 = None
    exp_20: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_222);  mul_222 = None
    mul_223: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
    mul_224: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_141, mul_223);  view_141 = mul_223 = None
    add_158: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_220, mul_224);  mul_220 = mul_224 = None
    mul_225: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_325, add_158);  view_325 = add_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_326: "f32[512, 3072]" = torch.ops.aten.view.default(mul_225, [512, 3072]);  mul_225 = None
    permute_254: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_94, [1, 0]);  permute_94 = None
    mm_48: "f32[512, 768]" = torch.ops.aten.mm.default(view_326, permute_254);  permute_254 = None
    permute_255: "f32[3072, 512]" = torch.ops.aten.permute.default(view_326, [1, 0])
    mm_49: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_255, view_140);  permute_255 = view_140 = None
    permute_256: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_92: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_326, [0], True);  view_326 = None
    view_327: "f32[3072]" = torch.ops.aten.view.default(sum_92, [3072]);  sum_92 = None
    permute_257: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_256, [1, 0]);  permute_256 = None
    view_328: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_48, [1, 512, 768]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    add_159: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_156, view_328);  add_156 = view_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_93: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_159, [0, 1], True)
    view_329: "f32[768]" = torch.ops.aten.view.default(sum_93, [768]);  sum_93 = None
    mul_226: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_159, primals_47);  primals_47 = None
    mul_227: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_159, div_31);  add_159 = div_31 = None
    sum_94: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_227, [0, 1], True);  mul_227 = None
    view_330: "f32[768]" = torch.ops.aten.view.default(sum_94, [768]);  sum_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_113: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_63, sqrt_23);  sub_63 = None
    div_114: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_113, sqrt_23);  div_113 = None
    neg_29: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_226)
    mul_228: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_29, div_114);  neg_29 = div_114 = None
    div_115: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_226, sqrt_23);  mul_226 = sqrt_23 = None
    sum_95: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_228, [2], True);  mul_228 = None
    alias_62: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_23);  alias_23 = None
    mul_229: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_62, 2);  alias_62 = None
    div_116: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_95, mul_229);  sum_95 = mul_229 = None
    neg_30: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_115)
    sum_96: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_30, [2], True);  neg_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_66: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_116, [1, 512, 768]);  div_116 = None
    div_117: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_66, 768);  expand_66 = None
    pow_35: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_62, 1.0);  sub_62 = None
    mul_230: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_35, 2.0);  pow_35 = None
    mul_231: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_117, mul_230);  div_117 = mul_230 = None
    neg_31: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_231)
    sum_97: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_31, [2], True);  neg_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_160: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_115, mul_231);  div_115 = mul_231 = None
    add_161: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_96, sum_97);  sum_96 = sum_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_67: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_161, [1, 512, 768]);  add_161 = None
    div_118: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_67, 768);  expand_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_162: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_160, div_118);  add_160 = div_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_70: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_82: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_31, scalar_tensor_70, add_162);  convert_element_type_31 = scalar_tensor_70 = None
    mul_232: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_82, 1.1111111111111112);  where_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_331: "f32[512, 768]" = torch.ops.aten.view.default(mul_232, [512, 768]);  mul_232 = None
    permute_258: "f32[768, 768]" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
    mm_50: "f32[512, 768]" = torch.ops.aten.mm.default(view_331, permute_258);  permute_258 = None
    permute_259: "f32[768, 512]" = torch.ops.aten.permute.default(view_331, [1, 0])
    mm_51: "f32[768, 768]" = torch.ops.aten.mm.default(permute_259, view_138);  permute_259 = view_138 = None
    permute_260: "f32[768, 768]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_98: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_331, [0], True);  view_331 = None
    view_332: "f32[768]" = torch.ops.aten.view.default(sum_98, [768]);  sum_98 = None
    permute_261: "f32[768, 768]" = torch.ops.aten.permute.default(permute_260, [1, 0]);  permute_260 = None
    view_333: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_50, [1, 512, 768]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_334: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_333, [1, 512, 12, 64]);  view_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_262: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_334, [0, 2, 1, 3]);  view_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_335: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_262, [12, 512, 64]);  permute_262 = None
    permute_263: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_134, [0, 2, 1]);  view_134 = None
    bmm_40: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_263, view_335);  permute_263 = None
    permute_264: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_135, [0, 2, 1]);  view_135 = None
    bmm_41: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_335, permute_264);  view_335 = permute_264 = None
    view_336: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_40, [1, 12, 512, 64]);  bmm_40 = None
    view_337: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_41, [1, 12, 512, 512]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_71: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_83: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_30, scalar_tensor_71, view_337);  convert_element_type_30 = scalar_tensor_71 = view_337 = None
    mul_233: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_83, 1.1111111111111112);  where_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_64: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(where_37);  where_37 = None
    alias_65: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_64);  alias_64 = None
    mul_234: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_233, alias_65);  mul_233 = None
    sum_99: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_234, [-1], True)
    mul_235: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_65, sum_99);  alias_65 = sum_99 = None
    sub_109: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_234, mul_235);  mul_234 = mul_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_338: "f32[12, 512, 512]" = torch.ops.aten.view.default(sub_109, [12, 512, 512]);  sub_109 = None
    permute_265: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_131, [0, 2, 1]);  view_131 = None
    bmm_42: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_265, view_338);  permute_265 = None
    permute_266: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_132, [0, 2, 1]);  view_132 = None
    bmm_43: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_338, permute_266);  view_338 = permute_266 = None
    view_339: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_42, [1, 12, 64, 512]);  bmm_42 = None
    view_340: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_43, [1, 12, 512, 64]);  bmm_43 = None
    permute_267: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_339, [0, 1, 3, 2]);  view_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_119: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(view_340, sqrt_22);  view_340 = sqrt_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    sum_100: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(view_336, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_268: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_100, [0, 2, 1, 3]);  sum_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_341: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_268, [1, 1, 768]);  permute_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    full_12: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_8: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_12, view_341, 2, 0, 9223372036854775807);  full_12 = view_341 = None
    squeeze_21: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_8, 1);  slice_scatter_8 = None
    squeeze_22: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_21, 0);  squeeze_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    sum_101: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(div_119, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_269: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_101, [0, 2, 1, 3]);  sum_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_342: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_269, [1, 1, 768]);  permute_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    full_13: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_9: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_13, view_342, 2, 0, 9223372036854775807);  full_13 = view_342 = None
    squeeze_23: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_9, 1);  slice_scatter_9 = None
    squeeze_24: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_23, 0);  squeeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    cat_5: "f32[1, 12, 512, 192]" = torch.ops.aten.cat.default([div_119, permute_267, view_336], 3);  div_119 = permute_267 = view_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_270: "f32[1, 512, 12, 192]" = torch.ops.aten.permute.default(cat_5, [0, 2, 1, 3]);  cat_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    clone_18: "f32[1, 512, 12, 192]" = torch.ops.aten.clone.default(permute_270, memory_format = torch.contiguous_format);  permute_270 = None
    view_343: "f32[1, 512, 2304]" = torch.ops.aten.view.default(clone_18, [1, 512, 2304]);  clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_344: "f32[512, 2304]" = torch.ops.aten.view.default(view_343, [512, 2304]);  view_343 = None
    permute_271: "f32[2304, 512]" = torch.ops.aten.permute.default(view_344, [1, 0])
    mm_52: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_271, view_126);  permute_271 = view_126 = None
    permute_272: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_52, [1, 0]);  mm_52 = None
    permute_273: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
    mm_53: "f32[512, 768]" = torch.ops.aten.mm.default(view_344, permute_273);  view_344 = permute_273 = None
    view_345: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_53, [1, 512, 768]);  mm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    add_163: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_162, view_345);  add_162 = view_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_274: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_272, [1, 0]);  permute_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_102: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_163, [0, 1], True)
    view_346: "f32[768]" = torch.ops.aten.view.default(sum_102, [768]);  sum_102 = None
    mul_236: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_163, primals_43);  primals_43 = None
    mul_237: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_163, div_28);  add_163 = div_28 = None
    sum_103: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_237, [0, 1], True);  mul_237 = None
    view_347: "f32[768]" = torch.ops.aten.view.default(sum_103, [768]);  sum_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_120: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_58, sqrt_21);  sub_58 = None
    div_121: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_120, sqrt_21);  div_120 = None
    neg_32: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_236)
    mul_238: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_32, div_121);  neg_32 = div_121 = None
    div_122: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_236, sqrt_21);  mul_236 = sqrt_21 = None
    sum_104: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_238, [2], True);  mul_238 = None
    alias_66: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_21);  alias_21 = None
    mul_239: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_66, 2);  alias_66 = None
    div_123: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_104, mul_239);  sum_104 = mul_239 = None
    neg_33: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_122)
    sum_105: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_33, [2], True);  neg_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_68: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_123, [1, 512, 768]);  div_123 = None
    div_124: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_68, 768);  expand_68 = None
    pow_36: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_57, 1.0);  sub_57 = None
    mul_240: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_36, 2.0);  pow_36 = None
    mul_241: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_124, mul_240);  div_124 = mul_240 = None
    neg_34: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_241)
    sum_106: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_34, [2], True);  neg_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_164: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_122, mul_241);  div_122 = mul_241 = None
    add_165: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_105, sum_106);  sum_105 = sum_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_69: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_165, [1, 512, 768]);  add_165 = None
    div_125: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_69, 768);  expand_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_166: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_164, div_125);  add_164 = div_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_72: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_84: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_28, scalar_tensor_72, add_166);  convert_element_type_28 = scalar_tensor_72 = None
    mul_242: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_84, 1.1111111111111112);  where_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_348: "f32[512, 768]" = torch.ops.aten.view.default(mul_242, [512, 768]);  mul_242 = None
    permute_275: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_83, [1, 0]);  permute_83 = None
    mm_54: "f32[512, 3072]" = torch.ops.aten.mm.default(view_348, permute_275);  permute_275 = None
    permute_276: "f32[768, 512]" = torch.ops.aten.permute.default(view_348, [1, 0])
    mm_55: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_276, view_124);  permute_276 = view_124 = None
    permute_277: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_107: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_348, [0], True);  view_348 = None
    view_349: "f32[768]" = torch.ops.aten.view.default(sum_107, [768]);  sum_107 = None
    permute_278: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_277, [1, 0]);  permute_277 = None
    view_350: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_54, [1, 512, 3072]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_243: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_123, 0.7071067811865476)
    erf_17: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_243);  mul_243 = None
    add_167: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_244: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_167, 0.5);  add_167 = None
    mul_245: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_123, view_123)
    mul_246: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_245, -0.5);  mul_245 = None
    exp_21: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_246);  mul_246 = None
    mul_247: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
    mul_248: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_123, mul_247);  view_123 = mul_247 = None
    add_168: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_244, mul_248);  mul_244 = mul_248 = None
    mul_249: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_350, add_168);  view_350 = add_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_351: "f32[512, 3072]" = torch.ops.aten.view.default(mul_249, [512, 3072]);  mul_249 = None
    permute_279: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_82, [1, 0]);  permute_82 = None
    mm_56: "f32[512, 768]" = torch.ops.aten.mm.default(view_351, permute_279);  permute_279 = None
    permute_280: "f32[3072, 512]" = torch.ops.aten.permute.default(view_351, [1, 0])
    mm_57: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_280, view_122);  permute_280 = view_122 = None
    permute_281: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_108: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_351, [0], True);  view_351 = None
    view_352: "f32[3072]" = torch.ops.aten.view.default(sum_108, [3072]);  sum_108 = None
    permute_282: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_281, [1, 0]);  permute_281 = None
    view_353: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_56, [1, 512, 768]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    add_169: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_166, view_353);  add_166 = view_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_109: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_169, [0, 1], True)
    view_354: "f32[768]" = torch.ops.aten.view.default(sum_109, [768]);  sum_109 = None
    mul_250: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_169, primals_41);  primals_41 = None
    mul_251: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_169, div_27);  add_169 = div_27 = None
    sum_110: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_251, [0, 1], True);  mul_251 = None
    view_355: "f32[768]" = torch.ops.aten.view.default(sum_110, [768]);  sum_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_126: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_55, sqrt_20);  sub_55 = None
    div_127: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_126, sqrt_20);  div_126 = None
    neg_35: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_250)
    mul_252: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_35, div_127);  neg_35 = div_127 = None
    div_128: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_250, sqrt_20);  mul_250 = sqrt_20 = None
    sum_111: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_252, [2], True);  mul_252 = None
    alias_67: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_20);  alias_20 = None
    mul_253: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_67, 2);  alias_67 = None
    div_129: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_111, mul_253);  sum_111 = mul_253 = None
    neg_36: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_128)
    sum_112: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_36, [2], True);  neg_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_70: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_129, [1, 512, 768]);  div_129 = None
    div_130: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_70, 768);  expand_70 = None
    pow_37: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_54, 1.0);  sub_54 = None
    mul_254: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_37, 2.0);  pow_37 = None
    mul_255: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_130, mul_254);  div_130 = mul_254 = None
    neg_37: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_255)
    sum_113: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_37, [2], True);  neg_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_170: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_128, mul_255);  div_128 = mul_255 = None
    add_171: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_112, sum_113);  sum_112 = sum_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_71: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_171, [1, 512, 768]);  add_171 = None
    div_131: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_71, 768);  expand_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_172: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_170, div_131);  add_170 = div_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_73: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_85: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_27, scalar_tensor_73, add_172);  convert_element_type_27 = scalar_tensor_73 = None
    mul_256: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_85, 1.1111111111111112);  where_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_356: "f32[512, 768]" = torch.ops.aten.view.default(mul_256, [512, 768]);  mul_256 = None
    permute_283: "f32[768, 768]" = torch.ops.aten.permute.default(permute_80, [1, 0]);  permute_80 = None
    mm_58: "f32[512, 768]" = torch.ops.aten.mm.default(view_356, permute_283);  permute_283 = None
    permute_284: "f32[768, 512]" = torch.ops.aten.permute.default(view_356, [1, 0])
    mm_59: "f32[768, 768]" = torch.ops.aten.mm.default(permute_284, view_120);  permute_284 = view_120 = None
    permute_285: "f32[768, 768]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_114: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_356, [0], True);  view_356 = None
    view_357: "f32[768]" = torch.ops.aten.view.default(sum_114, [768]);  sum_114 = None
    permute_286: "f32[768, 768]" = torch.ops.aten.permute.default(permute_285, [1, 0]);  permute_285 = None
    view_358: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_58, [1, 512, 768]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_359: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_358, [1, 512, 12, 64]);  view_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_287: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_359, [0, 2, 1, 3]);  view_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_360: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_287, [12, 512, 64]);  permute_287 = None
    permute_288: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_116, [0, 2, 1]);  view_116 = None
    bmm_44: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_288, view_360);  permute_288 = None
    permute_289: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_117, [0, 2, 1]);  view_117 = None
    bmm_45: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_360, permute_289);  view_360 = permute_289 = None
    view_361: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_44, [1, 12, 512, 64]);  bmm_44 = None
    view_362: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_45, [1, 12, 512, 512]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_74: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_86: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_26, scalar_tensor_74, view_362);  convert_element_type_26 = scalar_tensor_74 = view_362 = None
    mul_257: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_86, 1.1111111111111112);  where_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_69: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(where_32);  where_32 = None
    alias_70: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_69);  alias_69 = None
    mul_258: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_257, alias_70);  mul_257 = None
    sum_115: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_258, [-1], True)
    mul_259: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_70, sum_115);  alias_70 = sum_115 = None
    sub_110: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_258, mul_259);  mul_258 = mul_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_363: "f32[12, 512, 512]" = torch.ops.aten.view.default(sub_110, [12, 512, 512]);  sub_110 = None
    permute_290: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_113, [0, 2, 1]);  view_113 = None
    bmm_46: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_290, view_363);  permute_290 = None
    permute_291: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_114, [0, 2, 1]);  view_114 = None
    bmm_47: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_363, permute_291);  view_363 = permute_291 = None
    view_364: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_46, [1, 12, 64, 512]);  bmm_46 = None
    view_365: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_47, [1, 12, 512, 64]);  bmm_47 = None
    permute_292: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_364, [0, 1, 3, 2]);  view_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_132: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(view_365, sqrt_19);  view_365 = sqrt_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    sum_116: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(view_361, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_293: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_116, [0, 2, 1, 3]);  sum_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_366: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_293, [1, 1, 768]);  permute_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    full_14: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_10: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_14, view_366, 2, 0, 9223372036854775807);  full_14 = view_366 = None
    squeeze_25: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_10, 1);  slice_scatter_10 = None
    squeeze_26: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_25, 0);  squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    sum_117: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(div_132, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_294: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_117, [0, 2, 1, 3]);  sum_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_367: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_294, [1, 1, 768]);  permute_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    full_15: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_11: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_15, view_367, 2, 0, 9223372036854775807);  full_15 = view_367 = None
    squeeze_27: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_11, 1);  slice_scatter_11 = None
    squeeze_28: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_27, 0);  squeeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    cat_6: "f32[1, 12, 512, 192]" = torch.ops.aten.cat.default([div_132, permute_292, view_361], 3);  div_132 = permute_292 = view_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_295: "f32[1, 512, 12, 192]" = torch.ops.aten.permute.default(cat_6, [0, 2, 1, 3]);  cat_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    clone_19: "f32[1, 512, 12, 192]" = torch.ops.aten.clone.default(permute_295, memory_format = torch.contiguous_format);  permute_295 = None
    view_368: "f32[1, 512, 2304]" = torch.ops.aten.view.default(clone_19, [1, 512, 2304]);  clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_369: "f32[512, 2304]" = torch.ops.aten.view.default(view_368, [512, 2304]);  view_368 = None
    permute_296: "f32[2304, 512]" = torch.ops.aten.permute.default(view_369, [1, 0])
    mm_60: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_296, view_108);  permute_296 = view_108 = None
    permute_297: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_60, [1, 0]);  mm_60 = None
    permute_298: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_73, [1, 0]);  permute_73 = None
    mm_61: "f32[512, 768]" = torch.ops.aten.mm.default(view_369, permute_298);  view_369 = permute_298 = None
    view_370: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_61, [1, 512, 768]);  mm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    add_173: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_172, view_370);  add_172 = view_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_299: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_297, [1, 0]);  permute_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_118: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_173, [0, 1], True)
    view_371: "f32[768]" = torch.ops.aten.view.default(sum_118, [768]);  sum_118 = None
    mul_260: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_173, primals_37);  primals_37 = None
    mul_261: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_173, div_24);  add_173 = div_24 = None
    sum_119: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_261, [0, 1], True);  mul_261 = None
    view_372: "f32[768]" = torch.ops.aten.view.default(sum_119, [768]);  sum_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_133: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_50, sqrt_18);  sub_50 = None
    div_134: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_133, sqrt_18);  div_133 = None
    neg_38: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_260)
    mul_262: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_38, div_134);  neg_38 = div_134 = None
    div_135: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_260, sqrt_18);  mul_260 = sqrt_18 = None
    sum_120: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_262, [2], True);  mul_262 = None
    alias_71: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_18);  alias_18 = None
    mul_263: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_71, 2);  alias_71 = None
    div_136: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_120, mul_263);  sum_120 = mul_263 = None
    neg_39: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_135)
    sum_121: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_39, [2], True);  neg_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_72: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_136, [1, 512, 768]);  div_136 = None
    div_137: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_72, 768);  expand_72 = None
    pow_38: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_49, 1.0);  sub_49 = None
    mul_264: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_38, 2.0);  pow_38 = None
    mul_265: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_137, mul_264);  div_137 = mul_264 = None
    neg_40: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_265)
    sum_122: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_40, [2], True);  neg_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_174: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_135, mul_265);  div_135 = mul_265 = None
    add_175: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_121, sum_122);  sum_121 = sum_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_73: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_175, [1, 512, 768]);  add_175 = None
    div_138: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_73, 768);  expand_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_176: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_174, div_138);  add_174 = div_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_75: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_87: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_24, scalar_tensor_75, add_176);  convert_element_type_24 = scalar_tensor_75 = None
    mul_266: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_87, 1.1111111111111112);  where_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_373: "f32[512, 768]" = torch.ops.aten.view.default(mul_266, [512, 768]);  mul_266 = None
    permute_300: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
    mm_62: "f32[512, 3072]" = torch.ops.aten.mm.default(view_373, permute_300);  permute_300 = None
    permute_301: "f32[768, 512]" = torch.ops.aten.permute.default(view_373, [1, 0])
    mm_63: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_301, view_106);  permute_301 = view_106 = None
    permute_302: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_123: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_373, [0], True);  view_373 = None
    view_374: "f32[768]" = torch.ops.aten.view.default(sum_123, [768]);  sum_123 = None
    permute_303: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_302, [1, 0]);  permute_302 = None
    view_375: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_62, [1, 512, 3072]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_267: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_105, 0.7071067811865476)
    erf_18: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_267);  mul_267 = None
    add_177: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_268: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_177, 0.5);  add_177 = None
    mul_269: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_105, view_105)
    mul_270: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_269, -0.5);  mul_269 = None
    exp_22: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_270);  mul_270 = None
    mul_271: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
    mul_272: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_105, mul_271);  view_105 = mul_271 = None
    add_178: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_268, mul_272);  mul_268 = mul_272 = None
    mul_273: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_375, add_178);  view_375 = add_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_376: "f32[512, 3072]" = torch.ops.aten.view.default(mul_273, [512, 3072]);  mul_273 = None
    permute_304: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
    mm_64: "f32[512, 768]" = torch.ops.aten.mm.default(view_376, permute_304);  permute_304 = None
    permute_305: "f32[3072, 512]" = torch.ops.aten.permute.default(view_376, [1, 0])
    mm_65: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_305, view_104);  permute_305 = view_104 = None
    permute_306: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_124: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_376, [0], True);  view_376 = None
    view_377: "f32[3072]" = torch.ops.aten.view.default(sum_124, [3072]);  sum_124 = None
    permute_307: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_306, [1, 0]);  permute_306 = None
    view_378: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_64, [1, 512, 768]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    add_179: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_176, view_378);  add_176 = view_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_125: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_179, [0, 1], True)
    view_379: "f32[768]" = torch.ops.aten.view.default(sum_125, [768]);  sum_125 = None
    mul_274: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_179, primals_35);  primals_35 = None
    mul_275: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_179, div_23);  add_179 = div_23 = None
    sum_126: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_275, [0, 1], True);  mul_275 = None
    view_380: "f32[768]" = torch.ops.aten.view.default(sum_126, [768]);  sum_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_139: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_47, sqrt_17);  sub_47 = None
    div_140: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_139, sqrt_17);  div_139 = None
    neg_41: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_274)
    mul_276: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_41, div_140);  neg_41 = div_140 = None
    div_141: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_274, sqrt_17);  mul_274 = sqrt_17 = None
    sum_127: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_276, [2], True);  mul_276 = None
    alias_72: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_17);  alias_17 = None
    mul_277: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_72, 2);  alias_72 = None
    div_142: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_127, mul_277);  sum_127 = mul_277 = None
    neg_42: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_141)
    sum_128: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_42, [2], True);  neg_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_74: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_142, [1, 512, 768]);  div_142 = None
    div_143: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_74, 768);  expand_74 = None
    pow_39: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_46, 1.0);  sub_46 = None
    mul_278: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_39, 2.0);  pow_39 = None
    mul_279: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_143, mul_278);  div_143 = mul_278 = None
    neg_43: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_279)
    sum_129: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_43, [2], True);  neg_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_180: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_141, mul_279);  div_141 = mul_279 = None
    add_181: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_128, sum_129);  sum_128 = sum_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_75: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_181, [1, 512, 768]);  add_181 = None
    div_144: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_75, 768);  expand_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_182: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_180, div_144);  add_180 = div_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_76: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_88: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_23, scalar_tensor_76, add_182);  convert_element_type_23 = scalar_tensor_76 = None
    mul_280: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_88, 1.1111111111111112);  where_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_381: "f32[512, 768]" = torch.ops.aten.view.default(mul_280, [512, 768]);  mul_280 = None
    permute_308: "f32[768, 768]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
    mm_66: "f32[512, 768]" = torch.ops.aten.mm.default(view_381, permute_308);  permute_308 = None
    permute_309: "f32[768, 512]" = torch.ops.aten.permute.default(view_381, [1, 0])
    mm_67: "f32[768, 768]" = torch.ops.aten.mm.default(permute_309, view_102);  permute_309 = view_102 = None
    permute_310: "f32[768, 768]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_130: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_381, [0], True);  view_381 = None
    view_382: "f32[768]" = torch.ops.aten.view.default(sum_130, [768]);  sum_130 = None
    permute_311: "f32[768, 768]" = torch.ops.aten.permute.default(permute_310, [1, 0]);  permute_310 = None
    view_383: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_66, [1, 512, 768]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_384: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_383, [1, 512, 12, 64]);  view_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_312: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_384, [0, 2, 1, 3]);  view_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_385: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_312, [12, 512, 64]);  permute_312 = None
    permute_313: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_98, [0, 2, 1]);  view_98 = None
    bmm_48: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_313, view_385);  permute_313 = None
    permute_314: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_99, [0, 2, 1]);  view_99 = None
    bmm_49: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_385, permute_314);  view_385 = permute_314 = None
    view_386: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_48, [1, 12, 512, 64]);  bmm_48 = None
    view_387: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_49, [1, 12, 512, 512]);  bmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_77: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_89: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_22, scalar_tensor_77, view_387);  convert_element_type_22 = scalar_tensor_77 = view_387 = None
    mul_281: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_89, 1.1111111111111112);  where_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_74: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(where_27);  where_27 = None
    alias_75: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_74);  alias_74 = None
    mul_282: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_281, alias_75);  mul_281 = None
    sum_131: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_282, [-1], True)
    mul_283: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_75, sum_131);  alias_75 = sum_131 = None
    sub_111: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_282, mul_283);  mul_282 = mul_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_388: "f32[12, 512, 512]" = torch.ops.aten.view.default(sub_111, [12, 512, 512]);  sub_111 = None
    permute_315: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_95, [0, 2, 1]);  view_95 = None
    bmm_50: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_315, view_388);  permute_315 = None
    permute_316: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_96, [0, 2, 1]);  view_96 = None
    bmm_51: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_388, permute_316);  view_388 = permute_316 = None
    view_389: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_50, [1, 12, 64, 512]);  bmm_50 = None
    view_390: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_51, [1, 12, 512, 64]);  bmm_51 = None
    permute_317: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_389, [0, 1, 3, 2]);  view_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_145: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(view_390, sqrt_16);  view_390 = sqrt_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    sum_132: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(view_386, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_318: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_132, [0, 2, 1, 3]);  sum_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_391: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_318, [1, 1, 768]);  permute_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    full_16: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_12: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_16, view_391, 2, 0, 9223372036854775807);  full_16 = view_391 = None
    squeeze_29: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_12, 1);  slice_scatter_12 = None
    squeeze_30: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_29, 0);  squeeze_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    sum_133: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(div_145, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_319: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_133, [0, 2, 1, 3]);  sum_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_392: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_319, [1, 1, 768]);  permute_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    full_17: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_13: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_17, view_392, 2, 0, 9223372036854775807);  full_17 = view_392 = None
    squeeze_31: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_13, 1);  slice_scatter_13 = None
    squeeze_32: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_31, 0);  squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    cat_7: "f32[1, 12, 512, 192]" = torch.ops.aten.cat.default([div_145, permute_317, view_386], 3);  div_145 = permute_317 = view_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_320: "f32[1, 512, 12, 192]" = torch.ops.aten.permute.default(cat_7, [0, 2, 1, 3]);  cat_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    clone_20: "f32[1, 512, 12, 192]" = torch.ops.aten.clone.default(permute_320, memory_format = torch.contiguous_format);  permute_320 = None
    view_393: "f32[1, 512, 2304]" = torch.ops.aten.view.default(clone_20, [1, 512, 2304]);  clone_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_394: "f32[512, 2304]" = torch.ops.aten.view.default(view_393, [512, 2304]);  view_393 = None
    permute_321: "f32[2304, 512]" = torch.ops.aten.permute.default(view_394, [1, 0])
    mm_68: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_321, view_90);  permute_321 = view_90 = None
    permute_322: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_68, [1, 0]);  mm_68 = None
    permute_323: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_61, [1, 0]);  permute_61 = None
    mm_69: "f32[512, 768]" = torch.ops.aten.mm.default(view_394, permute_323);  view_394 = permute_323 = None
    view_395: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_69, [1, 512, 768]);  mm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    add_183: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_182, view_395);  add_182 = view_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_324: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_322, [1, 0]);  permute_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_134: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_183, [0, 1], True)
    view_396: "f32[768]" = torch.ops.aten.view.default(sum_134, [768]);  sum_134 = None
    mul_284: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_183, primals_31);  primals_31 = None
    mul_285: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_183, div_20);  add_183 = div_20 = None
    sum_135: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_285, [0, 1], True);  mul_285 = None
    view_397: "f32[768]" = torch.ops.aten.view.default(sum_135, [768]);  sum_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_146: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_42, sqrt_15);  sub_42 = None
    div_147: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_146, sqrt_15);  div_146 = None
    neg_44: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_284)
    mul_286: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_44, div_147);  neg_44 = div_147 = None
    div_148: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_284, sqrt_15);  mul_284 = sqrt_15 = None
    sum_136: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_286, [2], True);  mul_286 = None
    alias_76: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_15);  alias_15 = None
    mul_287: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_76, 2);  alias_76 = None
    div_149: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_136, mul_287);  sum_136 = mul_287 = None
    neg_45: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_148)
    sum_137: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_45, [2], True);  neg_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_76: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_149, [1, 512, 768]);  div_149 = None
    div_150: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_76, 768);  expand_76 = None
    pow_40: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_41, 1.0);  sub_41 = None
    mul_288: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_40, 2.0);  pow_40 = None
    mul_289: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_150, mul_288);  div_150 = mul_288 = None
    neg_46: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_289)
    sum_138: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_46, [2], True);  neg_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_184: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_148, mul_289);  div_148 = mul_289 = None
    add_185: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_137, sum_138);  sum_137 = sum_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_77: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_185, [1, 512, 768]);  add_185 = None
    div_151: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_77, 768);  expand_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_186: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_184, div_151);  add_184 = div_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_78: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_90: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_20, scalar_tensor_78, add_186);  convert_element_type_20 = scalar_tensor_78 = None
    mul_290: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_90, 1.1111111111111112);  where_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_398: "f32[512, 768]" = torch.ops.aten.view.default(mul_290, [512, 768]);  mul_290 = None
    permute_325: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
    mm_70: "f32[512, 3072]" = torch.ops.aten.mm.default(view_398, permute_325);  permute_325 = None
    permute_326: "f32[768, 512]" = torch.ops.aten.permute.default(view_398, [1, 0])
    mm_71: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_326, view_88);  permute_326 = view_88 = None
    permute_327: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_139: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_398, [0], True);  view_398 = None
    view_399: "f32[768]" = torch.ops.aten.view.default(sum_139, [768]);  sum_139 = None
    permute_328: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_327, [1, 0]);  permute_327 = None
    view_400: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_70, [1, 512, 3072]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_291: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_87, 0.7071067811865476)
    erf_19: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_291);  mul_291 = None
    add_187: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_292: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_187, 0.5);  add_187 = None
    mul_293: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_87, view_87)
    mul_294: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_293, -0.5);  mul_293 = None
    exp_23: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_294);  mul_294 = None
    mul_295: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
    mul_296: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_87, mul_295);  view_87 = mul_295 = None
    add_188: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_292, mul_296);  mul_292 = mul_296 = None
    mul_297: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_400, add_188);  view_400 = add_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_401: "f32[512, 3072]" = torch.ops.aten.view.default(mul_297, [512, 3072]);  mul_297 = None
    permute_329: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
    mm_72: "f32[512, 768]" = torch.ops.aten.mm.default(view_401, permute_329);  permute_329 = None
    permute_330: "f32[3072, 512]" = torch.ops.aten.permute.default(view_401, [1, 0])
    mm_73: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_330, view_86);  permute_330 = view_86 = None
    permute_331: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_140: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_401, [0], True);  view_401 = None
    view_402: "f32[3072]" = torch.ops.aten.view.default(sum_140, [3072]);  sum_140 = None
    permute_332: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_331, [1, 0]);  permute_331 = None
    view_403: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_72, [1, 512, 768]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    add_189: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_186, view_403);  add_186 = view_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_141: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_189, [0, 1], True)
    view_404: "f32[768]" = torch.ops.aten.view.default(sum_141, [768]);  sum_141 = None
    mul_298: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_189, primals_29);  primals_29 = None
    mul_299: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_189, div_19);  add_189 = div_19 = None
    sum_142: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_299, [0, 1], True);  mul_299 = None
    view_405: "f32[768]" = torch.ops.aten.view.default(sum_142, [768]);  sum_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_152: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_39, sqrt_14);  sub_39 = None
    div_153: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_152, sqrt_14);  div_152 = None
    neg_47: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_298)
    mul_300: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_47, div_153);  neg_47 = div_153 = None
    div_154: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_298, sqrt_14);  mul_298 = sqrt_14 = None
    sum_143: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_300, [2], True);  mul_300 = None
    alias_77: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_14);  alias_14 = None
    mul_301: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_77, 2);  alias_77 = None
    div_155: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_143, mul_301);  sum_143 = mul_301 = None
    neg_48: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_154)
    sum_144: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_48, [2], True);  neg_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_78: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_155, [1, 512, 768]);  div_155 = None
    div_156: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_78, 768);  expand_78 = None
    pow_41: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_38, 1.0);  sub_38 = None
    mul_302: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_41, 2.0);  pow_41 = None
    mul_303: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_156, mul_302);  div_156 = mul_302 = None
    neg_49: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_303)
    sum_145: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_49, [2], True);  neg_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_190: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_154, mul_303);  div_154 = mul_303 = None
    add_191: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_144, sum_145);  sum_144 = sum_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_79: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_191, [1, 512, 768]);  add_191 = None
    div_157: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_79, 768);  expand_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_192: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_190, div_157);  add_190 = div_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_79: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_91: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_19, scalar_tensor_79, add_192);  convert_element_type_19 = scalar_tensor_79 = None
    mul_304: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_91, 1.1111111111111112);  where_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_406: "f32[512, 768]" = torch.ops.aten.view.default(mul_304, [512, 768]);  mul_304 = None
    permute_333: "f32[768, 768]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    mm_74: "f32[512, 768]" = torch.ops.aten.mm.default(view_406, permute_333);  permute_333 = None
    permute_334: "f32[768, 512]" = torch.ops.aten.permute.default(view_406, [1, 0])
    mm_75: "f32[768, 768]" = torch.ops.aten.mm.default(permute_334, view_84);  permute_334 = view_84 = None
    permute_335: "f32[768, 768]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_146: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_406, [0], True);  view_406 = None
    view_407: "f32[768]" = torch.ops.aten.view.default(sum_146, [768]);  sum_146 = None
    permute_336: "f32[768, 768]" = torch.ops.aten.permute.default(permute_335, [1, 0]);  permute_335 = None
    view_408: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_74, [1, 512, 768]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_409: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_408, [1, 512, 12, 64]);  view_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_337: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_409, [0, 2, 1, 3]);  view_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_410: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_337, [12, 512, 64]);  permute_337 = None
    permute_338: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_80, [0, 2, 1]);  view_80 = None
    bmm_52: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_338, view_410);  permute_338 = None
    permute_339: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_81, [0, 2, 1]);  view_81 = None
    bmm_53: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_410, permute_339);  view_410 = permute_339 = None
    view_411: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_52, [1, 12, 512, 64]);  bmm_52 = None
    view_412: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_53, [1, 12, 512, 512]);  bmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_80: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_92: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_18, scalar_tensor_80, view_412);  convert_element_type_18 = scalar_tensor_80 = view_412 = None
    mul_305: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_92, 1.1111111111111112);  where_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_79: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(where_22);  where_22 = None
    alias_80: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_79);  alias_79 = None
    mul_306: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_305, alias_80);  mul_305 = None
    sum_147: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_306, [-1], True)
    mul_307: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_80, sum_147);  alias_80 = sum_147 = None
    sub_112: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_306, mul_307);  mul_306 = mul_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_413: "f32[12, 512, 512]" = torch.ops.aten.view.default(sub_112, [12, 512, 512]);  sub_112 = None
    permute_340: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_77, [0, 2, 1]);  view_77 = None
    bmm_54: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_340, view_413);  permute_340 = None
    permute_341: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
    bmm_55: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_413, permute_341);  view_413 = permute_341 = None
    view_414: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_54, [1, 12, 64, 512]);  bmm_54 = None
    view_415: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_55, [1, 12, 512, 64]);  bmm_55 = None
    permute_342: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_414, [0, 1, 3, 2]);  view_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_158: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(view_415, sqrt_13);  view_415 = sqrt_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    sum_148: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(view_411, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_343: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_148, [0, 2, 1, 3]);  sum_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_416: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_343, [1, 1, 768]);  permute_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    full_18: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_14: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_18, view_416, 2, 0, 9223372036854775807);  full_18 = view_416 = None
    squeeze_33: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_14, 1);  slice_scatter_14 = None
    squeeze_34: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_33, 0);  squeeze_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    sum_149: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(div_158, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_344: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_149, [0, 2, 1, 3]);  sum_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_417: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_344, [1, 1, 768]);  permute_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    full_19: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_15: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_19, view_417, 2, 0, 9223372036854775807);  full_19 = view_417 = None
    squeeze_35: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_15, 1);  slice_scatter_15 = None
    squeeze_36: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_35, 0);  squeeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    cat_8: "f32[1, 12, 512, 192]" = torch.ops.aten.cat.default([div_158, permute_342, view_411], 3);  div_158 = permute_342 = view_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_345: "f32[1, 512, 12, 192]" = torch.ops.aten.permute.default(cat_8, [0, 2, 1, 3]);  cat_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    clone_21: "f32[1, 512, 12, 192]" = torch.ops.aten.clone.default(permute_345, memory_format = torch.contiguous_format);  permute_345 = None
    view_418: "f32[1, 512, 2304]" = torch.ops.aten.view.default(clone_21, [1, 512, 2304]);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_419: "f32[512, 2304]" = torch.ops.aten.view.default(view_418, [512, 2304]);  view_418 = None
    permute_346: "f32[2304, 512]" = torch.ops.aten.permute.default(view_419, [1, 0])
    mm_76: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_346, view_72);  permute_346 = view_72 = None
    permute_347: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_76, [1, 0]);  mm_76 = None
    permute_348: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
    mm_77: "f32[512, 768]" = torch.ops.aten.mm.default(view_419, permute_348);  view_419 = permute_348 = None
    view_420: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_77, [1, 512, 768]);  mm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    add_193: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_192, view_420);  add_192 = view_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_349: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_347, [1, 0]);  permute_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_150: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_193, [0, 1], True)
    view_421: "f32[768]" = torch.ops.aten.view.default(sum_150, [768]);  sum_150 = None
    mul_308: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_193, primals_25);  primals_25 = None
    mul_309: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_193, div_16);  add_193 = div_16 = None
    sum_151: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_309, [0, 1], True);  mul_309 = None
    view_422: "f32[768]" = torch.ops.aten.view.default(sum_151, [768]);  sum_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_159: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_34, sqrt_12);  sub_34 = None
    div_160: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_159, sqrt_12);  div_159 = None
    neg_50: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_308)
    mul_310: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_50, div_160);  neg_50 = div_160 = None
    div_161: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_308, sqrt_12);  mul_308 = sqrt_12 = None
    sum_152: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_310, [2], True);  mul_310 = None
    alias_81: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    mul_311: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_81, 2);  alias_81 = None
    div_162: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_152, mul_311);  sum_152 = mul_311 = None
    neg_51: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_161)
    sum_153: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_51, [2], True);  neg_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_80: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_162, [1, 512, 768]);  div_162 = None
    div_163: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_80, 768);  expand_80 = None
    pow_42: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_33, 1.0);  sub_33 = None
    mul_312: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_42, 2.0);  pow_42 = None
    mul_313: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_163, mul_312);  div_163 = mul_312 = None
    neg_52: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_313)
    sum_154: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_52, [2], True);  neg_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_194: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_161, mul_313);  div_161 = mul_313 = None
    add_195: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_153, sum_154);  sum_153 = sum_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_81: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_195, [1, 512, 768]);  add_195 = None
    div_164: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_81, 768);  expand_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_196: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_194, div_164);  add_194 = div_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_81: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_93: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_16, scalar_tensor_81, add_196);  convert_element_type_16 = scalar_tensor_81 = None
    mul_314: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_93, 1.1111111111111112);  where_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_423: "f32[512, 768]" = torch.ops.aten.view.default(mul_314, [512, 768]);  mul_314 = None
    permute_350: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
    mm_78: "f32[512, 3072]" = torch.ops.aten.mm.default(view_423, permute_350);  permute_350 = None
    permute_351: "f32[768, 512]" = torch.ops.aten.permute.default(view_423, [1, 0])
    mm_79: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_351, view_70);  permute_351 = view_70 = None
    permute_352: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_155: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_423, [0], True);  view_423 = None
    view_424: "f32[768]" = torch.ops.aten.view.default(sum_155, [768]);  sum_155 = None
    permute_353: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_352, [1, 0]);  permute_352 = None
    view_425: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_78, [1, 512, 3072]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_315: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_69, 0.7071067811865476)
    erf_20: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_315);  mul_315 = None
    add_197: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_316: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_197, 0.5);  add_197 = None
    mul_317: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_69, view_69)
    mul_318: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_317, -0.5);  mul_317 = None
    exp_24: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_318);  mul_318 = None
    mul_319: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
    mul_320: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_69, mul_319);  view_69 = mul_319 = None
    add_198: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_316, mul_320);  mul_316 = mul_320 = None
    mul_321: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_425, add_198);  view_425 = add_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_426: "f32[512, 3072]" = torch.ops.aten.view.default(mul_321, [512, 3072]);  mul_321 = None
    permute_354: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
    mm_80: "f32[512, 768]" = torch.ops.aten.mm.default(view_426, permute_354);  permute_354 = None
    permute_355: "f32[3072, 512]" = torch.ops.aten.permute.default(view_426, [1, 0])
    mm_81: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_355, view_68);  permute_355 = view_68 = None
    permute_356: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_156: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_426, [0], True);  view_426 = None
    view_427: "f32[3072]" = torch.ops.aten.view.default(sum_156, [3072]);  sum_156 = None
    permute_357: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_356, [1, 0]);  permute_356 = None
    view_428: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_80, [1, 512, 768]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    add_199: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_196, view_428);  add_196 = view_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_157: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_199, [0, 1], True)
    view_429: "f32[768]" = torch.ops.aten.view.default(sum_157, [768]);  sum_157 = None
    mul_322: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_199, primals_23);  primals_23 = None
    mul_323: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_199, div_15);  add_199 = div_15 = None
    sum_158: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_323, [0, 1], True);  mul_323 = None
    view_430: "f32[768]" = torch.ops.aten.view.default(sum_158, [768]);  sum_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_165: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_31, sqrt_11);  sub_31 = None
    div_166: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_165, sqrt_11);  div_165 = None
    neg_53: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_322)
    mul_324: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_53, div_166);  neg_53 = div_166 = None
    div_167: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_322, sqrt_11);  mul_322 = sqrt_11 = None
    sum_159: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_324, [2], True);  mul_324 = None
    alias_82: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    mul_325: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_82, 2);  alias_82 = None
    div_168: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_159, mul_325);  sum_159 = mul_325 = None
    neg_54: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_167)
    sum_160: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_54, [2], True);  neg_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_82: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_168, [1, 512, 768]);  div_168 = None
    div_169: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_82, 768);  expand_82 = None
    pow_43: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_30, 1.0);  sub_30 = None
    mul_326: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_43, 2.0);  pow_43 = None
    mul_327: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_169, mul_326);  div_169 = mul_326 = None
    neg_55: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_327)
    sum_161: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_55, [2], True);  neg_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_200: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_167, mul_327);  div_167 = mul_327 = None
    add_201: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_160, sum_161);  sum_160 = sum_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_83: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_201, [1, 512, 768]);  add_201 = None
    div_170: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_83, 768);  expand_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_202: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_200, div_170);  add_200 = div_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_82: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_94: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_15, scalar_tensor_82, add_202);  convert_element_type_15 = scalar_tensor_82 = None
    mul_328: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_94, 1.1111111111111112);  where_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_431: "f32[512, 768]" = torch.ops.aten.view.default(mul_328, [512, 768]);  mul_328 = None
    permute_358: "f32[768, 768]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
    mm_82: "f32[512, 768]" = torch.ops.aten.mm.default(view_431, permute_358);  permute_358 = None
    permute_359: "f32[768, 512]" = torch.ops.aten.permute.default(view_431, [1, 0])
    mm_83: "f32[768, 768]" = torch.ops.aten.mm.default(permute_359, view_66);  permute_359 = view_66 = None
    permute_360: "f32[768, 768]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_162: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_431, [0], True);  view_431 = None
    view_432: "f32[768]" = torch.ops.aten.view.default(sum_162, [768]);  sum_162 = None
    permute_361: "f32[768, 768]" = torch.ops.aten.permute.default(permute_360, [1, 0]);  permute_360 = None
    view_433: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_82, [1, 512, 768]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_434: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_433, [1, 512, 12, 64]);  view_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_362: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_434, [0, 2, 1, 3]);  view_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_435: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_362, [12, 512, 64]);  permute_362 = None
    permute_363: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_62, [0, 2, 1]);  view_62 = None
    bmm_56: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_363, view_435);  permute_363 = None
    permute_364: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_63, [0, 2, 1]);  view_63 = None
    bmm_57: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_435, permute_364);  view_435 = permute_364 = None
    view_436: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_56, [1, 12, 512, 64]);  bmm_56 = None
    view_437: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_57, [1, 12, 512, 512]);  bmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_83: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_95: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_14, scalar_tensor_83, view_437);  convert_element_type_14 = scalar_tensor_83 = view_437 = None
    mul_329: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_95, 1.1111111111111112);  where_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_84: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(where_17);  where_17 = None
    alias_85: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_84);  alias_84 = None
    mul_330: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_329, alias_85);  mul_329 = None
    sum_163: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_330, [-1], True)
    mul_331: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_85, sum_163);  alias_85 = sum_163 = None
    sub_113: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_330, mul_331);  mul_330 = mul_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_438: "f32[12, 512, 512]" = torch.ops.aten.view.default(sub_113, [12, 512, 512]);  sub_113 = None
    permute_365: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_59, [0, 2, 1]);  view_59 = None
    bmm_58: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_365, view_438);  permute_365 = None
    permute_366: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_60, [0, 2, 1]);  view_60 = None
    bmm_59: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_438, permute_366);  view_438 = permute_366 = None
    view_439: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_58, [1, 12, 64, 512]);  bmm_58 = None
    view_440: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_59, [1, 12, 512, 64]);  bmm_59 = None
    permute_367: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_439, [0, 1, 3, 2]);  view_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_171: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(view_440, sqrt_10);  view_440 = sqrt_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    sum_164: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(view_436, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_368: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_164, [0, 2, 1, 3]);  sum_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_441: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_368, [1, 1, 768]);  permute_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    full_20: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_16: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_20, view_441, 2, 0, 9223372036854775807);  full_20 = view_441 = None
    squeeze_37: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_16, 1);  slice_scatter_16 = None
    squeeze_38: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_37, 0);  squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    sum_165: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(div_171, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_369: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_165, [0, 2, 1, 3]);  sum_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_442: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_369, [1, 1, 768]);  permute_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    full_21: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_17: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_21, view_442, 2, 0, 9223372036854775807);  full_21 = view_442 = None
    squeeze_39: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_17, 1);  slice_scatter_17 = None
    squeeze_40: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_39, 0);  squeeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    cat_9: "f32[1, 12, 512, 192]" = torch.ops.aten.cat.default([div_171, permute_367, view_436], 3);  div_171 = permute_367 = view_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_370: "f32[1, 512, 12, 192]" = torch.ops.aten.permute.default(cat_9, [0, 2, 1, 3]);  cat_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    clone_22: "f32[1, 512, 12, 192]" = torch.ops.aten.clone.default(permute_370, memory_format = torch.contiguous_format);  permute_370 = None
    view_443: "f32[1, 512, 2304]" = torch.ops.aten.view.default(clone_22, [1, 512, 2304]);  clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_444: "f32[512, 2304]" = torch.ops.aten.view.default(view_443, [512, 2304]);  view_443 = None
    permute_371: "f32[2304, 512]" = torch.ops.aten.permute.default(view_444, [1, 0])
    mm_84: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_371, view_54);  permute_371 = view_54 = None
    permute_372: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_84, [1, 0]);  mm_84 = None
    permute_373: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
    mm_85: "f32[512, 768]" = torch.ops.aten.mm.default(view_444, permute_373);  view_444 = permute_373 = None
    view_445: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_85, [1, 512, 768]);  mm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    add_203: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_202, view_445);  add_202 = view_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_374: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_372, [1, 0]);  permute_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_166: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_203, [0, 1], True)
    view_446: "f32[768]" = torch.ops.aten.view.default(sum_166, [768]);  sum_166 = None
    mul_332: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_203, primals_19);  primals_19 = None
    mul_333: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_203, div_12);  add_203 = div_12 = None
    sum_167: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_333, [0, 1], True);  mul_333 = None
    view_447: "f32[768]" = torch.ops.aten.view.default(sum_167, [768]);  sum_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_172: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_26, sqrt_9);  sub_26 = None
    div_173: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_172, sqrt_9);  div_172 = None
    neg_56: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_332)
    mul_334: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_56, div_173);  neg_56 = div_173 = None
    div_174: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_332, sqrt_9);  mul_332 = sqrt_9 = None
    sum_168: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_334, [2], True);  mul_334 = None
    alias_86: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    mul_335: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_86, 2);  alias_86 = None
    div_175: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_168, mul_335);  sum_168 = mul_335 = None
    neg_57: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_174)
    sum_169: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_57, [2], True);  neg_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_84: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_175, [1, 512, 768]);  div_175 = None
    div_176: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_84, 768);  expand_84 = None
    pow_44: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_25, 1.0);  sub_25 = None
    mul_336: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_44, 2.0);  pow_44 = None
    mul_337: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_176, mul_336);  div_176 = mul_336 = None
    neg_58: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_337)
    sum_170: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_58, [2], True);  neg_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_204: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_174, mul_337);  div_174 = mul_337 = None
    add_205: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_169, sum_170);  sum_169 = sum_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_85: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_205, [1, 512, 768]);  add_205 = None
    div_177: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_85, 768);  expand_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_206: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_204, div_177);  add_204 = div_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_84: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_96: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_12, scalar_tensor_84, add_206);  convert_element_type_12 = scalar_tensor_84 = None
    mul_338: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_96, 1.1111111111111112);  where_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_448: "f32[512, 768]" = torch.ops.aten.view.default(mul_338, [512, 768]);  mul_338 = None
    permute_375: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
    mm_86: "f32[512, 3072]" = torch.ops.aten.mm.default(view_448, permute_375);  permute_375 = None
    permute_376: "f32[768, 512]" = torch.ops.aten.permute.default(view_448, [1, 0])
    mm_87: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_376, view_52);  permute_376 = view_52 = None
    permute_377: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
    sum_171: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_448, [0], True);  view_448 = None
    view_449: "f32[768]" = torch.ops.aten.view.default(sum_171, [768]);  sum_171 = None
    permute_378: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_377, [1, 0]);  permute_377 = None
    view_450: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_86, [1, 512, 3072]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_339: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_51, 0.7071067811865476)
    erf_21: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_339);  mul_339 = None
    add_207: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_340: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_207, 0.5);  add_207 = None
    mul_341: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_51, view_51)
    mul_342: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_341, -0.5);  mul_341 = None
    exp_25: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_342);  mul_342 = None
    mul_343: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
    mul_344: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_51, mul_343);  view_51 = mul_343 = None
    add_208: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_340, mul_344);  mul_340 = mul_344 = None
    mul_345: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_450, add_208);  view_450 = add_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_451: "f32[512, 3072]" = torch.ops.aten.view.default(mul_345, [512, 3072]);  mul_345 = None
    permute_379: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    mm_88: "f32[512, 768]" = torch.ops.aten.mm.default(view_451, permute_379);  permute_379 = None
    permute_380: "f32[3072, 512]" = torch.ops.aten.permute.default(view_451, [1, 0])
    mm_89: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_380, view_50);  permute_380 = view_50 = None
    permute_381: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    sum_172: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_451, [0], True);  view_451 = None
    view_452: "f32[3072]" = torch.ops.aten.view.default(sum_172, [3072]);  sum_172 = None
    permute_382: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_381, [1, 0]);  permute_381 = None
    view_453: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_88, [1, 512, 768]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    add_209: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_206, view_453);  add_206 = view_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_173: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_209, [0, 1], True)
    view_454: "f32[768]" = torch.ops.aten.view.default(sum_173, [768]);  sum_173 = None
    mul_346: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_209, primals_17);  primals_17 = None
    mul_347: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_209, div_11);  add_209 = div_11 = None
    sum_174: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_347, [0, 1], True);  mul_347 = None
    view_455: "f32[768]" = torch.ops.aten.view.default(sum_174, [768]);  sum_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_178: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_23, sqrt_8);  sub_23 = None
    div_179: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_178, sqrt_8);  div_178 = None
    neg_59: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_346)
    mul_348: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_59, div_179);  neg_59 = div_179 = None
    div_180: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_346, sqrt_8);  mul_346 = sqrt_8 = None
    sum_175: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_348, [2], True);  mul_348 = None
    alias_87: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    mul_349: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_87, 2);  alias_87 = None
    div_181: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_175, mul_349);  sum_175 = mul_349 = None
    neg_60: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_180)
    sum_176: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_60, [2], True);  neg_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_86: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_181, [1, 512, 768]);  div_181 = None
    div_182: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_86, 768);  expand_86 = None
    pow_45: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_22, 1.0);  sub_22 = None
    mul_350: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_45, 2.0);  pow_45 = None
    mul_351: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_182, mul_350);  div_182 = mul_350 = None
    neg_61: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_351)
    sum_177: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_61, [2], True);  neg_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_210: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_180, mul_351);  div_180 = mul_351 = None
    add_211: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_176, sum_177);  sum_176 = sum_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_87: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_211, [1, 512, 768]);  add_211 = None
    div_183: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_87, 768);  expand_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_212: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_210, div_183);  add_210 = div_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_85: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_97: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_11, scalar_tensor_85, add_212);  convert_element_type_11 = scalar_tensor_85 = None
    mul_352: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_97, 1.1111111111111112);  where_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_456: "f32[512, 768]" = torch.ops.aten.view.default(mul_352, [512, 768]);  mul_352 = None
    permute_383: "f32[768, 768]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    mm_90: "f32[512, 768]" = torch.ops.aten.mm.default(view_456, permute_383);  permute_383 = None
    permute_384: "f32[768, 512]" = torch.ops.aten.permute.default(view_456, [1, 0])
    mm_91: "f32[768, 768]" = torch.ops.aten.mm.default(permute_384, view_48);  permute_384 = view_48 = None
    permute_385: "f32[768, 768]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_178: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_456, [0], True);  view_456 = None
    view_457: "f32[768]" = torch.ops.aten.view.default(sum_178, [768]);  sum_178 = None
    permute_386: "f32[768, 768]" = torch.ops.aten.permute.default(permute_385, [1, 0]);  permute_385 = None
    view_458: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_90, [1, 512, 768]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_459: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_458, [1, 512, 12, 64]);  view_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_387: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_459, [0, 2, 1, 3]);  view_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_460: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_387, [12, 512, 64]);  permute_387 = None
    permute_388: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_44, [0, 2, 1]);  view_44 = None
    bmm_60: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_388, view_460);  permute_388 = None
    permute_389: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_45, [0, 2, 1]);  view_45 = None
    bmm_61: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_460, permute_389);  view_460 = permute_389 = None
    view_461: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_60, [1, 12, 512, 64]);  bmm_60 = None
    view_462: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_61, [1, 12, 512, 512]);  bmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_86: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_98: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_10, scalar_tensor_86, view_462);  convert_element_type_10 = scalar_tensor_86 = view_462 = None
    mul_353: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_98, 1.1111111111111112);  where_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_89: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(where_12);  where_12 = None
    alias_90: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_89);  alias_89 = None
    mul_354: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_353, alias_90);  mul_353 = None
    sum_179: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_354, [-1], True)
    mul_355: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_90, sum_179);  alias_90 = sum_179 = None
    sub_114: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_354, mul_355);  mul_354 = mul_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_463: "f32[12, 512, 512]" = torch.ops.aten.view.default(sub_114, [12, 512, 512]);  sub_114 = None
    permute_390: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_41, [0, 2, 1]);  view_41 = None
    bmm_62: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_390, view_463);  permute_390 = None
    permute_391: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_42, [0, 2, 1]);  view_42 = None
    bmm_63: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_463, permute_391);  view_463 = permute_391 = None
    view_464: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_62, [1, 12, 64, 512]);  bmm_62 = None
    view_465: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_63, [1, 12, 512, 64]);  bmm_63 = None
    permute_392: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_464, [0, 1, 3, 2]);  view_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_184: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(view_465, sqrt_7);  view_465 = sqrt_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    sum_180: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(view_461, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_393: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_180, [0, 2, 1, 3]);  sum_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_466: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_393, [1, 1, 768]);  permute_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    full_22: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_18: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_22, view_466, 2, 0, 9223372036854775807);  full_22 = view_466 = None
    squeeze_41: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_18, 1);  slice_scatter_18 = None
    squeeze_42: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_41, 0);  squeeze_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    sum_181: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(div_184, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_394: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_181, [0, 2, 1, 3]);  sum_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_467: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_394, [1, 1, 768]);  permute_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    full_23: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_19: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_23, view_467, 2, 0, 9223372036854775807);  full_23 = view_467 = None
    squeeze_43: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_19, 1);  slice_scatter_19 = None
    squeeze_44: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_43, 0);  squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    cat_10: "f32[1, 12, 512, 192]" = torch.ops.aten.cat.default([div_184, permute_392, view_461], 3);  div_184 = permute_392 = view_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_395: "f32[1, 512, 12, 192]" = torch.ops.aten.permute.default(cat_10, [0, 2, 1, 3]);  cat_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    clone_23: "f32[1, 512, 12, 192]" = torch.ops.aten.clone.default(permute_395, memory_format = torch.contiguous_format);  permute_395 = None
    view_468: "f32[1, 512, 2304]" = torch.ops.aten.view.default(clone_23, [1, 512, 2304]);  clone_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_469: "f32[512, 2304]" = torch.ops.aten.view.default(view_468, [512, 2304]);  view_468 = None
    permute_396: "f32[2304, 512]" = torch.ops.aten.permute.default(view_469, [1, 0])
    mm_92: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_396, view_36);  permute_396 = view_36 = None
    permute_397: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_92, [1, 0]);  mm_92 = None
    permute_398: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
    mm_93: "f32[512, 768]" = torch.ops.aten.mm.default(view_469, permute_398);  view_469 = permute_398 = None
    view_470: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_93, [1, 512, 768]);  mm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    add_213: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_212, view_470);  add_212 = view_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_399: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_397, [1, 0]);  permute_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_182: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_213, [0, 1], True)
    view_471: "f32[768]" = torch.ops.aten.view.default(sum_182, [768]);  sum_182 = None
    mul_356: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_213, primals_13);  primals_13 = None
    mul_357: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_213, div_8);  add_213 = div_8 = None
    sum_183: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_357, [0, 1], True);  mul_357 = None
    view_472: "f32[768]" = torch.ops.aten.view.default(sum_183, [768]);  sum_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_185: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_18, sqrt_6);  sub_18 = None
    div_186: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_185, sqrt_6);  div_185 = None
    neg_62: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_356)
    mul_358: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_62, div_186);  neg_62 = div_186 = None
    div_187: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_356, sqrt_6);  mul_356 = sqrt_6 = None
    sum_184: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_358, [2], True);  mul_358 = None
    alias_91: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    mul_359: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_91, 2);  alias_91 = None
    div_188: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_184, mul_359);  sum_184 = mul_359 = None
    neg_63: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_187)
    sum_185: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_63, [2], True);  neg_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_88: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_188, [1, 512, 768]);  div_188 = None
    div_189: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_88, 768);  expand_88 = None
    pow_46: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_17, 1.0);  sub_17 = None
    mul_360: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_46, 2.0);  pow_46 = None
    mul_361: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_189, mul_360);  div_189 = mul_360 = None
    neg_64: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_361)
    sum_186: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_64, [2], True);  neg_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_214: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_187, mul_361);  div_187 = mul_361 = None
    add_215: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_185, sum_186);  sum_185 = sum_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_89: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_215, [1, 512, 768]);  add_215 = None
    div_190: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_89, 768);  expand_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_216: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_214, div_190);  add_214 = div_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_87: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_99: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_8, scalar_tensor_87, add_216);  convert_element_type_8 = scalar_tensor_87 = None
    mul_362: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_99, 1.1111111111111112);  where_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_473: "f32[512, 768]" = torch.ops.aten.view.default(mul_362, [512, 768]);  mul_362 = None
    permute_400: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    mm_94: "f32[512, 3072]" = torch.ops.aten.mm.default(view_473, permute_400);  permute_400 = None
    permute_401: "f32[768, 512]" = torch.ops.aten.permute.default(view_473, [1, 0])
    mm_95: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_401, view_34);  permute_401 = view_34 = None
    permute_402: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_187: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_473, [0], True);  view_473 = None
    view_474: "f32[768]" = torch.ops.aten.view.default(sum_187, [768]);  sum_187 = None
    permute_403: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_402, [1, 0]);  permute_402 = None
    view_475: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_94, [1, 512, 3072]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_363: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_33, 0.7071067811865476)
    erf_22: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_363);  mul_363 = None
    add_217: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_364: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_217, 0.5);  add_217 = None
    mul_365: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_33, view_33)
    mul_366: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_365, -0.5);  mul_365 = None
    exp_26: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_366);  mul_366 = None
    mul_367: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
    mul_368: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_33, mul_367);  view_33 = mul_367 = None
    add_218: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_364, mul_368);  mul_364 = mul_368 = None
    mul_369: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_475, add_218);  view_475 = add_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_476: "f32[512, 3072]" = torch.ops.aten.view.default(mul_369, [512, 3072]);  mul_369 = None
    permute_404: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    mm_96: "f32[512, 768]" = torch.ops.aten.mm.default(view_476, permute_404);  permute_404 = None
    permute_405: "f32[3072, 512]" = torch.ops.aten.permute.default(view_476, [1, 0])
    mm_97: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_405, view_32);  permute_405 = view_32 = None
    permute_406: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_188: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_476, [0], True);  view_476 = None
    view_477: "f32[3072]" = torch.ops.aten.view.default(sum_188, [3072]);  sum_188 = None
    permute_407: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_406, [1, 0]);  permute_406 = None
    view_478: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_96, [1, 512, 768]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    add_219: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_216, view_478);  add_216 = view_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_189: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_219, [0, 1], True)
    view_479: "f32[768]" = torch.ops.aten.view.default(sum_189, [768]);  sum_189 = None
    mul_370: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_219, primals_11);  primals_11 = None
    mul_371: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_219, div_7);  add_219 = div_7 = None
    sum_190: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_371, [0, 1], True);  mul_371 = None
    view_480: "f32[768]" = torch.ops.aten.view.default(sum_190, [768]);  sum_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_191: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_15, sqrt_5);  sub_15 = None
    div_192: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_191, sqrt_5);  div_191 = None
    neg_65: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_370)
    mul_372: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_65, div_192);  neg_65 = div_192 = None
    div_193: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_370, sqrt_5);  mul_370 = sqrt_5 = None
    sum_191: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_372, [2], True);  mul_372 = None
    alias_92: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    mul_373: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_92, 2);  alias_92 = None
    div_194: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_191, mul_373);  sum_191 = mul_373 = None
    neg_66: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_193)
    sum_192: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_66, [2], True);  neg_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_90: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_194, [1, 512, 768]);  div_194 = None
    div_195: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_90, 768);  expand_90 = None
    pow_47: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_14, 1.0);  sub_14 = None
    mul_374: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_47, 2.0);  pow_47 = None
    mul_375: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_195, mul_374);  div_195 = mul_374 = None
    neg_67: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_375)
    sum_193: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_67, [2], True);  neg_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_220: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_193, mul_375);  div_193 = mul_375 = None
    add_221: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_192, sum_193);  sum_192 = sum_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_91: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_221, [1, 512, 768]);  add_221 = None
    div_196: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_91, 768);  expand_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_222: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_220, div_196);  add_220 = div_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_88: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_100: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_7, scalar_tensor_88, add_222);  convert_element_type_7 = scalar_tensor_88 = None
    mul_376: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_100, 1.1111111111111112);  where_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_481: "f32[512, 768]" = torch.ops.aten.view.default(mul_376, [512, 768]);  mul_376 = None
    permute_408: "f32[768, 768]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    mm_98: "f32[512, 768]" = torch.ops.aten.mm.default(view_481, permute_408);  permute_408 = None
    permute_409: "f32[768, 512]" = torch.ops.aten.permute.default(view_481, [1, 0])
    mm_99: "f32[768, 768]" = torch.ops.aten.mm.default(permute_409, view_30);  permute_409 = view_30 = None
    permute_410: "f32[768, 768]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    sum_194: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_481, [0], True);  view_481 = None
    view_482: "f32[768]" = torch.ops.aten.view.default(sum_194, [768]);  sum_194 = None
    permute_411: "f32[768, 768]" = torch.ops.aten.permute.default(permute_410, [1, 0]);  permute_410 = None
    view_483: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_98, [1, 512, 768]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_484: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_483, [1, 512, 12, 64]);  view_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_412: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_484, [0, 2, 1, 3]);  view_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_485: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_412, [12, 512, 64]);  permute_412 = None
    permute_413: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_26, [0, 2, 1]);  view_26 = None
    bmm_64: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_413, view_485);  permute_413 = None
    permute_414: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_27, [0, 2, 1]);  view_27 = None
    bmm_65: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_485, permute_414);  view_485 = permute_414 = None
    view_486: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_64, [1, 12, 512, 64]);  bmm_64 = None
    view_487: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_65, [1, 12, 512, 512]);  bmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_89: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_101: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_6, scalar_tensor_89, view_487);  convert_element_type_6 = scalar_tensor_89 = view_487 = None
    mul_377: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_101, 1.1111111111111112);  where_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_94: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(where_7);  where_7 = None
    alias_95: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_94);  alias_94 = None
    mul_378: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_377, alias_95);  mul_377 = None
    sum_195: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_378, [-1], True)
    mul_379: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_95, sum_195);  alias_95 = sum_195 = None
    sub_115: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_378, mul_379);  mul_378 = mul_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_488: "f32[12, 512, 512]" = torch.ops.aten.view.default(sub_115, [12, 512, 512]);  sub_115 = None
    permute_415: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_23, [0, 2, 1]);  view_23 = None
    bmm_66: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_415, view_488);  permute_415 = None
    permute_416: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_24, [0, 2, 1]);  view_24 = None
    bmm_67: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_488, permute_416);  view_488 = permute_416 = None
    view_489: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_66, [1, 12, 64, 512]);  bmm_66 = None
    view_490: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_67, [1, 12, 512, 64]);  bmm_67 = None
    permute_417: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_489, [0, 1, 3, 2]);  view_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_197: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(view_490, sqrt_4);  view_490 = sqrt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    sum_196: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(view_486, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_418: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_196, [0, 2, 1, 3]);  sum_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_491: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_418, [1, 1, 768]);  permute_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    full_24: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_20: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_24, view_491, 2, 0, 9223372036854775807);  full_24 = view_491 = None
    squeeze_45: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_20, 1);  slice_scatter_20 = None
    squeeze_46: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_45, 0);  squeeze_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    sum_197: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(div_197, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_419: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_197, [0, 2, 1, 3]);  sum_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_492: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_419, [1, 1, 768]);  permute_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    full_25: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_21: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_25, view_492, 2, 0, 9223372036854775807);  full_25 = view_492 = None
    squeeze_47: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_21, 1);  slice_scatter_21 = None
    squeeze_48: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_47, 0);  squeeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    cat_11: "f32[1, 12, 512, 192]" = torch.ops.aten.cat.default([div_197, permute_417, view_486], 3);  div_197 = permute_417 = view_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_420: "f32[1, 512, 12, 192]" = torch.ops.aten.permute.default(cat_11, [0, 2, 1, 3]);  cat_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    clone_24: "f32[1, 512, 12, 192]" = torch.ops.aten.clone.default(permute_420, memory_format = torch.contiguous_format);  permute_420 = None
    view_493: "f32[1, 512, 2304]" = torch.ops.aten.view.default(clone_24, [1, 512, 2304]);  clone_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_494: "f32[512, 2304]" = torch.ops.aten.view.default(view_493, [512, 2304]);  view_493 = None
    permute_421: "f32[2304, 512]" = torch.ops.aten.permute.default(view_494, [1, 0])
    mm_100: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_421, view_18);  permute_421 = view_18 = None
    permute_422: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_100, [1, 0]);  mm_100 = None
    permute_423: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    mm_101: "f32[512, 768]" = torch.ops.aten.mm.default(view_494, permute_423);  view_494 = permute_423 = None
    view_495: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_101, [1, 512, 768]);  mm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    add_223: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_222, view_495);  add_222 = view_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_424: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_422, [1, 0]);  permute_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_198: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_223, [0, 1], True)
    view_496: "f32[768]" = torch.ops.aten.view.default(sum_198, [768]);  sum_198 = None
    mul_380: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_223, primals_7);  primals_7 = None
    mul_381: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_223, div_4);  add_223 = div_4 = None
    sum_199: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_381, [0, 1], True);  mul_381 = None
    view_497: "f32[768]" = torch.ops.aten.view.default(sum_199, [768]);  sum_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_198: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_10, sqrt_3);  sub_10 = None
    div_199: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_198, sqrt_3);  div_198 = None
    neg_68: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_380)
    mul_382: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_68, div_199);  neg_68 = div_199 = None
    div_200: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_380, sqrt_3);  mul_380 = sqrt_3 = None
    sum_200: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_382, [2], True);  mul_382 = None
    alias_96: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    mul_383: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_96, 2);  alias_96 = None
    div_201: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_200, mul_383);  sum_200 = mul_383 = None
    neg_69: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_200)
    sum_201: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_69, [2], True);  neg_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_92: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_201, [1, 512, 768]);  div_201 = None
    div_202: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_92, 768);  expand_92 = None
    pow_48: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_9, 1.0);  sub_9 = None
    mul_384: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_48, 2.0);  pow_48 = None
    mul_385: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_202, mul_384);  div_202 = mul_384 = None
    neg_70: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_385)
    sum_202: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_70, [2], True);  neg_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_224: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_200, mul_385);  div_200 = mul_385 = None
    add_225: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_201, sum_202);  sum_201 = sum_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_93: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_225, [1, 512, 768]);  add_225 = None
    div_203: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_93, 768);  expand_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_226: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_224, div_203);  add_224 = div_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_90: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_102: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_4, scalar_tensor_90, add_226);  convert_element_type_4 = scalar_tensor_90 = None
    mul_386: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_102, 1.1111111111111112);  where_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_498: "f32[512, 768]" = torch.ops.aten.view.default(mul_386, [512, 768]);  mul_386 = None
    permute_425: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    mm_102: "f32[512, 3072]" = torch.ops.aten.mm.default(view_498, permute_425);  permute_425 = None
    permute_426: "f32[768, 512]" = torch.ops.aten.permute.default(view_498, [1, 0])
    mm_103: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_426, view_16);  permute_426 = view_16 = None
    permute_427: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
    sum_203: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_498, [0], True);  view_498 = None
    view_499: "f32[768]" = torch.ops.aten.view.default(sum_203, [768]);  sum_203 = None
    permute_428: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_427, [1, 0]);  permute_427 = None
    view_500: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_102, [1, 512, 3072]);  mm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_387: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_15, 0.7071067811865476)
    erf_23: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_387);  mul_387 = None
    add_227: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_388: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_227, 0.5);  add_227 = None
    mul_389: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_15, view_15)
    mul_390: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_389, -0.5);  mul_389 = None
    exp_27: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_390);  mul_390 = None
    mul_391: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_27, 0.3989422804014327);  exp_27 = None
    mul_392: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_15, mul_391);  view_15 = mul_391 = None
    add_228: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_388, mul_392);  mul_388 = mul_392 = None
    mul_393: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_500, add_228);  view_500 = add_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_501: "f32[512, 3072]" = torch.ops.aten.view.default(mul_393, [512, 3072]);  mul_393 = None
    permute_429: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    mm_104: "f32[512, 768]" = torch.ops.aten.mm.default(view_501, permute_429);  permute_429 = None
    permute_430: "f32[3072, 512]" = torch.ops.aten.permute.default(view_501, [1, 0])
    mm_105: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_430, view_14);  permute_430 = view_14 = None
    permute_431: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
    sum_204: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_501, [0], True);  view_501 = None
    view_502: "f32[3072]" = torch.ops.aten.view.default(sum_204, [3072]);  sum_204 = None
    permute_432: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_431, [1, 0]);  permute_431 = None
    view_503: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_104, [1, 512, 768]);  mm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    add_229: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_226, view_503);  add_226 = view_503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_205: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_229, [0, 1], True)
    view_504: "f32[768]" = torch.ops.aten.view.default(sum_205, [768]);  sum_205 = None
    mul_394: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_229, primals_5);  primals_5 = None
    mul_395: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_229, div_3);  add_229 = div_3 = None
    sum_206: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_395, [0, 1], True);  mul_395 = None
    view_505: "f32[768]" = torch.ops.aten.view.default(sum_206, [768]);  sum_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_204: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_7, sqrt_2);  sub_7 = None
    div_205: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_204, sqrt_2);  div_204 = None
    neg_71: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_394)
    mul_396: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_71, div_205);  neg_71 = div_205 = None
    div_206: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_394, sqrt_2);  mul_394 = sqrt_2 = None
    sum_207: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_396, [2], True);  mul_396 = None
    alias_97: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    mul_397: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_97, 2);  alias_97 = None
    div_207: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_207, mul_397);  sum_207 = mul_397 = None
    neg_72: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_206)
    sum_208: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_72, [2], True);  neg_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_94: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_207, [1, 512, 768]);  div_207 = None
    div_208: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_94, 768);  expand_94 = None
    pow_49: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_6, 1.0);  sub_6 = None
    mul_398: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_49, 2.0);  pow_49 = None
    mul_399: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_208, mul_398);  div_208 = mul_398 = None
    neg_73: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_399)
    sum_209: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_73, [2], True);  neg_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_230: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_206, mul_399);  div_206 = mul_399 = None
    add_231: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_208, sum_209);  sum_208 = sum_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_95: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_231, [1, 512, 768]);  add_231 = None
    div_209: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_95, 768);  expand_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_232: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_230, div_209);  add_230 = div_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_91: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_103: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_3, scalar_tensor_91, add_232);  convert_element_type_3 = scalar_tensor_91 = None
    mul_400: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_103, 1.1111111111111112);  where_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_506: "f32[512, 768]" = torch.ops.aten.view.default(mul_400, [512, 768]);  mul_400 = None
    permute_433: "f32[768, 768]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    mm_106: "f32[512, 768]" = torch.ops.aten.mm.default(view_506, permute_433);  permute_433 = None
    permute_434: "f32[768, 512]" = torch.ops.aten.permute.default(view_506, [1, 0])
    mm_107: "f32[768, 768]" = torch.ops.aten.mm.default(permute_434, view_12);  permute_434 = view_12 = None
    permute_435: "f32[768, 768]" = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
    sum_210: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_506, [0], True);  view_506 = None
    view_507: "f32[768]" = torch.ops.aten.view.default(sum_210, [768]);  sum_210 = None
    permute_436: "f32[768, 768]" = torch.ops.aten.permute.default(permute_435, [1, 0]);  permute_435 = None
    view_508: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_106, [1, 512, 768]);  mm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_509: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_508, [1, 512, 12, 64]);  view_508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_437: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_509, [0, 2, 1, 3]);  view_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_510: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_437, [12, 512, 64]);  permute_437 = None
    permute_438: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_8, [0, 2, 1]);  view_8 = None
    bmm_68: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_438, view_510);  permute_438 = None
    permute_439: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
    bmm_69: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_510, permute_439);  view_510 = permute_439 = None
    view_511: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_68, [1, 12, 512, 64]);  bmm_68 = None
    view_512: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_69, [1, 12, 512, 512]);  bmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_92: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_104: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_2, scalar_tensor_92, view_512);  convert_element_type_2 = scalar_tensor_92 = view_512 = None
    mul_401: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_104, 1.1111111111111112);  where_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_99: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(where_2);  where_2 = None
    alias_100: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_99);  alias_99 = None
    mul_402: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_401, alias_100);  mul_401 = None
    sum_211: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_402, [-1], True)
    mul_403: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_100, sum_211);  alias_100 = sum_211 = None
    sub_116: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_402, mul_403);  mul_402 = mul_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_513: "f32[12, 512, 512]" = torch.ops.aten.view.default(sub_116, [12, 512, 512]);  sub_116 = None
    permute_440: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_5, [0, 2, 1]);  view_5 = None
    bmm_70: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_440, view_513);  permute_440 = None
    permute_441: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1]);  view_6 = None
    bmm_71: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_513, permute_441);  view_513 = permute_441 = None
    view_514: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_70, [1, 12, 64, 512]);  bmm_70 = None
    view_515: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_71, [1, 12, 512, 64]);  bmm_71 = None
    permute_442: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_514, [0, 1, 3, 2]);  view_514 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_210: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(view_515, sqrt_1);  view_515 = sqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    sum_212: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(view_511, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_443: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_212, [0, 2, 1, 3]);  sum_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_516: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_443, [1, 1, 768]);  permute_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    full_26: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_22: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_26, view_516, 2, 0, 9223372036854775807);  full_26 = view_516 = None
    squeeze_49: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_22, 1);  slice_scatter_22 = None
    squeeze_50: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_49, 0);  squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    sum_213: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(div_210, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_444: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_213, [0, 2, 1, 3]);  sum_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_517: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_444, [1, 1, 768]);  permute_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    full_27: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_23: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_27, view_517, 2, 0, 9223372036854775807);  full_27 = view_517 = None
    squeeze_51: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_23, 1);  slice_scatter_23 = None
    squeeze_52: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_51, 0);  squeeze_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    cat_12: "f32[1, 12, 512, 192]" = torch.ops.aten.cat.default([div_210, permute_442, view_511], 3);  div_210 = permute_442 = view_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_445: "f32[1, 512, 12, 192]" = torch.ops.aten.permute.default(cat_12, [0, 2, 1, 3]);  cat_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    clone_25: "f32[1, 512, 12, 192]" = torch.ops.aten.clone.default(permute_445, memory_format = torch.contiguous_format);  permute_445 = None
    view_518: "f32[1, 512, 2304]" = torch.ops.aten.view.default(clone_25, [1, 512, 2304]);  clone_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_519: "f32[512, 2304]" = torch.ops.aten.view.default(view_518, [512, 2304]);  view_518 = None
    permute_446: "f32[2304, 512]" = torch.ops.aten.permute.default(view_519, [1, 0])
    mm_108: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_446, view);  permute_446 = view = None
    permute_447: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_108, [1, 0]);  mm_108 = None
    permute_448: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    mm_109: "f32[512, 768]" = torch.ops.aten.mm.default(view_519, permute_448);  view_519 = permute_448 = None
    view_520: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_109, [1, 512, 768]);  mm_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    add_233: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_232, view_520);  add_232 = view_520 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_449: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_447, [1, 0]);  permute_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_93: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_105: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type, scalar_tensor_93, add_233);  convert_element_type = scalar_tensor_93 = add_233 = None
    mul_404: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_105, 1.1111111111111112);  where_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:812, code: embeddings = embeddings * mask
    mul_405: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_404, unsqueeze);  mul_404 = unsqueeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_214: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_405, [0, 1], True)
    view_521: "f32[768]" = torch.ops.aten.view.default(sum_214, [768]);  sum_214 = None
    mul_406: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_405, primals_1);  primals_1 = None
    mul_407: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_405, div);  mul_405 = div = None
    sum_215: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_407, [0, 1], True);  mul_407 = None
    view_522: "f32[768]" = torch.ops.aten.view.default(sum_215, [768]);  sum_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_211: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_1, sqrt);  sub_1 = None
    div_212: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_211, sqrt);  div_211 = None
    neg_74: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_406)
    mul_408: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_74, div_212);  neg_74 = div_212 = None
    div_213: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_406, sqrt);  mul_406 = sqrt = None
    sum_216: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_408, [2], True);  mul_408 = None
    alias_101: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias);  alias = None
    mul_409: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_101, 2);  alias_101 = None
    div_214: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_216, mul_409);  sum_216 = mul_409 = None
    neg_75: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_213)
    sum_217: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_75, [2], True);  neg_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_96: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_214, [1, 512, 768]);  div_214 = None
    div_215: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_96, 768);  expand_96 = None
    pow_50: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub, 1.0);  sub = None
    mul_410: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_50, 2.0);  pow_50 = None
    mul_411: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_215, mul_410);  div_215 = mul_410 = None
    neg_76: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_411)
    sum_218: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_76, [2], True);  neg_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_234: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_213, mul_411);  div_213 = mul_411 = None
    add_235: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_217, sum_218);  sum_217 = sum_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_97: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_235, [1, 512, 768]);  add_235 = None
    div_216: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_97, 768);  expand_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_236: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_234, div_216);  add_234 = div_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:789, code: position_embeddings = self.position_embeddings(position_ids.long())
    eq: "b8[1, 512]" = torch.ops.aten.eq.Scalar(slice_1, -1)
    unsqueeze_58: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
    scalar_tensor_94: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_106: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_58, scalar_tensor_94, add_236);  unsqueeze_58 = scalar_tensor_94 = None
    full_28: "f32[512, 768]" = torch.ops.aten.full.default([512, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put: "f32[512, 768]" = torch.ops.aten._unsafe_index_put.default(full_28, [slice_1], where_106, True);  full_28 = slice_1 = where_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:786, code: inputs_embeds = self.word_embeddings(input_ids)
    eq_1: "b8[1, 512]" = torch.ops.aten.eq.Scalar(primals_164, 0)
    unsqueeze_59: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
    scalar_tensor_95: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_107: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_59, scalar_tensor_95, add_236);  unsqueeze_59 = scalar_tensor_95 = add_236 = None
    full_29: "f32[50265, 768]" = torch.ops.aten.full.default([50265, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_1: "f32[50265, 768]" = torch.ops.aten._unsafe_index_put.default(full_29, [primals_164], where_107, True);  full_29 = primals_164 = where_107 = None
    return pytree.tree_unflatten([div_51, clone_12, clone_13, view_522, view_521, squeeze_52, squeeze_50, view_505, view_504, view_497, view_496, squeeze_48, squeeze_46, view_480, view_479, view_472, view_471, squeeze_44, squeeze_42, view_455, view_454, view_447, view_446, squeeze_40, squeeze_38, view_430, view_429, view_422, view_421, squeeze_36, squeeze_34, view_405, view_404, view_397, view_396, squeeze_32, squeeze_30, view_380, view_379, view_372, view_371, squeeze_28, squeeze_26, view_355, view_354, view_347, view_346, squeeze_24, squeeze_22, view_330, view_329, view_322, view_321, squeeze_20, squeeze_18, view_305, view_304, view_297, view_296, squeeze_16, squeeze_14, view_280, view_279, view_272, view_271, squeeze_12, squeeze_10, view_255, view_254, view_247, view_246, squeeze_8, squeeze_6, view_230, view_229, view_222, view_221, _unsafe_index_put_1, _unsafe_index_put, permute_449, permute_436, view_507, permute_432, view_502, permute_428, view_499, permute_424, permute_411, view_482, permute_407, view_477, permute_403, view_474, permute_399, permute_386, view_457, permute_382, view_452, permute_378, view_449, permute_374, permute_361, view_432, permute_357, view_427, permute_353, view_424, permute_349, permute_336, view_407, permute_332, view_402, permute_328, view_399, permute_324, permute_311, view_382, permute_307, view_377, permute_303, view_374, permute_299, permute_286, view_357, permute_282, view_352, permute_278, view_349, permute_274, permute_261, view_332, permute_257, view_327, permute_253, view_324, permute_249, permute_236, view_307, permute_232, view_302, permute_228, view_299, permute_224, permute_211, view_282, permute_207, view_277, permute_203, view_274, permute_199, permute_186, view_257, permute_182, view_252, permute_178, view_249, permute_174, permute_161, view_232, permute_157, view_227, permute_153, view_224, permute_149, view_219, None, None, None, None], self._out_spec)
    