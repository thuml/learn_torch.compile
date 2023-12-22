from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[768]"; primals_2: "f32[768]"; primals_3: "f32[768]"; primals_4: "f32[768]"; primals_5: "f32[768]"; primals_6: "f32[768]"; primals_7: "f32[768]"; primals_8: "f32[768]"; primals_9: "f32[768]"; primals_10: "f32[768]"; primals_11: "f32[768]"; primals_12: "f32[768]"; primals_13: "f32[768]"; primals_14: "f32[768]"; primals_15: "f32[768]"; primals_16: "f32[768]"; primals_17: "f32[768]"; primals_18: "f32[768]"; primals_19: "f32[768]"; primals_20: "f32[768]"; primals_21: "f32[768]"; primals_22: "f32[768]"; primals_23: "f32[768]"; primals_24: "f32[768]"; primals_25: "f32[768]"; primals_26: "f32[768]"; primals_27: "f32[768]"; primals_28: "f32[768]"; primals_29: "f32[768]"; primals_30: "f32[768]"; primals_31: "f32[768]"; primals_32: "f32[768]"; primals_33: "f32[768]"; primals_34: "f32[768]"; primals_35: "f32[768]"; primals_36: "f32[768]"; primals_37: "f32[768]"; primals_38: "f32[768]"; primals_39: "f32[768]"; primals_40: "f32[768]"; primals_41: "f32[768]"; primals_42: "f32[768]"; primals_43: "f32[768]"; primals_44: "f32[768]"; primals_45: "f32[768]"; primals_46: "f32[768]"; primals_47: "f32[768]"; primals_48: "f32[768]"; primals_49: "f32[768]"; primals_50: "f32[768]"; primals_51: "f32[768]"; primals_52: "f32[768]"; primals_53: "f32[768]"; primals_54: "f32[768]"; primals_55: "f32[768]"; primals_56: "f32[768]"; primals_57: "f32[768]"; primals_58: "f32[768]"; primals_59: "f32[768]"; primals_60: "f32[768]"; primals_61: "f32[768]"; primals_62: "f32[768]"; primals_63: "f32[768]"; primals_64: "f32[768]"; primals_65: "f32[768]"; primals_66: "f32[768]"; primals_67: "f32[768]"; primals_68: "f32[768]"; primals_69: "f32[768]"; primals_70: "f32[768]"; primals_71: "f32[768]"; primals_72: "f32[768]"; primals_73: "f32[768]"; primals_74: "f32[768]"; primals_75: "f32[50265, 768]"; primals_76: "f32[512, 768]"; primals_77: "f32[2304, 768]"; primals_78: "f32[768, 768]"; primals_79: "f32[768]"; primals_80: "f32[3072, 768]"; primals_81: "f32[3072]"; primals_82: "f32[768, 3072]"; primals_83: "f32[768]"; primals_84: "f32[2304, 768]"; primals_85: "f32[768, 768]"; primals_86: "f32[768]"; primals_87: "f32[3072, 768]"; primals_88: "f32[3072]"; primals_89: "f32[768, 3072]"; primals_90: "f32[768]"; primals_91: "f32[2304, 768]"; primals_92: "f32[768, 768]"; primals_93: "f32[768]"; primals_94: "f32[3072, 768]"; primals_95: "f32[3072]"; primals_96: "f32[768, 3072]"; primals_97: "f32[768]"; primals_98: "f32[2304, 768]"; primals_99: "f32[768, 768]"; primals_100: "f32[768]"; primals_101: "f32[3072, 768]"; primals_102: "f32[3072]"; primals_103: "f32[768, 3072]"; primals_104: "f32[768]"; primals_105: "f32[2304, 768]"; primals_106: "f32[768, 768]"; primals_107: "f32[768]"; primals_108: "f32[3072, 768]"; primals_109: "f32[3072]"; primals_110: "f32[768, 3072]"; primals_111: "f32[768]"; primals_112: "f32[2304, 768]"; primals_113: "f32[768, 768]"; primals_114: "f32[768]"; primals_115: "f32[3072, 768]"; primals_116: "f32[3072]"; primals_117: "f32[768, 3072]"; primals_118: "f32[768]"; primals_119: "f32[2304, 768]"; primals_120: "f32[768, 768]"; primals_121: "f32[768]"; primals_122: "f32[3072, 768]"; primals_123: "f32[3072]"; primals_124: "f32[768, 3072]"; primals_125: "f32[768]"; primals_126: "f32[2304, 768]"; primals_127: "f32[768, 768]"; primals_128: "f32[768]"; primals_129: "f32[3072, 768]"; primals_130: "f32[3072]"; primals_131: "f32[768, 3072]"; primals_132: "f32[768]"; primals_133: "f32[2304, 768]"; primals_134: "f32[768, 768]"; primals_135: "f32[768]"; primals_136: "f32[3072, 768]"; primals_137: "f32[3072]"; primals_138: "f32[768, 3072]"; primals_139: "f32[768]"; primals_140: "f32[2304, 768]"; primals_141: "f32[768, 768]"; primals_142: "f32[768]"; primals_143: "f32[3072, 768]"; primals_144: "f32[3072]"; primals_145: "f32[768, 3072]"; primals_146: "f32[768]"; primals_147: "f32[2304, 768]"; primals_148: "f32[768, 768]"; primals_149: "f32[768]"; primals_150: "f32[3072, 768]"; primals_151: "f32[3072]"; primals_152: "f32[768, 3072]"; primals_153: "f32[768]"; primals_154: "f32[2304, 768]"; primals_155: "f32[768, 768]"; primals_156: "f32[768]"; primals_157: "f32[3072, 768]"; primals_158: "f32[3072]"; primals_159: "f32[768, 3072]"; primals_160: "f32[768]"; primals_161: "f32[768, 768]"; primals_162: "f32[768]"; primals_163: "f32[768]"; primals_164: "f32[768]"; primals_165: "f32[50265, 768]"; primals_166: "f32[50265]"; primals_167: "i64[1, 512]"; primals_168: "i64[1, 512]"; primals_169: "i64[1, 512]"; tangents_1: "f32[]"; tangents_2: "f32[1, 512, 50265]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, tangents_1, tangents_2, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:972, code: attention_mask = torch.ones(input_shape, device=device)
    full: "f32[1, 512]" = torch.ops.aten.full.default([1, 512], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:780, code: position_ids = self.position_ids[:, :seq_length]
    slice_1: "i64[1, 512]" = torch.ops.aten.slice.Tensor(primals_167, 0, 0, 9223372036854775807);  primals_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:786, code: inputs_embeds = self.word_embeddings(input_ids)
    embedding: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(primals_75, primals_168, 0);  primals_75 = None
    
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
    empty: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    permute: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty, [0, 1, 2]);  empty = None
    bernoulli: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute, 0.9);  permute = None
    sub_2: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli);  bernoulli = None
    convert_element_type: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_2, torch.bool);  sub_2 = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
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
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_2: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(bitwise_not, scalar_tensor_1, div_2);  bitwise_not = scalar_tensor_1 = div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_1: "f32[1, 12, 512, 512]" = torch.ops.aten.empty.memory_format([1, 12, 512, 512], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    permute_6: "f32[1, 12, 512, 512]" = torch.ops.aten.permute.default(empty_1, [0, 1, 2, 3]);  empty_1 = None
    bernoulli_1: "f32[1, 12, 512, 512]" = torch.ops.aten.bernoulli.p(permute_6, 0.9);  permute_6 = None
    sub_4: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_1);  bernoulli_1 = None
    convert_element_type_2: "b8[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_4, torch.bool);  sub_4 = None
    scalar_tensor_2: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
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
    empty_2: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    permute_9: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_2, [0, 1, 2]);  empty_2 = None
    bernoulli_2: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_9, 0.9);  permute_9 = None
    sub_5: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_2);  bernoulli_2 = None
    convert_element_type_3: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_5, torch.bool);  sub_5 = None
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
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
    empty_3: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    permute_12: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_3, [0, 1, 2]);  empty_3 = None
    bernoulli_3: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_12, 0.9);  permute_12 = None
    sub_8: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_3);  bernoulli_3 = None
    convert_element_type_4: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_8, torch.bool);  sub_8 = None
    scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
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
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_7: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(bitwise_not_1, scalar_tensor_5, div_6);  bitwise_not_1 = scalar_tensor_5 = div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_4: "f32[1, 12, 512, 512]" = torch.ops.aten.empty.memory_format([1, 12, 512, 512], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    permute_18: "f32[1, 12, 512, 512]" = torch.ops.aten.permute.default(empty_4, [0, 1, 2, 3]);  empty_4 = None
    bernoulli_4: "f32[1, 12, 512, 512]" = torch.ops.aten.bernoulli.p(permute_18, 0.9);  permute_18 = None
    sub_12: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_4);  bernoulli_4 = None
    convert_element_type_6: "b8[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_12, torch.bool);  sub_12 = None
    scalar_tensor_6: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
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
    empty_5: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    permute_21: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_5, [0, 1, 2]);  empty_5 = None
    bernoulli_5: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_21, 0.9);  permute_21 = None
    sub_13: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_5);  bernoulli_5 = None
    convert_element_type_7: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_13, torch.bool);  sub_13 = None
    scalar_tensor_7: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
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
    empty_6: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    permute_24: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_6, [0, 1, 2]);  empty_6 = None
    bernoulli_6: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_24, 0.9);  permute_24 = None
    sub_16: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_6);  bernoulli_6 = None
    convert_element_type_8: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_16, torch.bool);  sub_16 = None
    scalar_tensor_8: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
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
    scalar_tensor_9: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_12: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(bitwise_not_2, scalar_tensor_9, div_10);  bitwise_not_2 = scalar_tensor_9 = div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_7: "f32[1, 12, 512, 512]" = torch.ops.aten.empty.memory_format([1, 12, 512, 512], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    permute_30: "f32[1, 12, 512, 512]" = torch.ops.aten.permute.default(empty_7, [0, 1, 2, 3]);  empty_7 = None
    bernoulli_7: "f32[1, 12, 512, 512]" = torch.ops.aten.bernoulli.p(permute_30, 0.9);  permute_30 = None
    sub_20: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_7);  bernoulli_7 = None
    convert_element_type_10: "b8[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_20, torch.bool);  sub_20 = None
    scalar_tensor_10: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
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
    empty_8: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    permute_33: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_8, [0, 1, 2]);  empty_8 = None
    bernoulli_8: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_33, 0.9);  permute_33 = None
    sub_21: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_8);  bernoulli_8 = None
    convert_element_type_11: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_21, torch.bool);  sub_21 = None
    scalar_tensor_11: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
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
    empty_9: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    permute_36: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_9, [0, 1, 2]);  empty_9 = None
    bernoulli_9: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_36, 0.9);  permute_36 = None
    sub_24: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_9);  bernoulli_9 = None
    convert_element_type_12: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_24, torch.bool);  sub_24 = None
    scalar_tensor_12: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
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
    scalar_tensor_13: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_17: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(bitwise_not_3, scalar_tensor_13, div_14);  bitwise_not_3 = scalar_tensor_13 = div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_10: "f32[1, 12, 512, 512]" = torch.ops.aten.empty.memory_format([1, 12, 512, 512], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    permute_42: "f32[1, 12, 512, 512]" = torch.ops.aten.permute.default(empty_10, [0, 1, 2, 3]);  empty_10 = None
    bernoulli_10: "f32[1, 12, 512, 512]" = torch.ops.aten.bernoulli.p(permute_42, 0.9);  permute_42 = None
    sub_28: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_10);  bernoulli_10 = None
    convert_element_type_14: "b8[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_28, torch.bool);  sub_28 = None
    scalar_tensor_14: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
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
    empty_11: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    permute_45: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_11, [0, 1, 2]);  empty_11 = None
    bernoulli_11: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_45, 0.9);  permute_45 = None
    sub_29: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_11);  bernoulli_11 = None
    convert_element_type_15: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_29, torch.bool);  sub_29 = None
    scalar_tensor_15: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
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
    empty_12: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    permute_48: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_12, [0, 1, 2]);  empty_12 = None
    bernoulli_12: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_48, 0.9);  permute_48 = None
    sub_32: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_12);  bernoulli_12 = None
    convert_element_type_16: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_32, torch.bool);  sub_32 = None
    scalar_tensor_16: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
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
    scalar_tensor_17: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_22: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(bitwise_not_4, scalar_tensor_17, div_18);  bitwise_not_4 = scalar_tensor_17 = div_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_13: "f32[1, 12, 512, 512]" = torch.ops.aten.empty.memory_format([1, 12, 512, 512], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    permute_54: "f32[1, 12, 512, 512]" = torch.ops.aten.permute.default(empty_13, [0, 1, 2, 3]);  empty_13 = None
    bernoulli_13: "f32[1, 12, 512, 512]" = torch.ops.aten.bernoulli.p(permute_54, 0.9);  permute_54 = None
    sub_36: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_13);  bernoulli_13 = None
    convert_element_type_18: "b8[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_36, torch.bool);  sub_36 = None
    scalar_tensor_18: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
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
    empty_14: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    permute_57: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_14, [0, 1, 2]);  empty_14 = None
    bernoulli_14: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_57, 0.9);  permute_57 = None
    sub_37: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_14);  bernoulli_14 = None
    convert_element_type_19: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_37, torch.bool);  sub_37 = None
    scalar_tensor_19: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
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
    empty_15: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    permute_60: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_15, [0, 1, 2]);  empty_15 = None
    bernoulli_15: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_60, 0.9);  permute_60 = None
    sub_40: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_15);  bernoulli_15 = None
    convert_element_type_20: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_40, torch.bool);  sub_40 = None
    scalar_tensor_20: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
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
    scalar_tensor_21: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_27: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(bitwise_not_5, scalar_tensor_21, div_22);  bitwise_not_5 = scalar_tensor_21 = div_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_16: "f32[1, 12, 512, 512]" = torch.ops.aten.empty.memory_format([1, 12, 512, 512], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    permute_66: "f32[1, 12, 512, 512]" = torch.ops.aten.permute.default(empty_16, [0, 1, 2, 3]);  empty_16 = None
    bernoulli_16: "f32[1, 12, 512, 512]" = torch.ops.aten.bernoulli.p(permute_66, 0.9);  permute_66 = None
    sub_44: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_16);  bernoulli_16 = None
    convert_element_type_22: "b8[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_44, torch.bool);  sub_44 = None
    scalar_tensor_22: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
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
    empty_17: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    permute_69: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_17, [0, 1, 2]);  empty_17 = None
    bernoulli_17: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_69, 0.9);  permute_69 = None
    sub_45: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_17);  bernoulli_17 = None
    convert_element_type_23: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_45, torch.bool);  sub_45 = None
    scalar_tensor_23: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
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
    empty_18: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    permute_72: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_18, [0, 1, 2]);  empty_18 = None
    bernoulli_18: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_72, 0.9);  permute_72 = None
    sub_48: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_18);  bernoulli_18 = None
    convert_element_type_24: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_48, torch.bool);  sub_48 = None
    scalar_tensor_24: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
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
    scalar_tensor_25: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_32: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(bitwise_not_6, scalar_tensor_25, div_26);  bitwise_not_6 = scalar_tensor_25 = div_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_19: "f32[1, 12, 512, 512]" = torch.ops.aten.empty.memory_format([1, 12, 512, 512], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    permute_78: "f32[1, 12, 512, 512]" = torch.ops.aten.permute.default(empty_19, [0, 1, 2, 3]);  empty_19 = None
    bernoulli_19: "f32[1, 12, 512, 512]" = torch.ops.aten.bernoulli.p(permute_78, 0.9);  permute_78 = None
    sub_52: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_19);  bernoulli_19 = None
    convert_element_type_26: "b8[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_52, torch.bool);  sub_52 = None
    scalar_tensor_26: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
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
    empty_20: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    permute_81: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_20, [0, 1, 2]);  empty_20 = None
    bernoulli_20: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_81, 0.9);  permute_81 = None
    sub_53: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_20);  bernoulli_20 = None
    convert_element_type_27: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_53, torch.bool);  sub_53 = None
    scalar_tensor_27: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
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
    empty_21: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    permute_84: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_21, [0, 1, 2]);  empty_21 = None
    bernoulli_21: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_84, 0.9);  permute_84 = None
    sub_56: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_21);  bernoulli_21 = None
    convert_element_type_28: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_56, torch.bool);  sub_56 = None
    scalar_tensor_28: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
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
    scalar_tensor_29: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_37: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(bitwise_not_7, scalar_tensor_29, div_30);  bitwise_not_7 = scalar_tensor_29 = div_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_22: "f32[1, 12, 512, 512]" = torch.ops.aten.empty.memory_format([1, 12, 512, 512], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    permute_90: "f32[1, 12, 512, 512]" = torch.ops.aten.permute.default(empty_22, [0, 1, 2, 3]);  empty_22 = None
    bernoulli_22: "f32[1, 12, 512, 512]" = torch.ops.aten.bernoulli.p(permute_90, 0.9);  permute_90 = None
    sub_60: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_22);  bernoulli_22 = None
    convert_element_type_30: "b8[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_60, torch.bool);  sub_60 = None
    scalar_tensor_30: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
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
    empty_23: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    permute_93: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_23, [0, 1, 2]);  empty_23 = None
    bernoulli_23: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_93, 0.9);  permute_93 = None
    sub_61: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_23);  bernoulli_23 = None
    convert_element_type_31: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_61, torch.bool);  sub_61 = None
    scalar_tensor_31: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
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
    empty_24: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    permute_96: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_24, [0, 1, 2]);  empty_24 = None
    bernoulli_24: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_96, 0.9);  permute_96 = None
    sub_64: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_24);  bernoulli_24 = None
    convert_element_type_32: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_64, torch.bool);  sub_64 = None
    scalar_tensor_32: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
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
    scalar_tensor_33: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_42: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(bitwise_not_8, scalar_tensor_33, div_34);  bitwise_not_8 = scalar_tensor_33 = div_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_25: "f32[1, 12, 512, 512]" = torch.ops.aten.empty.memory_format([1, 12, 512, 512], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    permute_102: "f32[1, 12, 512, 512]" = torch.ops.aten.permute.default(empty_25, [0, 1, 2, 3]);  empty_25 = None
    bernoulli_25: "f32[1, 12, 512, 512]" = torch.ops.aten.bernoulli.p(permute_102, 0.9);  permute_102 = None
    sub_68: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_25);  bernoulli_25 = None
    convert_element_type_34: "b8[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_68, torch.bool);  sub_68 = None
    scalar_tensor_34: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
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
    empty_26: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    permute_105: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_26, [0, 1, 2]);  empty_26 = None
    bernoulli_26: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_105, 0.9);  permute_105 = None
    sub_69: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_26);  bernoulli_26 = None
    convert_element_type_35: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_69, torch.bool);  sub_69 = None
    scalar_tensor_35: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
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
    empty_27: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    permute_108: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_27, [0, 1, 2]);  empty_27 = None
    bernoulli_27: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_108, 0.9);  permute_108 = None
    sub_72: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_27);  bernoulli_27 = None
    convert_element_type_36: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_72, torch.bool);  sub_72 = None
    scalar_tensor_36: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
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
    scalar_tensor_37: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_47: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(bitwise_not_9, scalar_tensor_37, div_38);  bitwise_not_9 = scalar_tensor_37 = div_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_28: "f32[1, 12, 512, 512]" = torch.ops.aten.empty.memory_format([1, 12, 512, 512], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    permute_114: "f32[1, 12, 512, 512]" = torch.ops.aten.permute.default(empty_28, [0, 1, 2, 3]);  empty_28 = None
    bernoulli_28: "f32[1, 12, 512, 512]" = torch.ops.aten.bernoulli.p(permute_114, 0.9);  permute_114 = None
    sub_76: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_28);  bernoulli_28 = None
    convert_element_type_38: "b8[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_76, torch.bool);  sub_76 = None
    scalar_tensor_38: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
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
    empty_29: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    permute_117: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_29, [0, 1, 2]);  empty_29 = None
    bernoulli_29: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_117, 0.9);  permute_117 = None
    sub_77: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_29);  bernoulli_29 = None
    convert_element_type_39: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_77, torch.bool);  sub_77 = None
    scalar_tensor_39: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
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
    empty_30: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    permute_120: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_30, [0, 1, 2]);  empty_30 = None
    bernoulli_30: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_120, 0.9);  permute_120 = None
    sub_80: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_30);  bernoulli_30 = None
    convert_element_type_40: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_80, torch.bool);  sub_80 = None
    scalar_tensor_40: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
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
    scalar_tensor_41: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_52: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(bitwise_not_10, scalar_tensor_41, div_42);  bitwise_not_10 = scalar_tensor_41 = div_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_31: "f32[1, 12, 512, 512]" = torch.ops.aten.empty.memory_format([1, 12, 512, 512], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    permute_126: "f32[1, 12, 512, 512]" = torch.ops.aten.permute.default(empty_31, [0, 1, 2, 3]);  empty_31 = None
    bernoulli_31: "f32[1, 12, 512, 512]" = torch.ops.aten.bernoulli.p(permute_126, 0.9);  permute_126 = None
    sub_84: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_31);  bernoulli_31 = None
    convert_element_type_42: "b8[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_84, torch.bool);  sub_84 = None
    scalar_tensor_42: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
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
    empty_32: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    permute_129: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_32, [0, 1, 2]);  empty_32 = None
    bernoulli_32: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_129, 0.9);  permute_129 = None
    sub_85: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_32);  bernoulli_32 = None
    convert_element_type_43: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_85, torch.bool);  sub_85 = None
    scalar_tensor_43: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
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
    empty_33: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    permute_132: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_33, [0, 1, 2]);  empty_33 = None
    bernoulli_33: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_132, 0.9);  permute_132 = None
    sub_88: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_33);  bernoulli_33 = None
    convert_element_type_44: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_88, torch.bool);  sub_88 = None
    scalar_tensor_44: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
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
    scalar_tensor_45: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_57: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(bitwise_not_11, scalar_tensor_45, div_46);  bitwise_not_11 = scalar_tensor_45 = div_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    empty_34: "f32[1, 12, 512, 512]" = torch.ops.aten.empty.memory_format([1, 12, 512, 512], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    permute_138: "f32[1, 12, 512, 512]" = torch.ops.aten.permute.default(empty_34, [0, 1, 2, 3]);  empty_34 = None
    bernoulli_34: "f32[1, 12, 512, 512]" = torch.ops.aten.bernoulli.p(permute_138, 0.9);  permute_138 = None
    sub_92: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(1, bernoulli_34);  bernoulli_34 = None
    convert_element_type_46: "b8[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_92, torch.bool);  sub_92 = None
    scalar_tensor_46: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
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
    empty_35: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    permute_141: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_35, [0, 1, 2]);  empty_35 = None
    bernoulli_35: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_141, 0.9);  permute_141 = None
    sub_93: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_35);  bernoulli_35 = None
    convert_element_type_47: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_93, torch.bool);  sub_93 = None
    scalar_tensor_47: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
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
    empty_36: "f32[1, 512, 768]" = torch.ops.aten.empty.memory_format([1, 512, 768], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    permute_144: "f32[1, 512, 768]" = torch.ops.aten.permute.default(empty_36, [0, 1, 2]);  empty_36 = None
    bernoulli_36: "f32[1, 512, 768]" = torch.ops.aten.bernoulli.p(permute_144, 0.9);  permute_144 = None
    sub_96: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, bernoulli_36);  bernoulli_36 = None
    convert_element_type_48: "b8[1, 512, 768]" = torch.ops.prims.convert_element_type.default(sub_96, torch.bool);  sub_96 = None
    scalar_tensor_48: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1116, code: hidden_states = self.dense(hidden_states)
    view_216: "f32[512, 768]" = torch.ops.aten.view.default(add_110, [512, 768]);  add_110 = None
    permute_145: "f32[768, 768]" = torch.ops.aten.permute.default(primals_161, [1, 0]);  primals_161 = None
    addmm_36: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_162, view_216, permute_145);  primals_162 = None
    view_217: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_36, [1, 512, 768]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_112: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_217, 0.5)
    mul_113: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476)
    erf_12: "f32[1, 512, 768]" = torch.ops.aten.erf.default(mul_113);  mul_113 = None
    add_111: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_114: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_112, add_111);  mul_112 = add_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1118, code: hidden_states = self.LayerNorm(hidden_states)
    var_mean = torch.ops.aten.var_mean.correction(mul_114, [2], correction = 0, keepdim = True)
    getitem_36: "f32[1, 512, 1]" = var_mean[0]
    getitem_37: "f32[1, 512, 1]" = var_mean[1];  var_mean = None
    add_112: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-07);  getitem_36 = None
    rsqrt: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
    sub_99: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_114, getitem_37)
    mul_115: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_99, rsqrt);  sub_99 = None
    mul_116: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_115, primals_163);  mul_115 = None
    add_113: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_116, primals_164);  mul_116 = primals_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1139, code: hidden_states = self.decoder(hidden_states)
    view_218: "f32[512, 768]" = torch.ops.aten.view.default(add_113, [512, 768]);  add_113 = None
    permute_146: "f32[768, 50265]" = torch.ops.aten.permute.default(primals_165, [1, 0]);  primals_165 = None
    addmm_37: "f32[512, 50265]" = torch.ops.aten.addmm.default(primals_166, view_218, permute_146);  primals_166 = None
    view_219: "f32[1, 512, 50265]" = torch.ops.aten.view.default(addmm_37, [1, 512, 50265]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1089, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    view_220: "f32[512, 50265]" = torch.ops.aten.view.default(view_219, [-1, 50265])
    view_221: "i64[512]" = torch.ops.aten.view.default(primals_169, [-1]);  primals_169 = None
    amax_12: "f32[512, 1]" = torch.ops.aten.amax.default(view_220, [1], True)
    sub_100: "f32[512, 50265]" = torch.ops.aten.sub.Tensor(view_220, amax_12);  view_220 = amax_12 = None
    exp_12: "f32[512, 50265]" = torch.ops.aten.exp.default(sub_100)
    sum_13: "f32[512, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [1], True);  exp_12 = None
    log: "f32[512, 1]" = torch.ops.aten.log.default(sum_13);  sum_13 = None
    sub_101: "f32[512, 50265]" = torch.ops.aten.sub.Tensor(sub_100, log);  sub_100 = log = None
    alias_37: "f32[512, 50265]" = torch.ops.aten.alias.default(sub_101)
    ne: "b8[512]" = torch.ops.aten.ne.Scalar(view_221, -100)
    scalar_tensor_49: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0))
    where_61: "i64[512]" = torch.ops.aten.where.self(ne, view_221, scalar_tensor_49);  ne = scalar_tensor_49 = None
    unsqueeze_52: "i64[512, 1]" = torch.ops.aten.unsqueeze.default(where_61, 1);  where_61 = None
    gather: "f32[512, 1]" = torch.ops.aten.gather.default(sub_101, 1, unsqueeze_52);  sub_101 = unsqueeze_52 = None
    squeeze_1: "f32[512]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[512]" = torch.ops.aten.neg.default(squeeze_1);  squeeze_1 = None
    ne_1: "b8[512]" = torch.ops.aten.ne.Scalar(view_221, -100)
    scalar_tensor_50: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_62: "f32[512]" = torch.ops.aten.where.self(ne_1, neg, scalar_tensor_50);  ne_1 = neg = scalar_tensor_50 = None
    ne_2: "b8[512]" = torch.ops.aten.ne.Scalar(view_221, -100)
    sum_14: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
    convert_element_type_49: "f32[]" = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None
    sum_15: "f32[]" = torch.ops.aten.sum.default(where_62);  where_62 = None
    div_49: "f32[]" = torch.ops.aten.div.Tensor(sum_15, convert_element_type_49);  sum_15 = None
    div_50: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_49);  tangents_1 = convert_element_type_49 = None
    unsqueeze_53: "i64[512, 1]" = torch.ops.aten.unsqueeze.default(view_221, 1);  view_221 = None
    ne_3: "b8[512, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_53, -100)
    scalar_tensor_51: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0))
    where_63: "i64[512, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_53, scalar_tensor_51);  ne_3 = scalar_tensor_51 = None
    full_2: "f32[512, 50265]" = torch.ops.aten.full.default([512, 50265], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    scatter: "f32[512, 50265]" = torch.ops.aten.scatter.value(full_2, 1, where_63, -1.0);  full_2 = where_63 = None
    ne_4: "b8[512, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_53, -100);  unsqueeze_53 = None
    scalar_tensor_52: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_64: "f32[512, 1]" = torch.ops.aten.where.self(ne_4, div_50, scalar_tensor_52);  ne_4 = div_50 = scalar_tensor_52 = None
    mul_117: "f32[512, 50265]" = torch.ops.aten.mul.Tensor(scatter, where_64);  scatter = where_64 = None
    alias_38: "f32[512, 50265]" = torch.ops.aten.alias.default(alias_37);  alias_37 = None
    exp_13: "f32[512, 50265]" = torch.ops.aten.exp.default(alias_38);  alias_38 = None
    sum_16: "f32[512, 1]" = torch.ops.aten.sum.dim_IntList(mul_117, [1], True)
    mul_118: "f32[512, 50265]" = torch.ops.aten.mul.Tensor(exp_13, sum_16);  exp_13 = sum_16 = None
    sub_102: "f32[512, 50265]" = torch.ops.aten.sub.Tensor(mul_117, mul_118);  mul_117 = mul_118 = None
    view_222: "f32[1, 512, 50265]" = torch.ops.aten.view.default(sub_102, [1, 512, 50265]);  sub_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1089, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    add_114: "f32[1, 512, 50265]" = torch.ops.aten.add.Tensor(tangents_2, view_222);  tangents_2 = view_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1139, code: hidden_states = self.decoder(hidden_states)
    view_223: "f32[512, 50265]" = torch.ops.aten.view.default(add_114, [512, 50265]);  add_114 = None
    permute_147: "f32[50265, 768]" = torch.ops.aten.permute.default(permute_146, [1, 0]);  permute_146 = None
    mm_12: "f32[512, 768]" = torch.ops.aten.mm.default(view_223, permute_147);  permute_147 = None
    permute_148: "f32[50265, 512]" = torch.ops.aten.permute.default(view_223, [1, 0])
    mm_13: "f32[50265, 768]" = torch.ops.aten.mm.default(permute_148, view_218);  permute_148 = view_218 = None
    permute_149: "f32[768, 50265]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_17: "f32[1, 50265]" = torch.ops.aten.sum.dim_IntList(view_223, [0], True);  view_223 = None
    view_224: "f32[50265]" = torch.ops.aten.view.default(sum_17, [50265]);  sum_17 = None
    permute_150: "f32[50265, 768]" = torch.ops.aten.permute.default(permute_149, [1, 0]);  permute_149 = None
    view_225: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_12, [1, 512, 768]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1118, code: hidden_states = self.LayerNorm(hidden_states)
    sub_103: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_114, getitem_37);  mul_114 = getitem_37 = None
    mul_119: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_103, rsqrt);  sub_103 = None
    mul_120: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_225, primals_163);  primals_163 = None
    mul_121: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_120, 768)
    sum_18: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_120, [2], True)
    mul_122: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_120, mul_119);  mul_120 = None
    sum_19: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_122, [2], True);  mul_122 = None
    mul_123: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_119, sum_19);  sum_19 = None
    sub_104: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_121, sum_18);  mul_121 = sum_18 = None
    sub_105: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_104, mul_123);  sub_104 = mul_123 = None
    div_51: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
    mul_124: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_51, sub_105);  div_51 = sub_105 = None
    mul_125: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_225, mul_119);  mul_119 = None
    sum_20: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_125, [0, 1]);  mul_125 = None
    sum_21: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_225, [0, 1]);  view_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_126: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476)
    erf_13: "f32[1, 512, 768]" = torch.ops.aten.erf.default(mul_126);  mul_126 = None
    add_115: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_127: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_115, 0.5);  add_115 = None
    mul_128: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_217, view_217)
    mul_129: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_128, -0.5);  mul_128 = None
    exp_14: "f32[1, 512, 768]" = torch.ops.aten.exp.default(mul_129);  mul_129 = None
    mul_130: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
    mul_131: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_217, mul_130);  view_217 = mul_130 = None
    add_116: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_127, mul_131);  mul_127 = mul_131 = None
    mul_132: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_124, add_116);  mul_124 = add_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1116, code: hidden_states = self.dense(hidden_states)
    view_226: "f32[512, 768]" = torch.ops.aten.view.default(mul_132, [512, 768]);  mul_132 = None
    permute_151: "f32[768, 768]" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
    mm_14: "f32[512, 768]" = torch.ops.aten.mm.default(view_226, permute_151);  permute_151 = None
    permute_152: "f32[768, 512]" = torch.ops.aten.permute.default(view_226, [1, 0])
    mm_15: "f32[768, 768]" = torch.ops.aten.mm.default(permute_152, view_216);  permute_152 = view_216 = None
    permute_153: "f32[768, 768]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_22: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_226, [0], True);  view_226 = None
    view_227: "f32[768]" = torch.ops.aten.view.default(sum_22, [768]);  sum_22 = None
    permute_154: "f32[768, 768]" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
    view_228: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_14, [1, 512, 768]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_23: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_228, [0, 1], True)
    view_229: "f32[768]" = torch.ops.aten.view.default(sum_23, [768]);  sum_23 = None
    mul_133: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_228, primals_73);  primals_73 = None
    mul_134: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_228, div_48);  view_228 = div_48 = None
    sum_24: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_134, [0, 1], True);  mul_134 = None
    view_230: "f32[768]" = torch.ops.aten.view.default(sum_24, [768]);  sum_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_52: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_98, sqrt_36);  sub_98 = None
    div_53: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_52, sqrt_36);  div_52 = None
    neg_1: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_133)
    mul_135: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_1, div_53);  neg_1 = div_53 = None
    div_54: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_133, sqrt_36);  mul_133 = sqrt_36 = None
    sum_25: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_135, [2], True);  mul_135 = None
    alias_39: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_36);  alias_36 = None
    mul_136: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_39, 2);  alias_39 = None
    div_55: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_25, mul_136);  sum_25 = mul_136 = None
    neg_2: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_54)
    sum_26: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_2, [2], True);  neg_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_48: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_55, [1, 512, 768]);  div_55 = None
    div_56: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_48, 768);  expand_48 = None
    pow_26: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_97, 1.0);  sub_97 = None
    mul_137: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_26, 2.0);  pow_26 = None
    mul_138: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_56, mul_137);  div_56 = mul_137 = None
    neg_3: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_138)
    sum_27: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_3, [2], True);  neg_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_117: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_54, mul_138);  div_54 = mul_138 = None
    add_118: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_26, sum_27);  sum_26 = sum_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_49: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_118, [1, 512, 768]);  add_118 = None
    div_57: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_49, 768);  expand_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_119: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_117, div_57);  add_117 = div_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_53: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_65: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_48, scalar_tensor_53, add_119);  convert_element_type_48 = scalar_tensor_53 = None
    mul_139: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_65, 1.1111111111111112);  where_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_231: "f32[512, 768]" = torch.ops.aten.view.default(mul_139, [512, 768]);  mul_139 = None
    permute_155: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
    mm_16: "f32[512, 3072]" = torch.ops.aten.mm.default(view_231, permute_155);  permute_155 = None
    permute_156: "f32[768, 512]" = torch.ops.aten.permute.default(view_231, [1, 0])
    mm_17: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_156, view_214);  permute_156 = view_214 = None
    permute_157: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_28: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_231, [0], True);  view_231 = None
    view_232: "f32[768]" = torch.ops.aten.view.default(sum_28, [768]);  sum_28 = None
    permute_158: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_157, [1, 0]);  permute_157 = None
    view_233: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_16, [1, 512, 3072]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_140: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_213, 0.7071067811865476)
    erf_14: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_140);  mul_140 = None
    add_120: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_141: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_120, 0.5);  add_120 = None
    mul_142: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_213, view_213)
    mul_143: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_142, -0.5);  mul_142 = None
    exp_15: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_143);  mul_143 = None
    mul_144: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_15, 0.3989422804014327);  exp_15 = None
    mul_145: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_213, mul_144);  view_213 = mul_144 = None
    add_121: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_141, mul_145);  mul_141 = mul_145 = None
    mul_146: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_233, add_121);  view_233 = add_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_234: "f32[512, 3072]" = torch.ops.aten.view.default(mul_146, [512, 3072]);  mul_146 = None
    permute_159: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_142, [1, 0]);  permute_142 = None
    mm_18: "f32[512, 768]" = torch.ops.aten.mm.default(view_234, permute_159);  permute_159 = None
    permute_160: "f32[3072, 512]" = torch.ops.aten.permute.default(view_234, [1, 0])
    mm_19: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_160, view_212);  permute_160 = view_212 = None
    permute_161: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_29: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_234, [0], True);  view_234 = None
    view_235: "f32[3072]" = torch.ops.aten.view.default(sum_29, [3072]);  sum_29 = None
    permute_162: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_161, [1, 0]);  permute_161 = None
    view_236: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_18, [1, 512, 768]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    add_122: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_119, view_236);  add_119 = view_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_30: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_122, [0, 1], True)
    view_237: "f32[768]" = torch.ops.aten.view.default(sum_30, [768]);  sum_30 = None
    mul_147: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_122, primals_71);  primals_71 = None
    mul_148: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_122, div_47);  add_122 = div_47 = None
    sum_31: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_148, [0, 1], True);  mul_148 = None
    view_238: "f32[768]" = torch.ops.aten.view.default(sum_31, [768]);  sum_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_58: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_95, sqrt_35);  sub_95 = None
    div_59: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_58, sqrt_35);  div_58 = None
    neg_4: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_147)
    mul_149: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_4, div_59);  neg_4 = div_59 = None
    div_60: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_147, sqrt_35);  mul_147 = sqrt_35 = None
    sum_32: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_149, [2], True);  mul_149 = None
    alias_40: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_35);  alias_35 = None
    mul_150: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_40, 2);  alias_40 = None
    div_61: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_32, mul_150);  sum_32 = mul_150 = None
    neg_5: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_60)
    sum_33: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_5, [2], True);  neg_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_50: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_61, [1, 512, 768]);  div_61 = None
    div_62: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_50, 768);  expand_50 = None
    pow_27: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_94, 1.0);  sub_94 = None
    mul_151: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_27, 2.0);  pow_27 = None
    mul_152: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_62, mul_151);  div_62 = mul_151 = None
    neg_6: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_152)
    sum_34: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_6, [2], True);  neg_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_123: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_60, mul_152);  div_60 = mul_152 = None
    add_124: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_33, sum_34);  sum_33 = sum_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_51: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_124, [1, 512, 768]);  add_124 = None
    div_63: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_51, 768);  expand_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_125: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_123, div_63);  add_123 = div_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_54: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_66: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_47, scalar_tensor_54, add_125);  convert_element_type_47 = scalar_tensor_54 = None
    mul_153: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_66, 1.1111111111111112);  where_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_239: "f32[512, 768]" = torch.ops.aten.view.default(mul_153, [512, 768]);  mul_153 = None
    permute_163: "f32[768, 768]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
    mm_20: "f32[512, 768]" = torch.ops.aten.mm.default(view_239, permute_163);  permute_163 = None
    permute_164: "f32[768, 512]" = torch.ops.aten.permute.default(view_239, [1, 0])
    mm_21: "f32[768, 768]" = torch.ops.aten.mm.default(permute_164, view_210);  permute_164 = view_210 = None
    permute_165: "f32[768, 768]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_35: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_239, [0], True);  view_239 = None
    view_240: "f32[768]" = torch.ops.aten.view.default(sum_35, [768]);  sum_35 = None
    permute_166: "f32[768, 768]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
    view_241: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_20, [1, 512, 768]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_242: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_241, [1, 512, 12, 64]);  view_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_167: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_242, [0, 2, 1, 3]);  view_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_243: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_167, [12, 512, 64]);  permute_167 = None
    permute_168: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_206, [0, 2, 1]);  view_206 = None
    bmm_24: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_168, view_243);  permute_168 = None
    permute_169: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_207, [0, 2, 1]);  view_207 = None
    bmm_25: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_243, permute_169);  view_243 = permute_169 = None
    view_244: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_24, [1, 12, 512, 64]);  bmm_24 = None
    view_245: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_25, [1, 12, 512, 512]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_55: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_67: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_46, scalar_tensor_55, view_245);  convert_element_type_46 = scalar_tensor_55 = view_245 = None
    mul_154: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_67, 1.1111111111111112);  where_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_42: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(where_57);  where_57 = None
    alias_43: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_42);  alias_42 = None
    mul_155: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_154, alias_43);  mul_154 = None
    sum_36: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_155, [-1], True)
    mul_156: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_43, sum_36);  alias_43 = sum_36 = None
    sub_106: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_155, mul_156);  mul_155 = mul_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_246: "f32[12, 512, 512]" = torch.ops.aten.view.default(sub_106, [12, 512, 512]);  sub_106 = None
    permute_170: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_203, [0, 2, 1]);  view_203 = None
    bmm_26: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_170, view_246);  permute_170 = None
    permute_171: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_204, [0, 2, 1]);  view_204 = None
    bmm_27: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_246, permute_171);  view_246 = permute_171 = None
    view_247: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_26, [1, 12, 64, 512]);  bmm_26 = None
    view_248: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_27, [1, 12, 512, 64]);  bmm_27 = None
    permute_172: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_247, [0, 1, 3, 2]);  view_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_64: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(view_248, sqrt_34);  view_248 = sqrt_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    sum_37: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(view_244, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_173: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_37, [0, 2, 1, 3]);  sum_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_249: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_173, [1, 1, 768]);  permute_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    full_3: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_3, view_249, 2, 0, 9223372036854775807);  full_3 = view_249 = None
    squeeze_2: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter, 1);  slice_scatter = None
    squeeze_3: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_2, 0);  squeeze_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    sum_38: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(div_64, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_174: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_38, [0, 2, 1, 3]);  sum_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_250: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_174, [1, 1, 768]);  permute_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    full_4: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_1: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_4, view_250, 2, 0, 9223372036854775807);  full_4 = view_250 = None
    squeeze_4: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_1, 1);  slice_scatter_1 = None
    squeeze_5: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_4, 0);  squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    cat: "f32[1, 12, 512, 192]" = torch.ops.aten.cat.default([div_64, permute_172, view_244], 3);  div_64 = permute_172 = view_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_175: "f32[1, 512, 12, 192]" = torch.ops.aten.permute.default(cat, [0, 2, 1, 3]);  cat = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    clone_12: "f32[1, 512, 12, 192]" = torch.ops.aten.clone.default(permute_175, memory_format = torch.contiguous_format);  permute_175 = None
    view_251: "f32[1, 512, 2304]" = torch.ops.aten.view.default(clone_12, [1, 512, 2304]);  clone_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_252: "f32[512, 2304]" = torch.ops.aten.view.default(view_251, [512, 2304]);  view_251 = None
    permute_176: "f32[2304, 512]" = torch.ops.aten.permute.default(view_252, [1, 0])
    mm_22: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_176, view_198);  permute_176 = view_198 = None
    permute_177: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_22, [1, 0]);  mm_22 = None
    permute_178: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
    mm_23: "f32[512, 768]" = torch.ops.aten.mm.default(view_252, permute_178);  view_252 = permute_178 = None
    view_253: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_23, [1, 512, 768]);  mm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    add_126: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_125, view_253);  add_125 = view_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_179: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_177, [1, 0]);  permute_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_39: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_126, [0, 1], True)
    view_254: "f32[768]" = torch.ops.aten.view.default(sum_39, [768]);  sum_39 = None
    mul_157: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_126, primals_67);  primals_67 = None
    mul_158: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_126, div_44);  add_126 = div_44 = None
    sum_40: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_158, [0, 1], True);  mul_158 = None
    view_255: "f32[768]" = torch.ops.aten.view.default(sum_40, [768]);  sum_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_65: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_90, sqrt_33);  sub_90 = None
    div_66: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_65, sqrt_33);  div_65 = None
    neg_7: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_157)
    mul_159: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_7, div_66);  neg_7 = div_66 = None
    div_67: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_157, sqrt_33);  mul_157 = sqrt_33 = None
    sum_41: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_159, [2], True);  mul_159 = None
    alias_44: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_33);  alias_33 = None
    mul_160: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_44, 2);  alias_44 = None
    div_68: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_41, mul_160);  sum_41 = mul_160 = None
    neg_8: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_67)
    sum_42: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_8, [2], True);  neg_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_52: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_68, [1, 512, 768]);  div_68 = None
    div_69: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_52, 768);  expand_52 = None
    pow_28: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_89, 1.0);  sub_89 = None
    mul_161: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_28, 2.0);  pow_28 = None
    mul_162: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_69, mul_161);  div_69 = mul_161 = None
    neg_9: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_162)
    sum_43: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_9, [2], True);  neg_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_127: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_67, mul_162);  div_67 = mul_162 = None
    add_128: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_42, sum_43);  sum_42 = sum_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_53: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_128, [1, 512, 768]);  add_128 = None
    div_70: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_53, 768);  expand_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_129: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_127, div_70);  add_127 = div_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_56: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_68: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_44, scalar_tensor_56, add_129);  convert_element_type_44 = scalar_tensor_56 = None
    mul_163: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_68, 1.1111111111111112);  where_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_256: "f32[512, 768]" = torch.ops.aten.view.default(mul_163, [512, 768]);  mul_163 = None
    permute_180: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
    mm_24: "f32[512, 3072]" = torch.ops.aten.mm.default(view_256, permute_180);  permute_180 = None
    permute_181: "f32[768, 512]" = torch.ops.aten.permute.default(view_256, [1, 0])
    mm_25: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_181, view_196);  permute_181 = view_196 = None
    permute_182: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_44: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_256, [0], True);  view_256 = None
    view_257: "f32[768]" = torch.ops.aten.view.default(sum_44, [768]);  sum_44 = None
    permute_183: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_182, [1, 0]);  permute_182 = None
    view_258: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_24, [1, 512, 3072]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_164: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476)
    erf_15: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_164);  mul_164 = None
    add_130: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_165: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_130, 0.5);  add_130 = None
    mul_166: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, view_195)
    mul_167: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_166, -0.5);  mul_166 = None
    exp_16: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_167);  mul_167 = None
    mul_168: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
    mul_169: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, mul_168);  view_195 = mul_168 = None
    add_131: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_165, mul_169);  mul_165 = mul_169 = None
    mul_170: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_258, add_131);  view_258 = add_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_259: "f32[512, 3072]" = torch.ops.aten.view.default(mul_170, [512, 3072]);  mul_170 = None
    permute_184: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
    mm_26: "f32[512, 768]" = torch.ops.aten.mm.default(view_259, permute_184);  permute_184 = None
    permute_185: "f32[3072, 512]" = torch.ops.aten.permute.default(view_259, [1, 0])
    mm_27: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_185, view_194);  permute_185 = view_194 = None
    permute_186: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_45: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_259, [0], True);  view_259 = None
    view_260: "f32[3072]" = torch.ops.aten.view.default(sum_45, [3072]);  sum_45 = None
    permute_187: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
    view_261: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_26, [1, 512, 768]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    add_132: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_129, view_261);  add_129 = view_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_46: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_132, [0, 1], True)
    view_262: "f32[768]" = torch.ops.aten.view.default(sum_46, [768]);  sum_46 = None
    mul_171: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_132, primals_65);  primals_65 = None
    mul_172: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_132, div_43);  add_132 = div_43 = None
    sum_47: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_172, [0, 1], True);  mul_172 = None
    view_263: "f32[768]" = torch.ops.aten.view.default(sum_47, [768]);  sum_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_71: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_87, sqrt_32);  sub_87 = None
    div_72: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_71, sqrt_32);  div_71 = None
    neg_10: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_171)
    mul_173: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_10, div_72);  neg_10 = div_72 = None
    div_73: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_171, sqrt_32);  mul_171 = sqrt_32 = None
    sum_48: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_173, [2], True);  mul_173 = None
    alias_45: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_32);  alias_32 = None
    mul_174: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_45, 2);  alias_45 = None
    div_74: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_48, mul_174);  sum_48 = mul_174 = None
    neg_11: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_73)
    sum_49: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_11, [2], True);  neg_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_54: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_74, [1, 512, 768]);  div_74 = None
    div_75: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_54, 768);  expand_54 = None
    pow_29: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_86, 1.0);  sub_86 = None
    mul_175: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_29, 2.0);  pow_29 = None
    mul_176: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_75, mul_175);  div_75 = mul_175 = None
    neg_12: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_176)
    sum_50: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_12, [2], True);  neg_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_133: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_73, mul_176);  div_73 = mul_176 = None
    add_134: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_49, sum_50);  sum_49 = sum_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_55: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_134, [1, 512, 768]);  add_134 = None
    div_76: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_55, 768);  expand_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_135: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_133, div_76);  add_133 = div_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_57: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_69: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_43, scalar_tensor_57, add_135);  convert_element_type_43 = scalar_tensor_57 = None
    mul_177: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_69, 1.1111111111111112);  where_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_264: "f32[512, 768]" = torch.ops.aten.view.default(mul_177, [512, 768]);  mul_177 = None
    permute_188: "f32[768, 768]" = torch.ops.aten.permute.default(permute_128, [1, 0]);  permute_128 = None
    mm_28: "f32[512, 768]" = torch.ops.aten.mm.default(view_264, permute_188);  permute_188 = None
    permute_189: "f32[768, 512]" = torch.ops.aten.permute.default(view_264, [1, 0])
    mm_29: "f32[768, 768]" = torch.ops.aten.mm.default(permute_189, view_192);  permute_189 = view_192 = None
    permute_190: "f32[768, 768]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_51: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_264, [0], True);  view_264 = None
    view_265: "f32[768]" = torch.ops.aten.view.default(sum_51, [768]);  sum_51 = None
    permute_191: "f32[768, 768]" = torch.ops.aten.permute.default(permute_190, [1, 0]);  permute_190 = None
    view_266: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_28, [1, 512, 768]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_267: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_266, [1, 512, 12, 64]);  view_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_192: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_267, [0, 2, 1, 3]);  view_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_268: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_192, [12, 512, 64]);  permute_192 = None
    permute_193: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None
    bmm_28: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_193, view_268);  permute_193 = None
    permute_194: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_189, [0, 2, 1]);  view_189 = None
    bmm_29: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_268, permute_194);  view_268 = permute_194 = None
    view_269: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_28, [1, 12, 512, 64]);  bmm_28 = None
    view_270: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_29, [1, 12, 512, 512]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_58: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_70: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_42, scalar_tensor_58, view_270);  convert_element_type_42 = scalar_tensor_58 = view_270 = None
    mul_178: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_70, 1.1111111111111112);  where_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_47: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(where_52);  where_52 = None
    alias_48: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_47);  alias_47 = None
    mul_179: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_178, alias_48);  mul_178 = None
    sum_52: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_179, [-1], True)
    mul_180: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_48, sum_52);  alias_48 = sum_52 = None
    sub_107: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_179, mul_180);  mul_179 = mul_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_271: "f32[12, 512, 512]" = torch.ops.aten.view.default(sub_107, [12, 512, 512]);  sub_107 = None
    permute_195: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_185, [0, 2, 1]);  view_185 = None
    bmm_30: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_195, view_271);  permute_195 = None
    permute_196: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1]);  view_186 = None
    bmm_31: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_271, permute_196);  view_271 = permute_196 = None
    view_272: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_30, [1, 12, 64, 512]);  bmm_30 = None
    view_273: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_31, [1, 12, 512, 64]);  bmm_31 = None
    permute_197: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_272, [0, 1, 3, 2]);  view_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_77: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(view_273, sqrt_31);  view_273 = sqrt_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    sum_53: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(view_269, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_198: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_53, [0, 2, 1, 3]);  sum_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_274: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_198, [1, 1, 768]);  permute_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    full_5: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_2: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_5, view_274, 2, 0, 9223372036854775807);  full_5 = view_274 = None
    squeeze_6: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_2, 1);  slice_scatter_2 = None
    squeeze_7: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_6, 0);  squeeze_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    sum_54: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(div_77, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_199: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_54, [0, 2, 1, 3]);  sum_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_275: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_199, [1, 1, 768]);  permute_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    full_6: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_3: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_6, view_275, 2, 0, 9223372036854775807);  full_6 = view_275 = None
    squeeze_8: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_3, 1);  slice_scatter_3 = None
    squeeze_9: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_8, 0);  squeeze_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    cat_1: "f32[1, 12, 512, 192]" = torch.ops.aten.cat.default([div_77, permute_197, view_269], 3);  div_77 = permute_197 = view_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_200: "f32[1, 512, 12, 192]" = torch.ops.aten.permute.default(cat_1, [0, 2, 1, 3]);  cat_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    clone_13: "f32[1, 512, 12, 192]" = torch.ops.aten.clone.default(permute_200, memory_format = torch.contiguous_format);  permute_200 = None
    view_276: "f32[1, 512, 2304]" = torch.ops.aten.view.default(clone_13, [1, 512, 2304]);  clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_277: "f32[512, 2304]" = torch.ops.aten.view.default(view_276, [512, 2304]);  view_276 = None
    permute_201: "f32[2304, 512]" = torch.ops.aten.permute.default(view_277, [1, 0])
    mm_30: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_201, view_180);  permute_201 = view_180 = None
    permute_202: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_30, [1, 0]);  mm_30 = None
    permute_203: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
    mm_31: "f32[512, 768]" = torch.ops.aten.mm.default(view_277, permute_203);  view_277 = permute_203 = None
    view_278: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_31, [1, 512, 768]);  mm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    add_136: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_135, view_278);  add_135 = view_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_204: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_202, [1, 0]);  permute_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_55: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_136, [0, 1], True)
    view_279: "f32[768]" = torch.ops.aten.view.default(sum_55, [768]);  sum_55 = None
    mul_181: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_136, primals_61);  primals_61 = None
    mul_182: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_136, div_40);  add_136 = div_40 = None
    sum_56: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_182, [0, 1], True);  mul_182 = None
    view_280: "f32[768]" = torch.ops.aten.view.default(sum_56, [768]);  sum_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_78: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_82, sqrt_30);  sub_82 = None
    div_79: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_78, sqrt_30);  div_78 = None
    neg_13: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_181)
    mul_183: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_13, div_79);  neg_13 = div_79 = None
    div_80: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_181, sqrt_30);  mul_181 = sqrt_30 = None
    sum_57: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_183, [2], True);  mul_183 = None
    alias_49: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_30);  alias_30 = None
    mul_184: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_49, 2);  alias_49 = None
    div_81: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_57, mul_184);  sum_57 = mul_184 = None
    neg_14: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_80)
    sum_58: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_14, [2], True);  neg_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_56: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_81, [1, 512, 768]);  div_81 = None
    div_82: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_56, 768);  expand_56 = None
    pow_30: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_81, 1.0);  sub_81 = None
    mul_185: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_30, 2.0);  pow_30 = None
    mul_186: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_82, mul_185);  div_82 = mul_185 = None
    neg_15: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_186)
    sum_59: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_15, [2], True);  neg_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_137: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_80, mul_186);  div_80 = mul_186 = None
    add_138: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_58, sum_59);  sum_58 = sum_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_57: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_138, [1, 512, 768]);  add_138 = None
    div_83: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_57, 768);  expand_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_139: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_137, div_83);  add_137 = div_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_59: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_71: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_40, scalar_tensor_59, add_139);  convert_element_type_40 = scalar_tensor_59 = None
    mul_187: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_71, 1.1111111111111112);  where_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_281: "f32[512, 768]" = torch.ops.aten.view.default(mul_187, [512, 768]);  mul_187 = None
    permute_205: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
    mm_32: "f32[512, 3072]" = torch.ops.aten.mm.default(view_281, permute_205);  permute_205 = None
    permute_206: "f32[768, 512]" = torch.ops.aten.permute.default(view_281, [1, 0])
    mm_33: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_206, view_178);  permute_206 = view_178 = None
    permute_207: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_60: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_281, [0], True);  view_281 = None
    view_282: "f32[768]" = torch.ops.aten.view.default(sum_60, [768]);  sum_60 = None
    permute_208: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_207, [1, 0]);  permute_207 = None
    view_283: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_32, [1, 512, 3072]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_188: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_177, 0.7071067811865476)
    erf_16: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_188);  mul_188 = None
    add_140: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_189: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_140, 0.5);  add_140 = None
    mul_190: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_177, view_177)
    mul_191: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_190, -0.5);  mul_190 = None
    exp_17: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_191);  mul_191 = None
    mul_192: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_17, 0.3989422804014327);  exp_17 = None
    mul_193: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_177, mul_192);  view_177 = mul_192 = None
    add_141: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_189, mul_193);  mul_189 = mul_193 = None
    mul_194: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_283, add_141);  view_283 = add_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_284: "f32[512, 3072]" = torch.ops.aten.view.default(mul_194, [512, 3072]);  mul_194 = None
    permute_209: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
    mm_34: "f32[512, 768]" = torch.ops.aten.mm.default(view_284, permute_209);  permute_209 = None
    permute_210: "f32[3072, 512]" = torch.ops.aten.permute.default(view_284, [1, 0])
    mm_35: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_210, view_176);  permute_210 = view_176 = None
    permute_211: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_61: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_284, [0], True);  view_284 = None
    view_285: "f32[3072]" = torch.ops.aten.view.default(sum_61, [3072]);  sum_61 = None
    permute_212: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None
    view_286: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_34, [1, 512, 768]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    add_142: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_139, view_286);  add_139 = view_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_62: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_142, [0, 1], True)
    view_287: "f32[768]" = torch.ops.aten.view.default(sum_62, [768]);  sum_62 = None
    mul_195: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_142, primals_59);  primals_59 = None
    mul_196: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_142, div_39);  add_142 = div_39 = None
    sum_63: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_196, [0, 1], True);  mul_196 = None
    view_288: "f32[768]" = torch.ops.aten.view.default(sum_63, [768]);  sum_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_84: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_79, sqrt_29);  sub_79 = None
    div_85: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_84, sqrt_29);  div_84 = None
    neg_16: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_195)
    mul_197: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_16, div_85);  neg_16 = div_85 = None
    div_86: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_195, sqrt_29);  mul_195 = sqrt_29 = None
    sum_64: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_197, [2], True);  mul_197 = None
    alias_50: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_29);  alias_29 = None
    mul_198: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_50, 2);  alias_50 = None
    div_87: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_64, mul_198);  sum_64 = mul_198 = None
    neg_17: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_86)
    sum_65: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_17, [2], True);  neg_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_58: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_87, [1, 512, 768]);  div_87 = None
    div_88: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_58, 768);  expand_58 = None
    pow_31: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_78, 1.0);  sub_78 = None
    mul_199: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_31, 2.0);  pow_31 = None
    mul_200: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_88, mul_199);  div_88 = mul_199 = None
    neg_18: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_200)
    sum_66: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_18, [2], True);  neg_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_143: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_86, mul_200);  div_86 = mul_200 = None
    add_144: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_65, sum_66);  sum_65 = sum_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_59: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_144, [1, 512, 768]);  add_144 = None
    div_89: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_59, 768);  expand_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_145: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_143, div_89);  add_143 = div_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_60: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_72: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_39, scalar_tensor_60, add_145);  convert_element_type_39 = scalar_tensor_60 = None
    mul_201: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_72, 1.1111111111111112);  where_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_289: "f32[512, 768]" = torch.ops.aten.view.default(mul_201, [512, 768]);  mul_201 = None
    permute_213: "f32[768, 768]" = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None
    mm_36: "f32[512, 768]" = torch.ops.aten.mm.default(view_289, permute_213);  permute_213 = None
    permute_214: "f32[768, 512]" = torch.ops.aten.permute.default(view_289, [1, 0])
    mm_37: "f32[768, 768]" = torch.ops.aten.mm.default(permute_214, view_174);  permute_214 = view_174 = None
    permute_215: "f32[768, 768]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_67: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_289, [0], True);  view_289 = None
    view_290: "f32[768]" = torch.ops.aten.view.default(sum_67, [768]);  sum_67 = None
    permute_216: "f32[768, 768]" = torch.ops.aten.permute.default(permute_215, [1, 0]);  permute_215 = None
    view_291: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_36, [1, 512, 768]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_292: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_291, [1, 512, 12, 64]);  view_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_217: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_292, [0, 2, 1, 3]);  view_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_293: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_217, [12, 512, 64]);  permute_217 = None
    permute_218: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_170, [0, 2, 1]);  view_170 = None
    bmm_32: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_218, view_293);  permute_218 = None
    permute_219: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_171, [0, 2, 1]);  view_171 = None
    bmm_33: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_293, permute_219);  view_293 = permute_219 = None
    view_294: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_32, [1, 12, 512, 64]);  bmm_32 = None
    view_295: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_33, [1, 12, 512, 512]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_61: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_73: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_38, scalar_tensor_61, view_295);  convert_element_type_38 = scalar_tensor_61 = view_295 = None
    mul_202: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_73, 1.1111111111111112);  where_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_52: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(where_47);  where_47 = None
    alias_53: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_52);  alias_52 = None
    mul_203: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_202, alias_53);  mul_202 = None
    sum_68: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_203, [-1], True)
    mul_204: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_53, sum_68);  alias_53 = sum_68 = None
    sub_108: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_203, mul_204);  mul_203 = mul_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_296: "f32[12, 512, 512]" = torch.ops.aten.view.default(sub_108, [12, 512, 512]);  sub_108 = None
    permute_220: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_167, [0, 2, 1]);  view_167 = None
    bmm_34: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_220, view_296);  permute_220 = None
    permute_221: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1]);  view_168 = None
    bmm_35: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_296, permute_221);  view_296 = permute_221 = None
    view_297: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_34, [1, 12, 64, 512]);  bmm_34 = None
    view_298: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_35, [1, 12, 512, 64]);  bmm_35 = None
    permute_222: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_297, [0, 1, 3, 2]);  view_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_90: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(view_298, sqrt_28);  view_298 = sqrt_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    sum_69: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(view_294, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_223: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_69, [0, 2, 1, 3]);  sum_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_299: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_223, [1, 1, 768]);  permute_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    full_7: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_4: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_7, view_299, 2, 0, 9223372036854775807);  full_7 = view_299 = None
    squeeze_10: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_4, 1);  slice_scatter_4 = None
    squeeze_11: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_10, 0);  squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    sum_70: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(div_90, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_224: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_70, [0, 2, 1, 3]);  sum_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_300: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_224, [1, 1, 768]);  permute_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    full_8: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_5: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_8, view_300, 2, 0, 9223372036854775807);  full_8 = view_300 = None
    squeeze_12: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_5, 1);  slice_scatter_5 = None
    squeeze_13: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_12, 0);  squeeze_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    cat_2: "f32[1, 12, 512, 192]" = torch.ops.aten.cat.default([div_90, permute_222, view_294], 3);  div_90 = permute_222 = view_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_225: "f32[1, 512, 12, 192]" = torch.ops.aten.permute.default(cat_2, [0, 2, 1, 3]);  cat_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    clone_14: "f32[1, 512, 12, 192]" = torch.ops.aten.clone.default(permute_225, memory_format = torch.contiguous_format);  permute_225 = None
    view_301: "f32[1, 512, 2304]" = torch.ops.aten.view.default(clone_14, [1, 512, 2304]);  clone_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_302: "f32[512, 2304]" = torch.ops.aten.view.default(view_301, [512, 2304]);  view_301 = None
    permute_226: "f32[2304, 512]" = torch.ops.aten.permute.default(view_302, [1, 0])
    mm_38: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_226, view_162);  permute_226 = view_162 = None
    permute_227: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_38, [1, 0]);  mm_38 = None
    permute_228: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
    mm_39: "f32[512, 768]" = torch.ops.aten.mm.default(view_302, permute_228);  view_302 = permute_228 = None
    view_303: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_39, [1, 512, 768]);  mm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    add_146: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_145, view_303);  add_145 = view_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_229: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_227, [1, 0]);  permute_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_71: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_146, [0, 1], True)
    view_304: "f32[768]" = torch.ops.aten.view.default(sum_71, [768]);  sum_71 = None
    mul_205: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_146, primals_55);  primals_55 = None
    mul_206: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_146, div_36);  add_146 = div_36 = None
    sum_72: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_206, [0, 1], True);  mul_206 = None
    view_305: "f32[768]" = torch.ops.aten.view.default(sum_72, [768]);  sum_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_91: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_74, sqrt_27);  sub_74 = None
    div_92: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_91, sqrt_27);  div_91 = None
    neg_19: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_205)
    mul_207: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_19, div_92);  neg_19 = div_92 = None
    div_93: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_205, sqrt_27);  mul_205 = sqrt_27 = None
    sum_73: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_207, [2], True);  mul_207 = None
    alias_54: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_27);  alias_27 = None
    mul_208: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_54, 2);  alias_54 = None
    div_94: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_73, mul_208);  sum_73 = mul_208 = None
    neg_20: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_93)
    sum_74: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_20, [2], True);  neg_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_60: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_94, [1, 512, 768]);  div_94 = None
    div_95: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_60, 768);  expand_60 = None
    pow_32: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_73, 1.0);  sub_73 = None
    mul_209: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_32, 2.0);  pow_32 = None
    mul_210: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_95, mul_209);  div_95 = mul_209 = None
    neg_21: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_210)
    sum_75: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_21, [2], True);  neg_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_147: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_93, mul_210);  div_93 = mul_210 = None
    add_148: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_74, sum_75);  sum_74 = sum_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_61: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_148, [1, 512, 768]);  add_148 = None
    div_96: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_61, 768);  expand_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_149: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_147, div_96);  add_147 = div_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_62: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_74: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_36, scalar_tensor_62, add_149);  convert_element_type_36 = scalar_tensor_62 = None
    mul_211: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_74, 1.1111111111111112);  where_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_306: "f32[512, 768]" = torch.ops.aten.view.default(mul_211, [512, 768]);  mul_211 = None
    permute_230: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
    mm_40: "f32[512, 3072]" = torch.ops.aten.mm.default(view_306, permute_230);  permute_230 = None
    permute_231: "f32[768, 512]" = torch.ops.aten.permute.default(view_306, [1, 0])
    mm_41: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_231, view_160);  permute_231 = view_160 = None
    permute_232: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_76: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_306, [0], True);  view_306 = None
    view_307: "f32[768]" = torch.ops.aten.view.default(sum_76, [768]);  sum_76 = None
    permute_233: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_232, [1, 0]);  permute_232 = None
    view_308: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_40, [1, 512, 3072]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_212: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_159, 0.7071067811865476)
    erf_17: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_212);  mul_212 = None
    add_150: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_213: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_150, 0.5);  add_150 = None
    mul_214: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_159, view_159)
    mul_215: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_214, -0.5);  mul_214 = None
    exp_18: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_215);  mul_215 = None
    mul_216: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
    mul_217: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_159, mul_216);  view_159 = mul_216 = None
    add_151: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_213, mul_217);  mul_213 = mul_217 = None
    mul_218: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_308, add_151);  view_308 = add_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_309: "f32[512, 3072]" = torch.ops.aten.view.default(mul_218, [512, 3072]);  mul_218 = None
    permute_234: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_106, [1, 0]);  permute_106 = None
    mm_42: "f32[512, 768]" = torch.ops.aten.mm.default(view_309, permute_234);  permute_234 = None
    permute_235: "f32[3072, 512]" = torch.ops.aten.permute.default(view_309, [1, 0])
    mm_43: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_235, view_158);  permute_235 = view_158 = None
    permute_236: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_77: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_309, [0], True);  view_309 = None
    view_310: "f32[3072]" = torch.ops.aten.view.default(sum_77, [3072]);  sum_77 = None
    permute_237: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_236, [1, 0]);  permute_236 = None
    view_311: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_42, [1, 512, 768]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    add_152: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_149, view_311);  add_149 = view_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_78: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_152, [0, 1], True)
    view_312: "f32[768]" = torch.ops.aten.view.default(sum_78, [768]);  sum_78 = None
    mul_219: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_152, primals_53);  primals_53 = None
    mul_220: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_152, div_35);  add_152 = div_35 = None
    sum_79: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_220, [0, 1], True);  mul_220 = None
    view_313: "f32[768]" = torch.ops.aten.view.default(sum_79, [768]);  sum_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_97: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_71, sqrt_26);  sub_71 = None
    div_98: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_97, sqrt_26);  div_97 = None
    neg_22: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_219)
    mul_221: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_22, div_98);  neg_22 = div_98 = None
    div_99: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_219, sqrt_26);  mul_219 = sqrt_26 = None
    sum_80: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_221, [2], True);  mul_221 = None
    alias_55: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_26);  alias_26 = None
    mul_222: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_55, 2);  alias_55 = None
    div_100: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_80, mul_222);  sum_80 = mul_222 = None
    neg_23: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_99)
    sum_81: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_23, [2], True);  neg_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_62: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_100, [1, 512, 768]);  div_100 = None
    div_101: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_62, 768);  expand_62 = None
    pow_33: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_70, 1.0);  sub_70 = None
    mul_223: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_33, 2.0);  pow_33 = None
    mul_224: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_101, mul_223);  div_101 = mul_223 = None
    neg_24: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_224)
    sum_82: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_24, [2], True);  neg_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_153: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_99, mul_224);  div_99 = mul_224 = None
    add_154: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_81, sum_82);  sum_81 = sum_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_63: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_154, [1, 512, 768]);  add_154 = None
    div_102: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_63, 768);  expand_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_155: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_153, div_102);  add_153 = div_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_63: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_75: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_35, scalar_tensor_63, add_155);  convert_element_type_35 = scalar_tensor_63 = None
    mul_225: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_75, 1.1111111111111112);  where_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_314: "f32[512, 768]" = torch.ops.aten.view.default(mul_225, [512, 768]);  mul_225 = None
    permute_238: "f32[768, 768]" = torch.ops.aten.permute.default(permute_104, [1, 0]);  permute_104 = None
    mm_44: "f32[512, 768]" = torch.ops.aten.mm.default(view_314, permute_238);  permute_238 = None
    permute_239: "f32[768, 512]" = torch.ops.aten.permute.default(view_314, [1, 0])
    mm_45: "f32[768, 768]" = torch.ops.aten.mm.default(permute_239, view_156);  permute_239 = view_156 = None
    permute_240: "f32[768, 768]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_83: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_314, [0], True);  view_314 = None
    view_315: "f32[768]" = torch.ops.aten.view.default(sum_83, [768]);  sum_83 = None
    permute_241: "f32[768, 768]" = torch.ops.aten.permute.default(permute_240, [1, 0]);  permute_240 = None
    view_316: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_44, [1, 512, 768]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_317: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_316, [1, 512, 12, 64]);  view_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_242: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_317, [0, 2, 1, 3]);  view_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_318: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_242, [12, 512, 64]);  permute_242 = None
    permute_243: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_152, [0, 2, 1]);  view_152 = None
    bmm_36: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_243, view_318);  permute_243 = None
    permute_244: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_153, [0, 2, 1]);  view_153 = None
    bmm_37: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_318, permute_244);  view_318 = permute_244 = None
    view_319: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_36, [1, 12, 512, 64]);  bmm_36 = None
    view_320: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_37, [1, 12, 512, 512]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_64: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_76: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_34, scalar_tensor_64, view_320);  convert_element_type_34 = scalar_tensor_64 = view_320 = None
    mul_226: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_76, 1.1111111111111112);  where_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_57: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(where_42);  where_42 = None
    alias_58: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_57);  alias_57 = None
    mul_227: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_226, alias_58);  mul_226 = None
    sum_84: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_227, [-1], True)
    mul_228: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_58, sum_84);  alias_58 = sum_84 = None
    sub_109: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_227, mul_228);  mul_227 = mul_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_321: "f32[12, 512, 512]" = torch.ops.aten.view.default(sub_109, [12, 512, 512]);  sub_109 = None
    permute_245: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_149, [0, 2, 1]);  view_149 = None
    bmm_38: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_245, view_321);  permute_245 = None
    permute_246: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_150, [0, 2, 1]);  view_150 = None
    bmm_39: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_321, permute_246);  view_321 = permute_246 = None
    view_322: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_38, [1, 12, 64, 512]);  bmm_38 = None
    view_323: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_39, [1, 12, 512, 64]);  bmm_39 = None
    permute_247: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_322, [0, 1, 3, 2]);  view_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_103: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(view_323, sqrt_25);  view_323 = sqrt_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    sum_85: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(view_319, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_248: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_85, [0, 2, 1, 3]);  sum_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_324: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_248, [1, 1, 768]);  permute_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    full_9: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_6: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_9, view_324, 2, 0, 9223372036854775807);  full_9 = view_324 = None
    squeeze_14: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_6, 1);  slice_scatter_6 = None
    squeeze_15: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_14, 0);  squeeze_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    sum_86: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(div_103, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_249: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_86, [0, 2, 1, 3]);  sum_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_325: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_249, [1, 1, 768]);  permute_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    full_10: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_7: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_10, view_325, 2, 0, 9223372036854775807);  full_10 = view_325 = None
    squeeze_16: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_7, 1);  slice_scatter_7 = None
    squeeze_17: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_16, 0);  squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    cat_3: "f32[1, 12, 512, 192]" = torch.ops.aten.cat.default([div_103, permute_247, view_319], 3);  div_103 = permute_247 = view_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_250: "f32[1, 512, 12, 192]" = torch.ops.aten.permute.default(cat_3, [0, 2, 1, 3]);  cat_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    clone_15: "f32[1, 512, 12, 192]" = torch.ops.aten.clone.default(permute_250, memory_format = torch.contiguous_format);  permute_250 = None
    view_326: "f32[1, 512, 2304]" = torch.ops.aten.view.default(clone_15, [1, 512, 2304]);  clone_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_327: "f32[512, 2304]" = torch.ops.aten.view.default(view_326, [512, 2304]);  view_326 = None
    permute_251: "f32[2304, 512]" = torch.ops.aten.permute.default(view_327, [1, 0])
    mm_46: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_251, view_144);  permute_251 = view_144 = None
    permute_252: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_46, [1, 0]);  mm_46 = None
    permute_253: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
    mm_47: "f32[512, 768]" = torch.ops.aten.mm.default(view_327, permute_253);  view_327 = permute_253 = None
    view_328: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_47, [1, 512, 768]);  mm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    add_156: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_155, view_328);  add_155 = view_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_254: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_252, [1, 0]);  permute_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_87: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_156, [0, 1], True)
    view_329: "f32[768]" = torch.ops.aten.view.default(sum_87, [768]);  sum_87 = None
    mul_229: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_156, primals_49);  primals_49 = None
    mul_230: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_156, div_32);  add_156 = div_32 = None
    sum_88: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_230, [0, 1], True);  mul_230 = None
    view_330: "f32[768]" = torch.ops.aten.view.default(sum_88, [768]);  sum_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_104: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_66, sqrt_24);  sub_66 = None
    div_105: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_104, sqrt_24);  div_104 = None
    neg_25: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_229)
    mul_231: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_25, div_105);  neg_25 = div_105 = None
    div_106: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_229, sqrt_24);  mul_229 = sqrt_24 = None
    sum_89: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_231, [2], True);  mul_231 = None
    alias_59: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_24);  alias_24 = None
    mul_232: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_59, 2);  alias_59 = None
    div_107: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_89, mul_232);  sum_89 = mul_232 = None
    neg_26: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_106)
    sum_90: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_26, [2], True);  neg_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_64: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_107, [1, 512, 768]);  div_107 = None
    div_108: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_64, 768);  expand_64 = None
    pow_34: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_65, 1.0);  sub_65 = None
    mul_233: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_34, 2.0);  pow_34 = None
    mul_234: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_108, mul_233);  div_108 = mul_233 = None
    neg_27: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_234)
    sum_91: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_27, [2], True);  neg_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_157: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_106, mul_234);  div_106 = mul_234 = None
    add_158: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_90, sum_91);  sum_90 = sum_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_65: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_158, [1, 512, 768]);  add_158 = None
    div_109: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_65, 768);  expand_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_159: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_157, div_109);  add_157 = div_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_65: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_77: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_32, scalar_tensor_65, add_159);  convert_element_type_32 = scalar_tensor_65 = None
    mul_235: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_77, 1.1111111111111112);  where_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_331: "f32[512, 768]" = torch.ops.aten.view.default(mul_235, [512, 768]);  mul_235 = None
    permute_255: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_95, [1, 0]);  permute_95 = None
    mm_48: "f32[512, 3072]" = torch.ops.aten.mm.default(view_331, permute_255);  permute_255 = None
    permute_256: "f32[768, 512]" = torch.ops.aten.permute.default(view_331, [1, 0])
    mm_49: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_256, view_142);  permute_256 = view_142 = None
    permute_257: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_92: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_331, [0], True);  view_331 = None
    view_332: "f32[768]" = torch.ops.aten.view.default(sum_92, [768]);  sum_92 = None
    permute_258: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_257, [1, 0]);  permute_257 = None
    view_333: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_48, [1, 512, 3072]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_236: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_141, 0.7071067811865476)
    erf_18: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_236);  mul_236 = None
    add_160: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_237: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_160, 0.5);  add_160 = None
    mul_238: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_141, view_141)
    mul_239: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_238, -0.5);  mul_238 = None
    exp_19: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_239);  mul_239 = None
    mul_240: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
    mul_241: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_141, mul_240);  view_141 = mul_240 = None
    add_161: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_237, mul_241);  mul_237 = mul_241 = None
    mul_242: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_333, add_161);  view_333 = add_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_334: "f32[512, 3072]" = torch.ops.aten.view.default(mul_242, [512, 3072]);  mul_242 = None
    permute_259: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_94, [1, 0]);  permute_94 = None
    mm_50: "f32[512, 768]" = torch.ops.aten.mm.default(view_334, permute_259);  permute_259 = None
    permute_260: "f32[3072, 512]" = torch.ops.aten.permute.default(view_334, [1, 0])
    mm_51: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_260, view_140);  permute_260 = view_140 = None
    permute_261: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_93: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_334, [0], True);  view_334 = None
    view_335: "f32[3072]" = torch.ops.aten.view.default(sum_93, [3072]);  sum_93 = None
    permute_262: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_261, [1, 0]);  permute_261 = None
    view_336: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_50, [1, 512, 768]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    add_162: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_159, view_336);  add_159 = view_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_94: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_162, [0, 1], True)
    view_337: "f32[768]" = torch.ops.aten.view.default(sum_94, [768]);  sum_94 = None
    mul_243: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_162, primals_47);  primals_47 = None
    mul_244: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_162, div_31);  add_162 = div_31 = None
    sum_95: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_244, [0, 1], True);  mul_244 = None
    view_338: "f32[768]" = torch.ops.aten.view.default(sum_95, [768]);  sum_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_110: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_63, sqrt_23);  sub_63 = None
    div_111: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_110, sqrt_23);  div_110 = None
    neg_28: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_243)
    mul_245: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_28, div_111);  neg_28 = div_111 = None
    div_112: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_243, sqrt_23);  mul_243 = sqrt_23 = None
    sum_96: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_245, [2], True);  mul_245 = None
    alias_60: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_23);  alias_23 = None
    mul_246: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_60, 2);  alias_60 = None
    div_113: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_96, mul_246);  sum_96 = mul_246 = None
    neg_29: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_112)
    sum_97: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_29, [2], True);  neg_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_66: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_113, [1, 512, 768]);  div_113 = None
    div_114: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_66, 768);  expand_66 = None
    pow_35: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_62, 1.0);  sub_62 = None
    mul_247: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_35, 2.0);  pow_35 = None
    mul_248: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_114, mul_247);  div_114 = mul_247 = None
    neg_30: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_248)
    sum_98: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_30, [2], True);  neg_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_163: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_112, mul_248);  div_112 = mul_248 = None
    add_164: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_97, sum_98);  sum_97 = sum_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_67: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_164, [1, 512, 768]);  add_164 = None
    div_115: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_67, 768);  expand_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_165: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_163, div_115);  add_163 = div_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_66: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_78: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_31, scalar_tensor_66, add_165);  convert_element_type_31 = scalar_tensor_66 = None
    mul_249: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_78, 1.1111111111111112);  where_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_339: "f32[512, 768]" = torch.ops.aten.view.default(mul_249, [512, 768]);  mul_249 = None
    permute_263: "f32[768, 768]" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
    mm_52: "f32[512, 768]" = torch.ops.aten.mm.default(view_339, permute_263);  permute_263 = None
    permute_264: "f32[768, 512]" = torch.ops.aten.permute.default(view_339, [1, 0])
    mm_53: "f32[768, 768]" = torch.ops.aten.mm.default(permute_264, view_138);  permute_264 = view_138 = None
    permute_265: "f32[768, 768]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_99: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_339, [0], True);  view_339 = None
    view_340: "f32[768]" = torch.ops.aten.view.default(sum_99, [768]);  sum_99 = None
    permute_266: "f32[768, 768]" = torch.ops.aten.permute.default(permute_265, [1, 0]);  permute_265 = None
    view_341: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_52, [1, 512, 768]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_342: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_341, [1, 512, 12, 64]);  view_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_267: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_342, [0, 2, 1, 3]);  view_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_343: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_267, [12, 512, 64]);  permute_267 = None
    permute_268: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_134, [0, 2, 1]);  view_134 = None
    bmm_40: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_268, view_343);  permute_268 = None
    permute_269: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_135, [0, 2, 1]);  view_135 = None
    bmm_41: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_343, permute_269);  view_343 = permute_269 = None
    view_344: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_40, [1, 12, 512, 64]);  bmm_40 = None
    view_345: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_41, [1, 12, 512, 512]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_67: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_79: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_30, scalar_tensor_67, view_345);  convert_element_type_30 = scalar_tensor_67 = view_345 = None
    mul_250: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_79, 1.1111111111111112);  where_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_62: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(where_37);  where_37 = None
    alias_63: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_62);  alias_62 = None
    mul_251: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_250, alias_63);  mul_250 = None
    sum_100: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_251, [-1], True)
    mul_252: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_63, sum_100);  alias_63 = sum_100 = None
    sub_110: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_251, mul_252);  mul_251 = mul_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_346: "f32[12, 512, 512]" = torch.ops.aten.view.default(sub_110, [12, 512, 512]);  sub_110 = None
    permute_270: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_131, [0, 2, 1]);  view_131 = None
    bmm_42: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_270, view_346);  permute_270 = None
    permute_271: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_132, [0, 2, 1]);  view_132 = None
    bmm_43: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_346, permute_271);  view_346 = permute_271 = None
    view_347: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_42, [1, 12, 64, 512]);  bmm_42 = None
    view_348: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_43, [1, 12, 512, 64]);  bmm_43 = None
    permute_272: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_347, [0, 1, 3, 2]);  view_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_116: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(view_348, sqrt_22);  view_348 = sqrt_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    sum_101: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(view_344, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_273: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_101, [0, 2, 1, 3]);  sum_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_349: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_273, [1, 1, 768]);  permute_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    full_11: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_8: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_11, view_349, 2, 0, 9223372036854775807);  full_11 = view_349 = None
    squeeze_18: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_8, 1);  slice_scatter_8 = None
    squeeze_19: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_18, 0);  squeeze_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    sum_102: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(div_116, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_274: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_102, [0, 2, 1, 3]);  sum_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_350: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_274, [1, 1, 768]);  permute_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    full_12: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_9: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_12, view_350, 2, 0, 9223372036854775807);  full_12 = view_350 = None
    squeeze_20: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_9, 1);  slice_scatter_9 = None
    squeeze_21: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_20, 0);  squeeze_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    cat_4: "f32[1, 12, 512, 192]" = torch.ops.aten.cat.default([div_116, permute_272, view_344], 3);  div_116 = permute_272 = view_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_275: "f32[1, 512, 12, 192]" = torch.ops.aten.permute.default(cat_4, [0, 2, 1, 3]);  cat_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    clone_16: "f32[1, 512, 12, 192]" = torch.ops.aten.clone.default(permute_275, memory_format = torch.contiguous_format);  permute_275 = None
    view_351: "f32[1, 512, 2304]" = torch.ops.aten.view.default(clone_16, [1, 512, 2304]);  clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_352: "f32[512, 2304]" = torch.ops.aten.view.default(view_351, [512, 2304]);  view_351 = None
    permute_276: "f32[2304, 512]" = torch.ops.aten.permute.default(view_352, [1, 0])
    mm_54: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_276, view_126);  permute_276 = view_126 = None
    permute_277: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_54, [1, 0]);  mm_54 = None
    permute_278: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
    mm_55: "f32[512, 768]" = torch.ops.aten.mm.default(view_352, permute_278);  view_352 = permute_278 = None
    view_353: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_55, [1, 512, 768]);  mm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    add_166: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_165, view_353);  add_165 = view_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_279: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_277, [1, 0]);  permute_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_103: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_166, [0, 1], True)
    view_354: "f32[768]" = torch.ops.aten.view.default(sum_103, [768]);  sum_103 = None
    mul_253: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_166, primals_43);  primals_43 = None
    mul_254: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_166, div_28);  add_166 = div_28 = None
    sum_104: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_254, [0, 1], True);  mul_254 = None
    view_355: "f32[768]" = torch.ops.aten.view.default(sum_104, [768]);  sum_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_117: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_58, sqrt_21);  sub_58 = None
    div_118: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_117, sqrt_21);  div_117 = None
    neg_31: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_253)
    mul_255: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_31, div_118);  neg_31 = div_118 = None
    div_119: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_253, sqrt_21);  mul_253 = sqrt_21 = None
    sum_105: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_255, [2], True);  mul_255 = None
    alias_64: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_21);  alias_21 = None
    mul_256: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_64, 2);  alias_64 = None
    div_120: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_105, mul_256);  sum_105 = mul_256 = None
    neg_32: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_119)
    sum_106: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_32, [2], True);  neg_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_68: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_120, [1, 512, 768]);  div_120 = None
    div_121: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_68, 768);  expand_68 = None
    pow_36: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_57, 1.0);  sub_57 = None
    mul_257: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_36, 2.0);  pow_36 = None
    mul_258: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_121, mul_257);  div_121 = mul_257 = None
    neg_33: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_258)
    sum_107: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_33, [2], True);  neg_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_167: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_119, mul_258);  div_119 = mul_258 = None
    add_168: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_106, sum_107);  sum_106 = sum_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_69: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_168, [1, 512, 768]);  add_168 = None
    div_122: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_69, 768);  expand_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_169: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_167, div_122);  add_167 = div_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_68: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_80: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_28, scalar_tensor_68, add_169);  convert_element_type_28 = scalar_tensor_68 = None
    mul_259: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_80, 1.1111111111111112);  where_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_356: "f32[512, 768]" = torch.ops.aten.view.default(mul_259, [512, 768]);  mul_259 = None
    permute_280: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_83, [1, 0]);  permute_83 = None
    mm_56: "f32[512, 3072]" = torch.ops.aten.mm.default(view_356, permute_280);  permute_280 = None
    permute_281: "f32[768, 512]" = torch.ops.aten.permute.default(view_356, [1, 0])
    mm_57: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_281, view_124);  permute_281 = view_124 = None
    permute_282: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_108: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_356, [0], True);  view_356 = None
    view_357: "f32[768]" = torch.ops.aten.view.default(sum_108, [768]);  sum_108 = None
    permute_283: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_282, [1, 0]);  permute_282 = None
    view_358: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_56, [1, 512, 3072]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_260: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_123, 0.7071067811865476)
    erf_19: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_260);  mul_260 = None
    add_170: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_261: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_170, 0.5);  add_170 = None
    mul_262: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_123, view_123)
    mul_263: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_262, -0.5);  mul_262 = None
    exp_20: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_263);  mul_263 = None
    mul_264: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
    mul_265: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_123, mul_264);  view_123 = mul_264 = None
    add_171: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_261, mul_265);  mul_261 = mul_265 = None
    mul_266: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_358, add_171);  view_358 = add_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_359: "f32[512, 3072]" = torch.ops.aten.view.default(mul_266, [512, 3072]);  mul_266 = None
    permute_284: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_82, [1, 0]);  permute_82 = None
    mm_58: "f32[512, 768]" = torch.ops.aten.mm.default(view_359, permute_284);  permute_284 = None
    permute_285: "f32[3072, 512]" = torch.ops.aten.permute.default(view_359, [1, 0])
    mm_59: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_285, view_122);  permute_285 = view_122 = None
    permute_286: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_109: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_359, [0], True);  view_359 = None
    view_360: "f32[3072]" = torch.ops.aten.view.default(sum_109, [3072]);  sum_109 = None
    permute_287: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_286, [1, 0]);  permute_286 = None
    view_361: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_58, [1, 512, 768]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    add_172: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_169, view_361);  add_169 = view_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_110: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_172, [0, 1], True)
    view_362: "f32[768]" = torch.ops.aten.view.default(sum_110, [768]);  sum_110 = None
    mul_267: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_172, primals_41);  primals_41 = None
    mul_268: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_172, div_27);  add_172 = div_27 = None
    sum_111: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_268, [0, 1], True);  mul_268 = None
    view_363: "f32[768]" = torch.ops.aten.view.default(sum_111, [768]);  sum_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_123: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_55, sqrt_20);  sub_55 = None
    div_124: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_123, sqrt_20);  div_123 = None
    neg_34: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_267)
    mul_269: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_34, div_124);  neg_34 = div_124 = None
    div_125: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_267, sqrt_20);  mul_267 = sqrt_20 = None
    sum_112: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_269, [2], True);  mul_269 = None
    alias_65: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_20);  alias_20 = None
    mul_270: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_65, 2);  alias_65 = None
    div_126: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_112, mul_270);  sum_112 = mul_270 = None
    neg_35: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_125)
    sum_113: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_35, [2], True);  neg_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_70: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_126, [1, 512, 768]);  div_126 = None
    div_127: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_70, 768);  expand_70 = None
    pow_37: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_54, 1.0);  sub_54 = None
    mul_271: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_37, 2.0);  pow_37 = None
    mul_272: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_127, mul_271);  div_127 = mul_271 = None
    neg_36: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_272)
    sum_114: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_36, [2], True);  neg_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_173: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_125, mul_272);  div_125 = mul_272 = None
    add_174: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_113, sum_114);  sum_113 = sum_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_71: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_174, [1, 512, 768]);  add_174 = None
    div_128: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_71, 768);  expand_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_175: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_173, div_128);  add_173 = div_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_69: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_81: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_27, scalar_tensor_69, add_175);  convert_element_type_27 = scalar_tensor_69 = None
    mul_273: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_81, 1.1111111111111112);  where_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_364: "f32[512, 768]" = torch.ops.aten.view.default(mul_273, [512, 768]);  mul_273 = None
    permute_288: "f32[768, 768]" = torch.ops.aten.permute.default(permute_80, [1, 0]);  permute_80 = None
    mm_60: "f32[512, 768]" = torch.ops.aten.mm.default(view_364, permute_288);  permute_288 = None
    permute_289: "f32[768, 512]" = torch.ops.aten.permute.default(view_364, [1, 0])
    mm_61: "f32[768, 768]" = torch.ops.aten.mm.default(permute_289, view_120);  permute_289 = view_120 = None
    permute_290: "f32[768, 768]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_115: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_364, [0], True);  view_364 = None
    view_365: "f32[768]" = torch.ops.aten.view.default(sum_115, [768]);  sum_115 = None
    permute_291: "f32[768, 768]" = torch.ops.aten.permute.default(permute_290, [1, 0]);  permute_290 = None
    view_366: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_60, [1, 512, 768]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_367: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_366, [1, 512, 12, 64]);  view_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_292: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_367, [0, 2, 1, 3]);  view_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_368: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_292, [12, 512, 64]);  permute_292 = None
    permute_293: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_116, [0, 2, 1]);  view_116 = None
    bmm_44: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_293, view_368);  permute_293 = None
    permute_294: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_117, [0, 2, 1]);  view_117 = None
    bmm_45: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_368, permute_294);  view_368 = permute_294 = None
    view_369: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_44, [1, 12, 512, 64]);  bmm_44 = None
    view_370: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_45, [1, 12, 512, 512]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_70: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_82: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_26, scalar_tensor_70, view_370);  convert_element_type_26 = scalar_tensor_70 = view_370 = None
    mul_274: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_82, 1.1111111111111112);  where_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_67: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(where_32);  where_32 = None
    alias_68: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_67);  alias_67 = None
    mul_275: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_274, alias_68);  mul_274 = None
    sum_116: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_275, [-1], True)
    mul_276: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_68, sum_116);  alias_68 = sum_116 = None
    sub_111: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_275, mul_276);  mul_275 = mul_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_371: "f32[12, 512, 512]" = torch.ops.aten.view.default(sub_111, [12, 512, 512]);  sub_111 = None
    permute_295: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_113, [0, 2, 1]);  view_113 = None
    bmm_46: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_295, view_371);  permute_295 = None
    permute_296: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_114, [0, 2, 1]);  view_114 = None
    bmm_47: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_371, permute_296);  view_371 = permute_296 = None
    view_372: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_46, [1, 12, 64, 512]);  bmm_46 = None
    view_373: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_47, [1, 12, 512, 64]);  bmm_47 = None
    permute_297: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_372, [0, 1, 3, 2]);  view_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_129: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(view_373, sqrt_19);  view_373 = sqrt_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    sum_117: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(view_369, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_298: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_117, [0, 2, 1, 3]);  sum_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_374: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_298, [1, 1, 768]);  permute_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    full_13: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_10: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_13, view_374, 2, 0, 9223372036854775807);  full_13 = view_374 = None
    squeeze_22: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_10, 1);  slice_scatter_10 = None
    squeeze_23: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_22, 0);  squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    sum_118: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(div_129, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_299: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_118, [0, 2, 1, 3]);  sum_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_375: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_299, [1, 1, 768]);  permute_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    full_14: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_11: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_14, view_375, 2, 0, 9223372036854775807);  full_14 = view_375 = None
    squeeze_24: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_11, 1);  slice_scatter_11 = None
    squeeze_25: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_24, 0);  squeeze_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    cat_5: "f32[1, 12, 512, 192]" = torch.ops.aten.cat.default([div_129, permute_297, view_369], 3);  div_129 = permute_297 = view_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_300: "f32[1, 512, 12, 192]" = torch.ops.aten.permute.default(cat_5, [0, 2, 1, 3]);  cat_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    clone_17: "f32[1, 512, 12, 192]" = torch.ops.aten.clone.default(permute_300, memory_format = torch.contiguous_format);  permute_300 = None
    view_376: "f32[1, 512, 2304]" = torch.ops.aten.view.default(clone_17, [1, 512, 2304]);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_377: "f32[512, 2304]" = torch.ops.aten.view.default(view_376, [512, 2304]);  view_376 = None
    permute_301: "f32[2304, 512]" = torch.ops.aten.permute.default(view_377, [1, 0])
    mm_62: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_301, view_108);  permute_301 = view_108 = None
    permute_302: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_62, [1, 0]);  mm_62 = None
    permute_303: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_73, [1, 0]);  permute_73 = None
    mm_63: "f32[512, 768]" = torch.ops.aten.mm.default(view_377, permute_303);  view_377 = permute_303 = None
    view_378: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_63, [1, 512, 768]);  mm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    add_176: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_175, view_378);  add_175 = view_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_304: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_302, [1, 0]);  permute_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_119: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_176, [0, 1], True)
    view_379: "f32[768]" = torch.ops.aten.view.default(sum_119, [768]);  sum_119 = None
    mul_277: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_176, primals_37);  primals_37 = None
    mul_278: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_176, div_24);  add_176 = div_24 = None
    sum_120: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_278, [0, 1], True);  mul_278 = None
    view_380: "f32[768]" = torch.ops.aten.view.default(sum_120, [768]);  sum_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_130: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_50, sqrt_18);  sub_50 = None
    div_131: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_130, sqrt_18);  div_130 = None
    neg_37: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_277)
    mul_279: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_37, div_131);  neg_37 = div_131 = None
    div_132: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_277, sqrt_18);  mul_277 = sqrt_18 = None
    sum_121: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_279, [2], True);  mul_279 = None
    alias_69: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_18);  alias_18 = None
    mul_280: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_69, 2);  alias_69 = None
    div_133: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_121, mul_280);  sum_121 = mul_280 = None
    neg_38: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_132)
    sum_122: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_38, [2], True);  neg_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_72: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_133, [1, 512, 768]);  div_133 = None
    div_134: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_72, 768);  expand_72 = None
    pow_38: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_49, 1.0);  sub_49 = None
    mul_281: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_38, 2.0);  pow_38 = None
    mul_282: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_134, mul_281);  div_134 = mul_281 = None
    neg_39: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_282)
    sum_123: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_39, [2], True);  neg_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_177: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_132, mul_282);  div_132 = mul_282 = None
    add_178: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_122, sum_123);  sum_122 = sum_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_73: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_178, [1, 512, 768]);  add_178 = None
    div_135: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_73, 768);  expand_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_179: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_177, div_135);  add_177 = div_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_71: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_83: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_24, scalar_tensor_71, add_179);  convert_element_type_24 = scalar_tensor_71 = None
    mul_283: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_83, 1.1111111111111112);  where_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_381: "f32[512, 768]" = torch.ops.aten.view.default(mul_283, [512, 768]);  mul_283 = None
    permute_305: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
    mm_64: "f32[512, 3072]" = torch.ops.aten.mm.default(view_381, permute_305);  permute_305 = None
    permute_306: "f32[768, 512]" = torch.ops.aten.permute.default(view_381, [1, 0])
    mm_65: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_306, view_106);  permute_306 = view_106 = None
    permute_307: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_124: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_381, [0], True);  view_381 = None
    view_382: "f32[768]" = torch.ops.aten.view.default(sum_124, [768]);  sum_124 = None
    permute_308: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_307, [1, 0]);  permute_307 = None
    view_383: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_64, [1, 512, 3072]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_284: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_105, 0.7071067811865476)
    erf_20: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_284);  mul_284 = None
    add_180: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_285: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_180, 0.5);  add_180 = None
    mul_286: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_105, view_105)
    mul_287: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_286, -0.5);  mul_286 = None
    exp_21: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_287);  mul_287 = None
    mul_288: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
    mul_289: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_105, mul_288);  view_105 = mul_288 = None
    add_181: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_285, mul_289);  mul_285 = mul_289 = None
    mul_290: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_383, add_181);  view_383 = add_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_384: "f32[512, 3072]" = torch.ops.aten.view.default(mul_290, [512, 3072]);  mul_290 = None
    permute_309: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
    mm_66: "f32[512, 768]" = torch.ops.aten.mm.default(view_384, permute_309);  permute_309 = None
    permute_310: "f32[3072, 512]" = torch.ops.aten.permute.default(view_384, [1, 0])
    mm_67: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_310, view_104);  permute_310 = view_104 = None
    permute_311: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_125: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_384, [0], True);  view_384 = None
    view_385: "f32[3072]" = torch.ops.aten.view.default(sum_125, [3072]);  sum_125 = None
    permute_312: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_311, [1, 0]);  permute_311 = None
    view_386: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_66, [1, 512, 768]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    add_182: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_179, view_386);  add_179 = view_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_126: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_182, [0, 1], True)
    view_387: "f32[768]" = torch.ops.aten.view.default(sum_126, [768]);  sum_126 = None
    mul_291: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_182, primals_35);  primals_35 = None
    mul_292: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_182, div_23);  add_182 = div_23 = None
    sum_127: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_292, [0, 1], True);  mul_292 = None
    view_388: "f32[768]" = torch.ops.aten.view.default(sum_127, [768]);  sum_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_136: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_47, sqrt_17);  sub_47 = None
    div_137: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_136, sqrt_17);  div_136 = None
    neg_40: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_291)
    mul_293: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_40, div_137);  neg_40 = div_137 = None
    div_138: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_291, sqrt_17);  mul_291 = sqrt_17 = None
    sum_128: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_293, [2], True);  mul_293 = None
    alias_70: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_17);  alias_17 = None
    mul_294: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_70, 2);  alias_70 = None
    div_139: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_128, mul_294);  sum_128 = mul_294 = None
    neg_41: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_138)
    sum_129: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_41, [2], True);  neg_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_74: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_139, [1, 512, 768]);  div_139 = None
    div_140: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_74, 768);  expand_74 = None
    pow_39: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_46, 1.0);  sub_46 = None
    mul_295: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_39, 2.0);  pow_39 = None
    mul_296: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_140, mul_295);  div_140 = mul_295 = None
    neg_42: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_296)
    sum_130: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_42, [2], True);  neg_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_183: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_138, mul_296);  div_138 = mul_296 = None
    add_184: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_129, sum_130);  sum_129 = sum_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_75: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_184, [1, 512, 768]);  add_184 = None
    div_141: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_75, 768);  expand_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_185: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_183, div_141);  add_183 = div_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_72: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_84: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_23, scalar_tensor_72, add_185);  convert_element_type_23 = scalar_tensor_72 = None
    mul_297: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_84, 1.1111111111111112);  where_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_389: "f32[512, 768]" = torch.ops.aten.view.default(mul_297, [512, 768]);  mul_297 = None
    permute_313: "f32[768, 768]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
    mm_68: "f32[512, 768]" = torch.ops.aten.mm.default(view_389, permute_313);  permute_313 = None
    permute_314: "f32[768, 512]" = torch.ops.aten.permute.default(view_389, [1, 0])
    mm_69: "f32[768, 768]" = torch.ops.aten.mm.default(permute_314, view_102);  permute_314 = view_102 = None
    permute_315: "f32[768, 768]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_131: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_389, [0], True);  view_389 = None
    view_390: "f32[768]" = torch.ops.aten.view.default(sum_131, [768]);  sum_131 = None
    permute_316: "f32[768, 768]" = torch.ops.aten.permute.default(permute_315, [1, 0]);  permute_315 = None
    view_391: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_68, [1, 512, 768]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_392: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_391, [1, 512, 12, 64]);  view_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_317: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_392, [0, 2, 1, 3]);  view_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_393: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_317, [12, 512, 64]);  permute_317 = None
    permute_318: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_98, [0, 2, 1]);  view_98 = None
    bmm_48: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_318, view_393);  permute_318 = None
    permute_319: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_99, [0, 2, 1]);  view_99 = None
    bmm_49: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_393, permute_319);  view_393 = permute_319 = None
    view_394: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_48, [1, 12, 512, 64]);  bmm_48 = None
    view_395: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_49, [1, 12, 512, 512]);  bmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_73: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_85: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_22, scalar_tensor_73, view_395);  convert_element_type_22 = scalar_tensor_73 = view_395 = None
    mul_298: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_85, 1.1111111111111112);  where_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_72: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(where_27);  where_27 = None
    alias_73: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_72);  alias_72 = None
    mul_299: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_298, alias_73);  mul_298 = None
    sum_132: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_299, [-1], True)
    mul_300: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_73, sum_132);  alias_73 = sum_132 = None
    sub_112: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_299, mul_300);  mul_299 = mul_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_396: "f32[12, 512, 512]" = torch.ops.aten.view.default(sub_112, [12, 512, 512]);  sub_112 = None
    permute_320: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_95, [0, 2, 1]);  view_95 = None
    bmm_50: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_320, view_396);  permute_320 = None
    permute_321: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_96, [0, 2, 1]);  view_96 = None
    bmm_51: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_396, permute_321);  view_396 = permute_321 = None
    view_397: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_50, [1, 12, 64, 512]);  bmm_50 = None
    view_398: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_51, [1, 12, 512, 64]);  bmm_51 = None
    permute_322: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_397, [0, 1, 3, 2]);  view_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_142: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(view_398, sqrt_16);  view_398 = sqrt_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    sum_133: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(view_394, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_323: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_133, [0, 2, 1, 3]);  sum_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_399: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_323, [1, 1, 768]);  permute_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    full_15: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_12: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_15, view_399, 2, 0, 9223372036854775807);  full_15 = view_399 = None
    squeeze_26: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_12, 1);  slice_scatter_12 = None
    squeeze_27: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_26, 0);  squeeze_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    sum_134: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(div_142, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_324: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_134, [0, 2, 1, 3]);  sum_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_400: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_324, [1, 1, 768]);  permute_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    full_16: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_13: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_16, view_400, 2, 0, 9223372036854775807);  full_16 = view_400 = None
    squeeze_28: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_13, 1);  slice_scatter_13 = None
    squeeze_29: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_28, 0);  squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    cat_6: "f32[1, 12, 512, 192]" = torch.ops.aten.cat.default([div_142, permute_322, view_394], 3);  div_142 = permute_322 = view_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_325: "f32[1, 512, 12, 192]" = torch.ops.aten.permute.default(cat_6, [0, 2, 1, 3]);  cat_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    clone_18: "f32[1, 512, 12, 192]" = torch.ops.aten.clone.default(permute_325, memory_format = torch.contiguous_format);  permute_325 = None
    view_401: "f32[1, 512, 2304]" = torch.ops.aten.view.default(clone_18, [1, 512, 2304]);  clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_402: "f32[512, 2304]" = torch.ops.aten.view.default(view_401, [512, 2304]);  view_401 = None
    permute_326: "f32[2304, 512]" = torch.ops.aten.permute.default(view_402, [1, 0])
    mm_70: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_326, view_90);  permute_326 = view_90 = None
    permute_327: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_70, [1, 0]);  mm_70 = None
    permute_328: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_61, [1, 0]);  permute_61 = None
    mm_71: "f32[512, 768]" = torch.ops.aten.mm.default(view_402, permute_328);  view_402 = permute_328 = None
    view_403: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_71, [1, 512, 768]);  mm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    add_186: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_185, view_403);  add_185 = view_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_329: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_327, [1, 0]);  permute_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_135: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_186, [0, 1], True)
    view_404: "f32[768]" = torch.ops.aten.view.default(sum_135, [768]);  sum_135 = None
    mul_301: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_186, primals_31);  primals_31 = None
    mul_302: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_186, div_20);  add_186 = div_20 = None
    sum_136: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_302, [0, 1], True);  mul_302 = None
    view_405: "f32[768]" = torch.ops.aten.view.default(sum_136, [768]);  sum_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_143: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_42, sqrt_15);  sub_42 = None
    div_144: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_143, sqrt_15);  div_143 = None
    neg_43: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_301)
    mul_303: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_43, div_144);  neg_43 = div_144 = None
    div_145: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_301, sqrt_15);  mul_301 = sqrt_15 = None
    sum_137: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_303, [2], True);  mul_303 = None
    alias_74: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_15);  alias_15 = None
    mul_304: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_74, 2);  alias_74 = None
    div_146: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_137, mul_304);  sum_137 = mul_304 = None
    neg_44: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_145)
    sum_138: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_44, [2], True);  neg_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_76: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_146, [1, 512, 768]);  div_146 = None
    div_147: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_76, 768);  expand_76 = None
    pow_40: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_41, 1.0);  sub_41 = None
    mul_305: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_40, 2.0);  pow_40 = None
    mul_306: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_147, mul_305);  div_147 = mul_305 = None
    neg_45: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_306)
    sum_139: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_45, [2], True);  neg_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_187: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_145, mul_306);  div_145 = mul_306 = None
    add_188: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_138, sum_139);  sum_138 = sum_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_77: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_188, [1, 512, 768]);  add_188 = None
    div_148: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_77, 768);  expand_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_189: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_187, div_148);  add_187 = div_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_74: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_86: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_20, scalar_tensor_74, add_189);  convert_element_type_20 = scalar_tensor_74 = None
    mul_307: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_86, 1.1111111111111112);  where_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_406: "f32[512, 768]" = torch.ops.aten.view.default(mul_307, [512, 768]);  mul_307 = None
    permute_330: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
    mm_72: "f32[512, 3072]" = torch.ops.aten.mm.default(view_406, permute_330);  permute_330 = None
    permute_331: "f32[768, 512]" = torch.ops.aten.permute.default(view_406, [1, 0])
    mm_73: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_331, view_88);  permute_331 = view_88 = None
    permute_332: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_140: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_406, [0], True);  view_406 = None
    view_407: "f32[768]" = torch.ops.aten.view.default(sum_140, [768]);  sum_140 = None
    permute_333: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_332, [1, 0]);  permute_332 = None
    view_408: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_72, [1, 512, 3072]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_308: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_87, 0.7071067811865476)
    erf_21: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_308);  mul_308 = None
    add_190: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_309: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_190, 0.5);  add_190 = None
    mul_310: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_87, view_87)
    mul_311: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_310, -0.5);  mul_310 = None
    exp_22: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_311);  mul_311 = None
    mul_312: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
    mul_313: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_87, mul_312);  view_87 = mul_312 = None
    add_191: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_309, mul_313);  mul_309 = mul_313 = None
    mul_314: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_408, add_191);  view_408 = add_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_409: "f32[512, 3072]" = torch.ops.aten.view.default(mul_314, [512, 3072]);  mul_314 = None
    permute_334: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
    mm_74: "f32[512, 768]" = torch.ops.aten.mm.default(view_409, permute_334);  permute_334 = None
    permute_335: "f32[3072, 512]" = torch.ops.aten.permute.default(view_409, [1, 0])
    mm_75: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_335, view_86);  permute_335 = view_86 = None
    permute_336: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_141: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_409, [0], True);  view_409 = None
    view_410: "f32[3072]" = torch.ops.aten.view.default(sum_141, [3072]);  sum_141 = None
    permute_337: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_336, [1, 0]);  permute_336 = None
    view_411: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_74, [1, 512, 768]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    add_192: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_189, view_411);  add_189 = view_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_142: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_192, [0, 1], True)
    view_412: "f32[768]" = torch.ops.aten.view.default(sum_142, [768]);  sum_142 = None
    mul_315: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_192, primals_29);  primals_29 = None
    mul_316: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_192, div_19);  add_192 = div_19 = None
    sum_143: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_316, [0, 1], True);  mul_316 = None
    view_413: "f32[768]" = torch.ops.aten.view.default(sum_143, [768]);  sum_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_149: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_39, sqrt_14);  sub_39 = None
    div_150: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_149, sqrt_14);  div_149 = None
    neg_46: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_315)
    mul_317: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_46, div_150);  neg_46 = div_150 = None
    div_151: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_315, sqrt_14);  mul_315 = sqrt_14 = None
    sum_144: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_317, [2], True);  mul_317 = None
    alias_75: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_14);  alias_14 = None
    mul_318: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_75, 2);  alias_75 = None
    div_152: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_144, mul_318);  sum_144 = mul_318 = None
    neg_47: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_151)
    sum_145: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_47, [2], True);  neg_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_78: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_152, [1, 512, 768]);  div_152 = None
    div_153: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_78, 768);  expand_78 = None
    pow_41: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_38, 1.0);  sub_38 = None
    mul_319: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_41, 2.0);  pow_41 = None
    mul_320: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_153, mul_319);  div_153 = mul_319 = None
    neg_48: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_320)
    sum_146: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_48, [2], True);  neg_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_193: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_151, mul_320);  div_151 = mul_320 = None
    add_194: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_145, sum_146);  sum_145 = sum_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_79: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_194, [1, 512, 768]);  add_194 = None
    div_154: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_79, 768);  expand_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_195: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_193, div_154);  add_193 = div_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_75: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_87: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_19, scalar_tensor_75, add_195);  convert_element_type_19 = scalar_tensor_75 = None
    mul_321: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_87, 1.1111111111111112);  where_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_414: "f32[512, 768]" = torch.ops.aten.view.default(mul_321, [512, 768]);  mul_321 = None
    permute_338: "f32[768, 768]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    mm_76: "f32[512, 768]" = torch.ops.aten.mm.default(view_414, permute_338);  permute_338 = None
    permute_339: "f32[768, 512]" = torch.ops.aten.permute.default(view_414, [1, 0])
    mm_77: "f32[768, 768]" = torch.ops.aten.mm.default(permute_339, view_84);  permute_339 = view_84 = None
    permute_340: "f32[768, 768]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_147: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_414, [0], True);  view_414 = None
    view_415: "f32[768]" = torch.ops.aten.view.default(sum_147, [768]);  sum_147 = None
    permute_341: "f32[768, 768]" = torch.ops.aten.permute.default(permute_340, [1, 0]);  permute_340 = None
    view_416: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_76, [1, 512, 768]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_417: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_416, [1, 512, 12, 64]);  view_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_342: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_417, [0, 2, 1, 3]);  view_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_418: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_342, [12, 512, 64]);  permute_342 = None
    permute_343: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_80, [0, 2, 1]);  view_80 = None
    bmm_52: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_343, view_418);  permute_343 = None
    permute_344: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_81, [0, 2, 1]);  view_81 = None
    bmm_53: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_418, permute_344);  view_418 = permute_344 = None
    view_419: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_52, [1, 12, 512, 64]);  bmm_52 = None
    view_420: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_53, [1, 12, 512, 512]);  bmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_76: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_88: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_18, scalar_tensor_76, view_420);  convert_element_type_18 = scalar_tensor_76 = view_420 = None
    mul_322: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_88, 1.1111111111111112);  where_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_77: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(where_22);  where_22 = None
    alias_78: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_77);  alias_77 = None
    mul_323: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_322, alias_78);  mul_322 = None
    sum_148: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_323, [-1], True)
    mul_324: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_78, sum_148);  alias_78 = sum_148 = None
    sub_113: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_323, mul_324);  mul_323 = mul_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_421: "f32[12, 512, 512]" = torch.ops.aten.view.default(sub_113, [12, 512, 512]);  sub_113 = None
    permute_345: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_77, [0, 2, 1]);  view_77 = None
    bmm_54: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_345, view_421);  permute_345 = None
    permute_346: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
    bmm_55: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_421, permute_346);  view_421 = permute_346 = None
    view_422: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_54, [1, 12, 64, 512]);  bmm_54 = None
    view_423: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_55, [1, 12, 512, 64]);  bmm_55 = None
    permute_347: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_422, [0, 1, 3, 2]);  view_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_155: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(view_423, sqrt_13);  view_423 = sqrt_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    sum_149: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(view_419, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_348: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_149, [0, 2, 1, 3]);  sum_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_424: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_348, [1, 1, 768]);  permute_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    full_17: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_14: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_17, view_424, 2, 0, 9223372036854775807);  full_17 = view_424 = None
    squeeze_30: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_14, 1);  slice_scatter_14 = None
    squeeze_31: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_30, 0);  squeeze_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    sum_150: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(div_155, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_349: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_150, [0, 2, 1, 3]);  sum_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_425: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_349, [1, 1, 768]);  permute_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    full_18: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_15: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_18, view_425, 2, 0, 9223372036854775807);  full_18 = view_425 = None
    squeeze_32: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_15, 1);  slice_scatter_15 = None
    squeeze_33: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_32, 0);  squeeze_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    cat_7: "f32[1, 12, 512, 192]" = torch.ops.aten.cat.default([div_155, permute_347, view_419], 3);  div_155 = permute_347 = view_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_350: "f32[1, 512, 12, 192]" = torch.ops.aten.permute.default(cat_7, [0, 2, 1, 3]);  cat_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    clone_19: "f32[1, 512, 12, 192]" = torch.ops.aten.clone.default(permute_350, memory_format = torch.contiguous_format);  permute_350 = None
    view_426: "f32[1, 512, 2304]" = torch.ops.aten.view.default(clone_19, [1, 512, 2304]);  clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_427: "f32[512, 2304]" = torch.ops.aten.view.default(view_426, [512, 2304]);  view_426 = None
    permute_351: "f32[2304, 512]" = torch.ops.aten.permute.default(view_427, [1, 0])
    mm_78: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_351, view_72);  permute_351 = view_72 = None
    permute_352: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_78, [1, 0]);  mm_78 = None
    permute_353: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
    mm_79: "f32[512, 768]" = torch.ops.aten.mm.default(view_427, permute_353);  view_427 = permute_353 = None
    view_428: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_79, [1, 512, 768]);  mm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    add_196: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_195, view_428);  add_195 = view_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_354: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_352, [1, 0]);  permute_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_151: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_196, [0, 1], True)
    view_429: "f32[768]" = torch.ops.aten.view.default(sum_151, [768]);  sum_151 = None
    mul_325: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_196, primals_25);  primals_25 = None
    mul_326: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_196, div_16);  add_196 = div_16 = None
    sum_152: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_326, [0, 1], True);  mul_326 = None
    view_430: "f32[768]" = torch.ops.aten.view.default(sum_152, [768]);  sum_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_156: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_34, sqrt_12);  sub_34 = None
    div_157: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_156, sqrt_12);  div_156 = None
    neg_49: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_325)
    mul_327: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_49, div_157);  neg_49 = div_157 = None
    div_158: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_325, sqrt_12);  mul_325 = sqrt_12 = None
    sum_153: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_327, [2], True);  mul_327 = None
    alias_79: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    mul_328: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_79, 2);  alias_79 = None
    div_159: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_153, mul_328);  sum_153 = mul_328 = None
    neg_50: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_158)
    sum_154: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_50, [2], True);  neg_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_80: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_159, [1, 512, 768]);  div_159 = None
    div_160: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_80, 768);  expand_80 = None
    pow_42: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_33, 1.0);  sub_33 = None
    mul_329: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_42, 2.0);  pow_42 = None
    mul_330: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_160, mul_329);  div_160 = mul_329 = None
    neg_51: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_330)
    sum_155: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_51, [2], True);  neg_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_197: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_158, mul_330);  div_158 = mul_330 = None
    add_198: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_154, sum_155);  sum_154 = sum_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_81: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_198, [1, 512, 768]);  add_198 = None
    div_161: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_81, 768);  expand_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_199: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_197, div_161);  add_197 = div_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_77: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_89: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_16, scalar_tensor_77, add_199);  convert_element_type_16 = scalar_tensor_77 = None
    mul_331: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_89, 1.1111111111111112);  where_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_431: "f32[512, 768]" = torch.ops.aten.view.default(mul_331, [512, 768]);  mul_331 = None
    permute_355: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
    mm_80: "f32[512, 3072]" = torch.ops.aten.mm.default(view_431, permute_355);  permute_355 = None
    permute_356: "f32[768, 512]" = torch.ops.aten.permute.default(view_431, [1, 0])
    mm_81: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_356, view_70);  permute_356 = view_70 = None
    permute_357: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_156: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_431, [0], True);  view_431 = None
    view_432: "f32[768]" = torch.ops.aten.view.default(sum_156, [768]);  sum_156 = None
    permute_358: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_357, [1, 0]);  permute_357 = None
    view_433: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_80, [1, 512, 3072]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_332: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_69, 0.7071067811865476)
    erf_22: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_332);  mul_332 = None
    add_200: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_333: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_200, 0.5);  add_200 = None
    mul_334: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_69, view_69)
    mul_335: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_334, -0.5);  mul_334 = None
    exp_23: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_335);  mul_335 = None
    mul_336: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
    mul_337: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_69, mul_336);  view_69 = mul_336 = None
    add_201: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_333, mul_337);  mul_333 = mul_337 = None
    mul_338: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_433, add_201);  view_433 = add_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_434: "f32[512, 3072]" = torch.ops.aten.view.default(mul_338, [512, 3072]);  mul_338 = None
    permute_359: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
    mm_82: "f32[512, 768]" = torch.ops.aten.mm.default(view_434, permute_359);  permute_359 = None
    permute_360: "f32[3072, 512]" = torch.ops.aten.permute.default(view_434, [1, 0])
    mm_83: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_360, view_68);  permute_360 = view_68 = None
    permute_361: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_157: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_434, [0], True);  view_434 = None
    view_435: "f32[3072]" = torch.ops.aten.view.default(sum_157, [3072]);  sum_157 = None
    permute_362: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_361, [1, 0]);  permute_361 = None
    view_436: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_82, [1, 512, 768]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    add_202: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_199, view_436);  add_199 = view_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_158: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_202, [0, 1], True)
    view_437: "f32[768]" = torch.ops.aten.view.default(sum_158, [768]);  sum_158 = None
    mul_339: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_202, primals_23);  primals_23 = None
    mul_340: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_202, div_15);  add_202 = div_15 = None
    sum_159: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_340, [0, 1], True);  mul_340 = None
    view_438: "f32[768]" = torch.ops.aten.view.default(sum_159, [768]);  sum_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_162: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_31, sqrt_11);  sub_31 = None
    div_163: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_162, sqrt_11);  div_162 = None
    neg_52: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_339)
    mul_341: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_52, div_163);  neg_52 = div_163 = None
    div_164: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_339, sqrt_11);  mul_339 = sqrt_11 = None
    sum_160: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_341, [2], True);  mul_341 = None
    alias_80: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    mul_342: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_80, 2);  alias_80 = None
    div_165: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_160, mul_342);  sum_160 = mul_342 = None
    neg_53: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_164)
    sum_161: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_53, [2], True);  neg_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_82: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_165, [1, 512, 768]);  div_165 = None
    div_166: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_82, 768);  expand_82 = None
    pow_43: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_30, 1.0);  sub_30 = None
    mul_343: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_43, 2.0);  pow_43 = None
    mul_344: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_166, mul_343);  div_166 = mul_343 = None
    neg_54: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_344)
    sum_162: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_54, [2], True);  neg_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_203: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_164, mul_344);  div_164 = mul_344 = None
    add_204: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_161, sum_162);  sum_161 = sum_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_83: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_204, [1, 512, 768]);  add_204 = None
    div_167: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_83, 768);  expand_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_205: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_203, div_167);  add_203 = div_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_78: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_90: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_15, scalar_tensor_78, add_205);  convert_element_type_15 = scalar_tensor_78 = None
    mul_345: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_90, 1.1111111111111112);  where_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_439: "f32[512, 768]" = torch.ops.aten.view.default(mul_345, [512, 768]);  mul_345 = None
    permute_363: "f32[768, 768]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
    mm_84: "f32[512, 768]" = torch.ops.aten.mm.default(view_439, permute_363);  permute_363 = None
    permute_364: "f32[768, 512]" = torch.ops.aten.permute.default(view_439, [1, 0])
    mm_85: "f32[768, 768]" = torch.ops.aten.mm.default(permute_364, view_66);  permute_364 = view_66 = None
    permute_365: "f32[768, 768]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_163: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_439, [0], True);  view_439 = None
    view_440: "f32[768]" = torch.ops.aten.view.default(sum_163, [768]);  sum_163 = None
    permute_366: "f32[768, 768]" = torch.ops.aten.permute.default(permute_365, [1, 0]);  permute_365 = None
    view_441: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_84, [1, 512, 768]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_442: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_441, [1, 512, 12, 64]);  view_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_367: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_442, [0, 2, 1, 3]);  view_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_443: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_367, [12, 512, 64]);  permute_367 = None
    permute_368: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_62, [0, 2, 1]);  view_62 = None
    bmm_56: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_368, view_443);  permute_368 = None
    permute_369: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_63, [0, 2, 1]);  view_63 = None
    bmm_57: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_443, permute_369);  view_443 = permute_369 = None
    view_444: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_56, [1, 12, 512, 64]);  bmm_56 = None
    view_445: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_57, [1, 12, 512, 512]);  bmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_79: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_91: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_14, scalar_tensor_79, view_445);  convert_element_type_14 = scalar_tensor_79 = view_445 = None
    mul_346: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_91, 1.1111111111111112);  where_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_82: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(where_17);  where_17 = None
    alias_83: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_82);  alias_82 = None
    mul_347: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_346, alias_83);  mul_346 = None
    sum_164: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_347, [-1], True)
    mul_348: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_83, sum_164);  alias_83 = sum_164 = None
    sub_114: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_347, mul_348);  mul_347 = mul_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_446: "f32[12, 512, 512]" = torch.ops.aten.view.default(sub_114, [12, 512, 512]);  sub_114 = None
    permute_370: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_59, [0, 2, 1]);  view_59 = None
    bmm_58: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_370, view_446);  permute_370 = None
    permute_371: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_60, [0, 2, 1]);  view_60 = None
    bmm_59: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_446, permute_371);  view_446 = permute_371 = None
    view_447: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_58, [1, 12, 64, 512]);  bmm_58 = None
    view_448: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_59, [1, 12, 512, 64]);  bmm_59 = None
    permute_372: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_447, [0, 1, 3, 2]);  view_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_168: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(view_448, sqrt_10);  view_448 = sqrt_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    sum_165: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(view_444, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_373: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_165, [0, 2, 1, 3]);  sum_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_449: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_373, [1, 1, 768]);  permute_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    full_19: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_16: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_19, view_449, 2, 0, 9223372036854775807);  full_19 = view_449 = None
    squeeze_34: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_16, 1);  slice_scatter_16 = None
    squeeze_35: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_34, 0);  squeeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    sum_166: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(div_168, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_374: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_166, [0, 2, 1, 3]);  sum_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_450: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_374, [1, 1, 768]);  permute_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    full_20: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_17: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_20, view_450, 2, 0, 9223372036854775807);  full_20 = view_450 = None
    squeeze_36: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_17, 1);  slice_scatter_17 = None
    squeeze_37: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_36, 0);  squeeze_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    cat_8: "f32[1, 12, 512, 192]" = torch.ops.aten.cat.default([div_168, permute_372, view_444], 3);  div_168 = permute_372 = view_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_375: "f32[1, 512, 12, 192]" = torch.ops.aten.permute.default(cat_8, [0, 2, 1, 3]);  cat_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    clone_20: "f32[1, 512, 12, 192]" = torch.ops.aten.clone.default(permute_375, memory_format = torch.contiguous_format);  permute_375 = None
    view_451: "f32[1, 512, 2304]" = torch.ops.aten.view.default(clone_20, [1, 512, 2304]);  clone_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_452: "f32[512, 2304]" = torch.ops.aten.view.default(view_451, [512, 2304]);  view_451 = None
    permute_376: "f32[2304, 512]" = torch.ops.aten.permute.default(view_452, [1, 0])
    mm_86: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_376, view_54);  permute_376 = view_54 = None
    permute_377: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_86, [1, 0]);  mm_86 = None
    permute_378: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
    mm_87: "f32[512, 768]" = torch.ops.aten.mm.default(view_452, permute_378);  view_452 = permute_378 = None
    view_453: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_87, [1, 512, 768]);  mm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    add_206: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_205, view_453);  add_205 = view_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_379: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_377, [1, 0]);  permute_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_167: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_206, [0, 1], True)
    view_454: "f32[768]" = torch.ops.aten.view.default(sum_167, [768]);  sum_167 = None
    mul_349: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_206, primals_19);  primals_19 = None
    mul_350: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_206, div_12);  add_206 = div_12 = None
    sum_168: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_350, [0, 1], True);  mul_350 = None
    view_455: "f32[768]" = torch.ops.aten.view.default(sum_168, [768]);  sum_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_169: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_26, sqrt_9);  sub_26 = None
    div_170: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_169, sqrt_9);  div_169 = None
    neg_55: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_349)
    mul_351: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_55, div_170);  neg_55 = div_170 = None
    div_171: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_349, sqrt_9);  mul_349 = sqrt_9 = None
    sum_169: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_351, [2], True);  mul_351 = None
    alias_84: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    mul_352: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_84, 2);  alias_84 = None
    div_172: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_169, mul_352);  sum_169 = mul_352 = None
    neg_56: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_171)
    sum_170: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_56, [2], True);  neg_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_84: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_172, [1, 512, 768]);  div_172 = None
    div_173: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_84, 768);  expand_84 = None
    pow_44: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_25, 1.0);  sub_25 = None
    mul_353: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_44, 2.0);  pow_44 = None
    mul_354: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_173, mul_353);  div_173 = mul_353 = None
    neg_57: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_354)
    sum_171: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_57, [2], True);  neg_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_207: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_171, mul_354);  div_171 = mul_354 = None
    add_208: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_170, sum_171);  sum_170 = sum_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_85: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_208, [1, 512, 768]);  add_208 = None
    div_174: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_85, 768);  expand_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_209: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_207, div_174);  add_207 = div_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_80: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_92: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_12, scalar_tensor_80, add_209);  convert_element_type_12 = scalar_tensor_80 = None
    mul_355: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_92, 1.1111111111111112);  where_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_456: "f32[512, 768]" = torch.ops.aten.view.default(mul_355, [512, 768]);  mul_355 = None
    permute_380: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
    mm_88: "f32[512, 3072]" = torch.ops.aten.mm.default(view_456, permute_380);  permute_380 = None
    permute_381: "f32[768, 512]" = torch.ops.aten.permute.default(view_456, [1, 0])
    mm_89: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_381, view_52);  permute_381 = view_52 = None
    permute_382: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    sum_172: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_456, [0], True);  view_456 = None
    view_457: "f32[768]" = torch.ops.aten.view.default(sum_172, [768]);  sum_172 = None
    permute_383: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_382, [1, 0]);  permute_382 = None
    view_458: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_88, [1, 512, 3072]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_356: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_51, 0.7071067811865476)
    erf_23: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_356);  mul_356 = None
    add_210: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_357: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_210, 0.5);  add_210 = None
    mul_358: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_51, view_51)
    mul_359: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_358, -0.5);  mul_358 = None
    exp_24: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_359);  mul_359 = None
    mul_360: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
    mul_361: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_51, mul_360);  view_51 = mul_360 = None
    add_211: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_357, mul_361);  mul_357 = mul_361 = None
    mul_362: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_458, add_211);  view_458 = add_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_459: "f32[512, 3072]" = torch.ops.aten.view.default(mul_362, [512, 3072]);  mul_362 = None
    permute_384: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    mm_90: "f32[512, 768]" = torch.ops.aten.mm.default(view_459, permute_384);  permute_384 = None
    permute_385: "f32[3072, 512]" = torch.ops.aten.permute.default(view_459, [1, 0])
    mm_91: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_385, view_50);  permute_385 = view_50 = None
    permute_386: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_173: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_459, [0], True);  view_459 = None
    view_460: "f32[3072]" = torch.ops.aten.view.default(sum_173, [3072]);  sum_173 = None
    permute_387: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_386, [1, 0]);  permute_386 = None
    view_461: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_90, [1, 512, 768]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    add_212: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_209, view_461);  add_209 = view_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_174: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_212, [0, 1], True)
    view_462: "f32[768]" = torch.ops.aten.view.default(sum_174, [768]);  sum_174 = None
    mul_363: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_212, primals_17);  primals_17 = None
    mul_364: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_212, div_11);  add_212 = div_11 = None
    sum_175: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_364, [0, 1], True);  mul_364 = None
    view_463: "f32[768]" = torch.ops.aten.view.default(sum_175, [768]);  sum_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_175: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_23, sqrt_8);  sub_23 = None
    div_176: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_175, sqrt_8);  div_175 = None
    neg_58: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_363)
    mul_365: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_58, div_176);  neg_58 = div_176 = None
    div_177: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_363, sqrt_8);  mul_363 = sqrt_8 = None
    sum_176: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_365, [2], True);  mul_365 = None
    alias_85: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    mul_366: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_85, 2);  alias_85 = None
    div_178: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_176, mul_366);  sum_176 = mul_366 = None
    neg_59: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_177)
    sum_177: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_59, [2], True);  neg_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_86: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_178, [1, 512, 768]);  div_178 = None
    div_179: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_86, 768);  expand_86 = None
    pow_45: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_22, 1.0);  sub_22 = None
    mul_367: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_45, 2.0);  pow_45 = None
    mul_368: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_179, mul_367);  div_179 = mul_367 = None
    neg_60: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_368)
    sum_178: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_60, [2], True);  neg_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_213: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_177, mul_368);  div_177 = mul_368 = None
    add_214: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_177, sum_178);  sum_177 = sum_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_87: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_214, [1, 512, 768]);  add_214 = None
    div_180: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_87, 768);  expand_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_215: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_213, div_180);  add_213 = div_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_81: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_93: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_11, scalar_tensor_81, add_215);  convert_element_type_11 = scalar_tensor_81 = None
    mul_369: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_93, 1.1111111111111112);  where_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_464: "f32[512, 768]" = torch.ops.aten.view.default(mul_369, [512, 768]);  mul_369 = None
    permute_388: "f32[768, 768]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    mm_92: "f32[512, 768]" = torch.ops.aten.mm.default(view_464, permute_388);  permute_388 = None
    permute_389: "f32[768, 512]" = torch.ops.aten.permute.default(view_464, [1, 0])
    mm_93: "f32[768, 768]" = torch.ops.aten.mm.default(permute_389, view_48);  permute_389 = view_48 = None
    permute_390: "f32[768, 768]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_179: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_464, [0], True);  view_464 = None
    view_465: "f32[768]" = torch.ops.aten.view.default(sum_179, [768]);  sum_179 = None
    permute_391: "f32[768, 768]" = torch.ops.aten.permute.default(permute_390, [1, 0]);  permute_390 = None
    view_466: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_92, [1, 512, 768]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_467: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_466, [1, 512, 12, 64]);  view_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_392: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_467, [0, 2, 1, 3]);  view_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_468: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_392, [12, 512, 64]);  permute_392 = None
    permute_393: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_44, [0, 2, 1]);  view_44 = None
    bmm_60: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_393, view_468);  permute_393 = None
    permute_394: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_45, [0, 2, 1]);  view_45 = None
    bmm_61: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_468, permute_394);  view_468 = permute_394 = None
    view_469: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_60, [1, 12, 512, 64]);  bmm_60 = None
    view_470: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_61, [1, 12, 512, 512]);  bmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_82: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_94: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_10, scalar_tensor_82, view_470);  convert_element_type_10 = scalar_tensor_82 = view_470 = None
    mul_370: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_94, 1.1111111111111112);  where_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_87: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(where_12);  where_12 = None
    alias_88: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_87);  alias_87 = None
    mul_371: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_370, alias_88);  mul_370 = None
    sum_180: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_371, [-1], True)
    mul_372: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_88, sum_180);  alias_88 = sum_180 = None
    sub_115: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_371, mul_372);  mul_371 = mul_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_471: "f32[12, 512, 512]" = torch.ops.aten.view.default(sub_115, [12, 512, 512]);  sub_115 = None
    permute_395: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_41, [0, 2, 1]);  view_41 = None
    bmm_62: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_395, view_471);  permute_395 = None
    permute_396: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_42, [0, 2, 1]);  view_42 = None
    bmm_63: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_471, permute_396);  view_471 = permute_396 = None
    view_472: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_62, [1, 12, 64, 512]);  bmm_62 = None
    view_473: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_63, [1, 12, 512, 64]);  bmm_63 = None
    permute_397: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_472, [0, 1, 3, 2]);  view_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_181: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(view_473, sqrt_7);  view_473 = sqrt_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    sum_181: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(view_469, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_398: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_181, [0, 2, 1, 3]);  sum_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_474: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_398, [1, 1, 768]);  permute_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    full_21: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_18: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_21, view_474, 2, 0, 9223372036854775807);  full_21 = view_474 = None
    squeeze_38: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_18, 1);  slice_scatter_18 = None
    squeeze_39: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_38, 0);  squeeze_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    sum_182: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(div_181, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_399: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_182, [0, 2, 1, 3]);  sum_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_475: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_399, [1, 1, 768]);  permute_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    full_22: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_19: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_22, view_475, 2, 0, 9223372036854775807);  full_22 = view_475 = None
    squeeze_40: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_19, 1);  slice_scatter_19 = None
    squeeze_41: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_40, 0);  squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    cat_9: "f32[1, 12, 512, 192]" = torch.ops.aten.cat.default([div_181, permute_397, view_469], 3);  div_181 = permute_397 = view_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_400: "f32[1, 512, 12, 192]" = torch.ops.aten.permute.default(cat_9, [0, 2, 1, 3]);  cat_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    clone_21: "f32[1, 512, 12, 192]" = torch.ops.aten.clone.default(permute_400, memory_format = torch.contiguous_format);  permute_400 = None
    view_476: "f32[1, 512, 2304]" = torch.ops.aten.view.default(clone_21, [1, 512, 2304]);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_477: "f32[512, 2304]" = torch.ops.aten.view.default(view_476, [512, 2304]);  view_476 = None
    permute_401: "f32[2304, 512]" = torch.ops.aten.permute.default(view_477, [1, 0])
    mm_94: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_401, view_36);  permute_401 = view_36 = None
    permute_402: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_94, [1, 0]);  mm_94 = None
    permute_403: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
    mm_95: "f32[512, 768]" = torch.ops.aten.mm.default(view_477, permute_403);  view_477 = permute_403 = None
    view_478: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_95, [1, 512, 768]);  mm_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    add_216: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_215, view_478);  add_215 = view_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_404: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_402, [1, 0]);  permute_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_183: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_216, [0, 1], True)
    view_479: "f32[768]" = torch.ops.aten.view.default(sum_183, [768]);  sum_183 = None
    mul_373: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_216, primals_13);  primals_13 = None
    mul_374: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_216, div_8);  add_216 = div_8 = None
    sum_184: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_374, [0, 1], True);  mul_374 = None
    view_480: "f32[768]" = torch.ops.aten.view.default(sum_184, [768]);  sum_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_182: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_18, sqrt_6);  sub_18 = None
    div_183: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_182, sqrt_6);  div_182 = None
    neg_61: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_373)
    mul_375: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_61, div_183);  neg_61 = div_183 = None
    div_184: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_373, sqrt_6);  mul_373 = sqrt_6 = None
    sum_185: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_375, [2], True);  mul_375 = None
    alias_89: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    mul_376: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_89, 2);  alias_89 = None
    div_185: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_185, mul_376);  sum_185 = mul_376 = None
    neg_62: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_184)
    sum_186: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_62, [2], True);  neg_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_88: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_185, [1, 512, 768]);  div_185 = None
    div_186: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_88, 768);  expand_88 = None
    pow_46: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_17, 1.0);  sub_17 = None
    mul_377: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_46, 2.0);  pow_46 = None
    mul_378: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_186, mul_377);  div_186 = mul_377 = None
    neg_63: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_378)
    sum_187: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_63, [2], True);  neg_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_217: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_184, mul_378);  div_184 = mul_378 = None
    add_218: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_186, sum_187);  sum_186 = sum_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_89: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_218, [1, 512, 768]);  add_218 = None
    div_187: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_89, 768);  expand_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_219: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_217, div_187);  add_217 = div_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_83: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_95: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_8, scalar_tensor_83, add_219);  convert_element_type_8 = scalar_tensor_83 = None
    mul_379: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_95, 1.1111111111111112);  where_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_481: "f32[512, 768]" = torch.ops.aten.view.default(mul_379, [512, 768]);  mul_379 = None
    permute_405: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    mm_96: "f32[512, 3072]" = torch.ops.aten.mm.default(view_481, permute_405);  permute_405 = None
    permute_406: "f32[768, 512]" = torch.ops.aten.permute.default(view_481, [1, 0])
    mm_97: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_406, view_34);  permute_406 = view_34 = None
    permute_407: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_188: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_481, [0], True);  view_481 = None
    view_482: "f32[768]" = torch.ops.aten.view.default(sum_188, [768]);  sum_188 = None
    permute_408: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_407, [1, 0]);  permute_407 = None
    view_483: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_96, [1, 512, 3072]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_380: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_33, 0.7071067811865476)
    erf_24: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_380);  mul_380 = None
    add_220: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
    mul_381: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_220, 0.5);  add_220 = None
    mul_382: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_33, view_33)
    mul_383: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_382, -0.5);  mul_382 = None
    exp_25: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_383);  mul_383 = None
    mul_384: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
    mul_385: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_33, mul_384);  view_33 = mul_384 = None
    add_221: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_381, mul_385);  mul_381 = mul_385 = None
    mul_386: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_483, add_221);  view_483 = add_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_484: "f32[512, 3072]" = torch.ops.aten.view.default(mul_386, [512, 3072]);  mul_386 = None
    permute_409: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    mm_98: "f32[512, 768]" = torch.ops.aten.mm.default(view_484, permute_409);  permute_409 = None
    permute_410: "f32[3072, 512]" = torch.ops.aten.permute.default(view_484, [1, 0])
    mm_99: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_410, view_32);  permute_410 = view_32 = None
    permute_411: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    sum_189: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_484, [0], True);  view_484 = None
    view_485: "f32[3072]" = torch.ops.aten.view.default(sum_189, [3072]);  sum_189 = None
    permute_412: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_411, [1, 0]);  permute_411 = None
    view_486: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_98, [1, 512, 768]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    add_222: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_219, view_486);  add_219 = view_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_190: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_222, [0, 1], True)
    view_487: "f32[768]" = torch.ops.aten.view.default(sum_190, [768]);  sum_190 = None
    mul_387: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_222, primals_11);  primals_11 = None
    mul_388: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_222, div_7);  add_222 = div_7 = None
    sum_191: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_388, [0, 1], True);  mul_388 = None
    view_488: "f32[768]" = torch.ops.aten.view.default(sum_191, [768]);  sum_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_188: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_15, sqrt_5);  sub_15 = None
    div_189: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_188, sqrt_5);  div_188 = None
    neg_64: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_387)
    mul_389: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_64, div_189);  neg_64 = div_189 = None
    div_190: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_387, sqrt_5);  mul_387 = sqrt_5 = None
    sum_192: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_389, [2], True);  mul_389 = None
    alias_90: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    mul_390: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_90, 2);  alias_90 = None
    div_191: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_192, mul_390);  sum_192 = mul_390 = None
    neg_65: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_190)
    sum_193: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_65, [2], True);  neg_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_90: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_191, [1, 512, 768]);  div_191 = None
    div_192: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_90, 768);  expand_90 = None
    pow_47: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_14, 1.0);  sub_14 = None
    mul_391: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_47, 2.0);  pow_47 = None
    mul_392: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_192, mul_391);  div_192 = mul_391 = None
    neg_66: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_392)
    sum_194: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_66, [2], True);  neg_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_223: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_190, mul_392);  div_190 = mul_392 = None
    add_224: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_193, sum_194);  sum_193 = sum_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_91: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_224, [1, 512, 768]);  add_224 = None
    div_193: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_91, 768);  expand_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_225: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_223, div_193);  add_223 = div_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_84: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_96: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_7, scalar_tensor_84, add_225);  convert_element_type_7 = scalar_tensor_84 = None
    mul_393: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_96, 1.1111111111111112);  where_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_489: "f32[512, 768]" = torch.ops.aten.view.default(mul_393, [512, 768]);  mul_393 = None
    permute_413: "f32[768, 768]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    mm_100: "f32[512, 768]" = torch.ops.aten.mm.default(view_489, permute_413);  permute_413 = None
    permute_414: "f32[768, 512]" = torch.ops.aten.permute.default(view_489, [1, 0])
    mm_101: "f32[768, 768]" = torch.ops.aten.mm.default(permute_414, view_30);  permute_414 = view_30 = None
    permute_415: "f32[768, 768]" = torch.ops.aten.permute.default(mm_101, [1, 0]);  mm_101 = None
    sum_195: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_489, [0], True);  view_489 = None
    view_490: "f32[768]" = torch.ops.aten.view.default(sum_195, [768]);  sum_195 = None
    permute_416: "f32[768, 768]" = torch.ops.aten.permute.default(permute_415, [1, 0]);  permute_415 = None
    view_491: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_100, [1, 512, 768]);  mm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_492: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_491, [1, 512, 12, 64]);  view_491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_417: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_492, [0, 2, 1, 3]);  view_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_493: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_417, [12, 512, 64]);  permute_417 = None
    permute_418: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_26, [0, 2, 1]);  view_26 = None
    bmm_64: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_418, view_493);  permute_418 = None
    permute_419: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_27, [0, 2, 1]);  view_27 = None
    bmm_65: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_493, permute_419);  view_493 = permute_419 = None
    view_494: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_64, [1, 12, 512, 64]);  bmm_64 = None
    view_495: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_65, [1, 12, 512, 512]);  bmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_85: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_97: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_6, scalar_tensor_85, view_495);  convert_element_type_6 = scalar_tensor_85 = view_495 = None
    mul_394: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_97, 1.1111111111111112);  where_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_92: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(where_7);  where_7 = None
    alias_93: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_92);  alias_92 = None
    mul_395: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_394, alias_93);  mul_394 = None
    sum_196: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_395, [-1], True)
    mul_396: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_93, sum_196);  alias_93 = sum_196 = None
    sub_116: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_395, mul_396);  mul_395 = mul_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_496: "f32[12, 512, 512]" = torch.ops.aten.view.default(sub_116, [12, 512, 512]);  sub_116 = None
    permute_420: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_23, [0, 2, 1]);  view_23 = None
    bmm_66: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_420, view_496);  permute_420 = None
    permute_421: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_24, [0, 2, 1]);  view_24 = None
    bmm_67: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_496, permute_421);  view_496 = permute_421 = None
    view_497: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_66, [1, 12, 64, 512]);  bmm_66 = None
    view_498: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_67, [1, 12, 512, 64]);  bmm_67 = None
    permute_422: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_497, [0, 1, 3, 2]);  view_497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_194: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(view_498, sqrt_4);  view_498 = sqrt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    sum_197: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(view_494, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_423: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_197, [0, 2, 1, 3]);  sum_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_499: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_423, [1, 1, 768]);  permute_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    full_23: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_20: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_23, view_499, 2, 0, 9223372036854775807);  full_23 = view_499 = None
    squeeze_42: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_20, 1);  slice_scatter_20 = None
    squeeze_43: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_42, 0);  squeeze_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    sum_198: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(div_194, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_424: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_198, [0, 2, 1, 3]);  sum_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_500: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_424, [1, 1, 768]);  permute_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    full_24: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_21: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_24, view_500, 2, 0, 9223372036854775807);  full_24 = view_500 = None
    squeeze_44: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_21, 1);  slice_scatter_21 = None
    squeeze_45: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_44, 0);  squeeze_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    cat_10: "f32[1, 12, 512, 192]" = torch.ops.aten.cat.default([div_194, permute_422, view_494], 3);  div_194 = permute_422 = view_494 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_425: "f32[1, 512, 12, 192]" = torch.ops.aten.permute.default(cat_10, [0, 2, 1, 3]);  cat_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    clone_22: "f32[1, 512, 12, 192]" = torch.ops.aten.clone.default(permute_425, memory_format = torch.contiguous_format);  permute_425 = None
    view_501: "f32[1, 512, 2304]" = torch.ops.aten.view.default(clone_22, [1, 512, 2304]);  clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_502: "f32[512, 2304]" = torch.ops.aten.view.default(view_501, [512, 2304]);  view_501 = None
    permute_426: "f32[2304, 512]" = torch.ops.aten.permute.default(view_502, [1, 0])
    mm_102: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_426, view_18);  permute_426 = view_18 = None
    permute_427: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_102, [1, 0]);  mm_102 = None
    permute_428: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    mm_103: "f32[512, 768]" = torch.ops.aten.mm.default(view_502, permute_428);  view_502 = permute_428 = None
    view_503: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_103, [1, 512, 768]);  mm_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    add_226: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_225, view_503);  add_225 = view_503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_429: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_427, [1, 0]);  permute_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_199: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_226, [0, 1], True)
    view_504: "f32[768]" = torch.ops.aten.view.default(sum_199, [768]);  sum_199 = None
    mul_397: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_226, primals_7);  primals_7 = None
    mul_398: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_226, div_4);  add_226 = div_4 = None
    sum_200: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_398, [0, 1], True);  mul_398 = None
    view_505: "f32[768]" = torch.ops.aten.view.default(sum_200, [768]);  sum_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_195: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_10, sqrt_3);  sub_10 = None
    div_196: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_195, sqrt_3);  div_195 = None
    neg_67: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_397)
    mul_399: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_67, div_196);  neg_67 = div_196 = None
    div_197: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_397, sqrt_3);  mul_397 = sqrt_3 = None
    sum_201: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_399, [2], True);  mul_399 = None
    alias_94: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    mul_400: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_94, 2);  alias_94 = None
    div_198: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_201, mul_400);  sum_201 = mul_400 = None
    neg_68: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_197)
    sum_202: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_68, [2], True);  neg_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_92: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_198, [1, 512, 768]);  div_198 = None
    div_199: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_92, 768);  expand_92 = None
    pow_48: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_9, 1.0);  sub_9 = None
    mul_401: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_48, 2.0);  pow_48 = None
    mul_402: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_199, mul_401);  div_199 = mul_401 = None
    neg_69: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_402)
    sum_203: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_69, [2], True);  neg_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_227: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_197, mul_402);  div_197 = mul_402 = None
    add_228: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_202, sum_203);  sum_202 = sum_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_93: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_228, [1, 512, 768]);  add_228 = None
    div_200: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_93, 768);  expand_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_229: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_227, div_200);  add_227 = div_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_86: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_98: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_4, scalar_tensor_86, add_229);  convert_element_type_4 = scalar_tensor_86 = None
    mul_403: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_98, 1.1111111111111112);  where_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_506: "f32[512, 768]" = torch.ops.aten.view.default(mul_403, [512, 768]);  mul_403 = None
    permute_430: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    mm_104: "f32[512, 3072]" = torch.ops.aten.mm.default(view_506, permute_430);  permute_430 = None
    permute_431: "f32[768, 512]" = torch.ops.aten.permute.default(view_506, [1, 0])
    mm_105: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_431, view_16);  permute_431 = view_16 = None
    permute_432: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
    sum_204: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_506, [0], True);  view_506 = None
    view_507: "f32[768]" = torch.ops.aten.view.default(sum_204, [768]);  sum_204 = None
    permute_433: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_432, [1, 0]);  permute_432 = None
    view_508: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_104, [1, 512, 3072]);  mm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_404: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_15, 0.7071067811865476)
    erf_25: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_404);  mul_404 = None
    add_230: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
    mul_405: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_230, 0.5);  add_230 = None
    mul_406: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_15, view_15)
    mul_407: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_406, -0.5);  mul_406 = None
    exp_26: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_407);  mul_407 = None
    mul_408: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
    mul_409: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_15, mul_408);  view_15 = mul_408 = None
    add_231: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_405, mul_409);  mul_405 = mul_409 = None
    mul_410: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_508, add_231);  view_508 = add_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_509: "f32[512, 3072]" = torch.ops.aten.view.default(mul_410, [512, 3072]);  mul_410 = None
    permute_434: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    mm_106: "f32[512, 768]" = torch.ops.aten.mm.default(view_509, permute_434);  permute_434 = None
    permute_435: "f32[3072, 512]" = torch.ops.aten.permute.default(view_509, [1, 0])
    mm_107: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_435, view_14);  permute_435 = view_14 = None
    permute_436: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
    sum_205: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_509, [0], True);  view_509 = None
    view_510: "f32[3072]" = torch.ops.aten.view.default(sum_205, [3072]);  sum_205 = None
    permute_437: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_436, [1, 0]);  permute_436 = None
    view_511: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_106, [1, 512, 768]);  mm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    add_232: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_229, view_511);  add_229 = view_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_206: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_232, [0, 1], True)
    view_512: "f32[768]" = torch.ops.aten.view.default(sum_206, [768]);  sum_206 = None
    mul_411: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_232, primals_5);  primals_5 = None
    mul_412: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_232, div_3);  add_232 = div_3 = None
    sum_207: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_412, [0, 1], True);  mul_412 = None
    view_513: "f32[768]" = torch.ops.aten.view.default(sum_207, [768]);  sum_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_201: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_7, sqrt_2);  sub_7 = None
    div_202: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_201, sqrt_2);  div_201 = None
    neg_70: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_411)
    mul_413: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_70, div_202);  neg_70 = div_202 = None
    div_203: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_411, sqrt_2);  mul_411 = sqrt_2 = None
    sum_208: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_413, [2], True);  mul_413 = None
    alias_95: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    mul_414: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_95, 2);  alias_95 = None
    div_204: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_208, mul_414);  sum_208 = mul_414 = None
    neg_71: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_203)
    sum_209: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_71, [2], True);  neg_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_94: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_204, [1, 512, 768]);  div_204 = None
    div_205: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_94, 768);  expand_94 = None
    pow_49: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_6, 1.0);  sub_6 = None
    mul_415: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_49, 2.0);  pow_49 = None
    mul_416: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_205, mul_415);  div_205 = mul_415 = None
    neg_72: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_416)
    sum_210: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_72, [2], True);  neg_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_233: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_203, mul_416);  div_203 = mul_416 = None
    add_234: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_209, sum_210);  sum_209 = sum_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_95: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_234, [1, 512, 768]);  add_234 = None
    div_206: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_95, 768);  expand_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_235: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_233, div_206);  add_233 = div_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_87: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_99: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_3, scalar_tensor_87, add_235);  convert_element_type_3 = scalar_tensor_87 = None
    mul_417: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_99, 1.1111111111111112);  where_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_514: "f32[512, 768]" = torch.ops.aten.view.default(mul_417, [512, 768]);  mul_417 = None
    permute_438: "f32[768, 768]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    mm_108: "f32[512, 768]" = torch.ops.aten.mm.default(view_514, permute_438);  permute_438 = None
    permute_439: "f32[768, 512]" = torch.ops.aten.permute.default(view_514, [1, 0])
    mm_109: "f32[768, 768]" = torch.ops.aten.mm.default(permute_439, view_12);  permute_439 = view_12 = None
    permute_440: "f32[768, 768]" = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
    sum_211: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_514, [0], True);  view_514 = None
    view_515: "f32[768]" = torch.ops.aten.view.default(sum_211, [768]);  sum_211 = None
    permute_441: "f32[768, 768]" = torch.ops.aten.permute.default(permute_440, [1, 0]);  permute_440 = None
    view_516: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_108, [1, 512, 768]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_517: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_516, [1, 512, 12, 64]);  view_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_442: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_517, [0, 2, 1, 3]);  view_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_518: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_442, [12, 512, 64]);  permute_442 = None
    permute_443: "f32[12, 512, 512]" = torch.ops.aten.permute.default(view_8, [0, 2, 1]);  view_8 = None
    bmm_68: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_443, view_518);  permute_443 = None
    permute_444: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
    bmm_69: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_518, permute_444);  view_518 = permute_444 = None
    view_519: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_68, [1, 12, 512, 64]);  bmm_68 = None
    view_520: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_69, [1, 12, 512, 512]);  bmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_88: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_100: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_2, scalar_tensor_88, view_520);  convert_element_type_2 = scalar_tensor_88 = view_520 = None
    mul_418: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_100, 1.1111111111111112);  where_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    alias_97: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(where_2);  where_2 = None
    alias_98: "f32[1, 12, 512, 512]" = torch.ops.aten.alias.default(alias_97);  alias_97 = None
    mul_419: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_418, alias_98);  mul_418 = None
    sum_212: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_419, [-1], True)
    mul_420: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_98, sum_212);  alias_98 = sum_212 = None
    sub_117: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_419, mul_420);  mul_419 = mul_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_521: "f32[12, 512, 512]" = torch.ops.aten.view.default(sub_117, [12, 512, 512]);  sub_117 = None
    permute_445: "f32[12, 64, 512]" = torch.ops.aten.permute.default(view_5, [0, 2, 1]);  view_5 = None
    bmm_70: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_445, view_521);  permute_445 = None
    permute_446: "f32[12, 512, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1]);  view_6 = None
    bmm_71: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_521, permute_446);  view_521 = permute_446 = None
    view_522: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_70, [1, 12, 64, 512]);  bmm_70 = None
    view_523: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_71, [1, 12, 512, 64]);  bmm_71 = None
    permute_447: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_522, [0, 1, 3, 2]);  view_522 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_207: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(view_523, sqrt_1);  view_523 = sqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    sum_213: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(view_519, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_448: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_213, [0, 2, 1, 3]);  sum_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_524: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_448, [1, 1, 768]);  permute_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    full_25: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_22: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_25, view_524, 2, 0, 9223372036854775807);  full_25 = view_524 = None
    squeeze_46: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_22, 1);  slice_scatter_22 = None
    squeeze_47: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_46, 0);  squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    sum_214: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(div_207, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_449: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_214, [0, 2, 1, 3]);  sum_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_525: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_449, [1, 1, 768]);  permute_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    full_26: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_23: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_26, view_525, 2, 0, 9223372036854775807);  full_26 = view_525 = None
    squeeze_48: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_23, 1);  slice_scatter_23 = None
    squeeze_49: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_48, 0);  squeeze_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    cat_11: "f32[1, 12, 512, 192]" = torch.ops.aten.cat.default([div_207, permute_447, view_519], 3);  div_207 = permute_447 = view_519 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_450: "f32[1, 512, 12, 192]" = torch.ops.aten.permute.default(cat_11, [0, 2, 1, 3]);  cat_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    clone_23: "f32[1, 512, 12, 192]" = torch.ops.aten.clone.default(permute_450, memory_format = torch.contiguous_format);  permute_450 = None
    view_526: "f32[1, 512, 2304]" = torch.ops.aten.view.default(clone_23, [1, 512, 2304]);  clone_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_527: "f32[512, 2304]" = torch.ops.aten.view.default(view_526, [512, 2304]);  view_526 = None
    permute_451: "f32[2304, 512]" = torch.ops.aten.permute.default(view_527, [1, 0])
    mm_110: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_451, view);  permute_451 = view = None
    permute_452: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_110, [1, 0]);  mm_110 = None
    permute_453: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    mm_111: "f32[512, 768]" = torch.ops.aten.mm.default(view_527, permute_453);  view_527 = permute_453 = None
    view_528: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_111, [1, 512, 768]);  mm_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    add_236: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_235, view_528);  add_235 = view_528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_454: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_452, [1, 0]);  permute_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    scalar_tensor_89: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_101: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type, scalar_tensor_89, add_236);  convert_element_type = scalar_tensor_89 = add_236 = None
    mul_421: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_101, 1.1111111111111112);  where_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:812, code: embeddings = embeddings * mask
    mul_422: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_421, unsqueeze);  mul_421 = unsqueeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_215: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_422, [0, 1], True)
    view_529: "f32[768]" = torch.ops.aten.view.default(sum_215, [768]);  sum_215 = None
    mul_423: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_422, primals_1);  primals_1 = None
    mul_424: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_422, div);  mul_422 = div = None
    sum_216: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_424, [0, 1], True);  mul_424 = None
    view_530: "f32[768]" = torch.ops.aten.view.default(sum_216, [768]);  sum_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_208: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_1, sqrt);  sub_1 = None
    div_209: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_208, sqrt);  div_208 = None
    neg_73: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_423)
    mul_425: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_73, div_209);  neg_73 = div_209 = None
    div_210: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_423, sqrt);  mul_423 = sqrt = None
    sum_217: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_425, [2], True);  mul_425 = None
    alias_99: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias);  alias = None
    mul_426: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_99, 2);  alias_99 = None
    div_211: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_217, mul_426);  sum_217 = mul_426 = None
    neg_74: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_210)
    sum_218: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_74, [2], True);  neg_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_96: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_211, [1, 512, 768]);  div_211 = None
    div_212: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_96, 768);  expand_96 = None
    pow_50: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub, 1.0);  sub = None
    mul_427: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_50, 2.0);  pow_50 = None
    mul_428: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_212, mul_427);  div_212 = mul_427 = None
    neg_75: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_428)
    sum_219: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_75, [2], True);  neg_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_237: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_210, mul_428);  div_210 = mul_428 = None
    add_238: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_218, sum_219);  sum_218 = sum_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_97: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_238, [1, 512, 768]);  add_238 = None
    div_213: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_97, 768);  expand_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_239: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_237, div_213);  add_237 = div_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:789, code: position_embeddings = self.position_embeddings(position_ids.long())
    eq: "b8[1, 512]" = torch.ops.aten.eq.Scalar(slice_1, -1)
    unsqueeze_54: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
    scalar_tensor_90: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_102: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_54, scalar_tensor_90, add_239);  unsqueeze_54 = scalar_tensor_90 = None
    full_27: "f32[512, 768]" = torch.ops.aten.full.default([512, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put: "f32[512, 768]" = torch.ops.aten._unsafe_index_put.default(full_27, [slice_1], where_102, True);  full_27 = slice_1 = where_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:786, code: inputs_embeds = self.word_embeddings(input_ids)
    eq_1: "b8[1, 512]" = torch.ops.aten.eq.Scalar(primals_168, 0)
    unsqueeze_55: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
    scalar_tensor_91: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_103: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_55, scalar_tensor_91, add_239);  unsqueeze_55 = scalar_tensor_91 = add_239 = None
    full_28: "f32[50265, 768]" = torch.ops.aten.full.default([50265, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put_1: "f32[50265, 768]" = torch.ops.aten._unsafe_index_put.default(full_28, [primals_168], where_103, True);  full_28 = primals_168 = where_103 = None
    return pytree.tree_unflatten([div_49, view_219, view_530, view_529, squeeze_49, squeeze_47, view_513, view_512, view_505, view_504, squeeze_45, squeeze_43, view_488, view_487, view_480, view_479, squeeze_41, squeeze_39, view_463, view_462, view_455, view_454, squeeze_37, squeeze_35, view_438, view_437, view_430, view_429, squeeze_33, squeeze_31, view_413, view_412, view_405, view_404, squeeze_29, squeeze_27, view_388, view_387, view_380, view_379, squeeze_25, squeeze_23, view_363, view_362, view_355, view_354, squeeze_21, squeeze_19, view_338, view_337, view_330, view_329, squeeze_17, squeeze_15, view_313, view_312, view_305, view_304, squeeze_13, squeeze_11, view_288, view_287, view_280, view_279, squeeze_9, squeeze_7, view_263, view_262, view_255, view_254, squeeze_5, squeeze_3, view_238, view_237, view_230, view_229, _unsafe_index_put_1, _unsafe_index_put, permute_454, permute_441, view_515, permute_437, view_510, permute_433, view_507, permute_429, permute_416, view_490, permute_412, view_485, permute_408, view_482, permute_404, permute_391, view_465, permute_387, view_460, permute_383, view_457, permute_379, permute_366, view_440, permute_362, view_435, permute_358, view_432, permute_354, permute_341, view_415, permute_337, view_410, permute_333, view_407, permute_329, permute_316, view_390, permute_312, view_385, permute_308, view_382, permute_304, permute_291, view_365, permute_287, view_360, permute_283, view_357, permute_279, permute_266, view_340, permute_262, view_335, permute_258, view_332, permute_254, permute_241, view_315, permute_237, view_310, permute_233, view_307, permute_229, permute_216, view_290, permute_212, view_285, permute_208, view_282, permute_204, permute_191, view_265, permute_187, view_260, permute_183, view_257, permute_179, permute_166, view_240, permute_162, view_235, permute_158, view_232, permute_154, view_227, sum_20, sum_21, permute_150, view_224, None, None, None], self._out_spec)
    