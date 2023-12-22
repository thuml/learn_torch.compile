from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[2304]"; primals_2: "f32[768, 2304]"; primals_3: "f32[768]"; primals_4: "f32[768, 768]"; primals_5: "f32[3072]"; primals_6: "f32[768, 3072]"; primals_7: "f32[768]"; primals_8: "f32[3072, 768]"; primals_9: "f32[2304]"; primals_10: "f32[768, 2304]"; primals_11: "f32[768]"; primals_12: "f32[768, 768]"; primals_13: "f32[3072]"; primals_14: "f32[768, 3072]"; primals_15: "f32[768]"; primals_16: "f32[3072, 768]"; primals_17: "f32[2304]"; primals_18: "f32[768, 2304]"; primals_19: "f32[768]"; primals_20: "f32[768, 768]"; primals_21: "f32[3072]"; primals_22: "f32[768, 3072]"; primals_23: "f32[768]"; primals_24: "f32[3072, 768]"; primals_25: "f32[2304]"; primals_26: "f32[768, 2304]"; primals_27: "f32[768]"; primals_28: "f32[768, 768]"; primals_29: "f32[3072]"; primals_30: "f32[768, 3072]"; primals_31: "f32[768]"; primals_32: "f32[3072, 768]"; primals_33: "f32[2304]"; primals_34: "f32[768, 2304]"; primals_35: "f32[768]"; primals_36: "f32[768, 768]"; primals_37: "f32[3072]"; primals_38: "f32[768, 3072]"; primals_39: "f32[768]"; primals_40: "f32[3072, 768]"; primals_41: "f32[2304]"; primals_42: "f32[768, 2304]"; primals_43: "f32[768]"; primals_44: "f32[768, 768]"; primals_45: "f32[3072]"; primals_46: "f32[768, 3072]"; primals_47: "f32[768]"; primals_48: "f32[3072, 768]"; primals_49: "f32[2304]"; primals_50: "f32[768, 2304]"; primals_51: "f32[768]"; primals_52: "f32[768, 768]"; primals_53: "f32[3072]"; primals_54: "f32[768, 3072]"; primals_55: "f32[768]"; primals_56: "f32[3072, 768]"; primals_57: "f32[2304]"; primals_58: "f32[768, 2304]"; primals_59: "f32[768]"; primals_60: "f32[768, 768]"; primals_61: "f32[3072]"; primals_62: "f32[768, 3072]"; primals_63: "f32[768]"; primals_64: "f32[3072, 768]"; primals_65: "f32[2304]"; primals_66: "f32[768, 2304]"; primals_67: "f32[768]"; primals_68: "f32[768, 768]"; primals_69: "f32[3072]"; primals_70: "f32[768, 3072]"; primals_71: "f32[768]"; primals_72: "f32[3072, 768]"; primals_73: "f32[2304]"; primals_74: "f32[768, 2304]"; primals_75: "f32[768]"; primals_76: "f32[768, 768]"; primals_77: "f32[3072]"; primals_78: "f32[768, 3072]"; primals_79: "f32[768]"; primals_80: "f32[3072, 768]"; primals_81: "f32[2304]"; primals_82: "f32[768, 2304]"; primals_83: "f32[768]"; primals_84: "f32[768, 768]"; primals_85: "f32[3072]"; primals_86: "f32[768, 3072]"; primals_87: "f32[768]"; primals_88: "f32[3072, 768]"; primals_89: "f32[2304]"; primals_90: "f32[768, 2304]"; primals_91: "f32[768]"; primals_92: "f32[768, 768]"; primals_93: "f32[3072]"; primals_94: "f32[768, 3072]"; primals_95: "f32[768]"; primals_96: "f32[3072, 768]"; primals_97: "f32[50257, 768]"; primals_98: "f32[1024, 768]"; primals_99: "f32[768]"; primals_100: "f32[768]"; primals_101: "f32[768]"; primals_102: "f32[768]"; primals_103: "f32[768]"; primals_104: "f32[768]"; primals_105: "f32[768]"; primals_106: "f32[768]"; primals_107: "f32[768]"; primals_108: "f32[768]"; primals_109: "f32[768]"; primals_110: "f32[768]"; primals_111: "f32[768]"; primals_112: "f32[768]"; primals_113: "f32[768]"; primals_114: "f32[768]"; primals_115: "f32[768]"; primals_116: "f32[768]"; primals_117: "f32[768]"; primals_118: "f32[768]"; primals_119: "f32[768]"; primals_120: "f32[768]"; primals_121: "f32[768]"; primals_122: "f32[768]"; primals_123: "f32[768]"; primals_124: "f32[768]"; primals_125: "f32[768]"; primals_126: "f32[768]"; primals_127: "f32[768]"; primals_128: "f32[768]"; primals_129: "f32[768]"; primals_130: "f32[768]"; primals_131: "f32[768]"; primals_132: "f32[768]"; primals_133: "f32[768]"; primals_134: "f32[768]"; primals_135: "f32[768]"; primals_136: "f32[768]"; primals_137: "f32[768]"; primals_138: "f32[768]"; primals_139: "f32[768]"; primals_140: "f32[768]"; primals_141: "f32[768]"; primals_142: "f32[768]"; primals_143: "f32[768]"; primals_144: "f32[768]"; primals_145: "f32[768]"; primals_146: "f32[768]"; primals_147: "f32[768]"; primals_148: "f32[768]"; primals_149: "f32[2, 768]"; primals_150: "b8[1, 1, 1024, 1024]"; primals_151: "b8[1, 1, 1024, 1024]"; primals_152: "b8[1, 1, 1024, 1024]"; primals_153: "b8[1, 1, 1024, 1024]"; primals_154: "b8[1, 1, 1024, 1024]"; primals_155: "b8[1, 1, 1024, 1024]"; primals_156: "b8[1, 1, 1024, 1024]"; primals_157: "b8[1, 1, 1024, 1024]"; primals_158: "b8[1, 1, 1024, 1024]"; primals_159: "b8[1, 1, 1024, 1024]"; primals_160: "b8[1, 1, 1024, 1024]"; primals_161: "b8[1, 1, 1024, 1024]"; primals_162: "i64[1, 1024]"; tangents_1: "f32[1, 1024, 768]"; tangents_2: "f32[1, 12, 1024, 64]"; tangents_3: "f32[1, 12, 1024, 64]"; tangents_4: "f32[1, 12, 1024, 64]"; tangents_5: "f32[1, 12, 1024, 64]"; tangents_6: "f32[1, 12, 1024, 64]"; tangents_7: "f32[1, 12, 1024, 64]"; tangents_8: "f32[1, 12, 1024, 64]"; tangents_9: "f32[1, 12, 1024, 64]"; tangents_10: "f32[1, 12, 1024, 64]"; tangents_11: "f32[1, 12, 1024, 64]"; tangents_12: "f32[1, 12, 1024, 64]"; tangents_13: "f32[1, 12, 1024, 64]"; tangents_14: "f32[1, 12, 1024, 64]"; tangents_15: "f32[1, 12, 1024, 64]"; tangents_16: "f32[1, 12, 1024, 64]"; tangents_17: "f32[1, 12, 1024, 64]"; tangents_18: "f32[1, 12, 1024, 64]"; tangents_19: "f32[1, 12, 1024, 64]"; tangents_20: "f32[1, 12, 1024, 64]"; tangents_21: "f32[1, 12, 1024, 64]"; tangents_22: "f32[1, 12, 1024, 64]"; tangents_23: "f32[1, 12, 1024, 64]"; tangents_24: "f32[1, 12, 1024, 64]"; tangents_25: "f32[1, 12, 1024, 64]"; tangents_26: "f32[1, 2]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:781, code: input_ids = input_ids.view(-1, input_shape[-1])
    view: "i64[1, 1024]" = torch.ops.aten.view.default(primals_162, [-1, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:802, code: position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
    iota: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:803, code: position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
    unsqueeze: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
    view_1: "i64[1, 1024]" = torch.ops.aten.view.default(unsqueeze, [-1, 1024]);  unsqueeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:843, code: inputs_embeds = self.wte(input_ids)
    embedding: "f32[1, 1024, 768]" = torch.ops.aten.embedding.default(primals_97, view);  primals_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:844, code: position_embeds = self.wpe(position_ids)
    embedding_1: "f32[1, 1024, 768]" = torch.ops.aten.embedding.default(primals_98, view_1);  primals_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:845, code: hidden_states = inputs_embeds + position_embeds
    add: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:851, code: hidden_states = self.drop(hidden_states)
    native_dropout = torch.ops.aten.native_dropout.default(add, 0.1, True);  add = None
    getitem: "f32[1, 1024, 768]" = native_dropout[0]
    getitem_1: "b8[1, 1024, 768]" = native_dropout[1];  native_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean = torch.ops.aten.var_mean.correction(getitem, [2], correction = 0, keepdim = True)
    getitem_2: "f32[1, 1024, 1]" = var_mean[0]
    getitem_3: "f32[1, 1024, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
    rsqrt: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(getitem, getitem_3)
    mul: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    mul_1: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul, primals_99);  mul = None
    add_2: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_1, primals_100);  mul_1 = primals_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_2: "f32[1024, 768]" = torch.ops.aten.view.default(add_2, [-1, 768]);  add_2 = None
    addmm: "f32[1024, 2304]" = torch.ops.aten.addmm.default(primals_1, view_2, primals_2);  primals_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_3: "f32[1, 1024, 2304]" = torch.ops.aten.view.default(addmm, [1, 1024, 2304]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes = torch.ops.aten.split_with_sizes.default(view_3, [768, 768, 768], 2);  view_3 = None
    getitem_4: "f32[1, 1024, 768]" = split_with_sizes[0]
    getitem_5: "f32[1, 1024, 768]" = split_with_sizes[1]
    getitem_6: "f32[1, 1024, 768]" = split_with_sizes[2];  split_with_sizes = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_4: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_4, [1, 1024, 12, 64]);  getitem_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_5: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_5, [1, 1024, 12, 64]);  getitem_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_1: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_6: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_6, [1, 1024, 12, 64]);  getitem_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_2: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_3: "f32[1, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_1, [0, 1, 3, 2])
    expand: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute, [1, 12, 1024, 64]);  permute = None
    view_7: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand, [12, 1024, 64]);  expand = None
    expand_1: "f32[1, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_3, [1, 12, 64, 1024]);  permute_3 = None
    view_8: "f32[12, 64, 1024]" = torch.ops.aten.view.default(expand_1, [12, 64, 1024]);  expand_1 = None
    bmm: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_7, view_8)
    view_9: "f32[1, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm, [1, 12, 1024, 1024]);  bmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    div: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_9, full);  view_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_1: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(primals_150, 0, 0, 9223372036854775807);  primals_150 = None
    slice_2: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 9223372036854775807);  slice_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_1: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_2, div, full_1);  div = full_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax: "f32[1, 12, 1024, 1]" = torch.ops.aten.amax.default(where, [-1], True)
    sub_1: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where, amax);  where = amax = None
    exp: "f32[1, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
    sum_1: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div_1: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(div_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    native_dropout_1 = torch.ops.aten.native_dropout.default(div_1, 0.1, True);  div_1 = None
    getitem_7: "f32[1, 12, 1024, 1024]" = native_dropout_1[0]
    getitem_8: "b8[1, 12, 1024, 1024]" = native_dropout_1[1];  native_dropout_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_2: "f32[1, 12, 1024, 1024]" = torch.ops.aten.expand.default(getitem_7, [1, 12, 1024, 1024]);  getitem_7 = None
    view_10: "f32[12, 1024, 1024]" = torch.ops.aten.view.default(expand_2, [12, 1024, 1024]);  expand_2 = None
    expand_3: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_2, [1, 12, 1024, 64])
    view_11: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_3, [12, 1024, 64]);  expand_3 = None
    bmm_1: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_10, view_11)
    view_12: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_1, [1, 12, 1024, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_4: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None
    clone: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_13: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone, [1, 1024, 768]);  clone = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_14: "f32[1024, 768]" = torch.ops.aten.view.default(view_13, [-1, 768]);  view_13 = None
    addmm_1: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_3, view_14, primals_4);  primals_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_15: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_1, [1, 1024, 768]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    native_dropout_2 = torch.ops.aten.native_dropout.default(view_15, 0.1, True);  view_15 = None
    getitem_9: "f32[1, 1024, 768]" = native_dropout_2[0]
    getitem_10: "b8[1, 1024, 768]" = native_dropout_2[1];  native_dropout_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_3: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_9, getitem);  getitem_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
    getitem_11: "f32[1, 1024, 1]" = var_mean_1[0]
    getitem_12: "f32[1, 1024, 1]" = var_mean_1[1];  var_mean_1 = None
    add_4: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_11, 1e-05);  getitem_11 = None
    rsqrt_1: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    sub_2: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_3, getitem_12)
    mul_2: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
    mul_3: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2, primals_101);  mul_2 = None
    add_5: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_3, primals_102);  mul_3 = primals_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_16: "f32[1024, 768]" = torch.ops.aten.view.default(add_5, [-1, 768]);  add_5 = None
    addmm_2: "f32[1024, 3072]" = torch.ops.aten.addmm.default(primals_5, view_16, primals_6);  primals_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_17: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_2, [1, 1024, 3072]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_4: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_17, 0.5)
    pow_1: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_17, 3.0)
    mul_5: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
    add_6: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(view_17, mul_5);  mul_5 = None
    mul_6: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_6, 0.7978845608028654);  add_6 = None
    tanh: "f32[1, 1024, 3072]" = torch.ops.aten.tanh.default(mul_6);  mul_6 = None
    alias_1: "f32[1, 1024, 3072]" = torch.ops.aten.alias.default(tanh)
    add_7: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
    mul_7: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_4, add_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_18: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_7, [-1, 3072]);  mul_7 = None
    addmm_3: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_7, view_18, primals_8);  primals_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_19: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_3, [1, 1024, 768]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    native_dropout_3 = torch.ops.aten.native_dropout.default(view_19, 0.1, True);  view_19 = None
    getitem_13: "f32[1, 1024, 768]" = native_dropout_3[0]
    getitem_14: "b8[1, 1024, 768]" = native_dropout_3[1];  native_dropout_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_8: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_3, getitem_13);  getitem_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
    getitem_15: "f32[1, 1024, 1]" = var_mean_2[0]
    getitem_16: "f32[1, 1024, 1]" = var_mean_2[1];  var_mean_2 = None
    add_9: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_15, 1e-05);  getitem_15 = None
    rsqrt_2: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
    sub_3: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_8, getitem_16)
    mul_8: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
    mul_9: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_8, primals_103);  mul_8 = None
    add_10: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_9, primals_104);  mul_9 = primals_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_20: "f32[1024, 768]" = torch.ops.aten.view.default(add_10, [-1, 768]);  add_10 = None
    addmm_4: "f32[1024, 2304]" = torch.ops.aten.addmm.default(primals_9, view_20, primals_10);  primals_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_21: "f32[1, 1024, 2304]" = torch.ops.aten.view.default(addmm_4, [1, 1024, 2304]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(view_21, [768, 768, 768], 2);  view_21 = None
    getitem_17: "f32[1, 1024, 768]" = split_with_sizes_1[0]
    getitem_18: "f32[1, 1024, 768]" = split_with_sizes_1[1]
    getitem_19: "f32[1, 1024, 768]" = split_with_sizes_1[2];  split_with_sizes_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_22: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_17, [1, 1024, 12, 64]);  getitem_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_5: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_22, [0, 2, 1, 3]);  view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_23: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_18, [1, 1024, 12, 64]);  getitem_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_6: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_23, [0, 2, 1, 3]);  view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_24: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_19, [1, 1024, 12, 64]);  getitem_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_7: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_8: "f32[1, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_6, [0, 1, 3, 2])
    expand_4: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_5, [1, 12, 1024, 64]);  permute_5 = None
    view_25: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_4, [12, 1024, 64]);  expand_4 = None
    expand_5: "f32[1, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_8, [1, 12, 64, 1024]);  permute_8 = None
    view_26: "f32[12, 64, 1024]" = torch.ops.aten.view.default(expand_5, [12, 64, 1024]);  expand_5 = None
    bmm_2: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_25, view_26)
    view_27: "f32[1, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_2, [1, 12, 1024, 1024]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_2: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    div_2: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_27, full_2);  view_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_3: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(primals_151, 0, 0, 9223372036854775807);  primals_151 = None
    slice_4: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_3, 1, 0, 9223372036854775807);  slice_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_3: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_1: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_4, div_2, full_3);  div_2 = full_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_1: "f32[1, 12, 1024, 1]" = torch.ops.aten.amax.default(where_1, [-1], True)
    sub_4: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where_1, amax_1);  where_1 = amax_1 = None
    exp_1: "f32[1, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
    sum_2: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_3: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    alias_2: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(div_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    native_dropout_4 = torch.ops.aten.native_dropout.default(div_3, 0.1, True);  div_3 = None
    getitem_20: "f32[1, 12, 1024, 1024]" = native_dropout_4[0]
    getitem_21: "b8[1, 12, 1024, 1024]" = native_dropout_4[1];  native_dropout_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_6: "f32[1, 12, 1024, 1024]" = torch.ops.aten.expand.default(getitem_20, [1, 12, 1024, 1024]);  getitem_20 = None
    view_28: "f32[12, 1024, 1024]" = torch.ops.aten.view.default(expand_6, [12, 1024, 1024]);  expand_6 = None
    expand_7: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_7, [1, 12, 1024, 64])
    view_29: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_7, [12, 1024, 64]);  expand_7 = None
    bmm_3: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_28, view_29)
    view_30: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_3, [1, 12, 1024, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_9: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
    clone_1: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_9, memory_format = torch.contiguous_format);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_31: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_1, [1, 1024, 768]);  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_32: "f32[1024, 768]" = torch.ops.aten.view.default(view_31, [-1, 768]);  view_31 = None
    addmm_5: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_11, view_32, primals_12);  primals_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_33: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_5, [1, 1024, 768]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    native_dropout_5 = torch.ops.aten.native_dropout.default(view_33, 0.1, True);  view_33 = None
    getitem_22: "f32[1, 1024, 768]" = native_dropout_5[0]
    getitem_23: "b8[1, 1024, 768]" = native_dropout_5[1];  native_dropout_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_11: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_22, add_8);  getitem_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_3 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
    getitem_24: "f32[1, 1024, 1]" = var_mean_3[0]
    getitem_25: "f32[1, 1024, 1]" = var_mean_3[1];  var_mean_3 = None
    add_12: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
    rsqrt_3: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
    sub_5: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_11, getitem_25)
    mul_10: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = None
    mul_11: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_10, primals_105);  mul_10 = None
    add_13: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_11, primals_106);  mul_11 = primals_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_34: "f32[1024, 768]" = torch.ops.aten.view.default(add_13, [-1, 768]);  add_13 = None
    addmm_6: "f32[1024, 3072]" = torch.ops.aten.addmm.default(primals_13, view_34, primals_14);  primals_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_35: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_6, [1, 1024, 3072]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_12: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_35, 0.5)
    pow_2: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_35, 3.0)
    mul_13: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
    add_14: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(view_35, mul_13);  mul_13 = None
    mul_14: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_14, 0.7978845608028654);  add_14 = None
    tanh_1: "f32[1, 1024, 3072]" = torch.ops.aten.tanh.default(mul_14);  mul_14 = None
    alias_3: "f32[1, 1024, 3072]" = torch.ops.aten.alias.default(tanh_1)
    add_15: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
    mul_15: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_12, add_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_36: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_15, [-1, 3072]);  mul_15 = None
    addmm_7: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_15, view_36, primals_16);  primals_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_37: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_7, [1, 1024, 768]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    native_dropout_6 = torch.ops.aten.native_dropout.default(view_37, 0.1, True);  view_37 = None
    getitem_26: "f32[1, 1024, 768]" = native_dropout_6[0]
    getitem_27: "b8[1, 1024, 768]" = native_dropout_6[1];  native_dropout_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_16: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_11, getitem_26);  getitem_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_4 = torch.ops.aten.var_mean.correction(add_16, [2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 1024, 1]" = var_mean_4[0]
    getitem_29: "f32[1, 1024, 1]" = var_mean_4[1];  var_mean_4 = None
    add_17: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
    rsqrt_4: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
    sub_6: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_16, getitem_29)
    mul_16: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = None
    mul_17: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_16, primals_107);  mul_16 = None
    add_18: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_17, primals_108);  mul_17 = primals_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_38: "f32[1024, 768]" = torch.ops.aten.view.default(add_18, [-1, 768]);  add_18 = None
    addmm_8: "f32[1024, 2304]" = torch.ops.aten.addmm.default(primals_17, view_38, primals_18);  primals_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_39: "f32[1, 1024, 2304]" = torch.ops.aten.view.default(addmm_8, [1, 1024, 2304]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_2 = torch.ops.aten.split_with_sizes.default(view_39, [768, 768, 768], 2);  view_39 = None
    getitem_30: "f32[1, 1024, 768]" = split_with_sizes_2[0]
    getitem_31: "f32[1, 1024, 768]" = split_with_sizes_2[1]
    getitem_32: "f32[1, 1024, 768]" = split_with_sizes_2[2];  split_with_sizes_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_40: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_30, [1, 1024, 12, 64]);  getitem_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_10: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_41: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_31, [1, 1024, 12, 64]);  getitem_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_11: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_42: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_32, [1, 1024, 12, 64]);  getitem_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_12: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_13: "f32[1, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_11, [0, 1, 3, 2])
    expand_8: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_10, [1, 12, 1024, 64]);  permute_10 = None
    view_43: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_8, [12, 1024, 64]);  expand_8 = None
    expand_9: "f32[1, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_13, [1, 12, 64, 1024]);  permute_13 = None
    view_44: "f32[12, 64, 1024]" = torch.ops.aten.view.default(expand_9, [12, 64, 1024]);  expand_9 = None
    bmm_4: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_43, view_44)
    view_45: "f32[1, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_4, [1, 12, 1024, 1024]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_4: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    div_4: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_45, full_4);  view_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_5: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(primals_152, 0, 0, 9223372036854775807);  primals_152 = None
    slice_6: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_5, 1, 0, 9223372036854775807);  slice_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_5: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_2: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_6, div_4, full_5);  div_4 = full_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_2: "f32[1, 12, 1024, 1]" = torch.ops.aten.amax.default(where_2, [-1], True)
    sub_7: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where_2, amax_2);  where_2 = amax_2 = None
    exp_2: "f32[1, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
    sum_3: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_5: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    alias_4: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(div_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    native_dropout_7 = torch.ops.aten.native_dropout.default(div_5, 0.1, True);  div_5 = None
    getitem_33: "f32[1, 12, 1024, 1024]" = native_dropout_7[0]
    getitem_34: "b8[1, 12, 1024, 1024]" = native_dropout_7[1];  native_dropout_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_10: "f32[1, 12, 1024, 1024]" = torch.ops.aten.expand.default(getitem_33, [1, 12, 1024, 1024]);  getitem_33 = None
    view_46: "f32[12, 1024, 1024]" = torch.ops.aten.view.default(expand_10, [12, 1024, 1024]);  expand_10 = None
    expand_11: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_12, [1, 12, 1024, 64])
    view_47: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_11, [12, 1024, 64]);  expand_11 = None
    bmm_5: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_46, view_47)
    view_48: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_5, [1, 12, 1024, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_14: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
    clone_2: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_49: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_2, [1, 1024, 768]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_50: "f32[1024, 768]" = torch.ops.aten.view.default(view_49, [-1, 768]);  view_49 = None
    addmm_9: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_19, view_50, primals_20);  primals_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_51: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_9, [1, 1024, 768]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    native_dropout_8 = torch.ops.aten.native_dropout.default(view_51, 0.1, True);  view_51 = None
    getitem_35: "f32[1, 1024, 768]" = native_dropout_8[0]
    getitem_36: "b8[1, 1024, 768]" = native_dropout_8[1];  native_dropout_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_19: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_35, add_16);  getitem_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_5 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
    getitem_37: "f32[1, 1024, 1]" = var_mean_5[0]
    getitem_38: "f32[1, 1024, 1]" = var_mean_5[1];  var_mean_5 = None
    add_20: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_37, 1e-05);  getitem_37 = None
    rsqrt_5: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
    sub_8: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_19, getitem_38)
    mul_18: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = None
    mul_19: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_18, primals_109);  mul_18 = None
    add_21: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_19, primals_110);  mul_19 = primals_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_52: "f32[1024, 768]" = torch.ops.aten.view.default(add_21, [-1, 768]);  add_21 = None
    addmm_10: "f32[1024, 3072]" = torch.ops.aten.addmm.default(primals_21, view_52, primals_22);  primals_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_53: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_10, [1, 1024, 3072]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_20: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_53, 0.5)
    pow_3: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_53, 3.0)
    mul_21: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
    add_22: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(view_53, mul_21);  mul_21 = None
    mul_22: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_22, 0.7978845608028654);  add_22 = None
    tanh_2: "f32[1, 1024, 3072]" = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
    alias_5: "f32[1, 1024, 3072]" = torch.ops.aten.alias.default(tanh_2)
    add_23: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
    mul_23: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_20, add_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_54: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_23, [-1, 3072]);  mul_23 = None
    addmm_11: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_23, view_54, primals_24);  primals_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_55: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_11, [1, 1024, 768]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    native_dropout_9 = torch.ops.aten.native_dropout.default(view_55, 0.1, True);  view_55 = None
    getitem_39: "f32[1, 1024, 768]" = native_dropout_9[0]
    getitem_40: "b8[1, 1024, 768]" = native_dropout_9[1];  native_dropout_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_24: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_19, getitem_39);  getitem_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_6 = torch.ops.aten.var_mean.correction(add_24, [2], correction = 0, keepdim = True)
    getitem_41: "f32[1, 1024, 1]" = var_mean_6[0]
    getitem_42: "f32[1, 1024, 1]" = var_mean_6[1];  var_mean_6 = None
    add_25: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_41, 1e-05);  getitem_41 = None
    rsqrt_6: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
    sub_9: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_24, getitem_42)
    mul_24: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = None
    mul_25: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_24, primals_111);  mul_24 = None
    add_26: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_25, primals_112);  mul_25 = primals_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_56: "f32[1024, 768]" = torch.ops.aten.view.default(add_26, [-1, 768]);  add_26 = None
    addmm_12: "f32[1024, 2304]" = torch.ops.aten.addmm.default(primals_25, view_56, primals_26);  primals_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_57: "f32[1, 1024, 2304]" = torch.ops.aten.view.default(addmm_12, [1, 1024, 2304]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_3 = torch.ops.aten.split_with_sizes.default(view_57, [768, 768, 768], 2);  view_57 = None
    getitem_43: "f32[1, 1024, 768]" = split_with_sizes_3[0]
    getitem_44: "f32[1, 1024, 768]" = split_with_sizes_3[1]
    getitem_45: "f32[1, 1024, 768]" = split_with_sizes_3[2];  split_with_sizes_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_58: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_43, [1, 1024, 12, 64]);  getitem_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_15: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_59: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_44, [1, 1024, 12, 64]);  getitem_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_16: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_59, [0, 2, 1, 3]);  view_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_60: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_45, [1, 1024, 12, 64]);  getitem_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_17: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_60, [0, 2, 1, 3]);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_18: "f32[1, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_16, [0, 1, 3, 2])
    expand_12: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_15, [1, 12, 1024, 64]);  permute_15 = None
    view_61: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_12, [12, 1024, 64]);  expand_12 = None
    expand_13: "f32[1, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_18, [1, 12, 64, 1024]);  permute_18 = None
    view_62: "f32[12, 64, 1024]" = torch.ops.aten.view.default(expand_13, [12, 64, 1024]);  expand_13 = None
    bmm_6: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_61, view_62)
    view_63: "f32[1, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_6, [1, 12, 1024, 1024]);  bmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_6: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    div_6: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_63, full_6);  view_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_7: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(primals_153, 0, 0, 9223372036854775807);  primals_153 = None
    slice_8: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_7, 1, 0, 9223372036854775807);  slice_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_7: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_3: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_8, div_6, full_7);  div_6 = full_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_3: "f32[1, 12, 1024, 1]" = torch.ops.aten.amax.default(where_3, [-1], True)
    sub_10: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where_3, amax_3);  where_3 = amax_3 = None
    exp_3: "f32[1, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
    sum_4: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_7: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    alias_6: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(div_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    native_dropout_10 = torch.ops.aten.native_dropout.default(div_7, 0.1, True);  div_7 = None
    getitem_46: "f32[1, 12, 1024, 1024]" = native_dropout_10[0]
    getitem_47: "b8[1, 12, 1024, 1024]" = native_dropout_10[1];  native_dropout_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_14: "f32[1, 12, 1024, 1024]" = torch.ops.aten.expand.default(getitem_46, [1, 12, 1024, 1024]);  getitem_46 = None
    view_64: "f32[12, 1024, 1024]" = torch.ops.aten.view.default(expand_14, [12, 1024, 1024]);  expand_14 = None
    expand_15: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_17, [1, 12, 1024, 64])
    view_65: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_15, [12, 1024, 64]);  expand_15 = None
    bmm_7: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_64, view_65)
    view_66: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_7, [1, 12, 1024, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_19: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None
    clone_3: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_67: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_3, [1, 1024, 768]);  clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_68: "f32[1024, 768]" = torch.ops.aten.view.default(view_67, [-1, 768]);  view_67 = None
    addmm_13: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_27, view_68, primals_28);  primals_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_69: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_13, [1, 1024, 768]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    native_dropout_11 = torch.ops.aten.native_dropout.default(view_69, 0.1, True);  view_69 = None
    getitem_48: "f32[1, 1024, 768]" = native_dropout_11[0]
    getitem_49: "b8[1, 1024, 768]" = native_dropout_11[1];  native_dropout_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_27: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_48, add_24);  getitem_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_7 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
    getitem_50: "f32[1, 1024, 1]" = var_mean_7[0]
    getitem_51: "f32[1, 1024, 1]" = var_mean_7[1];  var_mean_7 = None
    add_28: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
    rsqrt_7: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    sub_11: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_27, getitem_51)
    mul_26: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = None
    mul_27: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_26, primals_113);  mul_26 = None
    add_29: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_27, primals_114);  mul_27 = primals_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_70: "f32[1024, 768]" = torch.ops.aten.view.default(add_29, [-1, 768]);  add_29 = None
    addmm_14: "f32[1024, 3072]" = torch.ops.aten.addmm.default(primals_29, view_70, primals_30);  primals_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_71: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_14, [1, 1024, 3072]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_28: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_71, 0.5)
    pow_4: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_71, 3.0)
    mul_29: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
    add_30: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(view_71, mul_29);  mul_29 = None
    mul_30: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_30, 0.7978845608028654);  add_30 = None
    tanh_3: "f32[1, 1024, 3072]" = torch.ops.aten.tanh.default(mul_30);  mul_30 = None
    alias_7: "f32[1, 1024, 3072]" = torch.ops.aten.alias.default(tanh_3)
    add_31: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
    mul_31: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_28, add_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_72: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_31, [-1, 3072]);  mul_31 = None
    addmm_15: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_31, view_72, primals_32);  primals_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_73: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_15, [1, 1024, 768]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    native_dropout_12 = torch.ops.aten.native_dropout.default(view_73, 0.1, True);  view_73 = None
    getitem_52: "f32[1, 1024, 768]" = native_dropout_12[0]
    getitem_53: "b8[1, 1024, 768]" = native_dropout_12[1];  native_dropout_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_32: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_27, getitem_52);  getitem_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_8 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
    getitem_54: "f32[1, 1024, 1]" = var_mean_8[0]
    getitem_55: "f32[1, 1024, 1]" = var_mean_8[1];  var_mean_8 = None
    add_33: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
    rsqrt_8: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
    sub_12: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_32, getitem_55)
    mul_32: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = None
    mul_33: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_32, primals_115);  mul_32 = None
    add_34: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_33, primals_116);  mul_33 = primals_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_74: "f32[1024, 768]" = torch.ops.aten.view.default(add_34, [-1, 768]);  add_34 = None
    addmm_16: "f32[1024, 2304]" = torch.ops.aten.addmm.default(primals_33, view_74, primals_34);  primals_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_75: "f32[1, 1024, 2304]" = torch.ops.aten.view.default(addmm_16, [1, 1024, 2304]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_4 = torch.ops.aten.split_with_sizes.default(view_75, [768, 768, 768], 2);  view_75 = None
    getitem_56: "f32[1, 1024, 768]" = split_with_sizes_4[0]
    getitem_57: "f32[1, 1024, 768]" = split_with_sizes_4[1]
    getitem_58: "f32[1, 1024, 768]" = split_with_sizes_4[2];  split_with_sizes_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_76: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_56, [1, 1024, 12, 64]);  getitem_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_20: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_77: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_57, [1, 1024, 12, 64]);  getitem_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_21: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_77, [0, 2, 1, 3]);  view_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_78: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_58, [1, 1024, 12, 64]);  getitem_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_22: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_23: "f32[1, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_21, [0, 1, 3, 2])
    expand_16: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_20, [1, 12, 1024, 64]);  permute_20 = None
    view_79: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_16, [12, 1024, 64]);  expand_16 = None
    expand_17: "f32[1, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_23, [1, 12, 64, 1024]);  permute_23 = None
    view_80: "f32[12, 64, 1024]" = torch.ops.aten.view.default(expand_17, [12, 64, 1024]);  expand_17 = None
    bmm_8: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_79, view_80)
    view_81: "f32[1, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_8, [1, 12, 1024, 1024]);  bmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_8: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    div_8: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_81, full_8);  view_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_9: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(primals_154, 0, 0, 9223372036854775807);  primals_154 = None
    slice_10: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_9, 1, 0, 9223372036854775807);  slice_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_9: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_4: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_10, div_8, full_9);  div_8 = full_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_4: "f32[1, 12, 1024, 1]" = torch.ops.aten.amax.default(where_4, [-1], True)
    sub_13: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where_4, amax_4);  where_4 = amax_4 = None
    exp_4: "f32[1, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
    sum_5: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_9: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    alias_8: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(div_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    native_dropout_13 = torch.ops.aten.native_dropout.default(div_9, 0.1, True);  div_9 = None
    getitem_59: "f32[1, 12, 1024, 1024]" = native_dropout_13[0]
    getitem_60: "b8[1, 12, 1024, 1024]" = native_dropout_13[1];  native_dropout_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_18: "f32[1, 12, 1024, 1024]" = torch.ops.aten.expand.default(getitem_59, [1, 12, 1024, 1024]);  getitem_59 = None
    view_82: "f32[12, 1024, 1024]" = torch.ops.aten.view.default(expand_18, [12, 1024, 1024]);  expand_18 = None
    expand_19: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_22, [1, 12, 1024, 64])
    view_83: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_19, [12, 1024, 64]);  expand_19 = None
    bmm_9: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_82, view_83)
    view_84: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_9, [1, 12, 1024, 64]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_24: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_84, [0, 2, 1, 3]);  view_84 = None
    clone_4: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_85: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_4, [1, 1024, 768]);  clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_86: "f32[1024, 768]" = torch.ops.aten.view.default(view_85, [-1, 768]);  view_85 = None
    addmm_17: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_35, view_86, primals_36);  primals_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_87: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_17, [1, 1024, 768]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    native_dropout_14 = torch.ops.aten.native_dropout.default(view_87, 0.1, True);  view_87 = None
    getitem_61: "f32[1, 1024, 768]" = native_dropout_14[0]
    getitem_62: "b8[1, 1024, 768]" = native_dropout_14[1];  native_dropout_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_35: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_61, add_32);  getitem_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_9 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
    getitem_63: "f32[1, 1024, 1]" = var_mean_9[0]
    getitem_64: "f32[1, 1024, 1]" = var_mean_9[1];  var_mean_9 = None
    add_36: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_63, 1e-05);  getitem_63 = None
    rsqrt_9: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    sub_14: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_64)
    mul_34: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = None
    mul_35: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_34, primals_117);  mul_34 = None
    add_37: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_35, primals_118);  mul_35 = primals_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_88: "f32[1024, 768]" = torch.ops.aten.view.default(add_37, [-1, 768]);  add_37 = None
    addmm_18: "f32[1024, 3072]" = torch.ops.aten.addmm.default(primals_37, view_88, primals_38);  primals_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_89: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_18, [1, 1024, 3072]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_36: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_89, 0.5)
    pow_5: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_89, 3.0)
    mul_37: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
    add_38: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(view_89, mul_37);  mul_37 = None
    mul_38: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_38, 0.7978845608028654);  add_38 = None
    tanh_4: "f32[1, 1024, 3072]" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
    alias_9: "f32[1, 1024, 3072]" = torch.ops.aten.alias.default(tanh_4)
    add_39: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
    mul_39: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_36, add_39)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_90: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_39, [-1, 3072]);  mul_39 = None
    addmm_19: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_39, view_90, primals_40);  primals_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_91: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_19, [1, 1024, 768]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    native_dropout_15 = torch.ops.aten.native_dropout.default(view_91, 0.1, True);  view_91 = None
    getitem_65: "f32[1, 1024, 768]" = native_dropout_15[0]
    getitem_66: "b8[1, 1024, 768]" = native_dropout_15[1];  native_dropout_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_40: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_35, getitem_65);  getitem_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_10 = torch.ops.aten.var_mean.correction(add_40, [2], correction = 0, keepdim = True)
    getitem_67: "f32[1, 1024, 1]" = var_mean_10[0]
    getitem_68: "f32[1, 1024, 1]" = var_mean_10[1];  var_mean_10 = None
    add_41: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_67, 1e-05);  getitem_67 = None
    rsqrt_10: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
    sub_15: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_40, getitem_68)
    mul_40: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = None
    mul_41: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_40, primals_119);  mul_40 = None
    add_42: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_41, primals_120);  mul_41 = primals_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_92: "f32[1024, 768]" = torch.ops.aten.view.default(add_42, [-1, 768]);  add_42 = None
    addmm_20: "f32[1024, 2304]" = torch.ops.aten.addmm.default(primals_41, view_92, primals_42);  primals_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_93: "f32[1, 1024, 2304]" = torch.ops.aten.view.default(addmm_20, [1, 1024, 2304]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_5 = torch.ops.aten.split_with_sizes.default(view_93, [768, 768, 768], 2);  view_93 = None
    getitem_69: "f32[1, 1024, 768]" = split_with_sizes_5[0]
    getitem_70: "f32[1, 1024, 768]" = split_with_sizes_5[1]
    getitem_71: "f32[1, 1024, 768]" = split_with_sizes_5[2];  split_with_sizes_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_94: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_69, [1, 1024, 12, 64]);  getitem_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_25: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_95: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_70, [1, 1024, 12, 64]);  getitem_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_26: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_96: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_71, [1, 1024, 12, 64]);  getitem_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_27: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_28: "f32[1, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_26, [0, 1, 3, 2])
    expand_20: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_25, [1, 12, 1024, 64]);  permute_25 = None
    view_97: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_20, [12, 1024, 64]);  expand_20 = None
    expand_21: "f32[1, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_28, [1, 12, 64, 1024]);  permute_28 = None
    view_98: "f32[12, 64, 1024]" = torch.ops.aten.view.default(expand_21, [12, 64, 1024]);  expand_21 = None
    bmm_10: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_97, view_98)
    view_99: "f32[1, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_10, [1, 12, 1024, 1024]);  bmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_10: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    div_10: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_99, full_10);  view_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_11: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(primals_155, 0, 0, 9223372036854775807);  primals_155 = None
    slice_12: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_11, 1, 0, 9223372036854775807);  slice_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_11: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_5: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_12, div_10, full_11);  div_10 = full_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_5: "f32[1, 12, 1024, 1]" = torch.ops.aten.amax.default(where_5, [-1], True)
    sub_16: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where_5, amax_5);  where_5 = amax_5 = None
    exp_5: "f32[1, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
    sum_6: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_11: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    alias_10: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(div_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    native_dropout_16 = torch.ops.aten.native_dropout.default(div_11, 0.1, True);  div_11 = None
    getitem_72: "f32[1, 12, 1024, 1024]" = native_dropout_16[0]
    getitem_73: "b8[1, 12, 1024, 1024]" = native_dropout_16[1];  native_dropout_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_22: "f32[1, 12, 1024, 1024]" = torch.ops.aten.expand.default(getitem_72, [1, 12, 1024, 1024]);  getitem_72 = None
    view_100: "f32[12, 1024, 1024]" = torch.ops.aten.view.default(expand_22, [12, 1024, 1024]);  expand_22 = None
    expand_23: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_27, [1, 12, 1024, 64])
    view_101: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_23, [12, 1024, 64]);  expand_23 = None
    bmm_11: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_100, view_101)
    view_102: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_11, [1, 12, 1024, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_29: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
    clone_5: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_103: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_5, [1, 1024, 768]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_104: "f32[1024, 768]" = torch.ops.aten.view.default(view_103, [-1, 768]);  view_103 = None
    addmm_21: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_43, view_104, primals_44);  primals_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_105: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_21, [1, 1024, 768]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    native_dropout_17 = torch.ops.aten.native_dropout.default(view_105, 0.1, True);  view_105 = None
    getitem_74: "f32[1, 1024, 768]" = native_dropout_17[0]
    getitem_75: "b8[1, 1024, 768]" = native_dropout_17[1];  native_dropout_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_43: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_74, add_40);  getitem_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_11 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
    getitem_76: "f32[1, 1024, 1]" = var_mean_11[0]
    getitem_77: "f32[1, 1024, 1]" = var_mean_11[1];  var_mean_11 = None
    add_44: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
    rsqrt_11: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
    sub_17: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_43, getitem_77)
    mul_42: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = None
    mul_43: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_42, primals_121);  mul_42 = None
    add_45: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_43, primals_122);  mul_43 = primals_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_106: "f32[1024, 768]" = torch.ops.aten.view.default(add_45, [-1, 768]);  add_45 = None
    addmm_22: "f32[1024, 3072]" = torch.ops.aten.addmm.default(primals_45, view_106, primals_46);  primals_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_107: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_22, [1, 1024, 3072]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_44: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.5)
    pow_6: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_107, 3.0)
    mul_45: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
    add_46: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(view_107, mul_45);  mul_45 = None
    mul_46: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_46, 0.7978845608028654);  add_46 = None
    tanh_5: "f32[1, 1024, 3072]" = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
    alias_11: "f32[1, 1024, 3072]" = torch.ops.aten.alias.default(tanh_5)
    add_47: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
    mul_47: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_44, add_47)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_108: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_47, [-1, 3072]);  mul_47 = None
    addmm_23: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_47, view_108, primals_48);  primals_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_109: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_23, [1, 1024, 768]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    native_dropout_18 = torch.ops.aten.native_dropout.default(view_109, 0.1, True);  view_109 = None
    getitem_78: "f32[1, 1024, 768]" = native_dropout_18[0]
    getitem_79: "b8[1, 1024, 768]" = native_dropout_18[1];  native_dropout_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_48: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_43, getitem_78);  getitem_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_12 = torch.ops.aten.var_mean.correction(add_48, [2], correction = 0, keepdim = True)
    getitem_80: "f32[1, 1024, 1]" = var_mean_12[0]
    getitem_81: "f32[1, 1024, 1]" = var_mean_12[1];  var_mean_12 = None
    add_49: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
    rsqrt_12: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    sub_18: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_48, getitem_81)
    mul_48: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = None
    mul_49: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_48, primals_123);  mul_48 = None
    add_50: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_49, primals_124);  mul_49 = primals_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_110: "f32[1024, 768]" = torch.ops.aten.view.default(add_50, [-1, 768]);  add_50 = None
    addmm_24: "f32[1024, 2304]" = torch.ops.aten.addmm.default(primals_49, view_110, primals_50);  primals_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_111: "f32[1, 1024, 2304]" = torch.ops.aten.view.default(addmm_24, [1, 1024, 2304]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_6 = torch.ops.aten.split_with_sizes.default(view_111, [768, 768, 768], 2);  view_111 = None
    getitem_82: "f32[1, 1024, 768]" = split_with_sizes_6[0]
    getitem_83: "f32[1, 1024, 768]" = split_with_sizes_6[1]
    getitem_84: "f32[1, 1024, 768]" = split_with_sizes_6[2];  split_with_sizes_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_112: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_82, [1, 1024, 12, 64]);  getitem_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_30: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_113: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_83, [1, 1024, 12, 64]);  getitem_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_31: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_113, [0, 2, 1, 3]);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_114: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_84, [1, 1024, 12, 64]);  getitem_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_32: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_33: "f32[1, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_31, [0, 1, 3, 2])
    expand_24: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_30, [1, 12, 1024, 64]);  permute_30 = None
    view_115: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_24, [12, 1024, 64]);  expand_24 = None
    expand_25: "f32[1, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_33, [1, 12, 64, 1024]);  permute_33 = None
    view_116: "f32[12, 64, 1024]" = torch.ops.aten.view.default(expand_25, [12, 64, 1024]);  expand_25 = None
    bmm_12: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_115, view_116)
    view_117: "f32[1, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_12, [1, 12, 1024, 1024]);  bmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_12: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    div_12: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_117, full_12);  view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_13: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(primals_156, 0, 0, 9223372036854775807);  primals_156 = None
    slice_14: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_13, 1, 0, 9223372036854775807);  slice_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_13: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_6: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_14, div_12, full_13);  div_12 = full_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_6: "f32[1, 12, 1024, 1]" = torch.ops.aten.amax.default(where_6, [-1], True)
    sub_19: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where_6, amax_6);  where_6 = amax_6 = None
    exp_6: "f32[1, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
    sum_7: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_13: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    alias_12: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(div_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    native_dropout_19 = torch.ops.aten.native_dropout.default(div_13, 0.1, True);  div_13 = None
    getitem_85: "f32[1, 12, 1024, 1024]" = native_dropout_19[0]
    getitem_86: "b8[1, 12, 1024, 1024]" = native_dropout_19[1];  native_dropout_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_26: "f32[1, 12, 1024, 1024]" = torch.ops.aten.expand.default(getitem_85, [1, 12, 1024, 1024]);  getitem_85 = None
    view_118: "f32[12, 1024, 1024]" = torch.ops.aten.view.default(expand_26, [12, 1024, 1024]);  expand_26 = None
    expand_27: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_32, [1, 12, 1024, 64])
    view_119: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_27, [12, 1024, 64]);  expand_27 = None
    bmm_13: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_118, view_119)
    view_120: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_13, [1, 12, 1024, 64]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_34: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
    clone_6: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format);  permute_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_121: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_6, [1, 1024, 768]);  clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_122: "f32[1024, 768]" = torch.ops.aten.view.default(view_121, [-1, 768]);  view_121 = None
    addmm_25: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_51, view_122, primals_52);  primals_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_123: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_25, [1, 1024, 768]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    native_dropout_20 = torch.ops.aten.native_dropout.default(view_123, 0.1, True);  view_123 = None
    getitem_87: "f32[1, 1024, 768]" = native_dropout_20[0]
    getitem_88: "b8[1, 1024, 768]" = native_dropout_20[1];  native_dropout_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_51: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_87, add_48);  getitem_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_13 = torch.ops.aten.var_mean.correction(add_51, [2], correction = 0, keepdim = True)
    getitem_89: "f32[1, 1024, 1]" = var_mean_13[0]
    getitem_90: "f32[1, 1024, 1]" = var_mean_13[1];  var_mean_13 = None
    add_52: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_89, 1e-05);  getitem_89 = None
    rsqrt_13: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    sub_20: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_51, getitem_90)
    mul_50: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = None
    mul_51: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_50, primals_125);  mul_50 = None
    add_53: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_51, primals_126);  mul_51 = primals_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_124: "f32[1024, 768]" = torch.ops.aten.view.default(add_53, [-1, 768]);  add_53 = None
    addmm_26: "f32[1024, 3072]" = torch.ops.aten.addmm.default(primals_53, view_124, primals_54);  primals_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_125: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_26, [1, 1024, 3072]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_52: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_125, 0.5)
    pow_7: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_125, 3.0)
    mul_53: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
    add_54: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(view_125, mul_53);  mul_53 = None
    mul_54: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_54, 0.7978845608028654);  add_54 = None
    tanh_6: "f32[1, 1024, 3072]" = torch.ops.aten.tanh.default(mul_54);  mul_54 = None
    alias_13: "f32[1, 1024, 3072]" = torch.ops.aten.alias.default(tanh_6)
    add_55: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
    mul_55: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_52, add_55)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_126: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_55, [-1, 3072]);  mul_55 = None
    addmm_27: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_55, view_126, primals_56);  primals_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_127: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_27, [1, 1024, 768]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    native_dropout_21 = torch.ops.aten.native_dropout.default(view_127, 0.1, True);  view_127 = None
    getitem_91: "f32[1, 1024, 768]" = native_dropout_21[0]
    getitem_92: "b8[1, 1024, 768]" = native_dropout_21[1];  native_dropout_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_56: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_51, getitem_91);  getitem_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_14 = torch.ops.aten.var_mean.correction(add_56, [2], correction = 0, keepdim = True)
    getitem_93: "f32[1, 1024, 1]" = var_mean_14[0]
    getitem_94: "f32[1, 1024, 1]" = var_mean_14[1];  var_mean_14 = None
    add_57: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_93, 1e-05);  getitem_93 = None
    rsqrt_14: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
    sub_21: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_56, getitem_94)
    mul_56: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = None
    mul_57: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_56, primals_127);  mul_56 = None
    add_58: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_57, primals_128);  mul_57 = primals_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_128: "f32[1024, 768]" = torch.ops.aten.view.default(add_58, [-1, 768]);  add_58 = None
    addmm_28: "f32[1024, 2304]" = torch.ops.aten.addmm.default(primals_57, view_128, primals_58);  primals_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_129: "f32[1, 1024, 2304]" = torch.ops.aten.view.default(addmm_28, [1, 1024, 2304]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_7 = torch.ops.aten.split_with_sizes.default(view_129, [768, 768, 768], 2);  view_129 = None
    getitem_95: "f32[1, 1024, 768]" = split_with_sizes_7[0]
    getitem_96: "f32[1, 1024, 768]" = split_with_sizes_7[1]
    getitem_97: "f32[1, 1024, 768]" = split_with_sizes_7[2];  split_with_sizes_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_130: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_95, [1, 1024, 12, 64]);  getitem_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_35: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_131: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_96, [1, 1024, 12, 64]);  getitem_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_36: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_131, [0, 2, 1, 3]);  view_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_132: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_97, [1, 1024, 12, 64]);  getitem_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_37: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_132, [0, 2, 1, 3]);  view_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_38: "f32[1, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_36, [0, 1, 3, 2])
    expand_28: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_35, [1, 12, 1024, 64]);  permute_35 = None
    view_133: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_28, [12, 1024, 64]);  expand_28 = None
    expand_29: "f32[1, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_38, [1, 12, 64, 1024]);  permute_38 = None
    view_134: "f32[12, 64, 1024]" = torch.ops.aten.view.default(expand_29, [12, 64, 1024]);  expand_29 = None
    bmm_14: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_133, view_134)
    view_135: "f32[1, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_14, [1, 12, 1024, 1024]);  bmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_14: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    div_14: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_135, full_14);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_15: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(primals_157, 0, 0, 9223372036854775807);  primals_157 = None
    slice_16: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_15, 1, 0, 9223372036854775807);  slice_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_15: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_7: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_16, div_14, full_15);  div_14 = full_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_7: "f32[1, 12, 1024, 1]" = torch.ops.aten.amax.default(where_7, [-1], True)
    sub_22: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where_7, amax_7);  where_7 = amax_7 = None
    exp_7: "f32[1, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
    sum_8: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_15: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    alias_14: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(div_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    native_dropout_22 = torch.ops.aten.native_dropout.default(div_15, 0.1, True);  div_15 = None
    getitem_98: "f32[1, 12, 1024, 1024]" = native_dropout_22[0]
    getitem_99: "b8[1, 12, 1024, 1024]" = native_dropout_22[1];  native_dropout_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_30: "f32[1, 12, 1024, 1024]" = torch.ops.aten.expand.default(getitem_98, [1, 12, 1024, 1024]);  getitem_98 = None
    view_136: "f32[12, 1024, 1024]" = torch.ops.aten.view.default(expand_30, [12, 1024, 1024]);  expand_30 = None
    expand_31: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_37, [1, 12, 1024, 64])
    view_137: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_31, [12, 1024, 64]);  expand_31 = None
    bmm_15: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_136, view_137)
    view_138: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_15, [1, 12, 1024, 64]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_39: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None
    clone_7: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_39, memory_format = torch.contiguous_format);  permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_139: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_7, [1, 1024, 768]);  clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_140: "f32[1024, 768]" = torch.ops.aten.view.default(view_139, [-1, 768]);  view_139 = None
    addmm_29: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_59, view_140, primals_60);  primals_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_141: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_29, [1, 1024, 768]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    native_dropout_23 = torch.ops.aten.native_dropout.default(view_141, 0.1, True);  view_141 = None
    getitem_100: "f32[1, 1024, 768]" = native_dropout_23[0]
    getitem_101: "b8[1, 1024, 768]" = native_dropout_23[1];  native_dropout_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_59: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_100, add_56);  getitem_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_15 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
    getitem_102: "f32[1, 1024, 1]" = var_mean_15[0]
    getitem_103: "f32[1, 1024, 1]" = var_mean_15[1];  var_mean_15 = None
    add_60: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-05);  getitem_102 = None
    rsqrt_15: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    sub_23: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_59, getitem_103)
    mul_58: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = None
    mul_59: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_58, primals_129);  mul_58 = None
    add_61: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_59, primals_130);  mul_59 = primals_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_142: "f32[1024, 768]" = torch.ops.aten.view.default(add_61, [-1, 768]);  add_61 = None
    addmm_30: "f32[1024, 3072]" = torch.ops.aten.addmm.default(primals_61, view_142, primals_62);  primals_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_143: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_30, [1, 1024, 3072]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_60: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_143, 0.5)
    pow_8: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_143, 3.0)
    mul_61: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
    add_62: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(view_143, mul_61);  mul_61 = None
    mul_62: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_62, 0.7978845608028654);  add_62 = None
    tanh_7: "f32[1, 1024, 3072]" = torch.ops.aten.tanh.default(mul_62);  mul_62 = None
    alias_15: "f32[1, 1024, 3072]" = torch.ops.aten.alias.default(tanh_7)
    add_63: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
    mul_63: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_60, add_63)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_144: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_63, [-1, 3072]);  mul_63 = None
    addmm_31: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_63, view_144, primals_64);  primals_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_145: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_31, [1, 1024, 768]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    native_dropout_24 = torch.ops.aten.native_dropout.default(view_145, 0.1, True);  view_145 = None
    getitem_104: "f32[1, 1024, 768]" = native_dropout_24[0]
    getitem_105: "b8[1, 1024, 768]" = native_dropout_24[1];  native_dropout_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_64: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_59, getitem_104);  getitem_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_16 = torch.ops.aten.var_mean.correction(add_64, [2], correction = 0, keepdim = True)
    getitem_106: "f32[1, 1024, 1]" = var_mean_16[0]
    getitem_107: "f32[1, 1024, 1]" = var_mean_16[1];  var_mean_16 = None
    add_65: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-05);  getitem_106 = None
    rsqrt_16: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
    sub_24: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_64, getitem_107)
    mul_64: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = None
    mul_65: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_64, primals_131);  mul_64 = None
    add_66: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_65, primals_132);  mul_65 = primals_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_146: "f32[1024, 768]" = torch.ops.aten.view.default(add_66, [-1, 768]);  add_66 = None
    addmm_32: "f32[1024, 2304]" = torch.ops.aten.addmm.default(primals_65, view_146, primals_66);  primals_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_147: "f32[1, 1024, 2304]" = torch.ops.aten.view.default(addmm_32, [1, 1024, 2304]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_8 = torch.ops.aten.split_with_sizes.default(view_147, [768, 768, 768], 2);  view_147 = None
    getitem_108: "f32[1, 1024, 768]" = split_with_sizes_8[0]
    getitem_109: "f32[1, 1024, 768]" = split_with_sizes_8[1]
    getitem_110: "f32[1, 1024, 768]" = split_with_sizes_8[2];  split_with_sizes_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_148: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_108, [1, 1024, 12, 64]);  getitem_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_40: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_149: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_109, [1, 1024, 12, 64]);  getitem_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_41: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_149, [0, 2, 1, 3]);  view_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_150: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_110, [1, 1024, 12, 64]);  getitem_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_42: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_150, [0, 2, 1, 3]);  view_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_43: "f32[1, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_41, [0, 1, 3, 2])
    expand_32: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_40, [1, 12, 1024, 64]);  permute_40 = None
    view_151: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_32, [12, 1024, 64]);  expand_32 = None
    expand_33: "f32[1, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_43, [1, 12, 64, 1024]);  permute_43 = None
    view_152: "f32[12, 64, 1024]" = torch.ops.aten.view.default(expand_33, [12, 64, 1024]);  expand_33 = None
    bmm_16: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_151, view_152)
    view_153: "f32[1, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_16, [1, 12, 1024, 1024]);  bmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_16: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    div_16: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_153, full_16);  view_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_17: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(primals_158, 0, 0, 9223372036854775807);  primals_158 = None
    slice_18: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_17, 1, 0, 9223372036854775807);  slice_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_17: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_8: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_18, div_16, full_17);  div_16 = full_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_8: "f32[1, 12, 1024, 1]" = torch.ops.aten.amax.default(where_8, [-1], True)
    sub_25: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where_8, amax_8);  where_8 = amax_8 = None
    exp_8: "f32[1, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
    sum_9: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_17: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    alias_16: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(div_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    native_dropout_25 = torch.ops.aten.native_dropout.default(div_17, 0.1, True);  div_17 = None
    getitem_111: "f32[1, 12, 1024, 1024]" = native_dropout_25[0]
    getitem_112: "b8[1, 12, 1024, 1024]" = native_dropout_25[1];  native_dropout_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_34: "f32[1, 12, 1024, 1024]" = torch.ops.aten.expand.default(getitem_111, [1, 12, 1024, 1024]);  getitem_111 = None
    view_154: "f32[12, 1024, 1024]" = torch.ops.aten.view.default(expand_34, [12, 1024, 1024]);  expand_34 = None
    expand_35: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_42, [1, 12, 1024, 64])
    view_155: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_35, [12, 1024, 64]);  expand_35 = None
    bmm_17: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_154, view_155)
    view_156: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_17, [1, 12, 1024, 64]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_44: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None
    clone_8: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_44, memory_format = torch.contiguous_format);  permute_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_157: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_8, [1, 1024, 768]);  clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_158: "f32[1024, 768]" = torch.ops.aten.view.default(view_157, [-1, 768]);  view_157 = None
    addmm_33: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_67, view_158, primals_68);  primals_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_159: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_33, [1, 1024, 768]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    native_dropout_26 = torch.ops.aten.native_dropout.default(view_159, 0.1, True);  view_159 = None
    getitem_113: "f32[1, 1024, 768]" = native_dropout_26[0]
    getitem_114: "b8[1, 1024, 768]" = native_dropout_26[1];  native_dropout_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_67: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_113, add_64);  getitem_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_17 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
    getitem_115: "f32[1, 1024, 1]" = var_mean_17[0]
    getitem_116: "f32[1, 1024, 1]" = var_mean_17[1];  var_mean_17 = None
    add_68: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_115, 1e-05);  getitem_115 = None
    rsqrt_17: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
    sub_26: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_67, getitem_116)
    mul_66: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = None
    mul_67: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_66, primals_133);  mul_66 = None
    add_69: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_67, primals_134);  mul_67 = primals_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_160: "f32[1024, 768]" = torch.ops.aten.view.default(add_69, [-1, 768]);  add_69 = None
    addmm_34: "f32[1024, 3072]" = torch.ops.aten.addmm.default(primals_69, view_160, primals_70);  primals_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_161: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_34, [1, 1024, 3072]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_68: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_161, 0.5)
    pow_9: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_161, 3.0)
    mul_69: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
    add_70: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(view_161, mul_69);  mul_69 = None
    mul_70: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_70, 0.7978845608028654);  add_70 = None
    tanh_8: "f32[1, 1024, 3072]" = torch.ops.aten.tanh.default(mul_70);  mul_70 = None
    alias_17: "f32[1, 1024, 3072]" = torch.ops.aten.alias.default(tanh_8)
    add_71: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
    mul_71: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_68, add_71)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_162: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_71, [-1, 3072]);  mul_71 = None
    addmm_35: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_71, view_162, primals_72);  primals_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_163: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_35, [1, 1024, 768]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    native_dropout_27 = torch.ops.aten.native_dropout.default(view_163, 0.1, True);  view_163 = None
    getitem_117: "f32[1, 1024, 768]" = native_dropout_27[0]
    getitem_118: "b8[1, 1024, 768]" = native_dropout_27[1];  native_dropout_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_72: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_67, getitem_117);  getitem_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_18 = torch.ops.aten.var_mean.correction(add_72, [2], correction = 0, keepdim = True)
    getitem_119: "f32[1, 1024, 1]" = var_mean_18[0]
    getitem_120: "f32[1, 1024, 1]" = var_mean_18[1];  var_mean_18 = None
    add_73: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_119, 1e-05);  getitem_119 = None
    rsqrt_18: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
    sub_27: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_72, getitem_120)
    mul_72: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = None
    mul_73: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_72, primals_135);  mul_72 = None
    add_74: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_73, primals_136);  mul_73 = primals_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_164: "f32[1024, 768]" = torch.ops.aten.view.default(add_74, [-1, 768]);  add_74 = None
    addmm_36: "f32[1024, 2304]" = torch.ops.aten.addmm.default(primals_73, view_164, primals_74);  primals_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_165: "f32[1, 1024, 2304]" = torch.ops.aten.view.default(addmm_36, [1, 1024, 2304]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_9 = torch.ops.aten.split_with_sizes.default(view_165, [768, 768, 768], 2);  view_165 = None
    getitem_121: "f32[1, 1024, 768]" = split_with_sizes_9[0]
    getitem_122: "f32[1, 1024, 768]" = split_with_sizes_9[1]
    getitem_123: "f32[1, 1024, 768]" = split_with_sizes_9[2];  split_with_sizes_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_166: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_121, [1, 1024, 12, 64]);  getitem_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_45: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_166, [0, 2, 1, 3]);  view_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_167: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_122, [1, 1024, 12, 64]);  getitem_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_46: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_167, [0, 2, 1, 3]);  view_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_168: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_123, [1, 1024, 12, 64]);  getitem_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_47: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_48: "f32[1, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_46, [0, 1, 3, 2])
    expand_36: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_45, [1, 12, 1024, 64]);  permute_45 = None
    view_169: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_36, [12, 1024, 64]);  expand_36 = None
    expand_37: "f32[1, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_48, [1, 12, 64, 1024]);  permute_48 = None
    view_170: "f32[12, 64, 1024]" = torch.ops.aten.view.default(expand_37, [12, 64, 1024]);  expand_37 = None
    bmm_18: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_169, view_170)
    view_171: "f32[1, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_18, [1, 12, 1024, 1024]);  bmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_18: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    div_18: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_171, full_18);  view_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_19: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(primals_159, 0, 0, 9223372036854775807);  primals_159 = None
    slice_20: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_19, 1, 0, 9223372036854775807);  slice_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_19: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_9: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_20, div_18, full_19);  div_18 = full_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_9: "f32[1, 12, 1024, 1]" = torch.ops.aten.amax.default(where_9, [-1], True)
    sub_28: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where_9, amax_9);  where_9 = amax_9 = None
    exp_9: "f32[1, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
    sum_10: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_19: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    alias_18: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(div_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    native_dropout_28 = torch.ops.aten.native_dropout.default(div_19, 0.1, True);  div_19 = None
    getitem_124: "f32[1, 12, 1024, 1024]" = native_dropout_28[0]
    getitem_125: "b8[1, 12, 1024, 1024]" = native_dropout_28[1];  native_dropout_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_38: "f32[1, 12, 1024, 1024]" = torch.ops.aten.expand.default(getitem_124, [1, 12, 1024, 1024]);  getitem_124 = None
    view_172: "f32[12, 1024, 1024]" = torch.ops.aten.view.default(expand_38, [12, 1024, 1024]);  expand_38 = None
    expand_39: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_47, [1, 12, 1024, 64])
    view_173: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_39, [12, 1024, 64]);  expand_39 = None
    bmm_19: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_172, view_173)
    view_174: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_19, [1, 12, 1024, 64]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_49: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_174, [0, 2, 1, 3]);  view_174 = None
    clone_9: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_175: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_9, [1, 1024, 768]);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_176: "f32[1024, 768]" = torch.ops.aten.view.default(view_175, [-1, 768]);  view_175 = None
    addmm_37: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_75, view_176, primals_76);  primals_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_177: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_37, [1, 1024, 768]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    native_dropout_29 = torch.ops.aten.native_dropout.default(view_177, 0.1, True);  view_177 = None
    getitem_126: "f32[1, 1024, 768]" = native_dropout_29[0]
    getitem_127: "b8[1, 1024, 768]" = native_dropout_29[1];  native_dropout_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_75: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_126, add_72);  getitem_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_19 = torch.ops.aten.var_mean.correction(add_75, [2], correction = 0, keepdim = True)
    getitem_128: "f32[1, 1024, 1]" = var_mean_19[0]
    getitem_129: "f32[1, 1024, 1]" = var_mean_19[1];  var_mean_19 = None
    add_76: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-05);  getitem_128 = None
    rsqrt_19: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
    sub_29: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_75, getitem_129)
    mul_74: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = None
    mul_75: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_74, primals_137);  mul_74 = None
    add_77: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_75, primals_138);  mul_75 = primals_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_178: "f32[1024, 768]" = torch.ops.aten.view.default(add_77, [-1, 768]);  add_77 = None
    addmm_38: "f32[1024, 3072]" = torch.ops.aten.addmm.default(primals_77, view_178, primals_78);  primals_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_179: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_38, [1, 1024, 3072]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_76: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_179, 0.5)
    pow_10: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_179, 3.0)
    mul_77: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
    add_78: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(view_179, mul_77);  mul_77 = None
    mul_78: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_78, 0.7978845608028654);  add_78 = None
    tanh_9: "f32[1, 1024, 3072]" = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
    alias_19: "f32[1, 1024, 3072]" = torch.ops.aten.alias.default(tanh_9)
    add_79: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
    mul_79: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_76, add_79)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_180: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_79, [-1, 3072]);  mul_79 = None
    addmm_39: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_79, view_180, primals_80);  primals_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_181: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_39, [1, 1024, 768]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    native_dropout_30 = torch.ops.aten.native_dropout.default(view_181, 0.1, True);  view_181 = None
    getitem_130: "f32[1, 1024, 768]" = native_dropout_30[0]
    getitem_131: "b8[1, 1024, 768]" = native_dropout_30[1];  native_dropout_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_80: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_75, getitem_130);  getitem_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_20 = torch.ops.aten.var_mean.correction(add_80, [2], correction = 0, keepdim = True)
    getitem_132: "f32[1, 1024, 1]" = var_mean_20[0]
    getitem_133: "f32[1, 1024, 1]" = var_mean_20[1];  var_mean_20 = None
    add_81: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-05);  getitem_132 = None
    rsqrt_20: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
    sub_30: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_80, getitem_133)
    mul_80: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = None
    mul_81: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_80, primals_139);  mul_80 = None
    add_82: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_81, primals_140);  mul_81 = primals_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_182: "f32[1024, 768]" = torch.ops.aten.view.default(add_82, [-1, 768]);  add_82 = None
    addmm_40: "f32[1024, 2304]" = torch.ops.aten.addmm.default(primals_81, view_182, primals_82);  primals_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_183: "f32[1, 1024, 2304]" = torch.ops.aten.view.default(addmm_40, [1, 1024, 2304]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_10 = torch.ops.aten.split_with_sizes.default(view_183, [768, 768, 768], 2);  view_183 = None
    getitem_134: "f32[1, 1024, 768]" = split_with_sizes_10[0]
    getitem_135: "f32[1, 1024, 768]" = split_with_sizes_10[1]
    getitem_136: "f32[1, 1024, 768]" = split_with_sizes_10[2];  split_with_sizes_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_184: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_134, [1, 1024, 12, 64]);  getitem_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_50: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_185: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_135, [1, 1024, 12, 64]);  getitem_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_51: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_186: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_136, [1, 1024, 12, 64]);  getitem_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_52: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_53: "f32[1, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_51, [0, 1, 3, 2])
    expand_40: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_50, [1, 12, 1024, 64]);  permute_50 = None
    view_187: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_40, [12, 1024, 64]);  expand_40 = None
    expand_41: "f32[1, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_53, [1, 12, 64, 1024]);  permute_53 = None
    view_188: "f32[12, 64, 1024]" = torch.ops.aten.view.default(expand_41, [12, 64, 1024]);  expand_41 = None
    bmm_20: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_187, view_188)
    view_189: "f32[1, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_20, [1, 12, 1024, 1024]);  bmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_20: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    div_20: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_189, full_20);  view_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_21: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(primals_160, 0, 0, 9223372036854775807);  primals_160 = None
    slice_22: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_21, 1, 0, 9223372036854775807);  slice_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_21: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_10: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_22, div_20, full_21);  div_20 = full_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_10: "f32[1, 12, 1024, 1]" = torch.ops.aten.amax.default(where_10, [-1], True)
    sub_31: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where_10, amax_10);  where_10 = amax_10 = None
    exp_10: "f32[1, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
    sum_11: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_21: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    alias_20: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(div_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    native_dropout_31 = torch.ops.aten.native_dropout.default(div_21, 0.1, True);  div_21 = None
    getitem_137: "f32[1, 12, 1024, 1024]" = native_dropout_31[0]
    getitem_138: "b8[1, 12, 1024, 1024]" = native_dropout_31[1];  native_dropout_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_42: "f32[1, 12, 1024, 1024]" = torch.ops.aten.expand.default(getitem_137, [1, 12, 1024, 1024]);  getitem_137 = None
    view_190: "f32[12, 1024, 1024]" = torch.ops.aten.view.default(expand_42, [12, 1024, 1024]);  expand_42 = None
    expand_43: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_52, [1, 12, 1024, 64])
    view_191: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_43, [12, 1024, 64]);  expand_43 = None
    bmm_21: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_190, view_191)
    view_192: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_21, [1, 12, 1024, 64]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_54: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None
    clone_10: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_54, memory_format = torch.contiguous_format);  permute_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_193: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_10, [1, 1024, 768]);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_194: "f32[1024, 768]" = torch.ops.aten.view.default(view_193, [-1, 768]);  view_193 = None
    addmm_41: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_83, view_194, primals_84);  primals_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_195: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_41, [1, 1024, 768]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    native_dropout_32 = torch.ops.aten.native_dropout.default(view_195, 0.1, True);  view_195 = None
    getitem_139: "f32[1, 1024, 768]" = native_dropout_32[0]
    getitem_140: "b8[1, 1024, 768]" = native_dropout_32[1];  native_dropout_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_83: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_139, add_80);  getitem_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_21 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
    getitem_141: "f32[1, 1024, 1]" = var_mean_21[0]
    getitem_142: "f32[1, 1024, 1]" = var_mean_21[1];  var_mean_21 = None
    add_84: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_141, 1e-05);  getitem_141 = None
    rsqrt_21: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
    sub_32: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_83, getitem_142)
    mul_82: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = None
    mul_83: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_82, primals_141);  mul_82 = None
    add_85: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_83, primals_142);  mul_83 = primals_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_196: "f32[1024, 768]" = torch.ops.aten.view.default(add_85, [-1, 768]);  add_85 = None
    addmm_42: "f32[1024, 3072]" = torch.ops.aten.addmm.default(primals_85, view_196, primals_86);  primals_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_197: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_42, [1, 1024, 3072]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_84: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_197, 0.5)
    pow_11: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_197, 3.0)
    mul_85: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
    add_86: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(view_197, mul_85);  mul_85 = None
    mul_86: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_86, 0.7978845608028654);  add_86 = None
    tanh_10: "f32[1, 1024, 3072]" = torch.ops.aten.tanh.default(mul_86);  mul_86 = None
    alias_21: "f32[1, 1024, 3072]" = torch.ops.aten.alias.default(tanh_10)
    add_87: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
    mul_87: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_84, add_87)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_198: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_87, [-1, 3072]);  mul_87 = None
    addmm_43: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_87, view_198, primals_88);  primals_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_199: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_43, [1, 1024, 768]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    native_dropout_33 = torch.ops.aten.native_dropout.default(view_199, 0.1, True);  view_199 = None
    getitem_143: "f32[1, 1024, 768]" = native_dropout_33[0]
    getitem_144: "b8[1, 1024, 768]" = native_dropout_33[1];  native_dropout_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_88: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_83, getitem_143);  getitem_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    var_mean_22 = torch.ops.aten.var_mean.correction(add_88, [2], correction = 0, keepdim = True)
    getitem_145: "f32[1, 1024, 1]" = var_mean_22[0]
    getitem_146: "f32[1, 1024, 1]" = var_mean_22[1];  var_mean_22 = None
    add_89: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_145, 1e-05);  getitem_145 = None
    rsqrt_22: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
    sub_33: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_88, getitem_146)
    mul_88: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = None
    mul_89: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_88, primals_143);  mul_88 = None
    add_90: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_89, primals_144);  mul_89 = primals_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_200: "f32[1024, 768]" = torch.ops.aten.view.default(add_90, [-1, 768]);  add_90 = None
    addmm_44: "f32[1024, 2304]" = torch.ops.aten.addmm.default(primals_89, view_200, primals_90);  primals_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_201: "f32[1, 1024, 2304]" = torch.ops.aten.view.default(addmm_44, [1, 1024, 2304]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    split_with_sizes_11 = torch.ops.aten.split_with_sizes.default(view_201, [768, 768, 768], 2);  view_201 = None
    getitem_147: "f32[1, 1024, 768]" = split_with_sizes_11[0]
    getitem_148: "f32[1, 1024, 768]" = split_with_sizes_11[1]
    getitem_149: "f32[1, 1024, 768]" = split_with_sizes_11[2];  split_with_sizes_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_202: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_147, [1, 1024, 12, 64]);  getitem_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_55: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_202, [0, 2, 1, 3]);  view_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_203: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_148, [1, 1024, 12, 64]);  getitem_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_56: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_203, [0, 2, 1, 3]);  view_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    view_204: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_149, [1, 1024, 12, 64]);  getitem_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_57: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_204, [0, 2, 1, 3]);  view_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_58: "f32[1, 12, 64, 1024]" = torch.ops.aten.permute.default(permute_56, [0, 1, 3, 2])
    expand_44: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_55, [1, 12, 1024, 64]);  permute_55 = None
    view_205: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_44, [12, 1024, 64]);  expand_44 = None
    expand_45: "f32[1, 12, 64, 1024]" = torch.ops.aten.expand.default(permute_58, [1, 12, 64, 1024]);  permute_58 = None
    view_206: "f32[12, 64, 1024]" = torch.ops.aten.view.default(expand_45, [12, 64, 1024]);  expand_45 = None
    bmm_22: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_205, view_206)
    view_207: "f32[1, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_22, [1, 12, 1024, 1024]);  bmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    full_22: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    div_22: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(view_207, full_22);  view_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:197, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_23: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(primals_161, 0, 0, 9223372036854775807);  primals_161 = None
    slice_24: "b8[1, 1, 1024, 1024]" = torch.ops.aten.slice.Tensor(slice_23, 1, 0, 9223372036854775807);  slice_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:201, code: mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_23: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    where_11: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_24, div_22, full_23);  div_22 = full_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_11: "f32[1, 12, 1024, 1]" = torch.ops.aten.amax.default(where_11, [-1], True)
    sub_34: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(where_11, amax_11);  where_11 = amax_11 = None
    exp_11: "f32[1, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
    sum_12: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_23: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    alias_22: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(div_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    native_dropout_34 = torch.ops.aten.native_dropout.default(div_23, 0.1, True);  div_23 = None
    getitem_150: "f32[1, 12, 1024, 1024]" = native_dropout_34[0]
    getitem_151: "b8[1, 12, 1024, 1024]" = native_dropout_34[1];  native_dropout_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    expand_46: "f32[1, 12, 1024, 1024]" = torch.ops.aten.expand.default(getitem_150, [1, 12, 1024, 1024]);  getitem_150 = None
    view_208: "f32[12, 1024, 1024]" = torch.ops.aten.view.default(expand_46, [12, 1024, 1024]);  expand_46 = None
    expand_47: "f32[1, 12, 1024, 64]" = torch.ops.aten.expand.default(permute_57, [1, 12, 1024, 64])
    view_209: "f32[12, 1024, 64]" = torch.ops.aten.view.default(expand_47, [12, 1024, 64]);  expand_47 = None
    bmm_23: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_208, view_209)
    view_210: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_23, [1, 12, 1024, 64]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_59: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_210, [0, 2, 1, 3]);  view_210 = None
    clone_11: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_211: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_11, [1, 1024, 768]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_212: "f32[1024, 768]" = torch.ops.aten.view.default(view_211, [-1, 768]);  view_211 = None
    addmm_45: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_91, view_212, primals_92);  primals_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_213: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_45, [1, 1024, 768]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    native_dropout_35 = torch.ops.aten.native_dropout.default(view_213, 0.1, True);  view_213 = None
    getitem_152: "f32[1, 1024, 768]" = native_dropout_35[0]
    getitem_153: "b8[1, 1024, 768]" = native_dropout_35[1];  native_dropout_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:401, code: hidden_states = attn_output + residual
    add_91: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(getitem_152, add_88);  getitem_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    var_mean_23 = torch.ops.aten.var_mean.correction(add_91, [2], correction = 0, keepdim = True)
    getitem_154: "f32[1, 1024, 1]" = var_mean_23[0]
    getitem_155: "f32[1, 1024, 1]" = var_mean_23[1];  var_mean_23 = None
    add_92: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_154, 1e-05);  getitem_154 = None
    rsqrt_23: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
    sub_35: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_91, getitem_155)
    mul_90: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = None
    mul_91: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_90, primals_145);  mul_90 = None
    add_93: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_91, primals_146);  mul_91 = primals_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_214: "f32[1024, 768]" = torch.ops.aten.view.default(add_93, [-1, 768]);  add_93 = None
    addmm_46: "f32[1024, 3072]" = torch.ops.aten.addmm.default(primals_93, view_214, primals_94);  primals_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_215: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_46, [1, 1024, 3072]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_92: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_215, 0.5)
    pow_12: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_215, 3.0)
    mul_93: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
    add_94: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(view_215, mul_93);  mul_93 = None
    mul_94: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_94, 0.7978845608028654);  add_94 = None
    tanh_11: "f32[1, 1024, 3072]" = torch.ops.aten.tanh.default(mul_94);  mul_94 = None
    alias_23: "f32[1, 1024, 3072]" = torch.ops.aten.alias.default(tanh_11)
    add_95: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
    mul_95: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_92, add_95)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    view_216: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_95, [-1, 3072]);  mul_95 = None
    addmm_47: "f32[1024, 768]" = torch.ops.aten.addmm.default(primals_95, view_216, primals_96);  primals_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_217: "f32[1, 1024, 768]" = torch.ops.aten.view.default(addmm_47, [1, 1024, 768]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    native_dropout_36 = torch.ops.aten.native_dropout.default(view_217, 0.1, True);  view_217 = None
    getitem_156: "f32[1, 1024, 768]" = native_dropout_36[0]
    getitem_157: "b8[1, 1024, 768]" = native_dropout_36[1];  native_dropout_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:429, code: hidden_states = residual + feed_forward_hidden_states
    add_96: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_91, getitem_156);  getitem_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:926, code: hidden_states = self.ln_f(hidden_states)
    var_mean_24 = torch.ops.aten.var_mean.correction(add_96, [2], correction = 0, keepdim = True)
    getitem_158: "f32[1, 1024, 1]" = var_mean_24[0]
    getitem_159: "f32[1, 1024, 1]" = var_mean_24[1];  var_mean_24 = None
    add_97: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_158, 1e-05);  getitem_158 = None
    rsqrt_24: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
    sub_36: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_96, getitem_159)
    mul_96: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = None
    mul_97: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_96, primals_147);  mul_96 = None
    add_98: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_97, primals_148);  mul_97 = primals_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:928, code: hidden_states = hidden_states.view(output_shape)
    view_218: "f32[1, 1024, 768]" = torch.ops.aten.view.default(add_98, [-1, 1024, 768]);  add_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1433, code: logits = self.score(hidden_states)
    permute_60: "f32[768, 2]" = torch.ops.aten.permute.default(primals_149, [1, 0]);  primals_149 = None
    view_219: "f32[1024, 768]" = torch.ops.aten.view.default(view_218, [1024, 768])
    mm: "f32[1024, 2]" = torch.ops.aten.mm.default(view_219, permute_60)
    view_220: "f32[1, 1024, 2]" = torch.ops.aten.view.default(mm, [1, 1024, 2]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1447, code: sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1).to(
    eq: "b8[1, 1024]" = torch.ops.aten.eq.Scalar(primals_162, 0);  primals_162 = None
    convert_element_type: "i64[1, 1024]" = torch.ops.prims.convert_element_type.default(eq, torch.int64);  eq = None
    argmax: "i64[1]" = torch.ops.aten.argmax.default(convert_element_type, -1);  convert_element_type = None
    sub_37: "i64[1]" = torch.ops.aten.sub.Tensor(argmax, 1);  argmax = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1457, code: pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
    iota_1: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    index: "f32[1, 2]" = torch.ops.aten.index.Tensor(view_220, [iota_1, sub_37]);  view_220 = None
    full_24: "f32[1, 1024, 2]" = torch.ops.aten.full.default([1, 1024, 2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    index_put: "f32[1, 1024, 2]" = torch.ops.aten.index_put.default(full_24, [iota_1, sub_37], tangents_26, True);  full_24 = iota_1 = sub_37 = tangents_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1433, code: logits = self.score(hidden_states)
    view_221: "f32[1024, 2]" = torch.ops.aten.view.default(index_put, [1024, 2]);  index_put = None
    permute_61: "f32[2, 1024]" = torch.ops.aten.permute.default(view_221, [1, 0])
    mm_1: "f32[2, 768]" = torch.ops.aten.mm.default(permute_61, view_219);  permute_61 = view_219 = None
    permute_62: "f32[768, 2]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    permute_63: "f32[2, 768]" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None
    mm_2: "f32[1024, 768]" = torch.ops.aten.mm.default(view_221, permute_63);  view_221 = permute_63 = None
    view_222: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_2, [1, 1024, 768]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1433, code: logits = self.score(hidden_states)
    add_99: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(tangents_1, view_222);  tangents_1 = view_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1433, code: logits = self.score(hidden_states)
    permute_64: "f32[2, 768]" = torch.ops.aten.permute.default(permute_62, [1, 0]);  permute_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:928, code: hidden_states = hidden_states.view(output_shape)
    view_223: "f32[1, 1024, 768]" = torch.ops.aten.view.default(add_99, [1, 1024, 768]);  add_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:926, code: hidden_states = self.ln_f(hidden_states)
    sub_38: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_96, getitem_159);  add_96 = getitem_159 = None
    mul_98: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_24);  sub_38 = None
    mul_99: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_223, primals_147);  primals_147 = None
    mul_100: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_99, 768)
    sum_13: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_99, [2], True)
    mul_101: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_99, mul_98);  mul_99 = None
    sum_14: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_101, [2], True);  mul_101 = None
    mul_102: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_98, sum_14);  sum_14 = None
    sub_39: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_100, sum_13);  mul_100 = sum_13 = None
    sub_40: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_39, mul_102);  sub_39 = mul_102 = None
    div_24: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None
    mul_103: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_24, sub_40);  div_24 = sub_40 = None
    mul_104: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_223, mul_98);  mul_98 = None
    sum_15: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_104, [0, 1]);  mul_104 = None
    sum_16: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_223, [0, 1]);  view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_1: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_157, torch.float32);  getitem_157 = None
    mul_105: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
    mul_106: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_103, mul_105);  mul_105 = None
    clone_12: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_106, memory_format = torch.contiguous_format);  mul_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_224: "f32[1024, 768]" = torch.ops.aten.view.default(clone_12, [1024, 768]);  clone_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_65: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_96, [1, 0]);  primals_96 = None
    mm_3: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_224, permute_65);  permute_65 = None
    permute_66: "f32[3072, 1024]" = torch.ops.aten.permute.default(view_216, [1, 0]);  view_216 = None
    mm_4: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_66, view_224);  permute_66 = None
    sum_17: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_224, [0], True);  view_224 = None
    view_225: "f32[768]" = torch.ops.aten.view.default(sum_17, [768]);  sum_17 = None
    view_226: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(mm_3, [1, 1024, 3072]);  mm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_107: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_226, mul_92);  mul_92 = None
    mul_108: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_226, add_95);  view_226 = add_95 = None
    alias_24: "f32[1, 1024, 3072]" = torch.ops.aten.alias.default(alias_23);  alias_23 = None
    mul_109: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(alias_24, alias_24);  alias_24 = None
    sub_41: "f32[1, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_109);  mul_109 = None
    mul_110: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_107, sub_41);  mul_107 = sub_41 = None
    mul_111: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_110, 0.7978845608028654);  mul_110 = None
    mul_112: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_111, 0.044715)
    pow_13: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_215, 2.0);  view_215 = None
    mul_113: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_13, 3.0);  pow_13 = None
    mul_114: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_112, mul_113);  mul_112 = mul_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_100: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_111, mul_114);  mul_111 = mul_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_115: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_108, 0.5);  mul_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_101: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(add_100, mul_115);  add_100 = mul_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_227: "f32[1024, 3072]" = torch.ops.aten.view.default(add_101, [1024, 3072]);  add_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_67: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_94, [1, 0]);  primals_94 = None
    mm_5: "f32[1024, 768]" = torch.ops.aten.mm.default(view_227, permute_67);  permute_67 = None
    permute_68: "f32[768, 1024]" = torch.ops.aten.permute.default(view_214, [1, 0]);  view_214 = None
    mm_6: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_68, view_227);  permute_68 = None
    sum_18: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_227, [0], True);  view_227 = None
    view_228: "f32[3072]" = torch.ops.aten.view.default(sum_18, [3072]);  sum_18 = None
    view_229: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_5, [1, 1024, 768]);  mm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    sub_42: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_91, getitem_155);  add_91 = getitem_155 = None
    mul_116: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_23);  sub_42 = None
    mul_117: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_229, primals_145);  primals_145 = None
    mul_118: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_117, 768)
    sum_19: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_117, [2], True)
    mul_119: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_117, mul_116);  mul_117 = None
    sum_20: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_119, [2], True);  mul_119 = None
    mul_120: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_116, sum_20);  sum_20 = None
    sub_43: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_118, sum_19);  mul_118 = sum_19 = None
    sub_44: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_43, mul_120);  sub_43 = mul_120 = None
    div_25: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
    mul_121: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_25, sub_44);  div_25 = sub_44 = None
    mul_122: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_229, mul_116);  mul_116 = None
    sum_21: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_122, [0, 1]);  mul_122 = None
    sum_22: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_229, [0, 1]);  view_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    add_102: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_103, mul_121);  mul_103 = mul_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    convert_element_type_2: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_153, torch.float32);  getitem_153 = None
    mul_123: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
    mul_124: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_102, mul_123);  mul_123 = None
    clone_13: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_124, memory_format = torch.contiguous_format);  mul_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_230: "f32[1024, 768]" = torch.ops.aten.view.default(clone_13, [1024, 768]);  clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_69: "f32[768, 768]" = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
    mm_7: "f32[1024, 768]" = torch.ops.aten.mm.default(view_230, permute_69);  permute_69 = None
    permute_70: "f32[768, 1024]" = torch.ops.aten.permute.default(view_212, [1, 0]);  view_212 = None
    mm_8: "f32[768, 768]" = torch.ops.aten.mm.default(permute_70, view_230);  permute_70 = None
    sum_23: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_230, [0], True);  view_230 = None
    view_231: "f32[768]" = torch.ops.aten.view.default(sum_23, [768]);  sum_23 = None
    view_232: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_7, [1, 1024, 768]);  mm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_233: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(view_232, [1, 1024, 12, 64]);  view_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_71: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_233, [0, 2, 1, 3]);  view_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_234: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_71, [12, 1024, 64]);  permute_71 = None
    permute_72: "f32[12, 1024, 1024]" = torch.ops.aten.permute.default(view_208, [0, 2, 1]);  view_208 = None
    bmm_24: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(permute_72, view_234);  permute_72 = None
    permute_73: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_209, [0, 2, 1]);  view_209 = None
    bmm_25: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_234, permute_73);  view_234 = permute_73 = None
    view_235: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_24, [1, 12, 1024, 64]);  bmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    add_103: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_25, view_235);  tangents_25 = view_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_236: "f32[1, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_25, [1, 12, 1024, 1024]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    convert_element_type_3: "f32[1, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_151, torch.float32);  getitem_151 = None
    mul_125: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.1111111111111112);  convert_element_type_3 = None
    mul_126: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_236, mul_125);  view_236 = mul_125 = None
    clone_14: "f32[1, 12, 1024, 1024]" = torch.ops.aten.clone.default(mul_126, memory_format = torch.contiguous_format);  mul_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_25: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(alias_22);  alias_22 = None
    mul_127: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(clone_14, alias_25);  clone_14 = None
    sum_24: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_127, [-1], True)
    mul_128: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_25, sum_24);  alias_25 = sum_24 = None
    sub_45: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_127, mul_128);  mul_127 = mul_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_12: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_24, sub_45, scalar_tensor);  slice_24 = sub_45 = scalar_tensor = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_26: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(where_12, full_22);  where_12 = full_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_237: "f32[12, 1024, 1024]" = torch.ops.aten.view.default(div_26, [12, 1024, 1024]);  div_26 = None
    permute_74: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_205, [0, 2, 1]);  view_205 = None
    bmm_26: "f32[12, 64, 1024]" = torch.ops.aten.bmm.default(permute_74, view_237);  permute_74 = None
    permute_75: "f32[12, 1024, 64]" = torch.ops.aten.permute.default(view_206, [0, 2, 1]);  view_206 = None
    bmm_27: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_237, permute_75);  view_237 = permute_75 = None
    view_238: "f32[1, 12, 64, 1024]" = torch.ops.aten.view.default(bmm_26, [1, 12, 64, 1024]);  bmm_26 = None
    view_239: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_27, [1, 12, 1024, 64]);  bmm_27 = None
    permute_76: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_238, [0, 1, 3, 2]);  view_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_104: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_24, permute_76);  tangents_24 = permute_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_77: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_103, [0, 2, 1, 3]);  add_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_15: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_77, memory_format = torch.contiguous_format);  permute_77 = None
    view_240: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_15, [1, 1024, 768]);  clone_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_78: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_104, [0, 2, 1, 3]);  add_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_16: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_78, memory_format = torch.contiguous_format);  permute_78 = None
    view_241: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_16, [1, 1024, 768]);  clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_79: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_239, [0, 2, 1, 3]);  view_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_17: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None
    view_242: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_17, [1, 1024, 768]);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    cat: "f32[1, 1024, 2304]" = torch.ops.aten.cat.default([view_242, view_241, view_240], 2);  view_242 = view_241 = view_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_243: "f32[1024, 2304]" = torch.ops.aten.view.default(cat, [1024, 2304]);  cat = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_80: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_90, [1, 0]);  primals_90 = None
    mm_9: "f32[1024, 768]" = torch.ops.aten.mm.default(view_243, permute_80);  permute_80 = None
    permute_81: "f32[768, 1024]" = torch.ops.aten.permute.default(view_200, [1, 0]);  view_200 = None
    mm_10: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_81, view_243);  permute_81 = None
    sum_25: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_243, [0], True);  view_243 = None
    view_244: "f32[2304]" = torch.ops.aten.view.default(sum_25, [2304]);  sum_25 = None
    view_245: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_9, [1, 1024, 768]);  mm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    sub_46: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_88, getitem_146);  add_88 = getitem_146 = None
    mul_129: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_22);  sub_46 = None
    mul_130: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_245, primals_143);  primals_143 = None
    mul_131: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_130, 768)
    sum_26: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_130, [2], True)
    mul_132: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_130, mul_129);  mul_130 = None
    sum_27: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_132, [2], True);  mul_132 = None
    mul_133: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_129, sum_27);  sum_27 = None
    sub_47: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_131, sum_26);  mul_131 = sum_26 = None
    sub_48: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_47, mul_133);  sub_47 = mul_133 = None
    div_27: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
    mul_134: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_27, sub_48);  div_27 = sub_48 = None
    mul_135: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_245, mul_129);  mul_129 = None
    sum_28: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_135, [0, 1]);  mul_135 = None
    sum_29: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_245, [0, 1]);  view_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    add_105: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_102, mul_134);  add_102 = mul_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_4: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_144, torch.float32);  getitem_144 = None
    mul_136: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 1.1111111111111112);  convert_element_type_4 = None
    mul_137: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_105, mul_136);  mul_136 = None
    clone_18: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_137, memory_format = torch.contiguous_format);  mul_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_246: "f32[1024, 768]" = torch.ops.aten.view.default(clone_18, [1024, 768]);  clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_82: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_88, [1, 0]);  primals_88 = None
    mm_11: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_246, permute_82);  permute_82 = None
    permute_83: "f32[3072, 1024]" = torch.ops.aten.permute.default(view_198, [1, 0]);  view_198 = None
    mm_12: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_83, view_246);  permute_83 = None
    sum_30: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_246, [0], True);  view_246 = None
    view_247: "f32[768]" = torch.ops.aten.view.default(sum_30, [768]);  sum_30 = None
    view_248: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(mm_11, [1, 1024, 3072]);  mm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_138: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_248, mul_84);  mul_84 = None
    mul_139: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_248, add_87);  view_248 = add_87 = None
    alias_26: "f32[1, 1024, 3072]" = torch.ops.aten.alias.default(alias_21);  alias_21 = None
    mul_140: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(alias_26, alias_26);  alias_26 = None
    sub_49: "f32[1, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_140);  mul_140 = None
    mul_141: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_138, sub_49);  mul_138 = sub_49 = None
    mul_142: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_141, 0.7978845608028654);  mul_141 = None
    mul_143: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_142, 0.044715)
    pow_14: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_197, 2.0);  view_197 = None
    mul_144: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_14, 3.0);  pow_14 = None
    mul_145: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_143, mul_144);  mul_143 = mul_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_106: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_142, mul_145);  mul_142 = mul_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_146: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_139, 0.5);  mul_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_107: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(add_106, mul_146);  add_106 = mul_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_249: "f32[1024, 3072]" = torch.ops.aten.view.default(add_107, [1024, 3072]);  add_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_84: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_86, [1, 0]);  primals_86 = None
    mm_13: "f32[1024, 768]" = torch.ops.aten.mm.default(view_249, permute_84);  permute_84 = None
    permute_85: "f32[768, 1024]" = torch.ops.aten.permute.default(view_196, [1, 0]);  view_196 = None
    mm_14: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_85, view_249);  permute_85 = None
    sum_31: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_249, [0], True);  view_249 = None
    view_250: "f32[3072]" = torch.ops.aten.view.default(sum_31, [3072]);  sum_31 = None
    view_251: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_13, [1, 1024, 768]);  mm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    sub_50: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_83, getitem_142);  add_83 = getitem_142 = None
    mul_147: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_21);  sub_50 = None
    mul_148: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_251, primals_141);  primals_141 = None
    mul_149: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_148, 768)
    sum_32: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_148, [2], True)
    mul_150: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_148, mul_147);  mul_148 = None
    sum_33: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_150, [2], True);  mul_150 = None
    mul_151: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_147, sum_33);  sum_33 = None
    sub_51: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_149, sum_32);  mul_149 = sum_32 = None
    sub_52: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_51, mul_151);  sub_51 = mul_151 = None
    div_28: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
    mul_152: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_28, sub_52);  div_28 = sub_52 = None
    mul_153: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_251, mul_147);  mul_147 = None
    sum_34: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_153, [0, 1]);  mul_153 = None
    sum_35: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_251, [0, 1]);  view_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    add_108: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_105, mul_152);  add_105 = mul_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    convert_element_type_5: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_140, torch.float32);  getitem_140 = None
    mul_154: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 1.1111111111111112);  convert_element_type_5 = None
    mul_155: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_108, mul_154);  mul_154 = None
    clone_19: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_155, memory_format = torch.contiguous_format);  mul_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_252: "f32[1024, 768]" = torch.ops.aten.view.default(clone_19, [1024, 768]);  clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_86: "f32[768, 768]" = torch.ops.aten.permute.default(primals_84, [1, 0]);  primals_84 = None
    mm_15: "f32[1024, 768]" = torch.ops.aten.mm.default(view_252, permute_86);  permute_86 = None
    permute_87: "f32[768, 1024]" = torch.ops.aten.permute.default(view_194, [1, 0]);  view_194 = None
    mm_16: "f32[768, 768]" = torch.ops.aten.mm.default(permute_87, view_252);  permute_87 = None
    sum_36: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_252, [0], True);  view_252 = None
    view_253: "f32[768]" = torch.ops.aten.view.default(sum_36, [768]);  sum_36 = None
    view_254: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_15, [1, 1024, 768]);  mm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_255: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(view_254, [1, 1024, 12, 64]);  view_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_88: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_255, [0, 2, 1, 3]);  view_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_256: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_88, [12, 1024, 64]);  permute_88 = None
    permute_89: "f32[12, 1024, 1024]" = torch.ops.aten.permute.default(view_190, [0, 2, 1]);  view_190 = None
    bmm_28: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(permute_89, view_256);  permute_89 = None
    permute_90: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_191, [0, 2, 1]);  view_191 = None
    bmm_29: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_256, permute_90);  view_256 = permute_90 = None
    view_257: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_28, [1, 12, 1024, 64]);  bmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    add_109: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_23, view_257);  tangents_23 = view_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_258: "f32[1, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_29, [1, 12, 1024, 1024]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    convert_element_type_6: "f32[1, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_138, torch.float32);  getitem_138 = None
    mul_156: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 1.1111111111111112);  convert_element_type_6 = None
    mul_157: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_258, mul_156);  view_258 = mul_156 = None
    clone_20: "f32[1, 12, 1024, 1024]" = torch.ops.aten.clone.default(mul_157, memory_format = torch.contiguous_format);  mul_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_27: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(alias_20);  alias_20 = None
    mul_158: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(clone_20, alias_27);  clone_20 = None
    sum_37: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_158, [-1], True)
    mul_159: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_27, sum_37);  alias_27 = sum_37 = None
    sub_53: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_158, mul_159);  mul_158 = mul_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_13: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_22, sub_53, scalar_tensor_1);  slice_22 = sub_53 = scalar_tensor_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_29: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(where_13, full_20);  where_13 = full_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_259: "f32[12, 1024, 1024]" = torch.ops.aten.view.default(div_29, [12, 1024, 1024]);  div_29 = None
    permute_91: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_187, [0, 2, 1]);  view_187 = None
    bmm_30: "f32[12, 64, 1024]" = torch.ops.aten.bmm.default(permute_91, view_259);  permute_91 = None
    permute_92: "f32[12, 1024, 64]" = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None
    bmm_31: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_259, permute_92);  view_259 = permute_92 = None
    view_260: "f32[1, 12, 64, 1024]" = torch.ops.aten.view.default(bmm_30, [1, 12, 64, 1024]);  bmm_30 = None
    view_261: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_31, [1, 12, 1024, 64]);  bmm_31 = None
    permute_93: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_260, [0, 1, 3, 2]);  view_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_110: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_22, permute_93);  tangents_22 = permute_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_94: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_109, [0, 2, 1, 3]);  add_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_21: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_94, memory_format = torch.contiguous_format);  permute_94 = None
    view_262: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_21, [1, 1024, 768]);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_95: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_110, [0, 2, 1, 3]);  add_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_22: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
    view_263: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_22, [1, 1024, 768]);  clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_96: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_261, [0, 2, 1, 3]);  view_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_23: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_96, memory_format = torch.contiguous_format);  permute_96 = None
    view_264: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_23, [1, 1024, 768]);  clone_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    cat_1: "f32[1, 1024, 2304]" = torch.ops.aten.cat.default([view_264, view_263, view_262], 2);  view_264 = view_263 = view_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_265: "f32[1024, 2304]" = torch.ops.aten.view.default(cat_1, [1024, 2304]);  cat_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_97: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_82, [1, 0]);  primals_82 = None
    mm_17: "f32[1024, 768]" = torch.ops.aten.mm.default(view_265, permute_97);  permute_97 = None
    permute_98: "f32[768, 1024]" = torch.ops.aten.permute.default(view_182, [1, 0]);  view_182 = None
    mm_18: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_98, view_265);  permute_98 = None
    sum_38: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_265, [0], True);  view_265 = None
    view_266: "f32[2304]" = torch.ops.aten.view.default(sum_38, [2304]);  sum_38 = None
    view_267: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_17, [1, 1024, 768]);  mm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    sub_54: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_80, getitem_133);  add_80 = getitem_133 = None
    mul_160: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_20);  sub_54 = None
    mul_161: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_267, primals_139);  primals_139 = None
    mul_162: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_161, 768)
    sum_39: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_161, [2], True)
    mul_163: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_161, mul_160);  mul_161 = None
    sum_40: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_163, [2], True);  mul_163 = None
    mul_164: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_160, sum_40);  sum_40 = None
    sub_55: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_162, sum_39);  mul_162 = sum_39 = None
    sub_56: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_55, mul_164);  sub_55 = mul_164 = None
    div_30: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
    mul_165: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_30, sub_56);  div_30 = sub_56 = None
    mul_166: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_267, mul_160);  mul_160 = None
    sum_41: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_166, [0, 1]);  mul_166 = None
    sum_42: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_267, [0, 1]);  view_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    add_111: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_108, mul_165);  add_108 = mul_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_7: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_131, torch.float32);  getitem_131 = None
    mul_167: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_7, 1.1111111111111112);  convert_element_type_7 = None
    mul_168: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_111, mul_167);  mul_167 = None
    clone_24: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_168, memory_format = torch.contiguous_format);  mul_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_268: "f32[1024, 768]" = torch.ops.aten.view.default(clone_24, [1024, 768]);  clone_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_99: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_80, [1, 0]);  primals_80 = None
    mm_19: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_268, permute_99);  permute_99 = None
    permute_100: "f32[3072, 1024]" = torch.ops.aten.permute.default(view_180, [1, 0]);  view_180 = None
    mm_20: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_100, view_268);  permute_100 = None
    sum_43: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_268, [0], True);  view_268 = None
    view_269: "f32[768]" = torch.ops.aten.view.default(sum_43, [768]);  sum_43 = None
    view_270: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(mm_19, [1, 1024, 3072]);  mm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_169: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_270, mul_76);  mul_76 = None
    mul_170: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_270, add_79);  view_270 = add_79 = None
    alias_28: "f32[1, 1024, 3072]" = torch.ops.aten.alias.default(alias_19);  alias_19 = None
    mul_171: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(alias_28, alias_28);  alias_28 = None
    sub_57: "f32[1, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_171);  mul_171 = None
    mul_172: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_169, sub_57);  mul_169 = sub_57 = None
    mul_173: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_172, 0.7978845608028654);  mul_172 = None
    mul_174: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_173, 0.044715)
    pow_15: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_179, 2.0);  view_179 = None
    mul_175: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_15, 3.0);  pow_15 = None
    mul_176: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_174, mul_175);  mul_174 = mul_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_112: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_173, mul_176);  mul_173 = mul_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_177: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_170, 0.5);  mul_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_113: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(add_112, mul_177);  add_112 = mul_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_271: "f32[1024, 3072]" = torch.ops.aten.view.default(add_113, [1024, 3072]);  add_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_101: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_78, [1, 0]);  primals_78 = None
    mm_21: "f32[1024, 768]" = torch.ops.aten.mm.default(view_271, permute_101);  permute_101 = None
    permute_102: "f32[768, 1024]" = torch.ops.aten.permute.default(view_178, [1, 0]);  view_178 = None
    mm_22: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_102, view_271);  permute_102 = None
    sum_44: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_271, [0], True);  view_271 = None
    view_272: "f32[3072]" = torch.ops.aten.view.default(sum_44, [3072]);  sum_44 = None
    view_273: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_21, [1, 1024, 768]);  mm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    sub_58: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_75, getitem_129);  add_75 = getitem_129 = None
    mul_178: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_19);  sub_58 = None
    mul_179: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_273, primals_137);  primals_137 = None
    mul_180: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_179, 768)
    sum_45: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_179, [2], True)
    mul_181: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_179, mul_178);  mul_179 = None
    sum_46: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_181, [2], True);  mul_181 = None
    mul_182: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_178, sum_46);  sum_46 = None
    sub_59: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_180, sum_45);  mul_180 = sum_45 = None
    sub_60: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_59, mul_182);  sub_59 = mul_182 = None
    div_31: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
    mul_183: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_31, sub_60);  div_31 = sub_60 = None
    mul_184: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_273, mul_178);  mul_178 = None
    sum_47: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_184, [0, 1]);  mul_184 = None
    sum_48: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_273, [0, 1]);  view_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    add_114: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_111, mul_183);  add_111 = mul_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    convert_element_type_8: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_127, torch.float32);  getitem_127 = None
    mul_185: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.1111111111111112);  convert_element_type_8 = None
    mul_186: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_114, mul_185);  mul_185 = None
    clone_25: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_186, memory_format = torch.contiguous_format);  mul_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_274: "f32[1024, 768]" = torch.ops.aten.view.default(clone_25, [1024, 768]);  clone_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_103: "f32[768, 768]" = torch.ops.aten.permute.default(primals_76, [1, 0]);  primals_76 = None
    mm_23: "f32[1024, 768]" = torch.ops.aten.mm.default(view_274, permute_103);  permute_103 = None
    permute_104: "f32[768, 1024]" = torch.ops.aten.permute.default(view_176, [1, 0]);  view_176 = None
    mm_24: "f32[768, 768]" = torch.ops.aten.mm.default(permute_104, view_274);  permute_104 = None
    sum_49: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_274, [0], True);  view_274 = None
    view_275: "f32[768]" = torch.ops.aten.view.default(sum_49, [768]);  sum_49 = None
    view_276: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_23, [1, 1024, 768]);  mm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_277: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(view_276, [1, 1024, 12, 64]);  view_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_105: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_277, [0, 2, 1, 3]);  view_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_278: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_105, [12, 1024, 64]);  permute_105 = None
    permute_106: "f32[12, 1024, 1024]" = torch.ops.aten.permute.default(view_172, [0, 2, 1]);  view_172 = None
    bmm_32: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(permute_106, view_278);  permute_106 = None
    permute_107: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_173, [0, 2, 1]);  view_173 = None
    bmm_33: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_278, permute_107);  view_278 = permute_107 = None
    view_279: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_32, [1, 12, 1024, 64]);  bmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    add_115: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_21, view_279);  tangents_21 = view_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_280: "f32[1, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_33, [1, 12, 1024, 1024]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    convert_element_type_9: "f32[1, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_125, torch.float32);  getitem_125 = None
    mul_187: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_9, 1.1111111111111112);  convert_element_type_9 = None
    mul_188: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_280, mul_187);  view_280 = mul_187 = None
    clone_26: "f32[1, 12, 1024, 1024]" = torch.ops.aten.clone.default(mul_188, memory_format = torch.contiguous_format);  mul_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_29: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(alias_18);  alias_18 = None
    mul_189: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(clone_26, alias_29);  clone_26 = None
    sum_50: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_189, [-1], True)
    mul_190: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_29, sum_50);  alias_29 = sum_50 = None
    sub_61: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_189, mul_190);  mul_189 = mul_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    scalar_tensor_2: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_14: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_20, sub_61, scalar_tensor_2);  slice_20 = sub_61 = scalar_tensor_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_32: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(where_14, full_18);  where_14 = full_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_281: "f32[12, 1024, 1024]" = torch.ops.aten.view.default(div_32, [12, 1024, 1024]);  div_32 = None
    permute_108: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_169, [0, 2, 1]);  view_169 = None
    bmm_34: "f32[12, 64, 1024]" = torch.ops.aten.bmm.default(permute_108, view_281);  permute_108 = None
    permute_109: "f32[12, 1024, 64]" = torch.ops.aten.permute.default(view_170, [0, 2, 1]);  view_170 = None
    bmm_35: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_281, permute_109);  view_281 = permute_109 = None
    view_282: "f32[1, 12, 64, 1024]" = torch.ops.aten.view.default(bmm_34, [1, 12, 64, 1024]);  bmm_34 = None
    view_283: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_35, [1, 12, 1024, 64]);  bmm_35 = None
    permute_110: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_282, [0, 1, 3, 2]);  view_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_116: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_20, permute_110);  tangents_20 = permute_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_111: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_115, [0, 2, 1, 3]);  add_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_27: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_111, memory_format = torch.contiguous_format);  permute_111 = None
    view_284: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_27, [1, 1024, 768]);  clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_112: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_116, [0, 2, 1, 3]);  add_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_28: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_112, memory_format = torch.contiguous_format);  permute_112 = None
    view_285: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_28, [1, 1024, 768]);  clone_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_113: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_283, [0, 2, 1, 3]);  view_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_29: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_113, memory_format = torch.contiguous_format);  permute_113 = None
    view_286: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_29, [1, 1024, 768]);  clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    cat_2: "f32[1, 1024, 2304]" = torch.ops.aten.cat.default([view_286, view_285, view_284], 2);  view_286 = view_285 = view_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_287: "f32[1024, 2304]" = torch.ops.aten.view.default(cat_2, [1024, 2304]);  cat_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_114: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_74, [1, 0]);  primals_74 = None
    mm_25: "f32[1024, 768]" = torch.ops.aten.mm.default(view_287, permute_114);  permute_114 = None
    permute_115: "f32[768, 1024]" = torch.ops.aten.permute.default(view_164, [1, 0]);  view_164 = None
    mm_26: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_115, view_287);  permute_115 = None
    sum_51: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_287, [0], True);  view_287 = None
    view_288: "f32[2304]" = torch.ops.aten.view.default(sum_51, [2304]);  sum_51 = None
    view_289: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_25, [1, 1024, 768]);  mm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    sub_62: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_72, getitem_120);  add_72 = getitem_120 = None
    mul_191: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_18);  sub_62 = None
    mul_192: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_289, primals_135);  primals_135 = None
    mul_193: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_192, 768)
    sum_52: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_192, [2], True)
    mul_194: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_192, mul_191);  mul_192 = None
    sum_53: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_194, [2], True);  mul_194 = None
    mul_195: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_191, sum_53);  sum_53 = None
    sub_63: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_193, sum_52);  mul_193 = sum_52 = None
    sub_64: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_63, mul_195);  sub_63 = mul_195 = None
    div_33: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
    mul_196: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_33, sub_64);  div_33 = sub_64 = None
    mul_197: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_289, mul_191);  mul_191 = None
    sum_54: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_197, [0, 1]);  mul_197 = None
    sum_55: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_289, [0, 1]);  view_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    add_117: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_114, mul_196);  add_114 = mul_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_10: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_118, torch.float32);  getitem_118 = None
    mul_198: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.1111111111111112);  convert_element_type_10 = None
    mul_199: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_117, mul_198);  mul_198 = None
    clone_30: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_199, memory_format = torch.contiguous_format);  mul_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_290: "f32[1024, 768]" = torch.ops.aten.view.default(clone_30, [1024, 768]);  clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_116: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
    mm_27: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_290, permute_116);  permute_116 = None
    permute_117: "f32[3072, 1024]" = torch.ops.aten.permute.default(view_162, [1, 0]);  view_162 = None
    mm_28: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_117, view_290);  permute_117 = None
    sum_56: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_290, [0], True);  view_290 = None
    view_291: "f32[768]" = torch.ops.aten.view.default(sum_56, [768]);  sum_56 = None
    view_292: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(mm_27, [1, 1024, 3072]);  mm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_200: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_292, mul_68);  mul_68 = None
    mul_201: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_292, add_71);  view_292 = add_71 = None
    alias_30: "f32[1, 1024, 3072]" = torch.ops.aten.alias.default(alias_17);  alias_17 = None
    mul_202: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(alias_30, alias_30);  alias_30 = None
    sub_65: "f32[1, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_202);  mul_202 = None
    mul_203: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_200, sub_65);  mul_200 = sub_65 = None
    mul_204: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_203, 0.7978845608028654);  mul_203 = None
    mul_205: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_204, 0.044715)
    pow_16: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_161, 2.0);  view_161 = None
    mul_206: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_16, 3.0);  pow_16 = None
    mul_207: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_205, mul_206);  mul_205 = mul_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_118: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_204, mul_207);  mul_204 = mul_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_208: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_201, 0.5);  mul_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_119: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(add_118, mul_208);  add_118 = mul_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_293: "f32[1024, 3072]" = torch.ops.aten.view.default(add_119, [1024, 3072]);  add_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_118: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_70, [1, 0]);  primals_70 = None
    mm_29: "f32[1024, 768]" = torch.ops.aten.mm.default(view_293, permute_118);  permute_118 = None
    permute_119: "f32[768, 1024]" = torch.ops.aten.permute.default(view_160, [1, 0]);  view_160 = None
    mm_30: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_119, view_293);  permute_119 = None
    sum_57: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_293, [0], True);  view_293 = None
    view_294: "f32[3072]" = torch.ops.aten.view.default(sum_57, [3072]);  sum_57 = None
    view_295: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_29, [1, 1024, 768]);  mm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    sub_66: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_67, getitem_116);  add_67 = getitem_116 = None
    mul_209: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_17);  sub_66 = None
    mul_210: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_295, primals_133);  primals_133 = None
    mul_211: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_210, 768)
    sum_58: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_210, [2], True)
    mul_212: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_210, mul_209);  mul_210 = None
    sum_59: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_212, [2], True);  mul_212 = None
    mul_213: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_209, sum_59);  sum_59 = None
    sub_67: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_211, sum_58);  mul_211 = sum_58 = None
    sub_68: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_67, mul_213);  sub_67 = mul_213 = None
    div_34: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
    mul_214: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_34, sub_68);  div_34 = sub_68 = None
    mul_215: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_295, mul_209);  mul_209 = None
    sum_60: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_215, [0, 1]);  mul_215 = None
    sum_61: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_295, [0, 1]);  view_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    add_120: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_117, mul_214);  add_117 = mul_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    convert_element_type_11: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_114, torch.float32);  getitem_114 = None
    mul_216: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 1.1111111111111112);  convert_element_type_11 = None
    mul_217: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_120, mul_216);  mul_216 = None
    clone_31: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_217, memory_format = torch.contiguous_format);  mul_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_296: "f32[1024, 768]" = torch.ops.aten.view.default(clone_31, [1024, 768]);  clone_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_120: "f32[768, 768]" = torch.ops.aten.permute.default(primals_68, [1, 0]);  primals_68 = None
    mm_31: "f32[1024, 768]" = torch.ops.aten.mm.default(view_296, permute_120);  permute_120 = None
    permute_121: "f32[768, 1024]" = torch.ops.aten.permute.default(view_158, [1, 0]);  view_158 = None
    mm_32: "f32[768, 768]" = torch.ops.aten.mm.default(permute_121, view_296);  permute_121 = None
    sum_62: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_296, [0], True);  view_296 = None
    view_297: "f32[768]" = torch.ops.aten.view.default(sum_62, [768]);  sum_62 = None
    view_298: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_31, [1, 1024, 768]);  mm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_299: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(view_298, [1, 1024, 12, 64]);  view_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_122: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_299, [0, 2, 1, 3]);  view_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_300: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_122, [12, 1024, 64]);  permute_122 = None
    permute_123: "f32[12, 1024, 1024]" = torch.ops.aten.permute.default(view_154, [0, 2, 1]);  view_154 = None
    bmm_36: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(permute_123, view_300);  permute_123 = None
    permute_124: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_155, [0, 2, 1]);  view_155 = None
    bmm_37: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_300, permute_124);  view_300 = permute_124 = None
    view_301: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_36, [1, 12, 1024, 64]);  bmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    add_121: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_19, view_301);  tangents_19 = view_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_302: "f32[1, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_37, [1, 12, 1024, 1024]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    convert_element_type_12: "f32[1, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_112, torch.float32);  getitem_112 = None
    mul_218: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_12, 1.1111111111111112);  convert_element_type_12 = None
    mul_219: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_302, mul_218);  view_302 = mul_218 = None
    clone_32: "f32[1, 12, 1024, 1024]" = torch.ops.aten.clone.default(mul_219, memory_format = torch.contiguous_format);  mul_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_31: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(alias_16);  alias_16 = None
    mul_220: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(clone_32, alias_31);  clone_32 = None
    sum_63: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_220, [-1], True)
    mul_221: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_31, sum_63);  alias_31 = sum_63 = None
    sub_69: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_220, mul_221);  mul_220 = mul_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_15: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_18, sub_69, scalar_tensor_3);  slice_18 = sub_69 = scalar_tensor_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_35: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(where_15, full_16);  where_15 = full_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_303: "f32[12, 1024, 1024]" = torch.ops.aten.view.default(div_35, [12, 1024, 1024]);  div_35 = None
    permute_125: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_151, [0, 2, 1]);  view_151 = None
    bmm_38: "f32[12, 64, 1024]" = torch.ops.aten.bmm.default(permute_125, view_303);  permute_125 = None
    permute_126: "f32[12, 1024, 64]" = torch.ops.aten.permute.default(view_152, [0, 2, 1]);  view_152 = None
    bmm_39: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_303, permute_126);  view_303 = permute_126 = None
    view_304: "f32[1, 12, 64, 1024]" = torch.ops.aten.view.default(bmm_38, [1, 12, 64, 1024]);  bmm_38 = None
    view_305: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_39, [1, 12, 1024, 64]);  bmm_39 = None
    permute_127: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_304, [0, 1, 3, 2]);  view_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_122: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_18, permute_127);  tangents_18 = permute_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_128: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_121, [0, 2, 1, 3]);  add_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_33: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
    view_306: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_33, [1, 1024, 768]);  clone_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_129: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_122, [0, 2, 1, 3]);  add_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_34: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None
    view_307: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_34, [1, 1024, 768]);  clone_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_130: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_305, [0, 2, 1, 3]);  view_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_35: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_130, memory_format = torch.contiguous_format);  permute_130 = None
    view_308: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_35, [1, 1024, 768]);  clone_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    cat_3: "f32[1, 1024, 2304]" = torch.ops.aten.cat.default([view_308, view_307, view_306], 2);  view_308 = view_307 = view_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_309: "f32[1024, 2304]" = torch.ops.aten.view.default(cat_3, [1024, 2304]);  cat_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_131: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
    mm_33: "f32[1024, 768]" = torch.ops.aten.mm.default(view_309, permute_131);  permute_131 = None
    permute_132: "f32[768, 1024]" = torch.ops.aten.permute.default(view_146, [1, 0]);  view_146 = None
    mm_34: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_132, view_309);  permute_132 = None
    sum_64: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_309, [0], True);  view_309 = None
    view_310: "f32[2304]" = torch.ops.aten.view.default(sum_64, [2304]);  sum_64 = None
    view_311: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_33, [1, 1024, 768]);  mm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    sub_70: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_64, getitem_107);  add_64 = getitem_107 = None
    mul_222: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_16);  sub_70 = None
    mul_223: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_311, primals_131);  primals_131 = None
    mul_224: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_223, 768)
    sum_65: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_223, [2], True)
    mul_225: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_223, mul_222);  mul_223 = None
    sum_66: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_225, [2], True);  mul_225 = None
    mul_226: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_222, sum_66);  sum_66 = None
    sub_71: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_224, sum_65);  mul_224 = sum_65 = None
    sub_72: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_71, mul_226);  sub_71 = mul_226 = None
    div_36: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
    mul_227: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_36, sub_72);  div_36 = sub_72 = None
    mul_228: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_311, mul_222);  mul_222 = None
    sum_67: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_228, [0, 1]);  mul_228 = None
    sum_68: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_311, [0, 1]);  view_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    add_123: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_120, mul_227);  add_120 = mul_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_13: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_105, torch.float32);  getitem_105 = None
    mul_229: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
    mul_230: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_123, mul_229);  mul_229 = None
    clone_36: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_230, memory_format = torch.contiguous_format);  mul_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_312: "f32[1024, 768]" = torch.ops.aten.view.default(clone_36, [1024, 768]);  clone_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_133: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
    mm_35: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_312, permute_133);  permute_133 = None
    permute_134: "f32[3072, 1024]" = torch.ops.aten.permute.default(view_144, [1, 0]);  view_144 = None
    mm_36: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_134, view_312);  permute_134 = None
    sum_69: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_312, [0], True);  view_312 = None
    view_313: "f32[768]" = torch.ops.aten.view.default(sum_69, [768]);  sum_69 = None
    view_314: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(mm_35, [1, 1024, 3072]);  mm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_231: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_314, mul_60);  mul_60 = None
    mul_232: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_314, add_63);  view_314 = add_63 = None
    alias_32: "f32[1, 1024, 3072]" = torch.ops.aten.alias.default(alias_15);  alias_15 = None
    mul_233: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(alias_32, alias_32);  alias_32 = None
    sub_73: "f32[1, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_233);  mul_233 = None
    mul_234: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_231, sub_73);  mul_231 = sub_73 = None
    mul_235: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_234, 0.7978845608028654);  mul_234 = None
    mul_236: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_235, 0.044715)
    pow_17: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_143, 2.0);  view_143 = None
    mul_237: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_17, 3.0);  pow_17 = None
    mul_238: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_236, mul_237);  mul_236 = mul_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_124: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_235, mul_238);  mul_235 = mul_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_239: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_232, 0.5);  mul_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_125: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(add_124, mul_239);  add_124 = mul_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_315: "f32[1024, 3072]" = torch.ops.aten.view.default(add_125, [1024, 3072]);  add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_135: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_62, [1, 0]);  primals_62 = None
    mm_37: "f32[1024, 768]" = torch.ops.aten.mm.default(view_315, permute_135);  permute_135 = None
    permute_136: "f32[768, 1024]" = torch.ops.aten.permute.default(view_142, [1, 0]);  view_142 = None
    mm_38: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_136, view_315);  permute_136 = None
    sum_70: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_315, [0], True);  view_315 = None
    view_316: "f32[3072]" = torch.ops.aten.view.default(sum_70, [3072]);  sum_70 = None
    view_317: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_37, [1, 1024, 768]);  mm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    sub_74: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_59, getitem_103);  add_59 = getitem_103 = None
    mul_240: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_15);  sub_74 = None
    mul_241: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_317, primals_129);  primals_129 = None
    mul_242: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_241, 768)
    sum_71: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_241, [2], True)
    mul_243: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_241, mul_240);  mul_241 = None
    sum_72: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_243, [2], True);  mul_243 = None
    mul_244: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_240, sum_72);  sum_72 = None
    sub_75: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_242, sum_71);  mul_242 = sum_71 = None
    sub_76: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_75, mul_244);  sub_75 = mul_244 = None
    div_37: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
    mul_245: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_37, sub_76);  div_37 = sub_76 = None
    mul_246: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_317, mul_240);  mul_240 = None
    sum_73: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_246, [0, 1]);  mul_246 = None
    sum_74: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_317, [0, 1]);  view_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    add_126: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_123, mul_245);  add_123 = mul_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    convert_element_type_14: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_101, torch.float32);  getitem_101 = None
    mul_247: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 1.1111111111111112);  convert_element_type_14 = None
    mul_248: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_126, mul_247);  mul_247 = None
    clone_37: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_248, memory_format = torch.contiguous_format);  mul_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_318: "f32[1024, 768]" = torch.ops.aten.view.default(clone_37, [1024, 768]);  clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_137: "f32[768, 768]" = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
    mm_39: "f32[1024, 768]" = torch.ops.aten.mm.default(view_318, permute_137);  permute_137 = None
    permute_138: "f32[768, 1024]" = torch.ops.aten.permute.default(view_140, [1, 0]);  view_140 = None
    mm_40: "f32[768, 768]" = torch.ops.aten.mm.default(permute_138, view_318);  permute_138 = None
    sum_75: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_318, [0], True);  view_318 = None
    view_319: "f32[768]" = torch.ops.aten.view.default(sum_75, [768]);  sum_75 = None
    view_320: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_39, [1, 1024, 768]);  mm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_321: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(view_320, [1, 1024, 12, 64]);  view_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_139: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_321, [0, 2, 1, 3]);  view_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_322: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_139, [12, 1024, 64]);  permute_139 = None
    permute_140: "f32[12, 1024, 1024]" = torch.ops.aten.permute.default(view_136, [0, 2, 1]);  view_136 = None
    bmm_40: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(permute_140, view_322);  permute_140 = None
    permute_141: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_137, [0, 2, 1]);  view_137 = None
    bmm_41: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_322, permute_141);  view_322 = permute_141 = None
    view_323: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_40, [1, 12, 1024, 64]);  bmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    add_127: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_17, view_323);  tangents_17 = view_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_324: "f32[1, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_41, [1, 12, 1024, 1024]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    convert_element_type_15: "f32[1, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_99, torch.float32);  getitem_99 = None
    mul_249: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_15, 1.1111111111111112);  convert_element_type_15 = None
    mul_250: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_324, mul_249);  view_324 = mul_249 = None
    clone_38: "f32[1, 12, 1024, 1024]" = torch.ops.aten.clone.default(mul_250, memory_format = torch.contiguous_format);  mul_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_33: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(alias_14);  alias_14 = None
    mul_251: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(clone_38, alias_33);  clone_38 = None
    sum_76: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_251, [-1], True)
    mul_252: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_33, sum_76);  alias_33 = sum_76 = None
    sub_77: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_251, mul_252);  mul_251 = mul_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_16: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_16, sub_77, scalar_tensor_4);  slice_16 = sub_77 = scalar_tensor_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_38: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(where_16, full_14);  where_16 = full_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_325: "f32[12, 1024, 1024]" = torch.ops.aten.view.default(div_38, [12, 1024, 1024]);  div_38 = None
    permute_142: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_133, [0, 2, 1]);  view_133 = None
    bmm_42: "f32[12, 64, 1024]" = torch.ops.aten.bmm.default(permute_142, view_325);  permute_142 = None
    permute_143: "f32[12, 1024, 64]" = torch.ops.aten.permute.default(view_134, [0, 2, 1]);  view_134 = None
    bmm_43: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_325, permute_143);  view_325 = permute_143 = None
    view_326: "f32[1, 12, 64, 1024]" = torch.ops.aten.view.default(bmm_42, [1, 12, 64, 1024]);  bmm_42 = None
    view_327: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_43, [1, 12, 1024, 64]);  bmm_43 = None
    permute_144: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_326, [0, 1, 3, 2]);  view_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_128: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_16, permute_144);  tangents_16 = permute_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_145: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_127, [0, 2, 1, 3]);  add_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_39: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_145, memory_format = torch.contiguous_format);  permute_145 = None
    view_328: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_39, [1, 1024, 768]);  clone_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_146: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_128, [0, 2, 1, 3]);  add_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_40: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_146, memory_format = torch.contiguous_format);  permute_146 = None
    view_329: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_40, [1, 1024, 768]);  clone_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_147: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_327, [0, 2, 1, 3]);  view_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_41: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_147, memory_format = torch.contiguous_format);  permute_147 = None
    view_330: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_41, [1, 1024, 768]);  clone_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    cat_4: "f32[1, 1024, 2304]" = torch.ops.aten.cat.default([view_330, view_329, view_328], 2);  view_330 = view_329 = view_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_331: "f32[1024, 2304]" = torch.ops.aten.view.default(cat_4, [1024, 2304]);  cat_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_148: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
    mm_41: "f32[1024, 768]" = torch.ops.aten.mm.default(view_331, permute_148);  permute_148 = None
    permute_149: "f32[768, 1024]" = torch.ops.aten.permute.default(view_128, [1, 0]);  view_128 = None
    mm_42: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_149, view_331);  permute_149 = None
    sum_77: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_331, [0], True);  view_331 = None
    view_332: "f32[2304]" = torch.ops.aten.view.default(sum_77, [2304]);  sum_77 = None
    view_333: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_41, [1, 1024, 768]);  mm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    sub_78: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_56, getitem_94);  add_56 = getitem_94 = None
    mul_253: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_14);  sub_78 = None
    mul_254: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_333, primals_127);  primals_127 = None
    mul_255: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_254, 768)
    sum_78: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_254, [2], True)
    mul_256: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_254, mul_253);  mul_254 = None
    sum_79: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_256, [2], True);  mul_256 = None
    mul_257: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_253, sum_79);  sum_79 = None
    sub_79: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_255, sum_78);  mul_255 = sum_78 = None
    sub_80: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_79, mul_257);  sub_79 = mul_257 = None
    div_39: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
    mul_258: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_39, sub_80);  div_39 = sub_80 = None
    mul_259: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_333, mul_253);  mul_253 = None
    sum_80: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_259, [0, 1]);  mul_259 = None
    sum_81: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_333, [0, 1]);  view_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    add_129: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_126, mul_258);  add_126 = mul_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_16: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_92, torch.float32);  getitem_92 = None
    mul_260: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_16, 1.1111111111111112);  convert_element_type_16 = None
    mul_261: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_129, mul_260);  mul_260 = None
    clone_42: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_261, memory_format = torch.contiguous_format);  mul_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_334: "f32[1024, 768]" = torch.ops.aten.view.default(clone_42, [1024, 768]);  clone_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_150: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
    mm_43: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_334, permute_150);  permute_150 = None
    permute_151: "f32[3072, 1024]" = torch.ops.aten.permute.default(view_126, [1, 0]);  view_126 = None
    mm_44: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_151, view_334);  permute_151 = None
    sum_82: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_334, [0], True);  view_334 = None
    view_335: "f32[768]" = torch.ops.aten.view.default(sum_82, [768]);  sum_82 = None
    view_336: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(mm_43, [1, 1024, 3072]);  mm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_262: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_336, mul_52);  mul_52 = None
    mul_263: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_336, add_55);  view_336 = add_55 = None
    alias_34: "f32[1, 1024, 3072]" = torch.ops.aten.alias.default(alias_13);  alias_13 = None
    mul_264: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(alias_34, alias_34);  alias_34 = None
    sub_81: "f32[1, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_264);  mul_264 = None
    mul_265: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_262, sub_81);  mul_262 = sub_81 = None
    mul_266: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_265, 0.7978845608028654);  mul_265 = None
    mul_267: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_266, 0.044715)
    pow_18: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_125, 2.0);  view_125 = None
    mul_268: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_18, 3.0);  pow_18 = None
    mul_269: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_267, mul_268);  mul_267 = mul_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_130: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_266, mul_269);  mul_266 = mul_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_270: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_263, 0.5);  mul_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_131: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(add_130, mul_270);  add_130 = mul_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_337: "f32[1024, 3072]" = torch.ops.aten.view.default(add_131, [1024, 3072]);  add_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_152: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_54, [1, 0]);  primals_54 = None
    mm_45: "f32[1024, 768]" = torch.ops.aten.mm.default(view_337, permute_152);  permute_152 = None
    permute_153: "f32[768, 1024]" = torch.ops.aten.permute.default(view_124, [1, 0]);  view_124 = None
    mm_46: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_153, view_337);  permute_153 = None
    sum_83: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_337, [0], True);  view_337 = None
    view_338: "f32[3072]" = torch.ops.aten.view.default(sum_83, [3072]);  sum_83 = None
    view_339: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_45, [1, 1024, 768]);  mm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    sub_82: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_51, getitem_90);  add_51 = getitem_90 = None
    mul_271: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_13);  sub_82 = None
    mul_272: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_339, primals_125);  primals_125 = None
    mul_273: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_272, 768)
    sum_84: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_272, [2], True)
    mul_274: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_272, mul_271);  mul_272 = None
    sum_85: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_274, [2], True);  mul_274 = None
    mul_275: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_271, sum_85);  sum_85 = None
    sub_83: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_273, sum_84);  mul_273 = sum_84 = None
    sub_84: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_83, mul_275);  sub_83 = mul_275 = None
    div_40: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
    mul_276: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_40, sub_84);  div_40 = sub_84 = None
    mul_277: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_339, mul_271);  mul_271 = None
    sum_86: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_277, [0, 1]);  mul_277 = None
    sum_87: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_339, [0, 1]);  view_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    add_132: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_129, mul_276);  add_129 = mul_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    convert_element_type_17: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_88, torch.float32);  getitem_88 = None
    mul_278: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_17, 1.1111111111111112);  convert_element_type_17 = None
    mul_279: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_132, mul_278);  mul_278 = None
    clone_43: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_279, memory_format = torch.contiguous_format);  mul_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_340: "f32[1024, 768]" = torch.ops.aten.view.default(clone_43, [1024, 768]);  clone_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_154: "f32[768, 768]" = torch.ops.aten.permute.default(primals_52, [1, 0]);  primals_52 = None
    mm_47: "f32[1024, 768]" = torch.ops.aten.mm.default(view_340, permute_154);  permute_154 = None
    permute_155: "f32[768, 1024]" = torch.ops.aten.permute.default(view_122, [1, 0]);  view_122 = None
    mm_48: "f32[768, 768]" = torch.ops.aten.mm.default(permute_155, view_340);  permute_155 = None
    sum_88: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_340, [0], True);  view_340 = None
    view_341: "f32[768]" = torch.ops.aten.view.default(sum_88, [768]);  sum_88 = None
    view_342: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_47, [1, 1024, 768]);  mm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_343: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(view_342, [1, 1024, 12, 64]);  view_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_156: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_343, [0, 2, 1, 3]);  view_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_344: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_156, [12, 1024, 64]);  permute_156 = None
    permute_157: "f32[12, 1024, 1024]" = torch.ops.aten.permute.default(view_118, [0, 2, 1]);  view_118 = None
    bmm_44: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(permute_157, view_344);  permute_157 = None
    permute_158: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_119, [0, 2, 1]);  view_119 = None
    bmm_45: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_344, permute_158);  view_344 = permute_158 = None
    view_345: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_44, [1, 12, 1024, 64]);  bmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    add_133: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_15, view_345);  tangents_15 = view_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_346: "f32[1, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_45, [1, 12, 1024, 1024]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    convert_element_type_18: "f32[1, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_86, torch.float32);  getitem_86 = None
    mul_280: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_18, 1.1111111111111112);  convert_element_type_18 = None
    mul_281: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_346, mul_280);  view_346 = mul_280 = None
    clone_44: "f32[1, 12, 1024, 1024]" = torch.ops.aten.clone.default(mul_281, memory_format = torch.contiguous_format);  mul_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_35: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    mul_282: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(clone_44, alias_35);  clone_44 = None
    sum_89: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_282, [-1], True)
    mul_283: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_35, sum_89);  alias_35 = sum_89 = None
    sub_85: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_282, mul_283);  mul_282 = mul_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_17: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_14, sub_85, scalar_tensor_5);  slice_14 = sub_85 = scalar_tensor_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_41: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(where_17, full_12);  where_17 = full_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_347: "f32[12, 1024, 1024]" = torch.ops.aten.view.default(div_41, [12, 1024, 1024]);  div_41 = None
    permute_159: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_115, [0, 2, 1]);  view_115 = None
    bmm_46: "f32[12, 64, 1024]" = torch.ops.aten.bmm.default(permute_159, view_347);  permute_159 = None
    permute_160: "f32[12, 1024, 64]" = torch.ops.aten.permute.default(view_116, [0, 2, 1]);  view_116 = None
    bmm_47: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_347, permute_160);  view_347 = permute_160 = None
    view_348: "f32[1, 12, 64, 1024]" = torch.ops.aten.view.default(bmm_46, [1, 12, 64, 1024]);  bmm_46 = None
    view_349: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_47, [1, 12, 1024, 64]);  bmm_47 = None
    permute_161: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_348, [0, 1, 3, 2]);  view_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_134: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_14, permute_161);  tangents_14 = permute_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_162: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_133, [0, 2, 1, 3]);  add_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_45: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_162, memory_format = torch.contiguous_format);  permute_162 = None
    view_350: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_45, [1, 1024, 768]);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_163: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_134, [0, 2, 1, 3]);  add_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_46: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_163, memory_format = torch.contiguous_format);  permute_163 = None
    view_351: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_46, [1, 1024, 768]);  clone_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_164: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_349, [0, 2, 1, 3]);  view_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_47: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_164, memory_format = torch.contiguous_format);  permute_164 = None
    view_352: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_47, [1, 1024, 768]);  clone_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    cat_5: "f32[1, 1024, 2304]" = torch.ops.aten.cat.default([view_352, view_351, view_350], 2);  view_352 = view_351 = view_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_353: "f32[1024, 2304]" = torch.ops.aten.view.default(cat_5, [1024, 2304]);  cat_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_165: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_50, [1, 0]);  primals_50 = None
    mm_49: "f32[1024, 768]" = torch.ops.aten.mm.default(view_353, permute_165);  permute_165 = None
    permute_166: "f32[768, 1024]" = torch.ops.aten.permute.default(view_110, [1, 0]);  view_110 = None
    mm_50: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_166, view_353);  permute_166 = None
    sum_90: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_353, [0], True);  view_353 = None
    view_354: "f32[2304]" = torch.ops.aten.view.default(sum_90, [2304]);  sum_90 = None
    view_355: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_49, [1, 1024, 768]);  mm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    sub_86: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_48, getitem_81);  add_48 = getitem_81 = None
    mul_284: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_12);  sub_86 = None
    mul_285: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_355, primals_123);  primals_123 = None
    mul_286: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_285, 768)
    sum_91: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_285, [2], True)
    mul_287: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_285, mul_284);  mul_285 = None
    sum_92: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_287, [2], True);  mul_287 = None
    mul_288: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_284, sum_92);  sum_92 = None
    sub_87: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_286, sum_91);  mul_286 = sum_91 = None
    sub_88: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_87, mul_288);  sub_87 = mul_288 = None
    div_42: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
    mul_289: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_42, sub_88);  div_42 = sub_88 = None
    mul_290: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_355, mul_284);  mul_284 = None
    sum_93: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_290, [0, 1]);  mul_290 = None
    sum_94: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_355, [0, 1]);  view_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    add_135: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_132, mul_289);  add_132 = mul_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_19: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_79, torch.float32);  getitem_79 = None
    mul_291: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_19, 1.1111111111111112);  convert_element_type_19 = None
    mul_292: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_135, mul_291);  mul_291 = None
    clone_48: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_292, memory_format = torch.contiguous_format);  mul_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_356: "f32[1024, 768]" = torch.ops.aten.view.default(clone_48, [1024, 768]);  clone_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_167: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_48, [1, 0]);  primals_48 = None
    mm_51: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_356, permute_167);  permute_167 = None
    permute_168: "f32[3072, 1024]" = torch.ops.aten.permute.default(view_108, [1, 0]);  view_108 = None
    mm_52: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_168, view_356);  permute_168 = None
    sum_95: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_356, [0], True);  view_356 = None
    view_357: "f32[768]" = torch.ops.aten.view.default(sum_95, [768]);  sum_95 = None
    view_358: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(mm_51, [1, 1024, 3072]);  mm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_293: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_358, mul_44);  mul_44 = None
    mul_294: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_358, add_47);  view_358 = add_47 = None
    alias_36: "f32[1, 1024, 3072]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    mul_295: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(alias_36, alias_36);  alias_36 = None
    sub_89: "f32[1, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_295);  mul_295 = None
    mul_296: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_293, sub_89);  mul_293 = sub_89 = None
    mul_297: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_296, 0.7978845608028654);  mul_296 = None
    mul_298: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_297, 0.044715)
    pow_19: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_107, 2.0);  view_107 = None
    mul_299: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_19, 3.0);  pow_19 = None
    mul_300: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_298, mul_299);  mul_298 = mul_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_136: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_297, mul_300);  mul_297 = mul_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_301: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_294, 0.5);  mul_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_137: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(add_136, mul_301);  add_136 = mul_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_359: "f32[1024, 3072]" = torch.ops.aten.view.default(add_137, [1024, 3072]);  add_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_169: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_46, [1, 0]);  primals_46 = None
    mm_53: "f32[1024, 768]" = torch.ops.aten.mm.default(view_359, permute_169);  permute_169 = None
    permute_170: "f32[768, 1024]" = torch.ops.aten.permute.default(view_106, [1, 0]);  view_106 = None
    mm_54: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_170, view_359);  permute_170 = None
    sum_96: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_359, [0], True);  view_359 = None
    view_360: "f32[3072]" = torch.ops.aten.view.default(sum_96, [3072]);  sum_96 = None
    view_361: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_53, [1, 1024, 768]);  mm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    sub_90: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_43, getitem_77);  add_43 = getitem_77 = None
    mul_302: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_90, rsqrt_11);  sub_90 = None
    mul_303: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_361, primals_121);  primals_121 = None
    mul_304: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_303, 768)
    sum_97: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_303, [2], True)
    mul_305: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_303, mul_302);  mul_303 = None
    sum_98: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_305, [2], True);  mul_305 = None
    mul_306: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_302, sum_98);  sum_98 = None
    sub_91: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_304, sum_97);  mul_304 = sum_97 = None
    sub_92: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_91, mul_306);  sub_91 = mul_306 = None
    div_43: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
    mul_307: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_43, sub_92);  div_43 = sub_92 = None
    mul_308: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_361, mul_302);  mul_302 = None
    sum_99: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_308, [0, 1]);  mul_308 = None
    sum_100: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_361, [0, 1]);  view_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    add_138: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_135, mul_307);  add_135 = mul_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    convert_element_type_20: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_75, torch.float32);  getitem_75 = None
    mul_309: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.1111111111111112);  convert_element_type_20 = None
    mul_310: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_138, mul_309);  mul_309 = None
    clone_49: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_310, memory_format = torch.contiguous_format);  mul_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_362: "f32[1024, 768]" = torch.ops.aten.view.default(clone_49, [1024, 768]);  clone_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_171: "f32[768, 768]" = torch.ops.aten.permute.default(primals_44, [1, 0]);  primals_44 = None
    mm_55: "f32[1024, 768]" = torch.ops.aten.mm.default(view_362, permute_171);  permute_171 = None
    permute_172: "f32[768, 1024]" = torch.ops.aten.permute.default(view_104, [1, 0]);  view_104 = None
    mm_56: "f32[768, 768]" = torch.ops.aten.mm.default(permute_172, view_362);  permute_172 = None
    sum_101: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_362, [0], True);  view_362 = None
    view_363: "f32[768]" = torch.ops.aten.view.default(sum_101, [768]);  sum_101 = None
    view_364: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_55, [1, 1024, 768]);  mm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_365: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(view_364, [1, 1024, 12, 64]);  view_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_173: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_365, [0, 2, 1, 3]);  view_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_366: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_173, [12, 1024, 64]);  permute_173 = None
    permute_174: "f32[12, 1024, 1024]" = torch.ops.aten.permute.default(view_100, [0, 2, 1]);  view_100 = None
    bmm_48: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(permute_174, view_366);  permute_174 = None
    permute_175: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_101, [0, 2, 1]);  view_101 = None
    bmm_49: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_366, permute_175);  view_366 = permute_175 = None
    view_367: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_48, [1, 12, 1024, 64]);  bmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    add_139: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_13, view_367);  tangents_13 = view_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_368: "f32[1, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_49, [1, 12, 1024, 1024]);  bmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    convert_element_type_21: "f32[1, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_73, torch.float32);  getitem_73 = None
    mul_311: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 1.1111111111111112);  convert_element_type_21 = None
    mul_312: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_368, mul_311);  view_368 = mul_311 = None
    clone_50: "f32[1, 12, 1024, 1024]" = torch.ops.aten.clone.default(mul_312, memory_format = torch.contiguous_format);  mul_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_37: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    mul_313: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(clone_50, alias_37);  clone_50 = None
    sum_102: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_313, [-1], True)
    mul_314: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_37, sum_102);  alias_37 = sum_102 = None
    sub_93: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_313, mul_314);  mul_313 = mul_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    scalar_tensor_6: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_18: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_12, sub_93, scalar_tensor_6);  slice_12 = sub_93 = scalar_tensor_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_44: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(where_18, full_10);  where_18 = full_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_369: "f32[12, 1024, 1024]" = torch.ops.aten.view.default(div_44, [12, 1024, 1024]);  div_44 = None
    permute_176: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_97, [0, 2, 1]);  view_97 = None
    bmm_50: "f32[12, 64, 1024]" = torch.ops.aten.bmm.default(permute_176, view_369);  permute_176 = None
    permute_177: "f32[12, 1024, 64]" = torch.ops.aten.permute.default(view_98, [0, 2, 1]);  view_98 = None
    bmm_51: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_369, permute_177);  view_369 = permute_177 = None
    view_370: "f32[1, 12, 64, 1024]" = torch.ops.aten.view.default(bmm_50, [1, 12, 64, 1024]);  bmm_50 = None
    view_371: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_51, [1, 12, 1024, 64]);  bmm_51 = None
    permute_178: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_370, [0, 1, 3, 2]);  view_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_140: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_12, permute_178);  tangents_12 = permute_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_179: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_139, [0, 2, 1, 3]);  add_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_51: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_179, memory_format = torch.contiguous_format);  permute_179 = None
    view_372: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_51, [1, 1024, 768]);  clone_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_180: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_140, [0, 2, 1, 3]);  add_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_52: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_180, memory_format = torch.contiguous_format);  permute_180 = None
    view_373: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_52, [1, 1024, 768]);  clone_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_181: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_371, [0, 2, 1, 3]);  view_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_53: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_181, memory_format = torch.contiguous_format);  permute_181 = None
    view_374: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_53, [1, 1024, 768]);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    cat_6: "f32[1, 1024, 2304]" = torch.ops.aten.cat.default([view_374, view_373, view_372], 2);  view_374 = view_373 = view_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_375: "f32[1024, 2304]" = torch.ops.aten.view.default(cat_6, [1024, 2304]);  cat_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_182: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
    mm_57: "f32[1024, 768]" = torch.ops.aten.mm.default(view_375, permute_182);  permute_182 = None
    permute_183: "f32[768, 1024]" = torch.ops.aten.permute.default(view_92, [1, 0]);  view_92 = None
    mm_58: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_183, view_375);  permute_183 = None
    sum_103: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_375, [0], True);  view_375 = None
    view_376: "f32[2304]" = torch.ops.aten.view.default(sum_103, [2304]);  sum_103 = None
    view_377: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_57, [1, 1024, 768]);  mm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    sub_94: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_40, getitem_68);  add_40 = getitem_68 = None
    mul_315: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_94, rsqrt_10);  sub_94 = None
    mul_316: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_377, primals_119);  primals_119 = None
    mul_317: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_316, 768)
    sum_104: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_316, [2], True)
    mul_318: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_316, mul_315);  mul_316 = None
    sum_105: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_318, [2], True);  mul_318 = None
    mul_319: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_315, sum_105);  sum_105 = None
    sub_95: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_317, sum_104);  mul_317 = sum_104 = None
    sub_96: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_95, mul_319);  sub_95 = mul_319 = None
    div_45: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
    mul_320: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_45, sub_96);  div_45 = sub_96 = None
    mul_321: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_377, mul_315);  mul_315 = None
    sum_106: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_321, [0, 1]);  mul_321 = None
    sum_107: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_377, [0, 1]);  view_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    add_141: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_138, mul_320);  add_138 = mul_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_22: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_66, torch.float32);  getitem_66 = None
    mul_322: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 1.1111111111111112);  convert_element_type_22 = None
    mul_323: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_141, mul_322);  mul_322 = None
    clone_54: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_323, memory_format = torch.contiguous_format);  mul_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_378: "f32[1024, 768]" = torch.ops.aten.view.default(clone_54, [1024, 768]);  clone_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_184: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_40, [1, 0]);  primals_40 = None
    mm_59: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_378, permute_184);  permute_184 = None
    permute_185: "f32[3072, 1024]" = torch.ops.aten.permute.default(view_90, [1, 0]);  view_90 = None
    mm_60: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_185, view_378);  permute_185 = None
    sum_108: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_378, [0], True);  view_378 = None
    view_379: "f32[768]" = torch.ops.aten.view.default(sum_108, [768]);  sum_108 = None
    view_380: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(mm_59, [1, 1024, 3072]);  mm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_324: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_380, mul_36);  mul_36 = None
    mul_325: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_380, add_39);  view_380 = add_39 = None
    alias_38: "f32[1, 1024, 3072]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    mul_326: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(alias_38, alias_38);  alias_38 = None
    sub_97: "f32[1, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_326);  mul_326 = None
    mul_327: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_324, sub_97);  mul_324 = sub_97 = None
    mul_328: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_327, 0.7978845608028654);  mul_327 = None
    mul_329: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_328, 0.044715)
    pow_20: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_89, 2.0);  view_89 = None
    mul_330: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_20, 3.0);  pow_20 = None
    mul_331: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_329, mul_330);  mul_329 = mul_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_142: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_328, mul_331);  mul_328 = mul_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_332: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_325, 0.5);  mul_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_143: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(add_142, mul_332);  add_142 = mul_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_381: "f32[1024, 3072]" = torch.ops.aten.view.default(add_143, [1024, 3072]);  add_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_186: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_38, [1, 0]);  primals_38 = None
    mm_61: "f32[1024, 768]" = torch.ops.aten.mm.default(view_381, permute_186);  permute_186 = None
    permute_187: "f32[768, 1024]" = torch.ops.aten.permute.default(view_88, [1, 0]);  view_88 = None
    mm_62: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_187, view_381);  permute_187 = None
    sum_109: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_381, [0], True);  view_381 = None
    view_382: "f32[3072]" = torch.ops.aten.view.default(sum_109, [3072]);  sum_109 = None
    view_383: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_61, [1, 1024, 768]);  mm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    sub_98: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_64);  add_35 = getitem_64 = None
    mul_333: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_98, rsqrt_9);  sub_98 = None
    mul_334: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_383, primals_117);  primals_117 = None
    mul_335: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_334, 768)
    sum_110: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_334, [2], True)
    mul_336: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_334, mul_333);  mul_334 = None
    sum_111: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_336, [2], True);  mul_336 = None
    mul_337: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_333, sum_111);  sum_111 = None
    sub_99: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_335, sum_110);  mul_335 = sum_110 = None
    sub_100: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_99, mul_337);  sub_99 = mul_337 = None
    div_46: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
    mul_338: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_46, sub_100);  div_46 = sub_100 = None
    mul_339: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_383, mul_333);  mul_333 = None
    sum_112: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_339, [0, 1]);  mul_339 = None
    sum_113: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_383, [0, 1]);  view_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    add_144: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_141, mul_338);  add_141 = mul_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    convert_element_type_23: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_62, torch.float32);  getitem_62 = None
    mul_340: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_23, 1.1111111111111112);  convert_element_type_23 = None
    mul_341: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_144, mul_340);  mul_340 = None
    clone_55: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_341, memory_format = torch.contiguous_format);  mul_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_384: "f32[1024, 768]" = torch.ops.aten.view.default(clone_55, [1024, 768]);  clone_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_188: "f32[768, 768]" = torch.ops.aten.permute.default(primals_36, [1, 0]);  primals_36 = None
    mm_63: "f32[1024, 768]" = torch.ops.aten.mm.default(view_384, permute_188);  permute_188 = None
    permute_189: "f32[768, 1024]" = torch.ops.aten.permute.default(view_86, [1, 0]);  view_86 = None
    mm_64: "f32[768, 768]" = torch.ops.aten.mm.default(permute_189, view_384);  permute_189 = None
    sum_114: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_384, [0], True);  view_384 = None
    view_385: "f32[768]" = torch.ops.aten.view.default(sum_114, [768]);  sum_114 = None
    view_386: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_63, [1, 1024, 768]);  mm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_387: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(view_386, [1, 1024, 12, 64]);  view_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_190: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_387, [0, 2, 1, 3]);  view_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_388: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_190, [12, 1024, 64]);  permute_190 = None
    permute_191: "f32[12, 1024, 1024]" = torch.ops.aten.permute.default(view_82, [0, 2, 1]);  view_82 = None
    bmm_52: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(permute_191, view_388);  permute_191 = None
    permute_192: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_83, [0, 2, 1]);  view_83 = None
    bmm_53: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_388, permute_192);  view_388 = permute_192 = None
    view_389: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_52, [1, 12, 1024, 64]);  bmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    add_145: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_11, view_389);  tangents_11 = view_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_390: "f32[1, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_53, [1, 12, 1024, 1024]);  bmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    convert_element_type_24: "f32[1, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_60, torch.float32);  getitem_60 = None
    mul_342: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 1.1111111111111112);  convert_element_type_24 = None
    mul_343: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_390, mul_342);  view_390 = mul_342 = None
    clone_56: "f32[1, 12, 1024, 1024]" = torch.ops.aten.clone.default(mul_343, memory_format = torch.contiguous_format);  mul_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_39: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    mul_344: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(clone_56, alias_39);  clone_56 = None
    sum_115: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_344, [-1], True)
    mul_345: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_39, sum_115);  alias_39 = sum_115 = None
    sub_101: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_344, mul_345);  mul_344 = mul_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    scalar_tensor_7: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_19: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_10, sub_101, scalar_tensor_7);  slice_10 = sub_101 = scalar_tensor_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_47: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(where_19, full_8);  where_19 = full_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_391: "f32[12, 1024, 1024]" = torch.ops.aten.view.default(div_47, [12, 1024, 1024]);  div_47 = None
    permute_193: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_79, [0, 2, 1]);  view_79 = None
    bmm_54: "f32[12, 64, 1024]" = torch.ops.aten.bmm.default(permute_193, view_391);  permute_193 = None
    permute_194: "f32[12, 1024, 64]" = torch.ops.aten.permute.default(view_80, [0, 2, 1]);  view_80 = None
    bmm_55: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_391, permute_194);  view_391 = permute_194 = None
    view_392: "f32[1, 12, 64, 1024]" = torch.ops.aten.view.default(bmm_54, [1, 12, 64, 1024]);  bmm_54 = None
    view_393: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_55, [1, 12, 1024, 64]);  bmm_55 = None
    permute_195: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_392, [0, 1, 3, 2]);  view_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_146: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_10, permute_195);  tangents_10 = permute_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_196: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_145, [0, 2, 1, 3]);  add_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_57: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_196, memory_format = torch.contiguous_format);  permute_196 = None
    view_394: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_57, [1, 1024, 768]);  clone_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_197: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_146, [0, 2, 1, 3]);  add_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_58: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_197, memory_format = torch.contiguous_format);  permute_197 = None
    view_395: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_58, [1, 1024, 768]);  clone_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_198: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_393, [0, 2, 1, 3]);  view_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_59: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_198, memory_format = torch.contiguous_format);  permute_198 = None
    view_396: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_59, [1, 1024, 768]);  clone_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    cat_7: "f32[1, 1024, 2304]" = torch.ops.aten.cat.default([view_396, view_395, view_394], 2);  view_396 = view_395 = view_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_397: "f32[1024, 2304]" = torch.ops.aten.view.default(cat_7, [1024, 2304]);  cat_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_199: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
    mm_65: "f32[1024, 768]" = torch.ops.aten.mm.default(view_397, permute_199);  permute_199 = None
    permute_200: "f32[768, 1024]" = torch.ops.aten.permute.default(view_74, [1, 0]);  view_74 = None
    mm_66: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_200, view_397);  permute_200 = None
    sum_116: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_397, [0], True);  view_397 = None
    view_398: "f32[2304]" = torch.ops.aten.view.default(sum_116, [2304]);  sum_116 = None
    view_399: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_65, [1, 1024, 768]);  mm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    sub_102: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_32, getitem_55);  add_32 = getitem_55 = None
    mul_346: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_102, rsqrt_8);  sub_102 = None
    mul_347: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_399, primals_115);  primals_115 = None
    mul_348: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_347, 768)
    sum_117: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_347, [2], True)
    mul_349: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_347, mul_346);  mul_347 = None
    sum_118: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_349, [2], True);  mul_349 = None
    mul_350: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_346, sum_118);  sum_118 = None
    sub_103: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_348, sum_117);  mul_348 = sum_117 = None
    sub_104: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_103, mul_350);  sub_103 = mul_350 = None
    div_48: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
    mul_351: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_48, sub_104);  div_48 = sub_104 = None
    mul_352: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_399, mul_346);  mul_346 = None
    sum_119: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_352, [0, 1]);  mul_352 = None
    sum_120: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_399, [0, 1]);  view_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    add_147: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_144, mul_351);  add_144 = mul_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_25: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_53, torch.float32);  getitem_53 = None
    mul_353: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.1111111111111112);  convert_element_type_25 = None
    mul_354: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_147, mul_353);  mul_353 = None
    clone_60: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_354, memory_format = torch.contiguous_format);  mul_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_400: "f32[1024, 768]" = torch.ops.aten.view.default(clone_60, [1024, 768]);  clone_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_201: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_32, [1, 0]);  primals_32 = None
    mm_67: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_400, permute_201);  permute_201 = None
    permute_202: "f32[3072, 1024]" = torch.ops.aten.permute.default(view_72, [1, 0]);  view_72 = None
    mm_68: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_202, view_400);  permute_202 = None
    sum_121: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_400, [0], True);  view_400 = None
    view_401: "f32[768]" = torch.ops.aten.view.default(sum_121, [768]);  sum_121 = None
    view_402: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(mm_67, [1, 1024, 3072]);  mm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_355: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_402, mul_28);  mul_28 = None
    mul_356: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_402, add_31);  view_402 = add_31 = None
    alias_40: "f32[1, 1024, 3072]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    mul_357: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(alias_40, alias_40);  alias_40 = None
    sub_105: "f32[1, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_357);  mul_357 = None
    mul_358: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_355, sub_105);  mul_355 = sub_105 = None
    mul_359: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_358, 0.7978845608028654);  mul_358 = None
    mul_360: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_359, 0.044715)
    pow_21: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_71, 2.0);  view_71 = None
    mul_361: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_21, 3.0);  pow_21 = None
    mul_362: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_360, mul_361);  mul_360 = mul_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_148: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_359, mul_362);  mul_359 = mul_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_363: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_356, 0.5);  mul_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_149: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(add_148, mul_363);  add_148 = mul_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_403: "f32[1024, 3072]" = torch.ops.aten.view.default(add_149, [1024, 3072]);  add_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_203: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_30, [1, 0]);  primals_30 = None
    mm_69: "f32[1024, 768]" = torch.ops.aten.mm.default(view_403, permute_203);  permute_203 = None
    permute_204: "f32[768, 1024]" = torch.ops.aten.permute.default(view_70, [1, 0]);  view_70 = None
    mm_70: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_204, view_403);  permute_204 = None
    sum_122: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_403, [0], True);  view_403 = None
    view_404: "f32[3072]" = torch.ops.aten.view.default(sum_122, [3072]);  sum_122 = None
    view_405: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_69, [1, 1024, 768]);  mm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    sub_106: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_27, getitem_51);  add_27 = getitem_51 = None
    mul_364: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_106, rsqrt_7);  sub_106 = None
    mul_365: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_405, primals_113);  primals_113 = None
    mul_366: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_365, 768)
    sum_123: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_365, [2], True)
    mul_367: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_365, mul_364);  mul_365 = None
    sum_124: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_367, [2], True);  mul_367 = None
    mul_368: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_364, sum_124);  sum_124 = None
    sub_107: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_366, sum_123);  mul_366 = sum_123 = None
    sub_108: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_107, mul_368);  sub_107 = mul_368 = None
    div_49: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
    mul_369: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_49, sub_108);  div_49 = sub_108 = None
    mul_370: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_405, mul_364);  mul_364 = None
    sum_125: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_370, [0, 1]);  mul_370 = None
    sum_126: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_405, [0, 1]);  view_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    add_150: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_147, mul_369);  add_147 = mul_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    convert_element_type_26: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_49, torch.float32);  getitem_49 = None
    mul_371: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_26, 1.1111111111111112);  convert_element_type_26 = None
    mul_372: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_150, mul_371);  mul_371 = None
    clone_61: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_372, memory_format = torch.contiguous_format);  mul_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_406: "f32[1024, 768]" = torch.ops.aten.view.default(clone_61, [1024, 768]);  clone_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_205: "f32[768, 768]" = torch.ops.aten.permute.default(primals_28, [1, 0]);  primals_28 = None
    mm_71: "f32[1024, 768]" = torch.ops.aten.mm.default(view_406, permute_205);  permute_205 = None
    permute_206: "f32[768, 1024]" = torch.ops.aten.permute.default(view_68, [1, 0]);  view_68 = None
    mm_72: "f32[768, 768]" = torch.ops.aten.mm.default(permute_206, view_406);  permute_206 = None
    sum_127: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_406, [0], True);  view_406 = None
    view_407: "f32[768]" = torch.ops.aten.view.default(sum_127, [768]);  sum_127 = None
    view_408: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_71, [1, 1024, 768]);  mm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_409: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(view_408, [1, 1024, 12, 64]);  view_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_207: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_409, [0, 2, 1, 3]);  view_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_410: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_207, [12, 1024, 64]);  permute_207 = None
    permute_208: "f32[12, 1024, 1024]" = torch.ops.aten.permute.default(view_64, [0, 2, 1]);  view_64 = None
    bmm_56: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(permute_208, view_410);  permute_208 = None
    permute_209: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_65, [0, 2, 1]);  view_65 = None
    bmm_57: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_410, permute_209);  view_410 = permute_209 = None
    view_411: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_56, [1, 12, 1024, 64]);  bmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    add_151: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_9, view_411);  tangents_9 = view_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_412: "f32[1, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_57, [1, 12, 1024, 1024]);  bmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    convert_element_type_27: "f32[1, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_47, torch.float32);  getitem_47 = None
    mul_373: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_27, 1.1111111111111112);  convert_element_type_27 = None
    mul_374: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_412, mul_373);  view_412 = mul_373 = None
    clone_62: "f32[1, 12, 1024, 1024]" = torch.ops.aten.clone.default(mul_374, memory_format = torch.contiguous_format);  mul_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_41: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    mul_375: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(clone_62, alias_41);  clone_62 = None
    sum_128: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_375, [-1], True)
    mul_376: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_41, sum_128);  alias_41 = sum_128 = None
    sub_109: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_375, mul_376);  mul_375 = mul_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    scalar_tensor_8: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_20: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_8, sub_109, scalar_tensor_8);  slice_8 = sub_109 = scalar_tensor_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_50: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(where_20, full_6);  where_20 = full_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_413: "f32[12, 1024, 1024]" = torch.ops.aten.view.default(div_50, [12, 1024, 1024]);  div_50 = None
    permute_210: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_61, [0, 2, 1]);  view_61 = None
    bmm_58: "f32[12, 64, 1024]" = torch.ops.aten.bmm.default(permute_210, view_413);  permute_210 = None
    permute_211: "f32[12, 1024, 64]" = torch.ops.aten.permute.default(view_62, [0, 2, 1]);  view_62 = None
    bmm_59: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_413, permute_211);  view_413 = permute_211 = None
    view_414: "f32[1, 12, 64, 1024]" = torch.ops.aten.view.default(bmm_58, [1, 12, 64, 1024]);  bmm_58 = None
    view_415: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_59, [1, 12, 1024, 64]);  bmm_59 = None
    permute_212: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_414, [0, 1, 3, 2]);  view_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_152: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_8, permute_212);  tangents_8 = permute_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_213: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_151, [0, 2, 1, 3]);  add_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_63: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_213, memory_format = torch.contiguous_format);  permute_213 = None
    view_416: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_63, [1, 1024, 768]);  clone_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_214: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_152, [0, 2, 1, 3]);  add_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_64: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_214, memory_format = torch.contiguous_format);  permute_214 = None
    view_417: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_64, [1, 1024, 768]);  clone_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_215: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_415, [0, 2, 1, 3]);  view_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_65: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_215, memory_format = torch.contiguous_format);  permute_215 = None
    view_418: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_65, [1, 1024, 768]);  clone_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    cat_8: "f32[1, 1024, 2304]" = torch.ops.aten.cat.default([view_418, view_417, view_416], 2);  view_418 = view_417 = view_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_419: "f32[1024, 2304]" = torch.ops.aten.view.default(cat_8, [1024, 2304]);  cat_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_216: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_26, [1, 0]);  primals_26 = None
    mm_73: "f32[1024, 768]" = torch.ops.aten.mm.default(view_419, permute_216);  permute_216 = None
    permute_217: "f32[768, 1024]" = torch.ops.aten.permute.default(view_56, [1, 0]);  view_56 = None
    mm_74: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_217, view_419);  permute_217 = None
    sum_129: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_419, [0], True);  view_419 = None
    view_420: "f32[2304]" = torch.ops.aten.view.default(sum_129, [2304]);  sum_129 = None
    view_421: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_73, [1, 1024, 768]);  mm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    sub_110: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_24, getitem_42);  add_24 = getitem_42 = None
    mul_377: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_110, rsqrt_6);  sub_110 = None
    mul_378: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_421, primals_111);  primals_111 = None
    mul_379: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_378, 768)
    sum_130: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_378, [2], True)
    mul_380: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_378, mul_377);  mul_378 = None
    sum_131: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_380, [2], True);  mul_380 = None
    mul_381: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_377, sum_131);  sum_131 = None
    sub_111: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_379, sum_130);  mul_379 = sum_130 = None
    sub_112: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_111, mul_381);  sub_111 = mul_381 = None
    div_51: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
    mul_382: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_51, sub_112);  div_51 = sub_112 = None
    mul_383: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_421, mul_377);  mul_377 = None
    sum_132: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_383, [0, 1]);  mul_383 = None
    sum_133: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_421, [0, 1]);  view_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    add_153: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_150, mul_382);  add_150 = mul_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_28: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_40, torch.float32);  getitem_40 = None
    mul_384: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_28, 1.1111111111111112);  convert_element_type_28 = None
    mul_385: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_153, mul_384);  mul_384 = None
    clone_66: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_385, memory_format = torch.contiguous_format);  mul_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_422: "f32[1024, 768]" = torch.ops.aten.view.default(clone_66, [1024, 768]);  clone_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_218: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
    mm_75: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_422, permute_218);  permute_218 = None
    permute_219: "f32[3072, 1024]" = torch.ops.aten.permute.default(view_54, [1, 0]);  view_54 = None
    mm_76: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_219, view_422);  permute_219 = None
    sum_134: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_422, [0], True);  view_422 = None
    view_423: "f32[768]" = torch.ops.aten.view.default(sum_134, [768]);  sum_134 = None
    view_424: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(mm_75, [1, 1024, 3072]);  mm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_386: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_424, mul_20);  mul_20 = None
    mul_387: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_424, add_23);  view_424 = add_23 = None
    alias_42: "f32[1, 1024, 3072]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    mul_388: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(alias_42, alias_42);  alias_42 = None
    sub_113: "f32[1, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_388);  mul_388 = None
    mul_389: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_386, sub_113);  mul_386 = sub_113 = None
    mul_390: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_389, 0.7978845608028654);  mul_389 = None
    mul_391: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_390, 0.044715)
    pow_22: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_53, 2.0);  view_53 = None
    mul_392: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_22, 3.0);  pow_22 = None
    mul_393: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_391, mul_392);  mul_391 = mul_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_154: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_390, mul_393);  mul_390 = mul_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_394: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_387, 0.5);  mul_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_155: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(add_154, mul_394);  add_154 = mul_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_425: "f32[1024, 3072]" = torch.ops.aten.view.default(add_155, [1024, 3072]);  add_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_220: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_22, [1, 0]);  primals_22 = None
    mm_77: "f32[1024, 768]" = torch.ops.aten.mm.default(view_425, permute_220);  permute_220 = None
    permute_221: "f32[768, 1024]" = torch.ops.aten.permute.default(view_52, [1, 0]);  view_52 = None
    mm_78: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_221, view_425);  permute_221 = None
    sum_135: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_425, [0], True);  view_425 = None
    view_426: "f32[3072]" = torch.ops.aten.view.default(sum_135, [3072]);  sum_135 = None
    view_427: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_77, [1, 1024, 768]);  mm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    sub_114: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_19, getitem_38);  add_19 = getitem_38 = None
    mul_395: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_114, rsqrt_5);  sub_114 = None
    mul_396: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_427, primals_109);  primals_109 = None
    mul_397: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_396, 768)
    sum_136: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_396, [2], True)
    mul_398: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_396, mul_395);  mul_396 = None
    sum_137: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_398, [2], True);  mul_398 = None
    mul_399: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_395, sum_137);  sum_137 = None
    sub_115: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_397, sum_136);  mul_397 = sum_136 = None
    sub_116: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_115, mul_399);  sub_115 = mul_399 = None
    div_52: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
    mul_400: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_52, sub_116);  div_52 = sub_116 = None
    mul_401: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_427, mul_395);  mul_395 = None
    sum_138: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_401, [0, 1]);  mul_401 = None
    sum_139: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_427, [0, 1]);  view_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    add_156: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_153, mul_400);  add_153 = mul_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    convert_element_type_29: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_36, torch.float32);  getitem_36 = None
    mul_402: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_29, 1.1111111111111112);  convert_element_type_29 = None
    mul_403: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_156, mul_402);  mul_402 = None
    clone_67: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_403, memory_format = torch.contiguous_format);  mul_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_428: "f32[1024, 768]" = torch.ops.aten.view.default(clone_67, [1024, 768]);  clone_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_222: "f32[768, 768]" = torch.ops.aten.permute.default(primals_20, [1, 0]);  primals_20 = None
    mm_79: "f32[1024, 768]" = torch.ops.aten.mm.default(view_428, permute_222);  permute_222 = None
    permute_223: "f32[768, 1024]" = torch.ops.aten.permute.default(view_50, [1, 0]);  view_50 = None
    mm_80: "f32[768, 768]" = torch.ops.aten.mm.default(permute_223, view_428);  permute_223 = None
    sum_140: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_428, [0], True);  view_428 = None
    view_429: "f32[768]" = torch.ops.aten.view.default(sum_140, [768]);  sum_140 = None
    view_430: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_79, [1, 1024, 768]);  mm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_431: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(view_430, [1, 1024, 12, 64]);  view_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_224: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_431, [0, 2, 1, 3]);  view_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_432: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_224, [12, 1024, 64]);  permute_224 = None
    permute_225: "f32[12, 1024, 1024]" = torch.ops.aten.permute.default(view_46, [0, 2, 1]);  view_46 = None
    bmm_60: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(permute_225, view_432);  permute_225 = None
    permute_226: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_47, [0, 2, 1]);  view_47 = None
    bmm_61: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_432, permute_226);  view_432 = permute_226 = None
    view_433: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_60, [1, 12, 1024, 64]);  bmm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    add_157: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_7, view_433);  tangents_7 = view_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_434: "f32[1, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_61, [1, 12, 1024, 1024]);  bmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    convert_element_type_30: "f32[1, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_34, torch.float32);  getitem_34 = None
    mul_404: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_30, 1.1111111111111112);  convert_element_type_30 = None
    mul_405: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_434, mul_404);  view_434 = mul_404 = None
    clone_68: "f32[1, 12, 1024, 1024]" = torch.ops.aten.clone.default(mul_405, memory_format = torch.contiguous_format);  mul_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_43: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    mul_406: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(clone_68, alias_43);  clone_68 = None
    sum_141: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_406, [-1], True)
    mul_407: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_43, sum_141);  alias_43 = sum_141 = None
    sub_117: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_406, mul_407);  mul_406 = mul_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    scalar_tensor_9: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_21: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_6, sub_117, scalar_tensor_9);  slice_6 = sub_117 = scalar_tensor_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_53: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(where_21, full_4);  where_21 = full_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_435: "f32[12, 1024, 1024]" = torch.ops.aten.view.default(div_53, [12, 1024, 1024]);  div_53 = None
    permute_227: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_43, [0, 2, 1]);  view_43 = None
    bmm_62: "f32[12, 64, 1024]" = torch.ops.aten.bmm.default(permute_227, view_435);  permute_227 = None
    permute_228: "f32[12, 1024, 64]" = torch.ops.aten.permute.default(view_44, [0, 2, 1]);  view_44 = None
    bmm_63: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_435, permute_228);  view_435 = permute_228 = None
    view_436: "f32[1, 12, 64, 1024]" = torch.ops.aten.view.default(bmm_62, [1, 12, 64, 1024]);  bmm_62 = None
    view_437: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_63, [1, 12, 1024, 64]);  bmm_63 = None
    permute_229: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_436, [0, 1, 3, 2]);  view_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_158: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_6, permute_229);  tangents_6 = permute_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_230: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_157, [0, 2, 1, 3]);  add_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_69: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_230, memory_format = torch.contiguous_format);  permute_230 = None
    view_438: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_69, [1, 1024, 768]);  clone_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_231: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_158, [0, 2, 1, 3]);  add_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_70: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_231, memory_format = torch.contiguous_format);  permute_231 = None
    view_439: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_70, [1, 1024, 768]);  clone_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_232: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_437, [0, 2, 1, 3]);  view_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_71: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_232, memory_format = torch.contiguous_format);  permute_232 = None
    view_440: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_71, [1, 1024, 768]);  clone_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    cat_9: "f32[1, 1024, 2304]" = torch.ops.aten.cat.default([view_440, view_439, view_438], 2);  view_440 = view_439 = view_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_441: "f32[1024, 2304]" = torch.ops.aten.view.default(cat_9, [1024, 2304]);  cat_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_233: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
    mm_81: "f32[1024, 768]" = torch.ops.aten.mm.default(view_441, permute_233);  permute_233 = None
    permute_234: "f32[768, 1024]" = torch.ops.aten.permute.default(view_38, [1, 0]);  view_38 = None
    mm_82: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_234, view_441);  permute_234 = None
    sum_142: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_441, [0], True);  view_441 = None
    view_442: "f32[2304]" = torch.ops.aten.view.default(sum_142, [2304]);  sum_142 = None
    view_443: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_81, [1, 1024, 768]);  mm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    sub_118: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_16, getitem_29);  add_16 = getitem_29 = None
    mul_408: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_118, rsqrt_4);  sub_118 = None
    mul_409: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_443, primals_107);  primals_107 = None
    mul_410: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_409, 768)
    sum_143: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_409, [2], True)
    mul_411: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_409, mul_408);  mul_409 = None
    sum_144: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_411, [2], True);  mul_411 = None
    mul_412: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_408, sum_144);  sum_144 = None
    sub_119: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_410, sum_143);  mul_410 = sum_143 = None
    sub_120: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_119, mul_412);  sub_119 = mul_412 = None
    div_54: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
    mul_413: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_54, sub_120);  div_54 = sub_120 = None
    mul_414: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_443, mul_408);  mul_408 = None
    sum_145: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_414, [0, 1]);  mul_414 = None
    sum_146: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_443, [0, 1]);  view_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    add_159: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_156, mul_413);  add_156 = mul_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_31: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_27, torch.float32);  getitem_27 = None
    mul_415: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_31, 1.1111111111111112);  convert_element_type_31 = None
    mul_416: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_159, mul_415);  mul_415 = None
    clone_72: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_416, memory_format = torch.contiguous_format);  mul_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_444: "f32[1024, 768]" = torch.ops.aten.view.default(clone_72, [1024, 768]);  clone_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_235: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
    mm_83: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_444, permute_235);  permute_235 = None
    permute_236: "f32[3072, 1024]" = torch.ops.aten.permute.default(view_36, [1, 0]);  view_36 = None
    mm_84: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_236, view_444);  permute_236 = None
    sum_147: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_444, [0], True);  view_444 = None
    view_445: "f32[768]" = torch.ops.aten.view.default(sum_147, [768]);  sum_147 = None
    view_446: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(mm_83, [1, 1024, 3072]);  mm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_417: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_446, mul_12);  mul_12 = None
    mul_418: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_446, add_15);  view_446 = add_15 = None
    alias_44: "f32[1, 1024, 3072]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    mul_419: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(alias_44, alias_44);  alias_44 = None
    sub_121: "f32[1, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_419);  mul_419 = None
    mul_420: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_417, sub_121);  mul_417 = sub_121 = None
    mul_421: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_420, 0.7978845608028654);  mul_420 = None
    mul_422: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_421, 0.044715)
    pow_23: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_35, 2.0);  view_35 = None
    mul_423: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_23, 3.0);  pow_23 = None
    mul_424: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_422, mul_423);  mul_422 = mul_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_160: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_421, mul_424);  mul_421 = mul_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_425: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_418, 0.5);  mul_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_161: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(add_160, mul_425);  add_160 = mul_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_447: "f32[1024, 3072]" = torch.ops.aten.view.default(add_161, [1024, 3072]);  add_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_237: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None
    mm_85: "f32[1024, 768]" = torch.ops.aten.mm.default(view_447, permute_237);  permute_237 = None
    permute_238: "f32[768, 1024]" = torch.ops.aten.permute.default(view_34, [1, 0]);  view_34 = None
    mm_86: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_238, view_447);  permute_238 = None
    sum_148: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_447, [0], True);  view_447 = None
    view_448: "f32[3072]" = torch.ops.aten.view.default(sum_148, [3072]);  sum_148 = None
    view_449: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_85, [1, 1024, 768]);  mm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    sub_122: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_11, getitem_25);  add_11 = getitem_25 = None
    mul_426: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_122, rsqrt_3);  sub_122 = None
    mul_427: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_449, primals_105);  primals_105 = None
    mul_428: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_427, 768)
    sum_149: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_427, [2], True)
    mul_429: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_427, mul_426);  mul_427 = None
    sum_150: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_429, [2], True);  mul_429 = None
    mul_430: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_426, sum_150);  sum_150 = None
    sub_123: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_428, sum_149);  mul_428 = sum_149 = None
    sub_124: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_123, mul_430);  sub_123 = mul_430 = None
    div_55: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
    mul_431: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_55, sub_124);  div_55 = sub_124 = None
    mul_432: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_449, mul_426);  mul_426 = None
    sum_151: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_432, [0, 1]);  mul_432 = None
    sum_152: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_449, [0, 1]);  view_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    add_162: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_159, mul_431);  add_159 = mul_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    convert_element_type_32: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_23, torch.float32);  getitem_23 = None
    mul_433: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_32, 1.1111111111111112);  convert_element_type_32 = None
    mul_434: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_162, mul_433);  mul_433 = None
    clone_73: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_434, memory_format = torch.contiguous_format);  mul_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_450: "f32[1024, 768]" = torch.ops.aten.view.default(clone_73, [1024, 768]);  clone_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_239: "f32[768, 768]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
    mm_87: "f32[1024, 768]" = torch.ops.aten.mm.default(view_450, permute_239);  permute_239 = None
    permute_240: "f32[768, 1024]" = torch.ops.aten.permute.default(view_32, [1, 0]);  view_32 = None
    mm_88: "f32[768, 768]" = torch.ops.aten.mm.default(permute_240, view_450);  permute_240 = None
    sum_153: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_450, [0], True);  view_450 = None
    view_451: "f32[768]" = torch.ops.aten.view.default(sum_153, [768]);  sum_153 = None
    view_452: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_87, [1, 1024, 768]);  mm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_453: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(view_452, [1, 1024, 12, 64]);  view_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_241: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_453, [0, 2, 1, 3]);  view_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_454: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_241, [12, 1024, 64]);  permute_241 = None
    permute_242: "f32[12, 1024, 1024]" = torch.ops.aten.permute.default(view_28, [0, 2, 1]);  view_28 = None
    bmm_64: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(permute_242, view_454);  permute_242 = None
    permute_243: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_29, [0, 2, 1]);  view_29 = None
    bmm_65: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_454, permute_243);  view_454 = permute_243 = None
    view_455: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_64, [1, 12, 1024, 64]);  bmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    add_163: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_5, view_455);  tangents_5 = view_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_456: "f32[1, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_65, [1, 12, 1024, 1024]);  bmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    convert_element_type_33: "f32[1, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_21, torch.float32);  getitem_21 = None
    mul_435: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_33, 1.1111111111111112);  convert_element_type_33 = None
    mul_436: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_456, mul_435);  view_456 = mul_435 = None
    clone_74: "f32[1, 12, 1024, 1024]" = torch.ops.aten.clone.default(mul_436, memory_format = torch.contiguous_format);  mul_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_45: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    mul_437: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(clone_74, alias_45);  clone_74 = None
    sum_154: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_437, [-1], True)
    mul_438: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_45, sum_154);  alias_45 = sum_154 = None
    sub_125: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_437, mul_438);  mul_437 = mul_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    scalar_tensor_10: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_22: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_4, sub_125, scalar_tensor_10);  slice_4 = sub_125 = scalar_tensor_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_56: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(where_22, full_2);  where_22 = full_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_457: "f32[12, 1024, 1024]" = torch.ops.aten.view.default(div_56, [12, 1024, 1024]);  div_56 = None
    permute_244: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_25, [0, 2, 1]);  view_25 = None
    bmm_66: "f32[12, 64, 1024]" = torch.ops.aten.bmm.default(permute_244, view_457);  permute_244 = None
    permute_245: "f32[12, 1024, 64]" = torch.ops.aten.permute.default(view_26, [0, 2, 1]);  view_26 = None
    bmm_67: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_457, permute_245);  view_457 = permute_245 = None
    view_458: "f32[1, 12, 64, 1024]" = torch.ops.aten.view.default(bmm_66, [1, 12, 64, 1024]);  bmm_66 = None
    view_459: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_67, [1, 12, 1024, 64]);  bmm_67 = None
    permute_246: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_458, [0, 1, 3, 2]);  view_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_164: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_4, permute_246);  tangents_4 = permute_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_247: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_163, [0, 2, 1, 3]);  add_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_75: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_247, memory_format = torch.contiguous_format);  permute_247 = None
    view_460: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_75, [1, 1024, 768]);  clone_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_248: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_164, [0, 2, 1, 3]);  add_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_76: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_248, memory_format = torch.contiguous_format);  permute_248 = None
    view_461: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_76, [1, 1024, 768]);  clone_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_249: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_459, [0, 2, 1, 3]);  view_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_77: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None
    view_462: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_77, [1, 1024, 768]);  clone_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    cat_10: "f32[1, 1024, 2304]" = torch.ops.aten.cat.default([view_462, view_461, view_460], 2);  view_462 = view_461 = view_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_463: "f32[1024, 2304]" = torch.ops.aten.view.default(cat_10, [1024, 2304]);  cat_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_250: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
    mm_89: "f32[1024, 768]" = torch.ops.aten.mm.default(view_463, permute_250);  permute_250 = None
    permute_251: "f32[768, 1024]" = torch.ops.aten.permute.default(view_20, [1, 0]);  view_20 = None
    mm_90: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_251, view_463);  permute_251 = None
    sum_155: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_463, [0], True);  view_463 = None
    view_464: "f32[2304]" = torch.ops.aten.view.default(sum_155, [2304]);  sum_155 = None
    view_465: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_89, [1, 1024, 768]);  mm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    sub_126: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_8, getitem_16);  add_8 = getitem_16 = None
    mul_439: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_126, rsqrt_2);  sub_126 = None
    mul_440: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_465, primals_103);  primals_103 = None
    mul_441: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_440, 768)
    sum_156: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_440, [2], True)
    mul_442: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_440, mul_439);  mul_440 = None
    sum_157: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_442, [2], True);  mul_442 = None
    mul_443: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_439, sum_157);  sum_157 = None
    sub_127: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_441, sum_156);  mul_441 = sum_156 = None
    sub_128: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_127, mul_443);  sub_127 = mul_443 = None
    div_57: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
    mul_444: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_57, sub_128);  div_57 = sub_128 = None
    mul_445: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_465, mul_439);  mul_439 = None
    sum_158: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_445, [0, 1]);  mul_445 = None
    sum_159: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_465, [0, 1]);  view_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    add_165: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_162, mul_444);  add_162 = mul_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_34: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_14, torch.float32);  getitem_14 = None
    mul_446: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_34, 1.1111111111111112);  convert_element_type_34 = None
    mul_447: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_165, mul_446);  mul_446 = None
    clone_78: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_447, memory_format = torch.contiguous_format);  mul_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_466: "f32[1024, 768]" = torch.ops.aten.view.default(clone_78, [1024, 768]);  clone_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_252: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
    mm_91: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_466, permute_252);  permute_252 = None
    permute_253: "f32[3072, 1024]" = torch.ops.aten.permute.default(view_18, [1, 0]);  view_18 = None
    mm_92: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_253, view_466);  permute_253 = None
    sum_160: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_466, [0], True);  view_466 = None
    view_467: "f32[768]" = torch.ops.aten.view.default(sum_160, [768]);  sum_160 = None
    view_468: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(mm_91, [1, 1024, 3072]);  mm_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_448: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_468, mul_4);  mul_4 = None
    mul_449: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_468, add_7);  view_468 = add_7 = None
    alias_46: "f32[1, 1024, 3072]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    mul_450: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(alias_46, alias_46);  alias_46 = None
    sub_129: "f32[1, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_450);  mul_450 = None
    mul_451: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_448, sub_129);  mul_448 = sub_129 = None
    mul_452: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_451, 0.7978845608028654);  mul_451 = None
    mul_453: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_452, 0.044715)
    pow_24: "f32[1, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_17, 2.0);  view_17 = None
    mul_454: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_24, 3.0);  pow_24 = None
    mul_455: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_453, mul_454);  mul_453 = mul_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_166: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_452, mul_455);  mul_452 = mul_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_456: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_449, 0.5);  mul_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    add_167: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(add_166, mul_456);  add_166 = mul_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_469: "f32[1024, 3072]" = torch.ops.aten.view.default(add_167, [1024, 3072]);  add_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_254: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
    mm_93: "f32[1024, 768]" = torch.ops.aten.mm.default(view_469, permute_254);  permute_254 = None
    permute_255: "f32[768, 1024]" = torch.ops.aten.permute.default(view_16, [1, 0]);  view_16 = None
    mm_94: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_255, view_469);  permute_255 = None
    sum_161: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_469, [0], True);  view_469 = None
    view_470: "f32[3072]" = torch.ops.aten.view.default(sum_161, [3072]);  sum_161 = None
    view_471: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_93, [1, 1024, 768]);  mm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    sub_130: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(add_3, getitem_12);  add_3 = getitem_12 = None
    mul_457: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_130, rsqrt_1);  sub_130 = None
    mul_458: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_471, primals_101);  primals_101 = None
    mul_459: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_458, 768)
    sum_162: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_458, [2], True)
    mul_460: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_458, mul_457);  mul_458 = None
    sum_163: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_460, [2], True);  mul_460 = None
    mul_461: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_457, sum_163);  sum_163 = None
    sub_131: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_459, sum_162);  mul_459 = sum_162 = None
    sub_132: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_131, mul_461);  sub_131 = mul_461 = None
    div_58: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
    mul_462: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_58, sub_132);  div_58 = sub_132 = None
    mul_463: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_471, mul_457);  mul_457 = None
    sum_164: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_463, [0, 1]);  mul_463 = None
    sum_165: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_471, [0, 1]);  view_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:426, code: hidden_states = self.ln_2(hidden_states)
    add_168: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_165, mul_462);  add_165 = mul_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:335, code: attn_output = self.resid_dropout(attn_output)
    convert_element_type_35: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_10, torch.float32);  getitem_10 = None
    mul_464: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_35, 1.1111111111111112);  convert_element_type_35 = None
    mul_465: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_168, mul_464);  mul_464 = None
    clone_79: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_465, memory_format = torch.contiguous_format);  mul_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_472: "f32[1024, 768]" = torch.ops.aten.view.default(clone_79, [1024, 768]);  clone_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_256: "f32[768, 768]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
    mm_95: "f32[1024, 768]" = torch.ops.aten.mm.default(view_472, permute_256);  permute_256 = None
    permute_257: "f32[768, 1024]" = torch.ops.aten.permute.default(view_14, [1, 0]);  view_14 = None
    mm_96: "f32[768, 768]" = torch.ops.aten.mm.default(permute_257, view_472);  permute_257 = None
    sum_166: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_472, [0], True);  view_472 = None
    view_473: "f32[768]" = torch.ops.aten.view.default(sum_166, [768]);  sum_166 = None
    view_474: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_95, [1, 1024, 768]);  mm_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:288, code: return tensor.view(new_shape)
    view_475: "f32[1, 1024, 12, 64]" = torch.ops.aten.view.default(view_474, [1, 1024, 12, 64]);  view_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:286, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_258: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_475, [0, 2, 1, 3]);  view_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_476: "f32[12, 1024, 64]" = torch.ops.aten.view.default(permute_258, [12, 1024, 64]);  permute_258 = None
    permute_259: "f32[12, 1024, 1024]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
    bmm_68: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(permute_259, view_476);  permute_259 = None
    permute_260: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
    bmm_69: "f32[12, 1024, 1024]" = torch.ops.aten.bmm.default(view_476, permute_260);  view_476 = permute_260 = None
    view_477: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_68, [1, 12, 1024, 64]);  bmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    add_169: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_3, view_477);  tangents_3 = view_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:218, code: attn_output = torch.matmul(attn_weights, value)
    view_478: "f32[1, 12, 1024, 1024]" = torch.ops.aten.view.default(bmm_69, [1, 12, 1024, 1024]);  bmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:212, code: attn_weights = self.attn_dropout(attn_weights)
    convert_element_type_36: "f32[1, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(getitem_8, torch.float32);  getitem_8 = None
    mul_466: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_36, 1.1111111111111112);  convert_element_type_36 = None
    mul_467: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_478, mul_466);  view_478 = mul_466 = None
    clone_80: "f32[1, 12, 1024, 1024]" = torch.ops.aten.clone.default(mul_467, memory_format = torch.contiguous_format);  mul_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:208, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    alias_47: "f32[1, 12, 1024, 1024]" = torch.ops.aten.alias.default(alias);  alias = None
    mul_468: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(clone_80, alias_47);  clone_80 = None
    sum_167: "f32[1, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_468, [-1], True)
    mul_469: "f32[1, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(alias_47, sum_167);  alias_47 = sum_167 = None
    sub_133: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_468, mul_469);  mul_468 = mul_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:202, code: attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    scalar_tensor_11: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_23: "f32[1, 12, 1024, 1024]" = torch.ops.aten.where.self(slice_2, sub_133, scalar_tensor_11);  slice_2 = sub_133 = scalar_tensor_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:186, code: attn_weights = attn_weights / torch.full(
    div_59: "f32[1, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(where_23, full);  where_23 = full = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    view_479: "f32[12, 1024, 1024]" = torch.ops.aten.view.default(div_59, [12, 1024, 1024]);  div_59 = None
    permute_261: "f32[12, 64, 1024]" = torch.ops.aten.permute.default(view_7, [0, 2, 1]);  view_7 = None
    bmm_70: "f32[12, 64, 1024]" = torch.ops.aten.bmm.default(permute_261, view_479);  permute_261 = None
    permute_262: "f32[12, 1024, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1]);  view_8 = None
    bmm_71: "f32[12, 1024, 64]" = torch.ops.aten.bmm.default(view_479, permute_262);  view_479 = permute_262 = None
    view_480: "f32[1, 12, 64, 1024]" = torch.ops.aten.view.default(bmm_70, [1, 12, 64, 1024]);  bmm_70 = None
    view_481: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(bmm_71, [1, 12, 1024, 64]);  bmm_71 = None
    permute_263: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(view_480, [0, 1, 3, 2]);  view_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:183, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    add_170: "f32[1, 12, 1024, 64]" = torch.ops.aten.add.Tensor(tangents_2, permute_263);  tangents_2 = permute_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_264: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_169, [0, 2, 1, 3]);  add_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_81: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_264, memory_format = torch.contiguous_format);  permute_264 = None
    view_482: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_81, [1, 1024, 768]);  clone_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_265: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(add_170, [0, 2, 1, 3]);  add_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_82: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_265, memory_format = torch.contiguous_format);  permute_265 = None
    view_483: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_82, [1, 1024, 768]);  clone_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_266: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_481, [0, 2, 1, 3]);  view_481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:279, code: tensor = tensor.view(new_shape)
    clone_83: "f32[1, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_266, memory_format = torch.contiguous_format);  permute_266 = None
    view_484: "f32[1, 1024, 768]" = torch.ops.aten.view.default(clone_83, [1, 1024, 768]);  clone_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:312, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
    cat_11: "f32[1, 1024, 2304]" = torch.ops.aten.cat.default([view_484, view_483, view_482], 2);  view_484 = view_483 = view_482 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:107, code: x = x.view(size_out)
    view_485: "f32[1024, 2304]" = torch.ops.aten.view.default(cat_11, [1024, 2304]);  cat_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/pytorch_utils.py:106, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    permute_267: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
    mm_97: "f32[1024, 768]" = torch.ops.aten.mm.default(view_485, permute_267);  permute_267 = None
    permute_268: "f32[768, 1024]" = torch.ops.aten.permute.default(view_2, [1, 0]);  view_2 = None
    mm_98: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_268, view_485);  permute_268 = None
    sum_168: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_485, [0], True);  view_485 = None
    view_486: "f32[2304]" = torch.ops.aten.view.default(sum_168, [2304]);  sum_168 = None
    view_487: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_97, [1, 1024, 768]);  mm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    sub_134: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(getitem, getitem_3);  getitem = getitem_3 = None
    mul_470: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_134, rsqrt);  sub_134 = None
    mul_471: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_487, primals_99);  primals_99 = None
    mul_472: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_471, 768)
    sum_169: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_471, [2], True)
    mul_473: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_471, mul_470);  mul_471 = None
    sum_170: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_473, [2], True);  mul_473 = None
    mul_474: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_470, sum_170);  sum_170 = None
    sub_135: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_472, sum_169);  mul_472 = sum_169 = None
    sub_136: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_135, mul_474);  sub_135 = mul_474 = None
    div_60: "f32[1, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
    mul_475: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_60, sub_136);  div_60 = sub_136 = None
    mul_476: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(view_487, mul_470);  mul_470 = None
    sum_171: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_476, [0, 1]);  mul_476 = None
    sum_172: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_487, [0, 1]);  view_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:389, code: hidden_states = self.ln_1(hidden_states)
    add_171: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(add_168, mul_475);  add_168 = mul_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:851, code: hidden_states = self.drop(hidden_states)
    convert_element_type_37: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_1, torch.float32);  getitem_1 = None
    mul_477: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_37, 1.1111111111111112);  convert_element_type_37 = None
    mul_478: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_171, mul_477);  add_171 = mul_477 = None
    clone_84: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_478, memory_format = torch.contiguous_format);  mul_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:844, code: position_embeds = self.wpe(position_ids)
    eq_1: "b8[1, 1024]" = torch.ops.aten.eq.Scalar(view_1, -1)
    unsqueeze_1: "b8[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
    scalar_tensor_12: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_24: "f32[1, 1024, 768]" = torch.ops.aten.where.self(unsqueeze_1, scalar_tensor_12, clone_84);  unsqueeze_1 = scalar_tensor_12 = None
    full_25: "f32[1024, 768]" = torch.ops.aten.full.default([1024, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put: "f32[1024, 768]" = torch.ops.aten._unsafe_index_put.default(full_25, [view_1], where_24, True);  full_25 = view_1 = where_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:843, code: inputs_embeds = self.wte(input_ids)
    eq_2: "b8[1, 1024]" = torch.ops.aten.eq.Scalar(view, -1)
    unsqueeze_2: "b8[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_2, -1);  eq_2 = None
    scalar_tensor_13: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_25: "f32[1, 1024, 768]" = torch.ops.aten.where.self(unsqueeze_2, scalar_tensor_13, clone_84);  unsqueeze_2 = scalar_tensor_13 = clone_84 = None
    full_26: "f32[50257, 768]" = torch.ops.aten.full.default([50257, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_1: "f32[50257, 768]" = torch.ops.aten._unsafe_index_put.default(full_26, [view], where_25, True);  full_26 = view = where_25 = None
    return pytree.tree_unflatten([view_218, permute_1, permute_2, permute_6, permute_7, permute_11, permute_12, permute_16, permute_17, permute_21, permute_22, permute_26, permute_27, permute_31, permute_32, permute_36, permute_37, permute_41, permute_42, permute_46, permute_47, permute_51, permute_52, permute_56, permute_57, index, view_486, mm_98, view_473, mm_96, view_470, mm_94, view_467, mm_92, view_464, mm_90, view_451, mm_88, view_448, mm_86, view_445, mm_84, view_442, mm_82, view_429, mm_80, view_426, mm_78, view_423, mm_76, view_420, mm_74, view_407, mm_72, view_404, mm_70, view_401, mm_68, view_398, mm_66, view_385, mm_64, view_382, mm_62, view_379, mm_60, view_376, mm_58, view_363, mm_56, view_360, mm_54, view_357, mm_52, view_354, mm_50, view_341, mm_48, view_338, mm_46, view_335, mm_44, view_332, mm_42, view_319, mm_40, view_316, mm_38, view_313, mm_36, view_310, mm_34, view_297, mm_32, view_294, mm_30, view_291, mm_28, view_288, mm_26, view_275, mm_24, view_272, mm_22, view_269, mm_20, view_266, mm_18, view_253, mm_16, view_250, mm_14, view_247, mm_12, view_244, mm_10, view_231, mm_8, view_228, mm_6, view_225, mm_4, _unsafe_index_put_1, _unsafe_index_put, sum_171, sum_172, sum_164, sum_165, sum_158, sum_159, sum_151, sum_152, sum_145, sum_146, sum_138, sum_139, sum_132, sum_133, sum_125, sum_126, sum_119, sum_120, sum_112, sum_113, sum_106, sum_107, sum_99, sum_100, sum_93, sum_94, sum_86, sum_87, sum_80, sum_81, sum_73, sum_74, sum_67, sum_68, sum_60, sum_61, sum_54, sum_55, sum_47, sum_48, sum_41, sum_42, sum_34, sum_35, sum_28, sum_29, sum_21, sum_22, sum_15, sum_16, permute_64, None, None, None, None, None, None, None, None, None, None, None, None, None], self._out_spec)
    